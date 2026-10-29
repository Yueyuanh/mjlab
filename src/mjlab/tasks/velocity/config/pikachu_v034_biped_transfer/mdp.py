"""Task-local action/state adapters for transferring V033 walking weights."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import (
  POLICY_ACTION_JOINT_NAMES,
  POLICY_OBSERVATION_JOINT_NAMES,
)
from mjlab.envs import mdp as common
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .mapping import coordinate_matrix, foot_coordinate_matrices

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class TransferJointPositionActionCfg(JointPositionActionCfg):
  retarget_feet: bool = False

  def build(self, env: "ManagerBasedRlEnv") -> "TransferJointPositionAction":
    return TransferJointPositionAction(self, env)


class TransferJointPositionAction(JointPositionAction):
  def __init__(self, cfg: TransferJointPositionActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    if tuple(self._target_names) != POLICY_ACTION_JOINT_NAMES:
      raise ValueError("V033 transfer requires the explicit reference action order")
    self._action_map = torch.as_tensor(
      coordinate_matrix(POLICY_ACTION_JOINT_NAMES),
      device=self.device,
      dtype=torch.float32,
    )
    state_map = torch.as_tensor(
      coordinate_matrix(POLICY_OBSERVATION_JOINT_NAMES),
      device=self.device,
      dtype=torch.float32,
    )
    self.native_to_policy = torch.linalg.inv(state_map).T
    self.state_bias = torch.zeros(
      len(POLICY_OBSERVATION_JOINT_NAMES), device=self.device
    )
    if cfg.retarget_feet:
      action_map, observation_map, bias = foot_coordinate_matrices()
      self._action_map = torch.as_tensor(
        action_map, dtype=torch.float32, device=self.device
      )
      self.native_to_policy = torch.as_tensor(
        observation_map.T, dtype=torch.float32, device=self.device
      )
      self.state_bias = torch.as_tensor(bias, dtype=torch.float32, device=self.device)

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)
    residual = self._processed_actions - self._offset
    self._processed_actions[:] = residual @ self._action_map.T + self._offset


def joint_pos_reference(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg, biased: bool = False
) -> torch.Tensor:
  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, TransferJointPositionAction)
  return (
    common.joint_pos_rel(env, biased=biased, asset_cfg=asset_cfg)
    @ term.native_to_policy
  ) + term.state_bias


def joint_vel_reference(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, TransferJointPositionAction)
  return common.joint_vel_rel(env, asset_cfg=asset_cfg) @ term.native_to_policy
