"""V033 policy-coordinate adapters for the V034 crawl articulation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import (
  POLICY_ACTION_JOINT_NAMES,
)
from mjlab.envs import mdp as common
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .mapping import crawl_coordinate_matrices

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class CrawlTransferJointPositionActionCfg(JointPositionActionCfg):
  def build(self, env: "ManagerBasedRlEnv") -> "CrawlTransferJointPositionAction":
    return CrawlTransferJointPositionAction(self, env)


class CrawlTransferJointPositionAction(JointPositionAction):
  def __init__(
    self, cfg: CrawlTransferJointPositionActionCfg, env: "ManagerBasedRlEnv"
  ):
    super().__init__(cfg, env)
    if tuple(self._target_names) != POLICY_ACTION_JOINT_NAMES:
      raise ValueError("V033 crawl transfer requires the reference action order")
    action_map, observation_map = crawl_coordinate_matrices()
    self._action_map = torch.as_tensor(
      action_map, device=self.device, dtype=torch.float32
    )
    self.native_to_policy = torch.as_tensor(
      observation_map.T, device=self.device, dtype=torch.float32
    )

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)
    residual = self._processed_actions - self._offset
    self._processed_actions[:] = residual @ self._action_map.T + self._offset


def joint_pos_reference(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg, biased: bool = False
) -> torch.Tensor:
  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, CrawlTransferJointPositionAction)
  return common.joint_pos_rel(
    env, biased=biased, asset_cfg=asset_cfg
  ) @ term.native_to_policy


def joint_vel_reference(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, CrawlTransferJointPositionAction)
  return common.joint_vel_rel(env, asset_cfg=asset_cfg) @ term.native_to_policy
