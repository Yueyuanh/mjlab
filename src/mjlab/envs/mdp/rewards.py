"""Useful methods for MDP rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for being alive."""
  return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize terminated episodes that don't correspond to episodic timeouts."""
  return env.termination_manager.terminated.float()


def joint_torques_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint torques applied on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.square(asset.data.actuator_force[:, asset_cfg.actuator_ids]), dim=1
  )


def joint_vel_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint velocities on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint accelerations on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_power_l1(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize mechanical power draw (sum of |torque x joint velocity|), in W.

  Power is the instantaneous energy rate a joint either injects into
  (positive) or recovers from (negative, braking/regeneration) the system.
  Limiting ``sum_i |tau_i * dot_q_i|`` bounds the combined effort--velocity
  usage, which a torque-only or velocity-only penalty cannot see alone.

  Each actuator is paired with the **joint it actually drives** via
  ``mjModel.actuator_trnid``, so this is correct even when actuator indices
  are ordered differently from joint indices (as in the Pikachu robots).
  Only JOINT-transmission actuators are included; non-joint transmissions
  (tendon/site) are skipped.
  """
  asset: Entity = env.scene[asset_cfg.name]
  actuator_feedforward = torch.as_tensor(
    env.sim.mj_model.actuator_trnid, device=asset.data.device
  )
  actuator_ids = asset_cfg.actuator_ids
  if isinstance(actuator_ids, slice) or actuator_ids is None:
    actuator_ids = torch.arange(
      actuator_feedforward.shape[0], device=asset.data.device
    )
  else:
    actuator_ids = torch.as_tensor(actuator_ids, device=asset.data.device)
  driven_joint_ids = actuator_feedforward[actuator_ids, 0]
  joint_mask = driven_joint_ids >= 0
  torque = asset.data.actuator_force[:, actuator_ids[joint_mask]]
  joint_vel = asset.data.joint_vel[:, driven_joint_ids[joint_mask]]
  return torch.sum((torque * joint_vel).abs(), dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel.

  Operates on raw policy output (before per-term scale/offset).
  """
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def action_acc_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the acceleration of the actions using L2 squared kernel.

  Operates on raw policy output (before per-term scale/offset).
  """
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.sum(torch.square(action_acc), dim=1)


def joint_pos_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint positions if they cross the soft limits."""
  asset: Entity = env.scene[asset_cfg.name]
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  out_of_limits = -(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
  ).clip(max=0.0)
  out_of_limits += (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
  ).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


class posture:
  """Penalize the deviation of the joint positions from the default positions.

  Note: This is implemented as a class so that we can resolve the standard deviation
  dictionary into a tensor and thereafter use it in the __call__ method.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )

    _, _, std = resolve_matching_names_values(
      data=cfg.params["std"],
      list_of_strings=joint_names,
    )
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

  def __call__(
    self, env: ManagerBasedRlEnv, std, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    del std  # Unused.
    asset: Entity = env.scene[asset_cfg.name]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)
    return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


class electrical_power_cost:
  """Penalize electrical power consumption of actuators."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]

    joint_ids, _ = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.qfrc_actuator[:, self._joint_ids]
    qd = asset.data.joint_vel[:, self._joint_ids]
    mech = tau * qd
    mech_pos = torch.clamp(mech, min=0.0)  # Don't penalize regen.
    return torch.sum(mech_pos, dim=1)


def flat_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat base orientation."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
