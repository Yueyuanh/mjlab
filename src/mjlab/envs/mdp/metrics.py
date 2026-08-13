from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _single_joint_value(
  values: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  metric_name: str,
) -> torch.Tensor:
  selected = values[:, asset_cfg.joint_ids]
  if selected.ndim != 2 or selected.shape[1] != 1:
    raise ValueError(
      f"{metric_name} requires exactly one joint, got shape {tuple(selected.shape)}"
    )
  return selected[:, 0]


def joint_position(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return one joint's signed position in radians for each environment."""
  asset: Entity = env.scene[asset_cfg.name]
  return _single_joint_value(asset.data.joint_pos, asset_cfg, "joint_position")


def joint_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return one joint's signed velocity in radians per second."""
  asset: Entity = env.scene[asset_cfg.name]
  return _single_joint_value(asset.data.joint_vel, asset_cfg, "joint_velocity")


def joint_torque(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return one hinge joint's signed actuator torque in newton-metres.

  ``qfrc_actuator`` is the actuator contribution projected into joint space,
  including the output of MuJoCo's position-actuator PD law.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return _single_joint_value(asset.data.qfrc_actuator, asset_cfg, "joint_torque")


def root_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return root-link height above the environment origin in metres."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def mean_action_acc(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean absolute action acceleration.

  Lower values indicate smoother actions.

  Returns:
    Per-environment scalar. Shape: ``(B,)``.
  """
  # Discrete second derivative: a_t - 2 * a_{t-1} + a_{t-2}.  (B, N)
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.mean(torch.abs(action_acc), dim=-1)  # (B,)
