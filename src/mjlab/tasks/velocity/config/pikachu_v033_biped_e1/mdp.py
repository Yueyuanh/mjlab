"""Biped E1 soft arm windows, contact clearance and quiet-standing costs."""

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp as common
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def standing_mask(
  env: "ManagerBasedRlEnv", command_name: str = "twist", threshold: float = 0.05
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return (
    torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs() <= threshold
  )


def arm_position_window(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
  bounds_degrees: tuple[tuple[float, float], ...],
) -> torch.Tensor:
  """Mean squared excess beyond absolute, asymmetric joint windows.

  Normalize each excess by that window's half-width. Only actual positions
  are read: policy outputs, actuator targets and mechanical limits are intact.
  """
  robot = env.scene[asset_cfg.name]
  positions = robot.data.joint_pos[:, asset_cfg.joint_ids]
  if len(bounds_degrees) != positions.shape[-1] or any(
    low >= high for low, high in bounds_degrees
  ):
    raise ValueError("Each selected joint needs one increasing position window")
  bounds = positions.new_tensor(bounds_degrees) * (math.pi / 180)
  low, high = bounds.unbind(-1)
  half_width = (high - low) / 2
  excess = (low - positions).clamp_min(0) + (positions - high).clamp_min(0)
  return (excess / half_width).square().mean(-1)


def arm_leg_clearance(
  env: "ManagerBasedRlEnv", sensor_name: str, clearance: float = 0.01
) -> torch.Tensor:
  """Penalize closest mesh contacts within the detection-only gap zone."""
  if clearance <= 0:
    raise ValueError("clearance must be positive")
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  data = sensor.data
  assert data.found is not None and data.dist is not None
  shortfall = (1.0 - data.dist / clearance).clamp_min(0)
  return (shortfall.square() * (data.found > 0)).sum(-1)


def standing_joint_velocity(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg, velocity_std: float = 1.0
) -> torch.Tensor:
  velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
  return (velocity / velocity_std).square().mean(-1) * standing_mask(env)


def standing_base_motion(env: "ManagerBasedRlEnv") -> torch.Tensor:
  data = env.scene["robot"].data
  linear = (data.root_link_lin_vel_w / 0.05).square().sum(-1)
  angular = (data.root_link_ang_vel_w / 0.15).square().sum(-1)
  # Preserve strong curvature near zero without letting a fall's large speed
  # dominate the whole return and make short episodes artificially attractive.
  return torch.log1p(linear + angular) * standing_mask(env)


def standing_foot_motion(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg, velocity_std: float = 0.03
) -> torch.Tensor:
  velocity = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids]
  motion = (velocity / velocity_std).square().sum(-1).mean(-1)
  return torch.log1p(motion) * standing_mask(env)


def standing_support(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  found = sensor.data.found
  assert found is not None
  return (found == 0).float().mean(-1) * standing_mask(env)


def push_moving_robots(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  velocity_range: dict[str, tuple[float, float]],
) -> None:
  """Keep walking perturbations without kicking zero-command standing samples."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  moving = env_ids[~standing_mask(env)[env_ids]]
  if moving.numel():
    common.push_by_setting_velocity(env, moving, velocity_range)
