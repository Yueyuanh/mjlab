"""V034 crawl shaping ported from the selected V033 E3 smooth gait."""

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.config.pikachu_v035 import crawl_mdp
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def arm_posture_window(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
  tolerance_degrees: float = 5.0,
) -> torch.Tensor:
  if tolerance_degrees <= 0:
    raise ValueError("tolerance_degrees must be positive")
  robot = env.scene[asset_cfg.name]
  delta = (
    robot.data.joint_pos[:, asset_cfg.joint_ids]
    - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  tolerance = math.radians(tolerance_degrees)
  excess = (delta.abs() - tolerance).clamp_min(0)
  return (excess / tolerance).square().mean(-1)


def endpoint_distance_window(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
  minimum: float = 0.10,
  maximum: float = 0.25,
  std: float = 0.08,
) -> torch.Tensor:
  if not 0 < minimum < maximum or std <= 0:
    raise ValueError("Require 0 < minimum < maximum and std > 0")
  points = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids]
  distance = (points[:, 0::2] - points[:, 1::2]).norm(dim=-1)
  excess = (distance - maximum).clamp_min(0)
  shortage = (minimum - distance).clamp_min(0)
  return ((excess / std).square() + (shortage / std).square()).mean(-1)


def sole_geometry(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  z = quat_apply(quat, quat.new_tensor([0.0, 0.0, 1.0]).expand(*quat.shape[:-1], 3))
  x = quat_apply(quat, quat.new_tensor([1.0, 0.0, 0.0]).expand(*quat.shape[:-1], 3))
  tilt = torch.atan2(z[..., :2].norm(dim=-1), z[..., 2])
  return tilt, 0.08 * x[..., 2].abs()


def sole_stance_weight(env: "ManagerBasedRlEnv") -> torch.Tensor:
  phase = env.episode_length_buf[:, None] * env.step_dt / 1.6
  phase = torch.remainder(phase - phase.new_tensor([0.0, 0.5]), 1.0)
  weight = torch.minimum((phase - 0.20) / 0.10, (1.0 - phase) / 0.10).clamp(0, 1)
  return torch.where(crawl_mdp.moving(env)[:, None], weight, torch.ones_like(weight))


def sole_support_cost(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
  endpoint: bool = False,
  tolerance_degrees: float = 5.0,
  scale_degrees: float = 10.0,
) -> torch.Tensor:
  quat = env.scene[asset_cfg.name].data.body_link_quat_w[:, asset_cfg.body_ids]
  tilt, gap = sole_geometry(quat)
  error = (
    (gap - 0.008).clamp_min(0) / 0.01
    if endpoint
    else (tilt - math.radians(tolerance_degrees)).clamp_min(0)
    / math.radians(scale_degrees)
  )
  planned = sole_stance_weight(env)
  sensor = env.scene["feet_ground_contact"]
  assert isinstance(sensor, ContactSensor)
  indices = [
    next(
      i
      for i, name in enumerate(sensor.primary_names)
      if name.endswith(f"{side}_ankle_link")
    )
    for side in ("right", "left")
  ]
  measured = (sensor.data.current_contact_time[:, indices] / 0.08).clamp(0, 1)
  assert sensor.data.found is not None
  measured *= (sensor.data.found[:, indices] > 0).float()
  return (error.square() * torch.maximum(planned, measured)).mean(-1)


def ankle_limit_margin(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  q = robot.data.joint_pos[:, asset_cfg.joint_ids]
  limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  distance = torch.minimum(q - limits[..., 0], limits[..., 1] - q)
  return (
    ((math.radians(8.0) - distance).clamp_min(0) / math.radians(8.0)).square().mean(-1)
  )


def forward_weight(
  env: "ManagerBasedRlEnv", forward: float = 1.0, backward: float = 0.20
) -> torch.Tensor:
  command = env.command_manager.get_command("twist")
  assert command is not None
  longitudinal = command[:, 0].abs() > 0.04
  scale = torch.where(command[:, 0] > 0.04, forward, backward)
  return torch.where(longitudinal, scale, torch.zeros_like(scale))


def directional_sole_cost(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
  endpoint: bool = False,
) -> torch.Tensor:
  return sole_support_cost(env, asset_cfg, endpoint=endpoint) * forward_weight(env)


def smooth_swing_weight(
  env: "ManagerBasedRlEnv", transition_seconds: float = 0.10
) -> torch.Tensor:
  if not 0 < transition_seconds <= 0.16:
    raise ValueError("transition_seconds must be in (0, 0.16]")
  phase = env.episode_length_buf[:, None] * env.step_dt / 1.6
  offsets = phase.new_tensor([0.0, 0.5, 0.75, 0.25])
  relative = torch.remainder(phase - offsets, 1.0)
  inside = relative < 0.20
  distance = torch.minimum(relative, 0.20 - relative)
  alpha = (distance / (transition_seconds / 1.6)).clamp(0.0, 1.0)
  smooth = alpha.square() * (3.0 - 2.0 * alpha)
  return (
    torch.where(inside, smooth, torch.zeros_like(smooth))
    * crawl_mdp.moving(env)[:, None]
  )


def smooth_gait_mismatch(
  env: "ManagerBasedRlEnv", transition_seconds: float = 0.10
) -> torch.Tensor:
  sensor = env.scene["feet_ground_contact"]
  assert isinstance(sensor, ContactSensor) and sensor.data.found is not None
  indices = crawl_mdp.support_contact_indices(sensor.primary_names)
  contact = (sensor.data.found[:, indices] > 0).float()
  target_contact = 1.0 - smooth_swing_weight(env, transition_seconds)
  return (contact - target_contact).abs().mean(-1)


def smooth_gait_contacts(
  env: "ManagerBasedRlEnv", transition_seconds: float = 0.10
) -> torch.Tensor:
  return 1.0 - smooth_gait_mismatch(env, transition_seconds)


def directional_smooth_gait_mismatch(
  env: "ManagerBasedRlEnv", transition_seconds: float = 0.10
) -> torch.Tensor:
  return smooth_gait_mismatch(env, transition_seconds) * forward_weight(env)


def forward_velocity_shortfall(env: "ManagerBasedRlEnv") -> torch.Tensor:
  command = env.command_manager.get_command("twist")
  assert command is not None
  error = (command[:, 0] - crawl_mdp.heading_lin_vel(env)[:, 0]).clamp_min(0.0)
  return (error / 0.08).square() * (command[:, 0] > 0.04)


def directional_progress(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Reward signed longitudinal progress for both forward and backward commands.

  Exponential tracking gives a useful target basin but can leave a stable static
  policy with little incentive to discover locomotion.  This linear companion is
  zero at rest, positive only in the commanded direction, and normalized so that
  reaching the commanded speed returns one in either direction.
  """
  command = env.command_manager.get_command("twist")
  assert command is not None
  target = command[:, 0]
  moving = target.abs() > 0.04
  progress = target.sign() * crawl_mdp.heading_lin_vel(env)[:, 0]
  normalized = progress / target.abs().clamp_min(0.04)
  return normalized.clamp(-1.0, 1.0) * moving
