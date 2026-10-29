"""Distances use physical endpoint sites, independent of torso heading."""

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl.mdp import moving
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def endpoint_distance_window(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
  minimum: float = 0.10,
  maximum: float = 0.25,
  std: float = 0.08,
) -> torch.Tensor:
  """Soft band on 3D hand-foot distance, with a short-distance separation cost.

  The upper boundary encourages compact support without attracting points
  all the way to coincidence. Swing height remains free inside the band.
  """
  if not 0 < minimum < maximum or std <= 0:
    raise ValueError("Require 0 < minimum < maximum and std > 0")
  points = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids]
  distance = (points[:, 0::2] - points[:, 1::2]).norm(dim=-1)
  excess = (distance - maximum).clamp_min(0)
  shortage = (minimum - distance).clamp_min(0)
  return ((excess / std).square() + (shortage / std).square()).mean(-1)


def sole_geometry(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """True sole tilt and 80 mm toe/heel height separation, in world coordinates."""
  z = quat_apply(quat, quat.new_tensor([0.0, 0.0, 1.0]).expand(*quat.shape[:-1], 3))
  x = quat_apply(quat, quat.new_tensor([1.0, 0.0, 0.0]).expand(*quat.shape[:-1], 3))
  tilt = torch.atan2(z[..., :2].norm(dim=-1), z[..., 2])
  return tilt, 0.08 * x[..., 2].abs()


def sole_stance_weight(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Fade the planned support edges; missing contact cannot evade the cost.

  Rear-foot swing fractions are [0,.2] and [.5,.7] of the 1.6 s cycle.
  A 0.10-cycle fade leaves touchdown/toe-off free, but constrains mid-stance.
  Exact zero command constrains both feet continuously.
  """
  phase = env.episode_length_buf[:, None] * env.step_dt / 1.6
  phase = torch.remainder(phase - phase.new_tensor([0.0, 0.5]), 1.0)
  weight = torch.minimum((phase - 0.20) / 0.10, (1.0 - phase) / 0.10).clamp(0, 1)
  return torch.where(moving(env)[:, None], weight, torch.ones_like(weight))


def sole_support_cost(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
  endpoint: bool = False,
  tolerance_degrees: float = 5.0,
  scale_degrees: float = 10.0,
) -> torch.Tensor:
  """Constrain soles only in planned stance, without imposing a swing trajectory."""
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
  indices = [
    next(
      i
      for i, name in enumerate(sensor.primary_names)
      if name.endswith(f"{side}_ankle_link")
    )
    for side in ("right", "left")
  ]
  measured = (sensor.data.current_contact_time[:, indices] / 0.08).clamp(0, 1)
  measured *= (sensor.data.found[:, indices] > 0).float()
  # Actual stance can outlast or shift relative to the clock-based gait.
  # Keep the planned floor to stop contact avoidance, and also cover sustained
  # real contact even when the phase says this foot should be swinging.
  return (error.square() * torch.maximum(planned, measured)).mean(-1)


def ankle_limit_margin(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Penalize measured ankle angles within 8 degrees of a physical stop."""
  robot = env.scene[asset_cfg.name]
  q = robot.data.joint_pos[:, asset_cfg.joint_ids]
  limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  distance = torch.minimum(q - limits[..., 0], limits[..., 1] - q)
  return (
    ((math.radians(8.0) - distance).clamp_min(0) / math.radians(8.0)).square().mean(-1)
  )
