from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def linear_velocity_huber_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  delta: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return a non-saturating Huber cost for planar velocity tracking.

  The exponential tracking reward is precise near the target, but its gradient
  approaches zero for a large error.  This companion cost keeps a constant
  corrective gradient outside ``delta`` while remaining quadratic near zero.
  Errors are normalized by ``delta``, so the result is dimensionless.
  """
  if delta <= 0.0:
    raise ValueError(f"delta must be positive, got {delta}")
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  normalized = (command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]) / delta
  abs_error = torch.abs(normalized)
  huber = torch.where(
    abs_error <= 1.0,
    0.5 * torch.square(normalized),
    abs_error - 0.5,
  )
  return torch.sum(huber, dim=1)


def angular_velocity_huber_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  delta: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return a non-saturating Huber cost for yaw-rate tracking.

  The exponential tracking reward has a gradient that vanishes once the yaw
  error is a few times ``std``; a policy that has already spun up cannot feel
  a pull back toward the commanded heading.  This companion cost keeps a
  constant corrective gradient outside ``delta`` while staying quadratic near
  zero, mirroring :func:`linear_velocity_huber_error` on the yaw axis.

  Only the z (yaw) component is used: roll/pitch rates are handled by
  ``body_ang_vel`` and are not commanded.
  """
  if delta <= 0.0:
    raise ValueError(f"delta must be positive, got {delta}")
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  normalized = (command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]) / delta
  abs_error = torch.abs(normalized)
  return torch.where(
    abs_error <= 1.0,
    0.5 * torch.square(normalized),
    abs_error - 0.5,
  )


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def alternating_feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_air_time: float,
  std: float,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Reward timed landings only when another foot already supports the robot.

  Unlike a per-step air-time reward, this term cannot be accumulated by keeping
  both feet airborne. Simultaneous landings are also excluded because neither
  foot has an established contact at that point.
  """
  if std <= 0.0:
    raise ValueError(f"std must be positive, got {std}")

  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  last_air_time = sensor_data.last_air_time
  current_contact_time = sensor_data.current_contact_time
  assert current_air_time is not None
  assert last_air_time is not None
  assert current_contact_time is not None

  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  established_contact = current_contact_time > env.step_dt + 1e-6
  support_count = torch.sum(established_contact, dim=1, keepdim=True)
  has_other_support = support_count - established_contact.long() > 0
  valid_landing = first_contact & has_other_support

  timing_score = torch.exp(-torch.square((last_air_time - target_air_time) / std))
  reward = torch.sum(timing_score * valid_landing.float(), dim=1)

  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  num_valid_landings = torch.sum(valid_landing.float())
  mean_landing_air_time = torch.sum(
    last_air_time * valid_landing.float()
  ) / torch.clamp(num_valid_landings, min=1)
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  env.extras["log"]["Metrics/alternating_landing_air_time_mean"] = mean_landing_air_time

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      reward *= (total_command > command_threshold).float()
  return reward


def short_step_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_air_time: float,
  std: float,
  command_name: str | None = None,
  command_threshold: float = 0.15,
) -> torch.Tensor:
  """Penalize supported landings whose swing time is too short.

  This is a lower-bound hinge, not another target reward.  It therefore gives
  a direct gradient against high-cadence shuffling while leaving sufficiently
  long steps unpenalized.  A landing is counted only if the other foot already
  provides established support, matching :func:`alternating_feet_air_time` and
  excluding simultaneous hopping landings.
  """
  if min_air_time <= 0.0:
    raise ValueError(f"min_air_time must be positive, got {min_air_time}")
  if std <= 0.0:
    raise ValueError(f"std must be positive, got {std}")

  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  last_air_time = sensor_data.last_air_time
  current_contact_time = sensor_data.current_contact_time
  assert last_air_time is not None
  assert current_contact_time is not None
  # ``Sensor.data`` 的泛型参数在这里解析不出来，显式收窄成 Tensor。
  air_time = cast("torch.Tensor", last_air_time)

  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  established_contact = current_contact_time > env.step_dt + 1e-6
  support_count = torch.sum(established_contact, dim=1, keepdim=True)
  has_other_support = support_count - established_contact.long() > 0
  valid_landing = first_contact & has_other_support

  shortfall = torch.clamp((min_air_time - air_time) / std, min=0.0)
  cost = torch.sum(torch.square(shortfall) * valid_landing.float(), dim=1)

  num_landings = valid_landing.float().sum().clamp(min=1.0)
  env.extras["log"]["Metrics/short_step_air_time"] = (
    air_time * valid_landing.float()
  ).sum() / num_landings

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_speed = torch.norm(command[:, :2], dim=1)
      cost *= (linear_speed > command_threshold).float()
  return cost


def feet_flight_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  grace_period: float = 0.04,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize control steps where all feet remain airborne."""
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None
  cost = torch.all(current_air_time > grace_period, dim=1).float()
  env.extras["log"]["Metrics/flight_ratio"] = torch.mean(cost)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      cost *= (total_command > command_threshold).float()
  return cost


def base_vertical_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize vertical base velocity to suppress bouncing and hopping."""
  asset: Entity = env.scene[asset_cfg.name]
  vertical_velocity = asset.data.root_link_lin_vel_w[:, 2]
  env.extras["log"]["Metrics/base_vertical_velocity_mean"] = torch.mean(
    torch.abs(vertical_velocity)
  )
  return torch.square(vertical_velocity)


def base_height_penalty(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize normalized root-height error around a desired crouched posture."""
  if std <= 0.0:
    raise ValueError(f"std must be positive, got {std}")
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  env.extras["log"]["Metrics/base_height_mean"] = torch.mean(height)
  return torch.square((height - target_height) / std)


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def feet_lateral_distance(
  env: ManagerBasedRlEnv,
  target_distance: float,
  std: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation of the left-right foot separation from a target.

  Use as a **penalty** (negative weight) to hold a stance width. ``asset_cfg``
  must resolve to exactly two foot sites.

  **Measured in the base heading frame, not world.** The world-frame y
  separation shrinks by ``cos(yaw)`` as the robot turns, so a world-frame term
  would silently loosen as the robot rotates — and would read near-zero when
  the robot faces sideways.

  **Why not just tighten ``pose``?** ``variable_posture`` is
  ``exp(-mean(err²/std²))``, which saturates: once a joint is a few std off,
  the term is ~0 and its gradient with it. A stance width that is wrong by
  2x the std is exactly that regime — the reward has already given up. This
  term is the unbounded quadratic ``(err/std)²`` instead, so the gradient keeps
  growing with the error.

  Uses ``|Δy|`` rather than a signed difference so the result does not depend
  on which site resolves first. The cost of that choice: legs fully crossed
  (left foot at the right foot's nominal spot and vice versa) reads as a
  correct separation. That posture is already penalized by ``self_collisions``.

  ⚠️ **Gate on the command — this term must not be applied while walking.**
  Pass ``command_name`` and only the near-zero-command envs pay it. The gate is
  **inverted** relative to ``feet_flight_time``: there the command *enables*
  the cost, here it *disables* it.

  Why the gate is not optional: a walk cycle needs the feet to pass close
  together (the swing leg travels under the body), and ``hip_roll`` is also
  what the balance controller uses for lateral support and turning. An
  ungated constant outward push on ``hip_roll`` fights the gait directly, and
  the policy's way out is to stiffen into a march — feet lift on schedule but
  the body stops translating. Measured on v03 (r05, ungated, 15.3k iters vs
  r04, no term, 30k iters): ``alternating_air_time`` 0.091 → 0.155 (lifts
  *more*) while ``track_linear_velocity`` 0.72 → 0.38 and
  ``Metrics/twist/error_vel_xy`` 0.63 → 1.06. Net displacement collapses.
  Gated, the walking envs are bit-identical to having no term at all.
  """
  asset: Entity = env.scene[asset_cfg.name]
  site_ids = asset_cfg.site_ids
  assert isinstance(site_ids, list) and len(site_ids) == 2, (
    f"feet_lateral_distance needs exactly 2 foot sites, got {site_ids}"
  )
  pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, 2, 3]
  root_quat_w = asset.data.root_link_pose_w[:, 3:7]  # [B, 4] wxyz
  # 只去 yaw：俯仰/横滚会让 base 的 y 轴倾斜，但「双脚张开多少」是水平面上的量。
  delta_b = quat_apply_inverse(yaw_quat(root_quat_w), pos_w[:, 0] - pos_w[:, 1])
  separation = torch.abs(delta_b[:, 1])
  cost = torch.square((separation - target_distance) / std)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      standing = (total_command < command_threshold).float()
      cost = cost * standing
      # 门控后的指标才有意义：混着走路 env 的均值会被步态周期污染，
      # 看不出站姿到底挪没挪（r05 就是靠这个才发现「开度涨了但没到位」）。
      n_standing = standing.sum().clamp(min=1.0)
      env.extras["log"]["Metrics/foot_separation"] = (
        separation * standing
      ).sum() / n_standing
      env.extras["log"]["Metrics/stance_gate_frac"] = standing.mean()
      return cost

  env.extras["log"]["Metrics/foot_separation"] = separation.mean()
  return cost


def feet_lateral_separation_min(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_distance: float,
  std: float,
  command_name: str,
  command_threshold: float = 0.15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize a narrow walking stance only during double support.

  A constant target distance fights the swing leg as it passes under the body.
  This lower-bound hinge is active only while both feet contact the ground and
  a linear command is present.  The policy is free to swing naturally, but it
  is encouraged to land with a useful support polygon instead of converging to
  the narrow 9--11 cm stance observed in Pikachu v03 r04/r06.
  """
  if min_distance <= 0.0:
    raise ValueError(f"min_distance must be positive, got {min_distance}")
  if std <= 0.0:
    raise ValueError(f"std must be positive, got {std}")

  asset: Entity = env.scene[asset_cfg.name]
  site_ids = asset_cfg.site_ids
  assert isinstance(site_ids, list) and len(site_ids) == 2, (
    f"feet_lateral_separation_min needs exactly 2 foot sites, got {site_ids}"
  )
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  assert sensor.data.found.shape[1] == 2, (
    "feet_lateral_separation_min needs exactly 2 contact primaries, "
    f"got {sensor.data.found.shape[1]}"
  )

  pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  root_quat_w = asset.data.root_link_pose_w[:, 3:7]
  delta_b = quat_apply_inverse(yaw_quat(root_quat_w), pos_w[:, 0] - pos_w[:, 1])
  separation = torch.abs(delta_b[:, 1])
  shortfall = torch.clamp((min_distance - separation) / std, min=0.0)

  double_support = torch.all(sensor.data.found > 0, dim=1).float()
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  moving = (torch.norm(command[:, :2], dim=1) > command_threshold).float()
  active = double_support * moving
  cost = torch.square(shortfall) * active

  n_active = active.sum().clamp(min=1.0)
  env.extras["log"]["Metrics/walking_double_support_width"] = (
    separation * active
  ).sum() / n_active
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


class standing_posture_l2:
  """Unbounded default-pose cost gated to near-zero commands.

  ``variable_posture`` is an exponential reward and can saturate at zero once
  several joints drift far from their defaults.  This quadratic companion
  keeps a restoring gradient at large errors and is active only for standing
  environments, so normal walking joint excursions remain untouched.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, std = resolve_matching_names_values(
      data=cfg.params["std"],
      list_of_strings=joint_names,
    )
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)
    if torch.any(self.std <= 0.0):
      raise ValueError("standing_posture_l2 std values must all be positive")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.1,
  ) -> torch.Tensor:
    del std  # Resolved once in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing = (total_command < command_threshold).float()

    current = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired = self.default_joint_pos[:, asset_cfg.joint_ids]
    cost = torch.mean(torch.square((current - desired) / self.std), dim=1)
    return cost * standing


class joint_torque_soft_limit:
  """Penalize only the portion of joint torque above a soft utilization limit."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, limits = resolve_matching_names_values(
      data=cfg.params["effort_limits"],
      list_of_strings=joint_names,
    )
    self.effort_limits = torch.tensor(limits, device=env.device, dtype=torch.float32)
    if torch.any(self.effort_limits <= 0.0):
      raise ValueError("joint_torque_soft_limit effort limits must all be positive")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    effort_limits,
    soft_ratio: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del effort_limits  # Resolved once in __init__.
    if not 0.0 < soft_ratio < 1.0:
      raise ValueError(f"soft_ratio must be in (0, 1), got {soft_ratio}")
    asset: Entity = env.scene[asset_cfg.name]
    torque_ratio = (
      torch.abs(asset.data.qfrc_actuator[:, asset_cfg.joint_ids]) / self.effort_limits
    )
    normalized_excess = torch.clamp(
      (torque_ratio - soft_ratio) / (1.0 - soft_ratio), min=0.0
    )
    return torch.mean(torch.square(normalized_excess), dim=1)
