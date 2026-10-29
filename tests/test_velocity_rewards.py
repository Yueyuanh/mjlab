"""Tests for velocity task reward functions."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, PropertyMock

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastData, RayCastSensor
from mjlab.tasks.velocity.mdp.rewards import (
  alternating_feet_air_time,
  angular_velocity_huber_error,
  base_height_penalty,
  base_vertical_velocity_penalty,
  feet_flight_time,
  feet_lateral_separation_min,
  joint_torque_soft_limit,
  linear_velocity_huber_error,
  short_step_air_time,
  standing_posture_l2,
  upright,
)
from mjlab.utils.lab_api.math import quat_from_euler_xyz


def _identity_quat(B: int) -> torch.Tensor:
  """(w, x, y, z) = (1, 0, 0, 0)."""
  q = torch.zeros(B, 4)
  q[:, 0] = 1.0
  return q


def _quat_from_roll(roll_rad: float, B: int = 1) -> torch.Tensor:
  roll = torch.full((B,), roll_rad)
  zero = torch.zeros(B)
  return quat_from_euler_xyz(roll, zero, zero)


def _quat_from_pitch(pitch_rad: float, B: int = 1) -> torch.Tensor:
  pitch = torch.full((B,), pitch_rad)
  zero = torch.zeros(B)
  return quat_from_euler_xyz(zero, pitch, zero)


def _make_env_and_reward(
  terrain_sensor_names: tuple[str, ...] | None = None,
  body_quat_w: torch.Tensor | None = None,
  terrain_hit_z: float = 0.0,
  terrain_slope_x: float = 0.0,
):
  """Build mocked env + upright reward instance.

  Args:
    terrain_sensor_names: If set, enables terrain-aware mode.
    body_quat_w: [B, 4] root orientation. Defaults to identity.
    terrain_hit_z: Z value for flat terrain hits.
    terrain_slope_x: Slope in X (z = terrain_slope_x * x).
  """
  B = 1 if body_quat_w is None else body_quat_w.shape[0]
  if body_quat_w is None:
    body_quat_w = _identity_quat(B)

  # Mock asset data. Use explicit asset_cfg with no body_names so
  # body_ids stays None and the reward uses root_link_quat_w.
  asset = MagicMock()
  asset.data.root_link_quat_w = body_quat_w
  asset.data.root_link_pos_w = torch.zeros(B, 3)
  asset.data.gravity_vec_w = torch.tensor([0.0, 0.0, -1.0]).expand(B, 3)
  asset_cfg = SceneEntityCfg("robot", body_names=None, body_ids=[])

  # Mock terrain sensor if needed.
  sensors: dict = {"robot": asset}
  if terrain_sensor_names is not None:
    N = 100
    torch.manual_seed(0)
    hit_pos = torch.zeros(B, N, 3)
    hit_pos[:, :, 0] = torch.randn(B, N)
    hit_pos[:, :, 1] = torch.randn(B, N)
    hit_pos[:, :, 2] = terrain_hit_z + terrain_slope_x * hit_pos[:, :, 0]

    raycast_sensor = MagicMock(spec=RayCastSensor)
    raycast_data = RayCastData(
      distances=torch.ones(B, N),
      normals_w=torch.zeros(B, N, 3),
      hit_pos_w=hit_pos,
      pos_w=torch.zeros(B, 3),
      quat_w=torch.zeros(B, 4),
      frame_pos_w=torch.zeros(B, 1, 3),
      frame_quat_w=torch.zeros(B, 1, 4),
    )
    type(raycast_sensor).data = PropertyMock(return_value=raycast_data)
    for name in terrain_sensor_names:
      sensors[name] = raycast_sensor

  env = MagicMock()
  env.scene.__getitem__ = MagicMock(side_effect=lambda n: sensors[n])

  params: dict = {"std": 1.0, "asset_cfg": asset_cfg}
  if terrain_sensor_names is not None:
    params["terrain_sensor_names"] = terrain_sensor_names
  cfg = MagicMock(spec=RewardTermCfg)
  cfg.params = params

  reward_fn = upright(cfg, env)
  return env, reward_fn, params


def test_world_up_identity_gives_max_reward():
  """Perfectly upright robot on flat ground → reward ≈ 1."""
  env, reward, params = _make_env_and_reward()
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])
  assert r.shape == (1,)
  assert r.item() > 0.99


def test_world_up_tilted_gives_lower_reward():
  """30° roll → reward significantly below 1."""
  quat = _quat_from_roll(math.radians(30))
  env, reward, params = _make_env_and_reward(body_quat_w=quat)
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])
  assert r.item() < 0.8


def test_terrain_aware_aligned_with_slope():
  """Robot pitched to match a slope → terrain-aware reward ≈ 1."""
  slope = 0.5  # z = 0.5 * x
  tilt = math.atan(slope)  # Pitch to match slope in XZ plane.
  quat = _quat_from_pitch(-tilt)
  env, reward, params = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
    terrain_slope_x=slope,
  )
  r = reward(
    env,
    std=params["std"],
    asset_cfg=params["asset_cfg"],
    terrain_sensor_names=params["terrain_sensor_names"],
  )
  # Should be close to 1 since robot matches terrain.
  assert r.item() > 0.9


def test_terrain_aware_upright_on_slope_penalized():
  """Robot staying vertical on a slope → terrain-aware reward < 1."""
  slope = 0.5
  quat = _identity_quat(1)  # Robot is world-vertical, not matching slope.
  env, reward, params = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
    terrain_slope_x=slope,
  )
  r = reward(
    env,
    std=params["std"],
    asset_cfg=params["asset_cfg"],
    terrain_sensor_names=params["terrain_sensor_names"],
  )
  # Should be penalized since robot doesn't match terrain.
  assert r.item() < 0.95


def test_terrain_aware_flat_ground_matches_world_up():
  """On flat terrain, terrain-aware and world-up should give same reward."""
  quat = _quat_from_roll(math.radians(15))
  env_t, reward_t, params_t = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
  )
  env_w, reward_w, params_w = _make_env_and_reward(body_quat_w=quat)

  r_terrain = reward_t(
    env_t,
    std=params_t["std"],
    asset_cfg=params_t["asset_cfg"],
    terrain_sensor_names=params_t["terrain_sensor_names"],
  )
  r_world = reward_w(env_w, std=params_w["std"], asset_cfg=params_w["asset_cfg"])

  torch.testing.assert_close(r_terrain, r_world, atol=0.02, rtol=0.02)


def test_batch_consistency():
  """Multiple envs with different orientations get independent rewards."""
  B = 4
  quats = torch.zeros(B, 4)
  quats[:, 0] = 1.0  # All identity.
  # Tilt env 2 by 45°.
  quats[2] = _quat_from_roll(math.radians(45))[0]

  env, reward, params = _make_env_and_reward(body_quat_w=quats)
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])

  assert r.shape == (B,)
  # Env 0, 1, 3 should be ~1, env 2 should be lower.
  assert r[0].item() > 0.99
  assert r[1].item() > 0.99
  assert r[2].item() < 0.7
  assert r[3].item() > 0.99


def _make_gait_reward_env() -> MagicMock:
  sensor = MagicMock()
  sensor.data.current_air_time = torch.tensor(
    [
      [0.0, 0.0],
      [0.0, 0.0],
      [0.1, 0.1],
      [0.0, 0.0],
    ]
  )
  sensor.data.last_air_time = torch.tensor(
    [
      [0.22, 0.0],
      [0.22, 0.22],
      [0.0, 0.0],
      [0.22, 0.0],
    ]
  )
  sensor.data.current_contact_time = torch.tensor(
    [
      [0.02, 0.2],
      [0.02, 0.02],
      [0.0, 0.0],
      [0.02, 0.2],
    ]
  )
  sensor.compute_first_contact.return_value = torch.tensor(
    [
      [True, False],
      [True, True],
      [False, False],
      [True, False],
    ]
  )

  env = MagicMock()
  env.scene = {"feet": sensor}
  env.step_dt = 0.02
  env.extras = {"log": {}}
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.5, 0.0, 0.0],
      [0.5, 0.0, 0.0],
      [0.5, 0.0, 0.0],
      [0.0, 0.0, 0.0],
    ]
  )
  return env


def test_alternating_air_time_requires_existing_support():
  """Only a commanded landing with another established stance foot is rewarded."""
  env = _make_gait_reward_env()

  reward = alternating_feet_air_time(
    env,
    sensor_name="feet",
    target_air_time=0.22,
    std=0.08,
    command_name="twist",
    command_threshold=0.15,
  )

  torch.testing.assert_close(reward, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_flight_time_only_penalizes_sustained_double_flight():
  env = _make_gait_reward_env()

  cost = feet_flight_time(
    env,
    sensor_name="feet",
    grace_period=0.04,
    command_name="twist",
    command_threshold=0.15,
  )

  torch.testing.assert_close(cost, torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_short_step_air_time_penalizes_only_supported_moving_landing():
  env = _make_gait_reward_env()
  env.scene["feet"].data.last_air_time[0, 0] = 0.10

  cost = short_step_air_time(
    env,
    sensor_name="feet",
    min_air_time=0.18,
    std=0.08,
    command_name="twist",
    command_threshold=0.15,
  )

  torch.testing.assert_close(cost, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_linear_velocity_huber_keeps_large_error_gradient():
  env = MagicMock()
  asset = MagicMock()
  asset.data.root_link_lin_vel_b = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.4, 0.0]])
  env.scene = {"robot": asset}
  env.command_manager.get_command.return_value = torch.tensor(
    [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]
  )

  cost = linear_velocity_huber_error(env, command_name="twist", delta=0.2)

  torch.testing.assert_close(cost, torch.tensor([0.5, 2.0]))


def test_angular_velocity_huber_keeps_large_yaw_error_gradient():
  env = MagicMock()
  asset = MagicMock()
  asset.data.root_link_ang_vel_b = torch.tensor([[0.0, 0.0, 0.25], [0.0, 0.0, -0.75]])
  env.scene = {"robot": asset}
  env.command_manager.get_command.return_value = torch.tensor(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.25]]
  )

  cost = angular_velocity_huber_error(env, command_name="twist", delta=0.25)

  torch.testing.assert_close(cost, torch.tensor([0.5, 3.5]))


def test_walking_stance_width_is_double_support_lower_bound():
  env = MagicMock()
  asset = MagicMock()
  asset.data.site_pos_w = torch.tensor(
    [
      [[0.0, 0.05, 0.0], [0.0, -0.05, 0.0]],
      [[0.0, 0.08, 0.0], [0.0, -0.08, 0.0]],
      [[0.0, 0.05, 0.0], [0.0, -0.05, 0.0]],
    ]
  )
  asset.data.root_link_pose_w = torch.zeros(3, 7)
  asset.data.root_link_pose_w[:, 3] = 1.0
  sensor = MagicMock()
  sensor.data.found = torch.tensor([[1, 1], [1, 1], [1, 0]])
  env.scene.__getitem__ = MagicMock(
    side_effect=lambda name: {"robot": asset, "feet": sensor}[name]
  )
  env.command_manager.get_command.return_value = torch.tensor(
    [[0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]
  )
  env.extras = {"log": {}}

  cost = feet_lateral_separation_min(
    env,
    sensor_name="feet",
    min_distance=0.14,
    std=0.04,
    command_name="twist",
    asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
  )

  torch.testing.assert_close(cost, torch.tensor([1.0, 0.0, 0.0]))


def test_standing_posture_l2_is_unbounded_and_command_gated():
  env = MagicMock()
  env.device = "cpu"
  asset = MagicMock()
  asset.data.default_joint_pos = torch.zeros(2, 2)
  asset.data.joint_pos = torch.tensor([[0.1, 0.0], [0.1, 0.2]])
  asset.find_joints.return_value = ([0, 1], ["hip_joint", "knee_joint"])
  env.scene = {"robot": asset}
  env.command_manager.get_command.return_value = torch.tensor(
    [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]
  )
  asset_cfg = SceneEntityCfg(
    "robot",
    joint_names=("hip_joint", "knee_joint"),
    joint_ids=[0, 1],
  )
  cfg = MagicMock(spec=RewardTermCfg)
  cfg.params = {
    "asset_cfg": asset_cfg,
    "std": {".*": 0.1},
    "command_name": "twist",
  }
  reward = standing_posture_l2(cfg, env)

  cost = reward(
    env,
    std=cfg.params["std"],
    asset_cfg=asset_cfg,
    command_name="twist",
  )

  torch.testing.assert_close(cost, torch.tensor([0.5, 0.0]))


def test_joint_torque_soft_limit_is_free_below_threshold():
  env = MagicMock()
  env.device = "cpu"
  asset = MagicMock()
  asset.data.qfrc_actuator = torch.tensor([[10.0, 9.0], [12.5, 4.5]])
  asset.find_joints.return_value = ([0, 1], ["hip_pitch_joint", "ankle_joint"])
  env.scene = {"robot": asset}
  asset_cfg = SceneEntityCfg(
    "robot",
    joint_names=("hip_pitch_joint", "ankle_joint"),
    joint_ids=[0, 1],
  )
  cfg = MagicMock(spec=RewardTermCfg)
  cfg.params = {
    "asset_cfg": asset_cfg,
    "effort_limits": {".*hip_pitch.*": 12.5, ".*ankle.*": 9.0},
  }
  reward = joint_torque_soft_limit(cfg, env)

  cost = reward(
    env,
    effort_limits=cfg.params["effort_limits"],
    soft_ratio=0.8,
    asset_cfg=asset_cfg,
  )

  torch.testing.assert_close(cost, torch.tensor([0.5, 0.5]))


def test_base_vertical_velocity_penalty_is_squared():
  env = MagicMock()
  asset = MagicMock()
  asset.data.root_link_lin_vel_w = torch.tensor(
    [[0.0, 0.0, 0.2], [0.0, 0.0, -0.3], [0.0, 0.0, 0.0]]
  )
  env.scene = {"robot": asset}
  env.extras = {"log": {}}

  cost = base_vertical_velocity_penalty(env)

  torch.testing.assert_close(cost, torch.tensor([0.04, 0.09, 0.0]))


def test_base_height_penalty_tracks_lower_target():
  env = MagicMock()
  asset = MagicMock()
  asset.data.root_link_pos_w = torch.tensor(
    [[0.0, 0.0, 0.17], [0.0, 0.0, 0.185], [0.0, 0.0, 0.155]]
  )
  env.scene = MagicMock()
  env.scene.__getitem__.return_value = asset
  env.scene.env_origins = torch.tensor(
    [[0.0, 0.0, 0.02], [0.0, 0.0, 0.02], [0.0, 0.0, 0.02]]
  )
  env.extras = {"log": {}}

  cost = base_height_penalty(env, target_height=0.15, std=0.015)

  torch.testing.assert_close(cost, torch.tensor([0.0, 1.0, 1.0]))
