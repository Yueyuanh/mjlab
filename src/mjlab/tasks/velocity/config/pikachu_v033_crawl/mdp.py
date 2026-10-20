"""Crawl terms use a horizontal heading frame and a tilted rest posture."""

import math
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_mul,
)

from .robot import HEIGHT, PITCH

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def crawl_heading(quat: torch.Tensor) -> torch.Tensor:
  """Use the torso +Z projection; stays continuous across 90-degree pitch."""
  axis = quat.new_tensor([0.0, 0.0, 1.0]).expand(quat.shape[0], -1)
  forward = quat_apply(quat, axis)
  yaw = torch.atan2(forward[:, 1], forward[:, 0])
  return quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)


def heading_lin_vel(env: "ManagerBasedRlEnv") -> torch.Tensor:
  robot = env.scene["robot"]
  return quat_apply_inverse(
    crawl_heading(robot.data.root_link_quat_w), robot.data.root_link_lin_vel_w
  )


def heading_ang_vel(env: "ManagerBasedRlEnv") -> torch.Tensor:
  robot = env.scene["robot"]
  return quat_apply_inverse(
    crawl_heading(robot.data.root_link_quat_w), robot.data.root_link_ang_vel_w
  )


def tracking(env: "ManagerBasedRlEnv", angular: bool = False) -> torch.Tensor:
  command = env.command_manager.get_command("twist")
  assert command is not None
  if angular:
    error = (command[:, 2] - heading_ang_vel(env)[:, 2]).square()
    return torch.exp(-error / 0.3**2)
  error = (command[:, :2] - heading_lin_vel(env)[:, :2]).square().sum(-1)
  return torch.exp(-error / 0.08**2)


def tracking_huber(env: "ManagerBasedRlEnv") -> torch.Tensor:
  command = env.command_manager.get_command("twist")
  assert command is not None
  error = ((command[:, :2] - heading_lin_vel(env)[:, :2]) / 0.10).abs()
  return torch.where(error < 1, 0.5 * error.square(), error - 0.5).sum(-1)


def moving(env: "ManagerBasedRlEnv") -> torch.Tensor:
  command = env.command_manager.get_command("twist")
  assert command is not None
  return (command[:, :2].norm(dim=-1) + command[:, 2].abs()) > 0.04


def phase_observation(env: "ManagerBasedRlEnv") -> torch.Tensor:
  phase = env.episode_length_buf * env.step_dt / 1.6
  return (
    torch.stack((torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)), -1)
    * moving(env)[:, None]
  )


def swing_mask(env: "ManagerBasedRlEnv") -> torch.Tensor:
  # Sensor/site order: rear right, rear left, front right, front left.
  # Swing sequence: rear right -> front left -> rear left -> front right.
  phase = env.episode_length_buf[:, None] * env.step_dt / 1.6
  offsets = phase.new_tensor([0.0, 0.5, 0.75, 0.25])
  return (torch.remainder(phase - offsets, 1.0) < 0.20) & moving(env)[:, None]


def gait_contacts(env: "ManagerBasedRlEnv") -> torch.Tensor:
  sensor = env.scene["feet_ground_contact"]
  assert isinstance(sensor, ContactSensor) and sensor.data.found is not None
  return ((sensor.data.found > 0) == ~swing_mask(env)).float().mean(-1)


def swing_height(env: "ManagerBasedRlEnv", asset_cfg) -> torch.Tensor:
  robot = env.scene["robot"]
  heights = (
    robot.data.site_pos_w[:, asset_cfg.site_ids, 2] - env.scene.env_origins[:, 2:3]
  )
  target = swing_mask(env).float() * 0.02
  return ((heights - target) / 0.02).square().mean(-1)


def orientation_error(env: "ManagerBasedRlEnv") -> torch.Tensor:
  gravity = env.scene["robot"].data.projected_gravity_b
  target = gravity.new_tensor([math.sin(PITCH), 0, -math.cos(PITCH)])
  return (gravity - target).square().sum(-1)


def fell_over(env: "ManagerBasedRlEnv") -> torch.Tensor:
  z = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return (orientation_error(env) > 0.7) | (z < 0.10)


def posture(env: "ManagerBasedRlEnv") -> torch.Tensor:
  robot = env.scene["robot"]
  ids = robot.find_joints(
    (
      r".*(hip_pitch|hip_roll|hip_yaw|knee_up|ankle|arm_pitch|arm_roll|arm_yaw|elbow)_joint",
    )
  )[0]
  return (
    (robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids])
    .square()
    .mean(-1)
  )


def height_error(env: "ManagerBasedRlEnv") -> torch.Tensor:
  z = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return ((z - HEIGHT) / 0.05).square()


def support_count(env: "ManagerBasedRlEnv") -> torch.Tensor:
  sensor = env.scene["feet_ground_contact"]
  assert isinstance(sensor, ContactSensor) and sensor.data.found is not None
  return (sensor.data.found > 0).float().sum(-1)


def support_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
  return (3 - support_count(env)).clamp_min(0).square()


def contact_forces(env: "ManagerBasedRlEnv") -> torch.Tensor:
  sensor = env.scene["feet_ground_contact"]
  assert isinstance(sensor, ContactSensor) and sensor.data.force is not None
  force = sensor.data.force
  q = crawl_heading(env.scene["robot"].data.root_link_quat_w)[:, None, :].expand(
    -1, force.shape[1], -1
  )
  force = quat_apply_inverse(q, force)
  return (torch.sign(force) * torch.log1p(force.abs())).flatten(1)


def reset_base(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  yaw_range: tuple[float, float] = (-math.pi, math.pi),
  pose_range: dict[str, tuple[float, float]] | None = None,
) -> None:
  # The generic dummy-agent player supplies zero pose offsets to pin reset.
  offsets = pose_range or {}
  for axis in ("z", "roll", "pitch"):
    if offsets.get(axis, (0.0, 0.0)) != (0.0, 0.0):
      raise ValueError(f"Crawl reset does not support a {axis} pose offset")
  yaw_range = offsets.get("yaw", yaw_range)
  robot = env.scene["robot"]
  default = robot.data.default_root_state[env_ids].clone()
  n = len(env_ids)
  yaw = torch.rand(n, device=env.device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
  q = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
  default[:, 3:7] = quat_mul(q, default[:, 3:7])
  default[:, :3] += env.scene.env_origins[env_ids]
  for axis, name in enumerate(("x", "y")):
    low, high = offsets.get(name, (-0.1, 0.1))
    default[:, axis] += torch.rand(n, device=env.device) * (high - low) + low
  robot.write_root_link_pose_to_sim(default[:, :7], env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(default[:, 7:13], env_ids=env_ids)
