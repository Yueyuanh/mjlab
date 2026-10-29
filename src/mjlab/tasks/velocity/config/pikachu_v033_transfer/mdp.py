"""Goal-conditioned V033 transitions; no demonstration or reference trajectory."""

import math
from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl.robot import (
  HEIGHT,
  JOINT_POSE,
  PITCH,
)
from mjlab.tasks.velocity.config.pikachu_v033_crawl_e2.mdp import sole_geometry
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz

STAND_HEIGHT = 0.259412


def endpoint_joints(names: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
  stand = torch.tensor(
    [
      math.radians(53)
      if n == "right_arm_roll_joint"
      else -math.radians(53)
      if n == "left_arm_roll_joint"
      else 0.0
      for n in names
    ],
    device=device,
  )
  crawl = torch.tensor([JOINT_POSE.get(n, 0.0) for n in names], device=device)
  return stand, crawl


class PostureCommand(CommandTerm):
  """One extra network input: 0 = biped stand, 1 = quadruped support."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.value = torch.zeros(self.num_envs, 1, device=self.device)
    self._gui = None

  @property
  def command(self):
    return self.value

  def _resample_command(self, env_ids):
    self.value[env_ids, 0] = (
      torch.rand(len(env_ids), device=self.device) < 0.5
    ).float()

  def _update_metrics(self):
    pass

  def _update_command(self, env_ids):
    if self._gui is not None:
      enabled, target, get_index = self._gui
      if enabled.value:
        index = get_index()
        if env_ids is None or bool((env_ids == index).any()):
          self.value[index, 0] = float(target.value == "趴下（四足）")

  def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
    with server.gui.add_folder("姿态状态"):
      enabled = server.gui.add_checkbox("手动切换", initial_value=True)
      target = server.gui.add_dropdown(
        "目标状态",
        options=("起立（双足）", "趴下（四足）"),
        initial_value="趴下（四足）",
      )
    self._gui = (enabled, target, get_env_idx)


@dataclass(kw_only=True)
class PostureCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (4.0, 6.0)

  def build(self, env):
    return PostureCommand(self, env)


def state(env):
  command = env.command_manager.get_command("posture")
  assert command is not None
  return command[:, 0]


def target_errors(env):
  robot = env.scene["robot"]
  ids, names = robot.find_joints_by_actuator_names((".*",))
  stand, crawl = endpoint_joints(names, env.device)
  flag = state(env)
  target = torch.where(flag[:, None] > 0.5, crawl, stand)
  pose = (robot.data.joint_pos[:, ids] - target).square().mean(-1)
  pitch = flag * PITCH
  gravity = torch.stack((pitch.sin(), torch.zeros_like(pitch), -pitch.cos()), -1)
  orientation = (robot.data.projected_gravity_b - gravity).square().sum(-1)
  z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  height = (z - (STAND_HEIGHT + flag * (HEIGHT - STAND_HEIGHT))).abs()
  return pose, orientation, height


def goal_score(env):
  pose, orientation, height = target_errors(env)
  return torch.exp(-pose / 0.5**2 - orientation / 0.35**2 - (height / 0.035).square())


def goal_distance(env):
  pose, orientation, height = target_errors(env)
  # End-state joint accuracy must not force early hand retraction during rise.
  readiness = torch.exp(-orientation / 0.3**2)
  return readiness * pose.sqrt() + orientation.sqrt() + height / 0.05


def orientation_progress(env):
  """Command-aware change of orientation potential; holding pays zero."""
  current = -target_errors(env)[1]
  flag = state(env)
  previous = getattr(env, "_transfer_potential", current)
  previous_flag = getattr(env, "_transfer_potential_flag", flag)
  fresh = (env.episode_length_buf <= 1) | (flag != previous_flag)
  reward = torch.where(
    fresh, torch.zeros_like(current), (current - previous) / env.step_dt
  ).clamp(-2.0, 2.0)
  env._transfer_potential = current.clone()
  env._transfer_potential_flag = flag.clone()
  return reward


def settled_sole_cost(env):
  robot = env.scene["robot"]
  ids = robot.find_bodies(("right_ankle_link", "left_ankle_link"), preserve_order=True)[
    0
  ]
  tilt = sole_geometry(robot.data.body_link_quat_w[:, ids])[0]
  contact = (env.scene["feet_ground_contact"].data.found[:, :2] > 0).float()
  ready = torch.exp(-target_errors(env)[1] / 0.3**2)
  return ((tilt - math.radians(5)).clamp_min(0) / math.radians(10)).square().mul(
    contact
  ).mean(-1) * ready


def orientation_cost(env):
  return target_errors(env)[1]


def height_cost(env):
  return target_errors(env)[2] / 0.05


def contact_score(env):
  sensor = env.scene["feet_ground_contact"]
  found = sensor.data.found > 0
  rear = found[:, :2].all(-1)
  front = found[:, 2:].all(-1)
  front_clear = (~found[:, 2:]).all(-1)
  expected = rear & torch.where(state(env) > 0.5, front, front_clear)
  # Contact bonuses only pay near the complete commanded posture.
  return expected.float() * goal_score(env)


def quiet_at_goal(env):
  robot = env.scene["robot"]
  speed = robot.data.root_link_lin_vel_w.square().sum(
    -1
  ) + 0.1 * robot.data.root_link_ang_vel_w.square().sum(-1)
  return goal_score(env) * torch.exp(-speed / 0.1**2)


def motion_cost(env):
  robot = env.scene["robot"]
  # A backstop, not a trajectory: all paths within physical limits remain free.
  vz = robot.data.root_link_lin_vel_w[:, 2].abs()
  return (vz - 0.15).clamp_min(
    0
  ).square() + 0.01 * robot.data.root_link_ang_vel_w.square().sum(-1)


def unsafe_state(env):
  robot = env.scene["robot"]
  z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  torso_contact = (env.scene["torso_ground"].data.found > 0).flatten(1).any(-1)
  return (
    (z < 0.08)
    | (z > 0.6)
    | ~torch.isfinite(robot.data.joint_pos).all(-1)
    | torso_contact
  )


def body_velocity(env, angular=False):
  robot = env.scene["robot"]
  return quat_apply_inverse(
    robot.data.root_link_quat_w,
    robot.data.root_link_ang_vel_w if angular else robot.data.root_link_lin_vel_w,
  )


def reset_endpoints(env, env_ids, crawl_probability=0.5, fixed_state=None):
  """Only reset teleports; evaluation counts no reset as a transition."""
  robot = env.scene["robot"]
  n = len(env_ids)
  flag = (torch.rand(n, device=env.device) < crawl_probability).float()
  if fixed_state is not None:
    flag.fill_(fixed_state)
  reset_state = getattr(env, "_transfer_reset_state", None)
  if reset_state is None:
    reset_state = torch.zeros(env.num_envs, device=env.device)
    env._transfer_reset_state = reset_state
  reset_state[env_ids] = flag
  stand, crawl = endpoint_joints(robot.joint_names, env.device)
  joints = torch.where(flag[:, None] > 0.5, crawl, stand)
  robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
  root = robot.data.default_root_state[env_ids].clone()
  root[:, :3] = env.scene.env_origins[env_ids]
  root[:, 2] += STAND_HEIGHT + flag * (HEIGHT - STAND_HEIGHT) + 0.003
  pitch = flag * PITCH
  zero = torch.zeros_like(flag)
  root[:, 3:7] = quat_from_euler_xyz(zero, pitch, zero)
  root[:, 7:] = 0
  robot.write_root_link_pose_to_sim(root[:, :7], env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(root[:, 7:13], env_ids=env_ids)
