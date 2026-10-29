"""Smooth state targets and contact staging for gentle posture transitions."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.managers.command_manager import CommandTermCfg
from mjlab.tasks.velocity.config.pikachu_v033_transfer import mdp as base

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class SmoothPostureCommand(base.PostureCommand):
  """Slew a scalar endpoint target instead of exposing an instantaneous flip."""

  cfg: "SmoothPostureCommandCfg"

  def __init__(self, cfg: "SmoothPostureCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.goal = torch.zeros(self.num_envs, 1, device=self.device)
    self.start = torch.zeros_like(self.goal)
    self.elapsed = torch.zeros(self.num_envs, device=self.device)
    self.duration = torch.full(
      (self.num_envs,), cfg.down_duration_s, device=self.device
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    goal = (torch.rand(len(env_ids), device=self.device) < 0.5).float()
    reset_state = getattr(self._env, "_transfer_reset_state", None)
    # CommandTerm.reset() zeroes this counter immediately before resampling.
    # The environment resets episode_length_buf later, so that buffer still
    # contains the previous episode length at this point.
    fresh = self.command_counter[env_ids] == 0
    if reset_state is not None and fresh.any():
      self.value[env_ids[fresh], 0] = reset_state[env_ids[fresh]]
    self.start[env_ids, 0] = self.value[env_ids, 0]
    self.goal[env_ids, 0] = goal
    self.elapsed[env_ids] = 0.0
    rising = goal > self.start[env_ids, 0]
    self.duration[env_ids] = torch.where(
      rising,
      self.duration.new_full((len(env_ids),), self.cfg.down_duration_s),
      self.duration.new_full((len(env_ids),), self.cfg.up_duration_s),
    )

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if self._gui is not None:
      enabled, target, get_index = self._gui
      if enabled.value:
        index = get_index()
        if env_ids is None or bool((env_ids == index).any()):
          requested = float(target.value == "趴下（四足）")
          if requested != float(self.goal[index, 0]):
            self.start[index, 0] = self.value[index, 0]
            self.goal[index, 0] = requested
            self.elapsed[index] = 0.0
            self.duration[index] = (
              self.cfg.down_duration_s if requested else self.cfg.up_duration_s
            )
    if env_ids is not None:
      return
    active = self._env.episode_length_buf > 0
    self.elapsed[active] += self._env.step_dt
    progress = (self.elapsed / self.duration).clamp(0.0, 1.0)
    smooth = progress.square() * (3.0 - 2.0 * progress)
    self.value[:, 0] = self.start[:, 0] + (self.goal[:, 0] - self.start[:, 0]) * smooth


@dataclass(kw_only=True)
class SmoothPostureCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (14.0, 18.0)
  down_duration_s: float = 3.5
  up_duration_s: float = 5.0

  def build(self, env: "ManagerBasedRlEnv") -> SmoothPostureCommand:
    return SmoothPostureCommand(self, env)


def goal_state(env: "ManagerBasedRlEnv") -> torch.Tensor:
  term = env.command_manager.get_term("posture")
  if isinstance(term, SmoothPostureCommand):
    return term.goal[:, 0]
  return base.state(env)


def transition_direction(env: "ManagerBasedRlEnv") -> torch.Tensor:
  term = env.command_manager.get_term("posture")
  assert isinstance(term, SmoothPostureCommand)
  return torch.sign(term.goal[:, 0] - term.start[:, 0])


def endpoint_contact_score(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Pay endpoint contact only after the smooth target has nearly arrived."""
  score = base.contact_score(env)
  alpha = base.state(env)
  goal = goal_state(env)
  arrived = (alpha - goal).abs() < 0.02
  return score * arrived


def endpoint_quiet_score(env: "ManagerBasedRlEnv") -> torch.Tensor:
  alpha = base.state(env)
  arrived = (alpha - goal_state(env)).abs() < 0.02
  return base.quiet_at_goal(env) * arrived


def orientation_progress(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Reward tracking progress without treating every alpha step as a new goal."""
  current = -base.target_errors(env)[1]
  goal = goal_state(env)
  previous = getattr(env, "_transfer_e3_potential", current)
  previous_goal = getattr(env, "_transfer_e3_goal", goal)
  fresh = (env.episode_length_buf <= 1) | (goal != previous_goal)
  reward = torch.where(
    fresh, torch.zeros_like(current), (current - previous) / env.step_dt
  ).clamp(-2.0, 2.0)
  dynamic_env = cast(Any, env)
  dynamic_env._transfer_e3_potential = current.clone()
  dynamic_env._transfer_e3_goal = goal.clone()
  return reward


def staged_front_contact_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Keep hands loaded longer during rise and delay impact during descent."""
  sensor = env.scene["feet_ground_contact"]
  front = (sensor.data.found[:, 2:] > 0).all(-1)
  alpha = base.state(env)
  direction = transition_direction(env)
  should_contact = torch.where(direction < 0, alpha > 0.25, alpha > 0.78)
  transitioning = direction != 0
  return (front != should_contact).float() * transitioning


def vertical_speed_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Bound abrupt vertical launch/drop while leaving slow recovery available."""
  vz = env.scene["robot"].data.root_link_lin_vel_w[:, 2].abs()
  return ((vz - 0.10).clamp_min(0.0) / 0.10).square()


def torso_contact_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Penalize torso contact while leaving a short recovery window."""
  return (env.scene["torso_ground"].data.found > 0).flatten(1).any(-1).float()


def unrecoverable_state(
  env: "ManagerBasedRlEnv", torso_grace_s: float = 2.5
) -> torch.Tensor:
  """Terminate sustained collapse, not a recoverable transient contact."""
  robot = env.scene["robot"]
  z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  contact = (env.scene["torso_ground"].data.found > 0).flatten(1).any(-1)
  duration = getattr(env, "_transfer_e3_torso_contact_s", torch.zeros_like(z))
  duration = torch.where(contact, duration + env.step_dt, torch.zeros_like(duration))
  duration = torch.where(
    env.episode_length_buf <= 1, torch.zeros_like(duration), duration
  )
  cast(Any, env)._transfer_e3_torso_contact_s = duration
  return (
    (z < 0.05)
    | (z > 0.6)
    | ~torch.isfinite(robot.data.joint_pos).all(-1)
    | (duration > torso_grace_s)
  )
