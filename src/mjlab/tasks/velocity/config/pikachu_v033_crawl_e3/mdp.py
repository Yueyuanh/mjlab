"""Direction-aware costs that preserve the strong E2 backward gait."""

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.config.pikachu_v033_crawl.mdp import (
  gait_contacts,
  heading_lin_vel,
  moving,
)
from mjlab.tasks.velocity.config.pikachu_v033_crawl_e2.mdp import sole_support_cost

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def forward_weight(
  env: "ManagerBasedRlEnv", forward: float = 1.0, backward: float = 0.20
) -> torch.Tensor:
  """Weight forward corrections strongly while retaining backward rehearsal."""
  command = env.command_manager.get_command("twist")
  assert command is not None
  moving = command[:, 0].abs() > 0.04
  scale = torch.where(command[:, 0] > 0.04, forward, backward)
  return torch.where(moving, scale, torch.zeros_like(scale))


def directional_sole_cost(
  env: "ManagerBasedRlEnv",
  asset_cfg: "SceneEntityCfg",
  endpoint: bool = False,
  tolerance_degrees: float = 5.0,
  scale_degrees: float = 10.0,
) -> torch.Tensor:
  """Add sole shaping mainly to forward motion, where E2 is weaker."""
  cost = sole_support_cost(
    env,
    asset_cfg,
    endpoint=endpoint,
    tolerance_degrees=tolerance_degrees,
    scale_degrees=scale_degrees,
  )
  return cost * forward_weight(env)


def directional_gait_mismatch(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Penalize phase/contact mismatch without changing the E2 gait clock."""
  return (1.0 - gait_contacts(env)) * forward_weight(env)


def smooth_swing_weight(
  env: "ManagerBasedRlEnv", transition_seconds: float = 0.10
) -> torch.Tensor:
  """Return a continuous target swing confidence with softened phase edges."""
  if not 0 < transition_seconds <= 0.16:
    raise ValueError("transition_seconds must be in (0, 0.16]")
  phase = env.episode_length_buf[:, None] * env.step_dt / 1.6
  offsets = phase.new_tensor([0.0, 0.5, 0.75, 0.25])
  relative = torch.remainder(phase - offsets, 1.0)
  inside = relative < 0.20
  distance = torch.minimum(relative, 0.20 - relative)
  alpha = (distance / (transition_seconds / 1.6)).clamp(0.0, 1.0)
  smooth = alpha.square() * (3.0 - 2.0 * alpha)
  return torch.where(inside, smooth, torch.zeros_like(smooth)) * moving(env)[:, None]


def directional_smooth_gait_mismatch(
  env: "ManagerBasedRlEnv", transition_seconds: float = 0.10
) -> torch.Tensor:
  """Compare contact to a soft swing target instead of a Boolean phase mask."""
  return smooth_gait_mismatch(env, transition_seconds) * forward_weight(env)


def smooth_gait_mismatch(
  env: "ManagerBasedRlEnv", transition_seconds: float = 0.10
) -> torch.Tensor:
  """Measure contact error against a continuous four-limb gait target."""
  sensor = env.scene["feet_ground_contact"]
  assert sensor.data.found is not None
  contact = (sensor.data.found > 0).float()
  target_contact = 1.0 - smooth_swing_weight(env, transition_seconds)
  return (contact - target_contact).abs().mean(-1)


def smooth_gait_contacts(
  env: "ManagerBasedRlEnv", transition_seconds: float = 0.10
) -> torch.Tensor:
  """Reward continuous contact matching while retaining the original scale."""
  return 1.0 - smooth_gait_mismatch(env, transition_seconds)


def forward_velocity_shortfall(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Penalize only forward under-speed; overspeed and backward stay unchanged."""
  command = env.command_manager.get_command("twist")
  assert command is not None
  error = (command[:, 0] - heading_lin_vel(env)[:, 0]).clamp_min(0.0)
  return (error / 0.08).square() * (command[:, 0] > 0.04)
