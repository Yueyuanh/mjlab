"""V034 biped-specific locomotion rewards."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def directional_progress(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Reward normalized longitudinal progress in the commanded direction."""
  command = env.command_manager.get_command("twist")
  assert command is not None
  target = command[:, 0]
  moving = target.abs() > 0.04
  actual = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
  normalized = target.sign() * actual / target.abs().clamp_min(0.04)
  return normalized.clamp(-1.0, 1.0) * moving
