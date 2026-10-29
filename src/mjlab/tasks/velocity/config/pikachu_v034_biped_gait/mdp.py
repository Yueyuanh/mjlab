"""Gait-quality rewards derived from measured V034 failure modes."""

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.config.pikachu_v033_biped_e1.mdp import (
  straight_walking_mask,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def straight_base_roll_rate(
  env: "ManagerBasedRlEnv", deadzone: float = 0.45, std: float = 0.75
) -> torch.Tensor:
  """Penalize the excess roll rate used by the old side-to-side gait."""
  if deadzone < 0 or std <= 0:
    raise ValueError("deadzone must be non-negative and std must be positive")
  roll_rate = env.scene["robot"].data.root_link_ang_vel_b[:, 0].abs()
  excess = (roll_rate - deadzone).clamp_min(0) / std
  return excess.square() * straight_walking_mask(env)


def swing_knee_separation(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  margin: float = 0.07,
  std: float = 0.07,
) -> torch.Tensor:
  """Require the airborne knee to flex beyond the supporting knee.

  The term is active only in single support. It therefore cannot be satisfied by
  a permanent crouch, which was the failure measured in the previous policy.
  Joint order and contact order are both explicitly right, left in this task.
  """
  if margin <= 0 or std <= 0:
    raise ValueError("margin and std must be positive")
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  contact = sensor.data.found > 0
  assert contact.shape[-1] == 2
  position = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
  assert position.shape[-1] == 2
  flexion = position.abs()
  single_support = contact.sum(-1) == 1
  swing_flexion = (flexion * (~contact)).sum(-1)
  stance_flexion = (flexion * contact).sum(-1)
  shortfall = ((margin - (swing_flexion - stance_flexion)) / std).clamp_min(0)
  env.extras["log"]["Metrics/swing_knee_separation_rad"] = torch.mean(
    (swing_flexion - stance_flexion)[single_support]
  ) if single_support.any() else position.new_zeros(())
  return shortfall.square() * single_support * straight_walking_mask(env)
