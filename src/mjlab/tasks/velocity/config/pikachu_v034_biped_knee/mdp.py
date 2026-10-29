"""Rewards for an actively knee-flexing, low-roll V034 biped gait."""

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def walking_mask(
  env: "ManagerBasedRlEnv", command_name: str = "twist", threshold: float = 0.05
) -> torch.Tensor:
  """True for any non-standing command (forward, backward, lateral or turn)."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  magnitude = command[:, 0].abs() + command[:, 1].abs() + command[:, 2].abs()
  return magnitude > threshold


def _knee_flexion(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return (flexion, swing-flexion, stance-flexion) in the right,left order.

  V034's left knee_up axis is reversed: negative left angles are flexion.  The
  sign vector corrects this so ``flexion`` is positive for a bent knee on both
  sides.  The standing keyframe has zero knee_up, so no offset is needed.
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  contact = sensor.data.found > 0
  if contact.shape[-1] != 2:
    raise ValueError(f"Expected two feet, got {contact.shape[-1]}")
  position = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
  if position.shape[-1] != 2:
    raise ValueError(f"Expected two knee joints, got {position.shape[-1]}")
  sign = position.new_tensor((1.0, -1.0))
  flexion = position * sign
  swing = (flexion * (~contact)).sum(-1)
  stance = (flexion * contact).sum(-1)
  return flexion, swing, stance


def swing_knee_flexion(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  target: float = 0.35,
  std: float = 0.18,
) -> torch.Tensor:
  """Reward the airborne knee reaching a visible flexion target.

  Only single-support samples count.  A permanent crouch therefore cannot farm
  the term, because the supporting knee is not part of the reward and is
  separately penalized by :func:`stance_knee_extension`.
  """
  if target <= 0 or std <= 0:
    raise ValueError("target and std must be positive")
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  contact = sensor.data.found > 0
  _, swing, stance = _knee_flexion(env, sensor_name, asset_cfg)
  single_support = contact.sum(-1) == 1
  reward = torch.exp(-((swing - target) / std).square())
  reward = reward * single_support * walking_mask(env)
  log = env.extras.setdefault("log", {})
  if single_support.any():
    log["Metrics/swing_knee_flexion_rad"] = swing[single_support].mean()
    log["Metrics/stance_knee_flexion_rad"] = stance[single_support].mean()
  return reward


def swing_knee_separation(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  scale: float = 0.15,
) -> torch.Tensor:
  """Reward the airborne knee being bent more than the supporting knee.

  :func:`swing_knee_flexion` pays for absolute flexion, and the 150-iteration
  pilot (``research/gentle_lr_1024x150``) showed what that buys: both knees
  rose from 0.10/0.18 to about 0.18 rad, so the kernel sat at 90% of its
  maximum while swing-minus-stance moved from -0.07 to only +0.02 rad -- a
  symmetric crouch, not a lifted leg.  This term pays only for the difference,
  so crouching both legs earns nothing.

  ``tanh`` keeps a finite gradient at zero separation (``1/scale`` per rad) and
  saturates instead of punishing a large one.  At ``scale=0.15`` a 9 degree
  (0.15 rad) separation already earns 76% of the term's maximum, and the
  mechanical range allows up to 60 degrees of swing flexion.
  """
  if scale <= 0:
    raise ValueError("scale must be positive")
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  contact = sensor.data.found > 0
  _, swing, stance = _knee_flexion(env, sensor_name, asset_cfg)
  separation = swing - stance
  single_support = contact.sum(-1) == 1
  reward = torch.tanh(separation / scale).clamp_min(0.0)
  reward = reward * single_support * walking_mask(env)
  if single_support.any():
    log = env.extras.setdefault("log", {})
    log["Metrics/swing_minus_stance_knee_rad"] = separation[single_support].mean()
  return reward


def stance_knee_extension(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  tolerance: float = 0.15,
  std: float = 0.12,
) -> torch.Tensor:
  """Penalize a strongly bent supporting knee (squatting instead of stepping)."""
  if tolerance < 0 or std <= 0:
    raise ValueError("tolerance must be non-negative and std positive")
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  contact = sensor.data.found > 0
  _, _, stance = _knee_flexion(env, sensor_name, asset_cfg)
  excess = (stance - tolerance).clamp_min(0) / std
  return excess.square() * (contact.sum(-1) == 1) * walking_mask(env)
