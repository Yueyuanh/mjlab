"""Soft penalties measured on actual joints, without target or state clamping."""

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def arm_posture_window(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
  tolerance_degrees: float = 5.0,
) -> torch.Tensor:
  """Mean squared excess, normalized by the window half-width, over four joints.

  The center is the configured reset pose, not the last noisy reset sample.
  This term only reads physical positions; it never changes policy actions.
  """
  if tolerance_degrees <= 0:
    raise ValueError("tolerance_degrees must be positive")
  robot = env.scene[asset_cfg.name]
  delta = (
    robot.data.joint_pos[:, asset_cfg.joint_ids]
    - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  tolerance = math.radians(tolerance_degrees)
  excess = (delta.abs() - tolerance).clamp_min(0)
  return (excess / tolerance).square().mean(-1)
