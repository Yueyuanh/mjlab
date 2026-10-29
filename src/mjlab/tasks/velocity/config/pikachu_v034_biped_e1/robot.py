"""V034 biped collisions following the validated V033 E1 contact policy."""

from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import (
  FOOT_GEOMS,
  get_biped_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

ARM_GEOMS = r"(left|right)_(arm_(pitch|roll|yaw)|elbow)_link_collision"
LEG_GEOMS = r"(left|right)_(hip_(pitch|roll|yaw)|knee(_up|_mid)?)_link_collision"
CLEARANCE_M = 0.01


def biped_e1_robot_cfg(symmetric: bool = False) -> EntityCfg:
  """Enable feet-ground and selective arm-leg contacts for V034.

  ``symmetric=True`` selects the mirrored left-leg limit variant produced by
  ``scripts/build_v034_symmetric_variant.py``.
  """
  cfg = get_biped_robot_cfg(symmetric=symmetric)
  cfg.collisions = (
    CollisionCfg(
      geom_names_expr=(FOOT_GEOMS, ARM_GEOMS, LEG_GEOMS),
      contype={FOOT_GEOMS: 5, ARM_GEOMS: 2, LEG_GEOMS: 4},
      conaffinity={FOOT_GEOMS: 3, ARM_GEOMS: 4, LEG_GEOMS: 2},
      condim=3,
      priority=1,
      friction=(0.8,),
      margin=0.0,
      gap={ARM_GEOMS: CLEARANCE_M, ".*": 0.0},
    ),
  )
  return cfg
