"""Selective arm-leg mesh contact, isolated from E0 and crawl collision policies."""

from mjlab.asset_zoo.robots.pikachu.pikachu_v033_constants import (
  PIKACHU_V033_FOOT_GEOM_NAMES,
  get_pikachu_v033_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

ARM_GEOMS = r"(left|right)_(arm_(pitch|roll|yaw)|elbow)_link_collision"
LEG_GEOMS = r"(left|right)_(hip_(pitch|roll|yaw)|knee(_up|_mid)?)_link_collision"
ARM_BODIES = r"(left|right)_(arm_(pitch|roll|yaw)|elbow)_link"
CLEARANCE_M = 0.01


def biped_e1_robot_cfg() -> EntityCfg:
  cfg = get_pikachu_v033_robot_cfg()
  # Ground bit=1, arm bit=2, leg bit=4. Arms only meet legs/soles;
  # leg-leg and arm-arm joint assemblies retain their existing exclusions.
  cfg.collisions = (
    CollisionCfg(
      geom_names_expr=(PIKACHU_V033_FOOT_GEOM_NAMES, ARM_GEOMS, LEG_GEOMS),
      contype={PIKACHU_V033_FOOT_GEOM_NAMES: 5, ARM_GEOMS: 2, LEG_GEOMS: 4},
      conaffinity={PIKACHU_V033_FOOT_GEOM_NAMES: 3, ARM_GEOMS: 4, LEG_GEOMS: 2},
      condim=3,
      priority=1,
      friction=(0.6,),
      margin=0.0,
      # MuJoCo 3.11: detect inactive contacts within gap, generate forces
      # only at margin=0. This enables an anticipatory mesh-distance cost.
      gap={ARM_GEOMS: CLEARANCE_M, ".*": 0.0},
    ),
  )
  return cfg
