"""V035 直立任务：v033 E1 奖励框架 + 全直驱 18 关节本体。"""

from mjlab.asset_zoo.robots.pikachu.pikachu_v035_constants import (
  FOOT_GEOMS,
  get_biped_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

ARM_GEOMS = r"(left|right)_(arm_(pitch|roll|yaw)|elbow)_link_collision"
LEG_GEOMS = r"(left|right)_(hip_(pitch|roll|yaw)|knee|ankle)_link_collision"
CLEARANCE_M = 0.01


def biped_e1_robot_cfg() -> EntityCfg:
  """脚底接地 + 手臂/腿部选择性自碰撞（与 v034 E1 同一接触策略）。"""
  cfg = get_biped_robot_cfg()
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
