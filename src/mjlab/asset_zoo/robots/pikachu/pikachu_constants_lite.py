"""Pikachu 14-DOF humanoid constants — **lite 碰撞体**变体。

本文件与 ``pikachu_constants.py`` 的机器人完全一致（同一套执行器、
关键帧、动作缩放），唯一差异在**碰撞体 XML**：

  - ``Pikachu_V025_flat_14dof_lite.xml``：原始碰撞体。每个 link 一个
    ``<link>_collision`` geom（共 17 个），脚底就是 ``ankle_link`` 自身的
    网格（``left/right_ankle_link_collision``），没有 G1 风格的 foot1~7
    脚趾胶囊。
  - ``Pikachu_V025_flat_14dof_capsule.xml``：简化碰撞体（见
    ``pikachu_constants.py``）。碰撞 geom 稀疏（base / hip_roll / knee /
    arm_yaw），脚底为 foot1~7 胶囊（挂在 ankle body 子树内）。

用途：对比两种碰撞体对**训练与部署**的影响——lite 用踝关节自身网格做
脚底接触（面积/形状由 mesh 决定），capsule 用 7 个胶囊近似脚底。两者的
脚底摩擦随机化（``foot_friction`` 事件）也要匹配各自的脚底 geom，因此
这里导出 lite 版的 ``PIKACHU_FOOT_GEOM_NAMES`` 供任务 env_cfg 引用。

切换到 lite 碰撞体：任务 env_cfg（transition/velocity）里把
``get_pikachu_robot_cfg`` 与 ``PIKACHU_FOOT_GEOM_NAMES`` 的 import 来源从
``pikachu_constants`` 换成 ``pikachu_constants_lite`` 即可（两模块导出同名）。
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu.pikachu_constants import (
  ACTUATOR_J3507,
  ACTUATOR_J4310,
  ARMATURE_J3507,
  ARMATURE_J4310,
  BIPED_KEYFRAME,
  DAMPING_J3507,
  DAMPING_J4310,
  DAMPING_RATIO,
  GEARS_J3507,
  GEARS_J4310,
  HOME_KEYFRAME,
  KNEES_BENT_KEYFRAME,
  NATURAL_FREQ,
  PIKACHU_ACTION_SCALE,
  PIKACHU_ACTUATOR_J3507,
  PIKACHU_ACTUATOR_J4310,
  PIKACHU_ARTICULATION,
  QUAD_KEYFRAME,
  ROTOR_INERTIAS_J3507,
  ROTOR_INERTIAS_J4310,
  STIFFNESS_J3507,
  STIFFNESS_J4310,
)
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets — 覆盖为 lite 碰撞体。
##

# lite 碰撞体 XML：每个 link 一个 ``<link>_collision`` geom（17 个），
# 脚底为 ``ankle_link`` 自身网格，无 foot1~7 胶囊。
PIKACHU_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "pikachu"
  / "xmls"
  / "mjcf_14dof_lite"
  / "Pikachu_V025_flat_14dof_lite.xml"
)
assert PIKACHU_XML.exists()


def get_spec() -> mujoco.MjSpec:
  """从 lite 碰撞体 MJCF 文件编译出 MuJoCo 规格（每次调用返回新实例）。"""
  return mujoco.MjSpec.from_file(str(PIKACHU_XML))


##
# Actuator / Keyframe / Articulation — 与 capsule 版完全一致（re-export）。
##

# 脚底碰撞 geom 正则（lite：整块 ankle 网格当脚底）。
PIKACHU_FOOT_GEOM_NAMES = r"^(left|right)_ankle_link_collision$"

##
# Collision config — lite 版（脚底正则不同，其余策略与 capsule 版一致）。
##

# 全身碰撞：启用所有 ``.*_collision`` geom 的碰撞（含自碰撞）。
# 脚底（ankle_link 网格）用 condim=3（平面接触）、priority=1、friction=0.6，
# 其余用 condim=1（点接触）、priority=0。
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={PIKACHU_FOOT_GEOM_NAMES: 3, ".*_collision": 1},
  priority={PIKACHU_FOOT_GEOM_NAMES: 1, ".*": 0},
  friction={PIKACHU_FOOT_GEOM_NAMES: (0.6,)},
)

# 无自碰撞版本：contype=0 使机器人自身 geom 之间不产生接触，
# 但仍保留与地面（contype=1）的碰撞能力由 conaffinity=1 表达。
# 手（arm_yaw_link_collision）也配 condim=3 + friction：四足状态转换任务里
# 双手撑地，condim=1（点接触无摩擦）会打滑，无法稳定支撑。
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={
    PIKACHU_FOOT_GEOM_NAMES: 3,
    r"^(left|right)_arm_yaw_link_collision$": 3,  # 手撑地：平面接触（摩擦）
    ".*_collision": 1,
  },
  priority={
    PIKACHU_FOOT_GEOM_NAMES: 1,
    r"^(left|right)_arm_yaw_link_collision$": 1,
    ".*": 0,
  },
  friction={
    PIKACHU_FOOT_GEOM_NAMES: (0.6,),
    r"^(left|right)_arm_yaw_link_collision$": (0.6,),
  },
)

# 仅脚底碰撞：只保留脚底（ankle_link 网格）与地面接触，其余 geom 全部禁用
# （disable_other_geoms=True 会把未匹配 geom 的 contype/conaffinity 清零）。
# 适合平地行走训练——Pikachu 的自碰撞对已在 contact.xml 中大量排除，
# 训练时通常只需脚-地接触。
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(PIKACHU_FOOT_GEOM_NAMES,),
  contype=1,
  conaffinity=1,
  condim=3,  # 脚底为平面接触（3 维法向 + 切向）
  priority=1,
  friction=(0.6,),
)


def get_pikachu_robot_cfg() -> EntityCfg:
  """获取一个新的 **lite 碰撞体** Pikachu 机器人配置实例。

  与 ``pikachu_constants.get_pikachu_robot_cfg`` 签名一致，方便任务 env_cfg
  通过切换 import 来源来选择碰撞体。默认使用 ``FULL_COLLISION_WITHOUT_SELF``
  （手脚都可与地面接触、手 condim=3 + 摩擦，满足四足撑地需求）。
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,  # 默认从微蹲姿态起跑，更稳定
    collisions=(FULL_COLLISION_WITHOUT_SELF,),
    spec_fn=get_spec,
    articulation=PIKACHU_ARTICULATION,
  )


__all__ = [
  "ACTUATOR_J4310",
  "ACTUATOR_J3507",
  "ARMATURE_J4310",
  "ARMATURE_J3507",
  "BIPED_KEYFRAME",
  "DAMPING_J4310",
  "DAMPING_J3507",
  "DAMPING_RATIO",
  "FEET_ONLY_COLLISION",
  "FULL_COLLISION",
  "FULL_COLLISION_WITHOUT_SELF",
  "GEARS_J4310",
  "GEARS_J3507",
  "HOME_KEYFRAME",
  "KNEES_BENT_KEYFRAME",
  "NATURAL_FREQ",
  "PIKACHU_ACTION_SCALE",
  "PIKACHU_ACTUATOR_J4310",
  "PIKACHU_ACTUATOR_J3507",
  "PIKACHU_ARTICULATION",
  "PIKACHU_FOOT_GEOM_NAMES",
  "PIKACHU_XML",
  "QUAD_KEYFRAME",
  "ROTOR_INERTIAS_J4310",
  "ROTOR_INERTIAS_J3507",
  "STIFFNESS_J4310",
  "STIFFNESS_J3507",
  "get_pikachu_robot_cfg",
  "get_spec",
]


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_pikachu_robot_cfg())

  viewer.launch(robot.spec.compile())
