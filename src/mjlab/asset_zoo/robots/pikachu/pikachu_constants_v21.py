"""Pikachu **v2.1 sample-links** 训练常量（前缀 ``pikachu_sl_v21``）。

本模块为 `pikachu_sample_links_v2.1/pikachu_sl_v21_train.xml`（sample-links
设计版，每个 link 是彩色 box，含新质量/惯量/尺寸）提供机器人配置。与
``pikachu_constants.py``（capsule 碰撞体）和 ``pikachu_constants_lite.py``
（lite 碰撞体）并行存在，互不影响。

## 与旧机器人的关系——"同构型"适配

参考 npz（``joint_pos/vel (N,14)``、``body_pos_w (N,17,3)``）由旧 14 关节
机器人录制（见 ``scripts/csv_to_npz_pikachu.py`` 的 ``PIKACHU_JOINT_NAMES``）。
``tracking/mdp/commands.py::MotionCommand`` 用**当前机器人**的 body/joint 顺序
去索引 npz 列。为了让新模型**复用同一 npz**而列不错位，训练 XML
``pikachu_sl_v21_train.xml`` 做了三处适配（详见该文件头注释）：

  1. **丢弃 head**：源 XML 的 ``head_joint`` 是第 15 个 hinge，会让
     ``robot.data.joint_pos`` 变成 15 列、与 npz 的 14 列失配。
  2. **关节轴向/限位恢复旧 capsule(defaults.xml) 约定**：源 XML 限位方向反转
     且只正，会把参考值全 clip。本模块机器人仍为 14 关节、内部顺序 =
     ``PIKACHU_JOINT_NAMES``。
  3. **body 树顺序 = 旧 17-body**（右臂补 ``left/right_arm_yaw_link`` 刚性占位体），
     使 ``find_bodies`` 返回的 body id 与 npz 的 17 列对齐。

碰撞体：脚底 = ``*_ankle_link_collision`` box；手 = ``*_arm_yaw_link_collision``
box。执行器/电机物理完全复用 ``pikachu_constants.py``（同一套 DM 电机），只换
几何与碰撞正则。
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu.pikachu_constants import (
  HOME_KEYFRAME,
  KNEES_BENT_KEYFRAME,
  PIKACHU_ARTICULATION,
  QUAD_KEYFRAME,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# v2.1 sample-links 训练适配体（见模块 docstring 与 XML 头注释的适配说明）。
PIKACHU_SL21_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "pikachu"
  / "pikachu_sample_links_v2.1"
  / "pikachu_sl_v21_train.xml"
)
assert PIKACHU_SL21_XML.exists()


def get_spec() -> mujoco.MjSpec:
  """从 v2.1 训练 XML 编译出 MuJoCo 规格（每次调用返回新实例）。"""
  return mujoco.MjSpec.from_file(str(PIKACHU_SL21_XML))


##
# Joints.
##

# 14 个有效关节，顺序 = 机器人内部关节顺序 = 参考 npz 列序（同名同序同构）。
PIKACHU_SL21_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_joint",
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
)

##
# Geom regexes.
##

# 脚底碰撞 geom：v2.1 的脚是 ankle_link 上的 box（每脚一个），非 foot1~7 胶囊。
PIKACHU_SL21_FOOT_GEOM_NAMES = r"^(left|right)_ankle_link_collision$"
# 手/前腿支撑 geom：arm_yaw_link（刚性占位体）box，四足撑地时接触地面。
PIKACHU_SL21_HAND_GEOM_NAMES = r"^(left|right)_arm_yaw_link_collision$"

##
# Collision config.
##

# 全身碰撞：启用所有 ``.*_collision`` geom 的碰撞（含自碰撞）。
# 脚/手用 condim=3（平面接触）、priority=1、friction=0.6，其余点接触。
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={
    PIKACHU_SL21_FOOT_GEOM_NAMES: 3,
    PIKACHU_SL21_HAND_GEOM_NAMES: 3,
    ".*_collision": 1,
  },
  priority={PIKACHU_SL21_FOOT_GEOM_NAMES: 1, PIKACHU_SL21_HAND_GEOM_NAMES: 1, ".*": 0},
  friction={PIKACHU_SL21_FOOT_GEOM_NAMES: (0.6,), PIKACHU_SL21_HAND_GEOM_NAMES: (0.6,)},
)

# 无自碰撞版本（默认）：contype=0 使机器人自身 geom 之间不接触，但保持
# 与地面（contype=1）碰撞（conaffinity=1）。脚与手都 condim=3 + 摩擦，
# 满足双足行走与四足撑地（transition）两种需求。
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={
    PIKACHU_SL21_FOOT_GEOM_NAMES: 3,
    PIKACHU_SL21_HAND_GEOM_NAMES: 3,  # 手撑地：平面接触（摩擦）
    ".*_collision": 1,
  },
  priority={
    PIKACHU_SL21_FOOT_GEOM_NAMES: 1,
    PIKACHU_SL21_HAND_GEOM_NAMES: 1,
    ".*": 0,
  },
  friction={
    PIKACHU_SL21_FOOT_GEOM_NAMES: (0.6,),
    PIKACHU_SL21_HAND_GEOM_NAMES: (0.6,),
  },
)

# 仅脚底碰撞：只保留脚底与地面接触（平地行走训练常用）。
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(PIKACHU_SL21_FOOT_GEOM_NAMES,),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Keyframes — 与旧机器人同取值（保持了旧轴向/限位），仅按新腿长微调 pos.z。
#   新几何站立时 base_link 高度 ≈ 0.186：HOME 微抬、KNEES_BENT 微蹲。
#   数值为初值，落地配平用 viewer 微调。
##

HOME_SL21_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.19),
  joint_pos=HOME_KEYFRAME.joint_pos,
  joint_vel={".*": 0.0},
)

KNEES_BENT_SL21_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.17),
  joint_pos=KNEES_BENT_KEYFRAME.joint_pos,
  joint_vel={".*": 0.0},
)

# 双足模式标称姿态：复用微蹲起跑。
BIPED_SL21_KEYFRAME = KNEES_BENT_SL21_KEYFRAME

# 四足模式：手脚四点支撑。关节取值同旧 QUAD_KEYFRAME，pos.z 按新腿长微调。
QUAD_SL21_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.18),
  rot=QUAD_KEYFRAME.rot,
  joint_pos=QUAD_KEYFRAME.joint_pos,
  joint_vel={".*": 0.0},
)

##
# Action scale — 复用双足 velocity 已验证的幅度（与旧 tracking 任务一致）。
##
PIKACHU_SL21_ACTION_SCALE: dict[str, float] = {
  r".*_hip_pitch_joint": 0.5,
  r".*_knee_joint": 0.5,
  r".*_hip_roll_joint": 0.35,
  r".*_hip_yaw_joint": 0.35,
  r".*_ankle_joint": 0.35,
  r".*_arm_pitch_joint": 0.35,
  r".*_arm_roll_joint": 0.35,
}

##
# Final config.
##

# 执行器/电机物理完全复用旧 capsule 版（同一套 DM 电机、PD 增益、软限位因子）。
PIKACHU_SL21_ARTICULATION: EntityArticulationInfoCfg = PIKACHU_ARTICULATION


def get_pikachu_sl21_robot_cfg() -> EntityCfg:
  """获取一个新的 v2.1 sample-links 机器人配置实例。

  默认使用 ``FULL_COLLISION_WITHOUT_SELF``（双脚 + 双手都可与地面接触、
  手脚 condim=3 + 摩擦，满足双足行走与四足撑地）。起步姿态复用微蹲起跑。
  """
  return EntityCfg(
    init_state=KNEES_BENT_SL21_KEYFRAME,
    collisions=(FULL_COLLISION_WITHOUT_SELF,),
    spec_fn=get_spec,
    articulation=PIKACHU_SL21_ARTICULATION,
  )


__all__ = [
  "BIPED_SL21_KEYFRAME",
  "FEET_ONLY_COLLISION",
  "FULL_COLLISION",
  "FULL_COLLISION_WITHOUT_SELF",
  "HOME_SL21_KEYFRAME",
  "KNEES_BENT_SL21_KEYFRAME",
  "PIKACHU_SL21_ACTION_SCALE",
  "PIKACHU_SL21_ARTICULATION",
  "PIKACHU_SL21_FOOT_GEOM_NAMES",
  "PIKACHU_SL21_HAND_GEOM_NAMES",
  "PIKACHU_SL21_JOINT_NAMES",
  "PIKACHU_SL21_XML",
  "QUAD_SL21_KEYFRAME",
  "get_pikachu_sl21_robot_cfg",
  "get_spec",
]
