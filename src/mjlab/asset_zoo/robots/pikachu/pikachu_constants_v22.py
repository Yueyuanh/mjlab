"""Pikachu **v2.2 sample-links** 训练常量（前缀 ``pikachu_sl22``/``SL22``）。

本模块为 `pikachu_sample_14dof_links_v2.2/pikachu_sl_v22_train.xml`（sample-links
适配体，每个 link 是彩色 box，含新质量/惯量/尺寸）提供机器人配置。与
``pikachu_constants.py``（capsule 碰撞体）、``pikachu_constants_lite.py``
（lite）、``pikachu_constants_v21.py``（v2.1 sample-links）并行存在，互不影响。

## v2.1 → v2.2 的路口——为什么这版更灵活

用户反馈 v2.1（髋距 y=0.0974、腿长 L96）"左右腿间距太大、整体苯苯的"。v2.2
将髋距收窄到 ``y=0.0515``、腿部链条缩短（L71）、臂更短更细、base 质量降到
2.472 kg。**本模块不修改任何旧任务**，只是按同构复用规则把 v2.2 的几何/质量
重映射到旧 14 关节 / npz 运动学约定上（见下"与旧机器人的关系"）。

## 与旧机器人的关系——"同构型"适配

参考 npz（``joint_pos/vel (N,14)``、``body_pos_w (N,17,3)``）由旧 14 关节
机器人录制（见 ``scripts/csv_to_npz_pikachu.py`` 的 ``PIKACHU_JOINT_NAMES``）。
``tracking/mdp/commands.py::MotionCommand`` 用**当前机器人**的 body/joint 顺序
去索引 npz 列。为了让 v2.2 模型**复用同一 npz** 而列不错位，训练 XML
``pikachu_sl_v22_train.xml`` 做了三处适配（详见该文件头注释）：

  1. **丢弃 head**：源 XML 的 ``head_joint`` 是第 15 个 hinge，会让
     ``robot.data.joint_pos`` 变成 15 列、与 npz 的 14 列失配。
  2. **关节轴向/限位恢复旧 capsule(defaults.xml) 约定**：源 XML 限位方向反转
     且只正，会把参考值全 clip。本模块机器人仍为 14 关节、内部顺序 =
     ``PIKACHU_JOINT_NAMES``。
  3. **body 树顺序 = 旧 17-body**（右臂补 ``left/right_arm_yaw_link`` 刚性占位体），
     使 ``find_bodies`` 返回的 body id 与 npz 的 17 列对齐。

碰撞体：脚底 = ``*_ankle_link_collision`` box；手 = ``*_arm_yaw_link_collision``
box。执行器/电机物理完全复用 ``pikachu_constants.py``（同一套 DM 电机），只换
几何与碰撞正则。关键帧 ``pos.z`` 按 v2.2 更短腿下调（站立 base 帧高度
≈0.163，见落地配平冒烟脚本）；数值为初值，落地配平用 viewer 微调。

## 四足 velocity 复用

本模块同时导出四足全向 velocity 所需常量（``QUAD_SL22_*`` 与
``get_pikachu_sl22_quad_robot_cfg``），供 velocity quad 任务镜像旧
``pikachu_quad``。四足起步姿态复用共享 ``QUAD_KEYFRAME`` 的关节摆位（手脚四点
支撑），仅 ``pos.z`` 按 v2.2 更短腿下调。
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu.pikachu_constants import (
  HOME_KEYFRAME,
  KNEES_BENT_KEYFRAME,
  PIKACHU_ARTICULATION,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# v2.2 sample-links 训练适配体（见模块 docstring 与 XML 头注释的适配说明）。
PIKACHU_SL22_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "pikachu"
  / "pikachu_sample_14dof_links_v2.2"
  / "pikachu_sl_v22_train.xml"
)
assert PIKACHU_SL22_XML.exists()


def get_spec() -> mujoco.MjSpec:
  """从 v2.2 训练 XML 编译出 MuJoCo 规格（每次调用返回新实例）。"""
  return mujoco.MjSpec.from_file(str(PIKACHU_SL22_XML))


##
# Joints.
##

# 14 个有效关节，顺序 = 机器人内部关节顺序 = 参考 npz 列序（同名同序同构）。
PIKACHU_SL22_JOINT_NAMES: tuple[str, ...] = (
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

# 脚底碰撞 geom：v2.2 的脚是 ankle_link 上的 box（每脚一个），非 foot1~7 胶囊。
PIKACHU_SL22_FOOT_GEOM_NAMES = r"^(left|right)_ankle_link_collision$"
# 手/前腿支撑 geom：arm_yaw_link（刚性占位体）box，四足撑地时接触地面。
PIKACHU_SL22_HAND_GEOM_NAMES = r"^(left|right)_arm_yaw_link_collision$"

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
    PIKACHU_SL22_FOOT_GEOM_NAMES: 3,
    PIKACHU_SL22_HAND_GEOM_NAMES: 3,
    ".*_collision": 1,
  },
  priority={PIKACHU_SL22_FOOT_GEOM_NAMES: 1, PIKACHU_SL22_HAND_GEOM_NAMES: 1, ".*": 0},
  friction={PIKACHU_SL22_FOOT_GEOM_NAMES: (0.6,), PIKACHU_SL22_HAND_GEOM_NAMES: (0.6,)},
)

# 无自碰撞版本（默认）：contype=0 使机器人自身 geom 之间不接触，但保持
# 与地面（contype=1）碰撞（conaffinity=1）。脚与手都 condim=3 + 摩擦，
# 满足双足行走与四足撑地（transition）两种需求。
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={
    PIKACHU_SL22_FOOT_GEOM_NAMES: 3,
    PIKACHU_SL22_HAND_GEOM_NAMES: 3,  # 手撑地：平面接触（摩擦）
    ".*_collision": 1,
  },
  priority={
    PIKACHU_SL22_FOOT_GEOM_NAMES: 1,
    PIKACHU_SL22_HAND_GEOM_NAMES: 1,
    ".*": 0,
  },
  friction={
    PIKACHU_SL22_FOOT_GEOM_NAMES: (0.6,),
    PIKACHU_SL22_HAND_GEOM_NAMES: (0.6,),
  },
)

# 仅脚底碰撞：只保留脚底与地面接触（平地行走训练常用）。
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(PIKACHU_SL22_FOOT_GEOM_NAMES,),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Keyframes — 与旧机器人同取值（保持了旧轴向/限位），仅按 v2.2 更短腿微调 pos.z。
#   落地配平（/tmp/v22_height.py）：v2.2 双足站立脚底贴地时 base 帧高度 ≈ 0.163
#   （v2.1 是 ≈0.186，因腿更短），四足趴姿时 base 帧 ≈0.02。
#   数值为初值，落地配平用 viewer 微调。
##

HOME_SL22_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.175),
  joint_pos=HOME_KEYFRAME.joint_pos,
  joint_vel={".*": 0.0},
)

KNEES_BENT_SL22_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.17),
  joint_pos=KNEES_BENT_KEYFRAME.joint_pos,
  joint_vel={".*": 0.0},
)

# 双足模式标称姿态：复用微蹲起跑。
BIPED_SL22_KEYFRAME = KNEES_BENT_SL22_KEYFRAME

# 四足模式：手脚四点支撑。关节取值同旧 QUAD_KEYFRAME，pos.z 按更短腿下调。
# QUAD_SL22_KEYFRAME = EntityCfg.InitialStateCfg(
#   pos=(0, 0, 0.16),
#   rot=QUAD_KEYFRAME.rot,
#   joint_pos=QUAD_KEYFRAME.joint_pos,
#   joint_vel={".*": 0.0},
# )
QUAD_SL22_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.112),
  # rot=(0.593497, 0.0, 0.804837,0),# 117
  # rot=(0.7071, 0, 0.7071, 0), #90
  rot=(0.6756, 0, 0.7373, 0), #95
  joint_pos={
    # 腿：后支撑（四足时腿在 base 后方弯曲），左右取反
    "left_hip_pitch_joint": -2.4,
    "right_hip_pitch_joint": 2.4,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_knee_joint": -1.2,
    "right_knee_joint": 1.2,
    "left_ankle_joint": -0.3,
    "right_ankle_joint": 0.3,
    "left_arm_pitch_joint": -1.7,
    "right_arm_pitch_joint": 1.7,
    "left_arm_roll_joint": 0.0,
    "right_arm_roll_joint": -0.0,
  },
  joint_vel={".*": 0.0},
)

# 四足 velocity 起步姿态 = 四足关键帧。
QUAD_SL22_START = QUAD_SL22_KEYFRAME

# 四足躯干(基座)body 名。
QUAD_SL22_TORSO_BODY = "base_link"

# 四足"背轴"在当地机体的方向:机体 +Z。结构性定义,与几何/版本无关。
QUAD_SL22_BODY_AXIS: tuple[float, float, float] = (0.0, 0.0, 1.0)

# 四足 target 躯干前俯角(度) = viewer 中**紫色箭头目标方向**。
# 旧 capsule 用了 `QUAD_TILT_DEG = 110.0`,v2.2 **不复用旧值**。此值须按 v2.2
# 调好的四足趴姿标定：
#   - 初值取 v2.2 keyframe 自身实测背轴角 90.0°(keyframe 旋转把背轴转到世界 +X,
#     即"背水平");实测 /tmp/v22_quad_tilt.py:满幅 90.00°。
#   - ⚠️ 门禁标定 TODO：当前 QUAD_SL22_KEYFRAME 沿用旧摆位,套到 v2.2 更短腿上会
#     塌成趴地(实测静置背轴角 ≈75°、base_z≈0.04、base 触地),**不是有效四足姿态**。
#     必须先调好四足趴姿(手脚四点支撑、躯干不塌),再用 viewer 让紫箭头(目标)对准
#     实际目标背轴后再回填,不要直接照抄 keyframe 的 90°。
QUAD_SL22_TILT_DEG: float = 90.0+5.0

# 四足前腿 = 手的支撑几何：手掌(arm_yaw_link)撑地。
QUAD_SL22_HAND_GEOM_NAMES = PIKACHU_SL22_HAND_GEOM_NAMES

# 四足 4 点支撑的末端接地 geom 集：后脚(ankle box) + 双手手掌。
QUAD_SL22_SUPPORT_GEOM_NAMES = (
  PIKACHU_SL22_FOOT_GEOM_NAMES,  # 后脚
  QUAD_SL22_HAND_GEOM_NAMES,  # 双手撑地
)

##
# Action scale — 双足 velocity 复用已验证幅度（与旧 tracking 任务一致）。
##
PIKACHU_SL22_ACTION_SCALE: dict[str, float] = {
  r".*_hip_pitch_joint": 0.5,
  r".*_knee_joint": 0.5,
  r".*_hip_roll_joint": 0.35,
  r".*_hip_yaw_joint": 0.35,
  r".*_ankle_joint": 0.35,
  r".*_arm_pitch_joint": 0.35,
  r".*_arm_roll_joint": 0.35,
}

# 四足动作缩放：前腿(arm_pitch)作为支撑腿/抬脚放宽到 1.0，与 hip_pitch 同档。
QUAD_SL22_ACTION_SCALE: dict[str, float] = {
  r".*_hip_pitch_joint": 1.0,
  r".*_knee_joint": 1.0,
  r".*_hip_roll_joint": 0.35,
  r".*_hip_yaw_joint": 0.35,
  r".*_ankle_joint": 0.5,
  r".*_arm_pitch_joint": 1.0,  # 前腿(手)抬脚幅度
  r".*_arm_roll_joint": 0.35,
}

##
# Final config.
##

# 执行器/电机物理完全复用旧 capsule 版（同一套 DM 电机、PD 增益、软限位因子）。
PIKACHU_SL22_ARTICULATION: EntityArticulationInfoCfg = PIKACHU_ARTICULATION


def get_pikachu_sl22_robot_cfg() -> EntityCfg:
  """获取一个新的 v2.2 sample-links 机器人配置实例（双足/行走）。

  默认使用 ``FULL_COLLISION_WITHOUT_SELF``（双脚 + 双手都可与地面接触、
  手脚 condim=3 + 摩擦，满足双足行走与四足撑地）。起步姿态复用微蹲起跑。
  """
  return EntityCfg(
    init_state=KNEES_BENT_SL22_KEYFRAME,
    collisions=(FULL_COLLISION_WITHOUT_SELF,),
    spec_fn=get_spec,
    articulation=PIKACHU_SL22_ARTICULATION,
  )


def get_pikachu_sl22_quad_robot_cfg() -> EntityCfg:
  """获取 v2.2 四足 velocity 机器人配置（四足趴姿起步，无自碰撞，可安全修改）。"""
  return EntityCfg(
    init_state=QUAD_SL22_START,
    collisions=(FULL_COLLISION_WITHOUT_SELF,),
    spec_fn=get_spec,
    articulation=PIKACHU_SL22_ARTICULATION,
  )


__all__ = [
  "BIPED_SL22_KEYFRAME",
  "FEET_ONLY_COLLISION",
  "FULL_COLLISION",
  "FULL_COLLISION_WITHOUT_SELF",
  "HOME_SL22_KEYFRAME",
  "KNEES_BENT_SL22_KEYFRAME",
  "PIKACHU_SL22_ACTION_SCALE",
  "PIKACHU_SL22_ARTICULATION",
  "PIKACHU_SL22_FOOT_GEOM_NAMES",
  "PIKACHU_SL22_HAND_GEOM_NAMES",
  "PIKACHU_SL22_JOINT_NAMES",
  "PIKACHU_SL22_XML",
  "QUAD_SL22_ACTION_SCALE",
  "QUAD_SL22_BODY_AXIS",
  "QUAD_SL22_HAND_GEOM_NAMES",
  "QUAD_SL22_KEYFRAME",
  "QUAD_SL22_START",
  "QUAD_SL22_SUPPORT_GEOM_NAMES",
  "QUAD_SL22_TILT_DEG",
  "QUAD_SL22_TORSO_BODY",
  "get_pikachu_sl22_quad_robot_cfg",
  "get_pikachu_sl22_robot_cfg",
  "get_spec",
]
