"""Pikachu **27-dof sample-links** 训练常量（`pikachu_27dof_train.xml`）。

机器人定义取自 ``pikachu_xml/mjcf/pikachu_sample_links_27dof.xml``（皮卡丘
拟真全身 box 骨架：头 3 + 耳 4 + 尾 2 + 腿 10 + 臂 8 = 27 个 hinge 关节）。
本模块为 mjlab 训练提供一个收紧的 **训练体** ``pikachu_27dof_train.xml``
（见其文件头注释）：

  - 去掉 ``<include file="scene.xml"/>``：其自带 ``floor`` 与 mjlab 的
    ``TerrainEntity(plane)`` 重叠会形成双地面，故移除，地面由场景提供。
  - 给两个踝骨实心碰撞 box 显式命名 ``*_ankle_collision``：源 XML 的 geom
    全部未命名，CollisionCfg 靠 ``geom_names_expr`` 正则匹配，必须命名才能做
    只留脚底的碰撞裁剪。

与旧 14-dof / 14-joint 机器人的关系：腿 10 关节的名字/轴向与
``pikachu_constants_base.LEG_JOINT_NAMES`` 完全同名（left/right 髋 pitch、
roll、yaw、knee、ankle），因此**纯腿 10 关节的参考 npz**（``pikachu_walk_0.185_base.npz``）
可原样复用（``MotionCommand.joint_names`` 按名字索引，不依赖关节序）。

## 上肢作为"随机摆动干扰"

本机器人除行走腿外，上肢还有 17 个关节：臂 arm_pitch/roll/yaw/elbow（8）、
耳 ear_pitch/roll（4）、尾 tail_pitch/tail_yaw（2）、头 head_pitch/yaw/roll（3）。
这些不参与策略动作，而由两个零维副作用 ActionTerm 施加**随机摆动**作为外源
动力学干扰（见 ``envs/mdp/actions/random_flail.py``）：

  - "手动随机摆动" → 臂 8 关节（``ARM_JOINT_NAMES``）
  - "耳朵尾巴随机摆动" → 耳 4 + 尾 2 关节（``EXTREMITY_JOINT_NAMES``）

策略观测包含全部 27 关节位置/速度（能感知上肢摆动并主动平衡补偿），
但动作只驱动腿 10 关节。上肢执行器用 J3507 轻电机（Kp=12.19），足够带动小
质量臂/耳/尾做 PD 摆动；头虽然源组件质量 11.96 kg，但默认不参与摆动。
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.pikachu.pikachu_constants import (
  DAMPING_J3507,
  DAMPING_J4310,
  STIFFNESS_J3507,
  STIFFNESS_J4310,
)
from mjlab.asset_zoo.robots.pikachu.pikachu_constants_base import (
  LEG_JOINT_NAMES,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# 训练体 XML 与 assets。
#
# ``pikachu_27dof_train.xml`` 是源 ``pikachu_sample_links_27dof.xml`` 的
# mjlab 蒸馏版：去掉 floor include、给踝碰撞 geom 命名（见模块 docstring）。
##

PIKACHU_27DOF_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "pikachu_xml"
  / "mjcf"
  / "pikachu_27dof_train.xml"
)
assert PIKACHU_27DOF_XML.exists()


def get_spec() -> mujoco.MjSpec:
  """编译出训练体机器人规格（每次调用返回新实例）。"""
  return mujoco.MjSpec.from_file(str(PIKACHU_27DOF_XML))


##
# 关节分组（名字全部来自训练体 XML）。
##

# 腿 10（顺序 = 参考 npz 列序，见 pikachu_constants_base.LEG_JOINT_NAMES）。
# 直接复用共享 LEG_JOINT_NAMES，保证 tracking 参考对齐。
# 腿 = LEG_JOINT_NAMES

# 臂 8（"手动随机摆动"干扰组）：左/右 arm_pitch / arm_roll / arm_yaw / elbow。
ARM_JOINT_NAMES: tuple[str, ...] = (
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "left_arm_yaw_joint",
  "left_elbow_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
  "right_arm_yaw_joint",
  "right_elbow_joint",
)

# 耳 4 + 尾 2（"耳朵尾巴随机摆动"干扰组）。
EAR_JOINT_NAMES: tuple[str, ...] = (
  "left_ear_pitch_joint",
  "left_ear_roll_joint",
  "right_ear_pitch_joint",
  "right_ear_roll_joint",
)
TAIL_JOINT_NAMES: tuple[str, ...] = ("tail_pitch_joint", "tail_yaw_joint")
EXTREMITY_JOINT_NAMES: tuple[str, ...] = EAR_JOINT_NAMES + TAIL_JOINT_NAMES

# 头 3（不参与摆动，保持默认姿态由 pose 正则约束）。
HEAD_JOINT_NAMES: tuple[str, ...] = (
  "head_pitch_joint",
  "head_yaw_joint",
  "head_roll_joint",
)

# 全部 27 个执行器关节（策略观测覆盖全部，让策略感知上肢摆动并补偿）。
ALL_JOINT_NAMES: tuple[str, ...] = (
  HEAD_JOINT_NAMES
  + EAR_JOINT_NAMES
  + TAIL_JOINT_NAMES
  + LEG_JOINT_NAMES
  + ARM_JOINT_NAMES
)

##
# 执行器（PD 电机）：腿沿用 DM J4310/J3507；上肢（臂/头/耳/尾）用轻电机。
#
# 电机物理完全复用 ``pikachu_constants.py`` 推导的 Kp/Kd/effort/armature。
##

# 受人尊敬腿：髋 pitch + 膝 —— DM-J4310P 大扭矩。
_LEG_HEAVY_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_J4310,
  damping=DAMPING_J4310,
  effort_limit=10.0,
  armature=reflected_inertia(2.206e-5, 10.0),
)

# 腿轻：髋 roll / 髋 yaw / 踝 —— DM-J3507。
_LEG_LIGHT_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_ankle_joint"),
  stiffness=STIFFNESS_J3507,
  damping=DAMPING_J3507,
  effort_limit=15.0,
  armature=reflected_inertia(6.69169e-6, 21.0),
)

# 上肢：臂（pitch/roll/yaw/elbow）+ 头（pitch/yaw/roll）+ 耳（pitch/roll）+
# 尾（pitch/yaw）—— DM-J3507 轻电机，供随机摆动干扰驱动。
_UPPER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_arm_pitch_joint",
    ".*_arm_roll_joint",
    ".*_arm_yaw_joint",
    ".*_elbow_joint",
    "head_.*_joint",
    "left_ear_.*_joint",
    "right_ear_.*_joint",
    "tail_.*_joint",
  ),
  stiffness=STIFFNESS_J3507,
  damping=DAMPING_J3507,
  effort_limit=15.0,
  armature=reflected_inertia(6.69169e-6, 21.0),
)

PIKACHU_27DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(_LEG_HEAVY_ACTUATOR, _LEG_LIGHT_ACTUATOR, _UPPER_ACTUATOR),
  soft_joint_pos_limit_factor=1.0,
)

##
# 关键帧：腿微蹲起步（值与旧 KNEES_BENT 一致，left/right 轴反相），上肢零位。
##
KNEES_BENT_27DOF_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.165),  # 站立 geniz 
  joint_pos={
    # 腿：明显下蹲，踝反向补偿保持脚底水平，左右轴反相（同旧 KNEES_BENT）。
    "left_hip_pitch_joint": -0.0,
    "right_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_knee_joint": -0.0,
    "right_knee_joint": 0.0,
    "left_ankle_joint": -0.0,
    "right_ankle_joint": 0.0,
    # 臂：自然放松。
    "left_arm_pitch_joint": 0.0,
    "right_arm_pitch_joint": -0.0,
    "left_arm_roll_joint": 0.0,
    "right_arm_roll_joint": -0.0,
    "left_arm_yaw_joint": 0.0,
    "right_arm_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "right_elbow_joint": 0.0,
    # 头 / 耳 / 尾：零位。
    "head_pitch_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_roll_joint": 0.0,
    "left_ear_pitch_joint": 0.0,
    "left_ear_roll_joint": 0.0,
    "right_ear_pitch_joint": 0.0,
    "right_ear_roll_joint": 0.0,
    "tail_pitch_joint": 0.0,
    "tail_yaw_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

# 双足（biped）模式起步姿态 = 微蹲起跑。
PICK_START_KEYFRAME = KNEES_BENT_27DOF_KEYFRAME

##
# 碰撞配置：只保留双脚踝-地面（平地行走）。其余 geom（视觉 + 全身实心 box）
# 由 disable_other_geoms 清零 contype/conaffinity，不参与碰撞、不互相夹。
##
PIKACHU_27DOF_FOOT_GEOM_NAMES = r"^(left|right)_ankle_collision$"

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(PIKACHU_27DOF_FOOT_GEOM_NAMES,),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# 腿动作缩放（velocity 策略动作只驱动腿；与旧双足一致）。
##
LEG_ACTION_SCALE: dict[str, float] = {
  r".*_hip_pitch_joint": 0.5,
  r".*_knee_joint": 0.5,
  r".*_hip_roll_joint": 0.35,
  r".*_hip_yaw_joint": 0.35,
  r".*_ankle_joint": 0.35,
}


def get_pikachu_27dof_robot_cfg() -> EntityCfg:
  """获取 27-dof 双足机器人配置（微蹲起步，只脚底碰撞，全关节执行器）。"""
  return EntityCfg(
    init_state=KNEES_BENT_27DOF_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=PIKACHU_27DOF_ARTICULATION,
  )


__all__ = [
  "ALL_JOINT_NAMES",
  "ARM_JOINT_NAMES",
  "EAR_JOINT_NAMES",
  "EXTREMITY_JOINT_NAMES",
  "FEET_ONLY_COLLISION",
  "HEAD_JOINT_NAMES",
  "KNEES_BENT_27DOF_KEYFRAME",
  "LEG_ACTION_SCALE",
  "LEG_JOINT_NAMES",
  "PIKACHU_27DOF_ARTICULATION",
  "PIKACHU_27DOF_FOOT_GEOM_NAMES",
  "PIKACHU_27DOF_XML",
  "PICK_START_KEYFRAME",
  "TAIL_JOINT_NAMES",
  "get_pikachu_27dof_robot_cfg",
  "get_spec",
]