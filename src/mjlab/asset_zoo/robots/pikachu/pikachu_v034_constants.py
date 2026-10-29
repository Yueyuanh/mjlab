"""Pikachu v034 常量：18 个驱动关节、4 个被动四杆关节。

采用与 v033 相同的显式配置结构。质量、质心和惯量由训练 MJCF 提供；
此处列出电机参数、初始姿态、碰撞配置及策略接口顺序。
电机参数沿用本次接入值，10 Hz 控制增益仍是初始估计值。
"""

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

ASSET_DIR = MJLAB_SRC_PATH / "asset_zoo/robots/pikachu/xmls/pikachu_v034"
PIKACHU_V034_XML: Path = ASSET_DIR / "mjcf/pikachu_v034_train.xml"
# Mirrored left-leg hinge limits; see
# ``scripts/build_v034_symmetric_variant.py`` and
# ``provenance/symmetric_limits.json``.
PIKACHU_V034_SYMMETRIC_XML: Path = ASSET_DIR / "mjcf/pikachu_v034_train_symmetric.xml"


def get_spec() -> mujoco.MjSpec:
  """每次调用返回独立的 MuJoCo 规格。"""
  return mujoco.MjSpec.from_file(str(PIKACHU_V034_XML))


def get_symmetric_spec() -> mujoco.MjSpec:
  """Variant with physically mirrored left knee/ankle limits."""
  return mujoco.MjSpec.from_file(str(PIKACHU_V034_SYMMETRIC_XML))


##
# Actuator config.
##

ROTOR_INERTIAS_J4310 = 2.206e-5
GEARS_J4310 = (1, 10, 1)
ARMATURE_J4310 = reflected_inertia(
  ROTOR_INERTIAS_J4310, GEARS_J4310[1] * GEARS_J4310[2]
)

ROTOR_INERTIAS_J3507 = 6.69169e-6
GEARS_J3507 = (1, 7.0, 3.0)
ARMATURE_J3507 = reflected_inertia(
  ROTOR_INERTIAS_J3507, GEARS_J3507[1] * GEARS_J3507[2]
)

ACTUATOR_J4310 = ElectricActuator(
  reflected_inertia=ARMATURE_J4310,
  velocity_limit=12.57,
  effort_limit=12.5,
)
ACTUATOR_J3507 = ElectricActuator(
  reflected_inertia=ARMATURE_J3507,
  velocity_limit=15.71,
  effort_limit=3.0,
)

NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0
STIFFNESS_J4310 = ARMATURE_J4310 * NATURAL_FREQ**2
STIFFNESS_J3507 = ARMATURE_J3507 * NATURAL_FREQ**2
DAMPING_J4310 = 2.0 * DAMPING_RATIO * ARMATURE_J4310 * NATURAL_FREQ
DAMPING_J3507 = 2.0 * DAMPING_RATIO * ARMATURE_J3507 * NATURAL_FREQ

# knee / knee_mid 为被动关节，不配置驱动器。
PIKACHU_V034_ACTUATOR_J4310 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_up_joint",
    ".*_ankle_joint",
    ".*_arm_pitch_joint",
    ".*_arm_roll_joint",
  ),
  stiffness=STIFFNESS_J4310,
  damping=DAMPING_J4310,
  effort_limit=ACTUATOR_J4310.effort_limit,
  armature=ACTUATOR_J4310.reflected_inertia,
)
PIKACHU_V034_ACTUATOR_J3507 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_arm_yaw_joint", ".*_elbow_joint"),
  stiffness=STIFFNESS_J3507,
  damping=DAMPING_J3507,
  effort_limit=ACTUATOR_J3507.effort_limit,
  armature=ACTUATOR_J3507.reflected_inertia,
)

##
# Keyframe config.
##

# 站立高度沿用 geometry.json 的已测量值，初始化抬高量保持 3 mm。
STAND_HEIGHT = 0.23791378869623137
BIPED_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, STAND_HEIGHT + 0.003),
  joint_pos={
    "left_arm_roll_joint": math.radians(-53),
    "right_arm_roll_joint": math.radians(53),
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOOT_GEOMS = r"(left|right)_foot[1-5]_collision"
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(FOOT_GEOMS,),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.8,),
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(PIKACHU_V034_ACTUATOR_J4310, PIKACHU_V034_ACTUATOR_J3507),
  soft_joint_pos_limit_factor=0.95,
)


def get_biped_robot_cfg(symmetric: bool = False) -> EntityCfg:
  return EntityCfg(
    spec_fn=get_symmetric_spec if symmetric else get_spec,
    init_state=BIPED_KEYFRAME,
    articulation=ARTICULATION,
    collisions=(FEET_ONLY_COLLISION,),
  )


##
# Joint names and policy ordering.
##

# 保持已有接口名及顺序，避免影响历史模型和任务配置。
DRIVEN_KINDS = (
  "hip_pitch",
  "hip_roll",
  "hip_yaw",
  "knee_up",
  "ankle",
  "arm_pitch",
  "arm_roll",
  "arm_yaw",
  "elbow",
)
JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_up_joint",
  "left_ankle_joint",
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "left_arm_yaw_joint",
  "left_elbow_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_up_joint",
  "right_ankle_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
  "right_arm_yaw_joint",
  "right_elbow_joint",
)
PASSIVE_JOINTS = (
  "left_knee_joint",
  "left_knee_mid_joint",
  "right_knee_joint",
  "right_knee_mid_joint",
)

# v033 策略动作顺序：右腿、左腿、右臂、左臂。
POLICY_ACTION_JOINT_NAMES = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_up_joint",
  "right_ankle_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_up_joint",
  "left_ankle_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
  "right_arm_yaw_joint",
  "right_elbow_joint",
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "left_arm_yaw_joint",
  "left_elbow_joint",
)

# 观测顺序包含四个被动膝关节。
POLICY_OBSERVATION_JOINT_NAMES = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_up_joint",
  "right_knee_mid_joint",
  "right_knee_joint",
  "right_ankle_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_up_joint",
  "left_knee_mid_joint",
  "left_knee_joint",
  "left_ankle_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
  "right_arm_yaw_joint",
  "right_elbow_joint",
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "left_arm_yaw_joint",
  "left_elbow_joint",
)
