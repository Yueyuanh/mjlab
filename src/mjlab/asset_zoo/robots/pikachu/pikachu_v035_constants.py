"""Pikachu v035 常量：18 个全直驱关节，无闭链、无被动关节。

v035 相对 v034 的机构变化：膝关节取消四杆闭链，改为 DM-4340 直驱
（额定 12 N·m，峰值 40 N·m，额定 36 rpm）。因此整机只有 18 个驱动
关节（每腿 5 + 每臂 4），没有 ``knee`` / ``knee_mid`` 被动关节。

质量/质心/惯量全部来自训练 MJCF
``xmls/pikachu_v035/mjcf/pikachu_v035_train.xml``；左右惯量已由
``pikachu_urdf/pikachu_v035/scripts/prepare_v035_urdf.py`` 按左侧镜像
（原始 CAD 导出两侧最大差 73%）。关节限位也在同一脚本写入 URDF，
经 ``scripts/build_v035_train_xml.py`` 转入 MJCF。

注意：
  * DM-4340 的转子惯量 datasheet 未给出，这里按 4310 的 4 倍估
    （``ROTOR_INERTIAS_J4340 = 8.8e-5``，减速比 10），得到反射惯量
    ``8.8e-3 kg·m²``。该值只影响 PD 增益与 armature，拿到真实转子惯量
    后应替换。
  * 膝关节左右轴相反、限位不对称：左侧 ``[-30°, +90°]``，右侧
    ``[-90°, +30°]``（MJCF 已按 ``[-upper, -lower]`` 写入）。
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

ASSET_DIR = MJLAB_SRC_PATH / "asset_zoo/robots/pikachu/xmls/pikachu_v035"
PIKACHU_V035_XML: Path = ASSET_DIR / "mjcf/pikachu_v035_train.xml"


def get_spec() -> mujoco.MjSpec:
  """每次调用返回独立的 MuJoCo 规格。"""
  return mujoco.MjSpec.from_file(str(PIKACHU_V035_XML))


##
# Actuator config.
##

ROTOR_INERTIAS_J4310 = 2.206e-5
GEARS_J4310 = (1, 10, 1)
ARMATURE_J4310 = reflected_inertia(
  ROTOR_INERTIAS_J4310, GEARS_J4310[1] * GEARS_J4310[2]
)

# TODO(datasheet): 4340 转子惯量未公布，暂按 4310 的 4 倍估算。
ROTOR_INERTIAS_J4340 = 8.8e-5
GEARS_J4340 = (1, 10, 1)
ARMATURE_J4340 = reflected_inertia(
  ROTOR_INERTIAS_J4340, GEARS_J4340[1] * GEARS_J4340[2]
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
ACTUATOR_J4340 = ElectricActuator(
  reflected_inertia=ARMATURE_J4340,
  velocity_limit=3.7699111843,  # 36 rpm
  effort_limit=40.0,  # 峰值
)
ACTUATOR_J3507 = ElectricActuator(
  reflected_inertia=ARMATURE_J3507,
  velocity_limit=15.71,
  effort_limit=3.0,
)

NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0
STIFFNESS_J4310 = ARMATURE_J4310 * NATURAL_FREQ**2
STIFFNESS_J4340 = ARMATURE_J4340 * NATURAL_FREQ**2
STIFFNESS_J3507 = ARMATURE_J3507 * NATURAL_FREQ**2
DAMPING_J4310 = 2.0 * DAMPING_RATIO * ARMATURE_J4310 * NATURAL_FREQ
DAMPING_J4340 = 2.0 * DAMPING_RATIO * ARMATURE_J4340 * NATURAL_FREQ
DAMPING_J3507 = 2.0 * DAMPING_RATIO * ARMATURE_J3507 * NATURAL_FREQ

PIKACHU_V035_ACTUATOR_J4310 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_ankle_joint",
    ".*_arm_pitch_joint",
    ".*_arm_roll_joint",
  ),
  stiffness=STIFFNESS_J4310,
  damping=DAMPING_J4310,
  effort_limit=ACTUATOR_J4310.effort_limit,
  armature=ACTUATOR_J4310.reflected_inertia,
)
PIKACHU_V035_ACTUATOR_J4340 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=STIFFNESS_J4340,
  damping=DAMPING_J4340,
  effort_limit=ACTUATOR_J4340.effort_limit,
  armature=ACTUATOR_J4340.reflected_inertia,
)
PIKACHU_V035_ACTUATOR_J3507 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_arm_yaw_joint", ".*_elbow_joint"),
  stiffness=STIFFNESS_J3507,
  damping=DAMPING_J3507,
  effort_limit=ACTUATOR_J3507.effort_limit,
  armature=ACTUATOR_J3507.reflected_inertia,
)

##
# Keyframe config.
##

# qpos 全 0 时 base_link 的实测高度（CAD 零位：hip 20°/knee 20°/ankle 10°
# 相对完全直立，鞋底刚好贴地）。见 geometry.json。
STAND_HEIGHT = 0.22957707433403704
# 手臂默认下垂。v035 的 ``arm_roll`` 轴是 (-1,0,0)，而 v033/v034 是 (+1,0,0)，
# 所以从 v034 抄来的 L-53°/R+53° 在 v035 上符号整体反了：负角把上臂抬到水平面
# 上方 53°，正角压到下方 53°。实测肩 z=0.396 时旧位姿的手在 z=0.476–0.551（肩
# 上方 8–15.5 cm），渲染出来就是“双手举起”；改成 L+53°/R-53° 后手落在
# z=0.311（肩下方 8.4 cm），仍在 ±90° 限位内，限位不需要动。
# 动作是 ``use_default_offset=True`` 的绝对位姿（目标 = 本默认位 + action×scale），
# 所以策略输出 0 就指向这个下垂位，已有 checkpoint 的臂部动作会跟着翻到下垂侧；
# 观测也是相对默认位姿的。代价是这些策略的步态是按上举姿态学的：同 seed 实测前进
# 0.214→0.183（walk v1）、0.187→0.159（v5）m/s，无摔倒——要恢复原指标需按新默认
# 位姿重训（见 doc/pikachu_v035_20261028/README.md §7 第 3 条）。评估里的 arm_roll
# 包络是修复前的绝对角度口径，v035_biped_e1 的 ``arm_position_window`` 已同步镜像。
BIPED_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, STAND_HEIGHT + 0.003),
  joint_pos={
    "left_arm_roll_joint": math.radians(53),
    "right_arm_roll_joint": math.radians(-53),
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
  actuators=(
    PIKACHU_V035_ACTUATOR_J4310,
    PIKACHU_V035_ACTUATOR_J4340,
    PIKACHU_V035_ACTUATOR_J3507,
  ),
  soft_joint_pos_limit_factor=0.95,
)


def get_biped_robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
    init_state=BIPED_KEYFRAME,
    articulation=ARTICULATION,
    collisions=(FEET_ONLY_COLLISION,),
  )


##
# Joint names and policy ordering.
##

DRIVEN_KINDS = (
  "hip_pitch",
  "hip_roll",
  "hip_yaw",
  "knee",
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
  "left_knee_joint",
  "left_ankle_joint",
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "left_arm_yaw_joint",
  "left_elbow_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
  "right_arm_yaw_joint",
  "right_elbow_joint",
)

# 策略动作顺序：右腿、左腿、右臂、左臂。
POLICY_ACTION_JOINT_NAMES = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
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

# v035 无被动关节，观测顺序与动作顺序一致。
POLICY_OBSERVATION_JOINT_NAMES = POLICY_ACTION_JOINT_NAMES

ACTUATOR_GROUPS = (
  (r".*_knee_joint", ACTUATOR_J4340),
  (
    r".*(hip_pitch|hip_roll|hip_yaw|ankle|arm_pitch|arm_roll)_joint",
    ACTUATOR_J4310,
  ),
  (r".*(arm_yaw|elbow)_joint", ACTUATOR_J3507),
)
