"""Pikachu 14-DOF humanoid constants.

Pikachu 是一个浮动基的 14 自由度小型人形机器人，基于 DM 系列
电机（DM-J4310P 与 DM-J3507）驱动，结构见
``xmls/mjcf_14dof_lite/Pikachu_V025_flat_14dof_lite.xml``。

关节清单（左/右各 7 个，合计 14 个有效自由度）：
  - 腿部：``hip_pitch`` / ``hip_roll`` / ``hip_yaw`` / ``knee`` / ``ankle``
  - 臂部：``arm_pitch`` / ``arm_roll``

注意：``arm_yaw`` 与 ``elbow`` 关节在 XML 中被注释停用（对应 body 仍存在，
只是没有 joint），因此不要把它们写进执行器/关键帧配置。

与 Unitree G1 不同，Pikachu 的左右关节在 XML 中使用**相反的转轴方向**
（例如左 hip_pitch 轴为 ``0 1 0``，右为 ``0 -1 0``），因此关键帧里的
左右关节必须分开赋值，不能像 G1 那样用 ``.*_hip_pitch_joint`` 统配。
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# Pikachu 的 MJCF 文件位于
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
  """从 MJCF 文件编译出 MuJoCo 规格（每次调用返回新实例）。"""
  return mujoco.MjSpec.from_file(str(PIKACHU_XML))


##
# Actuator config.
##

# 电机规格（DM 系列）。
#
# TODO: 以下 ``ROTOR_INERTIAS_*``（转子惯量，单位 kg·m²）与 ``GEARS_*``
# （两级行星减速箱传动比，第 0 项固定为 1）为**占位参考值**，取自相近
# 尺寸的 Unitree 电机，正式训练前请替换为 DM-J4310P / DM-J3507 的
# datasheet 真实数值。执行器在 Python 侧创建（见 PIKACHU_ACTUATOR_*），
# 反射惯量 ARMATURE_* 会作为 <position> 执行器的 armature 写入模型。

# DM-J4310P-2EC-V1.1：腿部大扭矩电机（髋 pitch / 膝）。
# 峰值力矩 12.5 N·m（DM-J4310P 规格）。
ROTOR_INERTIAS_J4310 = (
  2.206e-5,  # 电机轴旋转惯量（占位，需 datasheet）
  0.02e-4,  # 中间级惯量（占位）
  0.17e-4,  # 输出级惯量（占位）
)
GEARS_J4310 = (
  1,  # 第一级（固定为 1，行星级从第二项开始）
  10,  # 第二级减速比（占位）
  1,  # 第三级减速比（占位）
)
ARMATURE_J4310 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_J4310, GEARS_J4310
)


# DM-J3507-2EC-3R：小扭矩电机（髋 roll / 髋 yaw / 踝 / 臂 pitch / 臂 roll）。
# 峰值力矩 9 N·m（DM-J3507-3R 规格），3R 表示三级减速。
ROTOR_INERTIAS_J3507 = (
  0.07e-4,  # 电机轴旋转惯量（占位，需 datasheet）
  0.02e-4,  # 中间级惯量（占位）
  0.17e-4,  # 输出级惯量（占位）
)
GEARS_J3507 = (
  1,  # 第一级（固定为 1）
  7.0,  # 第二级减速比
  3.0,  # 第三级减速比（外部，注意速度和扭矩）
)
ARMATURE_J3507 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_J3507, GEARS_J3507
)


# 执行器（电机的关节侧特性）。
#   reflected_inertia: 反射惯量（等效到关节轴的转子惯量，单位 kg·m²），
#                      决定 PD 增益与 armature。
#   velocity_limit:    关节最大角速度（rad/s）。TODO: 需按 datasheet 标定。
#   effort_limit:      关节最大力矩（N·m），来自 DM 电机峰值力矩规格。
ACTUATOR_J4310 = ElectricActuator(
  reflected_inertia=ARMATURE_J4310,
  velocity_limit=30.0,  #  TMax 30  限速63（600rpm）
  effort_limit=10,
)
ACTUATOR_J3507 = ElectricActuator(
  reflected_inertia=ARMATURE_J3507,
  # 最终的输出为经过1：3减速之后的关节速度和力矩
  velocity_limit=30.0 / 3, # 限速31(300rpm)  VMAX:50()  
  effort_limit=5.0 * 3,
)

# PD 位置控制器的固有频率与阻尼比。
#   NATURAL_FREQ: 期望闭环自然角频率 ω（rad/s）。10Hz → ω = 20π ≈ 62.8 rad/s。
#   DAMPING_RATIO: 阻尼比 ζ。2.0 表示过阻尼（临界阻尼为 1.0），
#                  过阻尼响应更快收敛、振荡更小，适合步态控制。
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

# PD 增益由惯量与自然频率推导（临界/过阻尼设计）：
#   Kp（刚度） = J * ω²        —— 位置刚度为惯量乘角频率平方
#   Kd（阻尼） = 2 * ζ * J * ω —— 阻尼系数随阻尼比线性放大
STIFFNESS_J4310 = ARMATURE_J4310 * NATURAL_FREQ**2
STIFFNESS_J3507 = ARMATURE_J3507 * NATURAL_FREQ**2

DAMPING_J4310 = 2.0 * DAMPING_RATIO * ARMATURE_J4310 * NATURAL_FREQ
DAMPING_J3507 = 2.0 * DAMPING_RATIO * ARMATURE_J3507 * NATURAL_FREQ

# 腿部执行器：髋 pitch 与膝（使用 DM-J4310P，大扭矩）。
# target_names_expr 为正则表达式，匹配被该执行器驱动的关节名。
PIKACHU_ACTUATOR_J4310 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",  # 左/右髋前后摆
    ".*_knee_joint",  # 左/右膝
  ),
  stiffness=STIFFNESS_J4310,
  damping=DAMPING_J4310,
  effort_limit=ACTUATOR_J4310.effort_limit,
  armature=ACTUATOR_J4310.reflected_inertia,
)

# 轻量执行器：髋 roll / 髋 yaw / 踝 / 臂 pitch / 臂 roll（DM-J3507-3R）。
PIKACHU_ACTUATOR_J3507 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll_joint",  # 髋内外展
    ".*_hip_yaw_joint",  # 髋旋转
    ".*_ankle_joint",  # 踝（Pikachu 为单自由度踝，无 pitch/roll 之分）
    ".*_arm_pitch_joint",  # 肩前后摆
    ".*_arm_roll_joint",  # 肩外展
  ),
  stiffness=STIFFNESS_J3507,
  damping=DAMPING_J3507,
  effort_limit=ACTUATOR_J3507.effort_limit,
  armature=ACTUATOR_J3507.reflected_inertia,
)


##
# Keyframe config.
##

# 初始状态说明：
#   Pikachu 的左右关节转轴符号相反（左 axis 为 +y/-y，右为镜像），
#   因此 keyframe 中 left/right 关节必须分别写出对称值（互为相反数），
#   不能使用 ``.*_`` 统配符。这些值已在关节限位范围内（见 defaults.xml），
#   具体数值为初步标定，若机器人落地姿态不平，请用 viewer 微调。
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.16),  # 基座(髋部)离地高度 ≈ 腿长 0.14 + 脚底厚度，m
  joint_pos={
    # 腿：微屈，左右对称（右值取反以匹配反向转轴）
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": 0.1,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_knee_joint": -0.1,
    "right_knee_joint": 0.1,
    "left_ankle_joint": -0.1,
    "right_ankle_joint": 0.1,
    # 臂：自然微摆（轻微前摆 + 外展）
    "left_arm_pitch_joint": 0.2,
    "right_arm_pitch_joint": -0.2,
    "left_arm_roll_joint": 0.2,
    "right_arm_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},  # 初始角速度全部为 0
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.14),  # 屈膝后髋部略微降低
  joint_pos={
    # 腿：明显下蹲（膝弯曲约 0.5 rad），踝随膝反向补偿以保持脚底水平
    "left_hip_pitch_joint": -0.3,
    "right_hip_pitch_joint": 0.3,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_knee_joint": -0.5,
    "right_knee_joint": 0.5,
    "left_ankle_joint": -0.5,
    "right_ankle_joint": 0.5,
    # 臂：与 HOME 相同
    "left_arm_pitch_joint": 0.2,
    "right_arm_pitch_joint": -0.2,
    "left_arm_roll_joint": 0.2,
    "right_arm_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# 注意：Pikachu 的碰撞 geom 在 XML 中已显式命名——每个 link 两个 geom，
# 视觉 geom 为 ``<link>_visual``（``contype=0, conaffinity=0, group=1``），
# 碰撞 geom 为 ``<link>_collision``（参与碰撞）。因此下面的正则统一匹配
# ``.*_collision``，视觉 geom 由 disable_other_geoms 自动禁用碰撞。
#
# 另外，主 XML 已移除对 ``scene.xml`` 的 include：Pikachu 资产自带的那个
# scene 只包含一个 ``floor`` 平面与灯光，而 mjlab 场景的地面由
# ``TerrainEntity`` 提供（见 tracking 任务 ``TerrainEntityCfg(terrain_type
# =\"plane\")``）。若保留该 floor，它会被这里的 disable_other_geoms 清零，
# 并与场景地形重叠产生双重地面。

# 全身碰撞：启用所有 ``.*_collision`` geom 的碰撞（含自碰撞）。
# 脚底（ankle）用 condim=3（平面接触），其余用 condim=1（点接触）。
# 注意 geom 名 = mesh 名（``left_ankle_link``）+ ``_collision`` 后缀。
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={r"^(left|right)_ankle_link_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_ankle_link_collision$": 1, ".*_collision": 0},
  friction={r"^(left|right)_ankle_link_collision$": (0.6,)},
)

# 无自碰撞版本：contype=0 使机器人自身 geom 之间不产生接触，
# 但仍保留与地面（contype=1）的碰撞能力由 conaffinity=1 表达。
# 注意：MuJoCo 要求双方 contype & conaffinity 同时匹配才产生接触，
# 因此此配置下机器人只与 contype=1 且 conaffinity=1 的环境物体接触。
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_ankle_link_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_ankle_link_collision$": 1, ".*_collision": 0},
  friction={r"^(left|right)_ankle_link_collision$": (0.6,)},
)

# 仅脚底碰撞：只保留左右踝（脚底）与地面接触，其余 geom 全部禁用
# （disable_other_geoms=True 会把未匹配 geom 的 contype/conaffinity 清零）。
# 适合平地行走训练——Pikachu 的自碰撞对已在 contact.xml 中大量排除，
# 训练时通常只需脚-地接触。
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_ankle_link_collision$",),
  contype=1,
  conaffinity=1,
  condim=3,  # 脚底为平面接触（3 维法向 + 切向）
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

PIKACHU_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PIKACHU_ACTUATOR_J4310,
    PIKACHU_ACTUATOR_J3507,
  ),
  soft_joint_pos_limit_factor=0.9,  # 关节限位软边界系数（保留 10% 余量）
)


def get_pikachu_robot_cfg() -> EntityCfg:
  """获取一个新的 Pikachu 机器人配置实例。

  每次调用返回新的 EntityCfg，避免配置被多处共享时发生意外变更。
  默认使用``FEET_ONLY_COLLISION``（见上文碰撞配置注释）：因为视觉 geom
  与碰撞 geom 同名，启用全身碰撞会误触发视觉 geom；且自碰撞已在
  contact.xml 中排除，训练平地行走只需脚-地接触。
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,  # 默认从微蹲姿态起跑，更稳定
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=PIKACHU_ARTICULATION,
  )


# 动作缩放系数：把 PD 目标位置（rad）映射到实际动作幅度。
# 每个执行器按 ``0.25 * effort_limit / stiffness`` 缩放，含义是：
#   在 PD 控制下，位置误差每产生 1 rad 偏差对应的力矩增益为 stiffness，
#   除以 effort_limit 得到"单位动作下占用的力矩预算比例"，再乘 0.25 作为
#   保守的动作幅度上限（避免单个动作指令直接顶满电机力矩）。
PIKACHU_ACTION_SCALE: dict[str, float] = {}
for a in PIKACHU_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    PIKACHU_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_pikachu_robot_cfg())

  viewer.launch(robot.spec.compile())
