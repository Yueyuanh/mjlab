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
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# Pikachu 的 MJCF 文件位于
#
# 本模块（pikachu_constants.py）固定使用 **capsule 碰撞体** XML。想对比
# lite 碰撞体（原始全身碰撞体）时，用独立模块 ``pikachu_constants_lite.py``
# （导出同名 get_pikachu_robot_cfg / PIKACHU_FOOT_GEOM_NAMES，切换只需改
# 任务 env_cfg 的 import 来源），不要再改这里的 XML 行。
PIKACHU_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "pikachu"
  / "xmls"
  / "mjcf_14dof_lite"
  / "Pikachu_V025_flat_14dof_capsule.xml"  # 简化碰撞体（胶囊脚底）
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
# datasheet 只给出**一个电机转子惯量**（减速箱内部各级齿轮惯量不提供），
# 这与 go1 的做法一致（见 unitree_go1/go1_constants.py：单一 ROTOR_INERTIA，
# 反射惯量 = I · 总减速比²）。因此：
#   - ROTOR_INERTIAS_* 为单一转子惯量（单位 kg·m²），TODO: 换成 datasheet 真实值。
#   - GEARS_* 保留两级行星减速比（总比 = 第 2 项 × 第 3 项，第 0 项固定为 1）。
#   - ARMATURE_* = ROTOR_INERTIAS_* × 总减速比²，忽略减速箱各级齿轮自身惯量
#     （占比 <5%，对 Kp/Kd 影响可忽略）。
# 执行器在 Python 侧创建（见 PIKACHU_ACTUATOR_*），反射惯量会作为
# <position> 执行器的 armature 写入模型。

# DM-J4310P-2EC-V1.1：腿部大扭矩电机（髋 pitch / 膝）。
# 峰值力矩 12.5 N·m（DM-J4310P 规格）。总减速比 = 10×1 = 10。
ROTOR_INERTIAS_J4310 = 2.206e-5  # 电机转子惯量（kg·m²，datasheet）
GEARS_J4310 = (
  1,  # 第一级（固定为 1，行星级从第二项开始）
  10,  # 第二级减速比（占位）
  1,  # 第三级减速比（占位）
)
ARMATURE_J4310 = reflected_inertia(
  ROTOR_INERTIAS_J4310, GEARS_J4310[1] * GEARS_J4310[2]
)


# DM-J3507-2EC-3R：小扭矩电机（髋 roll / 髋 yaw / 踝 / 臂 pitch / 臂 roll）。
# 峰值力矩 9 N·m（DM-J3507-3R 规格），3R 表示三级减速。
# 总减速比 = 7×3 = 21。
ROTOR_INERTIAS_J3507 = 6.69169e-6  # 电机转子惯量（kg·m²，datasheet）
GEARS_J3507 = (
  1,  # 第一级（固定为 1）
  7.0,  # 第二级减速比
  3.0,  # 第三级减速比（外部，注意速度和扭矩）
)
ARMATURE_J3507 = reflected_inertia(
  ROTOR_INERTIAS_J3507, GEARS_J3507[1] * GEARS_J3507[2]
)


# 执行器（电机的关节侧特性）。
#   reflected_inertia: 反射惯量（等效到关节轴的转子惯量，单位 kg·m²），
#                      决定 PD 增益与 armature。
#   velocity_limit:    关节最大角速度（rad/s）。TODO: 需按 datasheet 标定。
#   effort_limit:      关节最大力矩（N·m），来自 DM 电机峰值力矩规格。
ACTUATOR_J4310 = ElectricActuator(
  reflected_inertia=ARMATURE_J4310,
  velocity_limit=20.0,  #  TMax 30  限速20.94（200rpm）
  effort_limit=10,  #  TMax 12.5  软件力矩上限 10N·m
)
ACTUATOR_J3507 = ElectricActuator(
  reflected_inertia=ARMATURE_J3507,
  # 最终的输出为经过1：3减速之后的关节速度和力矩
  velocity_limit=30.0 / 3,  # 限速31(300rpm)  VMAX:50()
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

# 反射惯量:J = I_motor × 总减速比²(Go1 单惯量式)
# J4310:2.206e-5 × 10² = 0.002206 kg·m²
# J3507:0.07e-4 × 21² = 0.003087 kg·m²
# Kp:J × ω²
# J4310:0.002206 × 3947.8 = 8.709 N·m/rad
# J3507:0.003087 × 3947.8 = 12.187 N·m/rad
# Kd:2ζ × J × ω
# J4310:2×2×0.002206×62.83 = 0.554 N·m·s/rad
# J3507:2×2×0.003087×62.83 = 0.776 N·m·s/rad

STIFFNESS_J4310 = ARMATURE_J4310 * NATURAL_FREQ**2  # Kp 8.709
STIFFNESS_J3507 = ARMATURE_J3507 * NATURAL_FREQ**2  # Kp 12.187

DAMPING_J4310 = 2.0 * DAMPING_RATIO * ARMATURE_J4310 * NATURAL_FREQ  # Kd 0.554
DAMPING_J3507 = 2.0 * DAMPING_RATIO * ARMATURE_J3507 * NATURAL_FREQ  # Kd 0.776

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

# 双足（biped）模式标称姿态：状态转换任务的"双足态"基准。
# 复用 KNEES_BENT_KEYFRAME（微蹲起跑，左右轴反向已按 XML 对齐）。
BIPED_KEYFRAME = KNEES_BENT_KEYFRAME

# 四足（quadruped）模式标称姿态：base 前俯 + 双手撑地 + 双脚撑地。
#
# ✅ 几何已验证（scripts/calibrate_quad_keyframe.py）：base 前俯 90°（水平
# 趴地，g_b≈(+1,0,0)）+ arm_pitch ±1.2 + pos.z=0.075 时，双手/双脚同时着地、
# 躯干离地约 3.5cm，无 body 陷地。这是"手脚四点支撑"的几何自洽初值。
# ⚠️ (手动微调): 若希望更"四足感"（躯干前俯 <90°、臀部抬高），在
# viewer 中手动调 rot / arm_pitch / hip_pitch / pos.z，并用校准脚本复查
# 手脚着地 + 躯干不陷地。
#   - 符号约定：左右关节转轴反向，left/right 值互为相反数（left 为负=向前
#     屈/摆，与 KNEES_BENT 一致）。
#   - arm_pitch 陷阱：增大 |arm_pitch| 超过约 1.5 rad 时手臂会向前挥过头、
#     手反而上扬，因此手臂前伸角应控制在 [0, 1.5] 附近微调。
#   - 关节限位参考 defaults.xml：left_hip_pitch [-2.44, 0.698]、left_knee
#     [-1.57, 0]、left_ankle [-1.05, 0.52]、left_arm_pitch [-3.14, 1.05]。
QUAD_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.15),
  rot=(0.593497, 0.0, 0.804837, 0),
  # rot=(0.7071, 0, 0.7071, 0),
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
    "left_ankle_joint": -0.7,
    "right_ankle_joint": 0.7,
    "left_arm_pitch_joint": -2.0,
    "right_arm_pitch_joint": 2.0,
    "left_arm_roll_joint": 0.0,
    "right_arm_roll_joint": -0.0,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# 注意：Pikachu 的碰撞 geom 在 XML 中已显式命名——每个 link 两个 geom，
# 视觉 geom 为 ``<link>_visual``（``contype=0, conaffinity=0, group=2``），
# 碰撞 geom 为 ``<link>_collision``（参与碰撞）。因此下面的正则统一匹配
# ``.*_collision``，视觉 geom 由 disable_other_geoms 自动禁用碰撞。
#
# 另外，主 XML 已移除对 ``scene.xml`` 的 include：Pikachu 资产自带的那个
# scene 只包含一个 ``floor`` 平面与灯光，而 mjlab 场景的地面由
# ``TerrainEntity`` 提供（见 tracking 任务 ``TerrainEntityCfg(terrain_type
# =\"plane\")``）。若保留该 floor，它会被这里的 disable_other_geoms 清零，
# 并与场景地形重叠产生双重地面。


# 本文件（pikachu_constants.py）对应 **capsule 碰撞体** XML：
# Pikachu_V025_flat_14dof_capsule.xml —— 脚底为 G1 风格 7 个脚趾胶囊
# （foot1~7）。若想对比 **lite 碰撞体**（原始全身碰撞体，脚底为整块
# ankle 网格），用 ``pikachu_constants_lite.py``（见其模块 docstring）。

# 脚底碰撞 geom 正则（capsule：G1 风格脚趾胶囊，挂在 ankle body 子树内）。
# 供任务 env_cfg 的 foot_friction 事件（``geom_friction``）按机器人资产匹配，
# 也用于下面的碰撞配置。
PIKACHU_FOOT_GEOM_NAMES = r"^(left|right)_foot[1-7]_collision$"

# 全身碰撞：启用所有 ``.*_collision`` geom 的碰撞（含自碰撞）。
# 脚部胶囊（foot1~7）用 condim=3（平面接触）、priority=1、friction=0.6，
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
# 注意：MuJoCo 要求双方 contype & conaffinity 同时匹配才产生接触，
# 因此此配置下机器人只与 contype=1 且 conaffinity=1 的环境物体接触。
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

# 仅脚底碰撞：只保留脚部胶囊与地面接触，其余 geom 全部禁用
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

##
# Final config.
##

PIKACHU_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PIKACHU_ACTUATOR_J4310,
    PIKACHU_ACTUATOR_J3507,
  ),
  soft_joint_pos_limit_factor=1.0,  # 软限位 = 硬限位（不保留余量，避免参考动作被 clip）
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
    collisions=(
      FULL_COLLISION_WITHOUT_SELF,
    ),  # FEET_ONLY_COLLISION   FULL_COLLISION_WITHOUT_SELF
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
