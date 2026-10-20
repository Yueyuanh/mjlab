"""Pikachu v033（并联四杆膝 + 双臂 8-dof）常量。

v033 是 v031 的**整机升级版**（SolidWorks -> easymjcf 直出）：腿链沿用「并联
四杆膝、只有 ``knee_up_joint`` 驱动」的机构学，另外在躯干上加了**两条 4-dof
手臂**（``arm_pitch → arm_roll → arm_yaw → elbow``）。**本文件不 import 任何
v03 / v031 的东西**，各代机器人配置完全解耦。

1. **自由度**：22 个关节 = 腿 14（每腿 7）+ 臂 8（每臂 4）。

     驱动 18：hip_pitch / hip_roll / hip_yaw / knee_up / ankle（每腿 5）
              + arm_pitch / arm_roll / arm_yaw / elbow（每臂 4）
     被动  4：knee_joint / knee_mid_joint（每腿各 1，四杆闭合约束）

   ``knee`` / ``knee_mid`` 既没有 actuator，也不该进 pose 正则或超规格统计
   （它们的目标值由四杆约束决定，不由策略决定）。

2. **左右轴约定与 v031 又不同**（实测 ``m.jnt_axis``）：

       关节         右轴                     左轴
       hip_pitch    [0,-1, 0]                [0, 1, 0]       反号
       hip_roll     [1, 0, 0]                [1, 0, 0]       同号
       hip_yaw      [-0.36315, 0, 0.931731]  两侧相同（**斜轴**）
       knee_up      [0,-1, 0]                [0,-1, 0]       同号
       knee_mid     [0,-1, 0]                [0,-1, 0]       同号
       knee         [0,-1, 0]                [0, 1, 0]       反号
       ankle        [0, 1, 0]                [0,-1, 0]       反号
       arm_pitch    [0, 1, 0]                [0,-1, 0]       反号
       arm_roll     [1, 0, 0]                [1, 0, 0]       同号
       arm_yaw      [0,-1, 0]                [0, 1, 0]       反号
       elbow        [1, 0, 0]                [1, 0, 0]       同号

   写跨机器人通用代码时按实测轴取，不要沿用 v03 / v031 的符号表。

3. **腿部初始姿态为 CAD 零位**。以下为原始全零位几何测量：

     - 两只脚的鞋底世界 z = **-0.25941 m**，脚掌天然水平
     - 质心 x = +0.04982 m，脚心 x = +0.03307 m（质心在脚心前 1.68 cm，
       仍在足底支撑多边形内，但比 v031 略靠前）
     - 四杆闭合 site 间距 **1.001 mm**（Constraint 自洽，reset 不会被猛拽）
     - 脚底 AABB：x ±0.0465、y -0.0354/+0.0356、鞋底 z=-0.034

   ``BIPED_KEYFRAME`` 保留腿部零位及 base z=0.259412 m；2026-10-20
   按用户要求改为左 arm_roll=-53°、右=+53°，其余臂关节为 0。

4. **膝关节限位**。源模型 ``knee_up_joint`` 没有 limited/range（URDF 里
   lower=upper=0，等于锁死），由 ``scripts/build_v033_train_xml.py`` 沿用
   v031 的约定补成 ``[-0.65, 1.05]``；``knee_mid_joint`` 保持无限位。
   ⚠️ v033 实测四杆近似 1:1：``knee_up ≈ knee + 1.4°``，**正方向是屈膝**
   （knee_up 0 -> 60° 时，站地时 base 下沉约 4.9 cm）。给四足姿态填
   ``knee_up`` 时，必须同时给 ``knee`` / ``knee_mid`` 填一致的被动角
   （近似 ``knee ≈ knee_up - 1.4°``、``knee_mid ≈ -knee_up``），否则
   reset 瞬间四杆闭合会被猛拽；最稳妥的是用 IK / 数值求解回填。

5. **电机分两类**（取自 URDF ``<limit effort velocity>``，与 XML ctrlrange 一致）：

     hip_pitch / hip_roll / hip_yaw / knee_up / ankle / arm_pitch / arm_roll
                                                DM-J4310P  12.5 N·m / 12.57 rad/s
     arm_yaw / elbow                            DM-J3507   3.0 N·m / 15.71 rad/s

   注意 v033 的 **ankle 也是 12.5 N·m 档**（v031 的 ankle 是 3.0 N·m）。

6. **整机质量 4.702736 kg**（base_link 0.764587 kg）。

7. **两个 keyframe**（用户要求，四足姿态留待之后定义）：

     - ``BIPED_KEYFRAME``：双足站姿，腿部零位、双臂 roll=∓53°，base z=0.259412；
     - ``QUAD_KEYFRAME`` ：四足姿态**占位**，当前仍为全零位，各关节角度
       列成显式字典，直接改值即可（同时改 XML 里的 ``<key name="quad">``）。

   训练时真正生效的是本文件里的 ``*_KEYFRAME``；XML 里的 keyframe 只供
   MuJoCo viewer 预览，两边要同步。

8. **脚底是 5 条胶囊**，名字 ``{left,right}_foot1..5_collision``，由
   ``build_v033_train_xml.py`` 按 ankle mesh 的实测 AABB 生成，胶囊底面与
   mesh 鞋底精确同高。默认站姿左右脚心间距 **16.90 cm**。
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

PIKACHU_V033_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "pikachu"
  / "xmls"
  / "pikachu_v033"
  / "mjcf"
  # 训练模型由 scripts/build_v033_train_xml.py 从同目录的 pikachu_v033.xml
  # 生成：删 scene include / 删 <size> / 删 <actuator> 块 / 补 <default> /
  # 补 knee_up 限位 / 加 freejoint + imu site / 命名 geom / 加足部胶囊与 site /
  # 补 <sensor> / 保留 stand+quad 两个 keyframe。与源模型同目录，
  # meshdir 仍是 ../meshes/，不复制 STL。
  / "pikachu_v033_train.xml"
)

assert PIKACHU_V033_XML.exists(), (
  f"缺少训练模型: {PIKACHU_V033_XML}\n"
  "请先运行 `uv run python scripts/build_v033_train_xml.py` 生成。"
)


def get_spec() -> mujoco.MjSpec:
  """从 MJCF 文件编译出 MuJoCo 规格（每次调用返回新实例）。"""
  return mujoco.MjSpec.from_file(str(PIKACHU_V033_XML))


##
# Actuator config.
##

# 转子惯量与减速比沿用同一批 DM 电机（与 v03 / v031 物理上是同一颗电机）。
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

# effort / velocity 直接取 URDF 的 <limit>，也等于 XML 的 ctrlrange。
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

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_J4310 = ARMATURE_J4310 * NATURAL_FREQ**2
STIFFNESS_J3507 = ARMATURE_J3507 * NATURAL_FREQ**2

DAMPING_J4310 = 2.0 * DAMPING_RATIO * ARMATURE_J4310 * NATURAL_FREQ
DAMPING_J3507 = 2.0 * DAMPING_RATIO * ARMATURE_J3507 * NATURAL_FREQ

# 大扭矩档：髋三轴 + 膝驱动杆 + 踝 + 臂 pitch/roll。
# ⚠️ 匹配的是 ``knee_up_joint`` 而**不是** ``knee_joint`` —— v033 的膝同样
# 只有 knee_up 是驱动关节，knee / knee_mid 是被动关节，给它们加 actuator
# 会破坏四杆约束（两个位置源抢同一个机构）。
PIKACHU_V033_ACTUATOR_J4310 = BuiltinPositionActuatorCfg(
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

# 小扭矩档：臂 yaw + 肘（DM-J3507，3.0 N·m）。
PIKACHU_V033_ACTUATOR_J3507 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_arm_yaw_joint",
    ".*_elbow_joint",
  ),
  stiffness=STIFFNESS_J3507,
  damping=DAMPING_J3507,
  effort_limit=ACTUATOR_J3507.effort_limit,
  armature=ACTUATOR_J3507.reflected_inertia,
)


##
# Keyframe config.
##

# 两个 keyframe 槽位（用户要求）：
#   1) BIPED_KEYFRAME —— 双足站姿，直接可用；
#   2) QUAD_KEYFRAME  —— 四足姿态占位，等之后填各关节角度。
# mjlab 训练真正读取的是这两个 Python 对象；XML 里的 <key name="stand"/"quad">
# 只用于 viewer 预览，修改时两边同步。

# 2026-10-20: 用户指定的直立手臂初态；腿部仍保持 CAD 零位。
# Entity 对未列出的关节使用 0，避免 ".*" 与 arm_roll 名称重叠匹配。
# 活动范围在独立 Biped E1 奖励中定义，不在 keyframe 中裁剪关节。
BIPED_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.259412),
  joint_pos={
    "(left|right)_arm_pitch_joint": 0.0,
    "left_arm_roll_joint": math.radians(-53),
    "right_arm_roll_joint": math.radians(53),
    "(left|right)_arm_yaw_joint": 0.0,
    "(left|right)_elbow_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


# 兼容旧命名（ZERO_KEYFRAME 现在也引用用户指定的双足站姿）。
ZERO_KEYFRAME = BIPED_KEYFRAME

# 四足姿态占位：所有关节显式列出，方便直接改值。
# TODO(四足姿态): 按实际四足姿态填写。注意：
#   - base 通常需要前俯（改上面的 ``rot``，wxyz 四元数）并重新配平 ``pos.z``；
#   - v033 的 knee_up 正方向是屈膝，且四杆近似 1:1：
#       knee ≈ knee_up - 1.4°、knee_mid ≈ -knee_up（左右腿镜像取反）
#     只改 knee_up 而不同步这两个被动角，会让四杆闭合在 reset 瞬间被猛拽；
#   - 手臂前伸当"前腿"时，arm_pitch / arm_roll / elbow 决定手掌落点，
#     arm_yaw 一般固定 0。
# 当前全 0，仅作占位；独立 Crawl 任务使用自己的 robot.py。
QUAD_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.259412),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    # 左腿
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_up_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_knee_mid_joint": 0.0,
    "left_ankle_joint": 0.0,
    # 右腿
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_up_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_knee_mid_joint": 0.0,
    "right_ankle_joint": 0.0,
    # 左臂（四足时当前腿用）
    "left_arm_pitch_joint": 0.0,
    "left_arm_roll_joint": 0.0,
    "left_arm_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    # 右臂（四足时当前腿用）
    "right_arm_pitch_joint": 0.0,
    "right_arm_roll_joint": 0.0,
    "right_arm_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# v033 的脚底也是 5 条胶囊（build_v033_train_xml.py 按 ankle mesh AABB 生成）。
PIKACHU_V033_FOOT_GEOM_NAMES = r"^(left|right)_foot[1-5]_collision$"

# 四足时前腿（手臂）的支撑点是 elbow_link 的原始 mesh 碰撞体。
PIKACHU_V033_HAND_GEOM_NAMES = r"^(left|right)_elbow_link_collision$"

# 仅脚底碰撞：双足平地行走训练用。CollisionCfg 的 disable_other_geoms=True
# 会把未匹配 geom 的 contype/conaffinity 清零，所以源模型里那些高面数
# mesh 碰撞体不会参与接触，不拖慢仿真。
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(PIKACHU_V033_FOOT_GEOM_NAMES,),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

# 四足支撑（脚掌 + 手肘）：留给之后定义好四足 keyframe 的爬行/转换任务。
# 只开这两组高摩擦支撑面，其余身体 geom 不参与接触（平地用）。
QUAD_SUPPORT_COLLISION = CollisionCfg(
  geom_names_expr=(
    PIKACHU_V033_FOOT_GEOM_NAMES,
    PIKACHU_V033_HAND_GEOM_NAMES,
  ),
  contype=1,
  conaffinity=1,
  condim=3,
  priority={PIKACHU_V033_FOOT_GEOM_NAMES: 1, PIKACHU_V033_HAND_GEOM_NAMES: 1},
  friction={
    PIKACHU_V033_FOOT_GEOM_NAMES: (0.6,),
    PIKACHU_V033_HAND_GEOM_NAMES: (0.6,),
  },
)

# 全身碰撞（不含自碰撞）：留给以后的崎岖地形任务。
# ⚠️ 碰撞 geom 是**原始 mesh**，面数很高，开启前建议先换成胶囊。
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={PIKACHU_V033_FOOT_GEOM_NAMES: 3, ".*_collision": 1},
  priority={PIKACHU_V033_FOOT_GEOM_NAMES: 1, ".*": 0},
  friction={PIKACHU_V033_FOOT_GEOM_NAMES: (0.6,)},
)


##
# Final config.
##

PIKACHU_V033_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PIKACHU_V033_ACTUATOR_J4310,
    PIKACHU_V033_ACTUATOR_J3507,
  ),
  soft_joint_pos_limit_factor=1.0,
)


def get_pikachu_v033_robot_cfg() -> EntityCfg:
  """双足站姿配置（脚底碰撞，用于双足行走/过渡任务）。"""
  return EntityCfg(
    init_state=BIPED_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=PIKACHU_V033_ARTICULATION,
  )


def get_pikachu_v033_quad_robot_cfg() -> EntityCfg:
  """四足配置（``QUAD_KEYFRAME`` + 脚掌/手肘支撑碰撞）。

  ⚠️ ``QUAD_KEYFRAME`` 目前还是双足站姿占位，填好四足姿态后此函数才真正可用。
  """
  return EntityCfg(
    init_state=QUAD_KEYFRAME,
    collisions=(QUAD_SUPPORT_COLLISION,),
    spec_fn=get_spec,
    articulation=PIKACHU_V033_ARTICULATION,
  )


##
# 关节表 / 电机分组（供 telemetry metrics 使用）。
##

# 只有这 10 个腿关节被驱动。**不含** knee_joint / knee_mid_joint ——
# 它们是被动关节，既没有 actuator，也不该进 pose 正则或超规格统计
# （它们的"目标值"由四杆约束决定，不由策略决定）。
PIKACHU_V033_LEG_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_up_joint",
  "left_ankle_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_up_joint",
  "right_ankle_joint",
)

# 8 个手臂关节（全部被驱动）。
PIKACHU_V033_ARM_JOINT_NAMES = (
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "left_arm_yaw_joint",
  "left_elbow_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
  "right_arm_yaw_joint",
  "right_elbow_joint",
)

PIKACHU_V033_JOINT_NAMES = PIKACHU_V033_LEG_JOINT_NAMES + PIKACHU_V033_ARM_JOINT_NAMES

# 关节名正则 -> ElectricActuator（决定 joint_torque_ratio 等指标的分母）。
PIKACHU_V033_ACTUATOR_GROUPS: tuple[tuple[str, ElectricActuator], ...] = (
  (
    r".*(hip_pitch|hip_roll|hip_yaw|knee_up|ankle|arm_pitch|arm_roll)_joint",
    ACTUATOR_J4310,
  ),
  (r".*(arm_yaw|elbow)_joint", ACTUATOR_J3507),
)

# pose 奖励的关节正则：与 PIKACHU_V033_JOINT_NAMES 一一对应（18 个驱动关节）。
PIKACHU_V033_POSE_JOINT_NAMES = (
  r".*(hip_pitch|hip_roll|hip_yaw|knee_up|ankle)_joint",
  r".*(arm_pitch|arm_roll|arm_yaw|elbow)_joint",
)


# 动作缩放系数：``0.25 * effort_limit / stiffness``（与 v03 / v031 / V025 同式）。
PIKACHU_V033_ACTION_SCALE: dict[str, float] = {}
for _a in PIKACHU_V033_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e, _s = _a.effort_limit, _a.stiffness
  assert _e is not None
  for _n in _a.target_names_expr:
    PIKACHU_V033_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  viewer.launch(Entity(get_pikachu_v033_robot_cfg()).spec.compile())
