"""Pikachu v031（并联四杆膝）常量。

v031 是 v03 的**新一代整机**（SolidWorks -> easymjcf 直出），沿用「并联四杆膝、
只有 ``knee_up_joint`` 驱动」的机构学，但左右轴约定、质量、限位、电机分组全都
不同。**本文件不 import 任何 v03 的东西**，两个任务完全解耦。

1. **左右轴约定与 v03 相反**（实测 ``m.jnt_axis``）：

     关节         右轴              左轴              左右
     hip_pitch    [0, 1, 0]         [0, -1, 0]       反号
     hip_roll     [1, 0, 0]         [1, 0, 0]        反号（body 系镜像）
     hip_yaw      [0, 0, 1]         [0, 0, 1]        反号
     knee         [0, 1, 0]         [0, -1, 0]       反号
     ankle        [0, -1, 0]        [0, 1, 0]        反号
     knee_up      [0, -1, 0]        [0, -1, 0]       **同号**
     knee_mid     [0, -1, 0]        [0, -1, 0]       **同号**

   v03 是「髋三轴 + 膝同号，踝 / knee_up / knee_mid 反号」；v031 把前四项整个
   翻了过来，只留 knee_up / knee_mid 同号。**写跨机器人通用代码时按实测轴取，
   不要沿用 v03 的符号表。**

2. **初始姿态就是 qpos 全 0**。v03 的 qpos=0 是「腿后倾 27°、脚撑在身后」，
   必须标定站姿；v031 的 CAD 零位**本身就是站姿**，实测：

     - 两只脚的鞋底世界 z = **-0.25941 m**，脚掌俯仰 **-0.000°**（天然水平）
     - 脚心 x = +0.0331 m，质心 x = +0.0294 m —— 质心几乎正在脚心正上方
     - 四杆闭合 site 间距只有 **1.000 mm**（约束自洽，reset 不会被猛拽）
     - 零动作 3 s 静稳定测试（带地面、PD 压住 keyframe）：
       base z 0.25941 -> 0.25908、水平漂移 1.5 mm、roll 0.00°、
       pitch +2.09° 并**收敛**（不发散）

   所以 keyframe 不需要任何标定：全 0，base 抬到 0.25941 m 即可。
   ``scripts/probe_v031_keyframe.py`` 的双腿独立求解保留下来做姿态体检，
   但本文件的 keyframe 不用它的输出。

3. **膝关节限位**。源模型两个 ``knee_up_joint`` 都没有 ``limited``/``range``，
   由 ``scripts/build_v031_train_xml.py`` 按 URDF 补成 ``[-0.65, 1.05]``；
   ``knee_mid_joint`` 保持无限位（纯被动杆，行程由四杆闭合约束决定）。
   ⚠️ v031 的 ``knee_up`` **正方向是屈膝**（0 -> 0.6 时 base 从 0.2626 降到
   0.2352 m），与 v03 的正方向相反。

4. **源模型没有 ``<default>`` 块**，joint 的 damping / frictionloss /
   solreflimit 全是 MuJoCo 默认值（前两个 0）。``build_v031_train_xml.py``
   补齐为 ``damping=0.3 frictionloss=0.01 solreflimit="0.004 1"`` —— 缺了
   耗散，四杆被动杆在 PD 之外毫无阻尼；缺了 solreflimit，关节限位是软约束，
   小腿杆件惯量只有 1e-5 量级，满扭矩能直接顶穿。

5. **电机分两类**（取自 URDF ``<limit effort velocity>``，与 XML ctrlrange 一致）：

     hip_pitch / hip_roll / hip_yaw / knee_up   DM-J4310P  12.5 N·m / 12.57 rad/s
     ankle                                      DM-J3507    3.0 N·m / 15.71 rad/s

   ⚠️ v031 的 **hip_roll / hip_yaw 也在 12.5 N·m 档**（v03 是 3.0 N·m 降额），
   T6 的 ``joint_torque_soft_limit`` 分组必须跟着改，否则 80% 软限位
   （10 N·m）在 3 N·m 的执行器上永远达不到、该项恒为 0。

6. **整机质量 6.759964 kg**（base_link 4.623428 kg），已接近 v03 的
   「weight 6.3 kg」压舱版（6.38 kg），**不需要再加配重**。

7. **脚底是 5 条胶囊**，名字 ``{left,right}_foot1..5_collision``，由
   ``build_v031_train_xml.py`` 按 ankle mesh 的实测 AABB 生成，胶囊底面与
   mesh 鞋底精确同高。默认站姿左右脚心间距 **17.30 cm**。
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

PIKACHU_V031_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "pikachu"
  / "xmls"
  / "pikachu_v031"
  / "mjcf"
  # 训练模型由 scripts/build_v031_train_xml.py 从同目录的 pikachu_v031.xml
  # 生成：删 scene include / 删 <size> / 删 <actuator> 块 / 补 <default> /
  # 补 knee_up 限位 / 加 freejoint + imu site / 命名 geom / 加足部胶囊与 site /
  # 补 <sensor>。与源模型同目录，meshdir 仍是 ../meshes/，不复制 STL。
  / "pikachu_v031_train.xml"
)
assert PIKACHU_V031_XML.exists(), (
  f"缺少训练模型: {PIKACHU_V031_XML}\n"
  "请先运行 `uv run python scripts/build_v031_train_xml.py` 生成。"
)


def get_spec() -> mujoco.MjSpec:
  """从 MJCF 文件编译出 MuJoCo 规格（每次调用返回新实例）。"""
  return mujoco.MjSpec.from_file(str(PIKACHU_V031_XML))


##
# Actuator config.
##

# 转子惯量与减速比沿用同一批 DM 电机（与 v03 物理上是同一颗电机）。
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

# 大扭矩档：髋三轴 + 膝驱动杆。
# ⚠️ 匹配的是 ``knee_up_joint`` 而**不是** ``knee_joint`` —— v031 的膝同样
# 只有 knee_up 是驱动关节，knee / knee_mid 是被动关节，给它们加 actuator
# 会破坏四杆约束（两个位置源抢同一个机构）。
PIKACHU_V031_ACTUATOR_J4310 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_up_joint",
  ),
  stiffness=STIFFNESS_J4310,
  damping=DAMPING_J4310,
  effort_limit=ACTUATOR_J4310.effort_limit,
  armature=ACTUATOR_J4310.reflected_inertia,
)

# 小扭矩档：踝。
PIKACHU_V031_ACTUATOR_J3507 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_joint",),
  stiffness=STIFFNESS_J3507,
  damping=DAMPING_J3507,
  effort_limit=ACTUATOR_J3507.effort_limit,
  armature=ACTUATOR_J3507.reflected_inertia,
)


##
# Keyframe config.
##

# CAD 零位即站姿：全 0，base 抬到鞋底刚好贴地的高度。
#
# 这里**故意用正则** ``{".*": 0.0}`` 而不是逐个列出 14 个关节：被动的
# ``knee_joint`` / ``knee_mid_joint`` 不是策略能控制的自由度，把它们写进
# keyframe 会让人误以为它们是需要标定的量。全 0 时四杆闭合约束的违背量
# 只有 1.000 mm，reset 瞬间几乎无冲击。
ZERO_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.25941),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# v031 的脚底也是 5 条胶囊（build_v031_train_xml.py 按 ankle mesh AABB 生成）。
PIKACHU_V031_FOOT_GEOM_NAMES = r"^(left|right)_foot[1-5]_collision$"

# 仅脚底碰撞：平地行走训练用。CollisionCfg 的 disable_other_geoms=True
# 会把未匹配 geom 的 contype/conaffinity 清零，所以源模型里那些高面数
# mesh 碰撞体不会参与接触，不拖慢仿真。
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(PIKACHU_V031_FOOT_GEOM_NAMES,),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

# 全身碰撞（不含自碰撞）：留给以后的崎岖地形任务。
# ⚠️ 碰撞 geom 是**原始 mesh**，面数很高，开启前建议先换成胶囊。
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={PIKACHU_V031_FOOT_GEOM_NAMES: 3, ".*_collision": 1},
  priority={PIKACHU_V031_FOOT_GEOM_NAMES: 1, ".*": 0},
  friction={PIKACHU_V031_FOOT_GEOM_NAMES: (0.6,)},
)


##
# Final config.
##

PIKACHU_V031_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PIKACHU_V031_ACTUATOR_J4310,
    PIKACHU_V031_ACTUATOR_J3507,
  ),
  soft_joint_pos_limit_factor=1.0,
)


def get_pikachu_v031_robot_cfg() -> EntityCfg:
  """获取一个新的 Pikachu v031 机器人配置实例。"""
  return EntityCfg(
    init_state=ZERO_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=PIKACHU_V031_ARTICULATION,
  )


##
# 关节表 / 电机分组（供 telemetry metrics 使用）。
##

# 只有这 10 个关节被驱动。**不含** knee_joint / knee_mid_joint ——
# 它们是被动关节，既没有 actuator，也不该进 pose 正则或超规格统计
# （它们的"目标值"由四杆约束决定，不由策略决定）。
PIKACHU_V031_JOINT_NAMES = (
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

# 关节名正则 -> ElectricActuator（决定 joint_torque_ratio 等指标的分母）。
# ⚠️ 分组与 v03 **不同**：v031 的 hip_roll / hip_yaw 是 12.5 N·m 档。
PIKACHU_V031_ACTUATOR_GROUPS: tuple[tuple[str, ElectricActuator], ...] = (
  (r".*(hip_pitch|hip_roll|hip_yaw|knee_up)_joint", ACTUATOR_J4310),
  (r".*_ankle_joint", ACTUATOR_J3507),
)

# pose 奖励的关节正则：与 PIKACHU_V031_JOINT_NAMES 一一对应。
PIKACHU_V031_POSE_JOINT_NAMES = (r".*(hip_pitch|hip_roll|hip_yaw|knee_up|ankle)_joint",)


# 动作缩放系数：``0.25 * effort_limit / stiffness``（与 v03 / V025 同式）。
PIKACHU_V031_ACTION_SCALE: dict[str, float] = {}
for _a in PIKACHU_V031_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e, _s = _a.effort_limit, _a.stiffness
  assert _e is not None
  for _n in _a.target_names_expr:
    PIKACHU_V031_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  viewer.launch(Entity(get_pikachu_v031_robot_cfg()).spec.compile())
