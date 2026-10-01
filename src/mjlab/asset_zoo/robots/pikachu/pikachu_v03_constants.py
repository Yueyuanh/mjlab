"""Pikachu v03（并联四杆膝）常量。

与 ``pikachu_constants.py``（V025 串联膝 14dof）的关键差别：

1. **膝是并联四杆**。只有 ``knee_up_joint`` 是驱动关节；``knee_joint`` /
   ``knee_mid_joint`` 是被动关节，由 ``<equality><connect>`` 约束跟随。
   实测传动关系（``scripts/probe_v03_stance.py``，关重力标定）：
     - ``knee_mid = -knee_up``（精确成立）
     - ``knee/knee_up`` 在 0.45~1.16 之间随角度变化，零位有 +1.66° 偏置
   所以 **keyframe 里必须把 knee / knee_mid 也写出来**，且要与 knee_up 自洽，
   否则 reset 瞬间约束会把被动关节猛拽一下。

2. **左腿不是整体镜像** —— 这是最容易踩的坑。实测 ``jnt_axis``：

     关节         右轴              左轴              左右
     hip_pitch    [0, 1, 0]         [0, 1, 0]         同向
     hip_roll     [1, 0, 0]         [1, 0, 0]         同向
     hip_yaw      [-.571, 0, .821]  [-.571, 0, .821]  同向
     knee         [0, -1, 0]        [0, -1, 0]        同向
     ankle        [0, -1, 0]        [0,  1, 0]        **反号**
     knee_up      [0, -1, 0]        [0,  1, 0]        **反号**
     knee_mid     [0, -1, 0]        [0,  1, 0]        **反号**

   即髋三轴与膝轴左右**同号**（正方向在世界系里指向同一侧），只有
   踝 / knee_up / knee_mid 因为 CAD 里对应 link 被镜像过而反号。
   ``probe_v03_both_legs.py`` 实测比值证实：``knee_L/knee_R = +1.0000``、
   ``knee_mid_L/knee_mid_R = -1.0000``、``ankle_L/ankle_R = -1.0000``。
   关节限位同样只对这三个镜像：右 knee_up ``[-1.22, 0.52]`` ↔ 左 ``[-0.52, 1.22]``。

3. **站姿不是 qpos=0**。qpos=0 时踝在 base 原点后方 10.7 cm（腿后倾 27°），
   脚掌虽然水平但撑在身后，机器人会向后倒。真正的站立姿态要把脚掌摆到
   髋正下方 —— 见 ``STAND_KEYFRAME`` / ``KNEES_BENT_KEYFRAME``（由
   ``probe_v03_both_legs.py`` 双腿独立解出：脚掌俯仰 0°、脚中心 x=0）。
   高度在训练模型（带 freejoint）里复核过：直立 0.27187 m、微屈膝 0.26879 m。
   该姿态下质心在 x=-0.0195 m，落在脚掌支撑面 x∈[-0.052, +0.052] 内，静稳定。

4. **脚底是 5 条胶囊**（不是 V025 的 7 条脚趾胶囊），名字
   ``{left,right}_foot1..5_collision``，由 ``scripts/build_v03_train_xml.py``
   按 ankle mesh 的实测 AABB 生成，胶囊底面与 mesh 鞋底精确同高。

XML 由 ``scripts/build_v03_train_xml.py`` 从
``pikachu_urdf/pikachu_v03_base_series`` 生成（加 freejoint / 命名 geom /
加足部 site / 删 scene include / 删 actuator 块）。
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

PIKACHU_V03_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "pikachu"
  / "xmls"
  / "mjcf_v03"
  / "pikachu_v03_base_series_train.xml"
)
assert PIKACHU_V03_XML.exists(), f"缺少训练模型: {PIKACHU_V03_XML}"


def get_spec() -> mujoco.MjSpec:
  """从 MJCF 文件编译出 MuJoCo 规格（每次调用返回新实例）。"""
  return mujoco.MjSpec.from_file(str(PIKACHU_V03_XML))


##
# Actuator config.
##

# 电机规格与 pikachu_constants.py 相同（同一批 DM 电机），
# 但 effort_limit 用 v03 MJCF 里实际写入的力矩上限：
#   hip_pitch / knee_up -> DM-J4310P-2EC-V1.1  ±12.5 N·m
#   hip_roll / hip_yaw / ankle -> DM-J3507-2EC-3R  ±9 N·m
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
  velocity_limit=20.0,
  effort_limit=12.5,  # v03 MJCF ctrlrange ±12.5
)
ACTUATOR_J3507 = ElectricActuator(
  reflected_inertia=ARMATURE_J3507,
  velocity_limit=30.0 / 3,
  effort_limit=9.0,  # v03 MJCF ctrlrange ±9
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_J4310 = ARMATURE_J4310 * NATURAL_FREQ**2  # 8.709
STIFFNESS_J3507 = ARMATURE_J3507 * NATURAL_FREQ**2  # 12.187

DAMPING_J4310 = 2.0 * DAMPING_RATIO * ARMATURE_J4310 * NATURAL_FREQ  # 0.554
DAMPING_J3507 = 2.0 * DAMPING_RATIO * ARMATURE_J3507 * NATURAL_FREQ  # 0.776

# 大扭矩：髋 pitch + 膝。
# ⚠️ 这里匹配的是 ``knee_up_joint`` 而**不是** ``knee_joint`` —— v03 的膝
# 只有 knee_up 是驱动的，knee/knee_mid 是被动关节，给它们加 actuator
# 会破坏四杆约束（两个位置源抢同一个机构）。
PIKACHU_V03_ACTUATOR_J4310 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_knee_up_joint",
  ),
  stiffness=STIFFNESS_J4310,
  damping=DAMPING_J4310,
  effort_limit=ACTUATOR_J4310.effort_limit,
  armature=ACTUATOR_J4310.reflected_inertia,
)

# 小扭矩：髋 roll / 髋 yaw / 踝。
PIKACHU_V03_ACTUATOR_J3507 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_ankle_joint",
  ),
  stiffness=STIFFNESS_J3507,
  damping=DAMPING_J3507,
  effort_limit=ACTUATOR_J3507.effort_limit,
  armature=ACTUATOR_J3507.reflected_inertia,
)


##
# Keyframe config.
##

# 直立站姿（由 scripts/probe_v03_both_legs.py 双腿独立解出）：
#   脚掌俯仰 0.000°、脚中心 x = 0（脚掌在髋正下方）。
#   base 高 0.27187 m 是在**训练模型**里复核的（freejoint 版，权威值）。
# 被动关节 knee / knee_mid 的值也一并写出，与 knee_up 自洽。
STAND_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.27187),
  joint_pos={
    "right_hip_pitch_joint": -0.4822,
    "left_hip_pitch_joint": -0.4802,
    "right_hip_roll_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    # knee_up 左腿反号
    "right_knee_up_joint": 0.0,
    "left_knee_up_joint": 0.0,
    # ankle 左腿反号
    "right_ankle_joint": -0.5074,
    "left_ankle_joint": 0.5048,
    # 被动关节：knee 左右同号，knee_mid 左腿反号
    "right_knee_joint": 0.0219,
    "left_knee_joint": 0.0219,
    "right_knee_mid_joint": 0.0062,
    "left_knee_mid_joint": -0.0062,
  },
  joint_vel={".*": 0.0},
)

# 微屈膝站姿 + **静平衡**：knee_up = -0.20（膝屈 ~10.3°）。
#
# 这是训练的 init_state。它在 KNEES_BENT 的「脚心 x=0」解之上又做了一步
# 固定点迭代（``scripts/refine_v03_keyframe.py``），把脚心挪到**质心正下方**：
#
#     iter  cx_target    CoM_x      base_z
#       0    0.00000   -0.01744   0.26904
#       4   -0.02651   -0.02678   0.26788
#       7   -0.02690   -0.02691   0.26784   <- 收敛
#
# 为什么必须这么做：「脚心 x=0」只是把脚摆在 **base 原点**下方，而 v03 的
# 质心在 base 原点后方 2.7 cm。重力矩于是持续把机器人往后压 —— 零动作
# 开环测试里它先向后仰到 pitch=-0.167 rad、base 掉到 0.2248 m，再被踝关节
# PD 的弹性力矩拉回来，停在**非零**平衡点上。那不是静平衡：真正的静平衡
# 要求质心水平投影落在支撑中心，此时踝关节 PD 误差为 0，零动作就是不动点，
# episode 开局没有塌陷瞬态。
#
# 微屈膝本身也有理由：四杆膝在 knee_up≈0 附近传动比最低（0.45~1.17 的下沿），
# 微屈膝避开那个区段，也符合「走路时膝盖不锁死」的常识。
# 高度 0.26728 是在**训练模型**（带 freejoint）里复核的权威值；
# 源模型（base 焊死）给出 0.26784，差 0.56 mm 来自基座朝向的自由度。
KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.26728),
  joint_pos={
    "right_hip_pitch_joint": -0.4272,
    "left_hip_pitch_joint": -0.4252,
    "right_hip_roll_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_knee_up_joint": -0.2000,
    "left_knee_up_joint": 0.2000,  # 左腿 knee_up 反号
    "right_ankle_joint": -0.2747,
    "left_ankle_joint": 0.2727,  # 左腿 ankle 反号
    "right_knee_joint": -0.1775,
    "left_knee_joint": -0.1775,  # 左腿 knee 同号
    "right_knee_mid_joint": 0.2056,
    "left_knee_mid_joint": -0.2056,  # 左腿 knee_mid 反号
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# v03 的脚底是 5 条胶囊（build_v03_train_xml.py 按 ankle mesh AABB 生成）。
PIKACHU_V03_FOOT_GEOM_NAMES = r"^(left|right)_foot[1-5]_collision$"

# 仅脚底碰撞：平地行走训练用。CollisionCfg 的 disable_other_geoms=True
# 会把未匹配 geom 的 contype/conaffinity 清零，所以源模型里那些
# 高面数 mesh 碰撞体（base_link.STL 近 10 MB）不会参与接触，不拖慢仿真。
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(PIKACHU_V03_FOOT_GEOM_NAMES,),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

# 全身碰撞（不含自碰撞）：留给以后的崎岖地形任务。
# ⚠️ v03 的碰撞 geom 是**原始 mesh**，面数很高，开启前建议先换成胶囊。
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={PIKACHU_V03_FOOT_GEOM_NAMES: 3, ".*_collision": 1},
  priority={PIKACHU_V03_FOOT_GEOM_NAMES: 1, ".*": 0},
  friction={PIKACHU_V03_FOOT_GEOM_NAMES: (0.6,)},
)


##
# Final config.
##

PIKACHU_V03_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PIKACHU_V03_ACTUATOR_J4310,
    PIKACHU_V03_ACTUATOR_J3507,
  ),
  soft_joint_pos_limit_factor=1.0,
)


def get_pikachu_v03_robot_cfg() -> EntityCfg:
  """获取一个新的 Pikachu v03 机器人配置实例。"""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=PIKACHU_V03_ARTICULATION,
  )


##
# 关节表 / 电机分组（供 telemetry metrics 使用）。
##

# 只有这 10 个关节被驱动。**不含** knee_joint / knee_mid_joint ——
# 它们是被动关节，既没有 actuator，也不该进 pose 正则或超规格统计
# （它们的"目标值"由四杆约束决定，不由策略决定）。
PIKACHU_V03_JOINT_NAMES = (
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
# 注意是 ``knee_up`` 而不是 ``knee``：v03 的膝只有 knee_up 是驱动关节。
PIKACHU_V03_ACTUATOR_GROUPS: tuple[tuple[str, ElectricActuator], ...] = (
  (r".*(hip_pitch|knee_up)_joint", ACTUATOR_J4310),
  (r".*(hip_roll|hip_yaw|ankle)_joint", ACTUATOR_J3507),
)

# pose 奖励的关节正则：与 PIKACHU_V03_JOINT_NAMES 一一对应。
PIKACHU_V03_POSE_JOINT_NAMES = (r".*(hip_pitch|hip_roll|hip_yaw|knee_up|ankle)_joint",)


# 动作缩放系数：``0.25 * effort_limit / stiffness``（与 pikachu_constants.py 同式）。
PIKACHU_V03_ACTION_SCALE: dict[str, float] = {}
for _a in PIKACHU_V03_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e, _s = _a.effort_limit, _a.stiffness
  assert _e is not None
  for _n in _a.target_names_expr:
    PIKACHU_V03_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  viewer.launch(Entity(get_pikachu_v03_robot_cfg()).spec.compile())
