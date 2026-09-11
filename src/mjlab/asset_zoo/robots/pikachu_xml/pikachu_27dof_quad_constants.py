"""Pikachu **27-dof 四足（quadruped）** 专用常量。

独立于双足 ``pikachu_27dof_constants.py``：这里**只 import 其只读值引用**（关节分组、
执行器、XML/spec、脚底碰撞队形），但**起步姿态/碰撞/动作缩放用四足专属版** ——
不碰双足任务。

四足构型：前腿 = 手臂（27dof 臂链 `arm_pitch → arm_roll → arm_yaw → elbow`，
终端手掌 = ``elbow_link``），后腿 = 腿 10。base 前俯 ≈107°（``QUAD_27DOF_ROT``），
手脚四点撑地爬行。**策略动作驱动 16 个走步关节**（腿 10 + 臂 pitch/roll/elbow 6），
``arm_yaw`` 固定 0；头(3)/耳(4)/尾(2) 固定在关键帧姿态，不参与步态。

起步姿态的落地配平（rot / arm/leg 关节角度 / pos.z）沿用 14dof 四足已有「门禁标定」
哲学：先给初始值，viewer ``--agent zero`` 实测后回填，多开参数标注 TODO。
"""

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_constants import (
  ARM_JOINT_NAMES,
  EXTREMITY_JOINT_NAMES,
  HEAD_JOINT_NAMES,
  LEG_ACTION_SCALE,
  LEG_JOINT_NAMES,
  PIKACHU_27DOF_ARTICULATION,
  PIKACHU_27DOF_FOOT_GEOM_NAMES,
)
from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_constants import (
  get_spec as _get_27dof_spec,
)
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

__all__ = [
  "QUAD_27DOF_ACTION_SCALE",
  "QUAD_27DOF_COLLISION",
  "QUAD_27DOF_HAND_GEOM_NAMES",
  "QUAD_27DOF_KEYFRAME",
  "QUAD_27DOF_MOTION_FILE",
  "QUAD_27DOF_ROT_EULER_DEG",
  "QUAD_27DOF_SUPPORT_GEOM_NAMES",
  "QUAD_27DOF_TILT_DEG",
  "QUAD_27DOF_TORSO_BODY",
  "QUAD_27DOF_WALK_JOINT_NAMES",
  "get_spec",
  "get_pikachu_27dof_quad_robot_cfg",
  "pikachu_27dof_quad_walk_bodies",
]


# 四足躯干(基座)body 名(与双足一致)。
QUAD_27DOF_TORSO_BODY = "base_link"

# 四足起步姿态欧拉角(度):base 前俯 ≈107°(纯绕 +y 旋转,roll/yaw 为 0);
# 107.1889° 正是原硬编码四元数 (0.593497,0,0.804837,0) 的精确反推俯仰角。
# 以欧拉三元组(roll, pitch, yaw)作为唯一真源,``QUAD_27DOF_ROT`` 由它反算出
# 四元数 —— ``EntityCfg.InitialStateCfg.rot`` 字段只收 wxyz 四元数,见 entity.py。
QUAD_27DOF_ROT_EULER_DEG: tuple[float, float, float] = (0.0, 107.1889, 0.0)


def _quat_from_euler_xyz_deg(
  roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
  """欧拉角(度,XYZ extrinsic)转四元数 (w, x, y, z)。内联 ``lab_api.math.quat_from_euler_xyz``，
  避免在纯配置模块里引入 torch。"""
  r, p, y = map(math.radians, (roll, pitch, yaw))
  cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
  cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
  cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
  qw = cy * cr * cp + sy * sr * sp
  qx = cy * sr * cp - sy * cr * sp
  qy = cy * cr * sp + sy * sr * cp
  qz = sy * cr * cp - cy * sr * sp
  return (qw, qx, qy, qz)


# 四足起步姿态:base 前俯 ≈107°(绕 +y 旋转),value 由上面欧拉角反算,数值等价
# 原硬编码四元数 (0.593497, 0, 0.804837, 0),镜像 14dof ``QUAD_KEYFRAME``。
QUAD_27DOF_ROT: tuple[float, float, float, float] = _quat_from_euler_xyz_deg(
  *QUAD_27DOF_ROT_EULER_DEG
)


def get_spec() -> mujoco.MjSpec:
  """编译出 27dof 训练体机器人规格(四足/双足共用同一 XML)。"""
  return _get_27dof_spec()


##
# 四足走步关节集:策略动作驱动的 16 关节。
#
# 前腿 = 臂 pitch/roll/elbow(肘作前腿膝盖),``arm_yaw`` 固定 0(侧向撑开会绕)。
# 顺序 = 腿 10 + 左前腿 3 + 右前腿 3,供 tracking/velocity 共用。
##
_ARM_FRONT_LEG: tuple[str, ...] = (
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "left_elbow_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
  "right_elbow_joint",
)

QUAD_27DOF_WALK_JOINT_NAMES: tuple[str, ...] = LEG_JOINT_NAMES + _ARM_FRONT_LEG

# 固定(pitch/roll 之外)但仍需 PD 保持的关节:arm_yaw + 头 + 耳 + 尾。
QUAD_27DOF_FIXED_JOINT_NAMES: tuple[str, ...] = (
  tuple(j for j in ARM_JOINT_NAMES if j.endswith("_arm_yaw_joint"))
  + HEAD_JOINT_NAMES
  + EXTREMITY_JOINT_NAMES
)


def pikachu_27dof_quad_walk_bodies() -> tuple[str, ...]:
  """四足走步跟踪用的 body 布局(anchor + 四条腿的用户愿意 track 的 link)。"""
  return (
    QUAD_27DOF_TORSO_BODY,
    # 后腿:左/右 hip_pitch→knee→ankle。
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_link",
    # 前腿:左/右 arm_pitch→arm_roll→elbow(手掌)。
    "left_arm_pitch_link",
    "left_arm_roll_link",
    "left_elbow_link",
    "right_arm_pitch_link",
    "right_arm_roll_link",
    "right_elbow_link",
  )


# 四足 27dof 走步参考运动:由 ``scripts/make_27dof_quad_reference.py`` 从 14dof 四足
# mocap CSV(``pikachu_quad.csv``)在 27dof 四足机器人上 FK 重放生成。16 列(walk 序,
# = ``QUAD_27DOF_WALK_JOINT_NAMES``);elbow 录制未驱动,取关键帧默认,见脚本头告警。
QUAD_27DOF_MOTION_FILE: Path = (
  MJLAB_SRC_PATH / "mocap" / "npz" / "pikachu_walk_27dof_npz" / "pikachu_quad.npz"
)

##
# 碰撞配置:四足四点撑地 —— 后脚(ankle)与前手(elbow)均 condim=3,friction 防打滑;
# contype=0 机器人自身 geom 间不自碰撞(手臂贴地不夹腿)。其余身体几何由
# ``.*_collision`` 配 condim=1,保证下蹲/跌倒时肢体不穿地。
##
QUAD_27DOF_HAND_GEOM_NAMES = r"^(left|right)_elbow_link_collision$"

QUAD_27DOF_SUPPORT_GEOM_NAMES: tuple[str, str] = (
  PIKACHU_27DOF_FOOT_GEOM_NAMES,  # 后脚(left/right ankle_collision)
  QUAD_27DOF_HAND_GEOM_NAMES,  # 前手(left/right elbow_link_collision)
)

QUAD_27DOF_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={
    PIKACHU_27DOF_FOOT_GEOM_NAMES: 3,
    QUAD_27DOF_HAND_GEOM_NAMES: 3,
    ".*_collision": 1,
  },
  priority={
    PIKACHU_27DOF_FOOT_GEOM_NAMES: 1,
    QUAD_27DOF_HAND_GEOM_NAMES: 1,
    ".*": 0,
  },
  friction={
    PIKACHU_27DOF_FOOT_GEOM_NAMES: (0.6,),
    QUAD_27DOF_HAND_GEOM_NAMES: (0.6,),
  },
)

##
# 动作缩放(四足):16 个走步关节各放宽,arm_yaw 不在内(固定 0)。
# 腿沿用 LEG_ACTION_SCALE + 前腿放宽。
##
QUAD_27DOF_ACTION_SCALE: dict[str, float] = {
  **LEG_ACTION_SCALE,
  # 前腿:arm_pitch 当"前髋"前后摆、elbow 当前腿膝盖、arm_roll 侧向;均放宽。
  r".*_arm_pitch_joint": 1.0,
  r".*_arm_roll_joint": 0.35,
  r".*_elbow_joint": 0.5,
}

# 四足 target 躯干前俯角(度):``+Z`` 背轴与世界 up 的目标夹角。沿用 14dof 四足
# 110°;27dof 头部偏重(≈11.96kg),settle 角需 viewer 实测回填 —— 标校 TODO。
QUAD_27DOF_TILT_DEG: float = 110.0

# 四足"背轴"在当地机体的方向:机体 +Z(与 quad_mdp 一致)。
QUAD_27DOF_BODY_AXIS: tuple[float, float, float] = (0.0, 0.0, 1.0)

##
# 起步关键帧:base 前俯 ≈107°,手脚四点撑地。值镜像 14dof ``QUAD_KEYFRAME``(腿关节
# 左右反相、arm_pitch ±2.0),补 27dof 新增臂关节(arm_yaw=0)与头/耳/尾(前向下垂)。
# TODO(门禁标定):viewer --agent zero 实测后回填 elbow/arm_pitch/rot/pos.z 配平。
##
QUAD_27DOF_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.16),
  rot=QUAD_27DOF_ROT,
  joint_pos={
    # 腿:四足时腿在 base 后方弯曲,左右取反(镜像 14dof QUAD_KEYFRAME)。
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
    # 前腿:arm_pitch 前探、elbow 屈(手掌下压撑地)、arm_roll 0。
    "left_arm_pitch_joint": -2.0,
    "right_arm_pitch_joint": 2.0,
    "left_arm_roll_joint": 0.0,
    "right_arm_roll_joint": 0.0,
    "left_elbow_joint": 0.0,  # elbow 固定 0°,前腿伸直。
    "right_elbow_joint": 0.0,
    # arm_yaw + 头/耳/尾:固定 0(向前下方,头由 pose 保持)。
    "left_arm_yaw_joint": 0.0,
    "right_arm_yaw_joint": 0.0,
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


def get_pikachu_27dof_quad_robot_cfg() -> EntityCfg:
  """获取 27-dof 四足机器人配置(前俯四点撑地,手脚 condim=3,全关节执行器)。"""
  return EntityCfg(
    init_state=QUAD_27DOF_KEYFRAME,
    collisions=(QUAD_27DOF_COLLISION,),
    spec_fn=get_spec,
    articulation=PIKACHU_27DOF_ARTICULATION,
  )
