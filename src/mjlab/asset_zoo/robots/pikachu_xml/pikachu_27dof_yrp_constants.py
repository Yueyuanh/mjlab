"""Pikachu **27-dof hip-YRP** 训练常量（`pikachu_27dof_train_yrp.xml`）。

独立于双足 ``pikachu_27dof_constants.py``，专为 hip 拆分轴序 **Yaw→Roll→Pitch**
（YRP）的训练模型。YRP 与 PRY（``pikachu_27dof_train.xml``，Pitch→Roll→Yaw）的
body/关节名**完全相同**，区别只在 hip 三段的父子顺序与体素几何位置；同名 body 是
跨模型对齐的唯一锚。

- 关节分组、执行器、关键帧、碰撞、动作缩放等**只读值全部复用**
  ``pikachu_27dof_constants``（名字未变，值不重抄）。
- 仅新增：``PIKACHU_27DOF_YRP_XML``（yyrp 训练体）、``PIKACHU_27DOF_PRY_XML``
  （原 PRY 训练体，供读取 npz 参考布局名序）、``get_yrp_spec``（YRP spec）、
  ``get_body_column_names``（按 XML 读 body 名序）、以及换用 YRP spec 的 robot cfg。

tracking 参考 npz（``pikachu_walk_0.185_27dof.npz`` / ``pikachu_quad.npz``）是在**原
PRY 模型**上生成/重放的，其 ``body_*_w`` 列序 = PRY 模型的 body 先序（跳过 world，
28 个）。YRP 任务消费同一 npz 时，须在运行时按名字把它重排到当前模型 —— 具体由
``MotionCommandCfg.motion_body_column_names`` 触发（见 ``commands.py``），这里只提供
那份 PRY body 名序的来源 ``get_body_column_names(PIKACHU_27DOF_PRY_XML)``。
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_constants import (
  FEET_ONLY_COLLISION,
  LEG_ACTION_SCALE,
  LEG_JOINT_NAMES,
  PIKACHU_27DOF_ARTICULATION,
  PIKACHU_27DOF_FOOT_GEOM_NAMES,
  PIKACHU_27DOF_FULL_COLLISION,
)
from mjlab.entity import EntityCfg

# YRP hip 拆分序训练体（本次重建的目标模型）。
PIKACHU_27DOF_YRP_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "pikachu_xml"
  / "mjcf"
  / "pikachu_27dof_train_yrp.xml"
)
assert PIKACHU_27DOF_YRP_XML.exists()

# 原 PRY 训练体：现有 27dof 参考 npz 的 body 列序（生成/重放时的模型 body 先序）。
PIKACHU_27DOF_PRY_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "pikachu_xml"
  / "mjcf"
  / "pikachu_27dof_train.xml"
)
assert PIKACHU_27DOF_PRY_XML.exists()

# YRP 屈膝站姿（按 YRP 运动学 FK 搜索：脚底水平、CoM 落在支撑面中心附近）。
# 注意：不可复用 PRY 的 +0.52/+1.02/+0.50 —— hip 轴序不同，同角度脚位不对。
# 标定日期: 2026-09-17; hp=-0.50, kn=0.82, an=0.325, base_z=0.180, arm_roll=0.
HOME_27DOF_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.180),  # YRP 屈膝时脚底贴地的 base 高度
  joint_pos={
    # 腿：YRP 屈膝站姿（左右同号）。
    "left_hip_pitch_joint": -0.50,
    "right_hip_pitch_joint": -0.50,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.82,
    "right_knee_joint": 0.82,
    "left_ankle_joint": 0.325,
    "right_ankle_joint": 0.325,
    # 臂：自然放松（target 与 pos 一致，避免 PD 拉臂）。
    "left_arm_pitch_joint": 0.0,
    "right_arm_pitch_joint": 0.0,
    "left_arm_roll_joint": 0.0,
    "right_arm_roll_joint": 0.0,
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


def get_yrp_spec() -> mujoco.MjSpec:
  """编译出 YRP 训练体机器人规格（每次调用返回新实例）。"""
  return mujoco.MjSpec.from_file(str(PIKACHU_27DOF_YRP_XML))


def get_body_column_names(xml: Path) -> tuple[str, ...]:
  """返回 ``xml`` 模型的 body 名序（跳过 world，与 ``entity.body_names`` 同空间）。

  该名序与 mjlab 的 ``body_link_*_w`` 列布局一致，也即在此模型上生成/重放的参考
  npz 的 ``body_*_w`` 列序 —— 供 ``MotionCommandCfg.motion_body_column_names`` 把
  参考列按名字对齐到另一构型（YRP）模型的 ``body_names`` 序。
  """
  spec = mujoco.MjSpec.from_file(str(xml))
  return tuple(b.name.split("/")[-1] for b in spec.bodies[1:])


def get_pikachu_27dof_yrp_robot_cfg() -> EntityCfg:
  """获取 27-dof YRP 双足机器人配置（脚底碰撞，全关节执行器，YRP hip 拆分）。

  与 ``get_pikachu_27dof_robot_cfg`` 同构，仅把 spec 换成 YRP 训练体。keyframe /
  关节分组复用 PRY 版（关节名相同，按名字解析）。
  """
  return EntityCfg(
    init_state=HOME_27DOF_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_yrp_spec,
    articulation=PIKACHU_27DOF_ARTICULATION,
  )


def get_pikachu_27dof_yrp_mocap_robot_cfg() -> EntityCfg:
  """获取 mocap / 关键帧预览用的 27-dof YRP 配置（全身碰撞，换 YRP spec）。"""
  return EntityCfg(
    init_state=HOME_27DOF_KEYFRAME,
    collisions=(PIKACHU_27DOF_FULL_COLLISION,),
    spec_fn=get_yrp_spec,
    articulation=PIKACHU_27DOF_ARTICULATION,
  )


__all__ = [
  "LEG_ACTION_SCALE",
  "LEG_JOINT_NAMES",
  "PIKACHU_27DOF_FOOT_GEOM_NAMES",
  "PIKACHU_27DOF_PRY_XML",
  "PIKACHU_27DOF_YRP_XML",
  "get_body_column_names",
  "get_pikachu_27dof_yrp_mocap_robot_cfg",
  "get_pikachu_27dof_yrp_robot_cfg",
  "get_yrp_spec",
]
