"""Tests for the 27-dof quadruped constants module.

纯断言常量 sanity(不构建 env、不加载参考运动),防止把四足适配改坏:

- 走步关节集恰为 16(腿 10 + 前腿 arm_pitch/roll/elbow 6)。
- 起步关键帧覆盖全部 27 关节;动作缩放正则全部只落在走步关节上(arm_yaw 等不参与)。
- 手脚支撑 geom 正则能在训练体模型里命中真实的 ``*_collision`` geoms。
"""

from __future__ import annotations

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.pikachu_xml import pikachu_27dof_quad_constants as q
from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_constants import (
  ALL_JOINT_NAMES,
  LEG_JOINT_NAMES,
)

_FRONT_LEG_JOINTS = [
  f"{side}_{j}_joint"
  for side in ("left", "right")
  for j in ("arm_pitch", "arm_roll", "elbow")
]


def _all_geom_names(model: mujoco.MjModel) -> list[str]:
  """从 ``names`` 缓冲读全部 geom 名(mj 只存 adr,需自行切片)。"""
  out = []
  for g in range(model.ngeom):
    adr = int(model.name_geomadr[g])
    if adr < 0:
      out.append("")
      continue
    end = model.names.find(b"\x00", adr)
    out.append(model.names[adr:end].decode() if end >= adr else "")
  return out


def test_walk_joint_names_are_sixteen() -> None:
  """走步关节 = 腿 10 + 前腿 6,恰 16,顺序腿在前;与固定关节无重叠。"""
  walk = q.QUAD_27DOF_WALK_JOINT_NAMES
  assert len(walk) == 16
  assert walk[:10] == LEG_JOINT_NAMES
  assert set(walk[10:]) == set(_FRONT_LEG_JOINTS)
  assert set(walk) & set(q.QUAD_27DOF_FIXED_JOINT_NAMES) == set()


def test_keyframe_covers_all_27_joints() -> None:
  """起步关键帧的关节 dict 恰好覆盖 27 个关节名。"""
  joint_pos = q.QUAD_27DOF_KEYFRAME.joint_pos
  assert joint_pos is not None
  joints = set(joint_pos)
  assert joints == set(ALL_JOINT_NAMES)
  assert len(joints) == 27


def test_action_scale_only_touches_walk_joints() -> None:
  """每个缩放正则都命中、且只命中走步 16 关节;固定关节(arm_yaw/头/耳/尾)全部覆盖。"""
  walk = set(q.QUAD_27DOF_WALK_JOINT_NAMES)
  all_ = set(ALL_JOINT_NAMES)
  for pattern in q.QUAD_27DOF_ACTION_SCALE:
    r = re.compile(pattern)
    matched = {j for j in all_ if r.fullmatch(j)}
    assert matched, f"pattern {pattern!r} matches nothing"
    assert matched <= walk, f"pattern {pattern!r} matches {matched - walk}"
  for j in walk:
    assert any(re.compile(p).fullmatch(j) for p in q.QUAD_27DOF_ACTION_SCALE), (
      f"walk joint {j!r} has no action scale"
    )
  unmatched = {
    j
    for j in all_
    if not any(re.compile(p).fullmatch(j) for p in q.QUAD_27DOF_ACTION_SCALE)
  }
  assert unmatched == set(q.QUAD_27DOF_FIXED_JOINT_NAMES)


def test_support_geoms_exist_in_model() -> None:
  """编译训练体模型,确认手脚支撑 geom 正则能命中真实碰撞 geom。"""
  model = q.get_spec().compile()
  names = _all_geom_names(model)
  for pattern in (
    q.PIKACHU_27DOF_FOOT_GEOM_NAMES,
    q.QUAD_27DOF_HAND_GEOM_NAMES,
  ):
    r = re.compile(pattern)
    hits = [n for n in names if r.fullmatch(n)]
    assert len(hits) == 2, f"{pattern!r} -> {hits}"


def test_robot_cfg_builds() -> None:
  """四足机器人配置可构造,init_state 用四足关键帧(全 27 关节)。"""
  cfg = q.get_pikachu_27dof_quad_robot_cfg()
  assert cfg.init_state is q.QUAD_27DOF_KEYFRAME
  joint_pos = cfg.init_state.joint_pos
  assert joint_pos is not None
  assert set(joint_pos) == set(ALL_JOINT_NAMES)


def test_quad_keyframe_conserves_leg_pose() -> None:
  """四足起步姿态:base 前俯(绕 y 转动,w 分量 <1),腿关节左右反相。"""
  w, x, y, z = q.QUAD_27DOF_ROT
  assert np.isclose(x, 0.0) and np.isclose(z, 0.0)
  assert abs(y) > 0.5 and abs(w) < abs(y)  # 明显前俯而非竖直。
  kf = q.QUAD_27DOF_KEYFRAME.joint_pos
  assert kf is not None

  def leg_angle(side: str, joint: str) -> float:
    return kf[f"{side}_{joint}_joint"]

  assert leg_angle("left", "hip_pitch") == pytest.approx(
    -leg_angle("right", "hip_pitch")
  )
  assert leg_angle("left", "knee") == pytest.approx(-leg_angle("right", "knee"))
  assert leg_angle("left", "ankle") == pytest.approx(-leg_angle("right", "ankle"))
  assert leg_angle("left", "arm_pitch") == pytest.approx(
    -leg_angle("right", "arm_pitch")
  )
