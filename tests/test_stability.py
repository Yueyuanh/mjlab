"""Tests for CoM / inertia / ZMP computation in mocap.stability.

**Ground truth**：默认站立姿态（关节角全 0）的主惯量 (0.316, 0.262, 0.211) 不是
凭经验猜的——它已经用 MuJoCo 权威量交叉验证过：

- 把各 body 惯量平行轴合成到根原点 = (0.421, 0.367, 0.211)，与 ``mj_fullM``
  给 root free-joint 的 6×6 块精确一致；
- 减去整机 CoM 的平行轴 (m·z²≈0.1047，com_z=0.130) 得到绕整机 CoM 的
  (0.316, 0.262, 0.211)；
- CoM 与 ``subtree_com[0]`` 一致（(0.0023, 0, 0.130)）。

旧的 ``analyze_stability.py`` 报告 (0.380, 0.325, 0.211) 有 bug（平行轴参考系
搞混），本测试断言的是权威值。
"""

from __future__ import annotations

import numpy as np
import pytest

from mjlab.mocap.analyze_stability import DEFAULT_XML
from mjlab.mocap.stability import (
  _host_cinert,
  _host_geom_name,
  inertia_about_com,
  zmp_from_com,
)

# 27 个关节默认站立（model 关节序，qpos 第 7 位起）。
DEFAULT_POSE = np.zeros(27)


@pytest.fixture(scope="module")
def host_model():
  """宿主 27dof MjModel + 默认姿态 mj_forward 后的世界位姿。"""
  import mujoco

  model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  mujoco.mj_forward(model, data)
  return model, data


def test_total_mass(host_model) -> None:
  model, _ = host_model
  assert model.body_mass.sum() == pytest.approx(6.1965, abs=1e-3)


def test_com_matches_subtree(host_model) -> None:
  """整机质心与 MuJoCo 的 subtree_com[0] 一致。"""
  model, data = host_model
  com, _ = inertia_about_com(
    mass=model.body_mass,
    cinert=_host_cinert(model),
    body_ipos=model.body_ipos,
    body_iquat=model.body_iquat,
    xpos=np.asarray(data.xpos),
    xquat=np.asarray(data.xquat),
  )
  assert com == pytest.approx(np.asarray(data.subtree_com)[0], abs=1e-4)


def test_host_geom_name_finds_feet(host_model) -> None:
  """宿主 MjModel 无 geom_names；_host_geom_name 从 names 缓冲能取到足部 geom。"""
  model, _ = host_model
  assert not hasattr(model, "geom_names")  # 回归：宿主模型确实没有该字段
  names = [_host_geom_name(model, g) for g in range(model.ngeom)]
  feet = [nm for nm in names if "foot" in nm or "ankle" in nm]
  assert len(feet) == 2  # left_ankle_collision + right_ankle_collision
  assert all(nm.endswith("_collision") for nm in feet)


def test_host_cinert_shape(host_model) -> None:
  """宿主 body_inertia → cinert 6 分量（对角，离对角为 0）。"""
  model, _ = host_model
  cin = _host_cinert(model)
  assert cin.shape == (model.nbody, 6)
  assert np.all(cin[:, 3:] == 0.0)


def test_inertia_about_com_default_pose(host_model) -> None:
  """默认站立姿态绕整机 CoM 的主惯量 = MuJoCo 权威值 (0.316, 0.262, 0.211)。"""
  model, data = host_model
  _, inertia = inertia_about_com(
    mass=model.body_mass,
    cinert=_host_cinert(model),
    body_ipos=model.body_ipos,
    body_iquat=model.body_iquat,
    xpos=np.asarray(data.xpos),
    xquat=np.asarray(data.xquat),
  )
  principal = np.sort(np.linalg.eigvalsh(inertia))[::-1]
  assert principal[0] == pytest.approx(0.316, abs=0.002)
  assert principal[1] == pytest.approx(0.262, abs=0.002)
  assert principal[2] == pytest.approx(0.211, abs=0.002)


def test_zmp_static_com() -> None:
  """静止 CoM：无水平加速度 → ZMP = CoM 地面投影。"""
  com = np.repeat([[0.5, 0.0, 0.13]], 5, axis=0)
  zmp = zmp_from_com(com, dt=1.0 / 50.0)
  assert zmp[:, 0] == pytest.approx(0.5, abs=1e-9)
  assert zmp[:, 1] == pytest.approx(0.0, abs=1e-9)


def test_zmp_uniform_motion() -> None:
  """匀速水平 CoM：零水平加速度 → ZMP 沿 CoM 投影移动。"""
  t = np.arange(100) / 50.0
  com = np.column_stack((0.5 + 0.5 * t, np.zeros_like(t), np.full_like(t, 0.13)))
  zmp = zmp_from_com(com, dt=0.02)
  # 中心差分的端点噪声忽略，只看中间段贴着 CoM 投影。
  mid = slice(2, -2)
  assert zmp[mid, 0] == pytest.approx(com[mid, 0], abs=2e-3)


def test_zmp_lean_negative_direction() -> None:
  """以恒定加速度加速时，ZMP 落在 CoM 加速方向的负侧（支撑点滞后）。"""
  t = np.arange(120) / 50.0
  # x = 0.5 t^2 → a_x = 1.0，com_z=0.13 → zmp_x = com_x - 0.13/9.81 * 1.0。
  com = np.column_stack((0.5 * t**2, np.zeros_like(t), np.full_like(t, 0.13)))
  zmp = zmp_from_com(com, dt=0.02)
  # ZMP 在 CoM 之后（加速时支撑点滞后）。
  assert np.all(zmp[10:20, 0] <= com[10:20, 0])
  assert zmp[10, 0] == pytest.approx(com[10, 0] - 0.13 / 9.81, abs=2e-3)
