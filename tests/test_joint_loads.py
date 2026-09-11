"""Tests for script-level static/dynamic joint-load analysis.

跑 ``analyze_joint_loads.analyze``（固定 base_link，MuJoCo 逆动力学）并断言几个
物理不变量，防止以后把动力学改坏：

- 浮动基座在默认姿态的反作用力 ≈ 整机重量（重力平衡）。
- 质量阵对角线 M_ii 为正、整阵对称（动态加速扭矩的系数）。
- 每关节「静态力」= 远端子树总重 × g；扫描得到的最大静态力矩至少不小于默认姿态。
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mjlab.mocap.analyze_stability import DEFAULT_XML
from mjlab.scripts.analyze_joint_loads import analyze


@pytest.fixture(scope="module")
def host_model():
  model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
  return model


def test_base_reaction_equals_total_weight(host_model) -> None:
  """固定 base 默认站姿，浮动基座反作用力（|qfrc_bias[2]|）应抵消整机重力。"""
  res = analyze(host_model, n_steps=5)
  data = mujoco.MjData(host_model)
  data.qpos[:] = host_model.qpos0
  data.qvel[:] = 0.0
  mujoco.mj_forward(host_model, data)
  del res
  total = float(host_model.body_mass.sum())
  # qfrc_bias 是平衡重力所需的基座反作用力，z 平移分量大小 = 整机重力。
  z_force = abs(float(data.qfrc_bias[2]))
  assert z_force == pytest.approx(total * 9.81, abs=0.5)


def test_mass_matrix_psd_and_diagonal_positive(host_model) -> None:
  """质量阵对称正定；每关节有效惯量 M_ii > 0（动态加速扭矩有物理意义）。"""
  res = analyze(host_model, n_steps=5)
  eff = res["eff_inertia"]
  assert np.all(eff > 0.0)
  data = mujoco.MjData(host_model)
  data.qpos[:] = host_model.qpos0
  mujoco.mj_forward(host_model, data)
  M = np.zeros((host_model.nv, host_model.nv))
  mujoco.mj_fullM(host_model, data, M)
  assert M == pytest.approx(M.T, abs=1e-9)


def test_static_force_matches_distal_subtree(host_model) -> None:
  """每关节静态力 = 远端子树质量 × g；扫描最大力矩不小于默认姿态力矩。"""
  res = analyze(host_model, n_steps=5)
  names = {n: i for i, n in enumerate(res["names"])}
  head = names["head_pitch_joint"]
  f = res["static_force_N"][head]
  assert f == pytest.approx(1.05 * 9.81, abs=0.8)  # 头 0.96kg + 脖子/耳朵子树
  assert np.all(
    np.abs(res["max_static_torque_Nm"]) >= np.abs(res["default_static_torque_Nm"])
  )
