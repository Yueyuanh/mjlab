"""定位 pikachu_v03 源模型在刚性执行器下发散的原因。

⚠️ **原假设已被本脚本自己证伪，结论改过一次**：
最初以为是「源模型本身在 ``mj_step`` 下发散」，重跑后发现源模型（base 焊死在
世界、无执行器）**开/关重力都能稳跑 2000 步**。唯一不稳的组合是
**刚性 position 执行器 + 显式 ``euler`` 积分** —— 踝关节等效惯量只有
9.2e-5（比 hip_pitch 小两个数量级），第 ~21 步就报
``Nan, Inf or huge value in QACC at DOF 5``。

真正的解法是 **``implicitfast`` 积分器**（mjlab 默认），**不是 armature**：
euler 下加 armature 反而更差（9.8 vs 3.1），implicitfast 下两种都掉到 ~1e-10。
armature 管的是物理保真度与刚度反推（``kp = armature × ω²``）。

结论见 ``docs/pikachu_v03/01-model-conversion.md`` 坑 5。

用法：
    uv run python scripts/diag_v03_stability.py
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

SRC_DIR = Path("/home/finnox/Pikachu/PikachuRobot/pikachu_urdf/pikachu_v03_base_series")
SRC_XML = SRC_DIR / "mjcf" / "pikachu_v03_base_series.xml"
TMP = SRC_DIR / "mjcf" / "_diag_tmp.xml"

# mjlab 给 DM 电机算的反射惯量（见 pikachu_constants.py）。
ARM_J4310 = 2.206e-5 * 100.0
ARM_J3507 = 6.69169e-6 * 21.0**2


def build(*, armature: float | None, actuators: bool, kp: float) -> mujoco.MjModel:
  xml = SRC_XML.read_text()
  if armature is not None:
    xml = xml.replace(
      '<joint damping="0.3" frictionloss="0.01" solreflimit="0.004 1" />',
      f'<joint damping="0.3" frictionloss="0.01" solreflimit="0.004 1" '
      f'armature="{armature}" />',
    )
  if actuators:
    acts = "".join(
      f'\n        <position name="p_{j}" joint="{j}" kp="{kp}" kv="{kp/40}" '
      f'ctrlrange="-3 3" />'
      for j in (
        "right_hip_pitch_joint",
        "right_knee_up_joint",
        "right_ankle_joint",
      )
    )
    xml = re.sub(
      r"\s*<actuator>.*?</actuator>",
      f"\n    <actuator>{acts}\n    </actuator>",
      xml,
      flags=re.S,
    )
  TMP.write_text(xml)
  return mujoco.MjModel.from_xml_path(str(TMP))


def trial(tag: str, *, integrator: str = "euler", **kw) -> None:
  m = build(**kw)
  m.opt.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{integrator.upper()}")
  d = mujoco.MjData(m)
  m.opt.gravity[:] = 0.0
  if m.nu:
    d.ctrl[:] = 0.0
  bad = -1
  for s in range(2000):
    mujoco.mj_step(m, d)
    if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
      bad = s
      break
  if bad < 0:
    mx = float(np.abs(d.qvel).max())
    print(f"  {tag:<52} 稳定 2000 步 (max|qvel|={mx:.3e})")
  else:
    print(f"  {tag:<52} 第 {bad} 步发散")


def inertias() -> None:
  m = build(armature=None, actuators=False, kp=0.0)
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  M = np.zeros((m.nv, m.nv))
  mujoco.mj_fullM(m, d, M)
  print("=== 关节等效惯量（mj_forward 之后读 M）===")
  for j in range(m.njnt):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    dof = m.jnt_dofadr[j]
    print(f"  {n:<26} M[dof,dof]={M[dof,dof]:.4e}  armature={m.dof_armature[dof]:.4e}")


if __name__ == "__main__":
  try:
    inertias()
    print()
    print("=== 稳定性逐项排查（关重力，2000 步）===")
    trial("原样（无 armature、无 actuator）", armature=None, actuators=False, kp=0)
    trial("+ armature(J3507)", armature=ARM_J3507, actuators=False, kp=0)
    trial("+ armature(J4310)", armature=ARM_J4310, actuators=False, kp=0)
    trial("+ position kp=20（无 armature）", armature=None, actuators=True, kp=20)
    trial("+ position kp=20 + armature(J3507)",
          armature=ARM_J3507, actuators=True, kp=20)
    trial("+ position kp=150 + armature(J3507)",
          armature=ARM_J3507, actuators=True, kp=150)
    # ↓ 这一组是结论所在：稳不稳定看**积分器**，不看 armature。
    #   mjlab 默认 implicitfast（src/mjlab/sim/sim.py:103），v03 任务没覆盖它，
    #   所以这个坑在训练里是被默认值挡掉的 —— 换 euler 立刻踩。
    print()
    print("=== 分离「积分器」与「armature」各自的贡献（kp=150）===")
    for integ in ("euler", "implicitfast"):
      for arm, tag in ((None, "无 armature"), (ARM_J3507, "+ armature(J3507)")):
        trial(f"kp=150  {integ:<13} {tag}",
              integrator=integ, armature=arm, actuators=True, kp=150)
  finally:
    TMP.unlink(missing_ok=True)
