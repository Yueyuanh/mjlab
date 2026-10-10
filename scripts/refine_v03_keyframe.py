"""把 v03 的站立 keyframe 精修到**静平衡点**：质心落在脚掌支撑中心正上方。

为什么需要这一步：``probe_v03_both_legs.py`` 的解算目标是「脚掌俯仰 0° +
脚心 x = 0」，也就是把脚摆在 **base 原点**正下方。但 v03 的质心在
x = -0.0195 m（base 原点后方 2 cm），于是重力矩始终把机器人往后压 ——
零动作开环测试里机器人会先向后仰 9.6°、base 掉到 0.2248 m，再被踝关节
PD 的弹性力矩拉回来，最后停在 pitch = -0.167 rad 的**非零平衡点**上。

那不是静平衡点：真正的静平衡要求**质心水平投影落在支撑中心**，此时踝关节
不需要任何 PD 力矩（PD 误差为 0），零动作就是不动点。

解算思路（固定点迭代）：
    cx_target ← 0
    重复：
      解 hip/ankle 使 脚掌俯仰 = 0 且 脚心 x = cx_target
      量此时整机质心的 x  -> cx_target
   收敛后脚心就压在质心正下方。

用源模型（base_link 焊死、关重力）解算：base 焊死时 ``subtree_com`` 直接
给出 base 系下的质心，且关重力后 ``<connect>`` 约束精确收敛、PD 无稳态误差。

用法：
    uv run python scripts/refine_v03_keyframe.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

SRC_XML = (
  Path("/home/finnox/Pikachu/PikachuRobot/pikachu_urdf/pikachu_v03_base_series")
  / "mjcf"
  / "pikachu_v03_base_series.xml"
)

KP, KD, KI = 20.0, 0.5, 50.0
DRIVEN = ("hip_pitch_joint", "knee_up_joint", "ankle_joint")

# 迭代起点：probe_v03_both_legs.py 解出的 KNEES_BENT（knee_up_R = -0.20）。
START = {
  "right_hip_pitch_joint": -0.5430,
  "right_knee_up_joint": -0.2000,
  "right_ankle_joint": -0.3892,
  "left_hip_pitch_joint": -0.5410,
  "left_knee_up_joint": 0.2000,
  "left_ankle_joint": 0.3873,
}

SIDES = ("right", "left")


def main() -> None:
  m = mujoco.MjModel.from_xml_path(str(SRC_XML))
  d = mujoco.MjData(m)
  m.opt.gravity[:] = 0.0

  jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
  jid = {n: i for i, n in enumerate(jn)}
  qadr = {n: m.jnt_qposadr[jid[n]] for n in jn}
  vadr = {n: m.jnt_dofadr[jid[n]] for n in jn}
  aid = {jn[m.actuator_trnid[a, 0]]: a for a in range(m.nu)}

  feet: dict[str, tuple[int, np.ndarray]] = {}
  for side in SIDES:
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_link")
    g = next(
      g
      for g in range(m.ngeom)
      if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    )
    did = m.geom_dataid[g]
    va, vn = m.mesh_vertadr[did], m.mesh_vertnum[did]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.geom_quat[g])
    verts = (
      rot.reshape(3, 3) @ m.mesh_vert[va : va + vn].reshape(-1, 3).T
    ).T + m.geom_pos[g]
    feet[side] = (b, verts)

  joints = [f"{s}_{k}" for s in SIDES for k in DRIVEN]
  targets = {j: START[j] for j in joints}

  def settle(steps: int = 8000) -> None:
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    for j in joints:
      d.qpos[qadr[j]] = targets[j]
    integ = dict.fromkeys(joints, 0.0)
    for s in range(steps):
      for j in joints:
        e = targets[j] - d.qpos[qadr[j]]
        integ[j] = float(np.clip(integ[j] + e * m.opt.timestep, -0.4, 0.4))
        d.ctrl[aid[j]] = KP * e - KD * d.qvel[vadr[j]] + KI * integ[j]
      mujoco.mj_step(m, d)
      if (
        s % 500 == 499 and max(abs(d.qpos[qadr[j]] - targets[j]) for j in joints) < 1e-6
      ):
        break
    for j in joints:
      d.ctrl[aid[j]] = 0.0
    mujoco.mj_forward(m, d)

  def measure(side: str) -> tuple[float, float, float]:
    b, verts = feet[side]
    R = d.xmat[b].reshape(3, 3)
    w = (R @ verts.T).T + d.xpos[b]
    xa = R @ np.array([1.0, 0.0, 0.0])
    return (
      float(w[:, 2].min()),
      float(np.arctan2(xa[2], np.hypot(xa[0], xa[1]))),
      float(w[:, 0].mean()),
    )

  def com_x() -> float:
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    return float(d.subtree_com[base][0])

  def solve_leg(side: str, cx_target: float) -> None:
    """解 hip/ankle，使脚掌俯仰 = 0 且脚心 x = cx_target。"""
    hip = targets[f"{side}_hip_pitch_joint"]
    ank = targets[f"{side}_ankle_joint"]
    for _ in range(12):
      settle()
      _, pitch, cx = measure(side)
      if abs(pitch) < 1e-10 and abs(cx - cx_target) < 1e-10:
        break
      e = 1e-3
      targets[f"{side}_hip_pitch_joint"] = hip + e
      settle()
      _, ph, ch = measure(side)
      targets[f"{side}_hip_pitch_joint"] = hip
      targets[f"{side}_ankle_joint"] = ank + e
      settle()
      _, pa, ca = measure(side)
      targets[f"{side}_ankle_joint"] = ank
      J = np.array(
        [
          [(ph - pitch) / e, (pa - pitch) / e],
          [(ch - cx) / e, (ca - cx) / e],
        ]
      )
      try:
        step = np.linalg.solve(J, np.array([pitch, cx - cx_target]))
      except np.linalg.LinAlgError:
        break
      hip += float(np.clip(-step[0], -0.2, 0.2))
      ank += float(np.clip(-step[1], -0.2, 0.2))
      targets[f"{side}_hip_pitch_joint"] = hip
      targets[f"{side}_ankle_joint"] = ank

  print("=== 固定点迭代：把脚心解到质心正下方 ===")
  print(
    f"{'iter':>5} {'cx_target':>11} {'CoM_x':>10} {'脚心x_R':>10} "
    f"{'脚心x_L':>10} {'base_z':>9} {'俯仰_R':>9} {'俯仰_L':>9}"
  )

  cx_target = 0.0
  for it in range(8):
    for side in SIDES:
      solve_leg(side, cx_target)
    settle()
    com = com_x()
    _, pit_r, cx_r = measure("right")
    _, pit_l, cx_l = measure("left")
    lo = min(measure("right")[0], measure("left")[0])
    print(
      f"{it:>5} {cx_target:>11.5f} {com:>10.5f} {cx_r:>10.5f} {cx_l:>10.5f} "
      f"{-lo:>9.5f} {np.degrees(pit_r):>8.3f}° {np.degrees(pit_l):>8.3f}°"
    )
    if abs(com - cx_target) < 1e-5:
      break
    cx_target = com

  print()
  print("=== 静平衡 keyframe ===")
  base_z = -min(measure("right")[0], measure("left")[0])
  print(f"pos=(0, 0, {base_z:.5f})")
  for side in SIDES:
    print(
      f"  {side:<6}: hip_pitch={targets[f'{side}_hip_pitch_joint']:+.4f} "
      f"knee_up={targets[f'{side}_knee_up_joint']:+.4f} "
      f"ankle={targets[f'{side}_ankle_joint']:+.4f}"
    )
  for side in SIDES:
    print(
      f"  {side:<6}: knee={d.qpos[qadr[f'{side}_knee_joint']]:+.4f} "
      f"knee_mid={d.qpos[qadr[f'{side}_knee_mid_joint']]:+.4f}"
    )
  print(f"  质心 x = {com_x():+.5f}   脚心 x = {measure('right')[2]:+.5f}")


if __name__ == "__main__":
  main()
