"""为 pikachu_v03 标定并联四杆膝的传动关系 + 站立 keyframe。

v03 的膝是**并联四杆**：只有 knee_up 是驱动关节，knee / knee_mid 是被动关节，
靠 <equality><connect> 约束跟随。这个约束只在 mj_step 里求解 —— mj_forward
不会把被动关节拉到正确位置。

三个关键做法（都是踩坑后定的）：

1. **关重力 + 零外力**。此时系统里唯一的作用力就是 <connect> 的约束力，
   被动关节必然收敛到精确的约束流形上；而且 PD 没有稳态误差，可以用很小的
   增益。重力下 PD 永远有稳态误差，位置判据根本收敛不了。

2. **用力矩电机 + 显式 PD，不用 <position> 执行器**。
   实测 kp=20 的隐式位置执行器**顶不过冗余的 <connect> 约束**：
   knee_up 完全不动（err=1.2 rad），knee 被钉在零位偏置 1.66° 上，
   看起来像「机构锁死」。力矩电机没有这个问题。

3. **增益要小**。踝的等效惯量只有 9.2e-5 kg·m²，显式 PD 的稳定条件是
   KP < 4·I/dt²（dt=0.002 时踝上限 92，但经四杆耦合后有效惯量更小）。
   关重力后无稳态误差，KP=20/KD=0.5 既稳又准。

用源模型（base_link 焊死）而不是训练模型：训练模型有 freejoint，
只驱动一条腿时反作用力会让整机乱飞，纯机构学标定不需要那个自由度。

用法：
    uv run python scripts/probe_v03_stance.py
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
KNEE_UP_RANGE = (-1.20, 0.50)  # 右腿 knee_up 行程（限位 -1.22..0.52 留裕量）

DRV = (
  ("act_Rhip_pitch", "right_hip_pitch_joint"),
  ("act_Rknee_up", "right_knee_up_joint"),
  ("act_Rankle", "right_ankle_joint"),
)


def main() -> None:
  m = mujoco.MjModel.from_xml_path(str(SRC_XML))
  d = mujoco.MjData(m)
  m.opt.gravity[:] = 0.0  # 关重力：只剩约束力，且 PD 无稳态误差

  jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
  jid = {n: i for i, n in enumerate(jn)}
  qadr = {n: m.jnt_qposadr[jid[n]] for n in jn}
  vadr = {n: m.jnt_dofadr[jid[n]] for n in jn}
  an = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m.nu)]
  aid = {n: i for i, n in enumerate(an)}

  r_ankle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_link")
  g0 = next(
    g
    for g in range(m.ngeom)
    if m.geom_bodyid[g] == r_ankle and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
  )
  did = m.geom_dataid[g0]
  va, vn = m.mesh_vertadr[did], m.mesh_vertnum[did]
  rot0 = np.zeros(9)
  mujoco.mju_quat2Mat(rot0, m.geom_quat[g0])
  verts_body = (
    rot0.reshape(3, 3) @ m.mesh_vert[va : va + vn].reshape(-1, 3).T
  ).T + m.geom_pos[g0]

  targets = {
    "right_hip_pitch_joint": 0.0,
    "right_knee_up_joint": 0.0,
    "right_ankle_joint": 0.0,
  }

  def settle(hip: float, k: float, ank: float, steps: int = 8000) -> float:
    targets.update(
      {
        "right_hip_pitch_joint": hip,
        "right_knee_up_joint": k,
        "right_ankle_joint": ank,
      }
    )
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    integ = dict.fromkeys((j for _, j in DRV), 0.0)
    for s in range(steps):
      for a, j in DRV:
        e = targets[j] - d.qpos[qadr[j]]
        # 关重力后仍有 <connect> 约束反力要平衡，纯 PD 有 ~0.2 rad 稳态误差。
        # 加积分项把它消掉（积分限幅防止绕死）。
        integ[j] = float(np.clip(integ[j] + e * m.opt.timestep, -0.4, 0.4))
        d.ctrl[aid[a]] = KP * e - KD * d.qvel[vadr[j]] + KI * integ[j]
      mujoco.mj_step(m, d)
      if (
        s % 500 == 499 and max(abs(d.qpos[qadr[j]] - targets[j]) for _, j in DRV) < 1e-6
      ):
        break
    for a, _ in DRV:
      d.ctrl[aid[a]] = 0.0
    mujoco.mj_forward(m, d)
    return max(abs(d.qpos[qadr[j]] - targets[j]) for _, j in DRV)

  def measure() -> tuple[float, float, float]:
    """(最低鞋底 z, 脚掌俯仰 rad, 脚中心 x) —— 俯仰用 body 系 x 轴。"""
    R = d.geom_xmat[g0].reshape(3, 3)
    w = (R @ verts_body.T).T + d.geom_xpos[g0]
    Rb = d.xmat[r_ankle].reshape(3, 3)
    xa = Rb @ np.array([1.0, 0.0, 0.0])
    return (
      float(w[:, 2].min()),
      float(np.arctan2(xa[2], np.hypot(xa[0], xa[1]))),
      float(w[:, 0].mean()),
    )

  # ---- 1) knee_up -> knee / knee_mid 传动关系 ----
  print("=== knee_up -> 被动关节传动关系（关重力，约束精确收敛）===")
  print(
    f"{'knee_up':>9} {'knee':>10} {'knee_mid':>10} {'knee/knee_up':>13} {'收敛err':>10}"
  )
  ratio = []
  for k in np.linspace(KNEE_UP_RANGE[1], KNEE_UP_RANGE[0], 7):
    err = settle(0.0, float(k), 0.0)
    knee = float(d.qpos[qadr["right_knee_joint"]])
    kmid = float(d.qpos[qadr["right_knee_mid_joint"]])
    if abs(k) > 1e-6:
      ratio.append(knee / k)
    print(
      f"{k:>9.3f} {np.degrees(knee):>9.2f}° {np.degrees(kmid):>9.2f}° "
      f"{knee / k if abs(k) > 1e-6 else float('nan'):>13.4f} {err:>10.2e}"
    )
  print(f"传动比范围: {min(ratio):.4f} ~ {max(ratio):.4f}")

  settle(0.0, 0.0, 0.0)
  print(
    f"零位偏置 (knee_up=0): knee="
    f"{np.degrees(d.qpos[qadr['right_knee_joint']]):.3f}°  "
    f"knee_mid={np.degrees(d.qpos[qadr['right_knee_mid_joint']]):.3f}°"
  )

  # ---- 2) 站立姿态：脚掌水平 + 踝在髋正下方 ----
  print()
  print("=== 站立姿态标定（脚掌水平 & 踝 x 对齐髋）===")
  print(
    f"{'knee_up':>9} {'knee':>9} {'hip_pitch':>10} {'ankle':>9} "
    f"{'base_z':>9} {'俯仰(°)':>9} {'收敛err':>9}"
  )
  rows = []
  for k in (0.0, -0.20, -0.40, -0.60, -0.80, -1.00):
    hip, ank = -k / 2.0, -k / 2.0
    for _ in range(8):
      settle(hip, k, ank)
      lo, pitch, cx = measure()
      if abs(pitch) < 1e-10 and abs(cx) < 1e-10:
        break
      e = 1e-3
      settle(hip + e, k, ank)
      _, ph, ch = measure()
      settle(hip, k, ank + e)
      _, pa, ca = measure()
      J = np.array(
        [[(ph - pitch) / e, (pa - pitch) / e], [(ch - cx) / e, (ca - cx) / e]]
      )
      try:
        step = np.linalg.solve(J, np.array([pitch, cx]))
      except np.linalg.LinAlgError:
        break
      hip += float(np.clip(-step[0], -0.2, 0.2))
      ank += float(np.clip(-step[1], -0.2, 0.2))

    err = settle(hip, k, ank)
    lo, pitch, cx = measure()
    knee = float(d.qpos[qadr["right_knee_joint"]])
    kmid = float(d.qpos[qadr["right_knee_mid_joint"]])
    rows.append((k, knee, kmid, hip, ank, -lo))
    print(
      f"{k:>9.2f} {np.degrees(knee):>8.2f}° {hip:>10.4f} {ank:>9.4f} "
      f"{-lo:>9.5f} {np.degrees(pitch):>8.3f}° {err:>9.1e}"
    )

  print()
  print("=== 可直接抄进 constants 的 keyframe ===")
  for idx, tag in ((0, "STAND (直立)"), (1, "KNEES_BENT (knee_up=-0.20)")):
    k, knee, kmid, hip, ank, base_z = rows[idx]
    print(f"[{tag}]  pos=(0, 0, {base_z:.5f})")
    print(
      f"  right_hip_pitch={hip:+.4f}  right_knee_up={k:+.4f}  right_ankle={ank:+.4f}"
    )
    print(f"  left_hip_pitch={hip:+.4f}  left_knee_up={-k:+.4f}  left_ankle={ank:+.4f}")
    print(f"  被动 right_knee={knee:+.4f}  right_knee_mid={kmid:+.4f}")
    print(f"       left_knee={knee:+.4f}   left_knee_mid={kmid:+.4f}")
    print()


if __name__ == "__main__":
  main()
