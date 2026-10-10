"""标定 pikachu_v03 **左右双腿**的站立 keyframe，并验证镜像关系。

``probe_v03_stance.py`` 只驱动了右腿，keyframe 里左腿的值是按「整体镜像」
猜的 —— 实测证明这个猜测是错的。

从训练模型的关节轴表看（``jnt_axis``）：

    关节          右轴            左轴            左右是否同向
    hip_pitch     [0, 1, 0]       [0, 1, 0]       相同
    hip_roll      [1, 0, 0]       [1, 0, 0]       相同
    hip_yaw       [-.571,0,.821]  [-.571,0,.821]  相同
    knee          [0,-1, 0]       [0,-1, 0]       相同
    ankle         [0,-1, 0]       [0, 1, 0]       **相反**
    knee_up       [0,-1, 0]       [0, 1, 0]       **相反**
    knee_mid      [0,-1, 0]       [0, 1, 0]       **相反**

也就是说左腿**不是**整体镜像：髋三轴和膝轴左右同向（这两个关节的正方向
在世界系里指向同一侧），只有踝 / knee_up / knee_mid 因为 CAD 里对应的
link 被镜像过而反号。所以 keyframe 里：

    left_hip_pitch  =  right_hip_pitch      （同号）
    left_knee       =  right_knee           （同号，被动）
    left_knee_up    = -right_knee_up        （反号）
    left_knee_mid   = -right_knee_mid       （反号，被动）
    left_ankle      = -right_ankle          （反号）

本脚本用「与右腿完全相同的目标」（脚掌俯仰 = 0、脚中心 x = 0）独立求解
左腿，直接读出收敛后的全部左腿关节值，不依赖上面的推断。

用法：
    uv run python scripts/probe_v03_both_legs.py
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

KP, KD, KI = 20.0, 0.5, 50.0  # 关重力下标定，增益可取得很小

# 每条腿驱动的三个关节（顺序固定：hip_pitch, knee_up, ankle）。
DRIVEN = ("hip_pitch_joint", "knee_up_joint", "ankle_joint")


def main() -> None:
  m = mujoco.MjModel.from_xml_path(str(SRC_XML))
  d = mujoco.MjData(m)
  m.opt.gravity[:] = 0.0

  jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
  jid = {n: i for i, n in enumerate(jn)}
  qadr = {n: m.jnt_qposadr[jid[n]] for n in jn}
  vadr = {n: m.jnt_dofadr[jid[n]] for n in jn}

  # 关节 -> actuator 下标（源模型里每条腿 5 个电机，只有这 3 个是我们要驱动的）。
  aid: dict[str, int] = {}
  for a in range(m.nu):
    j = m.actuator_trnid[a, 0]
    aid[jn[j]] = a

  # 每个 ankle body 的 mesh 顶点（body 系），用于量鞋底高度 / 俯仰 / 脚中心。
  feet: dict[str, tuple[int, np.ndarray]] = {}
  for side in ("right", "left"):
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

  joints = [f"{s}_{k}" for s in ("right", "left") for k in DRIVEN]

  def settle(targets: dict[str, float], steps: int = 8000) -> float:
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
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
    return max(abs(d.qpos[qadr[j]] - targets[j]) for j in joints)

  def measure(side: str) -> tuple[float, float, float]:
    """(最低鞋底 z, 脚掌俯仰 rad, 脚中心 x)。俯仰用 body 系 x 轴。"""
    b, verts = feet[side]
    R = d.xmat[b].reshape(3, 3)
    w = (R @ verts.T).T + d.xpos[b]
    xa = R @ np.array([1.0, 0.0, 0.0])
    return (
      float(w[:, 2].min()),
      float(np.arctan2(xa[2], np.hypot(xa[0], xa[1]))),
      float(w[:, 0].mean()),
    )

  def solve_leg(side: str, k: float, hip: float, ank: float) -> tuple[float, float]:
    """在 knee_up=k 固定下，解出使「脚掌水平 + 脚中心 x=0」的 (hip, ankle)。"""
    for _ in range(10):
      targets[f"{side}_hip_pitch_joint"] = hip
      targets[f"{side}_knee_up_joint"] = k
      targets[f"{side}_ankle_joint"] = ank
      settle(targets)
      _, pitch, cx = measure(side)
      if abs(pitch) < 1e-10 and abs(cx) < 1e-10:
        break
      e = 1e-3
      targets[f"{side}_hip_pitch_joint"] = hip + e
      settle(targets)
      _, ph, ch = measure(side)
      targets[f"{side}_hip_pitch_joint"] = hip
      targets[f"{side}_ankle_joint"] = ank + e
      settle(targets)
      _, pa, ca = measure(side)
      targets[f"{side}_ankle_joint"] = ank
      J = np.array(
        [[(ph - pitch) / e, (pa - pitch) / e], [(ch - cx) / e, (ca - cx) / e]]
      )
      try:
        step = np.linalg.solve(J, np.array([pitch, cx]))
      except np.linalg.LinAlgError:
        break
      hip += float(np.clip(-step[0], -0.2, 0.2))
      ank += float(np.clip(-step[1], -0.2, 0.2))
    return hip, ank

  targets: dict[str, float] = dict.fromkeys(joints, 0.0)

  # 右腿 knee_up 的行程是 [-1.22, +0.52]；左腿镜像，用 [-0.52, +1.22]。
  print("=== 双腿站立标定（脚掌水平 & 脚中心 x=0）===")
  print(
    f"{'knee_up_R':>10} {'hip_R':>8} {'ank_R':>8} | "
    f"{'knee_up_L':>10} {'hip_L':>8} {'ank_L':>8} | {'base_z':>8}"
  )
  rows = []
  for k in (0.0, -0.20, -0.40, -0.60):
    hip_r, ank_r = solve_leg("right", k, -k / 2.0, -k / 2.0)
    hip_l, ank_l = solve_leg("left", -k, -k / 2.0, -k / 2.0)
    # 两腿同时摆好，量真实 base 高度（取两脚鞋底最低）。
    targets["right_hip_pitch_joint"] = hip_r
    targets["right_knee_up_joint"] = k
    targets["right_ankle_joint"] = ank_r
    targets["left_hip_pitch_joint"] = hip_l
    targets["left_knee_up_joint"] = -k
    targets["left_ankle_joint"] = ank_l
    err = settle(targets)
    lo_r, pit_r, cx_r = measure("right")
    lo_l, pit_l, cx_l = measure("left")
    base_z = -min(lo_r, lo_l)
    rows.append((k, hip_r, ank_r, hip_l, ank_l, base_z))
    print(
      f"{k:>10.2f} {hip_r:>8.4f} {ank_r:>8.4f} | "
      f"{-k:>10.2f} {hip_l:>8.4f} {ank_l:>8.4f} | {base_z:>8.5f}"
    )
    print(
      f"{'':>10} 俯仰 {np.degrees(pit_r):+.3f}° / {np.degrees(pit_l):+.3f}°  "
      f"脚心x {cx_r:+.5f} / {cx_l:+.5f}  收敛err {err:.1e}"
    )

  # ---- 镜像关系验证：左右被动关节值 ----
  print()
  print("=== 镜像关系验证（knee_up 取遍行程，两腿同时摆）===")
  print(
    f"{'k_R':>8} {'knee_R':>9} {'knee_L':>9} {'mid_R':>9} {'mid_L':>9} "
    f"{'ank_R':>8} {'ank_L':>8}  {'L/R 比值':>22}"
  )
  for k in (0.0, -0.20, -0.40, -0.60, -0.80):
    hip_r, ank_r = solve_leg("right", k, -k / 2.0, -k / 2.0)
    hip_l, ank_l = solve_leg("left", -k, -k / 2.0, -k / 2.0)
    targets.update(
      {
        "right_hip_pitch_joint": hip_r,
        "right_knee_up_joint": k,
        "right_ankle_joint": ank_r,
        "left_hip_pitch_joint": hip_l,
        "left_knee_up_joint": -k,
        "left_ankle_joint": ank_l,
      }
    )
    settle(targets)
    kr = d.qpos[qadr["right_knee_joint"]]
    kl = d.qpos[qadr["left_knee_joint"]]
    mr = d.qpos[qadr["right_knee_mid_joint"]]
    ml = d.qpos[qadr["left_knee_mid_joint"]]
    ar = d.qpos[qadr["right_ankle_joint"]]
    al = d.qpos[qadr["left_ankle_joint"]]

    def rat(a: float, b: float) -> str:
      return f"{a / b:+.4f}" if abs(b) > 1e-6 else "n/a"

    print(
      f"{k:>8.2f} {np.degrees(kr):>8.3f}° {np.degrees(kl):>8.3f}° "
      f"{np.degrees(mr):>8.3f}° {np.degrees(ml):>8.3f}° "
      f"{ar:>8.4f} {al:>8.4f}  "
      f"knee {rat(kl, kr):>7} mid {rat(ml, mr):>7} ank {rat(al, ar):>7}"
    )

  print()
  print("=== 可直接抄进 constants 的 keyframe ===")
  for idx, tag in ((0, "STAND (直立)"), (1, "KNEES_BENT (knee_up_R=-0.20)")):
    k, hip_r, ank_r, hip_l, ank_l, base_z = rows[idx]
    targets.update(
      {
        "right_hip_pitch_joint": hip_r,
        "right_knee_up_joint": k,
        "right_ankle_joint": ank_r,
        "left_hip_pitch_joint": hip_l,
        "left_knee_up_joint": -k,
        "left_ankle_joint": ank_l,
      }
    )
    settle(targets)
    kr = d.qpos[qadr["right_knee_joint"]]
    kl = d.qpos[qadr["left_knee_joint"]]
    mr = d.qpos[qadr["right_knee_mid_joint"]]
    ml = d.qpos[qadr["left_knee_mid_joint"]]
    print(f"[{tag}]  pos=(0, 0, {base_z:.5f})")
    print(f"  right: hip_pitch={hip_r:+.4f} knee_up={k:+.4f} ankle={ank_r:+.4f}")
    print(f"  left : hip_pitch={hip_l:+.4f} knee_up={-k:+.4f} ankle={ank_l:+.4f}")
    print(f"  right: knee={kr:+.4f} knee_mid={mr:+.4f}")
    print(f"  left : knee={kl:+.4f} knee_mid={ml:+.4f}")
    print()


if __name__ == "__main__":
  main()
