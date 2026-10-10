"""标定 pikachu_v031 的站立 keyframe（左右双腿独立解），并给出镜像关系。

v031 的关节轴与 v03 **完全不同**，v03 的 keyframe 数值与符号关系都不能照抄：

    关节          右轴        左轴        左右关系（实测）
    hip_pitch    0 1 0       0 -1 0      **反号**
    hip_roll     1 0 0       1  0 0      同号
    hip_yaw      0 0 1       0  0 1      同号
    knee         0 1 0       0 -1 0      **反号**（被动）
    ankle        0 -1 0      0  1 0      **反号**
    knee_up      0 -1 0      0 -1 0      同号
    knee_mid     0 -1 0      0 -1 0      同号（被动）

（v03 是「髋三轴+膝同号，踝/knee_up/knee_mid 反号」，正好互补。）

两个必须注意的点：

1. **源模型没有 ``<default>``**，joint damping / frictionloss / armature 全是 0。
   本脚本按训练模型的设定补齐（damping=0.3、frictionloss=0.01），并按 mjlab 的
   actuator 配置补 armature。不补的话位置伺服会发散（实测残差 0.5 rad 不收敛）。

2. **四杆膝在 knee_up≈0 附近有第二个分支**。左腿从 (0, 0) 出发求解会落到
   「左膝与右膝同向弯」的伪解上，它的脚掌同样水平、高度也一样。本脚本对左腿
   用右腿的镜像值做初值，并在最后核验 ``knee_L/knee_R ≈ -1``（真镜像）。

用法：
    uv run python scripts/probe_v031_keyframe.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

XML_DIR = (
  Path(__file__).resolve().parent.parent
  / "src"
  / "mjlab"
  / "asset_zoo"
  / "robots"
  / "pikachu"
  / "xmls"
  / "pikachu_v031"
)
SRC_XML = XML_DIR / "mjcf" / "pikachu_v031.xml"
TRAIN_XML = XML_DIR / "mjcf" / "pikachu_v031_train.xml"

KP, KD, KI = 20.0, 0.5, 50.0
DT = 5e-4
SETTLE_STEPS = 3000
CONVERGE_TOL = 1e-7
NEWTON_ITERS = 8
# Jacobian 变化很慢，不必每步重算（每步重算是 3 倍开销）。
JACOBIAN_REFRESH = 3

# mjlab actuator 配置里的 reflected inertia（与 pikachu_v031_constants.py 一致）。
ROTOR_J4310, GEARS_J4310 = 2.206e-5, 10.0
ROTOR_J3507, GEARS_J3507 = 6.69169e-6, 7.0 * 3.0

# 每条腿 5 个驱动关节。
DRIVEN = (
  "hip_pitch_joint",
  "hip_roll_joint",
  "hip_yaw_joint",
  "knee_up_joint",
  "ankle_joint",
)


class Rig:
  """源模型（base 焊死）+ 训练模型等效的关节属性，用于位置伺服标定。"""

  def __init__(self) -> None:
    self.m = mujoco.MjModel.from_xml_path(str(SRC_XML))
    self.d = mujoco.MjData(self.m)
    self.m.opt.gravity[:] = 0.0
    # 训练模型（build_v031_train_xml.py）补的 <default>。
    self.m.dof_damping[:] = 0.3
    self.m.dof_frictionloss[:] = 0.01
    self.names = [
      mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)
      for j in range(self.m.njnt)
    ]
    jid = {n: i for i, n in enumerate(self.names)}
    self.qadr = {n: self.m.jnt_qposadr[jid[n]] for n in self.names}
    self.vadr = {n: self.m.jnt_dofadr[jid[n]] for n in self.names}
    for n in self.names:
      a = ROTOR_J3507 * GEARS_J3507 if "ankle" in n else ROTOR_J4310 * GEARS_J4310
      self.m.dof_armature[self.vadr[n]] = a
    self.aid = {self.names[self.m.actuator_trnid[a, 0]]: a for a in range(self.m.nu)}
    self.joints = [f"{s}_{k}" for s in ("right", "left") for k in DRIVEN]
    self.targets: dict[str, float] = dict.fromkeys(self.joints, 0.0)
    self._integ = dict.fromkeys(self.joints, 0.0)
    self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    # 每个 ankle body 的 mesh 顶点（body 系），用于量鞋底高度 / 俯仰 / 脚心 x。
    self.feet: dict[str, tuple[int, np.ndarray]] = {}
    for side in ("right", "left"):
      b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_link")
      g = next(
        g
        for g in range(self.m.ngeom)
        if self.m.geom_bodyid[g] == b
        and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
      )
      did = self.m.geom_dataid[g]
      va, vn = self.m.mesh_vertadr[did], self.m.mesh_vertnum[did]
      rot = np.zeros(9)
      mujoco.mju_quat2Mat(rot, self.m.geom_quat[g])
      verts = (
        rot.reshape(3, 3) @ self.m.mesh_vert[va : va + vn].reshape(-1, 3).T
      ).T + self.m.geom_pos[g]
      self.feet[side] = (b, verts)

  def reset(self) -> None:
    self.d.qpos[:] = 0.0
    self.d.qvel[:] = 0.0
    self._integ = dict.fromkeys(self.joints, 0.0)

  def settle(self, steps: int = SETTLE_STEPS) -> float:
    """位置伺服推进到稳态。**不重置状态**（续跑收敛更快）。"""
    d, m = self.d, self.m
    for s in range(steps):
      for j in self.joints:
        e = self.targets[j] - d.qpos[self.qadr[j]]
        self._integ[j] = float(np.clip(self._integ[j] + e * DT, -0.4, 0.4))
        d.ctrl[self.aid[j]] = KP * e - KD * d.qvel[self.vadr[j]] + KI * self._integ[j]
      mujoco.mj_step(m, d)
      if s % 250 == 249 and self._err() < CONVERGE_TOL:
        break
    for j in self.joints:
      d.ctrl[self.aid[j]] = 0.0
    mujoco.mj_forward(m, d)
    return self._err()

  def _err(self) -> float:
    return max(abs(self.d.qpos[self.qadr[j]] - self.targets[j]) for j in self.joints)

  def measure(self, side: str) -> tuple[float, float, float]:
    """(最低鞋底 z, 脚掌俯仰 rad, 脚心 x)。俯仰用 body 系 x 轴。"""
    b, verts = self.feet[side]
    R = self.d.xmat[b].reshape(3, 3)
    w = (R @ verts.T).T + self.d.xpos[b]
    xa = R @ np.array([1.0, 0.0, 0.0])
    return (
      float(w[:, 2].min()),
      float(np.arctan2(xa[2], np.hypot(xa[0], xa[1]))),
      float(w[:, 0].mean()),
    )

  def com_x(self) -> float:
    return float(self.d.subtree_com[self.base][0])

  def q(self, name: str) -> float:
    return float(self.d.qpos[self.qadr[name]])


def _set_leg(rig: Rig, side: str, k: float, hip: float, ank: float) -> None:
  rig.targets[f"{side}_knee_up_joint"] = k
  rig.targets[f"{side}_hip_pitch_joint"] = hip
  rig.targets[f"{side}_ankle_joint"] = ank


def solve_leg(
  rig: Rig, side: str, k: float, cx: float, hip: float, ank: float
) -> tuple[float, float]:
  """固定 knee_up=k，解出使「脚掌水平 + 脚心 x=cx」的 (hip_pitch, ankle)。"""
  J: np.ndarray | None = None
  for it in range(NEWTON_ITERS):
    _set_leg(rig, side, k, hip, ank)
    rig.settle()
    _, pitch, cx_now = rig.measure(side)
    if abs(pitch) < 1e-12 and abs(cx_now - cx) < 1e-12:
      break
    if J is None or it % JACOBIAN_REFRESH == 0:
      e = 1e-3
      _set_leg(rig, side, k, hip + e, ank)
      rig.settle()
      _, ph, ch = rig.measure(side)
      _set_leg(rig, side, k, hip, ank + e)
      rig.settle()
      _, pa, ca = rig.measure(side)
      J = np.array(
        [[(ph - pitch) / e, (pa - pitch) / e], [(ch - cx_now) / e, (ca - cx_now) / e]]
      )
    try:
      step = np.linalg.solve(J, np.array([pitch, cx_now - cx]))
    except np.linalg.LinAlgError:
      break
    hip += float(np.clip(-step[0], -0.15, 0.15))
    ank += float(np.clip(-step[1], -0.15, 0.15))
  _set_leg(rig, side, k, hip, ank)
  rig.settle()
  return hip, ank


def solve_stance(rig: Rig, k: float, cx: float = 0.0) -> dict[str, float]:
  """解出双腿站姿：右腿从零起解，左腿以右腿镜像做初值（避开伪解分支）。"""
  hip_r, ank_r = solve_leg(rig, "right", k, cx, -k / 2.0, -k / 2.0)
  hip_l, ank_l = solve_leg(rig, "left", k, cx, -hip_r, -ank_r)
  rig.targets.update(
    {
      "right_hip_pitch_joint": hip_r,
      "right_knee_up_joint": k,
      "right_ankle_joint": ank_r,
      "left_hip_pitch_joint": hip_l,
      "left_knee_up_joint": k,
      "left_ankle_joint": ank_l,
    }
  )
  rig.settle()
  lo_r, _, cx_r = rig.measure("right")
  lo_l, _, cx_l = rig.measure("left")
  return {
    "knee_up": k,
    "hip_r": hip_r,
    "ank_r": ank_r,
    "hip_l": hip_l,
    "ank_l": ank_l,
    "knee_r": rig.q("right_knee_joint"),
    "knee_l": rig.q("left_knee_joint"),
    "mid_r": rig.q("right_knee_mid_joint"),
    "mid_l": rig.q("left_knee_mid_joint"),
    "base_z": -min(lo_r, lo_l),
    "com_x": rig.com_x(),
    "foot_x_r": cx_r,
    "foot_x_l": cx_l,
    "err": rig._err(),
  }


def refine(rig: Rig, k: float, iters: int = 8) -> dict[str, float]:
  """外层次固定点：把脚心解到质心正下方。"""
  cx = 0.0
  out = solve_stance(rig, k, cx)
  for i in range(iters):
    print(
      f"    iter {i}  cx_target={cx:+.5f}  com_x={out['com_x']:+.5f}  "
      f"foot_x={out['foot_x_r']:+.5f}/{out['foot_x_l']:+.5f}"
    )
    if abs(out["com_x"] - cx) < 1e-6:
      break
    cx = out["com_x"]
    out = solve_stance(rig, k, cx)
  out["cx"] = cx
  return out


def train_model_base_z(pose: dict[str, float]) -> tuple[float, float]:
  """在训练模型（freejoint 版）里复核：给定姿态下鞋底最低点与质心 x。"""
  tm = mujoco.MjModel.from_xml_path(str(TRAIN_XML))
  td = mujoco.MjData(tm)
  for jn, v in pose.items():
    j = mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_JOINT, jn)
    td.qpos[tm.jnt_qposadr[j]] = v
  mujoco.mj_forward(tm, td)
  sole_z: list[float] = []
  for g in range(tm.ngeom):
    name = mujoco.mj_id2name(tm, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if "ankle" not in name or tm.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    did = tm.geom_dataid[g]
    va, vn = tm.mesh_vertadr[did], tm.mesh_vertnum[did]
    v = tm.mesh_vert[va : va + vn].reshape(-1, 3)
    R = td.geom_xmat[g].reshape(3, 3)
    sole_z.append(float(((R @ v.T).T + td.geom_xpos[g])[:, 2].min()))
  base = mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  return -min(sole_z), float(td.subtree_com[base][0])


def keyframe_pose(out: dict[str, float]) -> dict[str, float]:
  return {
    "right_hip_pitch_joint": out["hip_r"],
    "left_hip_pitch_joint": out["hip_l"],
    "right_hip_roll_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_knee_up_joint": out["knee_up"],
    "left_knee_up_joint": out["knee_up"],
    "right_ankle_joint": out["ank_r"],
    "left_ankle_joint": out["ank_l"],
    "right_knee_joint": out["knee_r"],
    "left_knee_joint": out["knee_l"],
    "right_knee_mid_joint": out["mid_r"],
    "left_knee_mid_joint": out["mid_l"],
  }


def main() -> None:
  rig = Rig()
  m = rig.m
  jid = {n: i for i, n in enumerate(rig.names)}
  print("=== 关节轴（body 系）与左右镜像判定 ===")
  print(f"{'关节':<10} {'右轴':>16} {'左轴':>16} {'M(右轴)':>16}  L==M(R)?  关系")
  for b in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle", "knee_up", "knee_mid"):
    ar = m.jnt_axis[jid[f"right_{b}_joint"]]
    al = m.jnt_axis[jid[f"left_{b}_joint"]]
    mir = np.array([1.0, -1.0, 1.0]) * ar
    same = bool(np.allclose(al, mir, atol=1e-3))
    print(
      f"{b:<10} {str(np.round(ar, 4)):>16} {str(np.round(al, 4)):>16} "
      f"{str(np.round(mir, 4)):>16}  {str(same):>7}  "
      f"{'左=右' if not same else '左=-右'}"
    )

  print()
  print("=== 站姿扫描（脚掌水平 + 脚心在 base 原点正下方）===")
  print(
    f"{'knee_up':>8} {'hip_R':>8} {'ank_R':>8} {'knee_R':>8} {'mid_R':>8} | "
    f"{'hip_L':>8} {'ank_L':>8} {'knee_L':>8} {'mid_L':>8} | "
    f"{'base_z':>8} {'com_x':>8}  err   knee_L/R  mid_L/R"
  )
  rows: dict[float, dict[str, float]] = {}
  for k in (0.0, 0.15, 0.3, 0.45, 0.6, -0.2, -0.4, -0.65):
    rig.reset()
    out = solve_stance(rig, k)
    rows[k] = out

    def rat(a: float, b: float) -> str:
      return f"{a / b:+.4f}" if abs(b) > 1e-4 else "n/a"

    print(
      f"{k:>8.2f} {out['hip_r']:>8.4f} {out['ank_r']:>8.4f} {out['knee_r']:>8.4f} "
      f"{out['mid_r']:>8.4f} | {out['hip_l']:>8.4f} {out['ank_l']:>8.4f} "
      f"{out['knee_l']:>8.4f} {out['mid_l']:>8.4f} | "
      f"{out['base_z']:>8.5f} {out['com_x']:>8.5f}  {out['err']:.0e}  "
      f"{rat(out['knee_l'], out['knee_r']):>8} {rat(out['mid_l'], out['mid_r']):>8}"
    )

  print()
  print("=== 最终 keyframe（脚心对齐质心，固定点迭代）===")
  final: dict[str, dict[str, float]] = {}
  for k, tag in ((0.0, "STAND"), (0.45, "KNEES_BENT")):
    rig.reset()
    print(f"  [{tag}] knee_up={k:+.2f}")
    out = refine(rig, k)
    final[tag] = out
    pose = keyframe_pose(out)
    bz_train, com_train = train_model_base_z(pose)
    print(f"  [{tag}] 源模型 base_z={out['base_z']:.5f} com_x={out['com_x']:+.5f}")
    print(
      f"  [{tag}] 训练模型 base_z={bz_train:.5f} com_x={com_train:+.5f} "
      f"（差 {abs(bz_train - out['base_z']) * 1000:.2f} mm）"
    )
    print(f"  [{tag}] pos=(0, 0, {bz_train:.5f})")
    for jn, v in pose.items():
      print(f"    {jn:26s}= {v:+.4f},")
    print()

  print("=== 行程安全性 ===")
  for k, out in rows.items():
    print(
      f"  knee_up={k:+.2f}: knee_R={out['knee_r']:+.4f} "
      f"knee_L={out['knee_l']:+.4f} mid_R={out['mid_r']:+.4f} "
      f"mid_L={out['mid_l']:+.4f}  base_z={out['base_z']:.5f}"
    )


if __name__ == "__main__":
  main()
