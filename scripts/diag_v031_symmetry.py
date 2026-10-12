"""V031 训练模型左右对称性的数值体检。

回答一个问题：**"走路向一侧倾斜"是不是 MJCF 资产本身的左右不对称造成的？**

做法：在 qpos=0（CAD 零位 = 站姿）下编译 ``pikachu_v031_train.xml``，对每一对
同名 ``left_*`` / ``right_*`` body 比较：

  - 质量、质心（世界系，镜像后）
  - 世界系惯量张量（镜像后）
  - 关节轴（世界系，镜像后）、限位
  - 碰撞 geom 的世界 AABB（镜像后）

镜像面取 ``y = 0``，镜像算子 ``M = diag(1, -1, 1)``。向量镜像 ``v -> M v``，
张量镜像 ``T -> M T M``。

另外单独体检**脚底接触几何**：左右脚底胶囊的世界 AABB、接触点、以及
qpos=0 时的支撑多边形与整机质心投影 —— 这是"站姿是否天然平衡"的直接证据。

用法：
    uv run python scripts/diag_v031_symmetry.py
"""

from __future__ import annotations

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.pikachu.pikachu_v031_constants import PIKACHU_V031_XML

M = np.diag([1.0, -1.0, 1.0])


def world_inertia(m: mujoco.MjModel, body: int) -> np.ndarray:
  """body 在其自身质心处、世界系朝向下的惯量张量。"""
  R = m.body_iquat[body]
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, R)
  rot = rot.reshape(3, 3)
  return rot @ np.diag(m.body_inertia[body]) @ rot.T


def main() -> None:
  m = mujoco.MjModel.from_xml_path(str(PIKACHU_V031_XML))
  d = mujoco.MjData(m)
  mujoco.mj_resetData(m, d)
  mujoco.mj_forward(m, d)

  print(f"模型: {PIKACHU_V031_XML.name}")
  print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")
  print(f"整机质量 = {m.body_mass.sum():.6f} kg")

  name_of_body = {
    b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)
  }
  left_bodies = {
    n[5:]: b for b, n in name_of_body.items() if n and n.startswith("left_")
  }

  print("\n=== 1. body 质量 / 质心 / 惯量（镜像残差） ===")
  print(
    f"{'link':<22}{'mass L':>10}{'mass R':>10}{'dm(g)':>9}"
    f"{'dCOM(mm)':>10}{'dI(rel)':>10}"
  )
  worst: list[tuple[float, str]] = []
  for suffix, lb in sorted(left_bodies.items()):
    rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_" + suffix)
    if rb < 0:
      continue
    ml, mr = m.body_mass[lb], m.body_mass[rb]
    # 世界系质心
    cl = d.xipos[lb].copy()
    cr = d.xipos[rb].copy()
    dcom = np.linalg.norm(cl - M @ cr)
    Il, Ir = world_inertia(m, lb), world_inertia(m, rb)
    Ir_m = M @ Ir @ M
    dI = np.linalg.norm(Il - Ir_m) / max(np.linalg.norm(Il), 1e-12)
    print(
      f"{suffix:<22}{ml:>10.6f}{mr:>10.6f}{(ml - mr) * 1000:>9.2f}"
      f"{dcom * 1000:>10.3f}{dI:>10.4f}"
    )
    worst.append((abs(ml - mr) * 1000, f"mass {suffix}"))
    worst.append((dcom * 1000, f"com {suffix}"))

  print("\n=== 2. 关节轴（世界系）/ 限位 ===")
  print(
    f"{'joint':<24}{'axisL(world)':>26}{'axisR(world) mirror':>28}{'daxis':>9}{'range':>18}"
  )
  for j in range(m.njnt):
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    if not jn or not jn.startswith("left_"):
      continue
    rn = "right_" + jn[5:]
    jr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, rn)
    if jr < 0:
      continue
    al = d.xaxis[j].copy()
    ar = d.xaxis[jr].copy()
    dar = np.linalg.norm(al - M @ ar)
    rl = m.jnt_range[j]
    rr = m.jnt_range[jr]
    flag = "  <-- 轴非镜像" if dar > 1e-6 else ""
    print(
      f"{jn:<24}{np.array2string(al, precision=4):>26}"
      f"{np.array2string(M @ ar, precision=4):>28}{dar:>9.2e}"
      f"{np.array2string(rl, precision=3):>18}{flag}"
    )
    if not np.allclose(rl, rr):
      print(f"    !! 限位不等: L={rl} R={rr}")

  print("\n=== 3. 脚底碰撞胶囊世界 AABB ===")
  for side in ("left", "right"):
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    n = 0
    for g in range(m.ngeom):
      gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
      if not gn.startswith(f"{side}_foot") or "collision" not in gn:
        continue
      n += 1
      gtype, size, pos, mat = (
        m.geom_type[g],
        m.geom_size[g],
        d.geom_xpos[g],
        d.geom_xmat[g],
      )
      if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        half = size[1]
        axis = mat.reshape(3, 3)[:, 2]
        p1, p2 = pos - axis * half, pos + axis * half
        r = size[0]
        for p in (p1, p2):
          lo = np.minimum(lo, p - r)
          hi = np.maximum(hi, p + r)
      else:
        lo = np.minimum(lo, pos - size[:3])
        hi = np.maximum(hi, pos + size[:3])
    print(f"{side:<6} n_geom={n}  lo={np.round(lo, 5)}  hi={np.round(hi, 5)}")

  # 左右脚底镜像残差
  def foot_aabb(side: str) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(m.ngeom):
      gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
      if not gn.startswith(f"{side}_foot") or "collision" not in gn:
        continue
      size, pos, mat = m.geom_size[g], d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
      half = size[1]
      axis = mat[:, 2]
      r = size[0]
      for p in (pos - axis * half, pos + axis * half):
        lo = np.minimum(lo, p - r)
        hi = np.maximum(hi, p + r)
    return lo, hi

  loL, hiL = foot_aabb("left")
  loR, hiR = foot_aabb("right")
  # 左脚的 AABB 镜像后应该等于右脚
  loL_m, hiL_m = M @ hiL, M @ loL  # 镜像会交换 min/max
  print(f"\n左脚镜像 AABB lo={np.round(loL_m, 5)} hi={np.round(hiL_m, 5)}")
  print(f"右脚      AABB lo={np.round(loR, 5)} hi={np.round(hiR, 5)}")
  print(
    f"脚底镜像残差 max = {np.max(np.abs(loL_m - loR)) * 1000:.3f} mm / "
    f"{np.max(np.abs(hiL_m - hiR)) * 1000:.3f} mm"
  )

  print("\n=== 4. qpos=0 站姿体检 ===")
  print(f"base z = {d.qpos[2]:.5f} m")
  com = d.subtree_com[0]
  print(f"整机质心(world) = {np.round(com, 5)}")
  print(
    f"左脚心 x = {d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'left_foot')]}"
  )
  print(
    f"右脚心 x = {d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'right_foot')]}"
  )
  sup = np.array([[loL[0], hiL[0]], [loR[0], hiR[0]]])
  print(
    f"支撑多边形 x 范围 = [{sup.min():.5f}, {sup.max():.5f}], "
    f"质心 x = {com[0]:.5f} (余量 {min(com[0] - sup.min(), sup.max() - com[0]) * 1000:.2f} mm)"
  )
  print(f"质心 y = {com[1]:.6f} m (左右对称应为 0)")

  print("\n=== 5. 结论 ===")
  worst.sort(reverse=True)
  print("最大的左右差异项（前 5）：")
  for v, label in worst[:5]:
    print(f"  {label:<22}{v:.3f} (mm 或 g)")


if __name__ == "__main__":
  main()
