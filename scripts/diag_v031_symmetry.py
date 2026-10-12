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

from mjlab.asset_zoo.robots.pikachu.pikachu_v031_constants import (
  PIKACHU_V031_XML,
  ZERO_KEYFRAME,
)
from mjlab.tasks.velocity.config.pikachu_v031.symmetry import MIRROR_SIGN

M = np.diag([1.0, -1.0, 1.0])


def world_inertia(m: mujoco.MjModel, body: int) -> np.ndarray:
  """body 在其自身质心处、世界系朝向下的惯量张量。"""
  R = m.body_iquat[body]
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, R)
  rot = rot.reshape(3, 3)
  return rot @ np.diag(m.body_inertia[body]) @ rot.T


def mesh_solid(m: mujoco.MjModel, body_name: str) -> tuple[float, np.ndarray]:
  """把一个 body 的 STL 当**闭合实体**算 (体积, body frame 下的体积质心)。

  用发散定理对三角网格积分：对每个面 (a,b,c) 与原点围成的四面体，
  ``V_tet = a·(b×c)/6``、质心 ``(a+b+c)/4``，累加即可。这里取该 body 面数
  最多的 mesh geom（视觉/碰撞两个 geom 引用同一份 STL，结果相同）。

  为什么要算这个：XML 里的 ``inertial`` 是 CAD 在**精确 B-rep 实体**上算出来的，
  而 STL 只是它的三角化近似，两者天然差亚毫米级；左右两件的三角化误差还不一样，
  这个差会直接叠到第 1 节的镜像残差上。把 STL 实体算出来才能把残差拆成
  "真实几何差异"和"导出误差"两部分。
  """
  bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
  geoms = [
    g
    for g in range(m.ngeom)
    if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
  ]
  g = max(geoms, key=lambda i: m.mesh_facenum[m.geom_dataid[i]])
  mid = m.geom_dataid[g]
  a0, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, m.geom_quat[g])
  v = m.mesh_vert[a0 : a0 + n] @ rot.reshape(3, 3).T + m.geom_pos[g]
  f0, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
  f = m.mesh_face[f0 : f0 + fn]
  a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
  w = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
  vol = float(w.sum())
  return vol, ((a + b + c) * w[:, None]).sum(0) / 4.0 / vol


def main() -> None:
  m = mujoco.MjModel.from_xml_path(str(PIKACHU_V031_XML))
  d = mujoco.MjData(m)
  mujoco.mj_resetData(m, d)
  # 直接编译 XML 不会加载 keyframe（ZERO_KEYFRAME 由 EntityCfg 注入），base 会
  # 停在地面、脚陷到 z<0，第 4 节的站姿体检随之失真。手动抬到零位高度。
  d.qpos[2] = ZERO_KEYFRAME.pos[2]
  d.qpos[3] = 1.0
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
  # 左脚的 AABB 镜像后应该等于右脚。注意 M 只翻 y，所以**只有 y 分量**需要
  # 交换 min/max；x/z 分量的 lo/hi 必须保持原样。写成 ``M @ hiL, M @ loL``
  # 会把 x/z 的上界当下界，凭空造出 ~100 mm 的假残差。
  a, b = M @ loL, M @ hiL
  loL_m, hiL_m = np.minimum(a, b), np.maximum(a, b)
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

  print("\n=== 5. 非零关节角下的 FK 镜像验证 ===")
  # 前面 1~3 节都在 qpos=0（CAD 零位）下比较，而零位时**所有关节角都是 0**，
  # hip_pitch / hip_roll / hip_yaw 的符号差异在零位完全体现不出来 —— 静态量
  # 查不出运动学符号问题。必须在非零配置下验证。
  #
  # 符号表**由本脚本自己用 FK 逐关节测出**，再与 symmetry.MIRROR_SIGN 对照：
  # 拿被测对象自己的表去"验证"它是循环论证。
  joint_names = [
    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)
  ]
  qadr = {n: m.jnt_qposadr[i] for i, n in enumerate(joint_names) if n}

  def fk(assign: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """按给定驱动关节角算 FK；被动关节留在零位（四杆自然闭合）。"""
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    for n, v in assign.items():
      d.qpos[qadr[n]] = v
    mujoco.mj_kinematics(m, d)
    return d.xpos.copy(), d.xquat.copy()

  def mirror_quat(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], q[2], -q[3]])

  def body_off(b: int) -> int:
    nb = name_of_body[b]
    if not nb or not nb.startswith("right_"):
      return -1
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_" + nb[6:])

  def residual(ax, aq, bx, bq) -> float:
    """B 的右腿相对 A 的左腿的镜像残差。位置(mm)与姿态(rad)量纲不同，这里只
    用来比较两个候选符号的相对大小，不作绝对阈值。"""
    err = 0.0
    for b in range(m.nbody):
      lb = body_off(b)
      if lb < 0:
        continue
      dp = float(np.abs(bx[b] - M @ ax[lb]).max()) * 1000.0
      dq = (
        float(
          min(
            np.abs(mirror_quat(aq[lb]) - bq[b]).max(),
            np.abs(mirror_quat(aq[lb]) + bq[b]).max(),
          )
        )
        * 1000.0
      )
      err = max(err, dp, dq)
    return err

  # 只激励驱动关节：被动的 knee / knee_mid 由四杆约束决定，不能自由赋值
  # （随便赋值会让机构散架，残差爆到几百 mm，那是构造错误不是不对称）。
  left_driven = sorted(
    n
    for n in qadr
    if n.startswith("left_") and "knee_joint" not in n and "knee_mid" not in n
  )
  print(f"  {'左腿关节':<26}{'测得':>6}{'表值':>6}{'res(+1)':>11}{'res(-1)':>11}  判定")
  sign_mismatch: list[str] = []
  for ln in left_driven:
    rn = "right_" + ln[5:]
    ax, aq = fk({ln: 0.3})
    res = {}
    for s in (1.0, -1.0):
      bx, bq = fk({rn: s * 0.3})
      res[s] = residual(ax, aq, bx, bq)
    measured = 1.0 if res[1.0] < res[-1.0] else -1.0
    expect = MIRROR_SIGN.get(ln[5:-6], float("nan"))
    if measured != expect:
      sign_mismatch.append(ln)
    print(
      f"  {ln:<26}{measured:>6.0f}{expect:>6.0f}"
      f"{res[1.0]:>11.4f}{res[-1.0]:>11.4f}  "
      f"{'OK' if measured == expect else '**不符**'}"
    )

  rng = np.random.default_rng(20261012)
  worst_p = worst_q = 0.0
  for _ in range(20):
    ql = rng.uniform(-0.4, 0.4, size=len(left_driven))
    ax, aq = fk({ln: float(v) for ln, v in zip(left_driven, ql, strict=True)})
    bx, bq = fk(
      {
        "right_" + ln[5:]: float(v * MIRROR_SIGN.get(ln[5:-6], 1.0))
        for ln, v in zip(left_driven, ql, strict=True)
      }
    )
    for b in range(m.nbody):
      lb = body_off(b)
      if lb < 0:
        continue
      worst_p = max(worst_p, float(np.abs(bx[b] - M @ ax[lb]).max()) * 1000.0)
      worst_q = max(
        worst_q,
        float(
          min(
            np.abs(mirror_quat(aq[lb]) - bq[b]).max(),
            np.abs(mirror_quat(aq[lb]) + bq[b]).max(),
          )
        ),
      )
  print("\n  20 组随机驱动关节配置（±0.4 rad）下的整机镜像残差：")
  print(f"    位置 {worst_p:.9f} mm / 姿态 {worst_q:.3e}")

  print("\n=== 6. inertial 数据与 STL 几何的一致性 ===")
  # 第 1 节量到 hip_yaw_link 的质心镜像残差 0.58 mm。这一节回答它从哪来：
  # 把两个 STL 当实体算出真实几何，就能把残差拆成"几何本来就不同"与
  # "CAD 精确实体 vs STL 三角化"两层。
  vL, cL = mesh_solid(m, "left_hip_yaw_link")
  vR, cR = mesh_solid(m, "right_hip_yaw_link")
  bL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_hip_yaw_link")
  bR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_hip_yaw_link")
  iL, iR = m.body_inertia[bL], m.body_inertia[bR]
  mL, mR = m.body_mass[bL], m.body_mass[bR]

  print(
    f"  实体体积      L={vL * 1e6:.3f}  R={vR * 1e6:.3f} cm^3  ({(vL - vR) / vR * 100:+.3f}%)"
  )
  print(f"  网格体积质心  L={np.array2string(cL, precision=6)}")
  print(f"                R={np.array2string(cR, precision=6)}")
  print(f"  网格质心镜像残差 = {np.linalg.norm(cL - M @ cR) * 1000:.4f} mm")
  print(
    f"  XML ipos 镜像残差 = {np.linalg.norm(m.body_ipos[bL] - M @ m.body_ipos[bR]) * 1000:.4f} mm"
  )
  print(
    f"  网格质心 vs XML ipos 距离：L={np.linalg.norm(cL - m.body_ipos[bL]) * 1000:.4f} mm"
    f"  R={np.linalg.norm(cR - m.body_ipos[bR]) * 1000:.4f} mm"
  )
  # 质量与惯量是否自洽：形状相同时 I ∝ m·L² ∝ V^(5/3)，故 m ∝ V ∝ I^(3/5)。
  # 用**归一化**惯量确认两件形状一致，再用惯量迹反推密度比。
  tL, tR = float(iL.sum()), float(iR.sum())
  print(f"\n  归一化惯量形状  L={np.sort(iL)[::-1] / tL}")
  print(f"                  R={np.sort(iR)[::-1] / tR}")
  print(f"  惯量迹反推密度比 m_L/m_R = {(tL / tR) ** 0.6:.6f}")
  print(f"  体积反推质量比   m_L/m_R = {vL / vR:.6f}")
  print(f"  实际声明质量比   m_L/m_R = {mL / mR:.6f}  ({mL:.6f} / {mR:.6f} kg)")

  print("\n=== 7. 结论 ===")
  worst.sort(reverse=True)
  print("最大的左右差异项（前 5）：")
  for v, label in worst[:5]:
    print(f"  {label:<22}{v:.3f} (mm 或 g)")
  print(
    f"  运动学符号表：{'与 MIRROR_SIGN 一致' if not sign_mismatch else '不符 ' + str(sign_mismatch)}"
  )
  print(
    f"  整机镜像残差：位置 {worst_p:.6f} mm / 姿态 {worst_q:.3e}"
    f"  -> {'严格镜像机构' if worst_p < 1e-6 and worst_q < 1e-6 else '非严格镜像'}"
  )


if __name__ == "__main__":
  main()
