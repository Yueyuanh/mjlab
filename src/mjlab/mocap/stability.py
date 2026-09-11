"""实时稳定性分析：CoM / 惯量 / 足底接触面积 / ZMP 的可复用计算与绘制。

本模块把「重心 + 惯量 + 支撑多边形 + ZMP」这组稳定性量算出来，供两处复用：

  - ``play --com``：速度快，每帧叠加到 native / viser 查看器（见
    :func:`draw_stability`，两套查看器共用同一个 ``DebugVisualizer`` 接口）。
  - ``scripts/analyze_com_zmp.py``：离线逐帧算 CoM+ZMP 并导出 CSV。

**计算口径**（与 ``mocap/analyze_stability.py`` 一致）：

  - CoM 取 MuJoCo 的 ``subtree_com[0]`` / 逐 body 世界质心累加（整机质心）。
  - 绕 CoM 的惯量张量：用每 body 的 ``cinert``（绕自身质心惯性，存于
    ``body_iquat`` 主帧）+ 平行轴定理合成到整机。``cinert`` 内存序为
    ``[llz lly llx lrx lry lrz]``，对应对称阵
    ``[[llx,-lrx,-lry],[-lrx,lly,-lrz],[-lry,-lrz,llz]]``（离对角带负号）。
  - 足底接触面积：按足部 geom 的世界几何判断"着地"（最低点贴近地面），
    把着地足的足盒画在地面上作为支撑多边形。

**性能**：质量/惯性都是模型静态量（读取一次宿主 ``MjModel``）；每帧只从
GPU 拉当前环境的一行 ``xpos/xquat`` 与足 geom 姿态，量级很小，60Hz 无压力。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StabilityData:
  """一次采样的稳定性量。

  Attributes:
    com: 整机质心（世界系，(3,)）。
    inertia: 绕整机 CoM 的 3×3 惯量张量（世界系）。
    principal_moments: 主惯量，大到小。
    principal_axes: 对应主轴（3×3，行为主轴方向）。
    com_proj: CoM 在地面的投影 (x, y, ground_z)。
    foot_boxes: 着地脚的足盒世界几何；每项 (center(3), size(3), mat(3,3))。
    ground_z: 地面（脚底接触平面）高度。
  """

  com: np.ndarray
  inertia: np.ndarray
  principal_moments: np.ndarray
  principal_axes: np.ndarray
  com_proj: np.ndarray
  foot_boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
  ground_z: float


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
  """四元数 (w,x,y,z) → 旋转矩阵 R（把局部向量转到目标系）。"""
  q = q / np.linalg.norm(q)
  w, x, y, z = q
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )


def _cinert_to_tensor(cinert: np.ndarray) -> np.ndarray:
  """把一根 body 的 cinert 6-分量展开成 3×3 惯量张量（body_iquat 主帧）。"""
  llz, lly, llx, lrx, lry, lrz = cinert
  return np.array(
    [
      [llx, -lrx, -lry],
      [-lrx, lly, -lrz],
      [-lry, -lrz, llz],
    ]
  )


def _host_cinert(model: Any) -> np.ndarray:
  """由宿主 ``body_inertia``（主帧对角）合成 warp 风格的 cinert 6-分量。

  warp ``cinert`` 内存序为 ``[llz lly llx lrx lry lrz]``，对应主帧下离对角为 0，
  故 ``[Iz Iy Ix 0 0 0]``。宿主 ``MjModel`` 没有 ``cinert`` 字段（只有
  principal 帧对角 ``body_inertia``），在线 / 离线路径都用它喂
  :func:`inertia_about_com`。
  """
  bi = np.asarray(model.body_inertia, dtype=np.float64)
  cin = np.zeros((bi.shape[0], 6))
  cin[:, 0] = bi[:, 2]  # llz = Iz
  cin[:, 1] = bi[:, 1]  # lly = Iy
  cin[:, 2] = bi[:, 0]  # llx = Ix
  return cin


def _host_geom_name(model: Any, gid: int) -> str:
  """取宿主 ``MjModel`` 第 ``gid`` 个 geom 的名称（无后缀，'' 表示无名）。

  宿主 ``MjModel`` 没有 ``geom_names``；名称存在 ``names`` 缓冲里，按
  ``name_geomadr[gid]`` 定位到 NUL 结尾。跨 mujoco 版本通用。
  """
  if gid < 0 or gid >= int(getattr(model, "ngeom", 0)):
    return ""
  adr = int(model.name_geomadr[gid])
  if adr < 0:
    return ""
  buf = model.names
  end = buf.find(b"\x00", adr)
  if end < 0:
    end = len(buf)
  try:
    return buf[adr:end].decode()
  except UnicodeDecodeError:
    return ""


def inertia_about_com(
  mass: np.ndarray,
  cinert: np.ndarray,
  body_ipos: np.ndarray,
  body_iquat: np.ndarray,
  xpos: np.ndarray,
  xquat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """由逐 body 的质量/惯量与机体世界位姿，合成整机 CoM 与绕其惯量张量。

  Args:
    mass: (nbody,) 每 body 质量。
    cinert: (nbody,6) 每 body 绕自身质心的 com 惯性（``MjModel.cinert``）。
    body_ipos: (nbody,3) 质心在 body 系的位置（``MjModel.body_ipos``）。
    body_iquat: (nbody,4) 主帧相对 body 系朝向（wxyz，``MjModel.body_iquat``）。
    xpos: (nbody,3) 每 body 世界原点位置（``MjData.xpos``）。
    xquat: (nbody,4) 每 body 世界朝向（wxyz，``MjData.xquat``）。

  Returns:
    ``(com, I)``：整机质心 (3,)；绕质心的世界系 3×3 惯量张量。
  """
  total_mass = float(np.asarray(mass).sum())
  com = np.zeros(3)
  I_total = np.zeros((3, 3))
  for b in range(len(mass)):
    R_b = _quat_to_rot(np.asarray(xquat[b]))
    R_p = _quat_to_rot(np.asarray(body_iquat[b]))
    c_b = np.asarray(xpos[b]) + R_b @ np.asarray(body_ipos[b])  # 世界质心
    I_p = _cinert_to_tensor(np.asarray(cinert[b]))  # 主帧自身惯量
    R_w_from_p = R_b @ R_p.T  # 主帧 → 世界
    com = com + float(mass[b]) * c_b
    I_total = I_total + R_w_from_p @ I_p @ R_w_from_p.T

  com = com / total_mass
  # 平行轴：把每 body 自身惯量（关于各自质心）换算成关于整机质心。
  for b in range(len(mass)):
    d = (
      np.asarray(xpos[b])
      + _quat_to_rot(np.asarray(xquat[b])) @ np.asarray(body_ipos[b])
      - com
    )
    I_total = I_total + float(mass[b]) * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
  return com, np.asarray(I_total)


def com_and_inertia_from_sim(
  sim: Any, env_idx: int, ground_z: float = 0.0
) -> StabilityData:
  """从 warp 仿真 ``sim`` 取第 ``env_idx`` 个环境，算 CoM/惯量/足底接触。

  Args:
    sim: ``mjlab.sim.Simulation``。
    env_idx: 选中的环境索引。
    ground_z: 地面高度；<=0 时自动取当前足底最低点。
  """
  xpos = np.asarray(sim.data.xpos[env_idx].cpu().numpy(), dtype=np.float64)
  xquat = np.asarray(sim.data.xquat[env_idx].cpu().numpy(), dtype=np.float64)
  m = sim.mj_model
  com, I_total = inertia_about_com(
    mass=m.body_mass,
    cinert=_host_cinert(m),
    body_ipos=m.body_ipos,
    body_iquat=m.body_iquat,
    xpos=xpos,
    xquat=xquat,
  )
  moments, axes = np.linalg.eigh(I_total)
  order = np.argsort(moments)[::-1]
  # 足底接触：检测着地足盒，自动标定地面高度。
  foot_boxes, auto_ground = _planted_foot_boxes(sim, env_idx)
  gz = ground_z if ground_z > 0.0 else auto_ground
  if gz == 0.0 and auto_ground != 0.0:
    gz = auto_ground
  proj = np.array([com[0], com[1], gz], dtype=np.float64)
  return StabilityData(
    com=com,
    inertia=I_total,
    principal_moments=moments[order],
    principal_axes=axes[:, order].T,
    com_proj=proj,
    foot_boxes=foot_boxes,
    ground_z=gz,
  )


def _planted_foot_boxes(
  sim: Any, env_idx: int
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], float]:
  """返回 (着地足的足盒世界几何, 自动标定的地面高度)。

  筛选 geom 名含 ``ankle``/``foot`` 的 box，用其世界姿态算足盒底面四角；
  最低底面 z 视为地面。着地（足底贴近地面）的足盒返回给支撑多边形。
  """
  m = sim.mj_model
  # 用名称匹配足部 geom（宿主 MjModel 无 geom_names，从 names 缓冲按
  # name_geomadr 切片，仿 mujoco 官方取值约定）。
  foot_ids = [
    g
    for g in range(m.ngeom)
    if "ankle" in _host_geom_name(m, g) or "foot" in _host_geom_name(m, g)
  ]
  if not foot_ids:
    return [], 0.0

  gx = np.asarray(sim.data.geom_xpos[env_idx].cpu().numpy(), dtype=np.float64)
  gmat = np.asarray(sim.data.geom_xmat[env_idx].cpu().numpy(), dtype=np.float64)
  half = np.asarray(m.geom_size, dtype=np.float64)  # (ngeom,3)

  boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
  bottoms: list[float] = []
  for g in foot_ids:
    hx, hy, hz = half[g]
    if hz <= 0:
      continue
    R = gmat[g]  # (3,3) 世界系
    cen = gx[g]
    # 底面四角（局部 -hz 面）→ 世界。
    corners = []
    for sx in (hx, -hx):
      for sy in (hy, -hy):
        corners.append(cen + R @ np.array([sx, sy, -hz]))
    min_z = min(c[2] for c in corners)
    bottoms.append(min_z)
    boxes.append((np.array(cen), np.array([hx, hy, hz]), np.array(R)))
  gz = min(bottoms) if bottoms else 0.0
  # 着地判定：足盒底面最低点贴近地面 gz（< 2cm）。
  planted = [
    (cen, size, mat)
    for (cen, size, mat), bz in zip(boxes, bottoms, strict=True)
    if bz - gz < 0.02
  ]
  return planted, gz


def zmp_from_com(com_traj: np.ndarray, dt: float, g: float = 9.81) -> np.ndarray:
  """由整机 CoM 轨迹（(T,3)）用线性倒立摆近似求水平 ZMP ((T,2))。

  ``zmp_xy = com_xy - (com_z / g) * com_acc_xy``。加速度用二阶中心差分。
  """
  com = np.asarray(com_traj, dtype=np.float64)
  if com.ndim == 1:
    com = com[None, :]
  vel = np.gradient(com[:, :2], axis=0) / dt
  acc = np.gradient(vel, axis=0) / dt
  h = com[:, 2]
  return com[:, :2] - (h / g)[:, None] * acc


def _support_hull(points: np.ndarray) -> np.ndarray:
  """把足盒底面四角投影到地面(z=0)并取凸包（Andrew 单调链），返回顶点 (N,2)。"""
  xy = np.asarray(points, dtype=np.float64)[:, :2]
  if len(xy) < 3:
    return xy
  order = np.lexsort((xy[:, 1], xy[:, 0]))
  pts = xy[order]
  # 去重。
  keep = np.concatenate(([True], np.any(np.diff(pts, axis=0) != 0, axis=1)))
  pts = pts[keep]
  if len(pts) < 3:
    return pts

  def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

  lower: list[np.ndarray] = []
  for p in pts:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
      lower.pop()
    lower.append(p)
  upper: list[np.ndarray] = []
  for p in reversed(pts):
    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
      upper.pop()
    upper.append(p)
  return np.array(lower[:-1] + upper[:-1])


def draw_stability(scene: Any, data: StabilityData) -> None:
  """把稳定量画到 ``DebugVisualizer`` 接口（native / viser 共用）。

  Args:
    scene: 实现了 ``DebugVisualizer`` 的查看器场景（ignore env_idx 差异）。
    data: 由 :func:`com_and_inertia_from_sim` 算出的当前稳定性量。
  """
  com = data.com
  proj = data.com_proj
  gz = proj[2]

  # CoM 绿点。
  scene.add_sphere(com, radius=0.013, color=(0.15, 0.9, 0.3, 1.0))
  # CoM → 地面投影 黄竖线。
  scene.add_cylinder(
    com, np.array([com[0], com[1], gz]), radius=0.008, color=(0.95, 0.85, 0.2, 1.0)
  )
  # 地面投影红点。
  scene.add_sphere(proj, radius=0.013, color=(0.95, 0.2, 0.2, 1.0))

  # 绕 CoM 的惯量主轴椭球（半轴长 ∝ 主惯量开方，缩放到可见尺度）。
  moments = data.principal_moments
  axes = data.principal_axes  # 行为主轴
  scale = 0.06 * np.sqrt(moments / moments[0])
  scene.add_ellipsoid(
    center=com,
    size=scale,
    mat=axes.T,  # 列为主轴方向
    color=(0.3, 0.55, 0.95, 0.18),
  )
  # 主轴短线。
  for k in range(3):
    scene.add_cylinder(
      com, com + axes[k] * (scale[k] + 0.02), radius=0.006, color=(0.3, 0.55, 0.95, 0.9)
    )

  # 支撑多边形：着地足盒（半透明）围成的地面接触区 + 凸包连线。
  if data.foot_boxes:
    sizes = data.foot_boxes
    for cen, size, mat in sizes:
      scene.add_box(
        center=np.array([cen[0], cen[1], gz + 0.005]),
        size=np.array([size[0], size[1], 0.008]),
        mat=mat,
        color=(0.6, 0.9, 0.4, 0.25),
      )
    corners = []
    for cen, size, mat in sizes:
      hx, hy, _ = size
      for sx in (hx, -hx):
        for sy in (hy, -hy):
          corners.append(
            np.array([cen[0], cen[1], 0.0]) + mat @ np.array([sx, sy, 0.0])
          )
    hull = _support_hull(np.array(corners))
    n = len(hull)
    for i in range(n):
      a = np.array([hull[i][0], hull[i][1], gz + 0.01])
      b = np.array([hull[(i + 1) % n][0], hull[(i + 1) % n][1], gz + 0.01])
      scene.add_cylinder(a, b, radius=0.006, color=(0.15, 0.85, 0.25, 1.0))
