"""足底遥测的 3D 叠加层（play 时的 viser 观测）。

对应 ``pikachu_foot_telemetry`` 里那批 metric 列中**没法用折线图表达**的部分：

  - 足底 taxel 阵列热力图（压力分布、有效接触面积）—— 用户观测项 3；
  - CoP 与支撑多边形（支撑边界）—— 观测项 4；
  - 足底接触力/冲击力箭头 —— 观测项 5；
  - 摩擦锥（法向 + 摩擦角母线）—— 观测项 8；
  - 穿透量（脚陷进地面的深度）—— 观测项 9。

占空比/相位/横滚/摩擦系数这些**时序**量走 ``ViserTermOverlays`` 的 Metrics
标签页（metric 注册后自动出现），本模块只管几何。

与 ``ViserCoMOverlays`` 复用同一套生命周期：每帧往 ``scene`` 的 debug 队列里
追加图元，由 ``scene`` 的 ``_sync_debug_visualizations`` 统一重建，所以不会残留。
需要 viser 里勾上 "Debug visualization"（``scene.debug_visualization_enabled``），
否则 ``add_*`` 会静默丢弃。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import viser

_SIDES = ("left", "right")

# 压力色标（蓝 → 青 → 绿 → 黄 → 红）的锚点。
_RAMP = (
  (0.00, (0.13, 0.22, 0.55)),
  (0.25, (0.10, 0.60, 0.90)),
  (0.50, (0.18, 0.80, 0.32)),
  (0.75, (0.95, 0.80, 0.15)),
  (1.00, (0.90, 0.15, 0.10)),
)
_NO_CONTACT_RGBA = (0.30, 0.30, 0.32, 0.55)


class _EnvProtocol(Protocol):
  @property
  def unwrapped(self) -> Any: ...


class _SceneProtocol(Protocol):
  env_idx: int
  debug_visualization_enabled: bool

  def add_arrow(
    self,
    start: Any,
    end: Any,
    color: tuple[float, float, float, float],
    width: float = ...,
    label: str | None = ...,
  ) -> None: ...
  def add_sphere(
    self,
    center: Any,
    radius: float,
    color: tuple[float, float, float, float],
    label: str | None = ...,
  ) -> None: ...
  def add_cylinder(
    self,
    start: Any,
    end: Any,
    radius: float,
    color: tuple[float, float, float, float],
    label: str | None = ...,
  ) -> None: ...
  def clear_debug_all(self) -> None: ...


def _ramp_color(t: float) -> tuple[float, float, float, float]:
  """压力归一值 → RGBA（分段线性插值）。"""
  t = float(np.clip(t, 0.0, 1.0))
  for (t0, c0), (t1, c1) in zip(_RAMP, _RAMP[1:], strict=False):
    if t <= t1:
      f = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
      return (
        c0[0] + (c1[0] - c0[0]) * f,
        c0[1] + (c1[1] - c0[1]) * f,
        c0[2] + (c1[2] - c0[2]) * f,
        0.95,
      )
  c = _RAMP[-1][1]
  return (c[0], c[1], c[2], 0.95)


def _convex_hull(points: np.ndarray) -> np.ndarray:
  """Andrew monotone chain：逆时针凸包顶点（去掉共线点）。"""
  pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
  if len(pts) < 3:
    return pts
  pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

  def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

  lower: list[np.ndarray] = []
  for p in pts:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
      lower.pop()
    lower.append(p)
  upper: list[np.ndarray] = []
  for p in reversed(pts):
    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
      upper.pop()
    upper.append(p)
  return np.array(lower[:-1] + upper[:-1])


def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """给单位法向 n 造两个正交切向。"""
  ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
  t1 = np.cross(n, ref)
  t1 /= max(float(np.linalg.norm(t1)), 1e-9)
  t2 = np.cross(n, t1)
  return t1, t2


@dataclass
class ViserFootOverlays:
  """足底遥测的 3D 叠加层。"""

  server: viser.ViserServer
  env: _EnvProtocol
  scene: _SceneProtocol

  show_taxels: bool = True
  show_cop: bool = True
  show_force: bool = True
  show_cone: bool = False
  show_penetration: bool = True
  pressure_full_scale_pa: float = 60_000.0
  arrow_n_per_m: float = 0.004
  taxel_radius_m: float = 0.0042
  lift_m: float = 0.0012

  def __post_init__(self) -> None:
    self._status: Any = None

  @property
  def enabled(self) -> bool:
    return (
      self.show_taxels
      or self.show_cop
      or self.show_force
      or self.show_cone
      or self.show_penetration
    )

  # GUI。

  def setup_gui(self) -> None:
    """在 Visualization tab 加一个足底遥测折叠组。"""
    with self.server.gui.add_folder("足底遥测"):
      cb_tax = self.server.gui.add_checkbox(
        "传感器阵列热力图", initial_value=self.show_taxels
      )
      cb_cop = self.server.gui.add_checkbox(
        "CoP + 支撑多边形", initial_value=self.show_cop
      )
      cb_force = self.server.gui.add_checkbox(
        "接触力箭头", initial_value=self.show_force
      )
      cb_cone = self.server.gui.add_checkbox("摩擦锥", initial_value=self.show_cone)
      cb_pen = self.server.gui.add_checkbox(
        "穿透量", initial_value=self.show_penetration
      )
      sl_scale = self.server.gui.add_slider(
        "压力满量程 (kPa)",
        min=10.0,
        max=300.0,
        step=5.0,
        initial_value=self.pressure_full_scale_pa / 1000.0,
      )
      self._status = self.server.gui.add_text(
        "状态", initial_value="等待仿真…", disabled=True
      )

    @cb_tax.on_update
    def _(_ev: Any) -> None:
      self.show_taxels = bool(cb_tax.value)

    @cb_cop.on_update
    def _(_ev: Any) -> None:
      self.show_cop = bool(cb_cop.value)

    @cb_force.on_update
    def _(_ev: Any) -> None:
      self.show_force = bool(cb_force.value)

    @cb_cone.on_update
    def _(_ev: Any) -> None:
      self.show_cone = bool(cb_cone.value)

    @cb_pen.on_update
    def _(_ev: Any) -> None:
      self.show_penetration = bool(cb_pen.value)

    @sl_scale.on_update
    def _(_ev: Any) -> None:
      self.pressure_full_scale_pa = float(sl_scale.value) * 1000.0

  def on_env_switch(self) -> None:
    """切环境时清掉上一棵机体的图元。"""
    if self.enabled:
      self.scene.clear_debug_all()

  # 每帧绘制。

  def queue(self) -> None:
    """把当前帧的足底几何追加进 scene 的 debug 图元队列。"""
    if not (self.enabled and self.scene.debug_visualization_enabled):
      return
    env = self.env.unwrapped
    # 延迟 import：mjlab.tasks 的 __init__ 会级联 import envs，而 envs 又 import
    # mjlab.viewer（本模块所在包），顶层 import 会回环。运行时 env 已就绪。
    from mjlab.tasks.pikachu_foot_telemetry import foot_telemetry_registered

    tele = foot_telemetry_registered(env)
    if tele is None:
      return
    vis = tele.visual_state(env)
    if vis is None:
      return

    idx = int(self.scene.env_idx)
    if idx >= int(env.num_envs):
      return

    # 各字段的批次维度按脚聚合过（[B,2,...]）或按 taxel 展平过（[B,S,...]），
    # 两种形状在这里统一：先取 env_idx，再在循环里按脚取。
    ground_z = vis.ground_z
    taxel_w = vis.taxel_w[idx].cpu().numpy()
    pressure = vis.pressure[idx].cpu().numpy()
    in_contact = vis.in_contact[idx].cpu().numpy()
    cop_w = vis.cop_w[idx].cpu().numpy()
    has_contact = vis.has_contact[idx].cpu().numpy()
    sole_up = vis.sole_up[idx].cpu().numpy()
    force_w = vis.force_w[idx].cpu().numpy()
    force_n = vis.force_n[idx].cpu().numpy()
    force_t = vis.force_t[idx].cpu().numpy()
    mu = float(vis.mu[idx])
    penetration = vis.penetration[idx].cpu().numpy()
    n_per_foot = taxel_w.shape[0] // 2

    lines: list[str] = []
    for i, side in enumerate(_SIDES):
      sl = slice(i * n_per_foot, (i + 1) * n_per_foot)
      tw = taxel_w[sl]  # [N,3] 世界系
      ic = in_contact[sl]  # [N] bool
      pr = pressure[i]  # [N] Pa
      pen = penetration[sl]  # [N] m
      touching = bool(has_contact[i])
      if self.show_taxels:
        self._draw_taxels(tw, pr, ic)
      if self.show_cop and touching:
        self._draw_cop(tw, ic, cop_w[i])
      if self.show_force and touching:
        self._draw_force(cop_w[i], sole_up[i], force_n[i], force_t[i], force_w[i])
      if self.show_cone and touching and force_n[i] > 1e-3:
        self._draw_friction_cone(cop_w[i], sole_up[i], mu)
      if self.show_penetration:
        self._draw_penetration(tw, ic, ground_z, float(pen.max()))
      lines.append(
        f"{side[0].upper()}: Fn={force_n[i]:5.1f}N Ft={force_t[i]:4.1f}N "
        f"μ={mu:.2f} 接触={int(ic.sum()):2d}/{n_per_foot} "
        f"穿={pen.max() * 1e3:+5.2f}mm"
      )

    if self._status is not None:
      self._status.value = "\n".join(lines)

  def _draw_taxels(
    self, pos: np.ndarray, pressure: np.ndarray, in_contact: np.ndarray
  ) -> None:
    """足底平铺传感器：每个 taxel 一个小球，颜色按压力，未接触画暗灰。"""
    scale = max(self.pressure_full_scale_pa, 1.0)
    for k in range(pos.shape[0]):
      if not bool(in_contact[k]):
        color = _NO_CONTACT_RGBA
      else:
        color = _ramp_color(float(pressure[k]) / scale)
      center = pos[k].copy()
      center[2] += self.lift_m
      self.scene.add_sphere(center=center, radius=self.taxel_radius_m, color=color)

  def _draw_cop(
    self,
    taxel_w: np.ndarray,
    in_contact: np.ndarray,
    cop: np.ndarray,
  ) -> None:
    """CoP 红球 + 接触点凸包（支撑多边形）闭环。"""
    cop_draw = cop.copy()
    cop_draw[2] += self.lift_m * 2.0
    self.scene.add_sphere(center=cop_draw, radius=0.0075, color=(0.95, 0.10, 0.10, 1.0))

    hull = _convex_hull(taxel_w[in_contact][:, :2])
    if len(hull) < 3:
      return
    z = float(taxel_w[in_contact][:, 2].max()) + self.lift_m
    color = (0.98, 0.55, 0.10, 1.0)
    for a, b in zip(hull, np.roll(hull, -1, axis=0), strict=True):
      self.scene.add_arrow(
        start=np.array([a[0], a[1], z]),
        end=np.array([b[0], b[1], z]),
        color=color,
        width=0.0035,
      )

  def _draw_force(
    self,
    cop: np.ndarray,
    sole_up: np.ndarray,
    fn: float,
    ft: float,
    force_w: np.ndarray,
  ) -> None:
    """法向支持力（绿）+ 切向合力（橙）箭头，长度 ∝ 牛顿数。"""
    up = np.asarray(sole_up, dtype=np.float64)
    base = np.asarray(cop, dtype=np.float64).copy()
    base[2] += self.lift_m * 2.0
    n_len = float(fn) * self.arrow_n_per_m
    if n_len > 1e-4:
      self.scene.add_arrow(
        start=base,
        end=base + up * n_len,
        color=(0.20, 0.90, 0.30, 1.0),
        width=0.006,
      )
    tang = np.asarray(force_w, dtype=np.float64)[:2]
    t_len = float(ft) * self.arrow_n_per_m
    if t_len > 1e-4 and float(np.linalg.norm(tang)) > 1e-9:
      dir_t = tang / float(np.linalg.norm(tang))
      # 力是脚→地面（水平分量的反方向即地面给脚的切向力），箭头画反作用。
      self.scene.add_arrow(
        start=base,
        end=base + np.array([-dir_t[0], -dir_t[1], 0.0]) * t_len,
        color=(0.98, 0.60, 0.10, 1.0),
        width=0.005,
      )

  def _draw_friction_cone(
    self, cop: np.ndarray, sole_up: np.ndarray, mu: float
  ) -> None:
    """摩擦锥：顶点在 CoP，法向为轴，母线倾角 = atan(μ)。"""
    up = np.asarray(sole_up, dtype=np.float64)
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    base = np.asarray(cop, dtype=np.float64).copy()
    base[2] += self.lift_m * 2.0
    t1, t2 = _orthonormal_basis(up)
    h = 0.05
    r = h * max(float(mu), 1e-3)
    apex = base + up * h
    for ang in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
      d = np.cos(ang) * t1 + np.sin(ang) * t2
      self.scene.add_arrow(
        start=base,
        end=apex + d * r,
        color=(0.35, 0.75, 1.00, 0.55),
        width=0.0025,
      )
    self.scene.add_arrow(
      start=base, end=apex, color=(0.85, 0.85, 0.95, 0.9), width=0.0035
    )

  def _draw_penetration(
    self,
    taxel_w: np.ndarray,
    in_contact: np.ndarray,
    ground_z: float,
    max_pen: float,
  ) -> None:
    """穿透量：在陷得最深的 taxel 下方画一段红色圆柱（地表 → 该 taxel）。"""
    if max_pen <= 1e-6 or not bool(in_contact.any()):
      return
    idx = int(np.argmin(taxel_w[:, 2]))
    top = taxel_w[idx].copy()
    bottom = top.copy()
    bottom[2] = ground_z - max_pen
    self.scene.add_cylinder(
      start=bottom,
      end=top,
      radius=0.003,
      color=(0.95, 0.10, 0.10, 0.9),
    )
