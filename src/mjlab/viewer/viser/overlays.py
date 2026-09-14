"""Overlay managers for Viser viewer orchestration.

These managers intentionally coordinate *when* higher-level updates happen
(env switches, paused/running updates, etc.) while leaving low-level render
handle lifecycle ownership inside :mod:`scene.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np
import viser

from mjlab.sensor import CameraSensor, ContactSensor
from mjlab.viewer.viser.camera_viewer import ViserCameraViewer
from mjlab.viewer.viser.reward_bar_panel import RewardBarPanel
from mjlab.viewer.viser.term_plotter import ViserTermPlotter


class _EnvProtocol(Protocol):
  @property
  def unwrapped(self) -> Any: ...


class _SceneProtocol(Protocol):
  env_idx: int
  debug_visualization_enabled: bool
  needs_update: bool

  @property
  def show_contact_points(self) -> bool: ...
  @property
  def show_contact_forces(self) -> bool: ...

  def clear_debug_all(self) -> None: ...
  def clear(self) -> None: ...
  def add_sphere(self, center: Any, radius: float, color: Any) -> None: ...


@dataclass
class ViserTermOverlays:
  """Manage reward/metrics term plot tabs for Viser viewer."""

  server: viser.ViserServer
  env: _EnvProtocol
  scene: _SceneProtocol
  frame_time: float
  reward_bar_max_terms: int = 20
  metrics_output_dir: Path | None = None
  metrics_source_name: Callable[[], str] | None = None
  reward_plotter: ViserTermPlotter | None = None
  reward_bar_panel: RewardBarPanel | None = None
  metrics_plotter: ViserTermPlotter | None = None

  def setup_tabs(self, tabs: Any) -> None:
    """Create rewards/metrics tabs based on available managers."""
    if hasattr(self.env.unwrapped, "reward_manager"):
      with tabs.add_tab("Rewards", icon=viser.Icon.CHART_LINE):
        term_names = [
          name
          for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
            self.scene.env_idx
          )
        ]
        # Live bar panel (running-mean comparison).
        self.reward_bar_panel = RewardBarPanel(
          self.server,
          term_names,
          update_dt=self.frame_time,
          max_terms=self.reward_bar_max_terms,
        )
        self.reward_plotter = ViserTermPlotter(
          self.server, term_names, name="Reward", env_idx=self.scene.env_idx
        )

    if hasattr(self.env.unwrapped, "metrics_manager"):
      term_names = [
        name
        for name, _ in self.env.unwrapped.metrics_manager.get_active_iterable_terms(
          self.scene.env_idx
        )
      ]
      if term_names:
        with tabs.add_tab("Metrics", icon=viser.Icon.CHART_BAR):
          self.metrics_plotter = ViserTermPlotter(
            self.server,
            term_names,
            name="Metric",
            env_idx=self.scene.env_idx,
            csv_output_dir=self.metrics_output_dir,
            source_name=self.metrics_source_name,
            step_dt=self.env.unwrapped.step_dt,
          )

  def on_env_switch(self) -> None:
    """Clear histories when active environment changes."""
    env_idx = self.scene.env_idx
    if self.reward_plotter:
      self.reward_plotter.clear_histories()
      self.reward_plotter.update_env_idx(env_idx)
    if self.reward_bar_panel:
      self.reward_bar_panel.clear_histories()
    if self.metrics_plotter:
      self.metrics_plotter.clear_histories()
      self.metrics_plotter.update_env_idx(env_idx)

  def on_source_change(self) -> None:
    """Keep a recording scoped to exactly one policy checkpoint."""
    if self.metrics_plotter:
      self.metrics_plotter.on_source_change()

  def update(self, paused: bool, step_idx: int | None = None) -> None:
    """Update term plots from the selected environment."""
    if (
      self.reward_plotter is not None or self.reward_bar_panel is not None
    ) and not paused:
      terms = list(
        self.env.unwrapped.reward_manager.get_active_iterable_terms(self.scene.env_idx)
      )
      if self.reward_plotter is not None:
        self.reward_plotter.update(terms, step_idx=step_idx)
      if self.reward_bar_panel is not None:
        self.reward_bar_panel.update(terms)

    if self.metrics_plotter is not None and not paused:
      terms = list(
        self.env.unwrapped.metrics_manager.get_active_iterable_terms(self.scene.env_idx)
      )
      self.metrics_plotter.update(terms, step_idx=step_idx)

  def clear_histories(self) -> None:
    """Clear all overlay histories."""
    self.on_env_switch()

  def cleanup(self) -> None:
    """Cleanup plotter resources."""
    if self.reward_plotter:
      self.reward_plotter.cleanup()
    if self.reward_bar_panel:
      self.reward_bar_panel.cleanup()
    if self.metrics_plotter:
      self.metrics_plotter.cleanup()


@dataclass
class ViserCameraOverlays:
  """Manage camera feed widgets and updates for Viser viewer."""

  server: viser.ViserServer
  env: _EnvProtocol
  mj_model: mujoco.MjModel
  camera_viewers: list[ViserCameraViewer] | None = None

  @property
  def has_cameras(self) -> bool:
    """Whether the environment has any camera sensors."""
    return any(
      isinstance(s, CameraSensor) for s in self.env.unwrapped.scene.sensors.values()
    )

  def setup_controls(self) -> None:
    """Create camera feed controls under the active GUI folder."""
    camera_sensors = [
      sensor
      for sensor in self.env.unwrapped.scene.sensors.values()
      if isinstance(sensor, CameraSensor)
    ]
    if not camera_sensors:
      self.camera_viewers = []
      return

    self.camera_viewers = [
      ViserCameraViewer(self.server, sensor, self.mj_model) for sensor in camera_sensors
    ]

  def update(self, sim_data: Any, env_idx: int, scene_offset: Any) -> None:
    """Push latest camera images/frustums to GUI."""
    if not self.camera_viewers:
      return
    for camera_viewer in self.camera_viewers:
      camera_viewer.update(sim_data, env_idx, scene_offset)

  def cleanup(self) -> None:
    """Cleanup all camera feed widgets."""
    if not self.camera_viewers:
      return
    for camera_viewer in self.camera_viewers:
      camera_viewer.cleanup()


@dataclass
class ViserDebugOverlays:
  """Manage debug visualization queueing and env-switch behavior."""

  env: _EnvProtocol
  scene: _SceneProtocol

  def on_env_switch(self) -> None:
    """Reset debug visuals when switching selected environment."""
    if self.scene.debug_visualization_enabled:
      self.scene.clear_debug_all()

  def queue(self) -> None:
    """Queue environment debug visualizers for the current frame."""
    if self.scene.debug_visualization_enabled and hasattr(
      self.env.unwrapped, "update_visualizers"
    ):
      self.scene.clear()  # Clear queued arrows from previous frame.
      self.env.unwrapped.update_visualizers(self.scene)


@dataclass
class ViserCoMOverlays:
  """实时叠加整机质心/惯量/足底接触支撑多边形（play ``--com``）。

  每帧调用 :func:`mjlab.mocap.stability.com_and_inertia_from_sim` 从当前环境算
  CoM/惯量/着地足盒，再经 :func:`draw_stability` 画到 ``scene`` 的 debug 队列，
  与 ``ViserDebugOverlays`` 复用同一套 ``_sync_simple_primitives`` 生命周期
  （每帧重建当前队列中的图元，不会累积残留）。
  """

  server: viser.ViserServer
  env: _EnvProtocol
  scene: _SceneProtocol
  enabled: bool = False

  def setup_gui(self) -> None:
    """在 Visualization tab 加一个开启开关。"""
    cb = self.server.gui.add_checkbox(
      "CoM / 惯量 / 支撑多边形", initial_value=self.enabled
    )

    @cb.on_update
    def _on_update(_ev) -> None:
      self.enabled = cb.value

  def on_env_switch(self) -> None:
    """切换环境时清掉上一棵机体的叠加图元。"""
    if self.enabled:
      self.scene.clear_debug_all()

  def queue(self) -> None:
    """把当前环境稳定量追加进 scene 的 debug 图元队列（供每帧 sync）。"""
    if not (self.enabled and self.scene.debug_visualization_enabled):
      return
    env = self.env.unwrapped
    sim = env.sim
    # 延迟 import：mjlab.mocap 的 __init__ 会级联 import tasks → envs，在
    # envs 初始化期间顶层 import 会回环，这里只在运行时（env 已就绪）拉取。
    from mjlab.mocap.stability import com_and_inertia_from_sim, draw_stability

    data = com_and_inertia_from_sim(sim, self.scene.env_idx)
    draw_stability(self.scene, data)
    self._queue_foot_friction(env, data)

  def _queue_foot_friction(self, env: Any, data: Any) -> None:
    """逐脚摩擦锥裕度预警（打滑信号灯）。

    只用已提交 ground-truth 接触力的 ``feet_ground_contact`` ContactSensor
    （``reduce="netforce"``，世界系每 primary 净力 ``[N,3]``）。着地脚底近似朝上，
    故法向力 ≈ 净力 z 分量，切向 ≈ 水平合力；裕度 = ``μ·Fn − |Ft|``，μ 取足部
    geom 摩擦系数。把每只脚画一个信号灯球悬在足盒上方：

      - 绿：裕度充裕（摩擦锥利用率 < 80%）
      - 黄：临界（80%..100%）
      - 红：打滑（μ·Fn − |Ft| < 0，切向超锥）—— 正是足端滑移的直接诱因

    若无匹配的接触传感器或足盒，本层静默跳过（不阻塞 ``--com`` 其它叠加）。
    """
    foot_boxes = list(getattr(data, "foot_boxes", ()) or ())
    sensors = getattr(env.scene, "sensors", None)
    sensor = None
    if sensors:
      for s in sensors.values():
        if not isinstance(s, ContactSensor):
          continue
        name = getattr(s.cfg, "name", "")
        if name == "feet_ground_contact" or ("foot" in name and "ground" in name):
          if s.data.found is not None and s.data.force is not None:
            sensor = s
            break
    if (
      sensor is None
      or sensor.data.force is None
      or sensor.data.found is None
      or not foot_boxes
    ):
      return

    # 足部摩擦系数（第 0 分量即切向限制系数），取各足 geom 平均。
    m = env.sim.mj_model
    fr = [
      m.geom_friction[g, 0]
      for g in range(m.ngeom)
      if "ankle_collision" in _geom_name(m, g)
    ]
    mu = float(sum(fr) / len(fr)) if fr else 0.6

    idx = self.scene.env_idx
    force_data = sensor.data.force  # 已在 guard 里排除 None
    assert force_data is not None
    force = force_data[idx].float().cpu().numpy()  # (P,3) 世界系非线性力
    found = sensor.data.found[idx].float().cpu().numpy()  # (P,)
    for p in range(force.shape[0]):
      box = foot_boxes[p % len(foot_boxes)]
      if not bool(found[p]):
        continue
      fn = max(force[p, 2], 0.0)
      ft = float((force[p, :2] ** 2).sum() ** 0.5)
      if fn <= 0:
        continue
      margin = mu * fn - ft
      ratio = ft / (mu * fn)  # 摩擦锥利用率
      # 自 com 沿用足盒：取足盒底面中心上方抬 0.04，画信号灯。
      center = box[0]
      color = (
        (0.85, 0.2, 0.2, 1.0)
        if margin < 0
        else ((0.95, 0.85, 0.1, 1.0) if ratio > 0.8 else (0.25, 0.9, 0.3, 1.0))
      )
      self.scene.add_sphere(
        center=np.array([center[0], center[1], data.ground_z + 0.04]),
        radius=0.02,
        color=color,
      )


def _geom_name(model: Any, gid: int) -> str:
  """取宿主 ``MjModel`` 第 ``gid`` 个 geom 名称（与 stability._host_geom_name 一致）。"""
  if gid < 0 or gid >= int(getattr(model, "ngeom", 0)):
    return ""
  adr = int(model.name_geomadr[gid])
  if adr < 0:
    return ""
  end = model.names.find(b"\x00", adr)
  if end < 0:
    end = len(model.names)
  try:
    return model.names[adr:end].decode()
  except UnicodeDecodeError:
    return ""


@dataclass
class ViserContactOverlays:
  """Manage contact-visualization orchestration from the viewer layer.

  Note: contact mesh creation/update/removal stays in ``ViserMujocoScene``.
  This manager only requests scene refreshes at the right times.
  """

  scene: _SceneProtocol

  def is_enabled(self) -> bool:
    """Whether any contact visualization is currently enabled."""
    return self.scene.show_contact_points or self.scene.show_contact_forces

  def on_env_switch(self) -> None:
    """Request a scene refresh when switching environments with contacts enabled."""
    if self.is_enabled():
      self.scene.needs_update = True
