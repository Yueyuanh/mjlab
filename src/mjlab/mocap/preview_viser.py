"""Viser (浏览器 3D) 版关键帧预览：进度条 / 切 json / 参数热加载 / 重置。

做数据集（如倒地自起）时，原生 viewer（``launch_passive``）每改一次关键帧
JSON 或参数都要重新启动，很慢。这里用 mjlab 已依赖的
``viser`` + ``MjlabViserScene`` 把预览搬到浏览器：窗口自带相机拖拽
orbit/pan/zoom，GUI 面板提供进度条、Play/Pause、Reset、切换关键帧 JSON、
热加载 easing/fps/mode/write_root/loop 等参数——不用重开进程。

**不需要本地 GL/显示环境**：viser server 只跑 websocket，任意机器都能起，
浏览器打开打印的 URL 即可交互。物理/数据侧完全复用 ``MocapEnv`` +
``build_trajectory``；GUI 回调（viser 后台线程）只改 state 和置 dirty 标志，
主循环在自身线程里消费并重建轨迹/reset，避免跨线程改 traj 竞态。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import viser

from mjlab.entity import EntityCfg
from mjlab.mocap.keyframes import build_trajectory
from mjlab.mocap.mocap_env import MocapEnv
from mjlab.mocap.recorder import load_keyframes_json
from mjlab.viewer.viser.scene import MjlabViserScene

_SLEEP_S = 0.03  # 主循环刷新间隔（秒）；浏览器端画面平滑、播放推进仍按 fps。


def _json_options(json_dir: str | None, current: str) -> list[str]:
  """扫描 ``json_dir`` 下 ``*.json``，并把当前路径并入选项（可能在目录外）。"""
  paths = (
    list(Path(json_dir).glob("*.json")) if json_dir and Path(json_dir).is_dir() else []
  )
  options = sorted(str(p) for p in paths)
  if current not in options:
    options.append(current)
  return options


class _PreviewState:
  """GUI 回调（viser 后台线程）与主循环线程共享的播放状态。

  跨线程仅读写简单标量（GIL 保证原子）；``traj``/``cfg`` 只在主循环重建/替换，
  回调只置 ``dirty_*`` 标志，主循环消费后统一执行，避免 env/traj 竞态。
  """

  def __init__(self) -> None:
    self.playing: bool = False
    self.frame: int = 0
    self.mode: str = "kinematic"
    self.fps: float = 30.0
    self.easing: str = "linear"
    self.write_root: bool = False
    self.loop: bool = True
    self.loop_reset: bool = True
    self.fast: bool = False
    self.show_com: bool = True
    self.last_t: float = time.monotonic()
    self.dirty_rebuild: bool = False  # json/easing/fps/loop_reset 变 → 重建轨迹+reset
    self.dirty_frame: int | None = None  # GUI 拖进度条请求的帧
    self.kf_path: str = ""
    self.json_options: list[str] = []
    # fps 节流：拖动 FPS 滑块时仅记候选，离手（停更 >0.5s）才真正 rebuild。
    self.fps_candidate: float | None = None
    self.last_fps_change: float = time.monotonic()
    # 轨迹与插值配置：由 _rebuild 在**主线程**替换（GUI 只置 dirty，不直接写）。
    self.traj: Any = None
    self.cfg: Any = None


def _rebuild(env: MocapEnv, state: _PreviewState) -> None:
  """读取当前 kf_path + 参数重建轨迹，并摆回首帧（仅主线程调用）。"""
  kf_list, cfg = load_keyframes_json(state.kf_path)
  cfg.fps = state.fps
  cfg.easing = state.easing
  traj = build_trajectory(
    kf_list, env.joint_names, cfg=cfg, loop=(state.loop and not state.loop_reset)
  )
  _reset_to_first(env, traj)
  state.traj = traj
  state.cfg = cfg
  state.frame = 0


def _reset_to_first(env: MocapEnv, traj) -> None:
  """运动学恢复回首帧（含 root pos/rot + 关节角），base 归位。"""
  env.reset(
    joint_pos=traj.dof_pos[0],
    root_pos=traj.root_pos[0],
    root_quat_xyzw=traj.root_quat_xyzw[0],
  )


def _apply(env: MocapEnv, state: _PreviewState, physics_dt: float, frame: int) -> None:
  """把 ``frame`` 应用到仿真：pd 走 PD 执行（与录制一致），kinematic 运动学写入。"""
  traj = state.traj
  if state.mode == "pd":
    n_steps = max(1, round(1.0 / (state.fps * physics_dt)))
    env.pd_step(
      joint_target=traj.dof_pos[frame],
      root_pos=traj.root_pos[frame],
      root_quat_xyzw=traj.root_quat_xyzw[frame],
      n_steps=n_steps,
      write_root=state.write_root,
    )
  else:
    env.reset(
      joint_pos=traj.dof_pos[frame],
      root_pos=traj.root_pos[frame],
      root_quat_xyzw=traj.root_quat_xyzw[frame],
    )


def _add_com_overlay(scene: MjlabViserScene, mj_data) -> None:
  """叠加 COM 及地面投影（竖线 + 绿点/红点），语义同原生 viewer 覆盖。"""
  com = np.asarray(mj_data.subtree_com[0], dtype=np.float32)
  proj = np.array([com[0], com[1], 0.0], dtype=np.float32)
  scene.add_cylinder(com, proj, 0.003, (0.95, 0.85, 0.2, 1.0))
  scene.add_sphere(com, 0.015, (0.2, 0.9, 0.3, 1.0))
  scene.add_sphere(proj, 0.015, (0.95, 0.2, 0.2, 1.0))


def _sync_view(env: MocapEnv, scene: MjlabViserScene, state: _PreviewState) -> None:
  """把 warp 最新状态同步到 mj_data，叠加 COM，刷新 viser 场景。"""
  env.sync_to_mj_data()
  scene.debug_visualization_enabled = state.show_com
  if state.show_com:
    _add_com_overlay(scene, env.mj_data)
  else:
    scene.clear()
  scene.update_from_mjdata(env.mj_data)


def _tick(
  env: MocapEnv,
  scene: MjlabViserScene,
  state: _PreviewState,
  controls: dict,
  physics_dt: float,
) -> None:
  """主循环单步：消费 dirty、推进播放、apply、刷新 viser 场景与信息面板。"""
  # 1) fps 节流：FPS 滑块停更 >0.5s 才真正改 fps 并重建轨迹。
  if state.fps_candidate is not None and (
    time.monotonic() - state.last_fps_change > 0.5
  ):
    if abs(state.fps_candidate - state.fps) > 1e-9:
      state.fps = state.fps_candidate
      state.dirty_rebuild = True
    state.fps_candidate = None

  # 2) 重建/重载轨迹（GUI 改 json/easing/fps/loop_reset 时置位）。
  if state.dirty_rebuild:
    _rebuild(env, state)
    state.dirty_rebuild = False
    controls["frame_slider"].max = state.traj.num_frames - 1
    state.last_t = time.monotonic()

  # 3) 播放推进（按 fps 节流；fast=每 tick 快推）。
  now = time.monotonic()
  if state.playing:
    n = 1
    if not state.fast:
      interval = 1.0 / state.fps if state.fps > 0 else 0.0
      n = max(1, int((now - state.last_t) / interval)) if interval else 1
    state.last_t = now
    for _ in range(n):
      nxt = state.frame + 1
      if nxt >= state.traj.num_frames:
        if not state.loop:
          state.playing = False
          nxt = state.traj.num_frames - 1
          state.frame = nxt
          _apply(env, state, physics_dt, nxt)
          break
        state.frame = 0
        if state.loop_reset:
          _reset_to_first(env, state.traj)
        continue
      state.frame = nxt
      _apply(env, state, physics_dt, nxt)

  # 4) 跳帧请求（GUI 拖进度条 → 暂停并就地显示该帧）。
  if state.dirty_frame is not None:
    state.frame = int(np.clip(state.dirty_frame, 0, state.traj.num_frames - 1))
    state.dirty_frame = None
    state.playing = False
    _apply(env, state, physics_dt, state.frame)

  # 5) 刷新场景 + 信息。
  _sync_view(env, scene, state)
  info = controls["info_html"]
  info.content = (
    f"`{state.frame + 1} / {state.traj.num_frames}` 帧 · "
    f"mode `{state.mode}` · {state.fps:.1f} fps · "
    f"{'已循环' if state.playing else '暂停'}"
  )


def _build_gui(
  server: viser.ViserServer, state: _PreviewState, env: MocapEnv, controls: dict
) -> None:
  """构建 GUI 面板并绑定回调；句柄写入 ``controls`` 供主循环更新。"""
  with server.gui.add_folder("Transport"):
    # viser stub 未注解 text/icon 赋值，cast 成 Any 后统一用 _btn
    # （on_click 也走它，Any 上任何访问都合法）。
    _btn = cast(Any, server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY))

    @_btn.on_click
    def _(_) -> None:
      state.playing = not state.playing
      state.last_t = time.monotonic()
      _btn.text = "Pause" if state.playing else "Play"
      _btn.icon = viser.Icon.PLAYER_PAUSE if state.playing else viser.Icon.PLAYER_PLAY

    reset_btn = server.gui.add_button("Reset", icon=viser.Icon.PLAYER_SKIP_BACK)

    @reset_btn.on_click
    def _(_) -> None:
      _reset_to_first(env, state.traj)
      state.frame = 0
      state.playing = False
      _btn.text = "Play"
      _btn.icon = viser.Icon.PLAYER_PLAY

    frame_slider = server.gui.add_slider(
      "Frame", min=0, max=state.traj.num_frames - 1, step=1, initial_value=0
    )

    @frame_slider.on_update
    def _(event) -> None:
      state.dirty_frame = int(event.target.value)
      state.playing = False

    controls["frame_slider"] = frame_slider
    controls["info_html"] = server.gui.add_markdown("")

  with server.gui.add_folder("Motion"):
    json_dd = server.gui.add_dropdown(
      "JSON", options=state.json_options, initial_value=state.kf_path
    )

    @json_dd.on_update
    def _(event) -> None:
      new_path = str(event.target.value)
      if new_path != state.kf_path:
        state.kf_path = new_path
        state.dirty_rebuild = True
        state.playing = False

    mode_dd = server.gui.add_dropdown(
      "Mode", options=["kinematic", "pd"], initial_value=state.mode
    )

    @mode_dd.on_update
    def _(event) -> None:
      state.mode = str(event.target.value)

    easing_dd = server.gui.add_dropdown(
      "Easing", options=["linear", "ease_in_out"], initial_value=state.easing
    )

    @easing_dd.on_update
    def _(event) -> None:
      if str(event.target.value) != state.easing:
        state.easing = str(event.target.value)
        state.dirty_rebuild = True

    fps_input = server.gui.add_slider(
      "FPS", min=5.0, max=120.0, step=1.0, initial_value=state.fps
    )

    @fps_input.on_update
    def _(event) -> None:
      # 拖动过程高频触发：只记候选，主循环停更 0.5s 后才真正改 fps + rebuild。
      state.fps_candidate = float(event.target.value)
      state.last_fps_change = time.monotonic()

  with server.gui.add_folder("Options"):
    loop_cb = server.gui.add_checkbox("loop", initial_value=state.loop)

    @loop_cb.on_update
    def _(event) -> None:
      state.loop = bool(event.target.value)
      # loop 只影响播放尾部 wrap；loop_reset 才影响 build_trajectory 的回退段。
      if state.loop_reset and not state.loop:
        state.dirty_rebuild = True

    loop_reset_cb = server.gui.add_checkbox(
      "loop_reset", initial_value=state.loop_reset
    )

    @loop_reset_cb.on_update
    def _(event) -> None:
      if bool(event.target.value) != state.loop_reset:
        state.loop_reset = bool(event.target.value)
        state.dirty_rebuild = True

    write_root_cb = server.gui.add_checkbox(
      "write_root", initial_value=state.write_root
    )

    @write_root_cb.on_update
    def _(event) -> None:
      state.write_root = bool(event.target.value)

    fast_cb = server.gui.add_checkbox("fast", initial_value=state.fast)

    @fast_cb.on_update
    def _(event) -> None:
      state.fast = bool(event.target.value)

    show_com_cb = server.gui.add_checkbox("show_com", initial_value=state.show_com)

    @show_com_cb.on_update
    def _(event) -> None:
      state.show_com = bool(event.target.value)


def preview_viser(
  keyframes: str,
  fps: float = 30.0,
  easing: Literal["linear", "ease_in_out"] = "linear",
  mode: Literal["kinematic", "pd"] = "kinematic",
  loop: bool = True,
  fast: bool = False,
  physics_dt: float = 0.005,
  write_root: bool = False,
  show_com: bool = True,
  loop_reset: bool = True,
  json_dir: str | None = None,
  device: str = "cpu",
  robot_cfg: EntityCfg | None = None,
  pd_gain_multiplier: float = 1.0,
) -> None:
  """在 viser 浏览器预览一组关键帧动作（交互完整，无需重开进程）。

  Args:
    keyframes: 关键帧 JSON 路径（初始加载；GUI 里可切换其它 json）。
    fps / easing / mode / loop / fast / write_root / show_com / loop_reset:
      与 ``preview_cli`` 同语义，作为 GUI 各控件的**初始值**（可热改）。
    physics_dt: 物理步长（秒），pd 模式用。
    json_dir: GUI 下拉里列出的 json 目录；默认该模块 ``configs/``。
    device: 仿真设备（cpu/cuda）。viser 渲染本身无需 GPU。
    robot_cfg: 机器人配置；None 用默认 14dof。
    pd_gain_multiplier: PD 增益倍率（仅初始，不做热改——需重建环境）。
  """
  env = MocapEnv(
    device=device,
    physics_dt=physics_dt,
    robot_cfg=robot_cfg,
    pd_gain_multiplier=pd_gain_multiplier,
  )
  state = _PreviewState()
  state.kf_path = keyframes
  state.json_options = _json_options(json_dir, keyframes)
  state.fps = fps
  state.easing = easing
  state.mode = mode
  state.loop = loop
  state.loop_reset = loop_reset
  state.fast = fast
  state.write_root = write_root
  state.show_com = show_com

  _rebuild(env, state)  # 初始轨迹 + 摆首帧

  server = viser.ViserServer(label="mjlab-mocap-preview")
  scene = MjlabViserScene(server, env.mj_model, num_envs=1)

  @server.on_client_connect
  def _focus_camera(client: viser.ClientHandle) -> None:
    client.camera.position = (0.8, 0.5, 0.5)
    client.camera.look_at = (0.0, 0.0, 0.1)
    client.camera.fov = 55

  controls: dict = {}
  _build_gui(server, state, env, controls)

  print(
    f"[preview-viser] keyframes: {state.kf_path}"
    f"\n[preview-viser] {state.traj.num_frames} 帧 · mode={mode} · {fps:.1f}fps"
    f"\n[preview-viser] 浏览器打开上方 URL 交互；Ctrl+C 退出。"
  )
  try:
    while True:
      _tick(env, scene, state, controls, physics_dt)
      time.sleep(_SLEEP_S)
  except KeyboardInterrupt:
    print("[preview-viser] 退出")
