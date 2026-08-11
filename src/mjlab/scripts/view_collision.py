"""Live collision-body tuning viewer for mjlab tasks.

``uv run viewer <TASK> [OPTIONS]`` mimics :mod:`play.py`: the first positional
argument is a registered task id, and the task's environment is opened directly
(no raw XML loading). Optionally ``--xml <path>`` overrides the robot MJCF
(e.g. an ellipsoid-collision variant), with hot-reload on save.

By default the robot is **suspended in the air**: physics is not stepped, the
robot floats above the ground (``--suspend-height``) so its collision bodies can
be inspected and tuned from any angle. Pass ``--zero`` (zero actions, free
fall) or ``--random`` to enable simulation and let it land.

Collision bodies are identified by the codebase convention (geom names ending
in ``_collision``, or any geom in ``group=3``), tinted opaque green so they
show through the translucent visual shell.

Console commands:
    list                   List all collision bodies (name, pos, size)
    scale <f>              Scale all collision body sizes by f
    scale <name> <f>       Scale a single collision body by f
    pos <name> x y z       Set a single collision body's pos
    color <name> r g b a   Set a single collision body's color (0~1)
    r                      Force reload (re-reads ``--xml`` if given)
    q / Ctrl-C             Quit

Note: console edits only affect rendering and CPU-side contact display. To
change the simulated geometry, write the values back into the XML file — the
viewer hot-reloads on save.
"""

from __future__ import annotations

import select
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco.viewer
import torch
import tyro

from mjlab import TYRO_FLAGS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.types import VecEnvObs
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.sim import Simulation
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends

# Collision bodies: geoms named ``*_collision`` (Pikachu / g1 convention) or
# any geom in ``group=3`` (the group used by the ellipsoid variant XMLs).
COLLISION_SUFFIX = "_collision"
VISUAL_SUFFIX = "_visual"
COLLISION_GROUP = 3

# Bool flags for this interactive viewer should be presence flags (``--zero``
# rather than ``--zero True``), so drop FlagConversionOff for the config parse.
VIEWER_TYRO_FLAGS = tuple(c for c in TYRO_FLAGS if c is not tyro.conf.FlagConversionOff)


@dataclass(frozen=True)
class ViewerConfig:
  xml: str | None = None
  """Override the robot MJCF path (e.g. an ellipsoid-collision variant)."""

  zero: bool = False
  """Enable simulation with zero actions: the robot free-falls and lands."""

  random: bool = False
  """Enable simulation with a random action policy (also makes it land)."""

  suspend_height: float = 0.25
  """How far above the ground to hold the robot while suspended (m)."""

  num_envs: int | None = None
  """Override the number of parallel environments (viewer shows env 0)."""

  device: str | None = None
  """Torch device; defaults to cuda:0 when available, else cpu."""

  frame_rate: float = 60.0
  """Render frame rate in Hz."""

  transparent: bool = True
  """Render collision bodies semi-transparent to see through the shell."""


@dataclass
class _ViewerState:
  env: ManagerBasedRlEnv | None = None
  viewer: mujoco.viewer.Handle | None = None
  model: mujoco.MjModel | None = None
  data: mujoco.MjData | None = None
  obs: VecEnvObs | None = None
  need_reload: bool = False
  running: bool = True
  last_tick: float = field(default_factory=time.perf_counter)
  budget: float = 0.0


def _make_env(task_id: str, cfg: ViewerConfig, device: str) -> ManagerBasedRlEnv:
  """Build the task environment, applying the optional robot XML override."""
  env_cfg = load_env_cfg(task_id, play=True)
  if cfg.xml is not None:
    xml_path = Path(cfg.xml).resolve()
    if not xml_path.exists():
      raise FileNotFoundError(f"XML 文件不存在: {xml_path}")
    robot = env_cfg.scene.entities.get("robot")
    if robot is None:
      raise ValueError("该任务没有名为 'robot' 的实体，--xml 覆盖仅适用于机器人任务。")
    spec_path = xml_path
    robot.spec_fn = lambda: mujoco.MjSpec.from_file(str(spec_path))
    print(f"[INFO] 使用自定义机器人 XML: {spec_path.name}")
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def _collision_geom_ids(model: mujoco.MjModel) -> list[int]:
  """Return geom ids that are collision bodies."""
  ids: list[int] = []
  for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    if name.endswith(COLLISION_SUFFIX) or int(model.geom_group[i]) == COLLISION_GROUP:
      ids.append(i)
  return ids


def _visual_geom_ids(model: mujoco.MjModel) -> list[int]:
  """Return geom ids of the visual shell (named ``*_visual``)."""
  ids: list[int] = []
  for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    if name.endswith(VISUAL_SUFFIX):
      ids.append(i)
  return ids


def _style_geoms(model: mujoco.MjModel, transparent: bool) -> None:
  """Tint collision bodies opaque green; dim the visual shell."""
  for i in _collision_geom_ids(model):
    model.geom_rgba[i] = (0.2, 0.6, 0.2, 1.0)
  if transparent:
    for i in _visual_geom_ids(model):
      rgba = model.geom_rgba[i]
      model.geom_rgba[i] = (rgba[0], rgba[1], rgba[2], 0.18)


def _sync_sim_to_mjd(sim: Simulation, mjd: mujoco.MjData, env_idx: int = 0) -> None:
  """Copy one env's state from the GPU sim into the CPU MjData buffer."""
  model = sim.mj_model
  if model.nq > 0:
    mjd.qpos[:] = sim.data.qpos[env_idx].cpu().numpy()
    mjd.qvel[:] = sim.data.qvel[env_idx].cpu().numpy()
  if model.nu > 0:
    mjd.ctrl[:] = sim.data.ctrl[env_idx].cpu().numpy()
  if model.nmocap > 0:
    mjd.mocap_pos[:] = sim.data.mocap_pos[env_idx].cpu().numpy()
    mjd.mocap_quat[:] = sim.data.mocap_quat[env_idx].cpu().numpy()
  mjd.xfrc_applied[:] = sim.data.xfrc_applied[env_idx].cpu().numpy()


def run_viewer(task_id: str, cfg: ViewerConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  frame_time = 1.0 / max(cfg.frame_rate, 1.0)
  xml_path = Path(cfg.xml).resolve() if cfg.xml is not None else None

  # Default mode is "suspend": physics is not stepped and the robot floats in
  # the air. Only --zero / --random enable simulation (and let it land).
  simulate = cfg.random or cfg.zero

  state = _ViewerState()

  def open_viewer() -> tuple[
    ManagerBasedRlEnv, mujoco.viewer.Handle, mujoco.MjModel, mujoco.MjData, VecEnvObs
  ]:
    """(Re)build the env + viewer; returns strongly-typed locals."""
    if state.viewer is not None:
      try:
        state.viewer.close()
      except Exception:
        pass
      state.viewer = None
    if state.env is not None:
      state.env.close()

    env = _make_env(task_id, cfg, device)
    sim = env.unwrapped.sim
    model = sim.mj_model
    data = sim.mj_data
    _style_geoms(model, cfg.transparent)

    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=True)
    assert viewer is not None
    viewer.opt.geomgroup[:] = 1
    if cfg.transparent:
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1

    obs, _ = env.reset()
    _sync_sim_to_mjd(sim, data)
    if not simulate and cfg.suspend_height > 0:
      # Hold the robot off the ground so its collision bodies are easy to see.
      data.qpos[2] += cfg.suspend_height
    mujoco.mj_forward(model, data)
    viewer.sync(state_only=False)

    state.env = env
    state.viewer = viewer
    state.model = model
    state.data = data
    state.obs = obs
    n_coll = len(_collision_geom_ids(model))
    src = xml_path.name if xml_path is not None else "task默认XML"
    print(
      f"[{time.strftime('%H:%M:%S')}] 已打开任务 {task_id} | 来源 {src}"
      f" | 碰撞体 {n_coll} 个"
    )
    return env, viewer, model, data, obs

  def watch_loop() -> None:
    """Watch the --xml file's mtime; set a reload flag on change."""
    if xml_path is None:
      return
    mtime = 0.0
    try:
      mtime = xml_path.stat().st_mtime
    except FileNotFoundError:
      pass
    while state.running:
      time.sleep(0.3)
      try:
        new_mtime = xml_path.stat().st_mtime
      except FileNotFoundError:
        continue
      if new_mtime != mtime:
        mtime = new_mtime
        state.need_reload = True

  def make_policy(env: ManagerBasedRlEnv):
    unwrapped = env.unwrapped
    action_shape = unwrapped.action_space.shape
    dev = unwrapped.device
    if cfg.zero:
      return lambda obs: torch.zeros(action_shape, device=dev)
    return lambda obs: 2 * torch.rand(action_shape, device=dev) - 1

  env, viewer, model, data, obs = open_viewer()
  if simulate:
    policy = make_policy(env)
    step_dt = env.unwrapped.step_dt
  else:
    policy = None
    step_dt = 0.0

  threading.Thread(target=watch_loop, daemon=True).start()
  mode = "仿真(零动作)" if cfg.zero else ("仿真(随机动作)" if cfg.random else "悬浮")
  print(
    f"\n碰撞体实时查看器已启动 [{mode}]。编辑 --xml 保存即刷新；命令: list / scale / pos / color / r / q"
  )

  try:
    while state.running:
      if state.need_reload:
        state.need_reload = False
        print(f"[{time.strftime('%H:%M:%S')}] 检测到 XML 已修改，重载中...")
        try:
          env, viewer, model, data, obs = open_viewer()
          if simulate:
            policy = make_policy(env)
        except Exception as e:
          print(f"  !! 重载失败（保留旧环境）: {e}")

      if not viewer.is_running():
        break

      now = time.perf_counter()
      if simulate:
        assert policy is not None
        # Realtime pacing: accumulate sim-time budget, step until spent.
        state.budget += (now - state.last_tick) * 1.0  # 1x speed
        state.last_tick = now
        while state.budget >= step_dt:
          obs, *_ = env.step(policy(obs))
          state.obs = obs
          state.budget -= step_dt
        _sync_sim_to_mjd(env.unwrapped.sim, data)
        mujoco.mj_forward(model, data)
        viewer.sync(state_only=False)

      if select.select([sys.stdin], [], [], 0.0)[0]:
        line = sys.stdin.readline()
        if not line:
          break
        handle(state, line)

      time.sleep(max(frame_time - (time.perf_counter() - now), 0.001))
  except KeyboardInterrupt:
    pass
  finally:
    if state.viewer is not None:
      try:
        state.viewer.close()
      except Exception:
        pass
    if state.env is not None:
      state.env.close()


def live_sync(state: _ViewerState) -> None:
  """Refresh physics + display after an in-memory geom edit (no reload)."""
  model = state.model
  data = state.data
  viewer = state.viewer
  if model is not None and data is not None and viewer is not None:
    mujoco.mj_forward(model, data)
    viewer.sync(state_only=False)


def _geom_bare_name(model: mujoco.MjModel, i: int) -> str:
  """Geom name without the ``<entity>/`` prefix (e.g. ``robot/...``)."""
  name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"<{i}>"
  return name.rsplit("/", 1)[-1]


def cmd_list(state: _ViewerState) -> None:
  model = state.model
  if model is None:
    return
  for i in _collision_geom_ids(model):
    s = model.geom_size[i]
    p = model.geom_pos[i]
    print(
      f"  {_geom_bare_name(model, i):<28} size={s[0]:.4f} {s[1]:.4f} {s[2]:.4f}"
      f"  pos={p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
    )


def _find_geom(state: _ViewerState, name: str) -> int:
  model = state.model
  if model is None:
    return -1
  for i in _collision_geom_ids(model):
    full = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    if full == name or full.endswith("/" + name):
      return i
  print(f"  找不到碰撞体: {name}")
  return -1


def cmd_scale(state: _ViewerState, args: list[str]) -> None:
  model = state.model
  if model is None or not args:
    return
  if len(args) == 1:
    factor = float(args[0])
    ids = _collision_geom_ids(model)
    label = "全部"
  else:
    factor = float(args[-1])
    ids = [_find_geom(state, args[0])]
    label = args[0]
  ids = [i for i in ids if i >= 0]
  if not ids:
    return
  for i in ids:
    model.geom_size[i] *= factor
  live_sync(state)
  print(f"  {label}: size 缩放 x{factor}")


def cmd_pos(state: _ViewerState, args: list[str]) -> None:
  if len(args) != 4:
    print("  用法: pos <name> x y z")
    return
  i = _find_geom(state, args[0])
  if i < 0:
    return
  model = state.model
  assert model is not None
  model.geom_pos[i] = [float(v) for v in args[1:4]]
  live_sync(state)
  print(f"  {args[0]}: pos -> {' '.join(args[1:4])}")


def cmd_color(state: _ViewerState, args: list[str]) -> None:
  if len(args) != 5:
    print("  用法: color <name> r g b a")
    return
  i = _find_geom(state, args[0])
  if i < 0:
    return
  model = state.model
  assert model is not None
  model.geom_rgba[i] = [float(v) for v in args[1:5]]
  live_sync(state)
  print(f"  {args[0]}: rgba -> {' '.join(args[1:5])}")


def handle(state: _ViewerState, cmd: str) -> None:
  parts = cmd.strip().split()
  if not parts:
    return
  head = parts[0].lower()
  try:
    if head in ("q", "quit", "exit"):
      state.running = False
    elif head in ("list", "l"):
      cmd_list(state)
    elif head == "scale":
      cmd_scale(state, parts[1:])
    elif head == "pos":
      cmd_pos(state, parts[1:])
    elif head == "color":
      cmd_color(state, parts[1:])
    elif head == "r":
      state.need_reload = True
    else:
      print("未知命令。输入 list / scale / pos / color / r / q")
  except (ValueError, IndexError) as e:
    print(f"  命令解析失败: {e}")


def main():
  maybe_print_top_level_help("viewer")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=TYRO_FLAGS,
  )

  args = tyro.cli(
    ViewerConfig,
    args=remaining_args,
    default=ViewerConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=VIEWER_TYRO_FLAGS,
  )
  del remaining_args

  run_viewer(chosen_task, args)


if __name__ == "__main__":
  main()
