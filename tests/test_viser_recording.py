"""Control-step capture, command isolation and silent-loop restoration."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mjlab.managers.command_manager import CommandTerm
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.viewer.base import BaseViewer
from mjlab.viewer.viewer_config import ViewerConfig
from mjlab.viewer.viser import recording
from mjlab.viewer.viser.recording import (
  ControlStepRecording,
  RecordingCommandSchedule,
  RecordingOptions,
)
from mjlab.viewer.viser.term_plotter import (
  MetricsCsvRecorder,
  ViserTermPlotter,
  load_metrics_csv,
  metric_semantic_group,
)
from mjlab.viewer.viser.viewer import ViserPlayViewer


def make_env():
  env = SimpleNamespace(
    step_dt=0.02,
    num_envs=2,
    device="cpu",
    counter=0,
    cfg=SimpleNamespace(viewer=ViewerConfig(), auto_reset=True),
    reset_buf=torch.zeros(2, dtype=torch.bool),
    scene=SimpleNamespace(entities={}),
    sim=SimpleNamespace(mj_model=None, model=None, expanded_fields=set(), data=None),
  )
  env.unwrapped = env
  command = object.__new__(UniformVelocityCommand)
  command._env = env
  command.vel_command_b = torch.tensor([[0.2, 0.1, 0.0], [0.4, 0.0, 0.0]])
  command.recording_velocity_override = None
  env.command_manager = SimpleNamespace(get_term=lambda name: command)
  env.metrics_manager = SimpleNamespace(
    active_terms=["foot_contact/left", "joint_torque_nm/left_knee_joint"],
    get_active_iterable_terms=lambda idx: [
      ("foot_contact/left", [1.0]),
      ("joint_torque_nm/left_knee_joint", [float(env.counter)]),
    ],
  )
  env.get_observations = lambda: torch.zeros(2, 1)

  def step(action):
    env.counter += 1
    return None

  env.step = step
  return env, command


def test_live_capture_is_opt_in_and_reorders_batched_metrics():
  plotter = object.__new__(ViserTermPlotter)
  plotter._live_publisher = None
  # No manager access or device transfer while disabled.
  plotter.publish_control_step(None, 1, 0)
  env, _ = make_env()
  env.metrics_manager._step_values = torch.tensor([[1.0, 4.0], [0.0, 8.0]])
  env.metrics_manager.get_active_iterable_terms = MagicMock(
    side_effect=AssertionError("batched path must avoid per-term transfers")
  )
  publisher = MagicMock()
  publisher.names = list(reversed(env.metrics_manager.active_terms))
  plotter._live_publisher = publisher
  plotter._recorder = SimpleNamespace(_source_name=lambda: "modelA")
  plotter.publish_control_step(env, 12, 1)
  publisher.append.assert_called_once_with(12, 1, "modelA", [8.0, 0.0], reset=False)


@pytest.mark.parametrize(
  "kwargs",
  [
    {"duration_s": 0},
    {"duration_s": float("nan")},
    {"duration_s": 3601},
    {"command_interval_s": 0},
    {"lower": (1, 0, 0), "upper": (0, 0, 0)},
    {"velocity": (float("inf"), 0, 0)},
    {"seed": -1},
    {"video_fps": 30},
  ],
)
def test_recording_options_validate(kwargs):
  with pytest.raises(ValueError):
    RecordingOptions(**kwargs).validate()


def test_random_command_schedule_is_repeatable_and_held():
  cfg = RecordingOptions(command_mode="random", command_interval_s=0.1)
  a, b = RecordingCommandSchedule(cfg, 0.02), RecordingCommandSchedule(cfg, 0.02)
  values = [a.value(i) for i in range(20)]
  assert values == [b.value(i) for i in range(20)]
  assert values[0] == values[4] and values[0] != values[5]
  for value in values:
    assert all(
      lo <= x <= hi for lo, x, hi in zip(cfg.lower, value, cfg.upper, strict=True)
    )


def test_recording_command_wins_over_task_and_joystick(monkeypatch):
  _, command = make_env()
  monkeypatch.setattr(
    CommandTerm, "compute", lambda self, *args: self.vel_command_b.zero_()
  )
  command._joystick_enabled = SimpleNamespace(value=True)
  command._joystick_sliders = [SimpleNamespace(value=0.9) for _ in range(3)]
  command._joystick_get_env_idx = lambda: 0
  command.set_recording_velocity_override(0, (0.2, -0.1, 0.3))
  command.compute(0.02)
  torch.testing.assert_close(command.command[0], torch.tensor([0.2, -0.1, 0.3]))


def test_recording_captures_all_terms_and_restores_command(tmp_path):
  env, command = make_env()
  before = command.command.clone()
  recorder = MetricsCsvRecorder(tmp_path, lambda: "model", 0.02)
  cfg = RecordingOptions(
    prefix="试验A",
    video=False,
    duration_s=0.1,
    command_mode="fixed",
    velocity=(-0.2, 0.1, 0.3),
  )
  session = ControlStepRecording(env, recorder, cfg, 0)
  path = session.start()
  for i in range(5):
    session.prepare_command()
    env.counter = i
    reason = session.append(100 + i)
  assert reason == "duration_reached"
  assert session.samples == 5
  assert command.recording_velocity_override is not None
  torch.testing.assert_close(command.command[1], before[1])
  session.finish(reason)
  assert command.recording_velocity_override is None
  torch.testing.assert_close(command.command, before)
  loaded = load_metrics_csv(path.read_bytes(), env.metrics_manager.active_terms)
  assert len(loaded.time_s) == 5
  assert len(loaded.values) == 2
  np.testing.assert_allclose(loaded.time_s, np.arange(5) * 0.02)
  assert path.name.startswith("试验A_metrics_")
  assert path.with_suffix(".recording.json").exists()


def test_autoreset_terminal_pose_is_not_paired_with_terminal_metrics(tmp_path):
  env, _ = make_env()
  session = ControlStepRecording(
    env,
    MetricsCsvRecorder(tmp_path, lambda: "m", 0.02),
    RecordingOptions(video=False),
    0,
  )
  path = session.start()
  env.reset_buf[0] = True
  assert session.append(1) == "episode_ended_before_auto_reset_frame"
  session.finish("episode_ended_before_auto_reset_frame")
  assert session.samples == 0
  assert len(path.read_text().splitlines()) == 1


def test_video_frames_share_csv_steps_and_are_streamed(tmp_path, monkeypatch):
  env, _ = make_env()
  captured = []

  class Renderer:
    def __init__(self, model, cfg, *args):
      assert (cfg.width, cfg.height) == (1280, 720)

    def initialize(self):
      pass

    def update(self, data):
      pass

    def render(self):
      return np.array([env.counter])

    def close(self):
      pass

  class Writer:
    def __init__(self, path, shape, fps):
      assert shape == (720, 1280) and fps == 25

    def __enter__(self):
      return self

    def add_image(self, frame):
      captured.append(int(frame[0]))

    def __exit__(self, *args):
      pass

  monkeypatch.setattr(recording, "OffscreenRenderer", Renderer)
  monkeypatch.setattr(recording.media, "VideoWriter", Writer)
  session = ControlStepRecording(
    env,
    MetricsCsvRecorder(tmp_path, lambda: "m", 0.02),
    RecordingOptions(duration_s=0.2),
    0,
  )
  session.start()
  for i in range(10):
    env.counter = i
    session.append(200 + i)
  session.finish("duration_reached")
  assert captured == [0, 2, 4, 6, 8]
  assert [f["csv_sample"] for f in session.frame_map] == captured
  assert [f["sim_step"] for f in session.frame_map] == [200 + i for i in captured]
  assert session.samples == 10


def test_prefix_is_safe_and_does_not_overwrite_existing_sidecar(tmp_path):
  recorder = MetricsCsvRecorder(tmp_path, lambda: "model", 0.02)
  first = recorder.start(["metric"], prefix="../../test/one")
  recorder.stop()
  assert first.parent == tmp_path / "test-one"
  assert first.name.startswith("test-one_metrics_")
  second = recorder.start(["metric"], prefix="../../test/one")
  recorder.stop()
  assert first != second
  assert first.exists()


def make_viewer(tmp_path: Path):
  env, command = make_env()
  viewer = object.__new__(ViserPlayViewer)
  BaseViewer.__init__(viewer, env, lambda obs: torch.zeros(2, 1))
  viewer._recording = None
  viewer._recording_cancel = Event()
  viewer._sim_lock = Lock()
  viewer._scene = SimpleNamespace(env_idx=0, needs_update=False)
  viewer._server = SimpleNamespace(scene=MagicMock())
  viewer._pending_update_reasons = set()
  viewer._threadpool = ThreadPoolExecutor(max_workers=1)
  plotter = MagicMock()
  plotter._recorder = MetricsCsvRecorder(tmp_path, lambda: "model", 0.02)
  plotter._loaded_csv = None
  viewer._term_overlays = SimpleNamespace(metrics_plotter=plotter)
  viewer._sync_ui_state = MagicMock()
  viewer.sync_viewer_to_env = MagicMock()
  return viewer, env, command, plotter


@pytest.mark.parametrize("cancel", [False, True])
def test_silent_recording_samples_every_step_without_frontend_updates(tmp_path, cancel):
  viewer, env, command, plotter = make_viewer(tmp_path)
  viewer.pause()
  old = command.command.clone()

  def step(action):
    env.counter += 1
    if cancel and env.counter == 4:
      viewer._recording_cancel.set()

  env.step = step
  try:
    viewer._start_metrics_recording(
      RecordingOptions(
        silent=True,
        duration_s=0.2,
        video=False,
        command_mode="fixed",
        velocity=(0.3, 0, 0),
      )
    )
    expected = 4 if cancel else 10
    assert viewer._recording is None and viewer._is_paused
    assert env.counter == expected
    path = plotter._recorder.path
    loaded = load_metrics_csv(path.read_bytes(), env.metrics_manager.active_terms)
    assert len(loaded.time_s) == expected
    assert viewer._server.scene.set_global_visibility.call_args_list[0].args == (False,)
    assert viewer._server.scene.set_global_visibility.call_args_list[-1].args == (True,)
    plotter.recording_started.assert_called_once_with(True)
    plotter.update.assert_not_called()
    torch.testing.assert_close(command.command, old)
  finally:
    viewer._threadpool.shutdown()


def test_silent_recording_restores_after_step_failure(tmp_path):
  viewer, env, command, plotter = make_viewer(tmp_path)
  old = command.command.clone()

  def step(action):
    env.counter += 1
    if env.counter == 3:
      raise RuntimeError("injected physics failure")

  env.step = step
  try:
    viewer._start_metrics_recording(
      RecordingOptions(
        silent=True, video=False, command_mode="fixed", velocity=(0.3, 0, 0)
      )
    )
    assert viewer._recording is None and viewer._is_paused
    assert not plotter._recorder.active
    assert len(plotter._recorder.path.read_text().splitlines()) == 3
    torch.testing.assert_close(command.command, old)
    assert viewer._server.scene.set_global_visibility.call_args.args == (True,)
  finally:
    viewer._threadpool.shutdown()


def test_control_step_capture_is_not_duplicated_by_plot_refresh(tmp_path):
  recorder = MetricsCsvRecorder(tmp_path, lambda: "m", 0.02)
  path = recorder.start(["metric"])
  recorder.append({"metric": 1}, 0, 1)
  plotter = object.__new__(ViserTermPlotter)
  plotter._loaded_csv = None
  plotter._implicit_step_idx = 0
  plotter._last_step_idx = None
  plotter._terms = {"metric": SimpleNamespace(enabled=False, history=deque())}
  plotter._recorder = recorder
  plotter.control_step_recording = True
  plotter.update([("metric", np.array([2]))], step_idx=2)
  recorder.stop()
  assert len(path.read_text().splitlines()) == 2


@pytest.mark.parametrize(
  ("term", "group"),
  [
    ("foot_pressure_pa/left_foot0", "足端信息"),
    ("joint_torque_nm/left_ankle_joint", "腿部关节"),
    ("joint_target_rad/right_arm_pitch_joint", "手臂关节"),
    ("joint_torque_nm/unknown_joint", "其他关节"),
    ("vel/act_x", "指令与机身"),
    ("new_metric", "其他指标"),
  ],
)
def test_semantic_groups_keep_every_term(term, group):
  assert metric_semantic_group(term) == group
