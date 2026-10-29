"""Control-step recording sessions, independent of browser refresh frequency."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import mediapy as media
import numpy as np

from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

if TYPE_CHECKING:
  from mjlab.viewer.viser.term_plotter import MetricsCsvRecorder


@dataclass(frozen=True)
class RecordingOptions:
  prefix: str = "experiment"
  silent: bool = False
  duration_s: float = 10.0
  command_mode: Literal["current", "fixed", "random"] = "current"
  command_name: str = "twist"
  velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
  lower: tuple[float, float, float] = (-0.3, -0.2, -0.3)
  upper: tuple[float, float, float] = (0.5, 0.2, 0.3)
  command_interval_s: float = 2.0
  seed: int = 42
  video: bool = True
  video_fps: int = 25
  view: Literal["side", "front", "three_quarter"] = "three_quarter"

  def validate(self) -> None:
    if not math.isfinite(self.duration_s) or not 0 < self.duration_s <= 3600:
      raise ValueError("录制时长必须为 0–3600 秒之间的正数（仿真时间）")
    if self.command_mode not in ("current", "fixed", "random"):
      raise ValueError("未知速度指令模式")
    if self.view not in ("side", "front", "three_quarter"):
      raise ValueError("未知视频视角")
    if self.video_fps not in (10, 25, 50):
      raise ValueError("视频帧率请选择 10、25 或 50")
    if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
      raise ValueError("随机种子必须为 0–4294967295 的整数")
    if not math.isfinite(self.command_interval_s) or self.command_interval_s <= 0:
      raise ValueError("随机指令切换间隔必须为正数")
    if any(len(v) != 3 for v in (self.velocity, self.lower, self.upper)):
      raise ValueError("速度需要 vx、vy、yaw 三个分量")
    if not all(math.isfinite(v) for v in (*self.velocity, *self.lower, *self.upper)):
      raise ValueError("速度及范围必须为有限数值")
    if any(lo > hi for lo, hi in zip(self.lower, self.upper, strict=True)):
      raise ValueError("随机指令下限不能超过上限")


class RecordingCommandSchedule:
  """Dedicated RNG: command seeds do not reseed physics or the policy."""

  def __init__(self, options: RecordingOptions, step_dt: float):
    options.validate()
    self.options = options
    self.interval_steps = max(1, math.ceil(options.command_interval_s / step_dt))
    self._rng = np.random.default_rng(options.seed)
    self._segment = -1
    self._value = options.velocity

  def value(self, sample: int) -> tuple[float, float, float]:
    segment = sample // self.interval_steps
    if self.options.command_mode == "random" and segment != self._segment:
      values = self._rng.uniform(self.options.lower, self.options.upper)
      self._value = (float(values[0]), float(values[1]), float(values[2]))
    self._segment = segment
    return self._value


class ControlStepRecording:
  """Stream all metrics and optional video from one selected simulation env.

  Construct, append and close on the simulation thread. No second rollout is
  started. Video is streamed rather than retaining unbounded RGB arrays in RAM.
  """

  def __init__(
    self,
    env: Any,
    recorder: MetricsCsvRecorder,
    options: RecordingOptions,
    env_idx: int,
  ) -> None:
    options.validate()
    self.env, self.recorder, self.options = env, recorder, options
    self.env_idx = env_idx
    self.step_dt = float(env.step_dt)
    if not math.isfinite(self.step_dt) or self.step_dt <= 0:
      raise ValueError("无效的仿真控制周期")
    self.target_samples = math.ceil(options.duration_s / self.step_dt)
    self.samples = 0
    self.schedule = RecordingCommandSchedule(options, self.step_dt)
    self.video_stride = max(1, math.ceil(1 / (self.step_dt * options.video_fps)))
    self.video_fps = 1 / (self.step_dt * self.video_stride)
    self.renderer: OffscreenRenderer | None = None
    self.writer: media.VideoWriter | None = None
    self.path: Path | None = None
    self.frame_map: list[dict[str, int | float]] = []
    self.command_events: list[dict[str, Any]] = []
    self._command: Any = None
    self._previous_override: Any = None
    self._original_velocity: Any = None
    self.closed = False

  def start(self) -> Path:
    # Tasks import environments, which import viewer modules during bootstrap.
    # Resolve the concrete command only after application setup is complete.
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

    try:
      if self.options.command_mode != "current":
        command = self.env.command_manager.get_term(self.options.command_name)
        if not isinstance(command, UniformVelocityCommand):
          raise ValueError("该任务不是速度指令任务，请选择 current（沿用任务指令）")
        self._command = command
        self._previous_override = command.recording_velocity_override
        self._original_velocity = command.command[self.env_idx].clone()
      if self.options.video:
        cfg = copy.copy(self.env.cfg.viewer)
        cfg.width, cfg.height = 1280, 720
        cfg.env_idx, cfg.max_extra_envs = self.env_idx, 0
        cfg.enable_shadows = cfg.enable_reflections = False
        if "robot" in self.env.scene.entities:
          cfg.entity_name = "robot"
          cfg.origin_type = ViewerConfig.OriginType.ASSET_ROOT
          cfg.azimuth = {"side": 90.0, "front": 0.0, "three_quarter": 45.0}[
            self.options.view
          ]
          cfg.elevation, cfg.distance = -15.0, 1.1
        sim = self.env.sim
        self.renderer = OffscreenRenderer(
          sim.mj_model, cfg, self.env.scene, sim.model, sim.expanded_fields
        )
        self.renderer.initialize()
      self.path = self.recorder.start(
        list(self.env.metrics_manager.active_terms), prefix=self.options.prefix
      )
      if self.options.video:
        self.writer = media.VideoWriter(
          self.path.with_suffix(".mp4"), shape=(720, 1280), fps=self.video_fps
        )
        self.writer.__enter__()
      self.prepare_command()
      return self.path
    except Exception:
      self.finish("start_failed")
      raise

  def prepare_command(self) -> None:
    if self._command is None:
      return
    value = self.schedule.value(self.samples)
    self._command.set_recording_velocity_override(self.env_idx, value)
    if not self.command_events or self.command_events[-1]["velocity"] != list(value):
      self.command_events.append(
        {
          "sample": self.samples,
          "time_s": self.samples * self.step_dt,
          "velocity": list(value),
        }
      )

  def append(self, sim_step: int) -> str | None:
    # Auto-reset has already replaced the terminal pose. Do not pair terminal
    # metrics with a reset robot frame: stop before writing either of them.
    terminal = bool(self.env.reset_buf[self.env_idx])
    if terminal and self.env.cfg.auto_reset:
      return "episode_ended_before_auto_reset_frame"
    values = {
      name: float(value[0])
      for name, value in self.env.metrics_manager.get_active_iterable_terms(
        self.env_idx
      )
    }
    self.recorder.append(values, self.env_idx, sim_step)
    sample = self.samples
    self.samples += 1
    if self.renderer is not None and sample % self.video_stride == 0:
      self.renderer.update(self.env.sim.data)
      assert self.writer is not None
      self.writer.add_image(self.renderer.render())
      self.frame_map.append(
        {
          "video_frame": len(self.frame_map),
          "csv_sample": sample,
          "sim_step": sim_step,
          "time_s": sample * self.step_dt,
          "video_time_s": len(self.frame_map) / self.video_fps,
        }
      )
    if terminal:
      return "episode_ended"
    if self.samples >= self.target_samples:
      return "duration_reached"
    return None

  def finish(self, reason: str) -> None:
    if self.closed:
      return
    self.closed = True
    errors: list[str] = []
    if self._command is not None:
      try:
        self._command.recording_velocity_override = self._previous_override
        if self._original_velocity is not None:
          self._command.command[self.env_idx] = self._original_velocity
      except Exception as exc:
        errors.append(f"restore command: {exc}")
    closers = [self.recorder.stop]
    if self.writer is not None:
      writer = self.writer
      closers.append(lambda: writer.__exit__(None, None, None))
    if self.renderer is not None:
      closers.append(self.renderer.close)
    for close in closers:
      try:
        close()
      except Exception as exc:
        errors.append(str(exc))
    if self.path is not None:
      metadata = {
        "version": 1,
        "options": asdict(self.options),
        "reason": reason,
        "env_idx": self.env_idx,
        "step_dt": self.step_dt,
        "samples": self.samples,
        "recorded_duration_s": self.samples * self.step_dt,
        "effective_video_fps": self.video_fps,
        "csv": self.path.name,
        "video": self.path.with_suffix(".mp4").name if self.options.video else None,
        "frames": self.frame_map,
        "commands": self.command_events,
        "errors": errors,
      }
      self.path.with_suffix(".recording.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
      )
    if errors:
      raise RuntimeError("; ".join(errors))
