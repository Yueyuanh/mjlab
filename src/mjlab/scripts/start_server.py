"""Local web dashboard for composing and launching MjLab experiments."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import secrets
import shlex
import signal
import socket
import subprocess
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

_MODEL_RE = re.compile(r"model_(\d+)\.pt$")
_TASK_RE = re.compile(r"task_id\s*=\s*['\"]([^'\"]+)['\"]")
_MOTION_RE = re.compile(r"^\s*motion_file:\s*(.*?)\s*$", re.MULTILINE)
_VISER_URL_RE = re.compile(
  r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(?P<port>\d+)"
)
_CACHE_SECONDS = 5.0
_MAX_BODY_BYTES = 1_000_000


class BuildError(ValueError):
  """Raised when the requested experiment cannot be constructed safely."""


@dataclass(frozen=True)
class Checkpoint:
  """The newest checkpoint discovered in one local run directory."""

  path: str
  log_root: str
  experiment: str
  run: str
  iteration: int
  modified: float
  motion_file: str | None = None

  @property
  def label(self) -> str:
    source = "pre_train" if self.path.startswith("pre_train/") else "logs"
    return f"[{source}] {self.experiment} / {self.run} / model_{self.iteration}.pt"


@dataclass(frozen=True)
class MotionFile:
  """A locally discovered motion archive."""

  path: str
  modified: float

  @property
  def label(self) -> str:
    return self.path.removeprefix("src/mjlab/mocap/npz/")


@dataclass(frozen=True)
class Catalog:
  """Tasks and artifacts available in the current checkout."""

  tasks: tuple[str, ...]
  checkpoints: tuple[Checkpoint, ...]
  motions: tuple[MotionFile, ...]
  scanned_at: float

  def to_json(self) -> dict[str, Any]:
    return {
      "tasks": list(self.tasks),
      "checkpoints": [
        asdict(item) | {"label": item.label} for item in self.checkpoints
      ],
      "motions": [asdict(item) | {"label": item.label} for item in self.motions],
      "scanned_at": self.scanned_at,
    }


@dataclass(frozen=True)
class BuildResult:
  """Validated command and the artifacts used to construct it."""

  argv: tuple[str, ...]
  checkpoint: str | None = None
  motion: str | None = None
  motion_reason: str | None = None
  warnings: tuple[str, ...] = ()
  gpu_ids: tuple[int, ...] | None = ()

  def to_json(self) -> dict[str, Any]:
    return {
      "argv": list(self.argv),
      "command": shlex.join(self.argv),
      "checkpoint": self.checkpoint,
      "motion": self.motion,
      "motion_reason": self.motion_reason,
      "warnings": list(self.warnings),
      "gpu_ids": None if self.gpu_ids is None else list(self.gpu_ids),
    }


def find_repo_root(start: Path | None = None) -> Path:
  """Find the MjLab checkout without importing its GPU-heavy modules."""
  candidates = [start or Path.cwd(), Path(__file__).resolve()]
  for candidate in candidates:
    candidate = candidate if candidate.is_dir() else candidate.parent
    for parent in (candidate, *candidate.parents):
      pyproject = parent / "pyproject.toml"
      if pyproject.is_file() and (parent / "src/mjlab").is_dir():
        return parent.resolve()
  raise RuntimeError("Could not find the MjLab repository root.")


def _relative_path(repo_root: Path, path: Path) -> str:
  try:
    return str(path.resolve().relative_to(repo_root))
  except ValueError:
    return str(path.resolve())


def discover_tasks(repo_root: Path) -> tuple[str, ...]:
  """Read task IDs from registrations without importing MuJoCo or Torch."""
  found: set[str] = set()
  task_root = repo_root / "src/mjlab/tasks"
  for source in task_root.rglob("*.py"):
    try:
      found.update(_TASK_RE.findall(source.read_text(encoding="utf-8")))
    except OSError:
      continue
  return tuple(sorted(found))


def _saved_motion(repo_root: Path, env_yaml: Path) -> str | None:
  try:
    match = _MOTION_RE.search(env_yaml.read_text(encoding="utf-8"))
  except OSError:
    return None
  if match is None:
    return None
  raw = match.group(1).strip().strip("'\"")
  if not raw or raw.lower() in {"null", "none"}:
    return None
  candidate = Path(raw).expanduser()
  if not candidate.is_absolute():
    candidate = repo_root / candidate
  if candidate.is_file():
    return _relative_path(repo_root, candidate)

  marker = "src/mjlab/"
  if marker in raw:
    portable = repo_root / (marker + raw.split(marker, 1)[1])
    if portable.is_file():
      return _relative_path(repo_root, portable)
  return None


def discover_checkpoints(repo_root: Path) -> tuple[Checkpoint, ...]:
  """Return the highest numbered model from every run below local log roots."""
  newest: dict[Path, tuple[int, Path]] = {}
  for base_name in ("logs", "pre_train"):
    base = repo_root / base_name
    if not base.is_dir():
      continue
    for path in base.rglob("model_*.pt"):
      match = _MODEL_RE.search(path.name)
      if match is None:
        continue
      iteration = int(match.group(1))
      current = newest.get(path.parent)
      if current is None or iteration > current[0]:
        newest[path.parent] = (iteration, path)

  checkpoints: list[Checkpoint] = []
  for run_dir, (iteration, path) in newest.items():
    experiment_dir = run_dir.parent
    log_root = experiment_dir.parent
    checkpoints.append(
      Checkpoint(
        path=_relative_path(repo_root, path),
        log_root=_relative_path(repo_root, log_root),
        experiment=experiment_dir.name,
        run=run_dir.name,
        iteration=iteration,
        modified=path.stat().st_mtime,
        motion_file=_saved_motion(repo_root, run_dir / "params/env.yaml"),
      )
    )
  return tuple(sorted(checkpoints, key=lambda item: item.modified, reverse=True))


def discover_motions(repo_root: Path) -> tuple[MotionFile, ...]:
  """Discover local NPZ reference motions, newest first."""
  root = repo_root / "src/mjlab/mocap/npz"
  if not root.is_dir():
    return ()
  motions = [
    MotionFile(path=_relative_path(repo_root, path), modified=path.stat().st_mtime)
    for path in root.rglob("*.npz")
  ]
  return tuple(sorted(motions, key=lambda item: item.modified, reverse=True))


def discover_catalog(repo_root: Path) -> Catalog:
  """Scan the checkout for task registrations and reusable artifacts."""
  return Catalog(
    tasks=discover_tasks(repo_root),
    checkpoints=discover_checkpoints(repo_root),
    motions=discover_motions(repo_root),
    scanned_at=time.time(),
  )


def _words(value: str) -> set[str]:
  return set(re.findall(r"[a-z0-9]+", value.lower()))


def recommend_checkpoint(
  task: str, checkpoints: tuple[Checkpoint, ...]
) -> Checkpoint | None:
  """Choose a recent checkpoint whose experiment/run best matches a task."""
  if not checkpoints:
    return None
  task_words = _words(task)

  def score(item: Checkpoint) -> tuple[int, float]:
    haystack = _words(f"{item.experiment} {item.run}")
    points = 0
    for family in ("tracking", "velocity", "cartpole", "lift"):
      if family in task_words and family in haystack:
        points += 80
    for robot in ("pikachu", "g1", "go1", "yam"):
      if robot in task_words and robot in haystack:
        points += 35
    for intent in ("up", "down", "walk", "rough", "flat"):
      if intent in task_words and intent in haystack:
        points += 25
    return points, item.modified

  selected = max(checkpoints, key=score)
  return selected if score(selected)[0] >= 80 else None


def recommend_motion(
  task: str,
  checkpoint: Checkpoint | None,
  motions: tuple[MotionFile, ...],
) -> tuple[MotionFile | None, str | None]:
  """Resolve the recorded motion first, then fall back to semantic matching."""
  if not motions:
    return None, None
  by_path = {item.path: item for item in motions}
  if checkpoint is not None and checkpoint.motion_file in by_path:
    return by_path[checkpoint.motion_file], "来自 checkpoint 的 params/env.yaml"

  context = f"{task} {checkpoint.run if checkpoint else ''}".lower()

  def score(item: MotionFile) -> tuple[int, float]:
    name = item.path.lower()
    points = 0
    if "down" in context and "transition-down" in name:
      points += 120
    if "up" in context and "transition-up" in name:
      points += 120
    if "walk" in context and "walk" in name:
      points += 50
    if "stand" in context and "stand" in name:
      points += 90
    if "back" in context and re.search(r"walk_-|stand-m", name):
      points += 80
    if "forward" in context and re.search(r"walk_0\.|stand-0", name):
      points += 70
    if "pikachu" in context and "pikachu" in name:
      points += 10
    for robot in ("pikachu", "g1", "go1", "yam"):
      if robot in context and robot not in name:
        points -= 100
    return points, item.modified

  selected = max(motions, key=score)
  if score(selected)[0] < 50:
    return None, None
  reason = "按任务和 run 名称匹配（请在启动前确认）"
  return selected, reason


def _as_bool(value: Any) -> bool:
  if isinstance(value, bool):
    return value
  return str(value).lower() in {"1", "true", "yes", "on"}


def _text(payload: dict[str, Any], name: str) -> str:
  value = payload.get(name, "")
  return str(value).strip() if value is not None else ""


def _integer(payload: dict[str, Any], name: str, *, minimum: int = 1) -> int | None:
  raw = _text(payload, name)
  if not raw:
    return None
  try:
    value = int(raw)
  except ValueError as exc:
    raise BuildError(f"{name} 必须是整数。") from exc
  if value < minimum:
    raise BuildError(f"{name} 必须不小于 {minimum}。")
  return value


def _number(
  payload: dict[str, Any], name: str, *, minimum: float | None = None
) -> float | None:
  raw = _text(payload, name)
  if not raw:
    return None
  try:
    value = float(raw)
  except ValueError as exc:
    raise BuildError(f"{name} 必须是数字。") from exc
  if minimum is not None and value < minimum:
    raise BuildError(f"{name} 必须不小于 {minimum}。")
  return value


def _command_path(repo_root: Path, raw: str, suffix: str) -> str:
  if not raw:
    raise BuildError(f"请填写 {suffix} 文件路径。")
  path = Path(raw).expanduser()
  if not path.is_absolute():
    path = repo_root / path
  if not path.is_file():
    raise BuildError(f"文件不存在：{raw}")
  if path.suffix.lower() != suffix:
    raise BuildError(f"文件必须是 {suffix}：{raw}")
  return _relative_path(repo_root, path)


def _checkpoint_from_payload(
  repo_root: Path,
  payload: dict[str, Any],
  catalog: Catalog,
  task: str,
) -> Checkpoint | None:
  mode = _text(payload, "checkpoint_mode") or "auto"
  if mode == "none":
    return None
  if mode == "auto":
    return recommend_checkpoint(task, catalog.checkpoints)
  raw = _text(payload, "checkpoint_manual" if mode == "manual" else "checkpoint")
  path = _command_path(repo_root, raw, ".pt")
  known = next((item for item in catalog.checkpoints if item.path == path), None)
  if known is not None:
    return known
  resolved = repo_root / path if not Path(path).is_absolute() else Path(path)
  iteration_match = _MODEL_RE.search(resolved.name)
  iteration = int(iteration_match.group(1)) if iteration_match else -1
  run_dir = resolved.parent
  return Checkpoint(
    path=path,
    log_root=_relative_path(repo_root, run_dir.parent.parent),
    experiment=run_dir.parent.name,
    run=run_dir.name,
    iteration=iteration,
    modified=resolved.stat().st_mtime,
    motion_file=_saved_motion(repo_root, run_dir / "params/env.yaml"),
  )


def _motion_from_payload(
  repo_root: Path,
  payload: dict[str, Any],
  catalog: Catalog,
  task: str,
  checkpoint: Checkpoint | None,
) -> tuple[str | None, str | None]:
  mode = _text(payload, "motion_mode") or "auto"
  if mode == "none":
    return None, None
  if mode == "auto":
    motion, reason = recommend_motion(task, checkpoint, catalog.motions)
    return (motion.path if motion else None), reason
  raw = _text(payload, "motion_manual" if mode == "manual" else "motion")
  return _command_path(repo_root, raw, ".npz"), "手动选择"


def _gpu_ids(payload: dict[str, Any]) -> tuple[tuple[int, ...] | None, str]:
  raw = _text(payload, "gpu_ids") or "0"
  if raw.lower() == "all":
    return None, "all"
  if raw.lower() in {"cpu", "none", "[]"}:
    return (), "[]"
  try:
    values = tuple(dict.fromkeys(int(part.strip()) for part in raw.split(",")))
  except ValueError as exc:
    raise BuildError("GPU 必须填写 0、0,1、all 或 cpu。") from exc
  if any(value < 0 for value in values):
    raise BuildError("GPU 编号不能为负数。")
  return values, json.dumps(list(values))


def _add(command: list[str], flag: str, value: Any | None) -> None:
  if value is not None and str(value) != "":
    command.extend((flag, str(value)))


def _add_extra(command: list[str], payload: dict[str, Any]) -> None:
  raw = _text(payload, "extra_args")
  if not raw:
    return
  try:
    command.extend(shlex.split(raw))
  except ValueError as exc:
    raise BuildError(f"额外参数无法解析：{exc}") from exc


def build_experiment_command(
  repo_root: Path, payload: dict[str, Any], catalog: Catalog
) -> BuildResult:
  """Validate a dashboard request and turn it into a shell-free argv list."""
  kind = _text(payload, "kind") or "train"
  task = _text(payload, "task")
  if task not in catalog.tasks:
    raise BuildError("请选择一个已注册的任务。")
  tracking = "Tracking" in task
  warnings: list[str] = []
  checkpoint: Checkpoint | None = None
  motion: str | None = None
  motion_reason: str | None = None
  selected_gpus: tuple[int, ...] | None = ()

  if kind == "train":
    command = ["uv", "run", "train", task]
    resume = _as_bool(payload.get("resume"))
    checkpoint_for_motion = None
    if resume or (tracking and _text(payload, "motion_mode") in {"", "auto"}):
      checkpoint_for_motion = _checkpoint_from_payload(
        repo_root, payload, catalog, task
      )
    selected_gpus, gpu_arg = _gpu_ids(payload)
    _add(command, "--gpu-ids", gpu_arg)
    _add(command, "--env.scene.num-envs", _integer(payload, "num_envs"))
    _add(command, "--env.decimation", _integer(payload, "decimation"))
    _add(command, "--agent.max-iterations", _integer(payload, "max_iterations"))
    _add(command, "--agent.run-name", _text(payload, "run_name"))

    if tracking:
      motion, motion_reason = _motion_from_payload(
        repo_root, payload, catalog, task, checkpoint_for_motion
      )
      if motion is None and not _text(payload, "registry_name"):
        raise BuildError("Tracking 训练需要选择 NPZ，或填写 W&B motion registry。")
      _add(command, "--env.commands.motion.motion-file", motion)
      _add(command, "--registry-name", _text(payload, "registry_name"))

    logger = _text(payload, "logger")
    if logger and logger != "default":
      _add(command, "--agent.logger", logger)
    _add(command, "--agent.wandb-project", _text(payload, "wandb_project"))
    tags = [part.strip() for part in _text(payload, "wandb_tags").split(",")]
    tags = [tag for tag in tags if tag]
    if tags:
      _add(command, "--agent.wandb-tags", repr(tuple(tags)))
    _add(
      command,
      "--agent.algorithm.learning-rate",
      _number(payload, "learning_rate", minimum=0.0),
    )
    _add(
      command,
      "--agent.algorithm.clip-param",
      _number(payload, "clip_param", minimum=0.0),
    )
    _add(
      command,
      "--agent.algorithm.num-mini-batches",
      _integer(payload, "num_mini_batches"),
    )
    hidden_dims = _text(payload, "hidden_dims")
    if hidden_dims:
      dims = [part.strip() for part in hidden_dims.strip("()[]").split(",")]
      if not dims or any(not part.isdigit() or int(part) < 1 for part in dims):
        raise BuildError("hidden_dims 应类似 512,256,128。")
      _add(command, "--agent.actor.hidden-dims", f"({','.join(dims)})")
    if _as_bool(payload.get("obs_normalization")):
      _add(command, "--agent.actor.obs-normalization", "True")
    if _as_bool(payload.get("nan_guard")):
      _add(command, "--enable-nan-guard", "True")
    if _as_bool(payload.get("video")):
      _add(command, "--video", "True")

    if resume:
      checkpoint = checkpoint_for_motion
      if checkpoint is None:
        raise BuildError("续训需要选择一个本地 checkpoint。")
      _add(command, "--log-root", checkpoint.log_root)
      _add(command, "--agent.resume", "True")
      _add(command, "--agent.load-run", f"^{re.escape(checkpoint.run)}$")
      _add(
        command,
        "--agent.load-checkpoint",
        f"^{re.escape(Path(checkpoint.path).name)}$",
      )
      if checkpoint.log_root.startswith("pre_train"):
        warnings.append("续训日志也会写入 pre_train；如需隔离请先复制 run 到 logs。")

  elif kind == "play":
    command = ["uv", "run", "play", task]
    agent = _text(payload, "agent") or "trained"
    if agent not in {"trained", "zero", "random"}:
      raise BuildError("无效的 play agent。")
    _add(command, "--agent", agent)
    _add(command, "--viewer", _text(payload, "viewer") or "viser")
    _add(command, "--device", _text(payload, "device"))
    _add(command, "--num-envs", _integer(payload, "play_num_envs"))

    if agent == "trained":
      checkpoint = _checkpoint_from_payload(repo_root, payload, catalog, task)
      wandb_path = _text(payload, "wandb_run_path")
      if checkpoint is not None:
        _add(command, "--checkpoint-file", checkpoint.path)
      elif wandb_path:
        _add(command, "--wandb-run-path", wandb_path)
        _add(
          command,
          "--wandb-checkpoint-name",
          _text(payload, "wandb_checkpoint_name"),
        )
      else:
        raise BuildError("训练策略回放需要本地 checkpoint 或 W&B run path。")
    motion, motion_reason = _motion_from_payload(
      repo_root, payload, catalog, task, checkpoint
    )
    _add(command, "--motion-file", motion)
    _add(command, "--registry-name", _text(payload, "registry_name"))
    if tracking and motion is None and not _text(payload, "wandb_run_path"):
      if not _text(payload, "registry_name"):
        raise BuildError("Tracking 回放需要 NPZ、W&B run 或 motion registry。")
    sampling = _text(payload, "sampling_mode")
    if sampling and sampling != "default":
      _add(command, "--sampling-mode", sampling)
    if _as_bool(payload.get("no_terminations")):
      _add(command, "--no-terminations", "True")
    if _as_bool(payload.get("play_video")):
      _add(command, "--video", "True")
      _add(command, "--video-length", _integer(payload, "video_length"))
    if (_text(payload, "device") or "").startswith("cuda:"):
      try:
        selected_gpus = (int(_text(payload, "device").split(":", 1)[1]),)
      except ValueError:
        selected_gpus = ()

  elif kind == "response":
    command = ["uv", "run", "response", task]
    _add(command, "--mode", _text(payload, "response_mode") or "all")
    _add(command, "--joint-pattern", _text(payload, "joint_pattern") or ".*")
    _add(command, "--device", _text(payload, "response_device") or "cpu")
    _add(command, "--output-root", _text(payload, "output_root"))
    for name, flag in (
      ("suspend_height", "--suspend-height-m"),
      ("limit_margin", "--limit-margin-fraction"),
      ("step_amplitude", "--step-amplitude-fraction"),
      ("step_pre", "--step-pre-s"),
      ("step_duration", "--step-duration-s"),
      ("sine_amplitude", "--sine-amplitude-fraction"),
      ("sine_pre", "--sine-pre-s"),
      ("sine_cycles", "--sine-cycles"),
      ("sine_warmup", "--sine-warmup-cycles"),
      ("torque_safety", "--torque-safety-factor"),
      ("speed_safety", "--speed-safety-factor"),
    ):
      _add(command, flag, _number(payload, name, minimum=0.0))
    frequencies = _text(payload, "sine_frequencies")
    if frequencies:
      try:
        parsed = [float(part.strip()) for part in frequencies.split(",")]
      except ValueError as exc:
        raise BuildError("正弦频率应为逗号分隔的数字。") from exc
      if not parsed or any(value <= 0 for value in parsed):
        raise BuildError("正弦频率必须全部大于 0。")
      _add(
        command,
        "--sine-frequencies-hz",
        f"({','.join(str(value) for value in parsed)})",
      )
    _add(command, "--sample-stride", _integer(payload, "sample_stride"))
    _add(
      command,
      "--strict-no-contact",
      "True" if _as_bool(payload.get("strict_no_contact")) else "False",
    )
    response_device = _text(payload, "response_device")
    if response_device.startswith("cuda:"):
      try:
        selected_gpus = (int(response_device.split(":", 1)[1]),)
      except ValueError:
        selected_gpus = ()

  elif kind == "evaluate":
    if not tracking:
      raise BuildError("定量 evaluate 只支持 Tracking 任务。")
    script = "src/mjlab/tasks/tracking/scripts/evaluate.py"
    command = ["uv", "run", "python", script, task]
    wandb_path = _text(payload, "eval_wandb_run_path")
    if not wandb_path:
      raise BuildError("定量 evaluate 必须填写 W&B run path。")
    _add(command, "--wandb-run-path", wandb_path)
    _add(
      command,
      "--wandb-checkpoint-name",
      _text(payload, "eval_wandb_checkpoint_name"),
    )
    motion_checkpoint = _checkpoint_from_payload(repo_root, payload, catalog, task)
    motion, motion_reason = _motion_from_payload(
      repo_root, payload, catalog, task, motion_checkpoint
    )
    _add(command, "--motion-file", motion)
    _add(command, "--num-envs", _integer(payload, "eval_num_envs"))
    device = _text(payload, "eval_device")
    _add(command, "--device", device)
    _add(command, "--output-file", _text(payload, "eval_output_file"))
    if device.startswith("cuda:"):
      try:
        selected_gpus = (int(device.split(":", 1)[1]),)
      except ValueError:
        selected_gpus = ()
  else:
    raise BuildError("未知实验类型。")

  _add_extra(command, payload)
  return BuildResult(
    argv=tuple(command),
    checkpoint=checkpoint.path if checkpoint else None,
    motion=motion,
    motion_reason=motion_reason,
    warnings=tuple(warnings),
    gpu_ids=selected_gpus,
  )


@dataclass
class ManagedJob:
  """A process started by this dashboard."""

  job_id: str
  kind: str
  task: str
  command: str
  pid: int
  log_file: Path
  started_at: float
  process: subprocess.Popen[bytes] = field(repr=False)
  viewer_port: int | None = None

  def to_json(self) -> dict[str, Any]:
    log_tail = _tail(self.log_file)
    url_matches = list(_VISER_URL_RE.finditer(log_tail))
    actual_viewer_port = (
      int(url_matches[-1].group("port")) if url_matches else self.viewer_port
    )
    return {
      "id": self.job_id,
      "kind": self.kind,
      "task": self.task,
      "command": self.command,
      "pid": self.pid,
      "started_at": self.started_at,
      "status": _session_health(self.pid),
      "return_code": self.process.poll(),
      "log_file": str(self.log_file),
      "log_tail": log_tail,
      "viewer_port": actual_viewer_port,
      "viewer_ready": bool(url_matches),
    }

  def is_running(self) -> bool:
    return _session_health(self.pid) == "running"


def _tail(path: Path, byte_count: int = 16_000) -> str:
  try:
    with path.open("rb") as stream:
      stream.seek(0, os.SEEK_END)
      size = stream.tell()
      stream.seek(max(0, size - byte_count))
      return stream.read().decode("utf-8", errors="replace")
  except OSError:
    return ""


def _process_status(pid: int) -> tuple[str, int] | None:
  """Return the Linux process state and parent PID."""
  try:
    status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
  except (OSError, PermissionError):
    return None
  state_match = re.search(r"^State:\s+([A-Z])", status, re.MULTILINE)
  parent_match = re.search(r"^PPid:\s+(\d+)", status, re.MULTILINE)
  if state_match is None or parent_match is None:
    return None
  return state_match.group(1), int(parent_match.group(1))


def _process_snapshot() -> dict[int, dict[str, Any]]:
  """Read enough /proc metadata to identify live work and dead launchers."""
  snapshot: dict[int, dict[str, Any]] = {}
  for proc_dir in Path("/proc").glob("[0-9]*"):
    pid = int(proc_dir.name)
    status = _process_status(pid)
    if status is None:
      continue
    state, parent = status
    try:
      raw = (proc_dir / "cmdline").read_bytes()
      args = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
    except (OSError, PermissionError):
      args = []
    try:
      cwd: Path | None = (proc_dir / "cwd").resolve()
    except (OSError, PermissionError):
      cwd = None
    try:
      group = os.getpgid(pid)
      session = os.getsid(pid)
    except (OSError, ProcessLookupError):
      continue
    snapshot[pid] = {
      "state": state,
      "parent": parent,
      "args": args,
      "cwd": cwd,
      "group": group,
      "session": session,
    }
  return snapshot


def _descendant_pids(
  root_pid: int, snapshot: dict[int, dict[str, Any]]
) -> set[int]:
  children: dict[int, list[int]] = {}
  for pid, info in snapshot.items():
    children.setdefault(info["parent"], []).append(pid)
  found: set[int] = set()
  pending = list(children.get(root_pid, ()))
  while pending:
    pid = pending.pop()
    if pid in found:
      continue
    found.add(pid)
    pending.extend(children.get(pid, ()))
  return found


def _is_launcher_helper(args: list[str]) -> bool:
  """Return true for wrappers that do not themselves execute the workload."""
  if not args:
    return False
  command = " ".join(args).lower()
  executable = Path(args[0]).name.lower()
  if "multiprocessing.resource_tracker" in command:
    return True
  if "torchrunx" in command:
    return True
  if executable in {"uv", "timeout"}:
    return True
  return executable in {"sh", "bash", "dash"} and (
    "uv run" in command or "torchrunx" in command
  )


def _is_stale_process_tree(
  root_pid: int, snapshot: dict[int, dict[str, Any]]
) -> bool:
  """Detect a launcher whose real workers exited but were never reaped."""
  descendants = _descendant_pids(root_pid, snapshot)
  if not descendants:
    return False
  if not any(snapshot[pid]["state"] == "Z" for pid in descendants):
    return False
  live_descendants = [
    snapshot[pid]
    for pid in descendants
    if snapshot[pid]["state"] != "Z"
  ]
  return all(_is_launcher_helper(info["args"]) for info in live_descendants)


def _session_health(session_id: int) -> str:
  snapshot = _process_snapshot()
  members = [info for info in snapshot.values() if info["session"] == session_id]
  live = [info for info in members if info["state"] != "Z"]
  if not live:
    return "exited"
  has_zombie = any(info["state"] == "Z" for info in members)
  if has_zombie and all(_is_launcher_helper(info["args"]) for info in live):
    return "stale"
  return "running"


def _live_session_pids(session_id: int) -> list[int]:
  snapshot = _process_snapshot()
  return sorted(
    pid
    for pid, info in snapshot.items()
    if info["session"] == session_id and info["state"] != "Z"
  )


def _signal_session(session_id: int, sig: signal.Signals) -> None:
  """Signal every live member, including workers that changed process group."""
  for pid in reversed(_live_session_pids(session_id)):
    try:
      os.kill(pid, sig)
    except ProcessLookupError:
      continue


def _find_free_tcp_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
    stream.bind(("0.0.0.0", 0))
    return int(stream.getsockname()[1])


def inspect_gpus() -> dict[str, Any]:
  """Query NVIDIA GPUs and their compute processes without importing Torch."""
  query = "index,uuid,name,memory.used,memory.total,utilization.gpu"
  try:
    gpu_run = subprocess.run(
      ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
      check=True,
      capture_output=True,
      text=True,
      timeout=3,
    )
  except (FileNotFoundError, subprocess.SubprocessError):
    return {"available": False, "gpus": [], "processes": []}

  gpus: list[dict[str, Any]] = []
  uuid_to_index: dict[str, int] = {}
  for line in gpu_run.stdout.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 6:
      continue
    try:
      index = int(parts[0])
      memory_used = int(parts[3])
      memory_total = int(parts[4])
      utilization = int(parts[5])
    except ValueError:
      continue
    uuid_to_index[parts[1]] = index
    gpus.append(
      {
        "index": index,
        "name": parts[2],
        "memory_used_mb": memory_used,
        "memory_total_mb": memory_total,
        "utilization": utilization,
      }
    )

  processes: list[dict[str, Any]] = []
  try:
    proc_run = subprocess.run(
      [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
        "--format=csv,noheader,nounits",
      ],
      check=True,
      capture_output=True,
      text=True,
      timeout=3,
    )
    for line in proc_run.stdout.splitlines():
      parts = [part.strip() for part in line.split(",")]
      if len(parts) != 4 or not parts[0].isdigit():
        continue
      pid = int(parts[0])
      status = _process_status(pid)
      processes.append(
        {
          "pid": pid,
          "name": parts[1],
          "memory_mb": int(parts[2]),
          "gpu_index": uuid_to_index.get(parts[3]),
          "state": status[0] if status else None,
          "stale": status is None or status[0] == "Z",
          "session_id": os.getsid(pid) if status else None,
        }
      )
  except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
    pass
  return {"available": True, "gpus": gpus, "processes": processes}


def _process_group(pid: int) -> int | None:
  try:
    return os.getpgid(pid)
  except (OSError, ProcessLookupError):
    return None


def _experiment_kind(args: list[str]) -> str | None:
  """Classify supported MjLab commands, including absolute entry-point paths."""
  lowered = [arg.lower() for arg in args]
  command = " ".join(lowered)
  if "start_server" in command:
    return None
  basenames = {Path(arg).name.lower() for arg in args[:4]}
  if "train" in basenames or "mjlab.scripts.train" in lowered:
    return "train"
  if "play" in basenames or "mjlab.scripts.play" in lowered:
    return "play"
  if "response" in basenames or "mjlab.response.cli" in lowered:
    return "response"
  if "evaluate.py" in basenames or "tracking/scripts/evaluate.py" in command:
    return "evaluate"
  if "torchrunx" in command:
    return "train"
  return None


def discover_external_jobs(
  repo_root: Path, ignored_sessions: set[int]
) -> list[dict[str, Any]]:
  """Find MjLab train/play/response/evaluate processes not owned by this UI."""
  snapshot = _process_snapshot()
  candidates: dict[int, dict[str, Any]] = {}
  own_pid = os.getpid()
  for pid, info in snapshot.items():
    if (
      pid == own_pid
      or info["session"] in ignored_sessions
      or info["state"] == "Z"
    ):
      continue
    args = info["args"]
    command = " ".join(args)
    kind = _experiment_kind(args)
    if kind is None:
      continue
    cwd = info["cwd"]
    try:
      in_repo = cwd is not None and (cwd == repo_root or repo_root in cwd.parents)
    except OSError:
      in_repo = False
    if not in_repo and str(repo_root) not in command:
      continue
    candidates[pid] = {
      "pid": pid,
      "kind": kind,
      "command": command[:500],
      "stale": _is_stale_process_tree(pid, snapshot),
    }

  # A timeout/shell/uv wrapper and its child describe one task, not several.
  roots: list[dict[str, Any]] = []
  candidate_pids = set(candidates)
  for pid, item in candidates.items():
    parent = snapshot[pid]["parent"]
    seen: set[int] = set()
    nested = False
    while parent in snapshot and parent not in seen:
      if parent in candidate_pids:
        nested = True
        break
      seen.add(parent)
      parent = snapshot[parent]["parent"]
    if not nested:
      roots.append(item)

  # Collapse identical commands while retaining every PID for diagnostics.
  grouped: dict[tuple[str, str, bool], dict[str, Any]] = {}
  for item in roots:
    key = (item["kind"], item["command"], item["stale"])
    existing = grouped.get(key)
    if existing is None:
      grouped[key] = item | {"pids": [item["pid"]], "count": 1}
    else:
      existing["pids"].append(item["pid"])
      existing["count"] += 1
  return sorted(grouped.values(), key=lambda item: (item["stale"], item["pid"]))


class Dashboard:
  """Shared state for HTTP request handlers."""

  def __init__(self, repo_root: Path, ssh_target: str | None = None):
    self.repo_root = repo_root
    self.ssh_target = ssh_target or (
      f"{getpass.getuser()}@{socket.gethostname()}"
    )
    self.token = secrets.token_urlsafe(24)
    self._catalog: Catalog | None = None
    self._catalog_time = 0.0
    self._gpu: dict[str, Any] | None = None
    self._gpu_time = 0.0
    self._jobs: dict[str, ManagedJob] = {}
    self._lock = threading.RLock()

  def catalog(self, refresh: bool = False) -> Catalog:
    with self._lock:
      stale = time.monotonic() - self._catalog_time > _CACHE_SECONDS
      if refresh or self._catalog is None or stale:
        self._catalog = discover_catalog(self.repo_root)
        self._catalog_time = time.monotonic()
      return self._catalog

  def gpu_state(self, refresh: bool = False) -> dict[str, Any]:
    with self._lock:
      stale = time.monotonic() - self._gpu_time > 3.0
      if refresh or self._gpu is None or stale:
        self._gpu = inspect_gpus()
        self._gpu_time = time.monotonic()
      return self._gpu

  def _managed_sessions(self) -> set[int]:
    return {
      job.pid for job in self._jobs.values() if _live_session_pids(job.pid)
    }

  def conflicts(self, gpu_ids: tuple[int, ...] | None = ()) -> list[dict[str, Any]]:
    with self._lock:
      managed_sessions = self._managed_sessions()
      conflicts = [
        {
          "type": "managed",
          "pid": job.pid,
          "message": f"网页任务仍在运行：{job.kind} / {job.task}",
        }
        for job in self._jobs.values()
        if job.is_running()
      ]
      external = discover_external_jobs(self.repo_root, managed_sessions)
      conflicts.extend(
        {
          "type": "external",
          "pid": item["pid"],
          "message": f"检测到外部 {item['kind']}：{item['command']}",
        }
        for item in external
        if not item["stale"]
      )
      if gpu_ids == ():
        return conflicts
      gpu_state = self.gpu_state()
      for process in gpu_state["processes"]:
        if (
          process["pid"] == os.getpid()
          or process["stale"]
          or process["session_id"] in managed_sessions
        ):
          continue
        selected = gpu_ids is None or process["gpu_index"] in gpu_ids
        if selected and not any(item["pid"] == process["pid"] for item in conflicts):
          conflicts.append(
            {
              "type": "gpu",
              "pid": process["pid"],
              "message": (
                f"GPU {process['gpu_index']} 已被 PID {process['pid']} "
                f"占用 {process['memory_mb']} MiB（{process['name']}）"
              ),
            }
          )
      return conflicts

  def preview(self, payload: dict[str, Any]) -> dict[str, Any]:
    result = build_experiment_command(self.repo_root, payload, self.catalog())
    data = result.to_json()
    data["conflicts"] = self.conflicts(result.gpu_ids)
    return data

  def start(self, payload: dict[str, Any]) -> dict[str, Any]:
    with self._lock:
      result = build_experiment_command(self.repo_root, payload, self.catalog(True))
      conflicts = self.conflicts(result.gpu_ids)
      if conflicts and not _as_bool(payload.get("allow_conflict")):
        messages = "；".join(item["message"] for item in conflicts)
        raise BuildError(f"为避免任务冲突，已拒绝启动：{messages}")

      jobs_dir = self.repo_root / ".start_server/jobs"
      jobs_dir.mkdir(parents=True, exist_ok=True)
      kind = _text(payload, "kind") or "train"
      task = _text(payload, "task")
      stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      job_id = f"{stamp}_{secrets.token_hex(3)}"
      child_env: dict[str, str] | None = None
      viewer_port: int | None = None
      viewer = _text(payload, "viewer") or "auto"
      uses_viser = kind == "play" and (
        viewer == "viser"
        or (viewer == "auto" and not os.environ.get("DISPLAY"))
      )
      if uses_viser:
        viewer_port = _find_free_tcp_port()
        child_env = os.environ.copy() | {"_VISER_PORT_OVERRIDE": str(viewer_port)}
      log_file = jobs_dir / f"{job_id}_{kind}.log"
      stream = log_file.open("wb")
      stream.write(f"$ {shlex.join(result.argv)}\n\n".encode())
      stream.flush()
      try:
        process = subprocess.Popen(
          result.argv,
          cwd=self.repo_root,
          stdout=stream,
          stderr=subprocess.STDOUT,
          env=child_env,
          start_new_session=True,
        )
      except OSError:
        stream.close()
        raise
      stream.close()
      job = ManagedJob(
        job_id=job_id,
        kind=kind,
        task=task,
        command=shlex.join(result.argv),
        pid=process.pid,
        log_file=log_file,
        started_at=time.time(),
        process=process,
        viewer_port=viewer_port,
      )
      self._jobs[job_id] = job
      return {"job": job.to_json(), "preview": result.to_json()}

  def stop(self, job_id: str) -> dict[str, Any]:
    with self._lock:
      job = self._jobs.get(job_id)
      if job is None:
        raise BuildError("找不到这个网页启动的任务。")
      if _live_session_pids(job.pid):
        _signal_session(job.pid, signal.SIGTERM)
        threading.Thread(target=self._kill_later, args=(job,), daemon=True).start()
      return job.to_json()

  @staticmethod
  def _kill_later(job: ManagedJob) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
      if not _live_session_pids(job.pid):
        return
      time.sleep(0.2)
    _signal_session(job.pid, signal.SIGKILL)

  def state(self, refresh: bool = False) -> dict[str, Any]:
    with self._lock:
      catalog = self.catalog(refresh)
      gpu = self.gpu_state(refresh)
      external = discover_external_jobs(self.repo_root, self._managed_sessions())
      stale_jobs = [item for item in external if item["stale"]]
      stale_gpu_processes = [
        process
        for process in gpu["processes"]
        if process["stale"]
      ]
      return {
        "repo_root": str(self.repo_root),
        "ssh_target": self.ssh_target,
        "catalog": catalog.to_json(),
        "gpu": gpu,
        "jobs": [job.to_json() for job in reversed(self._jobs.values())],
        "conflicts": self.conflicts(),
        "external_jobs": [item for item in external if not item["stale"]],
        "stale_jobs": stale_jobs,
        "stale_gpu_processes": stale_gpu_processes,
      }


def make_handler(app: Dashboard) -> type[BaseHTTPRequestHandler]:
  """Bind dashboard state to a small same-origin JSON API."""

  class Handler(BaseHTTPRequestHandler):
    server_version = "MjLabStartServer/1.0"

    def log_message(self, format: str, *args: Any) -> None:
      print(f"[start_server] {self.address_string()} - {format % args}")

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
      encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
      self.send_response(status)
      self.send_header("Content-Type", "application/json; charset=utf-8")
      self.send_header("Content-Length", str(len(encoded)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(encoded)

    def _send_html(self) -> None:
      html = _HTML.replace("__TOKEN_JSON__", json.dumps(app.token))
      encoded = html.encode("utf-8")
      self.send_response(HTTPStatus.OK)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(encoded)))
      self.send_header("Cache-Control", "no-store")
      self.send_header("X-Content-Type-Options", "nosniff")
      self.send_header("Referrer-Policy", "no-referrer")
      self.end_headers()
      self.wfile.write(encoded)

    def _payload(self) -> dict[str, Any]:
      if self.headers.get("X-Start-Server-Token") != app.token:
        raise PermissionError("Invalid request token.")
      try:
        size = int(self.headers.get("Content-Length", "0"))
      except ValueError as exc:
        raise BuildError("无效的请求长度。") from exc
      if size <= 0 or size > _MAX_BODY_BYTES:
        raise BuildError("请求为空或过大。")
      try:
        value = json.loads(self.rfile.read(size))
      except json.JSONDecodeError as exc:
        raise BuildError("请求不是有效 JSON。") from exc
      if not isinstance(value, dict):
        raise BuildError("请求必须是 JSON object。")
      return value

    def do_GET(self) -> None:  # noqa: N802
      parsed = urlparse(self.path)
      if parsed.path in {"/", "/index.html"}:
        self._send_html()
        return
      if parsed.path == "/api/state":
        refresh = parse_qs(parsed.query).get("refresh") == ["1"]
        self._send_json(app.state(refresh))
        return
      if parsed.path == "/favicon.ico":
        self.send_error(HTTPStatus.NOT_FOUND)
        return
      self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
      try:
        payload = self._payload()
        if self.path == "/api/preview":
          self._send_json(app.preview(payload))
        elif self.path == "/api/start":
          self._send_json(app.start(payload), HTTPStatus.CREATED)
        elif self.path == "/api/stop":
          self._send_json(app.stop(_text(payload, "job_id")))
        else:
          self.send_error(HTTPStatus.NOT_FOUND)
      except PermissionError as exc:
        self._send_json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
      except (BuildError, OSError) as exc:
        self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
      except Exception as exc:  # Keep the local UI usable after one bad request.
        self._send_json(
          {"error": f"内部错误：{type(exc).__name__}: {exc}"},
          HTTPStatus.INTERNAL_SERVER_ERROR,
        )

  return Handler


def main() -> None:
  """Start the local experiment dashboard and optionally open a browser."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8010)
  parser.add_argument("--no-open", action="store_true")
  parser.add_argument(
    "--ssh-target",
    default=None,
    help="SSH destination used in generated Viser port-forward commands.",
  )
  parser.add_argument(
    "--allow-remote",
    action="store_true",
    help="Allow binding to a non-loopback host (POST requests still need a token).",
  )
  args = parser.parse_args()
  if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote:
    parser.error("Non-loopback --host requires --allow-remote.")

  app = Dashboard(find_repo_root(), ssh_target=args.ssh_target)
  server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
  actual_port = server.server_address[1]
  browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
  url = f"http://{browser_host}:{actual_port}/"
  print(f"MjLab experiment dashboard: {url}")
  print(f"Repository: {app.repo_root}")
  print("Press Ctrl+C to stop the dashboard; launched experiments keep running.")
  if not args.no_open:
    threading.Timer(0.4, webbrowser.open, args=(url,)).start()
  try:
    server.serve_forever(poll_interval=0.25)
  except KeyboardInterrupt:
    print("\nStopping dashboard (experiments are not terminated).")
  finally:
    server.server_close()


_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MjLab 实验启动台</title>
  <style>
    :root { --ink:#18201d; --muted:#69726e; --line:#dfe5e1; --paper:#fff;
      --bg:#f3f6f3; --brand:#f06c2e; --brand2:#d84d15; --green:#24805d;
      --red:#b8403b; --shadow:0 12px 36px rgba(24,32,29,.08); }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--bg); font:14px/1.5
      Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
    button,input,select,textarea { font:inherit; }
    .shell { max-width:1440px; margin:auto; padding:28px; }
    header { display:flex; align-items:flex-end; justify-content:space-between;
      gap:20px; margin-bottom:20px; }
    h1 { margin:0; font-size:28px; letter-spacing:-.04em; }
    .subtitle { color:var(--muted); margin-top:4px; }
    .status { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
    .pill { padding:6px 10px; border:1px solid var(--line); border-radius:999px;
      background:var(--paper); color:var(--muted); }
    .pill.good { color:var(--green); border-color:#b9ddcf; background:#effaf5; }
    .pill.bad { color:var(--red); border-color:#ecc5c2; background:#fff4f3; }
    .layout { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(340px,.65fr);
      gap:20px; align-items:start; }
    .card { background:var(--paper); border:1px solid var(--line); border-radius:18px;
      box-shadow:var(--shadow); padding:20px; }
    .tabs { display:grid; grid-template-columns:repeat(4,1fr); gap:7px; padding:6px;
      background:#edf1ee; border-radius:13px; margin-bottom:20px; }
    .tab { border:0; border-radius:9px; padding:10px 8px; background:transparent;
      color:var(--muted); cursor:pointer; }
    .tab.active { background:white; color:var(--ink); box-shadow:0 2px 8px #19211d14; }
    h2 { font-size:16px; margin:4px 0 14px; }
    h3 { font-size:14px; margin:20px 0 10px; padding-top:16px;
      border-top:1px solid var(--line); }
    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:13px; }
    .wide { grid-column:1/-1; }
    label { display:grid; gap:5px; color:#414a46; font-size:12px; font-weight:650; }
    input,select,textarea { width:100%; padding:9px 10px; color:var(--ink); background:white;
      border:1px solid #ccd4cf; border-radius:9px; outline:none; }
    textarea { min-height:64px; resize:vertical; }
    input:focus,select:focus,textarea:focus { border-color:var(--brand);
      box-shadow:0 0 0 3px #f06c2e1c; }
    .check { display:flex; align-items:center; gap:8px; min-height:38px; }
    .check input { width:16px; height:16px; accent-color:var(--brand); }
    .hint { font-size:12px; color:var(--muted); font-weight:400; }
    .hidden { display:none !important; }
    .side { display:grid; gap:16px; position:sticky; top:16px; }
    .command { margin:0; min-height:108px; max-height:260px; overflow:auto;
      white-space:pre-wrap; word-break:break-word; background:#19211e; color:#d9e4de;
      border-radius:12px; padding:14px; font:12px/1.55 ui-monospace,SFMono-Regular,monospace; }
    .match { display:grid; gap:8px; margin:12px 0; }
    .match div { padding:9px 10px; background:#f7f9f7; border-radius:9px;
      overflow-wrap:anywhere; }
    .actions { display:flex; gap:9px; margin-top:14px; }
    .primary,.secondary,.danger { border:0; border-radius:10px; cursor:pointer;
      padding:10px 14px; font-weight:700; }
    .primary { background:var(--brand); color:white; flex:1; }
    .primary:hover { background:var(--brand2); }
    .primary:disabled { opacity:.5; cursor:not-allowed; }
    .secondary { color:var(--ink); background:#edf1ee; }
    .danger { color:var(--red); background:#fff0ef; padding:7px 10px; }
    .message { margin-top:10px; padding:10px; border-radius:9px; background:#fff6e9;
      color:#875317; white-space:pre-wrap; }
    .message.error { color:var(--red); background:#fff0ef; }
    .jobs { margin-top:20px; }
    .job { border:1px solid var(--line); border-radius:13px; padding:13px; margin-top:10px; }
    .job-head { display:flex; gap:10px; justify-content:space-between; align-items:center; }
    .job-title { font-weight:750; overflow-wrap:anywhere; }
    .job-meta { color:var(--muted); font-size:12px; margin:4px 0 8px; }
    .viewer-actions { display:flex; gap:8px; align-items:center; flex-wrap:wrap;
      margin:8px 0; }
    .viewer-link { display:inline-block; border-radius:9px; padding:7px 10px;
      color:white; background:var(--green); text-decoration:none; font-weight:700; }
    .viewer-note { color:var(--muted); font:11px/1.4 ui-monospace,monospace;
      overflow-wrap:anywhere; }
    .log { max-height:220px; overflow:auto; white-space:pre-wrap; background:#f6f8f6;
      border-radius:8px; padding:10px; font:11px/1.45 ui-monospace,monospace; }
    .empty { color:var(--muted); text-align:center; padding:18px; }
    @media (max-width:980px) { .layout { grid-template-columns:1fr; }
      .side { position:static; } }
    @media (max-width:620px) { .shell { padding:14px; } header { align-items:flex-start;
      flex-direction:column; } .status { justify-content:flex-start; }
      .tabs { grid-template-columns:1fr 1fr; } .grid { grid-template-columns:1fr; }
      .wide { grid-column:auto; } }
  </style>
</head>
<body>
<main class="shell">
  <header>
    <div><h1>MjLab 实验启动台</h1><div class="subtitle" id="repo">正在连接本地仓库…</div></div>
    <div class="status" id="status"></div>
  </header>
  <div class="layout">
    <form class="card" id="form">
      <input type="hidden" name="kind" value="train">
      <div class="tabs">
        <button class="tab active" type="button" data-kind="train">训练</button>
        <button class="tab" type="button" data-kind="play">回放 / 验证</button>
        <button class="tab" type="button" data-kind="response">响应测试</button>
        <button class="tab" type="button" data-kind="evaluate">定量评估</button>
      </div>
      <h2>实验基础</h2>
      <div class="grid">
        <label class="wide">任务<select name="task" id="task"></select></label>
        <label data-kinds="train">GPU（0 / 0,1 / all / cpu）<input name="gpu_ids" value="0"></label>
        <label data-kinds="train">环境数<input name="num_envs" type="number" min="1" value="4096"></label>
        <label data-kinds="train">训练轮数<input name="max_iterations" type="number" min="1" value="5000"></label>
        <label data-kinds="train">Run 名<input name="run_name" placeholder="trial-pikachu-walk"></label>
        <label data-kinds="train">Decimation（留空用任务默认）<input name="decimation" type="number" min="1"></label>
        <label data-kinds="play">策略<select name="agent"><option value="trained">trained</option><option value="zero">zero</option><option value="random">random</option></select></label>
        <label data-kinds="play">Viewer<select name="viewer"><option value="viser">viser</option><option value="native">native</option><option value="auto">auto</option></select></label>
        <label data-kinds="play">设备<input name="device" value="cuda:0"></label>
        <label data-kinds="play">环境数<input name="play_num_envs" type="number" min="1" value="1"></label>
        <label data-kinds="response">模式<select name="response_mode"><option value="all">step + sine</option><option value="step">step</option><option value="sine">sine</option></select></label>
        <label data-kinds="response">关节正则<input name="joint_pattern" value=".*"></label>
        <label data-kinds="response">设备<input name="response_device" value="cpu"></label>
        <label data-kinds="response">输出目录<input name="output_root" value="logs/response"></label>
        <label data-kinds="evaluate">W&amp;B run path<input name="eval_wandb_run_path" placeholder="entity/project/run_id"></label>
        <label data-kinds="evaluate">W&amp;B checkpoint（可选）<input name="eval_wandb_checkpoint_name" placeholder="model_4999.pt"></label>
        <label data-kinds="evaluate">环境数<input name="eval_num_envs" type="number" min="1" value="512"></label>
        <label data-kinds="evaluate">设备<input name="eval_device" value="cuda:0"></label>
        <label class="wide" data-kinds="evaluate">JSON 输出<input name="eval_output_file" value="src/mjlab/mocap/eval/result.json"></label>
      </div>

      <div data-kinds="train,play,evaluate">
        <h3>本地模型与动作自动匹配</h3>
        <div class="grid">
          <label data-kinds="train,play,evaluate">Checkpoint 模式<select name="checkpoint_mode"><option value="auto">自动：最新匹配 run</option><option value="selected">从本地列表选择</option><option value="manual">手填路径</option><option value="none">不使用本地模型（W&amp;B / dummy）</option></select></label>
          <label data-kinds="train,play,evaluate">本地 checkpoint<select name="checkpoint" id="checkpoint"></select></label>
          <label class="wide" data-kinds="train,play,evaluate">手填 checkpoint<input name="checkpoint_manual" placeholder="/path/to/model_4999.pt"></label>
          <label data-kinds="train,play,evaluate">NPZ 模式<select name="motion_mode"><option value="auto">自动：env.yaml 优先</option><option value="selected">从本地列表选择</option><option value="manual">手填路径</option><option value="none">不传本地 NPZ</option></select></label>
          <label data-kinds="train,play,evaluate">本地 NPZ<select name="motion" id="motion"></select></label>
          <label class="wide" data-kinds="train,play,evaluate">手填 NPZ<input name="motion_manual" placeholder="/path/to/motion.npz"></label>
          <label class="wide" data-kinds="train,play">W&amp;B motion registry（可选）<input name="registry_name" placeholder="entity/project/motion-name"></label>
        </div>
      </div>

      <div data-kinds="train">
        <h3>训练与续训</h3>
        <div class="grid">
          <label>Logger<select name="logger"><option value="default">任务默认</option><option value="wandb">wandb</option><option value="tensorboard">tensorboard</option></select></label>
          <label>W&amp;B project<input name="wandb_project" placeholder="pikachu-transition"></label>
          <label class="wide">W&amp;B tags（逗号分隔）<input name="wandb_tags" placeholder="tracking,transition"></label>
          <label class="check"><input type="checkbox" name="resume">从所选 checkpoint 续训</label>
          <label class="check"><input type="checkbox" name="nan_guard" checked>开启 NaN guard</label>
          <label class="check"><input type="checkbox" name="video">训练中录视频</label>
          <label class="check"><input type="checkbox" name="obs_normalization">Actor observation normalization</label>
        </div>
        <h3>高级 PPO 参数 <span class="hint">留空即使用任务配置</span></h3>
        <div class="grid">
          <label>Learning rate<input name="learning_rate" type="number" step="any" placeholder="3e-4"></label>
          <label>Clip param<input name="clip_param" type="number" step="any" placeholder="0.2"></label>
          <label>Mini batches<input name="num_mini_batches" type="number" min="1" placeholder="4"></label>
          <label>Hidden dims<input name="hidden_dims" placeholder="512,256,128"></label>
        </div>
      </div>

      <div data-kinds="play">
        <h3>回放选项</h3>
        <div class="grid">
          <label>W&amp;B run path（本地模型为 none 时）<input name="wandb_run_path" placeholder="entity/project/run_id"></label>
          <label>W&amp;B checkpoint（可选）<input name="wandb_checkpoint_name" placeholder="model_4999.pt"></label>
          <label>采样方式<select name="sampling_mode"><option value="default">任务默认</option><option value="adaptive">adaptive</option><option value="uniform">uniform</option><option value="start">start</option></select></label>
          <label class="check"><input type="checkbox" name="no_terminations">禁用 termination</label>
          <label class="check"><input type="checkbox" name="play_video">录制视频</label>
          <label>视频步数<input name="video_length" type="number" min="1" value="200"></label>
        </div>
      </div>

      <div data-kinds="response">
        <h3>响应协议</h3>
        <div class="grid">
          <label>悬空高度 m<input name="suspend_height" type="number" step="any" value="1.0"></label>
          <label>关节限位余量<input name="limit_margin" type="number" step="any" value="0.1"></label>
          <label>阶跃幅值比例<input name="step_amplitude" type="number" step="any" value="0.25"></label>
          <label>阶跃前置 s<input name="step_pre" type="number" step="any" value="0.4"></label>
          <label>阶跃时长 s<input name="step_duration" type="number" step="any" value="1.6"></label>
          <label>正弦频率 Hz<input name="sine_frequencies" value="0.5,1,2,4,8"></label>
          <label>正弦幅值比例<input name="sine_amplitude" type="number" step="any" value="0.2"></label>
          <label>正弦前置 s<input name="sine_pre" type="number" step="any" value="0.4"></label>
          <label>正弦周期数<input name="sine_cycles" type="number" step="any" value="4"></label>
          <label>预热周期数<input name="sine_warmup" type="number" step="any" value="1"></label>
          <label>采样 stride<input name="sample_stride" type="number" min="1" value="1"></label>
          <label>扭矩安全系数<input name="torque_safety" type="number" step="any" value="1.5"></label>
          <label>速度安全系数<input name="speed_safety" type="number" step="any" value="1.2"></label>
          <label class="check"><input type="checkbox" name="strict_no_contact" checked>检测到接触时中止</label>
        </div>
      </div>

      <h3>命令行补充</h3>
      <div class="grid">
        <label class="wide">额外 Tyro 参数<textarea name="extra_args" placeholder="--env.some.option value"></textarea><span class="hint">按 shell 引号规则拆分，但启动时不会经过 shell。</span></label>
      </div>
    </form>

    <aside class="side">
      <section class="card">
        <h2>命令预览</h2>
        <pre class="command" id="command">正在生成…</pre>
        <div class="match" id="match"></div>
        <div id="message"></div>
        <label class="check"><input type="checkbox" name="allow_conflict" form="form">检测到冲突时仍强制启动</label>
        <div class="actions">
          <button class="secondary" type="button" id="copy">复制命令</button>
          <button class="primary" type="button" id="start" disabled>一键启动</button>
        </div>
      </section>
      <section class="card">
        <div class="job-head"><h2>本页启动的任务</h2><button class="secondary" type="button" id="refresh">刷新文件</button></div>
        <label>SSH 目标<input id="sshTarget" placeholder="user@remote-host"></label>
        <div class="hint">远程使用时，“复制 SSH 转发”会使用这里的目标。</div>
        <div id="jobs" class="empty">暂无任务</div>
      </section>
    </aside>
  </div>
</main>
<script>
const TOKEN = __TOKEN_JSON__;
const form = document.querySelector('#form');
const commandBox = document.querySelector('#command');
const startButton = document.querySelector('#start');
let state = null;
let previewTimer = null;
let lastPreview = null;

function esc(value) { return String(value ?? '').replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function payload() {
  const data = Object.fromEntries(new FormData(form).entries());
  form.querySelectorAll('input[type=checkbox]').forEach(el => data[el.name] = el.checked);
  return data;
}
async function api(path, body) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json',
    'X-Start-Server-Token':TOKEN}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function showMessage(text, error=false) {
  document.querySelector('#message').innerHTML = text ?
    `<div class="message ${error?'error':''}">${esc(text)}</div>` : '';
}
function setKind(kind) {
  form.elements.kind.value = kind;
  document.querySelectorAll('.tab').forEach(el => el.classList.toggle('active', el.dataset.kind===kind));
  document.querySelectorAll('[data-kinds]').forEach(el => {
    el.classList.toggle('hidden', !el.dataset.kinds.split(',').includes(kind));
  });
  if (kind === 'evaluate' && !form.elements.task.value.includes('Tracking')) {
    const tracking = [...form.elements.task.options].find(o => o.value.includes('Tracking'));
    if (tracking) form.elements.task.value = tracking.value;
  }
  schedulePreview();
}
function fillSelect(select, items, valueKey='path') {
  const old = select.value;
  select.innerHTML = items.map(item => `<option value="${esc(item[valueKey])}">${esc(item.label || item[valueKey])}</option>`).join('');
  if ([...select.options].some(option => option.value === old)) select.value = old;
}
function renderState(data) {
  state = data;
  document.querySelector('#repo').textContent = data.repo_root;
  const sshInput = document.querySelector('#sshTarget');
  if (!sshInput.dataset.edited) sshInput.value = data.ssh_target || '';
  sshInput.oninput = () => sshInput.dataset.edited = '1';
  const task = document.querySelector('#task');
  const oldTask = task.value;
  task.innerHTML = data.catalog.tasks.map(value => `<option value="${esc(value)}">${esc(value)}</option>`).join('');
  const preferred = data.catalog.tasks.includes(oldTask) ? oldTask :
    data.catalog.tasks.find(value => value === 'Mjlab-Tracking-Flat-Pikachu-Walk') || data.catalog.tasks[0];
  task.value = preferred || '';
  fillSelect(document.querySelector('#checkpoint'), data.catalog.checkpoints);
  fillSelect(document.querySelector('#motion'), data.catalog.motions);
  const gpuPills = data.gpu.available ? data.gpu.gpus.map(g =>
    `<span class="pill ${g.utilization<10?'good':''}">GPU ${g.index} · ${g.memory_used_mb}/${g.memory_total_mb} MiB · ${g.utilization}%</span>`).join('') :
    '<span class="pill">未检测到 nvidia-smi</span>';
  const staleJobCount = data.stale_jobs.reduce((sum, item) => sum + item.count, 0);
  const staleGpuCount = data.stale_gpu_processes.length;
  const stalePill = staleJobCount || staleGpuCount ?
    `<span class="pill">历史残留 ${staleJobCount} 组 · GPU stale ${staleGpuCount}</span>` : '';
  document.querySelector('#status').innerHTML =
    `<span class="pill good">本地服务已连接</span><span class="pill">${data.catalog.checkpoints.length} runs · ${data.catalog.motions.length} NPZ</span>` +
    `${gpuPills}${stalePill}`;
  renderJobs(data.jobs);
}
function viewerControls(job) {
  if (!job.viewer_port) return '';
  const rawHost = window.location.hostname || '127.0.0.1';
  const host = rawHost.includes(':') ? `[${rawHost}]` : rawHost;
  const viewerUrl = `http://${host}:${job.viewer_port}/`;
  const label = job.viewer_ready ? '打开 Viser' : '打开 Viser（启动中）';
  return `<div class="viewer-actions"><a class="viewer-link" href="${esc(viewerUrl)}" target="_blank" rel="noopener">${label}</a>
    <button type="button" class="secondary" data-forward-port="${job.viewer_port}">复制 SSH 转发</button>
    <span class="viewer-note">远端端口 ${job.viewer_port}</span></div>`;
}
function jobStatusText(job) {
  if (job.status === 'running') return '运行中';
  if (job.status === 'stale') return '历史残留（可停止清理）';
  return `已退出 (${job.return_code})`;
}
function renderJobs(jobs) {
  const root = document.querySelector('#jobs');
  if (!jobs.length) { root.className='empty'; root.textContent='暂无任务'; return; }
  root.className='';
  root.innerHTML = jobs.map(job => `<article class="job">
    <div class="job-head"><div class="job-title">${esc(job.kind)} · ${esc(job.task)}</div>
    ${job.status!=='exited'?`<button type="button" class="danger" data-stop="${esc(job.id)}">停止</button>`:''}</div>
    <div class="job-meta">PID ${job.pid} · ${jobStatusText(job)} · ${esc(job.log_file)}</div>
    ${viewerControls(job)}
    <div class="log">${esc(job.log_tail || '等待输出…')}</div></article>`).join('');
  root.querySelectorAll('[data-stop]').forEach(button => button.onclick = async () => {
    if (!confirm('向这个任务及其子进程发送终止信号？')) return;
    try { await api('/api/stop', {job_id:button.dataset.stop}); await loadState(); }
    catch (error) { showMessage(error.message, true); }
  });
  root.querySelectorAll('[data-forward-port]').forEach(button => button.onclick = async () => {
    const port = button.dataset.forwardPort;
    const target = document.querySelector('#sshTarget').value.trim() || state.ssh_target;
    const command = `ssh -N -L ${port}:127.0.0.1:${port} ${target}`;
    await navigator.clipboard.writeText(command);
    showMessage(`SSH 转发命令已复制：\n${command}\n建立隧道后点击“打开 Viser”。`);
  });
}
async function preview() {
  try {
    const data = await api('/api/preview', payload());
    lastPreview = data;
    commandBox.textContent = data.command;
    const checkpoint = data.checkpoint || '未使用本地 checkpoint';
    const motion = data.motion || '未使用本地 NPZ';
    document.querySelector('#match').innerHTML = `<div><b>模型</b><br>${esc(checkpoint)}</div><div><b>动作</b><br>${esc(motion)}${data.motion_reason?`<br><span class="hint">${esc(data.motion_reason)}</span>`:''}</div>`;
    const notes = [...data.warnings, ...data.conflicts.map(item => item.message)];
    showMessage(notes.join('\n'), data.conflicts.length > 0);
    startButton.disabled = false;
  } catch (error) {
    lastPreview = null; commandBox.textContent = '参数尚未就绪';
    document.querySelector('#match').innerHTML = '';
    showMessage(error.message, true); startButton.disabled = true;
  }
}
function schedulePreview() { clearTimeout(previewTimer); previewTimer=setTimeout(preview, 180); }
async function loadState(refresh=false) {
  try { const response=await fetch(`/api/state${refresh?'?refresh=1':''}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderState(await response.json()); schedulePreview();
  } catch (error) { showMessage(`无法连接服务：${error.message}`, true); }
}
form.addEventListener('input', schedulePreview);
form.addEventListener('change', schedulePreview);
document.querySelectorAll('.tab').forEach(tab => tab.onclick=()=>setKind(tab.dataset.kind));
document.querySelector('#copy').onclick = async () => {
  if (!lastPreview) return; await navigator.clipboard.writeText(lastPreview.command);
  showMessage('命令已复制。');
};
startButton.onclick = async () => {
  if (!lastPreview) return;
  startButton.disabled=true; showMessage('正在启动…');
  try { const result=await api('/api/start', payload());
    showMessage(`已启动 PID ${result.job.pid}，输出持续写入 ${result.job.log_file}`);
    await loadState();
  } catch (error) { showMessage(error.message, true); schedulePreview(); }
};
document.querySelector('#refresh').onclick=()=>loadState(true);
setKind('train'); loadState();
setInterval(()=>loadState(false), 3000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
  main()
