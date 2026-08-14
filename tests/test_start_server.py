"""Tests for the dependency-free local experiment dashboard."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from mjlab.scripts.start_server import (
  BuildError,
  _experiment_kind,
  _is_stale_process_tree,
  _VISER_URL_RE,
  build_experiment_command,
  discover_catalog,
  discover_checkpoints,
  discover_tasks,
  recommend_checkpoint,
  recommend_motion,
)


def _write(path: Path, content: str = "") -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(content, encoding="utf-8")


def _fixture_repo(tmp_path: Path) -> Path:
  _write(tmp_path / "pyproject.toml", "[project]\nname='mjlab'\n")
  _write(
    tmp_path / "src/mjlab/tasks/demo/__init__.py",
    'register(task_id="Mjlab-Tracking-Flat-Pikachu-Down")\n'
    'register(task_id="Mjlab-Velocity-Flat-Pikachu")\n',
  )
  motion = tmp_path / "src/mjlab/mocap/npz/Pikachu-Transition-Down.npz"
  _write(motion)
  run = tmp_path / "logs/rsl_rl/pikachu_tracking/2026-08-14_trial-transition-down"
  _write(run / "model_500.pt")
  _write(run / "model_4999.pt")
  _write(
    run / "params/env.yaml",
    "commands:\n  motion:\n    motion_file: "
    "src/mjlab/mocap/npz/Pikachu-Transition-Down.npz\n",
  )
  os.utime(run / "model_4999.pt", (2_000_000_000, 2_000_000_000))
  return tmp_path


def test_discovers_tasks_and_highest_checkpoint_per_run(tmp_path: Path) -> None:
  repo = _fixture_repo(tmp_path)

  tasks = discover_tasks(repo)
  checkpoints = discover_checkpoints(repo)

  assert tasks == (
    "Mjlab-Tracking-Flat-Pikachu-Down",
    "Mjlab-Velocity-Flat-Pikachu",
  )
  assert len(checkpoints) == 1
  assert checkpoints[0].iteration == 4999
  assert checkpoints[0].log_root == "logs/rsl_rl"
  assert checkpoints[0].motion_file == (
    "src/mjlab/mocap/npz/Pikachu-Transition-Down.npz"
  )


def test_matching_prefers_checkpoint_env_yaml_motion(tmp_path: Path) -> None:
  repo = _fixture_repo(tmp_path)
  catalog = discover_catalog(repo)
  task = "Mjlab-Tracking-Flat-Pikachu-Down"

  checkpoint = recommend_checkpoint(task, catalog.checkpoints)
  motion, reason = recommend_motion(task, checkpoint, catalog.motions)

  assert checkpoint is not None
  assert motion is not None
  assert motion.path.endswith("Pikachu-Transition-Down.npz")
  assert reason == "来自 checkpoint 的 params/env.yaml"


def test_matching_does_not_pick_an_unrelated_experiment(tmp_path: Path) -> None:
  repo = _fixture_repo(tmp_path)
  catalog = discover_catalog(repo)

  checkpoint = recommend_checkpoint("Mjlab-Velocity-Flat-Pikachu", catalog.checkpoints)

  assert checkpoint is None


def test_builds_canonical_resume_command(tmp_path: Path) -> None:
  repo = _fixture_repo(tmp_path)
  catalog = discover_catalog(repo)
  payload = {
    "kind": "train",
    "task": "Mjlab-Tracking-Flat-Pikachu-Down",
    "gpu_ids": "0,1",
    "num_envs": "4096",
    "max_iterations": "1000",
    "run_name": "resume-down",
    "resume": True,
    "checkpoint_mode": "auto",
    "motion_mode": "auto",
    "wandb_tags": "tracking,transition",
    "nan_guard": True,
  }

  result = build_experiment_command(repo, payload, catalog)

  assert result.argv[:4] == (
    "uv",
    "run",
    "train",
    "Mjlab-Tracking-Flat-Pikachu-Down",
  )
  assert "--env.scene.num-envs" in result.argv
  assert "--env.commands.motion.motion-file" in result.argv
  assert "--agent.resume" in result.argv
  assert "^2026\\-08\\-14_trial\\-transition\\-down$" in result.argv
  assert "^model_4999\\.pt$" in result.argv
  assert "('tracking', 'transition')" in result.argv
  assert result.gpu_ids == (0, 1)


def test_tracking_train_requires_motion_or_registry(tmp_path: Path) -> None:
  repo = _fixture_repo(tmp_path)
  catalog = discover_catalog(repo)
  payload = {
    "kind": "train",
    "task": "Mjlab-Tracking-Flat-Pikachu-Down",
    "motion_mode": "none",
    "gpu_ids": "cpu",
  }

  with pytest.raises(BuildError, match="Tracking 训练需要"):
    build_experiment_command(repo, payload, catalog)


def test_play_command_uses_local_model_without_shell(tmp_path: Path) -> None:
  repo = _fixture_repo(tmp_path)
  catalog = discover_catalog(repo)
  payload = {
    "kind": "play",
    "task": "Mjlab-Tracking-Flat-Pikachu-Down",
    "agent": "trained",
    "viewer": "viser",
    "device": "cuda:0",
    "play_num_envs": "1",
    "checkpoint_mode": "auto",
    "motion_mode": "auto",
    "extra_args": '--sampling-mode start --camera "tracking camera"',
  }

  result = build_experiment_command(repo, payload, catalog)

  assert result.argv[:4] == (
    "uv",
    "run",
    "play",


def test_stale_process_tree_requires_only_dead_workers() -> None:
  snapshot: dict[int, dict[str, Any]] = {
    10: {"state": "S", "parent": 1, "args": ["uv", "run", "play"]},
    11: {"state": "Z", "parent": 10, "args": []},
  }

  assert _is_stale_process_tree(10, snapshot)

  snapshot[12] = {
    "state": "S",
    "parent": 10,
    "args": ["/venv/bin/python", "/venv/bin/play", "Task"],
  }
  assert not _is_stale_process_tree(10, snapshot)


def test_classifies_torchrunx_launcher_as_training() -> None:
  args = ["/venv/bin/python", "-u", "-m", "torchrunx", "--world-size", "2"]

  assert _experiment_kind(args) == "train"


def test_extracts_viser_http_port() -> None:
  match = _VISER_URL_RE.search("HTTP  http://localhost:18432")

  assert match is not None
  assert int(match.group("port")) == 18432
    "Mjlab-Tracking-Flat-Pikachu-Down",
  )
  assert "--checkpoint-file" in result.argv
  assert result.argv[-2:] == ("--camera", "tracking camera")
