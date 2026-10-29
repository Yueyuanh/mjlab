"""Resume V034 tasks to an exact completed-update count, recording each checkpoint.

Run the queue in a detached session for long jobs. RSL-RL stores a zero-based
last-completed iteration; resume advances it once. Models/video names below use
completed updates (500, 1000, ..., 5000), while checkpoint internals retain the
upstream convention so ordinary play/eval tools remain compatible.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
  temporary = path.with_suffix(".tmp")
  temporary.write_text(json.dumps(value, indent=2) + "\n")
  temporary.replace(path)


def record(mode, checkpoint, folder, seconds):
  completed = int(checkpoint.stem.split("_")[-1])
  command = [
    sys.executable,
    str(ROOT / "scripts/eval_v034.py"),
    "--mode",
    mode,
    "--checkpoint",
    str(checkpoint),
    "--clips",
    "forward",
    "--num-envs",
    "1",
    "--seconds",
    str(seconds),
    "--output",
    str(folder / "videos" / f"model_{completed}.json"),
    "--video",
    str(folder / "videos" / f"model_{completed}.mp4"),
  ]
  with (folder / "videos" / f"model_{completed}.log").open("w") as log:
    subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def worker(args):
  import torch

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  folder = args.output / args.worker_mode
  folder.mkdir(parents=True, exist_ok=True)
  (folder / "videos").mkdir(exist_ok=True)
  task = f"Mjlab-Velocity-Flat-Pikachu-V034-{args.worker_mode.capitalize()}"
  env_cfg, agent_cfg = load_env_cfg(task), load_rl_cfg(task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = agent_cfg.seed
  env_cfg.sim.nan_guard.enabled = True
  agent_cfg.save_interval = args.save_interval
  agent_cfg.max_iterations = args.target_iterations
  agent_cfg.run_name = args.output.name
  agent_cfg.logger = "tensorboard"
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  state = {
    "task": task,
    "pid": os.getpid(),
    "source_checkpoint": str(args.checkpoint),
    "target_iterations": args.target_iterations,
    "save_interval": args.save_interval,
    "video_seconds": args.video_seconds,
    "num_envs": args.num_envs,
    "status": "loading",
    "artifacts": [],
  }
  status_file = folder / "progress.json"

  class MilestoneRunner(VelocityOnPolicyRunner):
    @staticmethod
    def _get_export_paths(checkpoint_path):
      path = Path(checkpoint_path)
      return path.parent, path.with_suffix(".onnx").name, path.with_suffix(".onnx")

    def save(self, path, infos=None):
      completed = self.current_learning_iteration + 1
      if completed % args.save_interval and completed != args.target_iterations:
        return
      checkpoint = folder / f"model_{completed}.pt"
      if checkpoint.exists():
        # The final upstream save repeats the last logger-triggered milestone.
        return
      state.update(status="saving", completed_iterations=completed)
      write_json(status_file, state)
      super().save(
        str(checkpoint), {**(infos or {}), "completed_iterations": completed}
      )
      state.update(status="recording", latest_checkpoint=str(checkpoint))
      write_json(status_file, state)
      record(args.worker_mode, checkpoint, folder, args.video_seconds)
      state["artifacts"].append(
        {
          "iteration": completed,
          "checkpoint": str(checkpoint),
          "onnx": str(checkpoint.with_suffix(".onnx")),
          "video": str(folder / "videos" / f"model_{completed}.mp4"),
        }
      )
      state.update(status="training", updated_at=time.time())
      write_json(status_file, state)

  try:
    dump_yaml(folder / "params/env.yaml", asdict(env_cfg))
    dump_yaml(folder / "params/agent.yaml", asdict(agent_cfg))
    runner = MilestoneRunner(wrapped, asdict(agent_cfg), str(folder), "cuda:0")
    runner.load(str(args.checkpoint), map_location="cuda:0")
    completed = runner.current_learning_iteration + 1
    if completed >= args.target_iterations:
      raise ValueError(
        f"Source already has {completed} updates; target must be greater"
      )
    # Restore optimizer, observation normalization and curriculum, then advance
    # the index so the checkpoint's last update is not counted twice.
    runner.current_learning_iteration = completed
    state.update(
      status="training",
      starting_iterations=completed,
      completed_iterations=completed,
      additional_iterations=args.target_iterations - completed,
      started_at=time.time(),
    )
    write_json(status_file, state)
    original_log = runner.logger.log

    def log_and_save(*positional, **keywords):
      original_log(*positional, **keywords, print_minimal=True)
      count = runner.current_learning_iteration + 1
      if count % 10 == 0:
        state.update(completed_iterations=count, updated_at=time.time())
        write_json(status_file, state)
      if count % args.save_interval == 0:
        runner.save(str(folder / f"model_{count}.pt"))

    runner.logger.log = log_and_save
    runner.learn(args.target_iterations - completed, init_at_random_ep_len=True)
    state.update(
      status="complete",
      completed_iterations=args.target_iterations,
      finished_at=time.time(),
    )
    write_json(status_file, state)
  except BaseException as exc:
    state.update(status="failed", error=repr(exc), updated_at=time.time())
    write_json(status_file, state)
    raise
  finally:
    wrapped.close()
    torch.cuda.empty_cache()


def queue(args):
  from run_v034_baselines import snapshot_sources

  args.output.mkdir(parents=True, exist_ok=True)
  if (args.output / "queue.json").exists():
    raise FileExistsError("Use a new output directory; preserve the previous queue")
  snapshot_sources(args.output)
  source = json.loads(args.manifest.read_text())
  state = {
    "pid": os.getpid(),
    "status": "running",
    "target_iterations": args.target_iterations,
    "save_interval": args.save_interval,
    "modes": {m: "pending" for m in ("biped", "crawl")},
  }
  status_file = args.output / "queue.json"
  write_json(status_file, state)
  try:
    for mode in ("biped", "crawl"):
      state["modes"][mode] = "training"
      write_json(status_file, state)
      command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--checkpoint",
        source[mode]["checkpoint"],
        "--output",
        str(args.output),
        "--target-iterations",
        str(args.target_iterations),
        "--save-interval",
        str(args.save_interval),
        "--num-envs",
        str(args.num_envs),
        "--video-seconds",
        str(args.video_seconds),
      ]
      with (args.output / f"{mode}_train.log").open("w") as log:
        subprocess.run(
          command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT
        )
      state["modes"][mode] = "evaluating"
      write_json(status_file, state)
      for randomized in (False, True):
        label = "randomized" if randomized else "nominal"
        command = [
          sys.executable,
          str(ROOT / "scripts/eval_v034.py"),
          "--mode",
          mode,
          "--checkpoint",
          str(args.output / mode / f"model_{args.target_iterations}.pt"),
          "--output",
          str(args.output / mode / f"final_{label}.json"),
        ]
        if randomized:
          command.append("--randomized")
        with (args.output / mode / f"final_{label}.log").open("w") as log:
          subprocess.run(
            command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT
          )
      state["modes"][mode] = "complete"
      write_json(status_file, state)
    state.update(status="complete", finished_at=time.time())
    write_json(status_file, state)
  except BaseException as exc:
    state.update(status="failed", error=repr(exc), updated_at=time.time())
    write_json(status_file, state)
    raise


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--manifest",
    type=Path,
    default=ROOT / "doc/pikachu_v034_baseline_20261023/data/runs.json",
  )
  parser.add_argument("--target-iterations", type=int, default=5000)
  parser.add_argument("--save-interval", type=int, default=500)
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--video-seconds", type=float, default=5)
  parser.add_argument("--worker-mode", choices=("biped", "crawl"))
  parser.add_argument("--checkpoint", type=Path)
  args = parser.parse_args()
  if (
    min(args.target_iterations, args.save_interval, args.video_seconds) <= 0
    or args.num_envs < 4
  ):
    parser.error(
      "Positive iterations/interval/video duration and num-envs >=4 are required"
    )
  if args.worker_mode and args.checkpoint is None:
    parser.error("worker-mode requires checkpoint")
  args.output = args.output.resolve()
  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
  os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
  worker(args) if args.worker_mode else queue(args)
