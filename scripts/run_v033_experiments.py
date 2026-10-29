"""Durable two-experiment training/evaluation workflow with explicit status files."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from typing import Any

from v033_artifacts import AUDIT, ROOT, TASK_DIRS, audit_status

ARTIFACTS = AUDIT
TASKS = {
  "biped": "Mjlab-Velocity-Flat-Pikachu-V033-E0",
  "crawl": "Mjlab-Velocity-Flat-Pikachu-V033-Crawl",
}


def now():
  return datetime.now().astimezone().isoformat()


def save_status(state):
  target = ARTIFACTS / "metadata/status.json"
  target.parent.mkdir(parents=True, exist_ok=True)
  tmp = target.with_suffix(".tmp")
  tmp.write_text(json.dumps(state, indent=2) + "\n")
  tmp.replace(target)


def run(cmd, log, env):
  with (ARTIFACTS / "logs/orchestration/commands.jsonl").open("a") as f:
    f.write(json.dumps({"time": now(), "argv": cmd, "log": str(log)}) + "\n")
  log.parent.mkdir(parents=True, exist_ok=True)
  with log.open("a") as stream:
    result = subprocess.run(
      cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
    )
  if result.returncode:
    raise RuntimeError(f"Exit {result.returncode}: {cmd}; inspect {log}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--iterations", type=int, default=3000)
  parser.add_argument("--num-envs", type=int, default=1024)
  args = parser.parse_args()
  if audit_status().exists():
    raise FileExistsError(
      "Recorded experiment exists; do not overwrite its audit trail."
    )
  (ARTIFACTS / "logs/orchestration").mkdir(parents=True, exist_ok=True)
  env = dict(os.environ, MUJOCO_GL="egl", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
  state: dict[str, Any] = {
    "started": now(),
    "pid": os.getpid(),
    "iterations": args.iterations,
    "num_envs": args.num_envs,
    "seed": 42,
    "status": "running",
    "experiments": {},
  }
  save_status(state)
  try:
    for name, task in TASKS.items():
      state["experiments"][name] = {
        "task": task,
        "status": "training",
        "started": now(),
      }
      save_status(state)
      train = [
        "uv",
        "run",
        "--no-sync",
        "train",
        task,
        "--env.scene.num-envs",
        str(args.num_envs),
        "--agent.max-iterations",
        str(args.iterations),
        "--agent.seed",
        "42",
        "--agent.logger",
        "tensorboard",
        "--agent.run-name",
        f"review-{name}",
        "--log-root",
        str(ROOT / "logs/rsl_rl"),
      ]
      run(train, ARTIFACTS / f"logs/training/{name}_train.log", env)
      runs = sorted(TASK_DIRS[name].glob(f"*review-{name}"))
      run_dir = runs[-1]
      checkpoints = sorted(
        run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1])
      )
      checkpoint = checkpoints[-1]
      state["experiments"][name].update(
        status="evaluating", run=str(run_dir), checkpoint=str(checkpoint)
      )
      save_status(state)
      review = run_dir / "review"
      base = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/review_v033_policy.py",
        "--task",
        task,
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(review),
      ]
      for clip in ("stand", "forward", "backward", "turn"):
        run(
          base + ["--clip", clip, "--seconds", "12", "--num-envs", "32"],
          ARTIFACTS / f"logs/evaluation/{name}_eval.log",
          env,
        )
        for view in ("side", "front"):
          run(
            base
            + [
              "--clip",
              clip,
              "--seconds",
              "8",
              "--num-envs",
              "1",
              "--video",
              "--view",
              view,
            ],
            ARTIFACTS / f"logs/evaluation/{name}_video.log",
            env,
          )
      state["experiments"][name].update(
        status="complete",
        finished=now(),
        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      )
      save_status(state)
    state.update(status="complete", finished=now())
    save_status(state)
  except Exception as exc:
    state.update(status="failed", error=repr(exc), failed=now())
    save_status(state)
    raise
  finally:
    run(
      ["uv", "run", "--no-sync", "python", "scripts/report_v033_experiments.py"],
      ARTIFACTS / "logs/orchestration/report.log",
      env,
    )


if __name__ == "__main__":
  main()
