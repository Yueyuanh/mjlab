"""Audited E1 training worker; explicit output, initialization and progress."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


def write_json(path: Path, value: Any) -> None:
  temporary = path.with_suffix(".tmp")
  temporary.write_text(json.dumps(value, indent=2) + "\n")
  temporary.replace(path)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--iterations", type=int, default=20000)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--learning-rate", type=float)
  parser.add_argument("--actor-init", type=Path)
  parser.add_argument("--save-interval", type=int, default=500)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=False)
  configure_torch_backends()
  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  # Read-only pressure telemetry is expensive at 4096 environments. Sensors,
  # observations, reward terms, action dimensions and network stay unchanged.
  cfg.metrics = {}
  agent = load_rl_cfg(args.task)
  assert isinstance(agent, RslRlOnPolicyRunnerCfg)
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.upload_model = False
  agent.max_iterations = args.iterations
  agent.save_interval = args.save_interval
  if args.learning_rate is not None:
    agent.algorithm.learning_rate = args.learning_rate
  dump_yaml(args.out / "params/env.yaml", asdict(cfg))
  dump_yaml(args.out / "params/agent.yaml", asdict(agent))
  write_json(
    args.out / "initialization.json",
    {
      "task": args.task,
      "num_envs": args.num_envs,
      "iterations": args.iterations,
      "transitions": args.num_envs * args.iterations * agent.num_steps_per_env,
      "seed": args.seed,
      "initialization": "actor-only" if args.actor_init else "random-from-scratch",
      "actor_checkpoint": str(args.actor_init) if args.actor_init else None,
      "actor_sha256": hashlib.sha256(args.actor_init.read_bytes()).hexdigest()
      if args.actor_init
      else None,
      "telemetry": "cfg.metrics disabled; task sensors/rewards unchanged",
      "pid": os.getpid(),
    },
  )
  started = time.monotonic()
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    cls = load_runner_cls(args.task)
    assert cls is not None
    runner = cls(RslRlVecEnvWrapper(env), asdict(agent), str(args.out), "cuda:0")
    if args.actor_init:
      runner.load(
        str(args.actor_init),
        load_cfg={
          "actor": True,
          "critic": False,
          "optimizer": False,
          "iteration": False,
        },
        map_location="cuda:0",
      )
      env.common_step_counter = 0
      runner.current_learning_iteration = 0
    original_log = runner.logger.log

    def log_progress(**kwargs: Any) -> None:
      original_log(**kwargs, print_minimal=True)
      write_json(
        args.out / "progress.json",
        {
          "iteration": int(kwargs["it"]),
          "completed_iterations": int(kwargs["it"]) + 1,
          "target_iterations": args.iterations,
          "num_envs": args.num_envs,
          "elapsed_seconds": time.monotonic() - started,
          "iteration_seconds": kwargs["collect_time"] + kwargs["learn_time"],
          "learning_rate": kwargs["learning_rate"],
          "peak_torch_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
          "losses": kwargs["loss_dict"],
        },
      )

    runner.logger.log = log_progress
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    checkpoint = args.out / f"model_{args.iterations - 1}.pt"
    write_json(
      args.out / "complete.json",
      {
        "checkpoint": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
      },
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()
