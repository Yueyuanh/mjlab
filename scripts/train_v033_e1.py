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
from rsl_rl.modules import GaussianDistribution

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


def parse_named_floats(values: list[str], label: str) -> dict[str, float]:
  """Parse repeatable NAME=VALUE overrides without hiding invalid names."""
  result: dict[str, float] = {}
  for value in values:
    name, separator, raw_weight = value.partition("=")
    if not separator or not name or not raw_weight:
      raise ValueError(f"Invalid {label} override {value!r}; expected NAME=VALUE")
    result[name] = float(raw_weight)
  return result


def parse_reward_weights(values: list[str]) -> dict[str, float]:
  """Parse repeatable reward overrides."""
  return parse_named_floats(values, "reward")


def write_json(path: Path, value: Any) -> None:
  temporary = path.with_suffix(".tmp")
  temporary.write_text(json.dumps(value, indent=2) + "\n")
  temporary.replace(path)


def available_memory_mib() -> float:
  for line in Path("/proc/meminfo").read_text().splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) / 1024
  raise RuntimeError("Cannot read available system memory")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--iterations", type=int, default=20000)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--learning-rate", type=float)
  parser.add_argument("--actor-init", type=Path)
  parser.add_argument("--load-critic", action="store_true")
  parser.add_argument(
    "--resume",
    action="store_true",
    help="Restore full state and train to --iterations total",
  )
  parser.add_argument(
    "--command-bucket",
    action="append",
    default=[],
    metavar="NAME=FRACTION",
    help="Override standing/forward/backward/lateral/heading/world sampling.",
  )
  parser.add_argument("--reset-exploration", action="store_true")
  parser.add_argument(
    "--exploration-std",
    type=float,
    help=(
      "Override the scalar Gaussian initial standard deviation. This also "
      "widens a configured std_range when necessary."
    ),
  )
  parser.add_argument("--save-interval", type=int, default=500)
  parser.add_argument("--gait-scale", type=float, default=1.0)
  parser.add_argument(
    "--reward-weight",
    action="append",
    default=[],
    metavar="NAME=WEIGHT",
    help="Override a configured reward weight; may be repeated.",
  )
  parser.add_argument(
    "--no-push",
    action="store_true",
    help=(
      "Remove the periodic push_robot event. Disturbance ablation: separates "
      "'cannot stand under 1-3 s pushes' from 'cannot learn at all'."
    ),
  )
  parser.add_argument("--min-available-memory-mib", type=float, default=1536)
  args = parser.parse_args()
  if args.resume:
    if args.actor_init is None or args.reset_exploration:
      parser.error("--resume requires --actor-init and retained exploration")
    args.load_critic = True
  if args.load_critic and args.actor_init is None:
    parser.error("--load-critic requires --actor-init")
  if args.exploration_std is not None and args.exploration_std <= 0:
    parser.error("--exploration-std must be positive")
  args.out.mkdir(parents=True, exist_ok=False)
  configure_torch_backends()
  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  # Read-only pressure telemetry is expensive at 4096 environments. Sensors,
  # observations, reward terms, action dimensions and network stay unchanged.
  cfg.metrics = {}
  if args.no_push:
    cfg.events.pop("push_robot", None)
  for name in ("straight_hip_roll", "straight_stance_width_max"):
    if name in cfg.rewards:
      cfg.rewards[name].weight *= args.gait_scale
  reward_weights = parse_reward_weights(args.reward_weight)
  unknown_rewards = sorted(set(reward_weights) - set(cfg.rewards))
  if unknown_rewards:
    raise ValueError(f"Unknown reward terms: {', '.join(unknown_rewards)}")
  for name, weight in reward_weights.items():
    cfg.rewards[name].weight = weight
  command_buckets = parse_named_floats(args.command_bucket, "command bucket")
  allowed_buckets = {
    "standing",
    "forward",
    "backward",
    "lateral",
    "heading",
    "world",
  }
  unknown_buckets = sorted(set(command_buckets) - allowed_buckets)
  if unknown_buckets:
    raise ValueError(f"Unknown command buckets: {', '.join(unknown_buckets)}")
  if any(not 0 <= value <= 1 for value in command_buckets.values()):
    raise ValueError("Command bucket fractions must be in [0, 1]")
  command = cfg.commands["twist"]
  for name, fraction in command_buckets.items():
    setattr(command, f"rel_{name}_envs", fraction)
  agent = load_rl_cfg(args.task)
  assert isinstance(agent, RslRlOnPolicyRunnerCfg)
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.upload_model = False
  agent.max_iterations = args.iterations
  agent.save_interval = args.save_interval
  if args.learning_rate is not None:
    agent.algorithm.learning_rate = args.learning_rate
  if args.exploration_std is not None:
    assert agent.actor.distribution_cfg is not None
    distribution_cfg = agent.actor.distribution_cfg
    distribution_cfg["init_std"] = args.exploration_std
    std_range = distribution_cfg.get("std_range")
    if std_range is not None:
      distribution_cfg["std_range"] = (
        min(std_range[0], args.exploration_std),
        max(std_range[1], args.exploration_std),
      )
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
      "initialization": (
        "resume-full-state"
        if args.resume
        else "actor-and-critic"
        if args.load_critic
        else "actor-only"
        if args.actor_init
        else "random-from-scratch"
      ),
      "actor_checkpoint": str(args.actor_init) if args.actor_init else None,
      "critic_loaded": args.load_critic,
      "exploration_reset": args.reset_exploration,
      "exploration_std_override": args.exploration_std,
      "actor_sha256": hashlib.sha256(args.actor_init.read_bytes()).hexdigest()
      if args.actor_init
      else None,
      "telemetry": (
        "cfg.metrics disabled; task sensors unchanged; reward overrides recorded "
        "separately"
      ),
      "reward_weight_overrides": reward_weights,
      "command_bucket_overrides": command_buckets,
      "push_robot": "disabled" if args.no_push else "enabled",
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
          "critic": args.load_critic,
          "optimizer": args.resume,
          "iteration": args.resume,
        },
        map_location="cuda:0",
      )
      if args.resume:
        # model_<it> is saved AFTER the update numbered <it>.
        runner.current_learning_iteration += 1
      else:
        env.common_step_counter = 0
        runner.current_learning_iteration = 0
    start_iteration = runner.current_learning_iteration
    if start_iteration >= args.iterations:
      raise ValueError("Checkpoint already reached the requested total iterations")
    write_json(
      args.out / "continuation.json",
      {
        "start_iteration": start_iteration,
        "target_iterations": args.iterations,
        "updates_this_run": args.iterations - start_iteration,
        "restored_optimizer": args.resume,
      },
    )
    if args.reset_exploration:
      distribution = runner.alg.actor.distribution
      assert isinstance(distribution, GaussianDistribution)
      assert distribution.std_type == "scalar"
      assert agent.actor.distribution_cfg is not None
      with torch.no_grad():
        distribution.std_param.fill_(agent.actor.distribution_cfg["init_std"])
    original_log = runner.logger.log

    def log_progress(**kwargs: Any) -> None:
      original_log(**kwargs, print_minimal=True)
      available_mib = available_memory_mib()
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
          "system_available_memory_mib": available_mib,
          "losses": kwargs["loss_dict"],
        },
      )
      if available_mib < args.min_available_memory_mib:
        runner.save(str(args.out / "model_memory_guard.pt"))
        raise MemoryError("Training stopped at the available-memory guard")

    runner.logger.log = log_progress
    runner.learn(
      num_learning_iterations=args.iterations - start_iteration,
      init_at_random_ep_len=True,
    )
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
