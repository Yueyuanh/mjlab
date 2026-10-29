"""Record a small, fully measured V033 crawl replay without video or training."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer.viser.recording import ControlStepRecording, RecordingOptions
from mjlab.viewer.viser.term_plotter import MetricsCsvRecorder


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--seconds", type=float, default=10.0)
  parser.add_argument("--forward", type=float, default=0.15)
  args = parser.parse_args()
  if args.task not in {
    "Mjlab-Velocity-Flat-Pikachu-V033-Crawl",
    "Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E1",
    "Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E2",
    "Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E3",
  }:
    raise ValueError("Only V033 crawl tasks are supported")
  if not 0 < args.seconds <= 10:
    raise ValueError("Limit the verification replay to 10 seconds")
  if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)
  available_kib = int(
    next(
      line
      for line in Path("/proc/meminfo").read_text().splitlines()
      if line.startswith("MemAvailable:")
    ).split()[1]
  )
  if available_kib < 2 * 1024**2:
    raise RuntimeError("At least 2 GiB available host RAM is required")
  torch.set_num_threads(2)

  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = 1
  cfg.seed = 42
  cfg.events.pop("push_robot", None)
  for name in ("foot_friction", "base_com", "encoder_bias"):
    cfg.events.pop(name, None)
  cfg.events["reset_base"].params["yaw_range"] = (0.0, 0.0)
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    runner_cls = load_runner_cls(args.task)
    assert runner_cls is not None
    runner = runner_cls(
      RslRlVecEnvWrapper(env), asdict(load_rl_cfg(args.task)), None, "cuda:0"
    )
    runner.load(
      str(args.checkpoint),
      load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
      map_location="cuda:0",
    )
    policy = runner.get_inference_policy("cuda:0")
    recorder = MetricsCsvRecorder(
      args.out, lambda: f"{args.task}__{args.checkpoint.stem}", env.step_dt
    )
    options = RecordingOptions(
      prefix="recheck",
      duration_s=args.seconds,
      command_mode="fixed",
      velocity=(args.forward, 0.0, 0.0),
      video=False,
    )
    recording = ControlStepRecording(env, recorder, options, 0)
    obs, _ = env.reset()
    print(f"Registered metrics: {len(env.metrics_manager.active_terms)}", flush=True)
    try:
      path = recording.start()
      for step in range(recording.target_samples):
        recording.prepare_command()
        with torch.inference_mode():
          obs, _, terminated, truncated, _ = env.step(policy(obs))
        reason = recording.append(step + 1)
        if reason or bool(terminated[0] or truncated[0]):
          break
    finally:
      recording.finish("replay_completed")
    print(f"Saved {recording.samples} rows: {path}", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
