"""Evaluate fixed commands with first-fall survival and reset-safe displacement.

No checkpoint means zero residual actions (PD rest-pose baseline). Nominal mode
disables startup randomization; --randomized retains training randomization.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.pikachu_v034.crawl_mdp import (
  crawl_heading,
  heading_ang_vel,
  heading_lin_vel,
)
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat


def evaluate(
  mode,
  checkpoint,
  output,
  num_envs=16,
  seconds=10.0,
  seed=42,
  device="cuda:0",
  randomized=False,
  clips=None,
  video=None,
):
  task = f"Mjlab-Velocity-Flat-Pikachu-V034-{mode.capitalize()}"
  cfg = load_env_cfg(task)
  cfg.scene.num_envs, cfg.seed = num_envs, seed
  cfg.episode_length_s = seconds + 1
  cfg.observations["actor"].enable_corruption = False
  cfg.curriculum = {}
  if not randomized:
    for name in ("foot_friction", "base_com", "encoder_bias", "push_robot"):
      cfg.events.pop(name, None)
  command = cfg.commands["twist"]
  command.resampling_time_range = (seconds + 2, seconds + 2)
  for name in (
    "rel_standing_envs",
    "rel_heading_envs",
    "rel_world_envs",
    "rel_forward_envs",
    "rel_backward_envs",
    "rel_lateral_envs",
  ):
    setattr(command, name, 0.0)
  command.heading_command, command.ranges.heading = False, None
  command.debug_vis = False
  if mode == "crawl":
    cfg.events["reset_base"].params["yaw_range"] = (0.0, 0.0)
    cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
  else:
    cfg.events["reset_base"].params["pose_range"] = {
      k: (0.0, 0.0) for k in ("x", "y", "z", "yaw")
    }
  agent_cfg = load_rl_cfg(task)
  if video:
    cfg.viewer.width, cfg.viewer.height = 640, 480
    cfg.viewer.max_extra_envs = 0
    cfg.viewer.elevation, cfg.viewer.azimuth = -18, 125
    cfg.viewer.site_group = (0, 0, 0, 0, 0, 0)
  env = ManagerBasedRlEnv(
    cfg=cfg, device=device, render_mode="rgb_array" if video else None
  )
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  writer = None
  try:
    if checkpoint:
      runner = load_runner_cls(task)(
        wrapped, asdict(agent_cfg), log_dir=None, device=device
      )
      runner.load(
        str(checkpoint),
        load_cfg={
          "actor": True,
          "critic": False,
          "optimizer": False,
          "iteration": False,
        },
        map_location=device,
      )
      runner.alg.eval_mode()
      policy = runner.get_inference_policy(device=device)
    else:

      def policy(obs):
        return torch.zeros(num_envs, env.action_manager.total_action_dim, device=device)

    speed = 0.20 if mode == "biped" else 0.10
    commands = {
      "stand": (0.0, 0.0, 0.0),
      "forward": (speed, 0.0, 0.0),
      "backward": (-speed * 0.6, 0.0, 0.0),
      "lateral": (0.0, 0.06 if mode == "biped" else 0.04, 0.0),
      "turn": (0.0, 0.0, 0.15),
    }
    selected = clips.split(",") if clips else list(commands)
    if not selected or any(name not in commands for name in selected):
      raise ValueError(f"clips must be selected from {list(commands)}")
    result = {
      "task": task,
      "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
      if checkpoint
      else None,
      "seed": seed,
      "num_envs": num_envs,
      "seconds": seconds,
      "randomized": randomized,
      "clips": {},
    }
    robot = env.scene["robot"]
    steps = round(seconds / env.step_dt)
    for name in selected:
      velocity = commands[name]
      for axis, value in zip(
        ("lin_vel_x", "lin_vel_y", "ang_vel_z"), velocity, strict=True
      ):
        setattr(command.ranges, axis, (value, value))
      obs, _ = wrapped.reset()
      expected = torch.tensor(velocity, device=device).expand(num_envs, -1)
      torch.testing.assert_close(env.command_manager.get_command("twist"), expected)
      failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
      first_survival = torch.zeros(num_envs, device=device)
      failures = 0
      sums = torch.zeros(3, device=device)
      abs_errors = torch.zeros(3, device=device)
      displacement = torch.zeros(2, device=device)
      support_sum = 0.0
      valid_count = 0
      if video and name == "forward":
        import imageio.v2 as imageio

        video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video), fps=round(1 / (2 * env.step_dt)))
      with torch.inference_mode():
        for step in range(steps):
          previous = robot.data.root_link_pos_w.clone()
          quat = robot.data.root_link_quat_w.clone()
          actions = policy(obs)
          assert torch.isfinite(actions).all(), "nonfinite policy action"
          obs, reward, done, _ = wrapped.step(actions)
          if writer is not None and step % 2 == 0:
            writer.append_data(env.render())
          assert (
            torch.isfinite(reward).all() and torch.isfinite(robot.data.joint_pos).all()
          ), "nonfinite simulation"
          valid = ~done.bool()
          first_survival += (~failed & valid) * env.step_dt
          failed |= done.bool()
          failures += int(done.sum())
          linear = (
            heading_lin_vel(env) if mode == "crawl" else robot.data.root_link_lin_vel_b
          )
          angular = (
            heading_ang_vel(env) if mode == "crawl" else robot.data.root_link_ang_vel_b
          )
          measured = torch.stack((linear[:, 0], linear[:, 1], angular[:, 2]), -1)
          sums += measured[valid].sum(0)
          abs_errors += (measured - expected)[valid].abs().sum(0)
          heading = crawl_heading(quat) if mode == "crawl" else yaw_quat(quat)
          delta = quat_apply_inverse(heading, robot.data.root_link_pos_w - previous)
          displacement += delta[valid, :2].sum(0)
          support_sum += float(
            (env.scene["feet_ground_contact"].data.found[valid] > 0).float().sum()
          )
          valid_count += int(valid.sum())
      if writer is not None:
        writer.close()
        writer = None
      denominator = max(valid_count, 1)
      metrics = {
        "command": list(velocity),
        "mean_velocity": (sums / denominator).tolist(),
        "velocity_mae": (abs_errors / denominator).tolist(),
        "displacement_velocity_xy": (
          displacement / (denominator * env.step_dt)
        ).tolist(),
        "falls": failures,
        "survival_fraction": float((~failed).float().mean()),
        "mean_time_to_first_fall_s": float(first_survival.mean()),
        "mean_support_count": support_sum / denominator,
        "finite": True,
      }
      result["clips"][name] = metrics
      print(name, json.dumps(metrics), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
  finally:
    if writer is not None:
      writer.close()
    wrapped.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("biped", "crawl"), required=True)
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--seconds", type=float, default=10)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--randomized", action="store_true")
  parser.add_argument(
    "--clips", help="Comma-separated subset: stand,forward,backward,lateral,turn"
  )
  parser.add_argument("--video", type=Path, help="Record the forward clip to this MP4")
  args = parser.parse_args()
  if args.num_envs < 1 or args.seconds <= 0:
    parser.error("num-envs and seconds must be positive")
  evaluate(**vars(args))
