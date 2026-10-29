"""Evaluate V035 biped / crawl policies with fixed commands, video and series.

For each clip (stand/forward/backward/lateral/turn) this script:

* pins the velocity command and disables resampling,
* rolls out the policy with nominal (no-randomization) startup,
* records per-step state/velocity/contact/joint series for envelope plots,
* optionally writes one MP4 and a 4x2 contact sheet per clip.

Example::

    uv run python scripts/eval_v035.py --mode biped \
        --checkpoint logs/rsl_rl/.../model_2999.pt \
        --output reports/v035/biped/eval.json --video-dir reports/v035/biped
"""

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

import mjlab.tasks  # noqa: F401
from mjlab.asset_zoo.robots.pikachu.pikachu_v035_constants import ACTUATOR_GROUPS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.pikachu_v035 import crawl_mdp
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from mjlab.viewer import ViewerConfig

TASKS = {
  "biped": "Mjlab-Velocity-Flat-Pikachu-V035-Biped-E1",
  "walk": "Mjlab-Velocity-Flat-Pikachu-V035-Biped-Walk",
  "crawl": "Mjlab-Velocity-Flat-Pikachu-V035-Crawl",
}
# 直立两档共用同一套机体坐标约定与地面站高。
BIPED_MODES = ("biped", "walk")


def _overlay(frame: np.ndarray, text: str) -> np.ndarray:
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image, "RGBA")
  draw.rectangle((0, 0, image.width, 26), fill=(255, 255, 255, 200))
  draw.text((8, 7), text, fill=(0, 0, 0, 255))
  return np.asarray(image)


def evaluate(
  mode,
  checkpoint,
  output,
  task=None,
  num_envs=16,
  seconds=8.0,
  seed=42,
  device="cuda:0",
  randomized=False,
  clips=None,
  video_dir=None,
  forward_speed=None,
  backward_speed=None,
  lateral_speed=None,
  turn_speed=None,
):
  task = task or TASKS[mode]
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs, cfg.seed = num_envs, seed
  cfg.episode_length_s = seconds + 2
  cfg.observations["actor"].enable_corruption = False
  cfg.curriculum = {}
  cfg.metrics = {}  # series are recorded directly below
  if not randomized:
    for name in ("foot_friction", "base_com", "encoder_bias", "push_robot"):
      cfg.events.pop(name, None)
  command = cfg.commands["twist"]
  command.resampling_time_range = (seconds + 3, seconds + 3)
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
    cfg.events["reset_base"].params["pose_range"] = {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "z": (0.0, 0.0),
    }
  else:
    cfg.events["reset_base"].params["pose_range"] = {
      k: (0.0, 0.0) for k in ("x", "y", "z", "yaw")
    }

  if video_dir is not None:
    video_dir = Path(video_dir)
    cfg.viewer.width, cfg.viewer.height = 640, 480
    cfg.viewer.max_extra_envs = 0
    cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    cfg.viewer.entity_name = "robot"
    cfg.viewer.site_group = (0, 0, 0, 0, 0, 0)
    cfg.viewer.distance = 0.95 if mode in BIPED_MODES else 0.62
    cfg.viewer.elevation = -12.0
    cfg.viewer.azimuth = 55.0 if mode in BIPED_MODES else 45.0

  agent_cfg = load_rl_cfg(task)
  env = ManagerBasedRlEnv(
    cfg=cfg, device=device, render_mode="rgb_array" if video_dir else None
  )
  if video_dir is not None:
    # 关掉调试叠加（命令箭头/触点小球），报告视频只要机器人本体。
    env.update_visualizers = lambda visualizer: None
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  try:
    if checkpoint is not None:
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

    speed = 0.20 if mode in BIPED_MODES else 0.10
    forward_speed = speed if forward_speed is None else forward_speed
    backward_speed = -speed * 0.6 if backward_speed is None else backward_speed
    lateral_speed = (
      (0.06 if mode in BIPED_MODES else 0.04)
      if lateral_speed is None
      else lateral_speed
    )
    turn_speed = 0.15 if turn_speed is None else turn_speed
    commands = {
      "stand": (0.0, 0.0, 0.0),
      "forward": (forward_speed, 0.0, 0.0),
      "backward": (backward_speed, 0.0, 0.0),
      "lateral": (0.0, lateral_speed, 0.0),
      "turn": (0.0, 0.0, turn_speed),
    }
    selected = clips.split(",") if clips else list(commands)
    if not selected or any(name not in commands for name in selected):
      raise ValueError(f"clips must be selected from {list(commands)}")

    robot = env.scene["robot"]
    joint_names = robot.joint_names
    ctrl_index = {name: robot.actuator_names.index(name) for name in joint_names}
    ctrl_ids = torch.tensor([ctrl_index[name] for name in joint_names], device=device)
    support_sensor = env.scene["feet_ground_contact"]
    steps = round(seconds / env.step_dt)
    result = {
      "task": task,
      "mode": mode,
      "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
      "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
      if checkpoint
      else None,
      "seed": seed,
      "num_envs": num_envs,
      "seconds": seconds,
      "randomized": randomized,
      "joint_names": list(joint_names),
      "soft_joint_pos_limits": robot.data.soft_joint_pos_limits[0].cpu().tolist(),
      "joint_specs": {
        joint: {
          "effort_limit": float(actuator.effort_limit),
          "velocity_limit": float(actuator.velocity_limit),
        }
        for joint in joint_names
        for pattern, actuator in ACTUATOR_GROUPS
        if re.fullmatch(pattern, joint)
      },
      "clips": {},
    }
    series = {}
    for name in selected:
      velocity = commands[name]
      for axis, value in zip(
        ("lin_vel_x", "lin_vel_y", "ang_vel_z"), velocity, strict=True
      ):
        setattr(command.ranges, axis, (value, value))
      obs, _ = wrapped.reset()
      expected = torch.tensor(velocity, device=device).expand(num_envs, -1)
      torch.testing.assert_close(env.command_manager.get_command("twist"), expected)

      writer = None
      sheet_indices: set[int] = set()
      if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
          video_dir / f"{mode}_{name}.mp4",
          fps=round(1.0 / env.step_dt),
          codec="libx264",
          quality=8,
          macro_block_size=None,
          ffmpeg_params=["-threads", "2"],
        )
        sheet_indices = set(np.linspace(0, steps - 1, 8, dtype=int))
      sheet_frames: list[Image.Image] = []

      failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
      first_survival = torch.zeros(num_envs, device=device)
      failures = 0
      sums = torch.zeros(3, device=device)
      abs_errors = torch.zeros(3, device=device)
      displacement = torch.zeros(2, device=device)
      support_sum = 0.0
      valid_count = 0
      rows = []
      with torch.no_grad():
        for step in range(steps):
          previous = robot.data.root_link_pos_w.clone()
          quat = robot.data.root_link_quat_w.clone()
          actions = policy(obs)
          assert torch.isfinite(actions).all(), "nonfinite policy action"
          obs, reward, done, _ = wrapped.step(actions)
          assert (
            torch.isfinite(reward).all() and torch.isfinite(robot.data.joint_pos).all()
          ), "nonfinite simulation"
          valid = ~done.bool()
          first_survival += (~failed & valid) * env.step_dt
          failed |= done.bool()
          failures += int(done.sum())
          linear = (
            crawl_mdp.heading_lin_vel(env)
            if mode == "crawl"
            else robot.data.root_link_lin_vel_b
          )
          angular = (
            crawl_mdp.heading_ang_vel(env)
            if mode == "crawl"
            else robot.data.root_link_ang_vel_b
          )
          measured = torch.stack((linear[:, 0], linear[:, 1], angular[:, 2]), -1)
          sums += measured[valid].sum(0)
          abs_errors += (measured - expected)[valid].abs().sum(0)
          heading = crawl_mdp.crawl_heading(quat) if mode == "crawl" else yaw_quat(quat)
          delta = quat_apply_inverse(heading, robot.data.root_link_pos_w - previous)
          displacement += delta[valid, :2].sum(0)
          found = support_sensor.data.found
          assert found is not None
          support_sum += float((found[valid] > 0).float().sum())
          valid_count += int(valid.sum())

          rows.append(
            {
              "t": (step + 1) * env.step_dt,
              "command": expected[0].cpu().numpy(),
              "velocity": measured.cpu().numpy(),
              "base_pos": robot.data.root_link_pos_w[0].cpu().numpy(),
              "base_quat": robot.data.root_link_quat_w[0].cpu().numpy(),
              "base_height": float(robot.data.root_link_pos_w[0, 2]),
              "joint_pos": robot.data.joint_pos[0].cpu().numpy(),
              "joint_vel": robot.data.joint_vel[0].cpu().numpy(),
              "joint_torque": robot.data.qfrc_actuator[0].cpu().numpy(),
              "joint_target": robot.data.ctrl[0][ctrl_ids].cpu().numpy(),
              "support_found": (found[0] > 0).cpu().numpy(),
              "action": actions[0].cpu().numpy(),
            }
          )

          if writer is not None:
            frame = _overlay(
              np.asarray(env.render()),
              f"{name}  t={((step + 1) * env.step_dt):.1f}s  "
              f"cmd=({velocity[0]:+.2f},{velocity[1]:+.2f},{velocity[2]:+.2f})  "
              f"act=({measured[0, 0]:+.2f},{measured[0, 1]:+.2f},{measured[0, 2]:+.2f})",
            )
            writer.append_data(frame)
            if step in sheet_indices:
              sheet_frames.append(
                Image.fromarray(frame).resize((320, 180), Image.Resampling.LANCZOS)
              )

      if writer is not None:
        writer.close()
        sheet = Image.new("RGB", (1280, 360), "white")
        for index, frame in enumerate(sheet_frames[:8]):
          sheet.paste(frame, ((index % 4) * 320, (index // 4) * 180))
        sheet.save(video_dir / f"{mode}_{name}_contact_sheet.png")

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
      series[name] = rows
      print(name, json.dumps(metrics), flush=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    # Per-step series: object arrays of dicts are awkward in npz, so flatten.
    npz = {
      f"{clip}/{key}": np.stack([row[key] for row in rows])
      for clip, rows in series.items()
      for key in rows[0]
    }
    np.savez_compressed(output.with_suffix(".npz"), **npz)
    (output.parent / "joint_names.json").write_text(
      json.dumps(list(joint_names), indent=2) + "\n"
    )
    return result
  finally:
    wrapped.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("biped", "walk", "crawl"), required=True)
  parser.add_argument("--task")
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--seconds", type=float, default=8)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--randomized", action="store_true")
  parser.add_argument("--forward-speed", type=float)
  parser.add_argument("--backward-speed", type=float)
  parser.add_argument("--lateral-speed", type=float)
  parser.add_argument("--turn-speed", type=float)
  parser.add_argument("--clips")
  parser.add_argument("--video-dir", type=Path)
  args = parser.parse_args()
  if args.num_envs < 1 or args.seconds <= 0:
    parser.error("num-envs and seconds must be positive")
  evaluate(**vars(args))
