"""Deterministic fixed-command evaluation, streamed video, and review metrics."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.pikachu_v033_crawl.mdp import crawl_heading
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from mjlab.viewer import ViewerConfig


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--seconds", type=float, default=12)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--video", action="store_true")
  parser.add_argument("--view", choices=("side", "front"), default="side")
  parser.add_argument(
    "--clip", choices=("stand", "forward", "backward", "turn"), default="forward"
  )
  args = parser.parse_args()
  cfg = load_env_cfg(args.task, play=True)
  # This script records its own measurements; full pressure telemetry is costly.
  cfg.metrics = {}
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.episode_length_s = args.seconds + 1
  cfg.curriculum = {}
  cfg.events.pop("push_robot", None)
  for name in ("foot_friction", "base_com", "encoder_bias"):
    cfg.events.pop(name, None)
  crawl = "Crawl" in args.task
  if args.video:
    if crawl:
      cfg.events["reset_base"].params["yaw_range"] = (0.0, 0.0)
    else:
      cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
  commands = {
    "stand": (0.0, 0.0, 0.0),
    "forward": (0.15 if crawl else 0.4, 0.0, 0.0),
    "backward": (-0.08 if crawl else -0.3, 0.0, 0.0),
    "turn": (0.0, 0.0, 0.15 if crawl else 0.3),
  }
  command = commands[args.clip]
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.ranges.heading = None
  for attr in (
    "rel_forward_envs",
    "rel_backward_envs",
    "rel_lateral_envs",
    "rel_standing_envs",
    "rel_heading_envs",
    "rel_world_envs",
  ):
    setattr(twist, attr, 0.0)
  twist.ranges.lin_vel_x = (command[0], command[0])
  twist.ranges.lin_vel_y = (command[1], command[1])
  twist.ranges.ang_vel_z = (command[2], command[2])
  cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
  cfg.viewer.entity_name = "robot"
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 0.85 if crawl else 1.0
  cfg.viewer.elevation = -15
  cfg.viewer.azimuth = 90 if args.view == "side" else 0
  cfg.viewer.width, cfg.viewer.height = 1280, 720
  env = ManagerBasedRlEnv(
    cfg, device="cuda:0", render_mode="rgb_array" if args.video else None
  )
  wrapped = RslRlVecEnvWrapper(env)
  if args.checkpoint:
    runner_cls = load_runner_cls(args.task)
    assert runner_cls is not None
    runner = runner_cls(wrapped, asdict(load_rl_cfg(args.task)), None, "cuda:0")
    runner.load(
      args.checkpoint,
      load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
      map_location="cuda:0",
    )
    policy = runner.get_inference_policy("cuda:0")
  else:

    def policy(obs):
      return torch.zeros(
        (args.num_envs, env.action_manager.total_action_dim), device=env.device
      )

  args.out.mkdir(parents=True, exist_ok=True)
  stem = f"{args.clip}_{args.view}" if args.video else f"{args.clip}_metrics"
  writer = (
    imageio.get_writer(
      args.out / f"{stem}.mp4",
      fps=50,
      codec="libx264",
      quality=8,
      ffmpeg_params=["-threads", "2"],
    )
    if args.video
    else None
  )
  obs, _ = env.reset()
  sums = np.zeros(3)
  errors = np.zeros(3)
  falls = np.zeros(args.num_envs, dtype=int)
  contacts = []
  heights = []
  slip_sum = None
  slip_count = None
  support_samples = []
  rows = []
  contact_names = env.scene["feet_ground_contact"].primary_names
  site_names = tuple(
    name.replace("_ankle_link", "_foot").replace("_elbow_link", "_hand")
    for name in contact_names
  )
  site_ids = env.scene["robot"].find_sites(site_names, preserve_order=True)[0]
  steps = int(args.seconds / env.step_dt)
  try:
    for step in range(steps):
      with torch.inference_mode():
        obs, _, terminated, _, _ = env.step(policy(obs))
      for value in obs.values():
        assert isinstance(value, torch.Tensor)
        assert torch.isfinite(value).all()
      robot = env.scene["robot"]
      q = (crawl_heading if crawl else yaw_quat)(robot.data.root_link_quat_w)
      vel = quat_apply_inverse(q, robot.data.root_link_lin_vel_w)
      ang = robot.data.root_link_ang_vel_w
      actual = torch.stack((vel[:, 0], vel[:, 1], ang[:, 2]), -1).cpu().numpy()
      sums += actual.mean(0)
      errors += np.abs(actual - np.array(command)).mean(0)
      falls += terminated.cpu().numpy().astype(int)
      contact = env.scene["feet_ground_contact"].data.found
      contact_mask = (contact > 0).float()
      contacts.append(contact_mask.mean(0).cpu().tolist())
      support_samples.append(contact_mask.sum(-1).cpu().numpy())
      speed = robot.data.site_lin_vel_w[:, site_ids, :2].norm(dim=-1)
      slip = (speed * contact_mask).sum(0).cpu().numpy()
      count = contact_mask.sum(0).cpu().numpy()
      slip_sum = slip if slip_sum is None else slip_sum + slip
      slip_count = count if slip_count is None else slip_count + count
      heights.append(
        float((robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).mean())
      )
      rows.append(
        [
          step * env.step_dt,
          *command,
          *actual.mean(0),
          *np.abs(actual - np.array(command)).mean(0),
          int(terminated.sum()),
          *contacts[-1],
          heights[-1],
        ]
      )
      if writer:
        writer.append_data(np.asarray(env.render()))
  finally:
    if writer:
      writer.close()
    env.close()
  result = {
    "task": args.task,
    "checkpoint": args.checkpoint,
    "seed": args.seed,
    "nominal": True,
    "seconds": args.seconds,
    "num_envs": args.num_envs,
    "command": command,
    "actual_mean": (sums / steps).tolist(),
    "mae": (errors / steps).tolist(),
    "falls_total": int(falls.sum()),
    "survival_fraction": float(np.mean(falls == 0)),
    "contacts_duty": np.mean(contacts, axis=0).tolist(),
    "contact_order": contact_names,
    "contact_slip_speed_mean_m_s": (slip_sum / np.maximum(slip_count, 1)).tolist(),
    "three_or_more_support_fraction": float((np.asarray(support_samples) >= 3).mean()),
    "zero_support_fraction": float((np.asarray(support_samples) == 0).mean()),
    "base_height_mean": float(np.mean(heights)),
    "note": "Auto-resets remain visible; actual/MAE includes reset transients. Nominal evaluation, not robustness certification.",
  }
  (args.out / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
  with (args.out / f"{stem}.csv").open("w") as stream:
    table = csv.writer(stream)
    table.writerow(
      [
        "time_s",
        "command_vx",
        "command_vy",
        "command_wz",
        "actual_vx",
        "actual_vy",
        "actual_wz",
        "mae_vx",
        "mae_vy",
        "mae_wz",
        "falls",
        *[f"contact/{name}" for name in contact_names],
        "base_height_m",
      ]
    )
    table.writerows(rows)
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
