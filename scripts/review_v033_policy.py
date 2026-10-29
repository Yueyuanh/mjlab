"""Deterministic fixed-command evaluation, streamed video, and review metrics."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from v033_video_overlay import VelocityOverlay

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.pikachu_v033_crawl.mdp import crawl_heading
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse, yaw_quat
from mjlab.viewer import ViewerConfig


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--name",
    help="Optional artifact stem; keeps checkpoint sweeps in one evaluation folder.",
  )
  parser.add_argument("--seconds", type=float, default=12)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--video", action="store_true")
  parser.add_argument(
    "--view", choices=("side", "three_quarter", "front"), default="side"
  )
  parser.add_argument(
    "--clip",
    choices=("stand", "forward", "backward", "turn", "lateral"),
    default="forward",
  )
  parser.add_argument(
    "--command-x",
    type=float,
    help="Override the selected clip's x velocity for fixed-speed reviews.",
  )
  parser.add_argument(
    "--disturbed",
    action="store_true",
    help=(
      "Keep push_robot, foot friction, base COM and encoder bias. The default "
      "(nominal) run measures gait capability only; this flag measures whether "
      "a policy survives the conditions it is actually trained in."
    ),
  )
  args = parser.parse_args()
  cfg = load_env_cfg(args.task, play=True)
  # This script records its own measurements; full pressure telemetry is costly.
  cfg.metrics = {}
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.episode_length_s = args.seconds + 1
  cfg.curriculum = {}
  if not args.disturbed:
    cfg.events.pop("push_robot", None)
    for name in ("foot_friction", "base_com", "encoder_bias"):
      cfg.events.pop(name, None)
  else:
    # play=True configs drop push_robot entirely, so "keep the events" alone
    # would silently test friction/COM only. Re-inject the training push
    # schedule to reproduce the conditions the policy is actually trained in.
    train_cfg = load_env_cfg(args.task, play=False)
    if "push_robot" in train_cfg.events:
      cfg.events["push_robot"] = train_cfg.events["push_robot"]
  crawl = "Crawl" in args.task
  if args.video and crawl and args.view != "side":
    parser.error("quadruped/crawl evaluation videos must use --view side")
  if args.video and not crawl and args.view == "side":
    parser.error("biped evaluation videos must use --view three_quarter or front")
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
    "lateral": (0.0, 0.10 if crawl else 0.2, 0.0),
  }
  command = commands[args.clip]
  if args.command_x is not None:
    command = (args.command_x, command[1], command[2])
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
  cfg.viewer.azimuth = {"side": 90, "three_quarter": 45, "front": 0}[args.view]
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
  stem = args.name or (
    f"{args.clip}_{args.view}" if args.video else f"{args.clip}_metrics"
  )
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
  velocity_overlay = VelocityOverlay(fps=50) if writer else None
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
  hip_samples, width_samples, contact_samples = [], [], []
  near_samples, penetration_samples, arm_samples = [], [], []
  elbow_samples, endpoint_samples = [], []
  leg_joint_samples, foot_contact_samples, foot_height_samples, foot_x_samples = (
    [],
    [],
    [],
    [],
  )
  base_euler_samples, base_ang_vel_samples = [], []
  robot = env.scene["robot"]
  elbow_ids = robot.find_joints(
    ("right_elbow_joint", "left_elbow_joint"), preserve_order=True
  )[0]
  paired_sites = (
    robot.find_sites(
      ("right_hand", "right_foot", "left_hand", "left_foot"),
      preserve_order=True,
    )[0]
    if crawl
    else []
  )
  hip_ids, hip_names = robot.find_joints(
    ("right_hip_roll_joint", "left_hip_roll_joint"), preserve_order=True
  )
  leg_joint_ids, leg_joint_names = robot.find_joints(
    tuple(
      f"{side}_{kind}_joint"
      for side in ("right", "left")
      for kind in ("hip_pitch", "knee_up", "knee_mid", "knee", "ankle")
    ),
    preserve_order=True,
  )
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
      if crawl and step * env.step_dt >= 1.0:
        elbow_samples.append(robot.data.joint_pos[:, elbow_ids].cpu().numpy())
        points = robot.data.site_pos_w[:, paired_sites]
        endpoint_samples.append(
          torch.stack(
            (
              (points[:, 0] - points[:, 1]).norm(dim=-1),
              (points[:, 2] - points[:, 3]).norm(dim=-1),
            ),
            -1,
          )
          .cpu()
          .numpy()
        )
      # Gait metrics used to be biped-only. The crawl experiments ask the same
      # question ("does the policy bend the knee to swing a limb, or does it
      # shuffle the whole body?"), and the crawl robot carries the same
      # hip/knee_up/knee_mid/knee/ankle chain, so sample it there too. For crawl
      # ``site_ids`` covers ankles *and* elbows, so the "foot" entries are
      # [R ankle, L ankle, R elbow, L elbow]; the per-side knee conditioning
      # still keys off the ankle contacts only.
      if step * env.step_dt >= 1.0:
        hip_samples.append(robot.data.joint_pos[:, hip_ids].cpu().numpy())
        leg_joint_samples.append(robot.data.joint_pos[:, leg_joint_ids].cpu().numpy())
        foot_contact_samples.append(contact_mask.bool().cpu().numpy())
        roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_link_quat_w)
        base_euler_samples.append(torch.stack((roll, pitch, yaw), -1).cpu().numpy())
        base_ang_vel_samples.append(robot.data.root_link_ang_vel_b.cpu().numpy())
        relative_feet = quat_apply_inverse(
          q[:, None, :].expand(-1, len(site_ids), -1),
          robot.data.site_pos_w[:, site_ids] - robot.data.root_link_pos_w[:, None, :],
        )
        foot_x_samples.append(relative_feet[:, :, 0].cpu().numpy())
        foot_height_samples.append(
          (robot.data.site_pos_w[:, site_ids, 2] - env.scene.env_origins[:, None, 2])
          .cpu()
          .numpy()
        )
        delta = (
          robot.data.site_pos_w[:, site_ids[0]] - robot.data.site_pos_w[:, site_ids[1]]
        )
        width_samples.append(quat_apply_inverse(q, delta)[:, 1].abs().cpu().numpy())
        contact_samples.append(contact_mask.bool().all(-1).cpu().numpy())
        if "arm_leg_contact" in env.scene.sensors:
          data = env.scene["arm_leg_contact"].data
          near_samples.append((data.found > 0).any(-1).cpu().numpy())
          penetration_samples.append(
            ((data.found > 0) & (data.dist < 0)).any(-1).cpu().numpy()
          )
        if "arm_position_window" in cfg.rewards:
          term = env.reward_manager.get_term_cfg("arm_position_window")
          arm_samples.append(term.func(env, **term.params).cpu().numpy())
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
        frame = np.asarray(env.render())
        assert velocity_overlay is not None
        writer.append_data(velocity_overlay.draw(frame, command, actual[0]))
  finally:
    if writer:
      writer.close()
    env.close()
  result = {
    "task": args.task,
    "checkpoint": args.checkpoint,
    "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    if args.checkpoint
    else None,
    "renderer": "mjlab OffscreenRenderer / MuJoCo" if args.video else None,
    "camera": {
      "azimuth": cfg.viewer.azimuth,
      "elevation": cfg.viewer.elevation,
      "body": "base_link",
      "fps": 50,
    }
    if args.video
    else None,
    "seed": args.seed,
    "nominal": not args.disturbed,
    "seconds": args.seconds,
    "num_envs": args.num_envs,
    "view": args.view if args.video else None,
    "velocity_overlay": {
      "position": "bottom_right",
      "history_seconds": 4.0,
      "panel_fraction": [0.30, 0.32],
      "font_pixels": {"label": 17, "value": 15},
      "series": ["vx", "vy", "wz"],
      "command": "dashed",
      "actual": "solid",
    }
    if args.video
    else None,
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
  if hip_samples:
    hips = np.concatenate(hip_samples)
    leg_joints = np.concatenate(leg_joint_samples)
    foot_contacts = np.concatenate(foot_contact_samples)
    foot_heights = np.concatenate(foot_height_samples)
    foot_x = np.concatenate(foot_x_samples)
    base_euler = np.concatenate(base_euler_samples)
    base_ang_vel = np.concatenate(base_ang_vel_samples)
    widths = np.concatenate(width_samples)
    support = np.concatenate(contact_samples)
    joint_p05 = np.quantile(leg_joints, 0.05, axis=0)
    joint_p95 = np.quantile(leg_joints, 0.95, axis=0)
    conditional_knee = {}
    for side_index, side in enumerate(("right", "left")):
      for joint_offset, kind in ((1, "knee_up"), (3, "knee")):
        values = leg_joints[:, side_index * 5 + joint_offset]
        stance = values[foot_contacts[:, side_index]]
        swing = values[~foot_contacts[:, side_index]]
        conditional_knee[f"{side}_{kind}"] = {
          "stance_abs_mean_deg": float(np.rad2deg(np.abs(stance)).mean()),
          "swing_abs_mean_deg": float(np.rad2deg(np.abs(swing)).mean()),
          "swing_minus_stance_abs_mean_deg": float(
            np.rad2deg(np.abs(swing)).mean() - np.rad2deg(np.abs(stance)).mean()
          ),
        }
    result["gait"] = {
      "settling_seconds": 1.0,
      "hip_order": hip_names,
      "hip_roll_mean_deg": np.rad2deg(hips).mean(0).tolist(),
      "hip_roll_abs_mean_deg": np.rad2deg(np.abs(hips)).mean(0).tolist(),
      "hip_roll_abs_p95_deg": np.rad2deg(
        np.quantile(np.abs(hips), 0.95, axis=0)
      ).tolist(),
      "hip_roll_near_limit_fraction": (np.abs(hips) > 0.24).mean(0).tolist(),
      "width_mean_m": float(widths.mean()),
      "width_p95_m": float(np.quantile(widths, 0.95)),
      "double_support_width_mean_m": float(widths[support].mean())
      if support.any()
      else None,
      "arm_leg_near_fraction": float(np.mean(near_samples)) if near_samples else None,
      "arm_leg_penetration_fraction": float(np.mean(penetration_samples))
      if penetration_samples
      else None,
      "arm_window_cost_mean": float(np.mean(arm_samples)) if arm_samples else None,
      "leg_joint_order": leg_joint_names,
      "leg_joint_mean_deg": np.rad2deg(leg_joints).mean(0).tolist(),
      "leg_joint_p05_deg": np.rad2deg(joint_p05).tolist(),
      "leg_joint_p95_deg": np.rad2deg(joint_p95).tolist(),
      "leg_joint_p05_p95_span_deg": np.rad2deg(joint_p95 - joint_p05).tolist(),
      "conditional_knee_flexion": conditional_knee,
      "foot_order": site_names,
      "foot_height_mean_m": foot_heights.mean(0).tolist(),
      "foot_height_p95_m": np.quantile(foot_heights, 0.95, axis=0).tolist(),
      "foot_sagittal_p05_p95_span_m": (
        np.quantile(foot_x, 0.95, axis=0) - np.quantile(foot_x, 0.05, axis=0)
      ).tolist(),
      "base_euler_order": ["roll", "pitch", "yaw"],
      "base_euler_mean_deg": np.rad2deg(base_euler).mean(0).tolist(),
      "base_euler_std_deg": np.rad2deg(base_euler).std(0).tolist(),
      "base_euler_abs_p95_deg": np.rad2deg(
        np.quantile(np.abs(base_euler), 0.95, axis=0)
      ).tolist(),
      "base_ang_vel_abs_mean_rad_s": np.abs(base_ang_vel).mean(0).tolist(),
      "base_ang_vel_abs_p95_rad_s": np.quantile(
        np.abs(base_ang_vel), 0.95, axis=0
      ).tolist(),
      "double_support_fraction": float(support.mean()),
    }
  if elbow_samples:
    elbows = np.rad2deg(np.asarray(elbow_samples))
    distances = np.asarray(endpoint_samples)
    result["crawl_posture"] = {
      "order": ["right", "left"],
      "elbow_mean_deg": elbows.mean(axis=(0, 1)).tolist(),
      "elbow_abs_mean_deg": np.abs(elbows).mean(axis=(0, 1)).tolist(),
      "elbow_p05_deg": np.quantile(elbows, 0.05, axis=(0, 1)).tolist(),
      "elbow_p95_deg": np.quantile(elbows, 0.95, axis=(0, 1)).tolist(),
      "elbow_abs_p95_deg": np.quantile(np.abs(elbows), 0.95, axis=(0, 1)).tolist(),
      "elbow_mean_per_env_p95_p05_span_deg": (
        np.quantile(elbows, 0.95, axis=0) - np.quantile(elbows, 0.05, axis=0)
      )
      .mean(axis=0)
      .tolist(),
      "same_side_endpoint_distance_mean_m": distances.mean(axis=(0, 1)).tolist(),
      "same_side_endpoint_distance_p05_m": np.quantile(
        distances, 0.05, axis=(0, 1)
      ).tolist(),
      "same_side_endpoint_distance_p95_m": np.quantile(
        distances, 0.95, axis=(0, 1)
      ).tolist(),
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
