"""Render recorded V033 transfer rollouts through mjlab's MuJoCo renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from v033_video_overlay import VelocityOverlay

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import ViewerConfig


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--metrics", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--task", default="Mjlab-Transfer-Flat-Pikachu-V033-E3")
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--case", default="continuous_cycle")
  parser.add_argument("--env-index", type=int, default=0)
  parser.add_argument("--source-fps", type=int, default=50)
  parser.add_argument("--fps", type=int, default=25)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--view", choices=("three_quarter", "front"), default="three_quarter"
  )
  args = parser.parse_args()

  if args.source_fps % args.fps:
    raise SystemExit("--source-fps must be divisible by --fps")
  stride = args.source_fps // args.fps

  rollout_path = args.metrics / "rollouts.npz"
  with np.load(rollout_path) as samples:
    key = f"{args.case}/qpos"
    if key not in samples:
      raise SystemExit(f"rollout has no case {args.case!r}")
    states = np.asarray(samples[key][:, args.env_index], dtype=np.float32)
    commands = np.asarray(
      samples[f"{args.case}/command"][:, args.env_index, 0], dtype=np.float32
    )

  position = states[:, :3]
  quaternion = states[:, 3:7]
  world_velocity = np.gradient(position, 1.0 / args.source_fps, axis=0)
  w, x, y, z = quaternion.T
  yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
  yaw_rate = np.gradient(yaw, 1.0 / args.source_fps)
  cos_yaw = np.cos(yaw)
  sin_yaw = np.sin(yaw)
  actual_velocity = np.column_stack(
    (
      cos_yaw * world_velocity[:, 0] + sin_yaw * world_velocity[:, 1],
      -sin_yaw * world_velocity[:, 0] + cos_yaw * world_velocity[:, 1],
      yaw_rate,
    )
  )

  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = 1
  cfg.metrics = {}
  cfg.events = {}
  cfg.curriculum = {}
  cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
  cfg.viewer.entity_name = "robot"
  cfg.viewer.width = args.width
  cfg.viewer.height = args.height
  cfg.viewer.distance = 1.0
  cfg.viewer.elevation = -15.0
  cfg.viewer.azimuth = 45.0 if args.view == "three_quarter" else 0.0
  cfg.viewer.max_extra_envs = 0

  env = ManagerBasedRlEnv(cfg, device=args.device, render_mode="rgb_array")
  args.out.mkdir(parents=True, exist_ok=True)
  model_tag = args.checkpoint.stem if args.checkpoint else "recorded"
  video_path = args.out / f"{args.case}_{model_tag}_{args.view}_mjlab_mujoco.mp4"
  image_dir = args.out.parent / "images"
  image_dir.mkdir(parents=True, exist_ok=True)
  contact_sheet_path = (
    image_dir / f"{args.case}_{model_tag}_{args.view}_contact_sheet.png"
  )

  frame_indices = list(range(0, len(states), stride))
  sheet_indices = set(np.linspace(0, len(frame_indices) - 1, 8, dtype=int))
  sheet_frames: list[Image.Image] = []
  writer = imageio.get_writer(
    video_path,
    fps=args.fps,
    codec="libx264",
    quality=8,
    macro_block_size=None,
    ffmpeg_params=["-threads", "2"],
  )
  velocity_overlay = VelocityOverlay(
    fps=args.fps, history_seconds=4.0, limits=(0.8, 0.5, 1.0)
  )
  try:
    env.reset()
    for output_index, source_index in enumerate(frame_indices):
      env.sim.data.qpos[0].copy_(
        torch.as_tensor(states[source_index], device=env.device)
      )
      env.sim.data.qvel[0].zero_()
      frame = velocity_overlay.draw(
        np.asarray(env.render()), (0.0, 0.0, 0.0), actual_velocity[source_index]
      )
      writer.append_data(frame)
      if output_index in sheet_indices:
        tile = Image.fromarray(frame).resize((320, 180))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 320, 30), fill=(255, 255, 255))
        draw.text(
          (8, 8),
          (
            f"t={source_index / args.source_fps:.1f}s  "
            f"posture={commands[source_index]:.2f}  recorded"
          ),
          fill=(0, 0, 0),
        )
        sheet_frames.append(tile)
  finally:
    writer.close()
    env.close()

  sheet = Image.new("RGB", (1280, 360), "white")
  for index, frame in enumerate(sheet_frames):
    sheet.paste(frame, ((index % 4) * 320, (index // 4) * 180))
  sheet.save(contact_sheet_path)

  metadata = {
    "task": args.task,
    "checkpoint": str(args.checkpoint) if args.checkpoint else None,
    "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
    "rollout": str(rollout_path),
    "rollout_sha256": sha256(rollout_path),
    "case": args.case,
    "environment_index": args.env_index,
    "source_frames": len(states),
    "rendered_frames": len(frame_indices),
    "source_fps": args.source_fps,
    "video_fps": args.fps,
    "duration_seconds": len(frame_indices) / args.fps,
    "resolution": [args.width, args.height],
    "view": args.view,
    "camera_azimuth_degrees": cfg.viewer.azimuth,
    "renderer": "mjlab.ManagerBasedRlEnv rgb_array / OffscreenRenderer / MuJoCo",
    "velocity_overlay": {
      "position": "bottom_right",
      "history_seconds": 4.0,
      "panel_fraction": [0.28, 0.30],
      "font_pixels": {"label": 15, "value": 13},
      "series": ["vx", "vy", "wz"],
      "command": "zero, dashed",
      "actual": "finite difference from recorded qpos, solid",
    },
    "video": str(video_path),
    "video_sha256": sha256(video_path),
    "contact_sheet": str(contact_sheet_path),
    "contact_sheet_sha256": sha256(contact_sheet_path),
    "note": (
      "Rendered from recorded simulated qpos without reference-motion "
      "interpolation. This is the last trained checkpoint, not an accepted model."
    ),
  }
  metadata_path = args.out.parent / "metrics" / f"render_metadata_{args.view}.json"
  metadata_path.parent.mkdir(parents=True, exist_ok=True)
  metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
  print(json.dumps(metadata, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
