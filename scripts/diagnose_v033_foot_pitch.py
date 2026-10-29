"""Bounded paired rollout; record states for CPU sole/contact geometry analysis."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.pikachu_v033_crawl.mdp import heading_lin_vel
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply


def parse_reward_weights(values: list[str]) -> dict[str, float]:
  result: dict[str, float] = {}
  for value in values:
    name, separator, raw_weight = value.partition("=")
    if not separator or not name or not raw_weight:
      raise ValueError(f"Invalid reward override {value!r}; expected NAME=WEIGHT")
    result[name] = float(raw_weight)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--baseline", type=Path)
  parser.add_argument("--task", default="Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E2")
  parser.add_argument("--label", default="candidate")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=2, choices=(1, 2, 4, 8, 16, 20))
  parser.add_argument("--seconds", type=float, default=8)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--randomized", action="store_true")
  parser.add_argument("--ankle-feedback", type=float, default=0.0)
  parser.add_argument("--speeds", type=float, nargs="+")
  parser.add_argument(
    "--reward-weight",
    action="append",
    default=[],
    metavar="NAME=WEIGHT",
  )
  args = parser.parse_args()
  if not 0 <= args.ankle_feedback <= 0.5:
    raise ValueError("Diagnostic feedback gain must be in [0, 0.5]")
  if not 3 <= args.seconds <= 12:
    raise ValueError("Keep this diagnostic between 3 and 12 seconds per case")
  mem_available_kib = int(
    next(
      line
      for line in Path("/proc/meminfo").read_text().splitlines()
      if line.startswith("MemAvailable:")
    ).split()[1]
  )
  if mem_available_kib < 2 * 1024**2:
    raise RuntimeError("Need at least 2 GiB available host RAM for this diagnostic")
  torch.set_num_threads(2)
  args.out.mkdir(parents=True, exist_ok=True)
  if (args.out / "rollouts.npz").exists():
    raise FileExistsError(args.out / "rollouts.npz")
  task = args.task
  cfg = load_env_cfg(task, play=True)
  reward_weights = parse_reward_weights(args.reward_weight)
  unknown_rewards = sorted(set(reward_weights) - set(cfg.rewards))
  if unknown_rewards:
    raise ValueError(f"Unknown reward terms: {', '.join(unknown_rewards)}")
  for name, weight in reward_weights.items():
    cfg.rewards[name].weight = weight
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.metrics = {}
  cfg.episode_length_s = args.seconds + 2
  cfg.curriculum = {}
  if not args.randomized:
    for name in ("foot_friction", "base_com", "encoder_bias", "push_robot"):
      cfg.events.pop(name, None)
  else:
    cfg.observations["actor"].enable_corruption = True
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.005, 0.005)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.02, 0.02)
  cfg.events["reset_base"].params["yaw_range"] = (0.0, 0.0)
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.resampling_time_range = (100.0, 100.0)
  for attr in (
    "rel_forward_envs",
    "rel_backward_envs",
    "rel_lateral_envs",
    "rel_standing_envs",
    "rel_heading_envs",
    "rel_world_envs",
  ):
    setattr(twist, attr, 0.0)
  twist.ranges.lin_vel_x = (0.0, 0.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  arrays = {}
  cases = []
  try:
    model = env.sim.mj_model
    # Save the exact compiled model for single-threaded CPU kinematic replay.
    mujoco.mj_saveModel(model, str(args.out / "compiled_model.mjb"), None)
    robot = env.scene["robot"]
    names = [
      f"{side}_{kind}_joint"
      for side in ("right", "left")
      for kind in ("hip_pitch", "knee_up", "knee", "ankle")
    ]
    ids, names = robot.find_joints(names, preserve_order=True)
    action = env.action_manager.get_term("joint_pos")
    assert isinstance(action, JointPositionAction)
    foot_ids = robot.find_bodies(
      ("right_ankle_link", "left_ankle_link"), preserve_order=True
    )[0]
    ankle_channels = [
      action.target_names.index(f"{side}_ankle_joint") for side in ("right", "left")
    ]
    contact_channels = [
      env.scene["feet_ground_contact"].primary_names.index(f"{side}_ankle_link")
      for side in ("right", "left")
    ]
    runner_cls = load_runner_cls(task)
    assert runner_cls is not None
    runner = runner_cls(
      RslRlVecEnvWrapper(env), asdict(load_rl_cfg(task)), None, "cuda:0"
    )
    checkpoints = [(args.label, args.checkpoint)]
    if args.baseline:
      checkpoints.append(("baseline", args.baseline))
    for label, checkpoint in checkpoints:
      runner.load(
        str(checkpoint),
        load_cfg={
          "actor": True,
          "critic": False,
          "optimizer": False,
          "iteration": False,
        },
        map_location="cuda:0",
      )
      policy = runner.get_inference_policy("cuda:0")
      speeds = [0.0, 0.08, -0.08, 0.15, -0.15]
      if args.speeds:
        speeds = args.speeds
      for speed in speeds:
        key = f"{label}_{speed:+.2f}"
        torch.manual_seed(args.seed)
        obs, _ = env.reset()
        command = env.command_manager.get_command("twist")
        assert command is not None
        command[:, 0] = speed
        # Recompute observation so the first action sees the paired command.
        obs = env.observation_manager.compute(update_history=True)
        records = {
          name: []
          for name in (
            "qpos",
            "qvel",
            "contact",
            "force",
            "joint",
            "target",
            "velocity",
            "terminated",
            "rewards",
            "actuator_force",
          )
        }
        for _ in range(round(args.seconds / env.step_dt)):
          with torch.inference_mode():
            outputs = policy(obs)
            if args.ankle_feedback:
              quat = robot.data.body_link_quat_w[:, foot_ids]
              x_axis = quat_apply(
                quat, quat.new_tensor([1.0, 0.0, 0.0]).expand(*quat.shape[:-1], 3)
              )
              z_axis = quat_apply(
                quat, quat.new_tensor([0.0, 0.0, 1.0]).expand(*quat.shape[:-1], 3)
              )
              delta = torch.atan2(x_axis[..., 2], z_axis[..., 2])
              delta *= delta.new_tensor([1.0, -1.0])
              contact = (
                env.scene["feet_ground_contact"].data.found[:, contact_channels] > 0
              )
              # Diagnostic feedback only: no policy weights or model edits.
              outputs = outputs.clone()
              outputs[:, ankle_channels] += args.ankle_feedback * delta * contact / 0.25
            obs, _, terminated, _, _ = env.step(outputs)
          records["qpos"].append(env.sim.data.qpos.cpu().numpy().copy())
          records["qvel"].append(env.sim.data.qvel.cpu().numpy().copy())
          contact = env.scene["feet_ground_contact"].data
          records["contact"].append(contact.found.cpu().numpy().copy())
          records["force"].append(contact.force.cpu().numpy().copy())
          records["joint"].append(robot.data.joint_pos[:, ids].cpu().numpy().copy())
          records["target"].append(robot.data.joint_pos_target.cpu().numpy().copy())
          records["velocity"].append(heading_lin_vel(env).cpu().numpy().copy())
          records["terminated"].append(terminated.cpu().numpy().copy())
          records["rewards"].append(
            env.reward_manager._step_reward.cpu().numpy().copy()
          )
          records["actuator_force"].append(
            robot.data.actuator_force.cpu().numpy().copy()
          )
        for name, values in records.items():
          arrays[f"{key}/{name}"] = np.asarray(values)
        cases.append(
          {"key": key, "checkpoint": str(checkpoint.resolve()), "speed": speed}
        )
        print(
          f"Completed {key}: falls={int(np.sum(records['terminated']))}", flush=True
        )
    metadata = {
      "task": task,
      "num_envs": args.num_envs,
      "seconds": args.seconds,
      "dt": env.step_dt,
      "warmup_s": 2.0,
      "seed": args.seed,
      "diagnostic_ankle_feedback_gain": args.ankle_feedback,
      "nominal": not args.randomized,
      "randomized": args.randomized,
      "rendering": False,
      "reward_weight_overrides": reward_weights,
      "cases": cases,
      "joint_names": names,
      "all_joint_names": robot.joint_names,
      "action_names": action.target_names,
      "contact_names": env.scene["feet_ground_contact"].primary_names,
      "reward_names": env.reward_manager.active_terms,
      "baseline_note": "Every checkpoint is evaluated in the selected task's identical dynamics and observations.",
    }
    np.savez_compressed(args.out / "rollouts.npz", **arrays)
    (args.out / "rollouts.json").write_text(json.dumps(metadata, indent=2) + "\n")
  finally:
    env.close()


if __name__ == "__main__":
  main()
