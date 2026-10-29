"""Four independent endpoint/command cases plus continuous cycles, no auto-reset."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.pikachu_v033_crawl_e2.mdp import sole_geometry
from mjlab.tasks.velocity.config.pikachu_v033_transfer import mdp


def set_posture_goal(term, goal: int) -> None:
  """Set an endpoint immediately for legacy commands or slew it for E3."""
  if all(hasattr(term, name) for name in ("goal", "start", "elapsed", "duration")):
    term.start[:, 0] = term.value[:, 0]
    term.goal[:, 0] = goal
    term.elapsed.zero_()
    duration = term.cfg.down_duration_s if goal else term.cfg.up_duration_s
    term.duration.fill_(duration)
  else:
    term.value.fill_(goal)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--task", default="Mjlab-Transfer-Flat-Pikachu-V033")
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--seconds", type=float, default=6.0)
  parser.add_argument("--settle-seconds", type=float, default=0.5)
  args = parser.parse_args()
  if args.settle_seconds <= 0 or args.settle_seconds > args.seconds:
    parser.error("--settle-seconds must be positive and no longer than --seconds")
  torch.set_num_threads(2)
  args.out.mkdir(parents=True, exist_ok=False)
  task = args.task
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.terminations = {}  # Failed trajectories continue; never reset into success.
  cfg.episode_length_s = 1000.0
  cfg.commands["posture"].resampling_time_range = (1000.0, 1000.0)
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    cls = load_runner_cls(task)
    assert cls is not None
    runner = cls(RslRlVecEnvWrapper(env), asdict(load_rl_cfg(task)), None, "cuda:0")
    runner.load(
      str(args.checkpoint),
      load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
      map_location="cuda:0",
    )
    policy = runner.get_inference_policy("cuda:0")
    robot = env.scene["robot"]
    feet = robot.find_bodies(
      ("right_ankle_link", "left_ankle_link"), preserve_order=True
    )[0]
    data, results = {}, []
    for start, goals, name in [
      (0, [0], "stand_hold"),
      (0, [1], "down"),
      (1, [0], "up"),
      (1, [1], "crawl_hold"),
      (1, [0, 1, 0, 1], "continuous_cycle"),
    ]:
      torch.manual_seed(args.seed)
      env.event_manager.get_term_cfg("reset_endpoints").params["fixed_state"] = start
      env.reset()
      # Small independent initial velocity perturbations; no teleports thereafter.
      velocity = torch.zeros(args.num_envs, 6, device=env.device)
      velocity[:, :3] = (torch.rand(args.num_envs, 3, device=env.device) - 0.5) * 0.02
      velocity[:, 3:] = (torch.rand(args.num_envs, 3, device=env.device) - 0.5) * 0.04
      robot.write_root_link_velocity_to_sim(velocity)
      term = env.command_manager.get_term("posture")
      assert isinstance(term, mdp.PostureCommand)
      set_posture_goal(term, goals[0])
      obs = env.observation_manager.compute(update_history=True)
      records = {
        key: []
        for key in (
          "qpos",
          "command",
          "orientation_error",
          "height_error",
          "pose_rmse",
          "speed",
          "angular_speed",
          "contact",
          "force",
          "sole_tilt_deg",
          "unsafe",
          "success_gate",
          "actuator_force",
          "ankle_near_limit",
          "closure_mm",
        )
      }
      for goal in goals:
        set_posture_goal(term, goal)
        obs = env.observation_manager.compute(update_history=True)
        for _ in range(round(args.seconds / env.step_dt)):
          with torch.inference_mode():
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            if terminated.any() or truncated.any():
              raise RuntimeError("Unexpected auto-reset during continuous evaluation")
            pose, orientation, height = mdp.target_errors(env)
            angular_error = 2 * torch.asin((orientation.sqrt() / 2).clamp(0, 1))
            speed = robot.data.root_link_lin_vel_w.norm(dim=-1)
            angspeed = robot.data.root_link_ang_vel_w.norm(dim=-1)
            sensor = env.scene["feet_ground_contact"].data
            contact = (sensor.found > 0) & (sensor.force[..., 2].abs() > 1.0)
            tilt = torch.rad2deg(sole_geometry(robot.data.body_link_quat_w[:, feet])[0])
            rear = contact[:, :2].all(-1)
            expected = rear & (
              contact[:, 2:].all(-1) if goal else (~contact[:, 2:]).all(-1)
            )
            safe = ~mdp.unsafe_state(env)
            gate = (
              (angular_error < 0.2)
              & (height < 0.035)
              & (speed < 0.1)
              & (angspeed < 0.5)
              & expected
              & (tilt < 12.0).all(-1)
              & safe
            )
            ankle_ids = robot.find_joints(
              ("right_ankle_joint", "left_ankle_joint"), preserve_order=True
            )[0]
            ankle = robot.data.joint_pos[:, ankle_ids]
            limits = robot.data.soft_joint_pos_limits[:, ankle_ids]
            near = torch.minimum(
              ankle - limits[..., 0], limits[..., 1] - ankle
            ) < np.deg2rad(3)
            closure = []
            for side in ("right", "left"):
              sites = robot.find_sites(
                (f"{side}_knee_mid_closure", f"{side}_knee_mid"), preserve_order=True
              )[0]
              closure.append(
                (
                  robot.data.site_pos_w[:, sites[0]]
                  - robot.data.site_pos_w[:, sites[1]]
                ).norm(dim=-1)
                * 1000
              )
            values = {
              "qpos": env.sim.data.qpos,
              "command": term.value,
              "orientation_error": angular_error,
              "height_error": height,
              "pose_rmse": pose.sqrt(),
              "speed": speed,
              "angular_speed": angspeed,
              "contact": contact,
              "force": sensor.force,
              "sole_tilt_deg": tilt,
              "unsafe": ~safe,
              "success_gate": gate,
              "actuator_force": robot.data.actuator_force,
              "ankle_near_limit": near,
              "closure_mm": torch.stack(closure, -1),
            }
            for key, value in values.items():
              records[key].append(value.cpu().numpy().copy())
      samples = {key: np.asarray(value) for key, value in records.items()}
      data.update({f"{name}/{key}": value for key, value in samples.items()})
      segment_steps = round(args.seconds / env.step_dt)
      segments = []
      for i, goal in enumerate(goals):
        end = (i + 1) * segment_steps
        settle_steps = round(args.settle_seconds / env.step_dt)
        segment_start = i * segment_steps
        stable = samples["success_gate"][segment_start:end]
        success = stable[-settle_steps:].all(0) & ~samples["unsafe"][:end].any(0)
        time_to_stable = []
        for env_id in range(args.num_envs):
          first = None
          for offset in range(settle_steps, segment_steps + 1):
            if stable[offset - settle_steps : offset, env_id].all():
              first = offset * env.step_dt
              break
          time_to_stable.append(first)
        segments.append(
          {
            "goal": goal,
            "successes": int(success.sum()),
            "trials": args.num_envs,
            "success_rate": float(success.mean()),
            "settle_seconds": args.settle_seconds,
            "time_to_stable_s": time_to_stable,
            "orientation_error_deg_final_mean": float(
              np.rad2deg(samples["orientation_error"][end - 25 : end]).mean()
            ),
            "height_error_mm_final_mean": float(
              samples["height_error"][end - 25 : end].mean() * 1000
            ),
            "unsafe_envs_through_segment": int(samples["unsafe"][:end].any(0).sum()),
          }
        )
      results.append(
        {
          "case": name,
          "start": start,
          "segments": segments,
          "max_abs_actuator_force_nm": float(np.abs(samples["actuator_force"]).max()),
          "max_closure_mm": float(samples["closure_mm"].max()),
          "ankle_near_limit_fraction": float(samples["ankle_near_limit"].mean()),
        }
      )
      print(results[-1], flush=True)
    np.savez_compressed(args.out / "rollouts.npz", **data)
    (args.out / "summary.json").write_text(
      json.dumps(
        {
          "checkpoint": str(args.checkpoint.resolve()),
          "seed": args.seed,
          "num_envs": args.num_envs,
          "segment_seconds": args.seconds,
          "settle_seconds": args.settle_seconds,
          "dt": env.step_dt,
          "no_auto_reset": True,
          "initial_linear_velocity_range": [-0.01, 0.01],
          "initial_angular_velocity_range": [-0.02, 0.02],
          "cases": results,
        },
        indent=2,
      )
      + "\n"
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()
