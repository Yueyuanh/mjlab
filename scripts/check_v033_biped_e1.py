"""Read-only E1 geometry probes and short physical standing/interface checks."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.sensor import ContactSensor
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.config.pikachu_v033_biped_e1.env_cfgs import (
  ARM_BOUNDS_DEGREES,
  ARM_JOINTS,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

TASK = "Mjlab-Velocity-Flat-Pikachu-V033-E1"


def geometry_probe(cfg) -> tuple[dict, dict[str, np.ndarray]]:
  model = Entity(cfg.scene.entities["robot"]).compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
  initial = data.qpos.copy()
  mujoco.mj_forward(model, data)
  assert data.ncon == 0
  rng = np.random.default_rng(42)
  indices = [model.joint(n).qposadr[0] for n in ARM_JOINTS]
  bounds = np.deg2rad(ARM_BOUNDS_DEGREES)
  hits = {"within_arm_windows": 0, "unrestricted_arms": 0}
  examples = {}
  distances = {}
  for mode, count in (("within_arm_windows", 256), ("unrestricted_arms", 2048)):
    for _ in range(count):
      data.qpos[:] = initial
      if mode == "within_arm_windows":
        data.qpos[indices] = rng.uniform(bounds[:, 0], bounds[:, 1])
      else:
        data.qpos[indices] = rng.uniform(-1.55, 1.55, len(indices))
      # Nominal leg pose: isolates the arm envelope, not a gait certificate.
      mujoco.mj_forward(model, data)
      if data.ncon:
        hits[mode] += 1
      for contact in data.contact:
        kind = "near" if 0 < contact.dist < 0.01 else "penetrating"
        if kind not in examples:
          examples[kind] = data.qpos.copy()
          distances[kind] = float(contact.dist)
  assert "near" in examples and "penetrating" in examples
  return {
    "mass_kg": float(model.body_mass.sum()),
    "initial_arm_leg_contact_count": 0,
    "random_pose_seed": 42,
    "random_pose_counts": {"within_arm_windows": 256, "unrestricted_arms": 2048},
    "poses_with_contacts_including_10mm_gap": hits,
    "example_signed_distances_m": distances,
    "scope": "Arm-only sampled poses with nominal legs; no guarantee for all walking configurations",
  }, examples


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  target = args.out / "physics_check.json"
  if target.exists():
    raise FileExistsError(target)
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 4
  cfg.seed = 42
  cfg.metrics = {}
  for name in ("foot_friction", "base_com", "encoder_bias"):
    cfg.events.pop(name, None)
  cfg.events["reset_base"].params["pose_range"] = {
    axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
  }
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.rel_standing_envs = 1.0
  geometry, examples = geometry_probe(cfg)
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    assert isinstance(action, JointPositionAction)
    assert env.action_manager.total_action_dim == 18
    ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)
    bounds = torch.deg2rad(torch.tensor(ARM_BOUNDS_DEGREES, device=env.device))
    feet, _ = robot.find_sites(("right_foot", "left_foot"), preserve_order=True)
    foot_start = robot.data.site_pos_w[:, feet].clone()
    root_start = robot.data.root_link_pos_w.clone()
    contacts, speeds, arm_angles, foot_speeds, drift, window_costs = (
      [],
      [],
      [],
      [],
      [],
      [],
    )
    falls = 0
    for step in range(250):
      obs, reward, terminated, _, _ = env.step(torch.zeros(4, 18, device=env.device))
      for value in obs.values():
        assert isinstance(value, torch.Tensor)
        assert torch.isfinite(value).all()
      assert torch.isfinite(reward).all()
      falls += int(terminated.sum())
      contacts.append((env.scene["feet_ground_contact"].data.found > 0).float().mean(0))
      arm_angles.append(robot.data.joint_pos[:, ids].clone())
      term = env.reward_manager.get_term_cfg("arm_position_window")
      window_costs.append(term.func(env, **term.params).mean().item())
      if step >= 50:
        speeds.append(robot.data.root_link_lin_vel_w.clone())
        foot_speeds.append(robot.data.site_lin_vel_w[:, feet].clone())
      drift.append((robot.data.root_link_pos_w[:, :2] - root_start[:, :2]).norm(dim=-1))
    angles = torch.stack(arm_angles)
    excess = (bounds[:, 0] - angles).clamp_min(0) + (angles - bounds[:, 1]).clamp_min(0)
    physical = {
      "num_envs": 4,
      "seconds": 5,
      "zero_action_falls": falls,
      "foot_contact_duty": torch.stack(contacts).mean(0).tolist(),
      "base_linear_speed_rms_after_1s_mps": torch.stack(speeds)
      .square()
      .sum(-1)
      .mean()
      .sqrt()
      .item(),
      "foot_linear_speed_rms_after_1s_mps": torch.stack(foot_speeds)
      .square()
      .sum(-1)
      .mean()
      .sqrt()
      .item(),
      "maximum_base_xy_drift_m": torch.stack(drift).max().item(),
      "final_foot_xy_drift_m": (
        robot.data.site_pos_w[:, feet, :2] - foot_start[:, :, :2]
      )
      .norm(dim=-1)
      .mean(0)
      .tolist(),
      "peak_arm_window_excess_degrees": torch.rad2deg(excess).amax(dim=(0, 1)).tolist(),
      "mean_arm_window_cost": sum(window_costs) / len(window_costs),
      "joint_names": ARM_JOINTS,
    }
    # Targets deliberately exceed soft windows: no hidden clipping or masking.
    probe = torch.full((4, 18), 0.75, device=env.device)
    action.process_actions(probe)
    action.apply_actions()
    targets = robot.data.joint_pos_target[:, action.target_ids]
    torch.testing.assert_close(targets, action.offset + probe * action.scale)
    contact_sensor = env.scene["arm_leg_contact"]
    assert isinstance(contact_sensor, ContactSensor)
    gpu_probes = {}
    # Native CPU examples must also appear in the real Warp contact sensor,
    # including the inactive positive-distance gap used by the reward.
    for kind, qpos in examples.items():
      joints = torch.tensor(qpos[7:], device=env.device, dtype=torch.float32).repeat(
        4, 1
      )
      robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
      env.sim.forward()
      env.scene.update(env.step_dt)
      data = contact_sensor.data
      assert data.found is not None and data.dist is not None
      assert (data.found > 0).any()
      term = env.reward_manager.get_term_cfg("arm_leg_clearance")
      cost = term.func(env, **term.params)
      assert cost.max() > 0
      found = data.found > 0
      gpu_probes[kind] = {
        "found": data.found[0].tolist(),
        "minimum_detected_distance_m": data.dist[found].min().item(),
        "clearance_cost": cost[0].item(),
      }
    actor_obs, critic_obs = obs["actor"], obs["critic"]
    assert isinstance(actor_obs, torch.Tensor) and isinstance(critic_obs, torch.Tensor)
    result = {
      "checked": datetime.now().astimezone().isoformat(),
      "task": TASK,
      "geometry": geometry,
      "zero_action_standing": physical,
      "gpu_mesh_contact_probes": gpu_probes,
      "action_dim": 18,
      "actor_dim": actor_obs.shape[-1],
      "critic_dim": critic_obs.shape[-1],
      "target_mapping": "Unchanged affine mapping on all 18 outputs, no window clipping",
      "note": "No balance/freeze assistance; no startup DR. Zero-action check is not a trained-policy result. Probe states only test sensors, not gait validity.",
    }
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
