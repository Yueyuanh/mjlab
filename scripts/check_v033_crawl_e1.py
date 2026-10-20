"""Small E1 physics/interface diagnostic; never substitutes for policy evaluation."""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl_e1.env_cfgs import (
  ARM_WINDOW_JOINTS,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  result_path = args.out / "physics_check.json"
  if result_path.exists():
    raise FileExistsError(result_path)
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E1", play=True)
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
  command.ranges.lin_vel_x = (0.0, 0.0)
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (0.0, 0.0)
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    assert isinstance(action, JointPositionAction)
    assert env.action_manager.total_action_dim == 18
    ids, _ = robot.find_joints(ARM_WINDOW_JOINTS, preserve_order=True)
    term = env.reward_manager.get_term_cfg("arm_posture_window")
    assert term.weight == -1.0
    costs = []
    contacts = []
    falls = 0
    peak = torch.zeros(4, device=env.device)
    for _ in range(250):
      obs, reward, terminated, _, _ = env.step(torch.zeros(4, 18, device=env.device))
      for value in obs.values():
        assert isinstance(value, torch.Tensor)
        assert torch.isfinite(value).all()
      assert torch.isfinite(reward).all()
      falls += int(terminated.sum())
      deviation = robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
      peak = torch.maximum(peak, deviation.abs().amax(0))
      costs.append(term.func(env, **term.params).mean().item())
      contacts.append((env.scene["feet_ground_contact"].data.found > 0).float().mean(0))

    # Exercise all 18 output channels without stepping: targets must retain the
    # affine action mapping even beyond the new reward window.
    probe = torch.full((4, 18), 0.75, device=env.device)
    action.process_actions(probe)
    action.apply_actions()
    targets = robot.data.joint_pos_target[:, action.target_ids]
    torch.testing.assert_close(targets, action.offset + probe * action.scale)
    joint_ids = action.target_ids.tolist()
    names = action.target_names
    offsets = targets[0] - robot.data.default_joint_pos[0, joint_ids]
    assert all(
      offsets[names.index(name)].abs() > math.radians(5) for name in ARM_WINDOW_JOINTS
    )
    elbow_names = [name for name in names if "elbow" in name]
    assert len(elbow_names) == 2
    assert all(offsets[names.index(name)] != 0 for name in elbow_names)
    actor_obs, critic_obs = obs["actor"], obs["critic"]
    assert isinstance(actor_obs, torch.Tensor)
    assert isinstance(critic_obs, torch.Tensor)
    result = {
      "checked": datetime.now().astimezone().isoformat(),
      "task": "Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E1",
      "num_envs": 4,
      "seconds": 5,
      "seed": 42,
      "action_dim": 18,
      "actor_dim": actor_obs.shape[-1],
      "critic_dim": critic_obs.shape[-1],
      "target_names": names,
      "zero_action_falls": falls,
      "zero_action_contact_duty": torch.stack(contacts).mean(0).tolist(),
      "window_joint_names": ARM_WINDOW_JOINTS,
      "zero_action_peak_deviation_degrees": torch.rad2deg(peak).tolist(),
      "zero_action_mean_window_cost_unweighted": sum(costs) / len(costs),
      "probe_raw_action": 0.75,
      "probe_target_offsets_degrees": dict(
        zip(names, torch.rad2deg(offsets).tolist(), strict=True)
      ),
      "target_clipping_to_window": False,
      "elbow_policy_outputs_active": True,
      "note": "Zero-action stability and target mapping only, not learned gait. "
      "No external balance or frozen joints. Target probe did not step physics.",
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
