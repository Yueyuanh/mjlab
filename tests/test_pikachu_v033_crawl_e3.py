"""E3 focuses forward corrections without deleting the E2 task."""

from types import SimpleNamespace
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl_e3 import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def test_e3_samples_both_longitudinal_directions_and_has_isolated_tracking():
  e2 = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E2")
  e3 = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E3")
  command = cast(UniformVelocityCommandCfg, e3.commands["twist"])
  assert command.rel_forward_envs == 0.40
  assert command.rel_backward_envs == 0.30
  assert command.ranges.lin_vel_x == (-0.15, 0.20)
  assert "forward_sole_normal" not in e2.rewards
  assert "forward_sole_normal" in e3.rewards
  assert e3.rewards["forward_velocity_shortfall"].weight == -0.25
  assert e3.rewards["action_acc_l2"].weight == 0.0
  assert e3.rewards["gait_contacts"].weight == 0.0
  assert e3.rewards["forward_gait_mismatch"].weight == 0.0
  assert e3.rewards["forward_gait_mismatch_smooth_080"].weight == 0.0
  assert e3.rewards["forward_gait_mismatch_smooth_120"].weight == -0.40
  assert e3.rewards["gait_contacts_smooth_080"].weight == 0.0
  assert e3.rewards["gait_contacts_smooth_120"].weight == 1.50
  agent = load_rl_cfg("Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E3")
  assert agent.wandb_project == "pikachu-v033-velocity-crawl-e3"
  assert "backward-retention" in agent.wandb_tags


def test_forward_weight_preserves_a_smaller_backward_rehearsal_signal():
  command = torch.tensor([[0.15, 0.0, 0.0], [-0.15, 0.0, 0.0], [0.0, 0.0, 0.0]])
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _: command)),
  )
  torch.testing.assert_close(mdp.forward_weight(env), torch.tensor([1.0, 0.20, 0.0]))


def test_smooth_swing_weight_fades_the_hard_phase_boundaries():
  command = torch.tensor([[0.15, 0.0, 0.0]] * 4)
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      command_manager=SimpleNamespace(get_command=lambda _: command),
      episode_length_buf=torch.tensor([0, 2, 4, 8]),
      step_dt=0.02,
    ),
  )
  weight = mdp.smooth_swing_weight(env, transition_seconds=0.08)[:, 0]
  torch.testing.assert_close(weight, torch.tensor([0.0, 0.5, 1.0, 1.0]))
