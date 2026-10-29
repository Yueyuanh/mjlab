"""Regression tests for the V031 gait-symmetry experiment chain."""

import torch

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.config.pikachu_v031.env_cfgs import (
  pikachu_v031_b1_beta_bounded_env_cfg,
  pikachu_v031_e0_mirror_env_cfg,
  pikachu_v031_e1_clip_env_cfg,
  pikachu_v031_e3_yaw_huber_env_cfg,
  pikachu_v031_e4_curriculum_env_cfg,
  pikachu_v031_t6_flat_env_cfg,
)
from mjlab.tasks.velocity.config.pikachu_v031.rl_cfg import (
  pikachu_v031_b1_beta_bounded_ppo_runner_cfg,
  pikachu_v031_e0_mirror005_ppo_runner_cfg,
  pikachu_v031_e0_mirror010_ppo_runner_cfg,
  pikachu_v031_e1_clip_ppo_runner_cfg,
  pikachu_v031_e2_symmetry_ppo_runner_cfg,
)
from mjlab.tasks.velocity.config.pikachu_v031.symmetry import (
  _ACTION_PERM,
  _ACTION_SIGN,
  _mirror_group,
  _swap_signed,
  foot_contact_forces_heading_frame,
)
from mjlab.utils.noise import UniformNoiseCfg


def test_v031_actor_and_critic_mirror_are_involutions():
  actor = torch.randn(7, 50)
  critic = torch.randn(7, 62)
  torch.testing.assert_close(
    _mirror_group(_mirror_group(actor, "actor"), "actor"), actor
  )
  torch.testing.assert_close(
    _mirror_group(_mirror_group(critic, "critic"), "critic"), critic
  )


def test_v031_action_mirror_swaps_sides_with_measured_signs():
  action = torch.arange(10, dtype=torch.float32).reshape(1, 10)
  mirrored = _swap_signed(action, _ACTION_PERM, _ACTION_SIGN)
  expected = torch.tensor([[-5.0, -6.0, -7.0, -8.0, 9.0, 0.0, -1.0, -2.0, -3.0, 4.0]])
  torch.testing.assert_close(mirrored, expected)
  torch.testing.assert_close(_swap_signed(mirrored, _ACTION_PERM, _ACTION_SIGN), action)


def test_e1_has_redundant_raw_and_processed_action_bounds():
  env_cfg = pikachu_v031_e1_clip_env_cfg()
  action = env_cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  assert action.clip is not None
  assert action.clip[r".*_hip_roll_joint"] == (-0.20, 0.20)
  assert action.clip[r".*_hip_yaw_joint"] == (-0.20, 0.20)
  assert pikachu_v031_e1_clip_ppo_runner_cfg().clip_actions == 1.0


def test_e0_mirror_only_adds_low_weight_symmetry_to_e0():
  baseline_env = pikachu_v031_t6_flat_env_cfg()
  mirror_env = pikachu_v031_e0_mirror_env_cfg()
  baseline_action = baseline_env.actions["joint_pos"]
  mirror_action = mirror_env.actions["joint_pos"]
  assert isinstance(baseline_action, JointPositionActionCfg)
  assert isinstance(mirror_action, JointPositionActionCfg)
  assert mirror_action.clip == baseline_action.clip
  assert (
    mirror_env.observations["critic"].terms["foot_contact_forces"].func
    is foot_contact_forces_heading_frame
  )

  runner005 = pikachu_v031_e0_mirror005_ppo_runner_cfg()
  runner010 = pikachu_v031_e0_mirror010_ppo_runner_cfg()
  assert runner005.clip_actions is None
  assert runner005.max_iterations == 50_000
  assert runner005.save_interval == 500
  assert runner005.algorithm.symmetry_cfg is not None
  assert runner005.algorithm.symmetry_cfg["mirror_loss_coeff"] == 0.05
  assert runner010.algorithm.symmetry_cfg is not None
  assert runner010.algorithm.symmetry_cfg["mirror_loss_coeff"] == 0.10
  assert runner005.wandb_project != runner010.wandb_project


def test_e2_enables_data_augmentation_and_mirror_loss():
  symmetry = pikachu_v031_e2_symmetry_ppo_runner_cfg().algorithm.symmetry_cfg
  assert symmetry is not None
  assert symmetry["use_data_augmentation"] is True
  assert symmetry["use_mirror_loss"] is True
  assert symmetry["mirror_loss_coeff"] == 0.30


def test_b1_uses_naturally_bounded_beta_with_redundant_safety_bounds():
  runner = pikachu_v031_b1_beta_bounded_ppo_runner_cfg()
  assert runner.actor.distribution_cfg == {
    "class_name": "BetaDistribution",
    "action_range": (-1.0, 1.0),
  }
  assert runner.clip_actions == 1.0
  assert runner.max_iterations == 50_000
  assert runner.save_interval == 500
  assert runner.algorithm.entropy_coef == 0.01
  assert runner.algorithm.symmetry_cfg is not None
  assert runner.algorithm.symmetry_cfg["mirror_loss_coeff"] == 0.05

  action = pikachu_v031_b1_beta_bounded_env_cfg().actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  assert action.clip is not None
  assert action.clip[r".*_hip_pitch_joint"] == (-0.50, 0.50)
  assert action.clip[r".*_hip_roll_joint"] == (-0.20, 0.20)
  assert action.clip[r".*_hip_yaw_joint"] == (-0.20, 0.20)
  assert action.clip[r".*_knee_up_joint"] == (-0.40, 0.40)
  assert action.clip[r".*_ankle_joint"] == (-0.35, 0.35)


def test_e3_and_e4_only_add_the_declared_factors():
  e3 = pikachu_v031_e3_yaw_huber_env_cfg()
  assert e3.rewards["angular_velocity_huber"].weight == -0.30
  assert e3.rewards["angular_velocity_huber"].params["delta"] == 0.25

  e4 = pikachu_v031_e4_curriculum_env_cfg()
  assert "v031_observation_noise" in e4.curriculum
  assert "v031_push" in e4.curriculum
  noise = e4.observations["actor"].terms["base_lin_vel"].noise
  assert isinstance(noise, UniformNoiseCfg)
  assert noise.n_min == -0.1
  assert noise.n_max == 0.1
