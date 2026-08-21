"""Tests for the Pikachu walk-base (leg-only + arm-swing) ablation task."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.arm_sine import ArmSineAction
from mjlab.tasks.tracking.config.pikachu_walk_base.env_cfgs import (
  pikachu_walk_base_env_cfg,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _build_env(play: bool) -> tuple[ManagerBasedRlEnv, ArmSineAction]:
  cfg = pikachu_walk_base_env_cfg(play=play)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
  term = env.action_manager.get_term("arm_sine")
  assert isinstance(term, ArmSineAction)
  return env, term


def test_walk_base_action_dim_is_10_legs_only() -> None:
  env, _term = _build_env(play=False)
  try:
    assert env.action_space.shape[1] == 10
  finally:
    env.close()


def test_walk_base_observer_has_no_arm_joints() -> None:
  env, _term = _build_env(play=False)
  try:
    cfgs = env.observation_manager._group_obs_term_cfgs["actor"]
    names = env.observation_manager._group_obs_term_names["actor"]
    terms = dict(zip(names, cfgs, strict=False))
    for term_name in ("joint_pos", "joint_vel"):
      joint_ids = terms[term_name].params["asset_cfg"].joint_ids
      assert len(joint_ids) == 10
      assert all(0 <= int(i) < 10 for i in joint_ids)
  finally:
    env.close()


def test_walk_base_arm_sine_is_zero_dim() -> None:
  env, term = _build_env(play=False)
  try:
    assert term.action_dim == 0
  finally:
    env.close()


def test_walk_base_arm_curriculum_ramp() -> None:
  env, term = _build_env(play=False)
  try:
    off_steps, ramp_steps = term.cfg.curriculum_steps
    sat_steps = off_steps + ramp_steps

    def factor(steps: int) -> float:
      env.common_step_counter = steps
      return term._curriculum_factor()

    assert factor(0) == 0.0
    assert 0.3 < factor(off_steps + ramp_steps // 2) < 0.7
    assert factor(sat_steps) == 1.0

    env.common_step_counter = sat_steps
    assert abs(term._effective_amp_pitch() - term.cfg.default_amp_pitch) < 1e-6
    assert abs(term._effective_amp_roll() - term.cfg.default_amp_roll) < 1e-6
  finally:
    env.close()


def test_walk_base_play_disables_curriculum() -> None:
  env, term = _build_env(play=True)
  try:
    assert not term.cfg.use_curriculum
    env.common_step_counter = 999_999
    assert term._curriculum_factor() == 0.0
  finally:
    env.close()
