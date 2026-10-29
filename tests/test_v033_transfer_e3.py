"""Slow E3 posture commands expose a continuous target and long hold time."""

from types import SimpleNamespace
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.pikachu_v033_transfer_e3 import mdp


def test_e3_transition_configuration_is_gentler_and_isolated():
  old = load_env_cfg("Mjlab-Transfer-Flat-Pikachu-V033")
  new = load_env_cfg("Mjlab-Transfer-Flat-Pikachu-V033-E3")
  command = new.commands["posture"]
  assert isinstance(command, mdp.SmoothPostureCommandCfg)
  assert command.up_duration_s == 5.0
  assert command.down_duration_s == 3.5
  assert command.resampling_time_range == (14.0, 18.0)
  assert new.episode_length_s == 24.0
  assert "action_acc" not in old.rewards
  assert "action_acc" in new.rewards
  assert new.terminations["unsafe"].func is mdp.unrecoverable_state
  assert new.terminations["unsafe"].params["torso_grace_s"] == 2.5
  agent = load_rl_cfg("Mjlab-Transfer-Flat-Pikachu-V033-E3")
  assert agent.wandb_project == "pikachu-v033-transfer-e3-gentle"


def test_smooth_command_has_zero_endpoint_velocity():
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      num_envs=1,
      device="cpu",
      step_dt=0.02,
      episode_length_buf=torch.ones(1, dtype=torch.long),
    ),
  )
  cfg = mdp.SmoothPostureCommandCfg(
    resampling_time_range=(14.0, 18.0), down_duration_s=4.0, up_duration_s=5.0
  )
  term = mdp.SmoothPostureCommand(cfg, env)
  term.start.fill_(0.0)
  term.goal.fill_(1.0)
  term.duration.fill_(4.0)
  term.elapsed.fill_(0.0)
  term._update_command(None)
  first = term.value.item()
  assert 0.0 < first < 0.001
  term.elapsed.fill_(3.98)
  term._update_command(None)
  assert 0.999 < term.value.item() <= 1.0


def test_reset_uses_endpoint_even_before_episode_length_is_cleared():
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      num_envs=1,
      device="cpu",
      step_dt=0.02,
      episode_length_buf=torch.tensor([137]),
      _transfer_reset_state=torch.ones(1),
    ),
  )
  term = mdp.SmoothPostureCommand(mdp.SmoothPostureCommandCfg(), env)
  term.value.zero_()
  term.command_counter.zero_()
  term._resample_command(torch.tensor([0]))
  assert term.value.item() == 1.0
  assert term.start.item() == 1.0
