"""Isolation and reward tests for the V034 gait-quality task."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.pikachu_v034_biped_gait import mdp

TASK = "Mjlab-Velocity-Flat-Pikachu-V034-Biped-Gait"


def test_gait_task_isolated_configuration():
  cfg = load_env_cfg(TASK)
  base = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V034-Biped-Retarget")
  assert set(cfg.rewards) == set(base.rewards) | {
    "straight_base_roll_rate",
    "swing_knee_separation",
  }
  assert cfg.rewards["foot_clearance"].weight == -4.0
  assert base.rewards["foot_clearance"].weight == -4.0
  command = cfg.commands["twist"]
  assert command.rel_forward_envs == pytest.approx(0.40)
  assert command.rel_backward_envs == pytest.approx(0.20)
  agent = load_rl_cfg(TASK)
  assert agent.algorithm.symmetry_cfg is not None
  assert agent.algorithm.symmetry_cfg["use_data_augmentation"] is False
  assert agent.algorithm.symmetry_cfg["mirror_loss_coeff"] == pytest.approx(0.02)
  assert agent.experiment_name == "pikachu_v034/biped_gait"


def test_swing_knee_separation_rejects_crouch_and_accepts_phase_flexion():
  sensor = Mock(spec=ContactSensor)
  # right stance / left swing, then left stance / right swing
  sensor.data = SimpleNamespace(found=torch.tensor([[1, 0], [0, 1]]))
  command = torch.tensor([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
  robot = SimpleNamespace(
    data=SimpleNamespace(
      # Equal 0.2 rad crouch fails; 0.12 rad swing over 0.02 stance passes.
      joint_pos=torch.tensor([[0.2, 0.2], [0.12, 0.02]])
    )
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={"robot": robot, "feet": sensor},
      command_manager=SimpleNamespace(get_command=lambda _: command),
      extras={"log": {}},
    ),
  )
  cost = mdp.swing_knee_separation(
    env,
    "feet",
    SceneEntityCfg("robot"),
    margin=0.07,
    std=0.07,
  )
  torch.testing.assert_close(cost, torch.tensor([1.0, 0.0]))


def test_roll_rate_has_deadzone_and_straight_command_gate():
  command = torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.2, 0.0]])
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_ang_vel_b=torch.tensor([[1.2, 0.0, 0.0], [1.2, 0.0, 0.0]])
    )
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={"robot": robot},
      command_manager=SimpleNamespace(get_command=lambda _: command),
    ),
  )
  cost = mdp.straight_base_roll_rate(env, deadzone=0.45, std=0.75)
  torch.testing.assert_close(cost, torch.tensor([1.0, 0.0]))
