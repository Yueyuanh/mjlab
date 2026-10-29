"""V034 tasks preserve their body while adopting the latest V033 framework."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.pikachu_v034_biped_e1 import mdp as biped_mdp
from mjlab.tasks.velocity.config.pikachu_v034_crawl_e3 import mdp as crawl_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

BIPED = "Mjlab-Velocity-Flat-Pikachu-V034-Biped-E1"
CRAWL = "Mjlab-Velocity-Flat-Pikachu-V034-Crawl-E3"


def test_v034_biped_uses_e1_rewards_without_unvalidated_symmetry():
  cfg = load_env_cfg(BIPED)
  model = Entity(cfg.scene.entities["robot"]).compile()
  assert model.body_mass.sum() == pytest.approx(8.890779, abs=1e-6)
  assert cfg.sim.mujoco.timestep == 0.0025
  assert cfg.decimation == 8
  assert cfg.scene.num_envs == 512
  assert cfg.rewards["base_height"].params["target_height"] == pytest.approx(
    0.237914, abs=1e-6
  )
  assert cfg.rewards["directional_progress"].weight == 2.0
  for term in (
    "arm_position_window",
    "standing_joint_velocity",
    "standing_base_motion",
    "standing_foot_motion",
    "standing_support",
  ):
    assert term in cfg.rewards
  agent = load_rl_cfg(BIPED)
  assert agent.actor.hidden_dims == (512, 256, 128)
  assert agent.algorithm.symmetry_cfg is None
  assert agent.wandb_project == "pikachu-v034-velocity-biped-e1"


@pytest.mark.parametrize("task", [BIPED, CRAWL])
def test_v034_policy_joint_order_matches_v033(task: str):
  reference = load_env_cfg(
    "Mjlab-Velocity-Flat-Pikachu-V033-E1"
    if task == BIPED
    else "Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E3"
  )
  cfg = load_env_cfg(task)
  actual_entity = Entity(cfg.scene.entities["robot"])
  reference_entity = Entity(reference.scene.entities["robot"])
  actual = cfg.actions["joint_pos"]
  expected = reference.actions["joint_pos"]
  assert isinstance(actual, JointPositionActionCfg)
  assert isinstance(expected, JointPositionActionCfg)
  _, actual_names = actual_entity.find_joints_by_actuator_names(
    actual.actuator_names, preserve_order=actual.preserve_order
  )
  _, expected_names = reference_entity.find_joints_by_actuator_names(
    expected.actuator_names, preserve_order=expected.preserve_order
  )
  assert actual_names == expected_names

  actual_obs = cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"]
  _, actual_obs_names = actual_entity.find_joints(
    actual_obs.joint_names, preserve_order=actual_obs.preserve_order
  )
  assert tuple(actual_obs_names) == reference_entity.joint_names


def test_v034_crawl_uses_selected_e3_smoothing_and_direction_buckets():
  cfg = load_env_cfg(CRAWL)
  command = cast(UniformVelocityCommandCfg, cfg.commands["twist"])
  assert command.rel_forward_envs == 0.40
  assert command.rel_backward_envs == 0.30
  assert command.ranges.lin_vel_x == (-0.15, 0.20)
  assert cfg.rewards["gait_contacts"].weight == 0.0
  assert cfg.rewards["gait_contacts_smooth_120"].weight == 1.50
  assert cfg.rewards["forward_gait_mismatch_smooth_120"].weight == -0.40
  assert cfg.rewards["forward_velocity_shortfall"].weight == -0.25
  assert cfg.rewards["directional_progress"].weight == 2.0
  assert cfg.rewards["action_acc_l2"].weight == 0.0
  agent = load_rl_cfg(CRAWL)
  assert agent.actor.hidden_dims == (512, 256, 128)
  assert agent.actor.distribution_cfg["std_range"] == (0.04, 0.25)
  assert agent.wandb_project == "pikachu-v034-velocity-crawl-e3"


def test_v034_smooth_phase_has_soft_edges():
  command = torch.tensor([[0.15, 0.0, 0.0]] * 4)
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      command_manager=SimpleNamespace(get_command=lambda _: command),
      episode_length_buf=torch.tensor([0, 2, 4, 8]),
      step_dt=0.02,
    ),
  )
  weight = crawl_mdp.smooth_swing_weight(env, transition_seconds=0.08)[:, 0]
  torch.testing.assert_close(weight, torch.tensor([0.0, 0.5, 1.0, 1.0]))


def test_v034_crawl_directional_progress_is_symmetric(monkeypatch):
  command = torch.tensor([[0.10, 0.0, 0.0], [-0.05, 0.0, 0.0], [0.0, 0.0, 0.0]])
  velocity = torch.tensor([[0.05, 0.0, 0.0], [-0.025, 0.0, 0.0], [0.5, 0.0, 0.0]])
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _: command)),
  )
  monkeypatch.setattr(crawl_mdp.crawl_mdp, "heading_lin_vel", lambda _: velocity)
  torch.testing.assert_close(
    crawl_mdp.directional_progress(env), torch.tensor([0.5, 0.5, 0.0])
  )


def test_v034_biped_directional_progress_is_symmetric():
  command = torch.tensor([[0.10, 0.0, 0.0], [-0.05, 0.0, 0.0], [0.0, 0.0, 0.0]])
  velocity = torch.tensor([[0.05, 0.0, 0.0], [-0.025, 0.0, 0.0], [0.5, 0.0, 0.0]])
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      command_manager=SimpleNamespace(get_command=lambda _: command),
      scene={
        "robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=velocity))
      },
    ),
  )
  torch.testing.assert_close(
    biped_mdp.directional_progress(env), torch.tensor([0.5, 0.5, 0.0])
  )
