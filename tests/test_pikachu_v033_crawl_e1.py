"""E1 deadband behavior and full-action task isolation."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl.robot import JOINT_POSE
from mjlab.tasks.velocity.config.pikachu_v033_crawl_e1.env_cfgs import (
  ARM_WINDOW_JOINTS,
)
from mjlab.tasks.velocity.config.pikachu_v033_crawl_e1.mdp import arm_posture_window

TASK = "Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E1"


@pytest.mark.parametrize("joint", range(4))
@pytest.mark.parametrize("sign", (-1, 1))
def test_window_boundary_and_excess_on_each_mirrored_joint(joint: int, sign: int):
  # Extra columns represent unselected joints, including elbows.
  ids = [1, 3, 4, 6]
  default = torch.zeros(6, 8, dtype=torch.float64)
  default[:, ids] = default.new_tensor([JOINT_POSE[name] for name in ARM_WINDOW_JOINTS])
  positions = default.clone()
  positions[:, ids[joint]] += sign * torch.deg2rad(
    torch.tensor([0, 4.9, 5, 10, 15, 25], dtype=torch.float64)
  )
  positions[:, 7] += 1.0  # Elbow changes must not enter this four-joint cost.
  original = positions.clone()
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={
        "robot": SimpleNamespace(
          data=SimpleNamespace(joint_pos=positions, default_joint_pos=default)
        )
      }
    ),
  )
  cost = arm_posture_window(env, SceneEntityCfg("robot", joint_ids=ids))
  torch.testing.assert_close(
    cost, cost.new_tensor([0, 0, 0, 0.25, 1, 4]), atol=1e-26, rtol=1e-12
  )
  torch.testing.assert_close(positions, original, atol=0, rtol=0)


def test_e1_preserves_full_interface_and_isolates_reward_and_logging():
  crawl = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V033-Crawl")
  e1 = load_env_cfg(TASK)
  e1_play = load_env_cfg(TASK, play=True)
  biped = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V033-E0")
  assert "arm_posture_window" not in crawl.rewards
  assert "arm_posture_window" not in biped.rewards
  term = e1.rewards["arm_posture_window"]
  assert term.params["asset_cfg"].joint_names == ARM_WINDOW_JOINTS
  assert term.params["tolerance_degrees"] == 5.0
  assert term.weight == e1_play.rewards["arm_posture_window"].weight == -1.0
  for cfg in (e1, e1_play):
    robot = Entity(cfg.scene.entities["robot"])
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    _, controlled = robot.find_joints_by_actuator_names(action.actuator_names)
    assert len(controlled) == 18
    assert sum("elbow" in name for name in controlled) == 2
    assert action.clip is None
    # E1 uses exactly the same physical robot, not additional joint limits.
    assert cfg.scene.entities["robot"] == crawl.scene.entities["robot"]
  e1_rl = load_rl_cfg(TASK)
  crawl_rl = load_rl_cfg("Mjlab-Velocity-Flat-Pikachu-V033-Crawl")
  assert isinstance(e1_rl, RslRlOnPolicyRunnerCfg)
  assert isinstance(crawl_rl, RslRlOnPolicyRunnerCfg)
  assert e1_rl.actor == crawl_rl.actor
  assert e1_rl.critic == crawl_rl.critic
  assert e1_rl.algorithm == crawl_rl.algorithm
  assert e1_rl.clip_actions is None
  assert e1_rl.experiment_name == "pikahcu_v033/crawl_e1"
  assert e1_rl.wandb_project != crawl_rl.wandb_project
  assert e1_rl.wandb_tags != crawl_rl.wandb_tags
  term.weight = -7
  assert load_env_cfg(TASK).rewards["arm_posture_window"].weight == -1.0
