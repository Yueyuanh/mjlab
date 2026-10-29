"""Posture command contract and mutually exclusive static endpoint rewards."""

import math
from types import SimpleNamespace

import torch

from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.config.pikachu_v033_transfer import mdp


def test_static_targets_require_the_commanded_endpoint():
  names = ["right_arm_roll_joint", "left_arm_roll_joint", "right_arm_yaw_joint"]
  stand, crawl = mdp.endpoint_joints(names, "cpu")
  robot = SimpleNamespace(
    find_joints_by_actuator_names=lambda _: (list(range(3)), names),
    data=SimpleNamespace(
      joint_pos=torch.stack((stand, crawl, crawl, stand)),
      projected_gravity_b=torch.tensor(
        [
          [0.0, 0.0, -1.0],
          [math.sin(mdp.PITCH), 0.0, -math.cos(mdp.PITCH)],
          [math.sin(mdp.PITCH), 0.0, -math.cos(mdp.PITCH)],
          [0.0, 0.0, -1.0],
        ]
      ),
      root_link_pos_w=torch.tensor(
        [
          [0.0, 0.0, mdp.STAND_HEIGHT],
          [0.0, 0.0, mdp.HEIGHT],
          [0.0, 0.0, mdp.HEIGHT],
          [0.0, 0.0, mdp.STAND_HEIGHT],
        ]
      ),
    ),
  )
  env = SimpleNamespace(
    device="cpu",
    scene={"robot": robot},
    command_manager=SimpleNamespace(
      get_command=lambda _: torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    ),
  )

  class Scene(dict):
    env_origins = torch.zeros(4, 3)

  env.scene = Scene(env.scene)
  score = mdp.goal_score(env)
  torch.testing.assert_close(score[:2], torch.ones(2))
  assert (score[2:] < 0.001).all()
  assert (mdp.goal_distance(env) >= 0).all()


def test_transition_has_one_separate_command_and_no_motion_reference():
  cfg = load_env_cfg("Mjlab-Transfer-Flat-Pikachu-V033")
  assert set(cfg.commands) == {"posture"}
  assert "posture" in cfg.observations["actor"].terms
  assert "crawl_phase" not in cfg.observations["actor"].terms
  assert set(cfg.events) == {"reset_endpoints"}
  assert set(cfg.terminations) == {"time_out", "unsafe"}
  assert not cfg.actions["joint_pos"].use_default_offset


def test_progress_does_not_pay_for_command_flips_resets_or_holding(monkeypatch):
  env = SimpleNamespace(
    episode_length_buf=torch.tensor([0]),
    step_dt=0.02,
    error=torch.tensor([1.0]),
    flag=torch.tensor([0.0]),
  )
  monkeypatch.setattr(
    mdp, "target_errors", lambda env: (env.error * 0, env.error, env.error * 0)
  )
  monkeypatch.setattr(mdp, "state", lambda env: env.flag)
  assert mdp.orientation_progress(env).item() == 0
  env.episode_length_buf.fill_(2)
  assert mdp.orientation_progress(env).item() == 0
  env.error.fill_(0.99)
  assert mdp.orientation_progress(env).item() > 0
  env.flag.fill_(1)
  env.error.fill_(0.5)
  assert mdp.orientation_progress(env).item() == 0
  env.episode_length_buf.fill_(0)
  env.error.fill_(0.2)
  assert mdp.orientation_progress(env).item() == 0
