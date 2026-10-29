"""Sole geometry, swing freedom and missing-contact anti-evasion."""

import math
from types import SimpleNamespace

import torch

from mjlab.tasks.velocity.config.pikachu_v033_crawl_e2.mdp import (
  sole_geometry,
  sole_stance_weight,
  sole_support_cost,
)


def test_true_sole_normal_and_endpoint_geometry():
  angle = torch.tensor([0.0, math.pi / 6, -math.pi / 6])
  zero = torch.zeros_like(angle)
  q = torch.stack((torch.cos(angle / 2), zero, torch.sin(angle / 2), zero), -1)
  tilt, gap = sole_geometry(q)
  torch.testing.assert_close(tilt, angle.abs())
  torch.testing.assert_close(gap, 0.08 * angle.sin().abs())


def test_expected_stance_does_not_depend_on_measured_contact():
  env = SimpleNamespace(
    episode_length_buf=torch.tensor([0.0, 40.0, 0.0]),
    step_dt=0.02,
    command_manager=SimpleNamespace(
      get_command=lambda _: torch.tensor(
        [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]
      )
    ),
  )
  weight = sole_stance_weight(env)
  torch.testing.assert_close(weight, torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))


def test_real_sustained_contact_is_constrained_even_in_planned_swing():
  angle = math.radians(25)
  quat = torch.tensor(
    [[[math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0], [1.0, 0.0, 0.0, 0.0]]]
  )
  sensor = SimpleNamespace(
    primary_names=["robot/right_ankle_link", "robot/left_ankle_link"],
    data=SimpleNamespace(
      current_contact_time=torch.tensor([[0.2, 0.2]]), found=torch.ones(1, 2)
    ),
  )
  env = SimpleNamespace(
    episode_length_buf=torch.tensor([0.0]),
    step_dt=0.02,
    scene={
      "robot": SimpleNamespace(data=SimpleNamespace(body_link_quat_w=quat)),
      "feet_ground_contact": sensor,
    },
    command_manager=SimpleNamespace(
      get_command=lambda _: torch.tensor([[0.1, 0.0, 0.0]])
    ),
  )
  cfg = SimpleNamespace(name="robot", body_ids=[0, 1])
  assert sole_support_cost(env, cfg).item() > 1
  sensor.data.found.zero_()
  assert sole_support_cost(env, cfg).item() == 0
