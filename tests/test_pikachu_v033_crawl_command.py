"""Debug arrows remain horizontal and agree with crawl command semantics."""

import math
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl.command import (
  CrawlVelocityCommand,
  CrawlVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul
from mjlab.viewer.debug_visualizer import DebugVisualizer


@pytest.mark.parametrize("pitch_degrees", (75, 95))
def test_crawl_arrows_follow_horizontal_heading_and_world_up(pitch_degrees: float):
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E1", play=True)
  command_cfg = cfg.commands["twist"]
  assert isinstance(command_cfg, CrawlVelocityCommandCfg)
  assert command_cfg.debug_vis
  yaw = torch.tensor([0.7])
  zero = torch.zeros(1)
  yaw_q = quat_from_euler_xyz(zero, zero, yaw)
  root_q = quat_mul(
    yaw_q, quat_from_euler_xyz(zero, zero + math.radians(pitch_degrees), zero)
  )
  data = SimpleNamespace(
    root_link_pos_w=torch.tensor([[0.0, 0.0, 0.23]]),
    root_link_quat_w=root_q,
    root_link_lin_vel_w=quat_apply(yaw_q, torch.tensor([[0.10, -0.02, 0.03]])),
    root_link_ang_vel_w=torch.tensor([[0.0, 0.0, 0.2]]),
  )
  robot = SimpleNamespace(data=data)
  term = cast(
    CrawlVelocityCommand,
    SimpleNamespace(
      num_envs=1,
      robot=robot,
      _env=SimpleNamespace(scene={"robot": robot}),
      cfg=command_cfg,
      command=torch.tensor([[0.15, 0.04, 0.3]]),
    ),
  )
  visualizer = Mock(spec=DebugVisualizer)
  visualizer.get_env_indices.return_value = [0]
  CrawlVelocityCommand._debug_vis_impl(term, visualizer)
  calls = visualizer.add_arrow.call_args_list
  assert len(calls) == 4
  rotation = np.array(
    [[math.cos(0.7), -math.sin(0.7), 0], [math.sin(0.7), math.cos(0.7), 0], [0, 0, 1]]
  )
  expected = (
    rotation @ [0.15, 0.04, 0],
    np.array([0, 0, 0.3]),
    rotation @ [0.10, -0.02, 0],
    np.array([0, 0, 0.2]),
  )
  for call, direction in zip(calls, expected, strict=True):
    start, end = call.args
    np.testing.assert_allclose(start, [0, 0, 0.23 + command_cfg.viz.z_offset])
    np.testing.assert_allclose(
      end - start, direction * command_cfg.viz.scale, atol=1e-7
    )
