"""Crawl velocity arrows use the same horizontal frame as the policy task."""

from typing import TYPE_CHECKING

import numpy as np

from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from .mdp import crawl_heading, heading_ang_vel, heading_lin_vel

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class CrawlVelocityCommand(UniformVelocityCommand):
  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    indices = visualizer.get_env_indices(self.num_envs)
    commands = self.command.cpu().numpy()
    positions = self.robot.data.root_link_pos_w.cpu().numpy()
    rotation = (
      matrix_from_quat(crawl_heading(self.robot.data.root_link_quat_w)).cpu().numpy()
    )
    linear = heading_lin_vel(self._env).cpu().numpy()
    angular = heading_ang_vel(self._env).cpu().numpy()
    for idx in indices:
      origin = positions[idx] + np.array([0.0, 0.0, self.cfg.viz.z_offset])
      vectors = (
        (rotation[idx] @ np.array([*commands[idx, :2], 0.0]), (0.2, 0.2, 0.6, 0.6)),
        (np.array([0.0, 0.0, commands[idx, 2]]), (0.2, 0.6, 0.2, 0.6)),
        (rotation[idx] @ np.array([*linear[idx, :2], 0.0]), (0.0, 0.6, 1.0, 0.7)),
        (np.array([0.0, 0.0, angular[idx, 2]]), (0.0, 1.0, 0.4, 0.7)),
      )
      for vector, color in vectors:
        visualizer.add_arrow(
          origin, origin + vector * self.cfg.viz.scale, color=color, width=0.015
        )


class CrawlVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env: "ManagerBasedRlEnv") -> CrawlVelocityCommand:
    return CrawlVelocityCommand(self, env)
