"""V034 crawl with mirrored left-leg limits and stronger forward progress."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.pikachu_v034.robot import get_crawl_robot_cfg
from mjlab.tasks.velocity.config.pikachu_v034_crawl_e3.env_cfgs import (
  v034_crawl_e3_env_cfg,
)


def v034_crawl_knee_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = v034_crawl_e3_env_cfg(play=play)
  cfg.scene.entities["robot"] = get_crawl_robot_cfg(symmetric=True)
  cfg.scene.num_envs = 1 if play else 1024
  # The zero-shot V033 crawl policy stands still on V034: the command error is
  # real but the support/slip terms are already satisfied by standing.  Raise
  # the progress terms enough that standing is not a local optimum.
  cfg.rewards["forward_velocity_shortfall"].weight = -0.60
  cfg.rewards["linear_velocity_huber"].weight = -1.0
  cfg.rewards["directional_progress"].weight = 3.0
  cfg.rewards["gait_contacts_smooth_120"].weight = 1.8
  cfg.rewards["support_slip"].weight = -20.0
  return cfg
