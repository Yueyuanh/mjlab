"""Compact crawl posture with full joint control and isolated training logs."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import crawl_e2_env_cfg
from .rl_cfg import crawl_e2_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E2",
  env_cfg=crawl_e2_env_cfg(),
  play_env_cfg=crawl_e2_env_cfg(play=True),
  rl_cfg=crawl_e2_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
