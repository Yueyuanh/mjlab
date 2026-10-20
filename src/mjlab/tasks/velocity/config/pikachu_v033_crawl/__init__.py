"""V033 crawl registration, isolated from the E0 biped task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import crawl_env_cfg
from .rl_cfg import crawl_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V033-Crawl",
  env_cfg=crawl_env_cfg(),
  play_env_cfg=crawl_env_cfg(play=True),
  rl_cfg=crawl_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
