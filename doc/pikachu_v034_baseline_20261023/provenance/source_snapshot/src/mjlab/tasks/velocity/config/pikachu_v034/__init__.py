"""Register the V034 biped and four-support crawl baselines."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import biped_env_cfg, crawl_env_cfg
from .rl_cfg import ppo_runner_cfg

for mode, suffix, factory in (
  ("biped", "Biped", biped_env_cfg),
  ("crawl", "Crawl", crawl_env_cfg),
):
  register_mjlab_task(
    task_id=f"Mjlab-Velocity-Flat-Pikachu-V034-{suffix}",
    env_cfg=factory(),
    play_env_cfg=factory(play=True),
    rl_cfg=ppo_runner_cfg(mode),
    runner_cls=VelocityOnPolicyRunner,
  )
