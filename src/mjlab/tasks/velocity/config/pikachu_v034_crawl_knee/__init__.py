"""Register the V034 crawl task with mirrored left-leg limits."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import v034_crawl_knee_env_cfg
from .rl_cfg import v034_crawl_knee_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V034-Crawl-Knee",
  env_cfg=v034_crawl_knee_env_cfg(),
  play_env_cfg=v034_crawl_knee_env_cfg(play=True),
  rl_cfg=v034_crawl_knee_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
