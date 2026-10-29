"""Register the isolated V033-coordinate V034 crawl transfer task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import v034_crawl_transfer_env_cfg
from .rl_cfg import v034_crawl_transfer_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V034-Crawl-Retarget",
  env_cfg=v034_crawl_transfer_env_cfg(),
  play_env_cfg=v034_crawl_transfer_env_cfg(play=True),
  rl_cfg=v034_crawl_transfer_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
