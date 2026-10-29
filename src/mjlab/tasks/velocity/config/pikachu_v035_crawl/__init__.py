"""注册 V035 四足 crawl 任务。"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import v035_crawl_env_cfg
from .rl_cfg import v035_crawl_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V035-Crawl",
  env_cfg=v035_crawl_env_cfg(),
  play_env_cfg=v035_crawl_env_cfg(play=True),
  rl_cfg=v035_crawl_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
