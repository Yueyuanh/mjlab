"""注册 V035 直立行走 v3（偏航 Huber 微调）任务。"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import v035_biped_walk_v3_env_cfg
from .rl_cfg import v035_biped_walk_v3_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V035-Biped-Walk-V3",
  env_cfg=v035_biped_walk_v3_env_cfg(),
  play_env_cfg=v035_biped_walk_v3_env_cfg(play=True),
  rl_cfg=v035_biped_walk_v3_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
