"""Register the V034 upright task based on the latest V033 E1 framework."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import v034_biped_e1_env_cfg
from .rl_cfg import v034_biped_e1_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V034-Biped-E1",
  env_cfg=v034_biped_e1_env_cfg(),
  play_env_cfg=v034_biped_e1_env_cfg(play=True),
  rl_cfg=v034_biped_e1_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
