"""Register the V034 biped gait-quality refinement task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import v034_biped_gait_env_cfg
from .rl_cfg import v034_biped_gait_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V034-Biped-Gait",
  env_cfg=v034_biped_gait_env_cfg(),
  play_env_cfg=v034_biped_gait_env_cfg(play=True),
  rl_cfg=v034_biped_gait_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
