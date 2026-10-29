"""Register the V034 knee-led biped tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  v034_biped_knee_env_cfg,
  v034_biped_knee_retarget_env_cfg,
  v034_biped_knee_sep_retarget_env_cfg,
)
from .rl_cfg import (
  v034_biped_knee_ppo_cfg,
  v034_biped_knee_retarget_ppo_cfg,
  v034_biped_knee_sep_ppo_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V034-Biped-Knee",
  env_cfg=v034_biped_knee_env_cfg(),
  play_env_cfg=v034_biped_knee_env_cfg(play=True),
  rl_cfg=v034_biped_knee_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V034-Biped-Knee-Retarget",
  env_cfg=v034_biped_knee_retarget_env_cfg(),
  play_env_cfg=v034_biped_knee_retarget_env_cfg(play=True),
  rl_cfg=v034_biped_knee_retarget_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V034-Biped-Knee-Sep",
  env_cfg=v034_biped_knee_sep_retarget_env_cfg(),
  play_env_cfg=v034_biped_knee_sep_retarget_env_cfg(play=True),
  rl_cfg=v034_biped_knee_sep_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
