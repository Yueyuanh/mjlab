"""Register V034 walking transfer without changing historical task interfaces."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import biped_transfer_env_cfg
from .rl_cfg import biped_transfer_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V034-Biped-Transfer",
  env_cfg=biped_transfer_env_cfg(),
  play_env_cfg=biped_transfer_env_cfg(play=True),
  rl_cfg=biped_transfer_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

for retarget, label in ((False, "Transfer"), (True, "Retarget")):
  register_mjlab_task(
    task_id=f"Mjlab-Velocity-Flat-Pikachu-V034-Biped-{label}-Load",
    env_cfg=biped_transfer_env_cfg(retarget_feet=retarget, gain_scale=1.9),
    play_env_cfg=biped_transfer_env_cfg(
      play=True, retarget_feet=retarget, gain_scale=1.9
    ),
    rl_cfg=biped_transfer_ppo_cfg(retarget_feet=retarget, load_adapted=True),
    runner_cls=VelocityOnPolicyRunner,
  )

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V034-Biped-Retarget",
  env_cfg=biped_transfer_env_cfg(retarget_feet=True),
  play_env_cfg=biped_transfer_env_cfg(play=True, retarget_feet=True),
  rl_cfg=biped_transfer_ppo_cfg(retarget_feet=True),
  runner_cls=VelocityOnPolicyRunner,
)
