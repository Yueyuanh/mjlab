"""Register the slow-command E3 quadruped-to-biped transition task."""

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import transfer_e3_env_cfg
from .rl_cfg import transfer_e3_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Transfer-Flat-Pikachu-V033-E3",
  env_cfg=transfer_e3_env_cfg(),
  play_env_cfg=transfer_e3_env_cfg(play=True),
  rl_cfg=transfer_e3_ppo_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
