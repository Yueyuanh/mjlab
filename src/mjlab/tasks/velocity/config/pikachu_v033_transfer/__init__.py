"""Register a separate posture-conditioned no-reference transition task."""

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import transfer_env_cfg
from .rl_cfg import transfer_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Transfer-Flat-Pikachu-V033",
  env_cfg=transfer_env_cfg(),
  play_env_cfg=transfer_env_cfg(play=True),
  rl_cfg=transfer_ppo_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
