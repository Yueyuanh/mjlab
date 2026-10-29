"""PPO configuration for V034 gait-quality refinement."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v033.rl_cfg import MIRROR_FUNC
from mjlab.tasks.velocity.config.pikachu_v034_biped_transfer.rl_cfg import (
  biped_transfer_ppo_cfg,
)


def v034_biped_gait_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = biped_transfer_ppo_cfg(retarget_feet=True)
  # The transfer adapter exposes V033 policy coordinates on both observations
  # and actions, so the validated V033 mirror transform applies at this boundary.
  cfg.algorithm.symmetry_cfg = {
    "data_augmentation_func": MIRROR_FUNC,
    # Start from an asymmetric but functional policy. A mirror loss corrects it
    # gradually; doubling every PPO batch caused the 500-update gate to stall.
    "use_data_augmentation": False,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 0.02,
  }
  cfg.max_iterations = 5000
  cfg.save_interval = 500
  cfg.experiment_name = "pikachu_v034/biped_gait"
  cfg.wandb_project = "pikachu-v034-velocity-biped-gait"
  cfg.wandb_tags = (
    "v034",
    "biped",
    "gait-quality",
    "symmetric",
    "swing-knee",
  )
  return cfg
