"""PPO configuration for the native-action V034 knee-led biped task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v033_biped_e1.rl_cfg import (
  biped_e1_ppo_cfg,
)
from mjlab.tasks.velocity.config.pikachu_v034_biped_transfer.rl_cfg import (
  biped_transfer_ppo_cfg,
)

MIRROR_FUNC = (
  "mjlab.tasks.velocity.config.pikachu_v034.symmetry:pikachu_v034_data_augmentation"
)


def v034_biped_knee_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = biped_e1_ppo_cfg()
  # Exploration noise must stay near the validated V033 E1 production range
  # (init 0.25, cap 0.35, entropy 0.001).  A first pilot with init 0.30, cap
  # 0.45 and entropy 0.003 drove the scalar std to the 0.45 cap; the
  # deterministic policy stood securely (0 falls, 0.998 contact duty) while
  # training rollouts fell about every 4.4 s purely from the injected noise.
  # See experiments/biped_knee_led_20261027/reports/exploration_noise_pilot.md.
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.25,
    "std_type": "scalar",
    "std_range": (0.05, 0.35),
  }
  cfg.algorithm.entropy_coef = 0.001
  # The V034 leg axes differ from V033.  The dedicated mirror module was
  # validated by closed-chain FK (residual < 1e-6) before enabling this.
  cfg.algorithm.symmetry_cfg = {
    "data_augmentation_func": MIRROR_FUNC,
    "use_data_augmentation": True,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 0.05,
  }
  cfg.max_iterations = 6000
  cfg.save_interval = 500
  cfg.experiment_name = "pikachu_v034/biped_knee"
  cfg.wandb_project = "pikachu-v034-velocity-biped-knee"
  cfg.wandb_tags = (
    "v034",
    "biped",
    "native-action",
    "knee-led",
    "symmetric-asset",
    "low-roll",
  )
  return cfg


def v034_biped_knee_retarget_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO for the knee-led fine-tune; inherits the transfer (mirror-free) setup."""
  cfg = biped_transfer_ppo_cfg(retarget_feet=True)
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.25,
    "std_type": "scalar",
    "std_range": (0.05, 0.35),
  }
  cfg.algorithm.entropy_coef = 0.001
  cfg.max_iterations = 2000
  cfg.save_interval = 250
  cfg.experiment_name = "pikachu_v034/biped_knee_retarget"
  cfg.wandb_project = "pikachu-v034-velocity-biped-knee-retarget"
  cfg.wandb_tags = (
    "v034",
    "biped",
    "retarget-coordinates",
    "knee-led",
    "symmetric-asset",
    "low-roll",
  )
  return cfg


def v034_biped_knee_sep_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO for the swing-stance separation fine-tune (same interface as v3)."""
  cfg = v034_biped_knee_retarget_ppo_cfg()
  cfg.experiment_name = "pikachu_v034/biped_knee_separation"
  cfg.wandb_project = "pikachu-v034-velocity-biped-knee-separation"
  cfg.wandb_tags = (
    "v034",
    "biped",
    "retarget-coordinates",
    "knee-led",
    "swing-stance-gap",
    "symmetric-asset",
    "low-roll",
  )
  return cfg
