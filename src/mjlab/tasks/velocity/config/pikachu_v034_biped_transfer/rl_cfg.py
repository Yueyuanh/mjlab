"""Independent experiment identity for V033-to-V034 walking transfer."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v034_biped_e1.rl_cfg import (
  v034_biped_e1_ppo_cfg,
)


def biped_transfer_ppo_cfg(
  retarget_feet: bool = False, load_adapted: bool = False
) -> RslRlOnPolicyRunnerCfg:
  cfg = v034_biped_e1_ppo_cfg()
  cfg.experiment_name = "pikachu_v034/biped_transfer"
  cfg.wandb_project = "pikachu-v034-velocity-biped-transfer"
  cfg.wandb_tags = ("v034", "biped", "v033-transfer", "coordinate-map")
  if retarget_feet:
    cfg.experiment_name = "pikachu_v034/biped_retarget"
    cfg.wandb_project = "pikachu-v034-velocity-biped-retarget"
    cfg.wandb_tags += ("foot-jacobian",)
  if load_adapted:
    cfg.experiment_name += "_load"
    cfg.wandb_project += "-load"
    cfg.wandb_tags += ("load-adapted-gains",)
  return cfg
