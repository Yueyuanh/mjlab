"""V034 crawl E3 with a local V033 policy-coordinate interface."""

from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.config.pikachu_v034_crawl_e3.env_cfgs import (
  v034_crawl_e3_env_cfg,
)

from . import mdp


def v034_crawl_transfer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = v034_crawl_e3_env_cfg(play=play)
  original = cfg.actions["joint_pos"]
  assert isinstance(original, JointPositionActionCfg)
  cfg.actions["joint_pos"] = mdp.CrawlTransferJointPositionActionCfg(
    **{field.name: getattr(original, field.name) for field in fields(original)}
  )
  # The transferred actor's normalized outputs were trained with these V033
  # residual amplitudes.  Mapping is applied after this reference scaling.
  cfg.actions["joint_pos"].scale = {
    r".*_hip_pitch_joint": 0.35,
    r".*_hip_roll_joint": 0.15,
    r".*_hip_yaw_joint": 0.15,
    r".*_knee_up_joint": 0.25,
    r".*_ankle_joint": 0.25,
    r".*_arm_pitch_joint": 0.25,
    r".*_arm_roll_joint": 0.25,
    r".*_arm_yaw_joint": 0.20,
    r".*_elbow_joint": 0.25,
  }
  for group in cfg.observations.values():
    group.terms["joint_pos"].func = mdp.joint_pos_reference
    group.terms["joint_vel"].func = mdp.joint_vel_reference
  return cfg
