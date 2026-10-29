"""E3 transition environment with slow targets and long endpoint recovery."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.config.pikachu_v033_transfer import mdp as base
from mjlab.tasks.velocity.config.pikachu_v033_transfer.env_cfgs import transfer_env_cfg

from . import mdp


def transfer_e3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = transfer_env_cfg(play=play)
  cfg.scene.num_envs = 256 if not play else 16
  cfg.episode_length_s = 24.0
  cfg.commands["posture"] = mdp.SmoothPostureCommandCfg()
  cfg.events["reset_endpoints"] = EventTermCfg(func=base.reset_endpoints, mode="reset")
  cfg.rewards["contacts_at_goal"] = RewardTermCfg(
    func=mdp.endpoint_contact_score, weight=1.5
  )
  cfg.rewards["quiet_at_goal"] = RewardTermCfg(
    func=mdp.endpoint_quiet_score, weight=1.5
  )
  cfg.rewards["orientation_progress"] = RewardTermCfg(
    func=mdp.orientation_progress, weight=3.0
  )
  cfg.rewards["front_contact_schedule"] = RewardTermCfg(
    func=mdp.staged_front_contact_cost, weight=-0.6
  )
  cfg.rewards["vertical_speed"] = RewardTermCfg(
    func=mdp.vertical_speed_cost, weight=-0.10
  )
  cfg.rewards["torso_contact"] = RewardTermCfg(func=mdp.torso_contact_cost, weight=-2.0)
  cfg.rewards["action_rate"].weight = -0.04
  cfg.rewards["action_acc"] = RewardTermCfg(func=common.action_acc_l2, weight=-0.015)
  cfg.rewards["joint_velocity"] = RewardTermCfg(func=common.joint_vel_l2, weight=-2e-4)
  cfg.terminations["unsafe"] = TerminationTermCfg(
    func=mdp.unrecoverable_state, params={"torso_grace_s": 2.5}
  )
  return cfg
