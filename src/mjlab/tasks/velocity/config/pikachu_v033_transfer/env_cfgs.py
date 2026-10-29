"""Fixed physical model and independent start-state/goal sampling."""

from mjlab.envs import mdp as common
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl.env_cfgs import crawl_env_cfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl_e2.mdp import ankle_limit_margin

from . import mdp


def transfer_env_cfg(play=False):
  cfg = crawl_env_cfg(play=False)
  cfg.scene.num_envs = 128 if not play else 16
  cfg.episode_length_s = 12.0
  cfg.metrics = {}
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    ContactSensorCfg(
      name="torso_ground",
      primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )
  cfg.curriculum = {}
  cfg.commands = {"posture": mdp.PostureCommandCfg()}
  # Reset start and command are sampled independently. No motion files loaded.
  cfg.events = {"reset_endpoints": EventTermCfg(func=mdp.reset_endpoints, mode="reset")}
  for group in cfg.observations.values():
    group.terms.pop("crawl_phase", None)
    group.terms.pop("command", None)
    group.terms["posture"] = ObservationTermCfg(
      func=common.generated_commands, params={"command_name": "posture"}
    )
    group.terms["base_lin_vel"] = ObservationTermCfg(func=mdp.body_velocity)
    group.terms["base_ang_vel"] = ObservationTermCfg(
      func=mdp.body_velocity, params={"angular": True}
    )
    # The heading-contact transform used by crawl degenerates when upright.
    group.terms.pop("foot_contact_forces", None)
    group.enable_corruption = False
  # Fixed action map, never switched or interpolated according to command.
  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=1.0,
    use_default_offset=False,
    offset=0.0,
    clip={".*": (-2.44, 2.44)},
  )
  cfg.rewards = {
    "goal_composite": RewardTermCfg(func=mdp.goal_score, weight=5.0),
    "goal_distance": RewardTermCfg(func=mdp.goal_distance, weight=-1.5),
    "orientation_progress": RewardTermCfg(func=mdp.orientation_progress, weight=3.0),
    "settled_soles": RewardTermCfg(func=mdp.settled_sole_cost, weight=-0.5),
    "ankle_margin": RewardTermCfg(
      func=ankle_limit_margin,
      weight=-1.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=("right_ankle_joint", "left_ankle_joint"),
          preserve_order=True,
        )
      },
    ),
    "orientation": RewardTermCfg(func=mdp.orientation_cost, weight=-1.0),
    "height": RewardTermCfg(func=mdp.height_cost, weight=-1.0),
    "contacts_at_goal": RewardTermCfg(func=mdp.contact_score, weight=1.0),
    "quiet_at_goal": RewardTermCfg(func=mdp.quiet_at_goal, weight=1.0),
    "motion": RewardTermCfg(func=mdp.motion_cost, weight=-0.5),
    "action_rate": RewardTermCfg(func=common.action_rate_l2, weight=-0.015),
    "joint_limits": RewardTermCfg(func=common.joint_pos_limits, weight=-2.0),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=common.time_out, time_out=True),
    "unsafe": TerminationTermCfg(func=mdp.unsafe_state),
  }
  return cfg
