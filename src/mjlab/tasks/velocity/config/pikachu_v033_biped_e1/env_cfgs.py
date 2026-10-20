"""Full 18-action upright E1 with soft arm windows and explicit quiet standing."""

from mjlab.asset_zoo.robots.pikachu.pikachu_v033_constants import (
  PIKACHU_V033_POSE_JOINT_NAMES,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp as velocity
from mjlab.tasks.velocity.config.pikachu_v033.env_cfgs import pikachu_v033_e0_env_cfg

from . import mdp
from .robot import ARM_BODIES, CLEARANCE_M, biped_e1_robot_cfg

ARM_JOINTS = tuple(
  f"{side}_{kind}_joint"
  for side in ("right", "left")
  for kind in ("arm_pitch", "arm_roll", "arm_yaw", "elbow")
)
ARM_BOUNDS_DEGREES = (
  (-10.0, 10.0),
  (45.0, 53.0),
  (-5.0, 5.0),
  (0.0, 10.0),
  (-10.0, 10.0),
  (-53.0, -45.0),
  (-5.0, 5.0),
  (-10.0, 0.0),
)
FOOT_SITES = ("right_foot", "left_foot")


def biped_e1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pikachu_v033_e0_env_cfg(play=play)
  cfg.scene.entities["robot"] = biped_e1_robot_cfg()
  cfg.scene.num_envs = 1024
  # Contact/air-time columns are right,left. Align the height scan and all
  # height/velocity pairings with them, fixing E0's left,right scan mismatch.
  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, TerrainHeightSensorCfg):
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES
      )
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = FOOT_SITES
    cfg.rewards[name].params["asset_cfg"].preserve_order = True
  # The mask policy allows arm contacts only with legs/soles. Keeping one
  # primary per arm body covers both same-side and opposite-side collisions.
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name="arm_leg_contact",
      primary=ContactMatch(mode="body", pattern=ARM_BODIES, entity="robot"),
      fields=("found", "dist", "force"),
      reduce="mindist",
      history_length=cfg.decimation,
    ),
  )
  command = cfg.commands["twist"]
  assert isinstance(command, velocity.UniformVelocityCommandCfg)
  command.rel_standing_envs = 0.25
  cfg.rewards["arm_position_window"] = RewardTermCfg(
    func=mdp.arm_position_window,
    weight=-2.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS, preserve_order=True),
      "bounds_degrees": ARM_BOUNDS_DEGREES,
    },
  )
  cfg.rewards["arm_leg_clearance"] = RewardTermCfg(
    func=mdp.arm_leg_clearance,
    weight=-2.0,
    params={"sensor_name": "arm_leg_contact", "clearance": CLEARANCE_M},
  )
  # Replace broad, previously ineffective self-contact detection with the
  # selective arm-leg sensor, including brief impacts inside a policy step.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=velocity.self_collision_cost,
    weight=-2.0,
    params={"sensor_name": "arm_leg_contact", "force_threshold": 1.0},
  )
  cfg.rewards["standing_posture_l2"].weight = -1.0
  cfg.rewards["standing_posture_l2"].params.update(
    std={
      r".*hip_pitch.*": 0.06,
      r".*hip_roll.*": 0.05,
      r".*hip_yaw.*": 0.05,
      r".*knee_up.*": 0.06,
      r".*ankle.*": 0.06,
      r".*arm_pitch.*": 0.08,
      r".*arm_roll.*": 0.05,
      r".*arm_yaw.*": 0.04,
      r".*elbow.*": 0.05,
    },
    command_threshold=0.05,
  )
  cfg.rewards["pose"].params["walking_threshold"] = 0.05
  for name in (
    "air_time",
    "alternating_air_time",
    "flight",
    "short_step_air_time",
    "foot_clearance",
    "foot_swing_height",
    "walking_stance_width",
  ):
    cfg.rewards[name].params["command_threshold"] = 0.05
  cfg.rewards["standing_joint_velocity"] = RewardTermCfg(
    func=mdp.standing_joint_velocity,
    weight=-0.5,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=PIKACHU_V033_POSE_JOINT_NAMES)
    },
  )
  cfg.rewards["standing_base_motion"] = RewardTermCfg(
    func=mdp.standing_base_motion,
    weight=-0.5,
  )
  cfg.rewards["standing_foot_motion"] = RewardTermCfg(
    func=mdp.standing_foot_motion,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES)},
  )
  cfg.rewards["standing_support"] = RewardTermCfg(
    func=mdp.standing_support,
    weight=-3.0,
    params={"sensor_name": "feet_ground_contact"},
  )
  # New soft costs must not make terminating early cheaper than balancing.
  # RewardManager scales by dt=.02: a non-timeout fall contributes -50.
  cfg.rewards["alive"] = RewardTermCfg(func=common.is_alive, weight=1.0)
  cfg.rewards["fall"] = RewardTermCfg(func=common.is_terminated, weight=-2500.0)
  # A 25-mm COM shift is large relative to this robot's soles. Use a small
  # physical COM randomization while learning the new quiet-standing task.
  cfg.events["base_com"].params["ranges"] = {i: (-0.005, 0.005) for i in range(3)}
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].func = mdp.push_moving_robots
  return cfg
