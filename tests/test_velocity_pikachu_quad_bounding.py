"""Configuration and isolation tests for Pikachu quadruped bounding."""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.pikachu_quad.quad_mdp import (
  quad_track_linear_velocity,
)
from mjlab.tasks.velocity.config.pikachu_quad_bounding.bounding_mdp import (
  bounding_contact_phases,
  bounding_pair_air_time,
  bounding_pair_desync_fraction,
  bounding_track_linear_velocity,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

BOUNDING_TASK = "Mjlab-Velocity-Flat-Pikachu-Quad-Bounding"
QUAD_TASK = "Mjlab-Velocity-Flat-Pikachu-Quad"


def test_quad_bounding_task_registered_with_private_rl_namespace() -> None:
  assert BOUNDING_TASK in list_tasks()
  rl_cfg = load_rl_cfg(BOUNDING_TASK)
  assert rl_cfg.experiment_name == "pikachu_velocity_quad_bounding"
  assert rl_cfg.wandb_project == "pikachu-velocity-quad-bounding"
  assert rl_cfg.wandb_tags == ("velocity", "quad", "bounding", "flat")


def test_quad_bounding_has_front_and_rear_contact_sensors() -> None:
  cfg = load_env_cfg(BOUNDING_TASK)
  sensor_names = {sensor.name for sensor in (cfg.scene.sensors or ())}
  assert "hands_ground_contact" in sensor_names
  assert "feet_ground_contact" in sensor_names
  assert "hand_contact" in cfg.observations["critic"].terms
  assert "hand_contact" not in cfg.observations["actor"].terms


def test_quad_bounding_uses_ordered_pair_gait_rewards() -> None:
  cfg = load_env_cfg(BOUNDING_TASK)
  assert "air_time" not in cfg.rewards

  linear = cfg.rewards["track_linear_velocity"]
  phase = cfg.rewards["bounding_phase"]
  pair_air = cfg.rewards["bounding_pair_air_time"]
  pair_desync = cfg.rewards["bounding_pair_desync"]
  assert isinstance(linear, RewardTermCfg)
  assert linear.func is bounding_track_linear_velocity
  assert linear.weight == 3.0
  assert phase.func is bounding_contact_phases
  assert phase.params["target_stance_time"] == 0.12
  assert phase.params["target_flight_time"] == 0.06
  assert pair_air.func is bounding_pair_air_time
  assert pair_air.params["target_air_time"] == 0.2
  assert pair_desync.func is bounding_pair_desync_fraction
  assert pair_desync.weight == -1.0


def test_quad_bounding_training_commands_focus_on_forward_motion() -> None:
  cfg = load_env_cfg(BOUNDING_TASK)
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == (0.15, 0.65)
  assert command.ranges.lin_vel_y == (-0.05, 0.05)
  assert command.ranges.ang_vel_z == (-0.1, 0.1)
  assert "command_vel" not in cfg.curriculum


def test_quad_bounding_play_keeps_independent_command_ranges() -> None:
  cfg = load_env_cfg(BOUNDING_TASK, play=True)
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == (-0.4, 0.8)
  assert command.ranges.lin_vel_y == (-0.2, 0.2)
  assert command.ranges.ang_vel_z == (-0.3, 0.3)
  assert cfg.curriculum == {}


def test_quad_parent_configuration_is_unchanged() -> None:
  cfg = load_env_cfg(QUAD_TASK)
  sensor_names = {sensor.name for sensor in (cfg.scene.sensors or ())}
  assert "hands_ground_contact" not in sensor_names
  assert "bounding_phase" not in cfg.rewards
  assert cfg.rewards["air_time"].weight == 3.0
  assert cfg.rewards["air_time"].params["target_air_time"] == 0.22
  assert cfg.rewards["track_linear_velocity"].func is quad_track_linear_velocity
  assert cfg.rewards["track_linear_velocity"].weight == 2.0
  assert cfg.rewards["base_height"].weight == -0.2
  assert cfg.rewards["base_height"].params == {
    "target_height": 0.177,
    "std": 0.02,
  }
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == (-0.4, 0.5)
