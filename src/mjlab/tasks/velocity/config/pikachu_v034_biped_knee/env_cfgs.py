"""V034 biped tasks that reward active swing-knee flexion.

Two variants share the corrected left-leg asset and the knee rewards:

* ``v034_biped_knee_env_cfg`` uses the native 18-actuator interface.  It cannot
  be warm-started from any existing V034 walker, and from-scratch training in
  this task family does not learn to survive the 1-3 s push schedule at any
  affordable budget (see the experiment's ``reports/training_probe.md``).
* ``v034_biped_knee_retarget_env_cfg`` keeps the V033-transfer action
  coordinates, so the robust ``confirmed_mass_forward_5000`` walker transfers
  exactly.  Use it for fine-tuning; use the native variant only when an exact
  actor conversion exists.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.pikachu_v034_biped_e1.env_cfgs import (
  v034_biped_e1_env_cfg,
)
from mjlab.tasks.velocity.config.pikachu_v034_biped_e1.robot import (
  biped_e1_robot_cfg,
)
from mjlab.tasks.velocity.config.pikachu_v034_biped_gait.mdp import (
  straight_base_roll_rate,
)
from mjlab.tasks.velocity.config.pikachu_v034_biped_transfer.env_cfgs import (
  biped_transfer_env_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from . import mdp

KNEE_JOINTS = ("right_knee_up_joint", "left_knee_up_joint")


def v034_biped_knee_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = v034_biped_e1_env_cfg(play=play)
  cfg.scene.entities["robot"] = biped_e1_robot_cfg(symmetric=True)
  cfg.scene.num_envs = 1 if play else 1024

  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.rel_forward_envs = 0.45
  command.rel_backward_envs = 0.15
  command.rel_lateral_envs = 0.10
  command.rel_standing_envs = 0.10
  command.rel_heading_envs = 1.0
  # Fixed command ranges: a 6000-iteration run should not silently widen its
  # own task halfway through.  The previous curriculum doubled the ranges at
  # iteration 2000 and made evaluation incomparable across checkpoints.
  cfg.curriculum.pop("command_vel", None)
  if play:
    command.ranges.lin_vel_x = (-0.6, 0.8)
    command.ranges.lin_vel_y = (-0.4, 0.4)
    command.ranges.ang_vel_z = (-0.5, 0.5)
  else:
    command.ranges.lin_vel_x = (-0.40, 0.50)
    command.ranges.lin_vel_y = (-0.20, 0.20)
    command.ranges.ang_vel_z = (-0.30, 0.30)

  knee_cfg = SceneEntityCfg("robot", joint_names=KNEE_JOINTS, preserve_order=True)
  _add_knee_rewards(cfg, knee_cfg)
  _add_low_roll_rewards(cfg)
  return cfg


def v034_biped_knee_retarget_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Knee-led fine-tune in the V033-transfer action coordinates.

  The locomotion interface is byte-for-byte the one the robust
  ``confirmed_mass_forward_5000`` walker was trained on, so that actor and
  critic transfer exactly; only the asset, the command envelope and the gait
  quality rewards change.  Commands stay forward-only and inside the proven
  ``lin_vel_x`` envelope because the goal is a better forward gait, not new
  directions.
  """
  cfg = biped_transfer_env_cfg(play=play, retarget_feet=True)
  cfg.scene.entities["robot"] = biped_e1_robot_cfg(symmetric=True)
  cfg.scene.num_envs = 1 if play else 1024

  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.rel_forward_envs = 1.0
  command.rel_backward_envs = 0.0
  command.rel_lateral_envs = 0.0
  command.rel_standing_envs = 0.0
  command.rel_heading_envs = 0.0
  # The inherited curriculum *widens* the range to +-1.0 m/s at step 0, so
  # dropping it is what keeps the range below fixed and checkpoints comparable.
  cfg.curriculum.pop("command_vel", None)
  if play:
    command.ranges.lin_vel_x = (-0.6, 0.8)
    command.ranges.lin_vel_y = (-0.4, 0.4)
    command.ranges.ang_vel_z = (-0.5, 0.5)
  else:
    command.ranges.lin_vel_x = (-0.40, 0.50)
    command.ranges.lin_vel_y = (-0.20, 0.20)
    command.ranges.ang_vel_z = (-0.30, 0.30)

  knee_cfg = SceneEntityCfg("robot", joint_names=KNEE_JOINTS, preserve_order=True)
  _add_knee_rewards(cfg, knee_cfg)
  _add_low_roll_rewards(cfg)
  return cfg


def v034_biped_knee_sep_retarget_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Knee fine-tune whose carrot is the swing-minus-stance knee gap.

  Same asset, action interface, commands and penalties as
  :func:`v034_biped_knee_retarget_env_cfg`; only the positive knee term
  changes.  The 150-iteration pilot showed the absolute-flexion kernel
  saturating on a symmetric crouch (both knees about 0.18 rad, gap +0.02 rad),
  so this variant pays for the gap itself.  Everything else is held fixed, so
  a difference between the two runs is attributable to that term alone.
  """
  cfg = v034_biped_knee_retarget_env_cfg(play=play)
  knee_cfg = SceneEntityCfg("robot", joint_names=KNEE_JOINTS, preserve_order=True)
  cfg.rewards.pop("swing_knee_flexion")
  cfg.rewards["swing_knee_separation"] = RewardTermCfg(
    func=mdp.swing_knee_separation,
    weight=1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": knee_cfg,
      "scale": 0.15,
    },
  )
  return cfg


def _add_knee_rewards(cfg: ManagerBasedRlEnvCfg, knee_cfg: SceneEntityCfg) -> None:
  """Swing-flexion reward with a kernel that has gradient at the current gait.

  ``swing_knee_flexion`` is an exponential kernel.  With ``target=0.35`` and
  ``std=0.18`` the measured V034 gait (swing flexion about -0.09 rad, i.e. the
  swing knee *less* bent than the stance knee) sits 6 sigma away, where the
  reward is 0.0025 and its gradient is about 1e-4 per rad: the term was dead
  and the observed reward was 0.000 for the whole run.  ``target=0.30`` with
  ``std=0.35`` puts the current gait at 0.29 so flexion is actually rewarded.
  The weight is 1.5 (not 1.0) so the carrot reaches about 40% of the progress
  reward (see ``_add_low_roll_rewards`` for the budget it is compared with);
  the kernel is unimodal, so an over-flexed knee is not paid extra.
  """
  cfg.rewards["swing_knee_flexion"] = RewardTermCfg(
    func=mdp.swing_knee_flexion,
    weight=1.5,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": knee_cfg,
      "target": 0.30,
      "std": 0.35,
    },
  )
  cfg.rewards["stance_knee_extension"] = RewardTermCfg(
    func=mdp.stance_knee_extension,
    weight=-0.40,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": knee_cfg,
      "tolerance": 0.15,
      "std": 0.12,
    },
  )


def _add_low_roll_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Penalize the torso roll that the old policy used instead of knee flexion.

  Every weight here is budgeted against the progress reward the warm-started
  walker already earns, because a penalty that outbids progress turns the task
  optimum into standing still.  Measured on ``confirmed_mass_forward_5000``
  (nominal, forward clip): progress terms pay about 3.5 per step
  (``directional_progress`` 2.1 + ``track_linear_velocity`` 1.4), base roll
  rate is 1.19 rad/s.

  A first attempt used ``straight_base_roll_rate`` with weight -0.60, i.e.
  ``((1.19-0.20)/0.35)^2 * 0.6 = 4.8`` per step against 3.5 of progress.  That
  run (``aborted_roll_penalty_dominant_1024x300``) destroyed the walker within
  70 iterations and settled into a near-standing policy.  At -0.15 the same
  gait costs 1.2 per step (34% of progress) and halving the roll rate still
  saves about 1.0 per step, so the incentive survives without outbidding
  locomotion.
  """
  cfg.rewards["straight_base_roll_rate"] = RewardTermCfg(
    func=straight_base_roll_rate,
    weight=-0.15,
    params={"deadzone": 0.20, "std": 0.35},
  )
  # Hip roll is already small (5.3/7.2 deg mean, 9.5/13.1 deg p95), so this
  # term costs about 0.09 per step at the known gait: cheap, keeps its job.
  cfg.rewards["straight_hip_roll"].weight = -0.50
  cfg.rewards["straight_hip_roll"].params.update(deadzone=0.08, std=0.08)
  cfg.rewards["body_ang_vel"].weight = -0.20
  cfg.rewards["angular_momentum"].weight = -0.05
  cfg.rewards["action_rate_l2"].weight = -0.12
