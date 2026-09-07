"""Pikachu v2.2 sample-links 四足全向 velocity 环境配置。

镜像旧 ``pikachu_quad``（capsule）：基于 ``make_velocity_env_cfg()`` 起步，
四足定制（``quad_upright``/``quad_bad_orientation``/世界系命令/脚接地）全部落地
本文件，复用 ``quad_mdp.py``。差异仅在机器人：v2.2 sample-links
（``pikachu_sl_v22_train.xml``，髋距收窄、腿更短）。

⚠️ 门禁标定 TODO：v2.2 比例与旧 capsule 不同，``base_height`` target 与
``QUAD_SL22_TILT_DEG``（紫箭头目标方向）需训练前用 ``--agent zero`` viewer
实测后回填，**不复用旧 capsule 的 QUAD_TILT_DEG=110.0**。先调好四足趴姿（手脚
四点支撑、躯干不塌）再开训——旧 ``QUAD_KEYFRAME`` 的关节摆位是按旧腿长标定的，
对 v2.2 更短的腿会塌成趴地（实测 zero-action 静置背轴角≈75°、base_z≈0.04、
base 触地）。
"""

import dataclasses
import math

from mjlab.asset_zoo.robots.pikachu.pikachu_constants_v22 import (
  QUAD_SL22_ACTION_SCALE,
  QUAD_SL22_BODY_AXIS,
  QUAD_SL22_SUPPORT_GEOM_NAMES,
  QUAD_SL22_TILT_DEG,
  QUAD_SL22_TORSO_BODY,
  get_pikachu_sl22_quad_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.velocity.config.pikachu_quad.quad_mdp import (
  QuadUniformVelocityCommandCfg,
  quad_bad_orientation,
  quad_base_height,
  quad_front_ang_vel,
  quad_front_command,
  quad_front_lin_vel,
  quad_reset_root_state_uniform,
  quad_track_angular_velocity,
  quad_track_linear_velocity,
  quad_upright,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

__all__ = ["pikachu_sl_v22_quad_flat_env_cfg"]

# 后脚 site(脚接地/脚抬高/脚打滑奖励载体;手暂无 site)。
_LEG_FOOT_SITES = ("left_foot", "right_foot")


def pikachu_sl_v22_quad_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """平坦地形四足 velocity 配置(快速起步:后脚 2 通道 + 姿态支撑手)。"""
  cfg = make_velocity_env_cfg()

  # 四足起步姿态(手脚四点、base 前俯)。
  cfg.scene.entities = {"robot": get_pikachu_sl22_quad_robot_cfg()}

  # 平滑平地接触仿真参数(与双足 flat 一致)。
  # njmax 沿用 make_velocity_env_cfg 的 1500：v2.2 box 碰撞体(脚/手 condim=3)
  #   EFC 行数比旧 capsule 多,收得太小 (300) 会 "nefc overflow" 丢约束。
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  # flat 无地形:删掉依赖 terrain_generator 的地形课程。
  cfg.curriculum.pop("terrain_levels", None)
  # 纯 flat 简化:移除一切地形/脚高度监测。
  cfg.scene.sensors = tuple(
    s
    for s in (cfg.scene.sensors or ())
    if s.name not in ("terrain_scan", "foot_height_scan")
  )
  for grp in cfg.observations.values():
    grp.terms.pop("height_scan", None)
    grp.terms.pop("foot_height", None)

  # 4 点支撑脚接地传感器。起步只跟踪**后脚**(左右 ankle,2 通道)。
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left|right)_ankle_link$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=QUAD_SL22_TORSO_BODY, entity="robot"),
    secondary=ContactMatch(
      mode="subtree", pattern=QUAD_SL22_TORSO_BODY, entity="robot"
    ),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  # 动作缩放:四足(前腿=arm 放宽抬脚幅度)。
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = QUAD_SL22_ACTION_SCALE

  cfg.viewer.body_name = QUAD_SL22_TORSO_BODY

  # 速度命令:四足走**世界系**(is_world_env)。
  twist_cfg = cfg.commands["twist"]
  assert isinstance(twist_cfg, UniformVelocityCommandCfg)
  twist_cmd = QuadUniformVelocityCommandCfg(
    **{f.name: getattr(twist_cfg, f.name) for f in dataclasses.fields(twist_cfg)}
  )
  cfg.commands["twist"] = twist_cmd
  twist_cmd.rel_standing_envs = 0.1
  twist_cmd.rel_world_envs = 1.0
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.rel_backward_envs = 0.0
  twist_cmd.rel_lateral_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.ranges.lin_vel_x = (-0.4, 0.5)
  twist_cmd.ranges.lin_vel_y = (-0.35, 0.35)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  # 四足前方参考系闭环:命令/线速/角速观测统一到"背轴水平投影"为 X 的前方系。
  for grp in ("actor", "critic"):
    cfg.observations[grp].terms["base_lin_vel"] = ObservationTermCfg(
      func=quad_front_lin_vel,
      noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
    )
    cfg.observations[grp].terms["base_ang_vel"] = ObservationTermCfg(
      func=quad_front_ang_vel,
      noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
    )
    cfg.observations[grp].terms["command"] = ObservationTermCfg(
      func=quad_front_command, params={"command_name": "twist"}
    )

  # 重置事件:四足版(yaw 绕世界竖直,保持脚贴地只转朝向)。
  cfg.events["reset_base"] = EventTermCfg(
    func=quad_reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.01, 0.05),
        "yaw": (-3.14, 3.14),
      },
      "velocity_range": {},
    },
  )

  # 速度跟踪:四足前方系版。
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=quad_track_linear_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=quad_track_angular_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.5)},
  )

  # 朝向奖励:四足躯干背轴带 tilt_deg 偏角(紫箭头目标方向)。v2.2 用专属
  # QUAD_SL22_TILT_DEG,**不复用旧 capsule 的 QUAD_TILT_DEG=110.0**(见常量模块
  # 注释;初值 90.0 = v2.2 keyframe 实测背轴角,调好趴姿后再回填)。
  cfg.rewards["upright"] = RewardTermCfg(
    func=quad_upright,
    weight=1.0,
    params={
      "std": math.sqrt(0.2),
      "tilt_deg": QUAD_SL22_TILT_DEG,
      "body_axis": QUAD_SL22_BODY_AXIS,
      "asset_cfg": SceneEntityCfg("robot", body_names=(QUAD_SL22_TORSO_BODY,)),
    },
  )

  # 摔倒终止:四足背轴偏离 tilt 超过阈值即止。
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=quad_bad_orientation,
    params={
      "limit_angle": math.radians(40.0),
      "tilt_deg": QUAD_SL22_TILT_DEG,
      "body_axis": QUAD_SL22_BODY_AXIS,
      "asset_cfg": SceneEntityCfg("robot", body_names=(QUAD_SL22_TORSO_BODY,)),
    },
  )

  # 四足躯干角速度/角动量在 base_link 上。
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (QUAD_SL22_TORSO_BODY,)
  cfg.rewards["angular_momentum"].weight = -0.02

  # 动作速率惩罚:加大 base 的 -0.1。四足 4 肢体、支撑/挥摆都要动,策略容易钻空子用
  # 高频小幅摆腿刷 track/air_time 奖励,导致步态"快速捣腾"。提高到 -0.25 压住连步跳变,
  # 换取稳定灵活的前进(仍低于双足方向上的激进压制,保留活动度)。可再调到 -0.3。
  cfg.rewards["action_rate_l2"].weight = -0.25

  # pose 正则:四足 clamp。
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.4,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle.*": 0.2,
    r".*arm_pitch.*": 0.4,  # 前腿前后摆
    r".*arm_roll.*": 0.2,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle.*": 0.2,
    r".*arm_pitch.*": 0.5,
    r".*arm_roll.*": 0.3,
  }

  # 纯 flat 简化:删掉依赖已移除 foot_height_scan 的脚高奖励。
  for name in ("foot_clearance", "foot_swing_height"):
    cfg.rewards.pop(name, None)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = _LEG_FOOT_SITES

  # 脚触地节奏(后脚 air_time)。
  cfg.rewards["air_time"] = RewardTermCfg(
    func=__import__(
      "mjlab.tasks.velocity.mdp", fromlist=["alternating_feet_air_time"]
    ).alternating_feet_air_time,  # noqa: E501
    weight=3.0,
    params={
      "sensor_name": "feet_ground_contact",
      "target_air_time": 0.22,
      "std": 0.08,
      "command_name": "twist",
      "command_threshold": 0.15,
    },
  )
  # base 高度:用 quad_base_height(class term,数学同共享 base_height_penalty),在
  # viser 画 target 高度参考盘 + 当前 base 点,便于 --agent zero 标定。**先调好四足趴姿
  # (手脚四点、躯干不塌)再回填此值**——0.15 只是初值,见 TODO。
  cfg.rewards["base_height"] = RewardTermCfg(
    func=quad_base_height,
    weight=-0.2,
    params={"target_height": 0.133, "std": 0.02},  # TODO(门禁标定 v2.2)
  )

  # 脚摩擦:四足用后脚+手支撑 geom。
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = QUAD_SL22_SUPPORT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (QUAD_SL22_TORSO_BODY,)

  # 共享 Pikachu 逐关节位置/速度/扭矩 Metrics。
  add_pikachu_telemetry_metrics(cfg)

  # 评估(play)模式覆盖:无限回合、关随机化。
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    twist_cmd.ranges.lin_vel_x = (-0.6, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
