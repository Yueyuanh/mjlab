"""Pikachu 四足全向 velocity 环境配置。

不触碰共享文件:基于 ``make_velocity_env_cfg()`` 起步,四足定制全部在本文件
落地。共享 velocity 的奖励/终止/命令/观测/事件结构全保留,只:
  1. 机器人换四足起步姿态(``QUAD_START``,手脚四点、base 前俯);
  2. ``upright`` 奖励换四足私有 ``quad_upright``(目标带 ``tilt_deg`` 偏角);
  3. ``fell_over`` 终止换四足私有 ``quad_bad_orientation``;
  4. 脚接地/空气时间**保持后脚 2 通道**(复用共享 ``feet_air_time``/``feet_clearance``),
     前手由姿态 + pose/upright 维持支撑,四足交替脚 air_time 留第二版
     (需先给手部加 site,手目前无可用的 foot site,避免 critic 维数错配)。

四足 target 偏角 ``tilt_deg``/``body_axis`` 与 ``base_height`` target 是
**待门禁标定 TODO**:训练前需 ``--agent zero`` viewer 保持初始姿态实测后回填,
不要直接开训。
"""

import dataclasses
import math

from mjlab.asset_zoo.robots.pikachu.pikachu_quad_constants import (
  QUAD_ACTION_SCALE,
  QUAD_SUPPORT_GEOM_NAMES,
  QUAD_TORSO_BODY,
  get_pikachu_quad_robot_cfg,
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
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from .quad_mdp import (
  QUAD_BODY_AXIS,
  QUAD_TILT_DEG,
  QuadUniformVelocityCommandCfg,
  quad_bad_orientation,
  quad_front_ang_vel,
  quad_front_command,
  quad_front_lin_vel,
  quad_reset_root_state_uniform,
  quad_track_angular_velocity,
  quad_track_linear_velocity,
  quad_upright,
)

__all__ = ["pikachu_quad_flat_env_cfg"]

# 后脚 site(脚接地/脚抬高/脚打滑奖励载体;手暂无 site)。
_LEG_FOOT_SITES = ("left_foot", "right_foot")


def pikachu_quad_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """平坦地形四足 velocity 配置(快速起步:后脚 2 通道 + 姿态支撑手)。"""
  cfg = make_velocity_env_cfg()

  # 四足起步姿态(手脚四点、base 前俯)。
  cfg.scene.entities = {"robot": get_pikachu_quad_robot_cfg()}

  # 平滑平地接触仿真参数(与双足 flat 一致)。
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  # flat 无地形:删掉依赖 terrain_generator 的地形课程(共享 ``terrain_levels_vel``
  # 在 ``curriculum_manager`` 里 assert 非空,不删会一 reset 就崩)。
  cfg.curriculum.pop("terrain_levels", None)
  # 纯 flat 简化:移除一切地形/脚高度监测——terrain_scan(地形扫描)与
  # foot_height_scan(脚下方地面扫描,其洋红射线正是 viewer 里的"紫箭头")。
  # 相关观测(foot_height)随传感器一并去掉,避免引用缺失。起步只靠纯物理接触。
  cfg.scene.sensors = tuple(
    s
    for s in (cfg.scene.sensors or ())
    if s.name not in ("terrain_scan", "foot_height_scan")
  )
  for grp in cfg.observations.values():
    grp.terms.pop("height_scan", None)
    grp.terms.pop("foot_height", None)

  # 4 点支撑脚接地传感器。起步只跟踪**后脚**(左右 ankle,2 通道),与 critic
  # 的 foot_height(后脚 site)对齐;手由四足姿态维持支撑。手部要参与空气
  # 时间需先加手 site,见模块 docstring(第二版)。
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
    primary=ContactMatch(mode="subtree", pattern=QUAD_TORSO_BODY, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=QUAD_TORSO_BODY, entity="robot"),
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
  joint_pos_action.scale = QUAD_ACTION_SCALE

  cfg.viewer.body_name = QUAD_TORSO_BODY

  # 速度命令:四足走**世界系**(is_world_env)。四足 base 前俯,共享 base-frame
  # 命令的"前进=base+X"在 base 前俯后竖直、失效;改世界系后命令前进=世界平面
  # 方向(quad_mdp 据此用世界速度跟踪)。因此关掉 base-frame 的
  # forward/lateral/heading 划分,全部走 is_world_env 分支。
  # 换成四足命令类(覆写 debug_vis 用世界命令画箭头,build 返回子类);字段整体
  # 拷贝自共享 cfg(dataclass.replace 不能换类)。采样/更新语义与父类一致。
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
  twist_cmd.rel_heading_envs = (
    0.0  # 关 heading:世界命令已含方向,避免产生 base-frame heading env
  )
  twist_cmd.ranges.lin_vel_x = (-0.4, 0.5)
  twist_cmd.ranges.lin_vel_y = (-0.35, 0.35)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  # 四足前方参考系闭环:命令/线速/角速观测统一到"背轴水平投影"为 X 的前方系。
  # 共享 obs 里 body 系(base_link)变量被 base 前俯污染、世界系变量又与 base 朝向
  # 无关;这里统一绕竖直 yaw 旋转,让策略看到"相对四足前方的 前进/横移/转向"。
  # 噪声梯度沿用共享 body 档(domain randomization 不退化)。
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

  # 重置事件:共享 `reset_base` 把 yaw 加在 base 局部帧,前俯后变翻滚;换四足版
  # (yaw 绕世界竖直,保持脚贴地只转朝向)。位置/速度随机仍沿用共享逻辑。
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

  # 速度跟踪:四足 base 前俯,共享 base-frame 版失效,换四足前方系版
  # (读 vel_command_w + root_link_lin_vel_w,方向=世界前向)。std 沿用双足值。
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

  # 朝向奖励:四足躯干背轴带 tilt_deg 偏角(门禁标定后回填)。
  cfg.rewards["upright"] = RewardTermCfg(
    func=quad_upright,
    weight=1.0,
    params={
      "std": math.sqrt(0.2),
      "tilt_deg": QUAD_TILT_DEG,  # TODO(门禁标定)
      "body_axis": QUAD_BODY_AXIS,  # TODO(门禁标定)
      "asset_cfg": SceneEntityCfg("robot", body_names=(QUAD_TORSO_BODY,)),
    },
  )

  # 摔倒终止:四足背轴偏离 tilt_deg 超过阈值即止(防侧翻/背朝天)。
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=quad_bad_orientation,
    params={
      "limit_angle": math.radians(40.0),
      "tilt_deg": QUAD_TILT_DEG,
      "body_axis": QUAD_BODY_AXIS,
      "asset_cfg": SceneEntityCfg("robot", body_names=(QUAD_TORSO_BODY,)),
    },
  )

  # 四足躯干角速度/角动量在 base_link 上。
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (QUAD_TORSO_BODY,)
  cfg.rewards["angular_momentum"].weight = -0.02

  # pose 正则:四足 clamp——前腿(arm)也与后腿同档参与迈步,可实现自然四足
  # 步态;hip_pitch/knee 放宽,arm_pitch/arm_roll 放宽(前腿抬/落)。
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.4,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle.*": 0.35,
    r".*arm_pitch.*": 0.4,  # 前腿前后摆
    r".*arm_roll.*": 0.2,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle.*": 0.35,
    r".*arm_pitch.*": 0.5,
    r".*arm_roll.*": 0.3,
  }

  # 纯 flat 简化:脚抬高约束(foot_clearance/foot_swing_height)依赖已移除的
  # foot_height_scan,一并删掉;脚打滑仍用 feet_ground_contact(纯接触,保留)。
  # 后脚 air_time(接触触发)负责脚触地节奏。
  for name in ("foot_clearance", "foot_swing_height"):
    cfg.rewards.pop(name, None)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = _LEG_FOOT_SITES

  # 脚触地传感器/奖励参数挂在后脚上。
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
  # base 高度:前俯 base 起步的稳定实测 root 离地高度。门禁标定(/tmp/quad_diag.py
  # 训练 cfg)zero-action 120 步 settle base_z≈0.177。旧值 0.062 是早期旧测量,
  # 会让策略把 base 强压到贴地、leg 蜷起 → 顶在原地怪姿态,已按实测回填。
  cfg.rewards["base_height"] = RewardTermCfg(
    func=__import__(
      "mjlab.tasks.velocity.mdp", fromlist=["base_height_penalty"]
    ).base_height_penalty,  # noqa: E501
    weight=-0.2,
    params={"target_height": 0.177, "std": 0.02},
  )

  # 脚摩擦:四足也用后脚+手支撑 geom(脚 foot1~7 + 手 arm_yaw)。
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = QUAD_SUPPORT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (QUAD_TORSO_BODY,)

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
