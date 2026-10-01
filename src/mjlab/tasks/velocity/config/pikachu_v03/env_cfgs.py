"""Pikachu **v03**（并联四杆膝）velocity 环境配置。

结构上照抄 ``config/pikachu/env_cfgs.py``（14dof V025），只改 v03 与 V025
真正不同的地方。差异清单及原因：

1. **资产来源**：``pikachu_v03_constants``（XML 为
   ``xmls/mjcf_v03/pikachu_v03_base_series_train.xml``）。

2. **脚底碰撞 geom 是 5 条胶囊**：``^(left|right)_foot[1-5]_collision$``
   （V025 是 7 条脚趾胶囊）。数量不同，正则必须跟着改，否则
   ``foot_friction`` 域随机化会匹配不到任何 geom。

3. **没有手臂关节**。v03 是纯腿平台（14 个关节 = 2×7，其中 4 个是被动的
   ``knee`` / ``knee_mid``），所以动作缩放表里不再有 ``arm_pitch`` /
   ``arm_roll`` 条目，``pose`` 奖励的正则也不含手臂。

4. **膝是并联四杆，驱动关节叫 ``knee_up_joint``**。策略能控制的只有
   ``hip_pitch / hip_roll / hip_yaw / knee_up / ankle``（左右各一，共 10 个）。
   被动的 ``knee_joint`` / ``knee_mid_joint`` **不参与**动作、姿态正则和
   超规格统计 —— 它们的值是四杆约束决定的，惩罚它们等于惩罚机构学。

5. **动作缩放按真实限位收窄**。v03 的 ``hip_roll`` / ``hip_yaw`` 限位只有
   ±0.26 rad（±15°），V025 用的 0.35 rad 会直接把目标顶到限位上，
   招来 ``dof_pos_limits`` 惩罚并让策略学出"贴限位"的畸形姿态。
   这里按限位逐关节定标：

     hip_pitch 0.50（限位 ±2.45）
     knee_up   0.40（限位 -1.22~+0.52，默认 -0.20）
     hip_roll  0.20（限位 ±0.26）
     hip_yaw   0.20（限位 ±0.26）
     ankle     0.35（限位 -2.36~+0.79，默认 -0.39）

6. **``base_height`` 目标高度按 v03 实测重定**。v03 微屈膝站姿的
   ``base_link`` 高度是 0.26879 m（``KNEES_BENT_KEYFRAME``），
   沿用 V025 的 0.17 m 会让机器人被奖励压到地上。
"""

from mjlab.asset_zoo.robots.pikachu.pikachu_v03_constants import (
  PIKACHU_V03_ACTUATOR_GROUPS,
  PIKACHU_V03_FOOT_GEOM_NAMES,
  PIKACHU_V03_JOINT_NAMES,
  PIKACHU_V03_POSE_JOINT_NAMES,
  get_pikachu_v03_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# 足部 site 名（v03 XML 里已有 left_foot / right_foot）。
SITE_NAMES = ("left_foot", "right_foot")
# 躯干（基座）body 名。
TORSO_BODY = "base_link"

# v03 微屈膝站姿下 base_link 的实测高度（见 pikachu_v03_constants.py）。
STAND_HEIGHT = 0.26879


def pikachu_v03_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """创建 Pikachu v03 粗糙地形（rough）velocity 配置。"""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_pikachu_v03_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = TORSO_BODY

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

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
    primary=ContactMatch(mode="subtree", pattern=TORSO_BODY, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=TORSO_BODY, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # 动作缩放：见模块 docstring 第 5 条（按 v03 真实限位逐关节定标）。
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = {
    r".*_hip_pitch_joint": 0.50,
    r".*_knee_up_joint": 0.40,
    r".*_hip_roll_joint": 0.20,
    r".*_hip_yaw_joint": 0.20,
    r".*_ankle_joint": 0.35,
  }

  cfg.viewer.body_name = TORSO_BODY

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.15
  # 方向桶的总量要留够给「全随机」环境 —— 三个桶在 `_resample_command` 里
  # **都把 ang_vel_z 强制置 0**，桶占比越高，能见到 yaw 命令的环境就越少。
  # r02 用 0.25×3 = 0.75，只剩 25% 的环境练过转向，结果是
  # 「原地转向命令 0.30 → 实测 0.005」完全不会转。收到 0.50 给 yaw 让路；
  # 前进/后退已经 100% 跟踪，不需要那么宽的覆盖。
  twist_cmd.rel_forward_envs = 0.20
  twist_cmd.rel_backward_envs = 0.15
  twist_cmd.rel_lateral_envs = 0.15
  twist_cmd.ranges.lin_vel_x = (-0.4, 0.5)
  twist_cmd.ranges.lin_vel_y = (-0.35, 0.35)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-0.4, 0.5),
      "lin_vel_y": (-0.35, 0.35),
      "ang_vel_z": (-0.3, 0.3),
    },
    {
      "step": 2000 * 24,
      "lin_vel_x": (-0.6, 0.8),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-0.5, 0.5),
    },
  ]

  # 域随机化：脚底摩擦作用在 v03 的 5 条脚底胶囊上。
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    PIKACHU_V03_FOOT_GEOM_NAMES
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  # pose 奖励：只覆盖 10 个驱动关节（排除被动的 knee / knee_mid）。
  pose = cfg.rewards["pose"]
  pose.params["asset_cfg"].joint_names = PIKACHU_V03_POSE_JOINT_NAMES
  pose.params["std_standing"] = {".*": 0.05}
  # std_walking 要比「步态里这些关节的真实摆幅」略宽一点：收得太紧等于
  # 惩罚正常的迈步动作，策略会为了保住 pose 分而不敢抬腿。
  #
  # ⚠️ `hip_roll` / `hip_yaw` 尤其不能用 0.15 收着 —— v03 这两个关节的机械
  # 限位只有 **±0.26**，而**转向必须动 hip_yaw、侧移必须动 hip_roll**。
  # 给 0.15 等于「一动就罚」，r02 实测就是原地转向命令 0.30 → 实际 0.005、
  # 侧移只有 41%。std 要放到接近限位的量级才算「不惩罚」。
  pose.params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.30,
    # knee_up 的行程比串联膝小（±0.87 rad 全行程），std 相应收窄。
    r".*knee_up.*": 0.5,
    r".*ankle.*": 0.45,
  }
  pose.params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee_up.*": 0.5,
    r".*ankle.*": 0.35,
  }
  pose.params["running_threshold"] = 2.0

  cfg.rewards["upright"].params["asset_cfg"].body_names = (TORSO_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (TORSO_BODY,)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = SITE_NAMES

  # --- track 奖励的 std 必须随命令范围缩放 ---------------------------------
  # make_velocity_env_cfg 的默认值（lin std=√0.25=0.5、ang std=√0.5=0.707）
  # 是配 G1 那种大命令（±3 m/s）的。v03 的命令收窄到 lin ±0.5 / ang ±0.3 之后，
  # 「原地站住」的误差就只有 mean|cmd| ≈ 0.29 m/s：
  #     exp(-0.29² / 0.5²) = 0.72   → 站桩白拿 72% 的跟踪分
  # 梯度压力随之消失，策略直接落向「站着不动」（v03-flat-r01 实测：
  # error_vel_xy 平在 0.7 不降、air_time ≈ 0、episode_length 却顶满 1000）。
  # 这不是新坑 —— AMP 侧踩过一模一样的，见记忆 amp-track-std-must-match-command。
  # 收到命令上限量级：lin 0.20 → 站桩只剩 0.13；ang 0.25 → 站桩 0.70。
  cfg.rewards["track_linear_velocity"].params["std"] = 0.20
  #
  # ang 的 std 比 lin 宽：r02 实测 yaw 误差落在 0.2~0.6 rad/s 区间
  # （行走时未被命令的偏航 0.2~0.33，原地转向命令 0.30 则完全没跟）。
  # std=0.25 时该区间 exp(-err²/0.0625) 已经很小、梯度趋零：
  # 在 err=0.5 处梯度 -0.29，而 std=0.40 时是 -1.31。
  # 放宽的代价是站桩白拿的角速度分变多（exp(-0.15²/0.16)=0.87），
  # 但站桩已经被 lin std=0.20 和 air_time 摁死了，不靠这一项。
  cfg.rewards["track_angular_velocity"].params["std"] = 0.40

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02

  # 抬脚奖励拆成两项，缺一不可：
  #
  #   air_time  = feet_air_time（**每步**给分）
  #     ``alternating_feet_air_time`` 只在【落地瞬间】给分，策略一次都不抬脚时
  #     该项恒为 0、梯度也为 0 —— 光靠它推不出第一步（这是典型的死梯度，
  #     v03-flat-r01 里 air_time 实测 +0.0016）。每步给分的那版只要脚离地
  #     超过 threshold_min 就有梯度，用来把策略推离「站着不动」的鞍点。
  #
  #   alternating_air_time = alternating_feet_air_time（落地质量）
  #     它**不能**靠「双脚一起腾空」刷分（见其 docstring：要求另一只脚已有
  #     稳定支撑），所以作为高质量项保留，负责把步态收成真正的交替步。
  cfg.rewards["air_time"] = RewardTermCfg(
    func=mdp.feet_air_time,
    weight=1.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "threshold_min": 0.05,
      "threshold_max": 0.5,
      "command_name": "twist",
      "command_threshold": 0.15,
    },
  )
  cfg.rewards["alternating_air_time"] = RewardTermCfg(
    func=mdp.alternating_feet_air_time,
    weight=3.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "target_air_time": 0.22,
      "std": 0.08,
      "command_name": "twist",
      "command_threshold": 0.15,
    },
  )
  cfg.rewards["flight"] = RewardTermCfg(
    func=mdp.feet_flight_time,
    weight=-2.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "grace_period": 0.04,
      "command_name": "twist",
      "command_threshold": 0.15,
    },
  )
  cfg.rewards["base_vertical_velocity"] = RewardTermCfg(
    func=mdp.base_vertical_velocity_penalty,
    weight=-0.5,
  )
  cfg.rewards["base_height"] = RewardTermCfg(
    func=mdp.base_height_penalty,
    weight=-0.2,
    # v03 微屈膝站姿 base_link 高 0.26879 m；目标比站姿低约 1.9 cm。
    params={"target_height": 0.25, "std": 0.02},
  )

  cfg.rewards["foot_clearance"].params["target_height"] = 0.04
  cfg.rewards["foot_clearance"].weight = -4.0
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.04
  cfg.rewards["foot_swing_height"].weight = -2.0
  cfg.rewards["soft_landing"].weight = -2e-3
  cfg.rewards["action_rate_l2"].weight = -0.15
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=envs_mdp.action_acc_l2,
    weight=-0.02,
  )

  # 遥测指标：用 v03 的关节表与电机分组（J4310 12.5 N·m / J3507 9 N·m）。
  add_pikachu_telemetry_metrics(
    cfg,
    joint_names=PIKACHU_V03_JOINT_NAMES,
    actuator_groups=PIKACHU_V03_ACTUATOR_GROUPS,
  )

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def pikachu_v03_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """创建 Pikachu v03 平坦地形（flat）velocity 配置。"""
  cfg = pikachu_v03_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.6, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
