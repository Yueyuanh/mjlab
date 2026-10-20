"""Pikachu **v033**（并联四杆膝 + 双臂 8-dof，4.70 kg 整机）velocity 环境配置。

注册任务：``Mjlab-Velocity-Flat-Pikachu-V033-E0``。

**与 v03 / v031 完全解耦**：本包只 import ``pikachu_v033_constants``，不引用
其它代际的任何常量、类或函数。奖励档从 v031 的 T6 逐条移植，但所有与机器
本体相关的量（站高、脚距、关节表、电机分组、pose 正则、动作缩放）都按 v033
重新取。

v033 相对 v031 的实质差异：

1. **资产**：``xmls/pikachu_v033/mjcf/pikachu_v033_train.xml``，18 个驱动关节
   = 腿 10（``hip_pitch / hip_roll / hip_yaw / knee_up / ankle`` × 左右）
   + 臂 8（``arm_pitch / arm_roll / arm_yaw / elbow`` × 左右）。另有 4 个被动
   四杆关节 ``knee`` / ``knee_mid``（每腿各 1），既没有 actuator，也不进 pose
   正则或超规格统计。

2. **初始姿态读取 BIPED_KEYFRAME**，``base_link`` 高 0.259412 m。
   2026-10-20 用户将左右 arm_roll 改为 -53°/+53°，其余关节为 0；
   ``standing_posture_l2`` 的正则目标随 keyframe 更新。

3. **默认站姿脚心间距 16.90 cm**（v031 是 17.30 cm），
   ``walking_stance_width`` 的下限按比例从 0.12 收到 0.11 m。

4. **电机分组与 v031 不同**：
   ``hip_pitch / hip_roll / hip_yaw / knee_up / ankle / arm_pitch / arm_roll``
   是 DM-J4310P **12.5 N·m** 档（v033 的 ankle 也是 12.5，v031 是 3.0），
   ``arm_yaw / elbow`` 是 DM-J3507 **3.0 N·m** 档。

5. **动作缩放**：腿部五项沿用 v031 的逐关节定标（机械限位完全相同：
   hip_pitch ±2.44、hip_roll/hip_yaw ±0.26、knee_up -0.65~1.05、
   ankle -0.79~1.05）；手臂四项按 ±1.5708 的限位单独给值，手臂不参与行走
   推进，缩放比腿略小，避免初始探索把手臂甩到限位。

E0 档 = T6 奖励档 + critic 接触力的 heading 系修正（镜像增强的必要条件），
与 v031 的 ``E0-Mirror005`` 环境侧改动一一对应；runner 侧见 ``rl_cfg.py``。
"""

from mjlab.asset_zoo.robots.pikachu.pikachu_v033_constants import (
  PIKACHU_V033_ACTUATOR_GROUPS,
  PIKACHU_V033_FOOT_GEOM_NAMES,
  PIKACHU_V033_JOINT_NAMES,
  PIKACHU_V033_POSE_JOINT_NAMES,
  get_pikachu_v033_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.pikachu_foot_telemetry import add_pikachu_foot_telemetry_metrics
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from .symmetry import foot_contact_forces_heading_frame

# 足部 site 名（v033 训练 XML 里已有 left_foot / right_foot）。
SITE_NAMES = ("left_foot", "right_foot")
# 躯干（基座）body 名。
TORSO_BODY = "base_link"

# qpos 全 0 时 base_link 的实测高度（鞋底刚好贴地，见 constants 文件第 3 条）。
STAND_HEIGHT = 0.259412


def pikachu_v033_t6_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """创建 Pikachu v033 平地 velocity 配置（T6 奖励档）。

  T6 是「精确跟踪 + 不靠窄步幅碎步糊弄」的奖励档：在平地基线上改奖励与
  命令覆盖，不动 actor/critic 形状。逐条改动见下方带 ``[T6-n]`` 的注释。
  """
  cfg = make_velocity_env_cfg()

  # --- 平地基线 -----------------------------------------------------------
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  cfg.scene.entities = {"robot": get_pikachu_v033_robot_cfg()}
  # BIPED_KEYFRAME already places the capsule soles on the ground. The generic
  # velocity reset adds 1--5 cm for larger robots; V033 must not inherit it.
  cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # 平地档没有地形高度扫描，但 `foot_height_scan` 这个射线传感器仍在场景里，
  # 必须把它的 frame 从基线的 G1 躯干名改到 v033 的足部 site，否则射线源为
  # 空，`prepare_rays()` 会以 "stack expects a non-empty TensorList" 崩掉。
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

  cfg.viewer.body_name = TORSO_BODY

  # 动作缩放：腿部按 v031 的逐关节定标值（两代腿链限位完全相同），手臂按
  # v033 的 ±1.5708 限位单独定标。每个正则必须恰好命中一种关节，不能重叠：
  # resolve_matching_names_values 对「一个名字命中多条正则」直接报错。
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = {
    r".*_hip_pitch_joint": 0.50,
    r".*_knee_up_joint": 0.40,
    r".*_hip_roll_joint": 0.20,
    r".*_hip_yaw_joint": 0.20,
    r".*_ankle_joint": 0.35,
    r".*_arm_pitch_joint": 0.35,
    r".*_arm_roll_joint": 0.30,
    r".*_arm_yaw_joint": 0.20,
    r".*_elbow_joint": 0.30,
  }

  # --- 命令覆盖与课程 -----------------------------------------------------
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.15
  # 方向桶恢复 0.25×3（后退/侧移样本），非方向桶全部用 heading 控制器驱动 yaw。
  # 「有多少环境能见到非零 yaw 命令」= rel_heading × (1 − 桶总量) ×
  # (1 − rel_standing)；方向桶在 _resample_command 里把 ang_vel_z 强制置 0，
  # 只有 heading 环境拿得到非零 yaw。
  twist_cmd.rel_forward_envs = 0.25
  twist_cmd.rel_backward_envs = 0.25
  twist_cmd.rel_lateral_envs = 0.25
  twist_cmd.rel_heading_envs = 1.0
  # [T6-1] 站桩样本 0.05 -> 0.10：门控后的站姿项只由 standing 环境携带，
  # 5% 样本收敛太慢。代价是 yaw 暴露面 23.75% -> 22.5%，可忽略。
  twist_cmd.rel_standing_envs = 0.10
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

  # --- 域随机化 -----------------------------------------------------------
  # 脚底摩擦作用在 v033 的 5 条脚底胶囊上。
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = PIKACHU_V033_FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  # --- pose 奖励：只覆盖 18 个驱动关节（排除被动的 knee / knee_mid）-------
  pose = cfg.rewards["pose"]
  pose.params["asset_cfg"].joint_names = PIKACHU_V033_POSE_JOINT_NAMES
  pose.params["std_standing"] = {".*": 0.05}
  # std_walking 要比步态里这些关节的真实摆幅略宽一点：收得太紧等于惩罚正常
  # 的迈步动作。hip_roll / hip_yaw 尤其不能用 0.15 收着 —— 它们限位只有
  # ±0.26，而转向必须动 hip_yaw、侧移必须动 hip_roll。手臂只要求"别僵、
  # 别甩飞"，给宽一点的 std，避免把自然摆臂也罚掉。
  pose.params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.30,
    r".*knee_up.*": 0.5,
    r".*ankle.*": 0.45,
    r".*arm_pitch.*": 0.50,
    r".*arm_roll.*": 0.30,
    r".*arm_yaw.*": 0.30,
    r".*elbow.*": 0.40,
  }
  pose.params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee_up.*": 0.5,
    r".*ankle.*": 0.35,
    r".*arm_pitch.*": 0.5,
    r".*arm_roll.*": 0.25,
    r".*arm_yaw.*": 0.2,
    r".*elbow.*": 0.35,
  }
  pose.params["running_threshold"] = 2.0

  cfg.rewards["upright"].params["asset_cfg"].body_names = (TORSO_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (TORSO_BODY,)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = SITE_NAMES

  # --- track 奖励的 std 必须随命令范围缩放 --------------------------------
  # 默认值（lin std=0.5、ang std=0.707）是配 G1 那种 ±3 m/s 大命令的。命令
  # 收窄到 lin ±0.5 / ang ±0.3 后，「原地站住」的误差就只有 ≈0.29 m/s：
  #     exp(-0.29² / 0.5²) = 0.72   → 站桩白拿 72% 的跟踪分
  # 梯度压力随之消失，策略直接落向「站着不动」（曲线健康但行为垮掉）。
  # 收到命令上限量级：lin 0.20 → 站桩只剩 0.13。
  cfg.rewards["track_linear_velocity"].params["std"] = 0.20
  # ang 的 std 比 lin 宽：行走时未被命令的偏航本底就有 0.2~0.33 rad/s，
  # std=0.25 时该区间梯度已趋零；放宽的代价是站桩白拿的角速度分变多，
  # 但站桩已被 lin std=0.20 和 air_time 摁死，不靠这一项。
  cfg.rewards["track_angular_velocity"].params["std"] = 0.40

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02

  # --- 抬脚奖励 -----------------------------------------------------------
  # 拆成两项，缺一不可：
  #   air_time = feet_air_time（**每步**给分）—— ``alternating_feet_air_time``
  #     只在落地瞬间给分，策略一次都不抬脚时恒为 0、梯度也为 0，光靠它推不出
  #     第一步。每步给分的那版只要脚离地超过 threshold_min 就有梯度，用来把
  #     策略推离「站着不动」的鞍点。
  #   alternating_air_time（落地质量）—— 不能靠「双脚一起腾空」刷分，负责把
  #     步态收成真正的交替步。
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

  # 遥测指标：用 v033 的关节表（18 个驱动关节）与电机分组。
  add_pikachu_telemetry_metrics(
    cfg,
    joint_names=PIKACHU_V033_JOINT_NAMES,
    actuator_groups=PIKACHU_V033_ACTUATOR_GROUPS,
  )
  # 速度追踪主线（命令/实测/误差逐分量），只观测不影响训练。
  mdp.add_velocity_tracking_metrics(cfg, command_name="twist")

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # =======================================================================
  # T6 奖励档
  # =======================================================================
  driven_joints = SceneEntityCfg("robot", joint_names=PIKACHU_V033_POSE_JOINT_NAMES)
  foot_sites = SceneEntityCfg("robot", site_names=SITE_NAMES)

  # [T6-2] 指数型 pose 奖励会饱和（站立关节 RMSE 一大，exp 项就归零、梯度也
  # 归零），补一个**无界二次罚**，只在命令≈0 的环境生效，走路环境恒为 0。
  # 它的目标是整个 BIPED_KEYFRAME，包括用户指定的手臂初态。
  cfg.rewards["standing_posture_l2"] = RewardTermCfg(
    func=mdp.standing_posture_l2,
    weight=-0.25,
    params={
      "std": {
        r".*hip_pitch.*": 0.12,
        r".*hip_roll.*": 0.08,
        r".*hip_yaw.*": 0.08,
        r".*knee_up.*": 0.12,
        r".*ankle.*": 0.12,
        r".*arm_pitch.*": 0.20,
        r".*arm_roll.*": 0.15,
        r".*arm_yaw.*": 0.15,
        r".*elbow.*": 0.20,
      },
      "asset_cfg": driven_joints,
      "command_name": "twist",
      "command_threshold": 0.10,
    },
  )
  # [T6-3] 原来的 0.25 m 目标等于**奖励下蹲**（v03 站姿 0.267）。改成与
  # v033 的静止平衡姿态一致的高度（全 0 姿态下 base_link = 0.259412 m）。
  cfg.rewards["base_height"].params["target_height"] = STAND_HEIGHT
  cfg.rewards["base_height"].params["std"] = 0.02
  # [T6-4] 完整 keyframe 正则已经隐含了标称站宽，删掉那条会恒定外推、把步态
  # 推成"原地踏步"的精确距离力。
  cfg.rewards.pop("feet_lateral_distance", None)

  # [T6-5] 保留有用的支撑多边形，但只在**双支撑**且**窄于下限**时生效；
  # 高于下限恒为 0，不会在摆动相一路把摆动腿往外推。
  # 下限按 v033 的默认站宽 16.90 cm 取 0.11（v031 是 17.30 -> 0.12）。
  cfg.rewards["walking_stance_width"] = RewardTermCfg(
    func=mdp.feet_lateral_separation_min,
    weight=-0.20,
    params={
      "sensor_name": "feet_ground_contact",
      "min_distance": 0.11,
      "std": 0.04,
      "command_name": "twist",
      "command_threshold": 0.15,
      "asset_cfg": foot_sites,
    },
  )

  # [T6-6..8] 步频过高 = 每步推进太少。把「短促的支撑摆动」显式罚掉，再把
  # 落地奖励的中心挪到更长的 0.28 s 摆动上；速度跟踪会迫使降低的步频转化
  # 成更长的步幅，而不是更慢的行进。
  cfg.rewards["air_time"].weight = 0.50
  cfg.rewards["alternating_air_time"].params["target_air_time"] = 0.28
  cfg.rewards["alternating_air_time"].params["std"] = 0.10
  cfg.rewards["short_step_air_time"] = RewardTermCfg(
    func=mdp.short_step_air_time,
    weight=-2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "min_air_time": 0.18,
      "std": 0.08,
      "command_name": "twist",
      "command_threshold": 0.15,
    },
  )

  # [T6-9] 指数跟踪器在弱档（后退/侧移）饱和后梯度趋零，补一个 Huber 项，
  # 同时保住 0 附近的精度。
  cfg.rewards["linear_velocity_huber"] = RewardTermCfg(
    func=mdp.linear_velocity_huber_error,
    weight=-0.25,
    params={"command_name": "twist", "delta": 0.20},
  )

  # [T6-10] 只在扭矩用到上限 80% 以上时罚，正常步态的出力是免费的。
  # 上限必须与 pikachu_v033_constants.py 的 effort_limit 一致：
  #   J4310 = 12.5 N·m（hip_pitch/hip_roll/hip_yaw/knee_up/ankle/arm_pitch/arm_roll）
  #   J3507 = 3.0 N·m（arm_yaw/elbow）
  cfg.rewards["torque_soft_limit"] = RewardTermCfg(
    func=mdp.joint_torque_soft_limit,
    weight=-0.25,
    params={
      "effort_limits": {
        r".*(hip_pitch|hip_roll|hip_yaw|knee_up|ankle|arm_pitch|arm_roll)_joint": (
          12.5
        ),
        r".*(arm_yaw|elbow)_joint": 3.0,
      },
      "soft_ratio": 0.80,
      "asset_cfg": driven_joints,
    },
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist_cmd.ranges.lin_vel_x = (-0.6, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
    add_pikachu_foot_telemetry_metrics(
      cfg, joint_names=PIKACHU_V033_JOINT_NAMES, entity_name="robot"
    )

  return cfg


def pikachu_v033_e0_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """E0：T6 环境 + 镜像增强所需的 critic 接触力坐标系修正。

  ``ContactSensor(reduce="netforce")`` 报的是世界系净力。reset 带随机 yaw，
  世界系力在矢状面镜像下不能直接交换左右脚 —— 必须先转到 base heading 系再
  做 ``sign(x) * log1p(|x|)`` 压缩。这是 E0-Mirror005 相对 T6 唯一的环境改动。
  """
  cfg = pikachu_v033_t6_flat_env_cfg(play=play)
  cfg.observations["critic"].terms[
    "foot_contact_forces"
  ].func = foot_contact_forces_heading_frame
  return cfg
