"""Pikachu velocity environment configurations.

模仿 Unitree G1 的 velocity 任务配置（``config/g1/env_cfgs.py``），
把机器人相关部分替换为 Pikachu 的资产。整体结构（观测/动作/命令/事件/
奖励/终止/课程）全部复用 ``make_velocity_env_cfg()`` 基础工厂，这里只做
机器人相关的定制。

与 G1 的主要差异（Pikachu 资产特性）：
  - 躯干 body 名是 ``base_link``（G1 是 ``torso_link``）。
  - 足部 site 是 ``left_foot`` / ``right_foot``（已加入 ankle body）。
  - 脚底碰撞几何在 ``left_ankle_link`` / ``right_ankle_link`` body 的子树内
    （即 ``left_ankle_link_collision`` / ``right_ankle_link_collision``），
    因此脚底接触传感器用 ``subtree`` 匹配整个 ankle body。
  - 关节为 14 个：``hip_pitch / hip_roll / hip_yaw / knee / ankle /
    arm_pitch / arm_roll``（左右各一），没有 waist/shoulder/elbow/wrist，
    因此 pose 奖励的正则按这些关节名配置。
"""

from mjlab.asset_zoo.robots import (
  PIKACHU_ACTION_SCALE,
  get_pikachu_robot_cfg,
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
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# 足部 site 与碰撞几何名（Pikachu 资产）。
SITE_NAMES = ("left_foot", "right_foot")
FOOT_GEOM_NAMES = ("left_ankle_link_collision", "right_ankle_link_collision")
# 躯干（基座）body 名。
TORSO_BODY = "base_link"


def pikachu_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """创建 Pikachu 粗糙地形（rough）velocity 配置。"""
  cfg = make_velocity_env_cfg()

  # 粗糙地形接触较多，提高 CCD（连续碰撞检测）迭代与接触槽上限。
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  # 机器人实体：用 get_pikachu_robot_cfg() 返回的默认配置
  # （FEET_ONLY_COLLISION：只保留脚底与地面碰撞）。
  cfg.scene.entities = {"robot": get_pikachu_robot_cfg()}

  # 地形扫描（terrain_scan）以躯干为参考系（做速度指令 yaw 对齐）。
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = TORSO_BODY

  # 足底高度扫描（foot_height_scan）挂在左右足部 site 上。
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  # 脚底接地接触传感器：primary 为左右踝 body 子树（含脚掌碰撞 geom），
  # secondary 为地形（body 名 "terrain"）。
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
  # 自碰撞传感器：检测整机（base_link 子树内）的自接触。
  # Pikachu 的相邻 link 接触已在 contact.xml 中排除，剩下的接触即异常碰撞。
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

  # 启用地形课程（随学习进度提升地形难度）。
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # 关节位置动作使用 Pikachu 的动作缩放系数（由执行器 PD 增益推导）。
  #
  # 重要：PIKACHU_ACTION_SCALE = 0.25 * effort_limit / stiffness 是按
  # 电机力矩预算推导的，对小电机（J3507 仅 5 N·m）会得到很保守的数值
  # （踝/髋roll/yaw 仅 0.101 rad，约 5.8°，相对关节限位 1.57 rad 只用了
  # 6%）。实测 PD 对阶跃目标跟踪率 >99%，物理完全能迈步——卡住前进的
  # 瓶颈是动作幅度不足。因此这里放宽到与 G1 相当的步态幅度：
  #   hip_pitch / knee：±0.5 rad（G1 为 0.55）
  #   hip_roll / hip_yaw / ankle / arm：±0.35 rad（G1 为 0.35~0.44）
  # 若电机力矩跟不上升级为抽搐/打滑，再回调。
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = {
    r".*_hip_pitch_joint": 0.5,
    r".*_knee_joint": 0.5,
    r".*_hip_roll_joint": 0.35,
    r".*_hip_yaw_joint": 0.35,
    r".*_ankle_joint": 0.35,
    r".*_arm_pitch_joint": 0.35,
    r".*_arm_roll_joint": 0.35,
  }

  # 查看器跟踪躯干。
  cfg.viewer.body_name = TORSO_BODY

  # 速度指令的调试可视化偏移量（相对躯干高度）。
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.15

  # 域随机化：脚底摩擦系数 + 质心偏移作用在躯干上。
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  # pose 奖励的关节标准差（rad）：规定各关节在站立/行走/跑步时的"放松度"。
  # 与 G1 的取值逻辑一致：
  #   - hip_pitch/knee 放宽以允许自然迈步弯曲；
  #   - hip_roll/hip_yaw 收紧以抑制横向摇摆、稳定步态；
  #   - ankle 中等（Pikachu 踝为单自由度，兼顾平衡与抬脚）；
  #   - arm_pitch/arm_roll 中等以允许自然摆臂。
  # 跑步（running）约为行走（walking）的 1.5~2 倍，以适应更大动作幅度。
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle.*": 0.25,
    r".*arm_pitch.*": 0.15,
    r".*arm_roll.*": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle.*": 0.35,
    r".*arm_pitch.*": 0.3,
    r".*arm_roll.*": 0.2,
  }

  # 躯干直立、躯干角速度惩罚作用在 base_link 上。
  cfg.rewards["upright"].params["asset_cfg"].body_names = (TORSO_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (TORSO_BODY,)

  # 抬脚高度 / 打滑奖励依赖足部 site。
  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = SITE_NAMES

  # 奖励权重：轻微抑制躯干旋转与角动量。
  # air_time 设为 1.0：激励抬脚迈步（配合 command_threshold=0.5 门控，
  # 只在指令速度 >0.5 m/s 时生效，站立时不会诱导跳跃）。
  # 上一轮（trial-02）air_time=0 且 action scale 过小，策略学会"站着转"
  # 的局部最优；放宽动作幅度后配合 air_time 正奖励，引导它真正抬脚前进。
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 1.0

  # 自碰撞惩罚（力超过阈值计为一次碰撞）。
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # 评估（play）模式覆盖：无限 episode、关闭随机化、固定 5x5 地形。
  if play:
    # 无限回合时长。
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


def pikachu_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """创建 Pikachu 平坦地形（flat）velocity 配置。"""
  cfg = pikachu_rough_env_cfg(play=play)

  # 平地上接触少，放宽 CCD 与接触槽。
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # 切换到平坦地形（关闭地形生成器）。
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # 平地上无地形可扫描：移除地形扫描传感器与对应观测。
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # 关闭地形课程（play 模式下 rough 已清空 curriculum）。
  cfg.curriculum.pop("terrain_levels", None)

  # 评估（play）模式下放宽速度指令范围以充分展示能力。
  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
