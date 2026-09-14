"""Pikachu **27-dof 四足** 前进 tracking 环境配置。

基于 ``make_tracking_env_cfg()`` 起步,机器人同 velocity-quad:27-dof 拟真全身取
四足构型(base 前俯 ≈107°,手脚四点撑地)。核心设定:

  - **策略动作驱动 16 走步关节**(腿 10 + 前腿 arm_pitch/roll/elbow 6)跟踪四足
    参考走步;``arm_yaw`` 固定 0、头/耳/尾固定在关键帧姿态。
  - **参考运动**:由 ``scripts/make_27dof_quad_reference.py`` 从 14dof 四足 mocap
    CSV FK 重放生成的四足 27dof 参考,``QUAD_27DOF_MOTION_FILE`` 指向其 npz。
  - 跟踪体/关节为四足布局:anchor=base_link,body = anchor + 四条腿链,
    joint = 16 走步关节。

脚底:27dof 无 foot site,tracking 的 body 跟踪奖励/终止直接基于 ``base_link`` 与
四腿 link,不依赖足部 site。
"""

from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_quad_constants import (
  QUAD_27DOF_ACTION_SCALE,
  QUAD_27DOF_MOTION_FILE,
  QUAD_27DOF_SUPPORT_GEOM_NAMES,
  QUAD_27DOF_WALK_JOINT_NAMES,
  get_pikachu_27dof_quad_robot_cfg,
  pikachu_27dof_quad_walk_bodies,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# 躯干(基座)body 名。
ANCHOR_BODY_NAME = "base_link"

# 后端终止只保留四腿末端:后脚踝 + 前手掌(肘)。与参考 body 布局对齐。
EE_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_link",
  "right_ankle_link",
  "left_elbow_link",
  "right_elbow_link",
)


def pikachu_27dof_quad_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Pikachu 27-dof quadrilateral forward-walk tracking config.

  Reference motion is a placeholder (see ``QUAD_27DOF_MOTION_FILE``); the task is
  scaffolded but not trainable until a real 27-dof quad reference is provided.
  """
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_pikachu_27dof_quad_robot_cfg()}

  # 27dof box 碰撞体多,放宽接触容量。
  cfg.sim.njmax = 800
  cfg.sim.nconmax = 70
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  # 自碰撞传感器(tracking 奖励引用 name="self_collision")。
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=ANCHOR_BODY_NAME, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=ANCHOR_BODY_NAME, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  # -- 动作层:裁剪驱动到 16 走步关节 ---------------------------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = QUAD_27DOF_WALK_JOINT_NAMES
  joint_pos_action.scale = QUAD_27DOF_ACTION_SCALE

  # -- 参考运动命令:占位四足参考 + 四腿 body/关节布局 -------------------------
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(QUAD_27DOF_MOTION_FILE)
  motion_cmd.anchor_body_name = ANCHOR_BODY_NAME
  motion_cmd.body_names = pikachu_27dof_quad_walk_bodies()
  motion_cmd.joint_names = QUAD_27DOF_WALK_JOINT_NAMES
  # 跳过参考运动开头几帧(录制起步常夹杂沉地/未落定),避免 reset 一上来就被
  # contact 弹开。见 commands.py MotionCommandCfg.start_frame。
  motion_cmd.start_frame = 4

  # -- 事件 / 奖励 / 终止 ------------------------------------------------------
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = QUAD_27DOF_SUPPORT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (ANCHOR_BODY_NAME,)

  # joint limit 只限走步 16 关节(固定关节由 pose 锁住)。
  cfg.rewards["joint_limit"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=QUAD_27DOF_WALK_JOINT_NAMES
  )

  # 功率限制:惩罚 16 个走步关节的机械注入功率之和 Σ max(0, τ_actuator * qd) (W)。
  # electrical_power_cost 按关节名解析 joint_ids,用 qfrc_actuator 取每关节执行器广义力,
  # 不依赖 actuator_ids/joint_ids 平行对齐,对 Pikachu 正确。仅惩罚正功率(不罚再生制动),
  # 既限转发力/高频摆臂,又不和大尺度跟踪奖励抢权重。
  # reset 沉降期实测均值≈84W,权 -2e-3 使峰值惩罚≈-0.17(≈跟踪奖励的 4%),温和压功率;
  # 训练可 `--env.rewards.power.weight` 收紧/放宽。
  cfg.rewards["power"] = RewardTermCfg(
    func=mdp.electrical_power_cost,
    weight=-2e-3,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=QUAD_27DOF_WALK_JOINT_NAMES)
    },
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = EE_BODY_NAMES
  cfg.viewer.body_name = ANCHOR_BODY_NAME

  add_pikachu_telemetry_metrics(cfg)
  # 27dof 无 foot site,移除依赖它的脚滑遥测。
  cfg.metrics.pop("foot_slip_velocity_m_s", None)
  cfg.metrics.pop("foot_slide_fraction", None)

  # -- 评估(play)模式:无限 episode、关随机化 ---------------------------------
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
