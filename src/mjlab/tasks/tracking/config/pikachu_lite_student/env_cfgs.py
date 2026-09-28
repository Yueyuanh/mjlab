"""Pikachu lite **学生**(速度命令驱动)的 tracking/蒸馏环境配置。

与教师档 ``pikachu_lite`` 的差异只在**命令项**与**观测组**:

教师(``MultiMotionCommand``)
  每个 env 随机抽一条 clip 跟踪;``actor`` 观测 = 完整参考(anchor 位姿 + 参考
  关节角/角速度 + 本体感受),共 65 维。它见过整库,但**不可部署** —— 观测里
  带参考动作,真机上拿不到。

学生(``VelocityMatchMotionCommand``)
  ``actor`` 观测 = 速度命令 ``(vx, vy, wz)`` + 本体感受,共 **39 维**,全部是
  真机可得量。命令项在参考库里做最近邻匹配,自动把该 env 的参考切到对应 clip
  —— 这就是"发送速度控制命令自动匹配模型和数据"。

蒸馏通过 ``teacher`` 观测组实现:该组逐项复刻教师 ``actor`` 组的 65 维
(``command`` 换成语义相同的 ``motion_reference``,关掉噪声),让冻结教师能在
学生的状态上给出动作,作为 BC 目标(见 ``mjlab.rl.dagger_ppo``)。

三组观测的分工::

    actor   (39)  可部署  速度命令 + 本体感受            -> 学生策略
    teacher (65)  仅训练  复刻教师 actor 组              -> 冻结教师前向
    critic  (128) 仅训练  特权信息(参考 body 位姿等)   -> 价值函数
"""

from __future__ import annotations

from mjlab.asset_zoo.robots.pikachu.pikachu_constants_base import LEG_JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.config.pikachu_lite.env_cfgs import (
  ANCHOR_BODY_NAME,
  LEG_BODY_NAMES,
  pikachu_lite_env_cfg,
)
from mjlab.tasks.tracking.mdp.velocity_match import VelocityMatchMotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# 学生可部署观测的维度:3 命令 + 3 角速度 + 3 投影重力 + 10 + 10 关节 + 10 动作。
STUDENT_OBS_DIM = 39
# 教师 actor 组维度(与训练日志中 Actor Model 的 in_features 一致)。
TEACHER_OBS_DIM = 65


def _student_actor_terms() -> dict[str, ObservationTermCfg]:
  """学生策略的观测:真机可得的 39 维。"""
  return {
    # 唯一的任务指令。``VelocityMatchMotionCommand.command`` 返回 (vx, vy, wz)。
    "velocity_command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }


def _teacher_terms() -> dict[str, ObservationTermCfg]:
  """逐项复刻教师 ``actor`` 组(65 维),只把 ``command`` 换成 ``motion_reference``。

  教师的 ``command`` 项就是 ``cat([joint_pos, joint_vel])``(20 维),而学生 env
  的命令项返回 3 维速度命令,所以这里必须改用 ``motion_reference`` 才能凑出
  同构的 65 维。噪声全部关掉:教师部署/回放时也是干净观测,且 BC 目标应当确定。
  """
  return {
    "command": ObservationTermCfg(
      func=mdp.motion_reference, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"biased": True}),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }


def _critic_terms() -> dict[str, ObservationTermCfg]:
  """特权价值观测(128 维):与教师 ``critic`` 组逐项同构。

  唯一改动同样是把 ``command`` 换成语义相同的 ``motion_reference`` —— 学生 env 的
  命令项返回 3 维速度命令,直接用会让 ``command`` 项从 20 维缩到 3 维,价值函数的
  输入分布跟着变。换成 ``motion_reference`` 后 critic 与教师完全一致(128 维),
  参考关节轨迹这一路信息也不丢。
  """
  return {
    "command": ObservationTermCfg(
      func=mdp.motion_reference, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }


def pikachu_lite_student_env_cfg(
  play: bool = False, start_clip_id: int = 0
) -> ManagerBasedRlEnvCfg:
  """Create the Pikachu lite velocity-command (student) configuration.

  Args:
    play: 回放模式(关噪声/关推力/固定 clip 起点)。
    start_clip_id: ``play`` 时固定跟踪第几条 clip(录像用)。
  """
  cfg = pikachu_lite_env_cfg(play=play)

  # -- 命令项:速度命令 -> 最近邻参考 clip ------------------------------------
  base_cmd = cfg.commands["motion"]
  assert isinstance(base_cmd, VelocityMatchMotionCommandCfg) or hasattr(
    base_cmd, "motion_files"
  )
  motion_files = base_cmd.motion_files  # type: ignore[attr-defined]
  assert motion_files, "student env needs the multi-clip motion library"

  student_cmd = VelocityMatchMotionCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=True,
    pose_range=dict(base_cmd.pose_range),
    velocity_range=dict(base_cmd.velocity_range),
    joint_position_range=base_cmd.joint_position_range,
    motion_file=motion_files[0],
    motion_files=motion_files,
    anchor_body_name=ANCHOR_BODY_NAME,
    body_names=LEG_BODY_NAMES,
    joint_names=LEG_JOINT_NAMES,
    # 蒸馏要求**均匀技能采样**(PHP Module 3):25 条 clip 等概率,
    # 不用 adaptive —— 否则采样分布会朝少数难帧塌缩,BC 覆盖面变窄。
    sampling_mode="uniform",
    start_clip_id=start_clip_id,
  )
  cfg.commands["motion"] = student_cmd

  # -- 观测组:actor(学生) / teacher(冻结教师) / critic(价值) ---------------
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=_student_actor_terms(),
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "teacher": ObservationGroupCfg(
      terms=_teacher_terms(), concatenate_terms=True, enable_corruption=False
    ),
    "critic": ObservationGroupCfg(
      terms=_critic_terms(), concatenate_terms=True, enable_corruption=False
    ),
  }

  # -- 只裁腿关节(三个组的 joint_pos / joint_vel 都要裁到 10 腿) --------------
  leg_obs_cfg = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
  for group in cfg.observations.values():
    group.terms["joint_pos"].params["asset_cfg"] = leg_obs_cfg
    group.terms["joint_vel"].params["asset_cfg"] = leg_obs_cfg

  cfg.viewer.body_name = ANCHOR_BODY_NAME

  if play:
    # 回放:干净观测 + 固定 clip + 固定帧起点(与教师档一致)。
    cfg.observations["actor"].enable_corruption = False
    student_cmd.sampling_mode = "start"
    student_cmd.pose_range = {}
    student_cmd.velocity_range = {}

  return cfg


__all__ = [
  "STUDENT_OBS_DIM",
  "TEACHER_OBS_DIM",
  "pikachu_lite_student_env_cfg",
]
