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
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.config.pikachu_lite.env_cfgs import (
  ANCHOR_BODY_NAME,
  LEG_BODY_NAMES,
  pikachu_lite_env_cfg,
)
from mjlab.tasks.tracking.mdp.velocity_match import VelocityMatchMotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# 学生可部署观测的维度:3 命令 + 2 步态相位 + 3 角速度 + 3 投影重力
# + 10 + 10 关节 + 10 动作。
STUDENT_OBS_DIM = 41
# 教师 actor 组维度(与训练日志中 Actor Model 的 in_features 一致)。
TEACHER_OBS_DIM = 65


def _student_actor_terms() -> dict[str, ObservationTermCfg]:
  """学生策略的观测:真机可得的 39 维。"""
  return {
    # 唯一的任务指令。``VelocityMatchMotionCommand.command`` 返回 (vx, vy, wz)。
    "velocity_command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    # 参考步态的节拍 (cos, sin)。**这条是学生能不能动起来的关键** ——
    # 教师是相位锁定的(它的观测里有整段参考),而学生原本既没有参考动作也
    # 没有相位,只能靠关节角自激振荡维持步态。实测后果是塌缩成"只会往一个
    # 方向转":转向档 vel_err 0.05(能转),后退档 0.19(完全不动),横向
    # 0.09~0.13,而且同一 ckpt 换一个质心偏移就从 +0.166 m/s 掉到 0.000。
    # 补上节拍后学生才有能力复现教师的相位锁定步态。
    "gait_phase": ObservationTermCfg(
      func=mdp.motion_gait_phase, params={"command_name": "motion"}
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
    # 收窄命令抖动:默认 1.0 会让命令落在 Voronoi 格任意位置,格边缘处
    # "命令速度 != 该 clip 实测速度",于是 BC(朝 clip)与奖励(朝参考)互相拉扯。
    # 0.3 让二者基本一致,同时保留对非格点命令的插值能力。
    command_jitter=0.3,
  )
  cfg.commands["motion"] = student_cmd

  # -- 奖励:让"真的产生位移 / 真的转向"有梯度 ------------------------------
  # 教师档继承来的 6 项 motion_* 奖励里,权重最大的两项(motion_body_pos /
  # motion_body_ori,各 1.0)**不要求位移**:它们比较的 body_pos_relative_w 在
  # ``commands.py:463-471`` 被重锚定到机器人当前 anchor,xy 漂移被完全抹掉。
  # 名义上管速度的 motion_body_lin_vel σ=1.0 m/s,对本库 0.19 m/s 的速度量级
  # 等于常数(站着不动 exp(-0.19²/1²)=0.965)。于是唯一真正要求世界系位移的
  # ``motion_global_root_pos`` 就成了全部压力所在。
  #
  # **实测:旧学生(``pikachu_lite_distill/2026-09-29_07-45-13``)到底差在哪**
  #
  #   | 档位 | 条数 | 学生 anchor_mae | 教师 anchor_mae | 倍数 |
  #   |---|---|---|---|---|
  #   | \|v\|<0.05(含原地转) | 9 | 0.079 | 0.079 | **1.0x** |
  #   | \|v\|>=0.05(有位移) | 16 | 0.520 | 0.087 | **6.0x** |
  #
  # 零速档**一模一样**,有位移档实速全是 ~0.000;而 ``joint_mae`` 只差 5%
  # (0.542 vs 0.515)—— 腿的轨迹是对的,**就是原地踏步**。原因在观测:学生 39 维
  # 里没有位置、没有相位、没有线速度,参考帧又按墙钟 ``time_steps += 1``
  # (``commands.py:489-493``)推进,学生看不见相位,只能靠自身关节角自激振荡维持
  # 步态。**关节空间的"走"与"原地摆腿"是同一条轨迹**,唯一能打破对称性的就是奖励。
  #
  # **所以问题不是权重不够,是核宽 σ 太窄 —— 学生已经落在饱和区**
  #
  # 有位移档上学生平均位置误差 0.52 m、教师 0.087 m。σ=0.3 时:
  #
  #   | σ | 教师值 | 学生值 | 学生处梯度 | 站着不动 |
  #   |---|---|---|---|---|
  #   | **0.3(原)** | 0.920 | **0.049** | 0.571 | 0.140 |
  #   | **0.5(改)** | 0.970 | **0.338** | **1.409** | 0.233 |
  #   | 0.6 | 0.979 | 0.471 | 1.363 | 0.280 |
  #   | 1.0 | 0.993 | 0.763 | 0.794 | 0.463 |
  #
  # ``exp(-d²/σ²)`` 在 σ=0.3、d=0.52 处已经掉到 0.049,**权重从 0.5 抬到 3.0
  # 只是把 0.049 乘 3,梯度一分没多**(3.0×0.049=0.148,仍接近 0)。σ 放宽到 0.5
  # 把学生处的梯度拉到 1.409(2.5 倍,且是全表最大),同时教师 0.970、"站着不动"
  # 0.233 都保住 —— 教师与站桩的差距 0.737,加权 3.0 后 2.21 分,远大于其余项
  # 合计的 0.229 分差,足以把策略推离原地踏步。
  #
  # σ 不能更大:到 1.0 时学生值 0.763、与教师只差 0.230,区分度被抹平。
  #
  # **朝向项同理**。原地转的 clip(``0_0_0.2`` 命令 wz=0.366)学生 vel_err 0.412,
  # 即基本没转;10 s 累积朝向误差 ~3.7 rad,σ=0.4 时该项同样饱和。按同样的比例
  # 放宽到 0.7(教师 0.939 / 学生 0.194 / 站桩 0.194),权重 0.5 -> 3.0。
  #
  # **为什么不用"瞬时速度"项**:曾按 base_link 世界系速度加过两项(σ=0.2/0.5),
  # 用 ``scripts/probe_reward_gap.py`` 实测教师 0.607 / 学生 0.519(只差 0.088),
  # 角速度那项甚至反向。原因是**相位**:参考速度锁在步态相位上,逐帧 std 达
  # 0.117 m/s(机器人只有 0.025~0.104),教师的均速完全正确、瞬时误差仍有
  # 0.144 m/s。逐帧比速度等于在考"相位对齐"而不是"有没有位移"。位置/朝向误差
  # 天然是速度误差的积分,不受相位干扰。
  #
  # 只改学生档:教师职责是产出可被 BC 模仿的参考步态,它已做到 vel_err=0.0144;
  # 改教师的目标函数等于同时移动 BC 的目标。
  cfg.rewards["motion_global_root_pos"].weight = 3.0
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.5
  cfg.rewards["motion_global_root_ori"].weight = 3.0
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.7
  # 位置/朝向项管"长期别漂",但它们在学生已落后时是饱和的;真正把机器人从
  # "站着不动"推起来的是下面两项**命令速度跟踪** —— 见 ``rewards.py`` 里
  # ``track_command_linear_velocity`` 的推导。命令是常量,没有相位噪声。
  cfg.rewards["track_lin_vel"] = RewardTermCfg(
    func=mdp.track_command_linear_velocity,
    weight=2.0,
    params={"command_name": "motion", "std": 0.25},
  )
  # 角速度项权重压到 1.0、σ 放到 0.6:步态本身的 yaw 摆动(std ~0.3 rad/s)
  # 与库里的转向命令(0.18~0.37 rad/s)同量级,σ 收紧会变成"走路就扣分"。
  # 转向的位移压力主要由 ``motion_global_root_ori`` 承担 —— 那是朝向误差的
  # 积分,无相位噪声:原地不转 10 s 累积 3.7 rad,σ=0.7 下该项直接归零。
  cfg.rewards["track_ang_vel"] = RewardTermCfg(
    func=mdp.track_command_angular_velocity,
    weight=1.0,
    params={"command_name": "motion", "std": 0.6},
  )

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
