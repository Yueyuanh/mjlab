"""V035 直立行走任务：去掉 v033 E1 的「安静站立」奖励族，改用 v031 T6 行走档。

背景：v035 Biped-E1 用 3000 轮训练后策略完全站桩（0.2 m/s 命令实测
≈0.0007 m/s），因为 E1 为安静站立加入了 5 项只在 |cmd|<0.05 生效的奖励、
25% 站桩样本、init_std=0.25 的低探索，以及只推「移动中」机器人的推扰。
这套组合把「站着不动」变成了稳定局部最优。

本任务保留 E1 的机器人、动作/观测、手臂软窗口与安全项，但：
  * 删除 standing_* 奖励族，站桩不再额外加分；
  * 站桩样本降到 10%，恢复默认的全员推扰；
  * 跟踪权重 3→5，Huber -0.25→-1.0，前进进度 2→3；
  * 动作平滑罚 -0.15→-0.05，避免抑制迈步；
  * 课程速度上限收到 (±0.5, 0.6) / (±0.35) / (±0.35) rad/s，保证可跟踪；
  * 探索回到 v031 E0 的 init_std=0.8 / entropy=0.005。
"""

from mjlab.envs import ManagerBasedRlEnvCfg, mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.velocity.config.pikachu_v035_biped_e1.env_cfgs import (
  v035_biped_e1_env_cfg,
)

_STANDING_REWARDS = (
  "standing_posture_l2",
  "standing_joint_velocity",
  "standing_base_motion",
  "standing_foot_motion",
  "standing_support",
)


def v035_biped_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = v035_biped_e1_env_cfg(play=play)

  # --- 站桩不再有任何额外奖励 ----------------------------------------------
  for name in _STANDING_REWARDS:
    cfg.rewards.pop(name, None)
  command = cfg.commands["twist"]
  command.rel_standing_envs = 0.10

  # --- 推扰对全员生效（E1 只推移动中的机器人，等于保护站桩） --------------
  cfg.events["push_robot"] = EventTermCfg(
    func=envs_mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={
      "velocity_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
      },
    },
  )

  # --- 奖励重配：把梯度压到「真的走起来」上 --------------------------------
  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["linear_velocity_huber"].weight = -1.0
  cfg.rewards["directional_progress"].weight = 3.0
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["short_step_air_time"].weight = -1.0

  # --- 课程：速度上限收到这台 0.23 m 机器人追得上的量级 --------------------
  if "command_vel" in cfg.curriculum:
    stages = cfg.curriculum["command_vel"].params["velocity_stages"]
    stages.clear()
    stages.extend(
      [
        {
          "step": 0,
          "lin_vel_x": (-0.30, 0.40),
          "lin_vel_y": (-0.20, 0.20),
          "ang_vel_z": (-0.25, 0.25),
        },
        {
          "step": 1500 * 24,
          "lin_vel_x": (-0.40, 0.55),
          "lin_vel_y": (-0.30, 0.30),
          "ang_vel_z": (-0.35, 0.35),
        },
      ]
    )
  if play:
    cfg.curriculum = {}
    command.ranges.lin_vel_x = (-0.4, 0.55)
    command.ranges.lin_vel_y = (-0.30, 0.30)
    command.ranges.ang_vel_z = (-0.35, 0.35)
  return cfg
