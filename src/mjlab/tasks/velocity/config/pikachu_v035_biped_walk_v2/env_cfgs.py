"""V035 直立行走 v2：在 walk 档基础上专治「前进时偏航漂移」。

v1（``pikachu_v035_biped_walk``）跑到 2000 轮时确定性评估已经能走：
前进命令 0.2 实测 0.204 m/s（MAE 0.012）、后退 −0.12 实测 −0.135，
0 摔倒。但同一组评估里偏航是失控的：

  * 前进档实测 wz = +0.269 rad/s 而命令是 0，8 s 会拐出约 120°；
  * 站立档 wz = +0.159 rad/s，站着也在缓慢自转；
  * 1000 轮时这两项分别是 +0.190 / +0.143，即随训练变差。

根因在跟踪奖励的尺度：``track_angular_velocity`` 默认 ``std=sqrt(0.5)=0.707``，
而本任务角速度命令只有 ±0.35 rad/s。0.27 rad/s 的漂移仍能拿
``exp(-0.073/0.5)=0.86``，几乎等于满分，于是「边走边转」没有梯度去纠正。
线性跟踪同理：默认 ``std=sqrt(0.25)=0.5`` 对 ±0.2~0.35 的命令过宽。

本档只改三处，其余（机器人、动作/观测、课程、推扰、奖励族）全部继承 v1：

  * ``track_angular_velocity``: std 0.707 → 0.35，权重 2.0 → 2.5；
  * ``track_linear_velocity``: std 0.50 → 0.30；
  * ``body_ang_vel``: 权重 −0.05 → −0.12。

权重的预算参照 ``pikachu_v034_biped_knee`` 的教训：那里
``straight_base_roll_rate`` 用 −0.60 时单步罚 4.8 压过了 3.5 的进度奖励，
70 轮就把行走策略打回站桩；−0.15 才安全。这里 std=0.35 时，
0.27 rad/s 漂移损失 ``(1-exp(-0.073/0.1225))*2.5 ≈ 1.1`` 每步，相对
进度奖励约 3.5 仍是三成量级，不会反客为主。
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.pikachu_v035_biped_walk.env_cfgs import (
  v035_biped_walk_env_cfg,
)


def v035_biped_walk_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = v035_biped_walk_env_cfg(play=play)
  cfg.rewards["track_angular_velocity"].params["std"] = 0.35
  cfg.rewards["track_angular_velocity"].weight = 2.5
  cfg.rewards["track_linear_velocity"].params["std"] = 0.30
  cfg.rewards["body_ang_vel"].weight = -0.12
  return cfg
