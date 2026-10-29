"""V035 直立行走 v5：提高角速度跟踪的收益，而不是加重误差的代价。

前两轮微调都失败，但失败方式一致，而 tensorboard 给出了原因：

v2（收紧 std + 加权惩罚）和 v3（单项偏航 Huber）都在**加重「跟踪不上」的
代价**。v3 的中间 checkpoint 还留下了塌陷轨迹——热启动后 250 轮转向仍有
0.046（31%）、侧移 0.064（106%，比 v1 还好），到 1000 轮转向只剩 1%、
侧移 5%，支撑数从 1.23 涨到 1.97（双足着地 = 站住了）。

同一窗口内 31 个奖励项**全部平台化**（Δ<0.01），训练奖励看不出任何异常。
根因是量级：``track_linear_velocity`` 权重 5.0、训练中实际贡献 3.72；
``track_angular_velocity`` 权重 2.0、实际只贡献 0.40——**角速度跟踪的份量
是线速度的十分之一**。在这个配比下，「把直线走好」的边际收益远大于「学会
转向」，站桩（wz≈0，在 yaw 命令下仍能拿 46% 的角速度奖励）就是最省事的解。
加重代价只会让它更快地放弃转向。

本档只改一处：``track_angular_velocity`` 权重 **2.0 → 6.0**，与线速度同量级。
其余奖励、命令分布、执行器、观测全部沿用 v1，训练超参也沿用 v1
（init_std 0.8 / entropy 0.005），这样与 v2、v3 的差异是干净的单点。
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.pikachu_v035_biped_walk.env_cfgs import (
  v035_biped_walk_env_cfg,
)


def v035_biped_walk_v5_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = v035_biped_walk_env_cfg(play=play)
  cfg.rewards["track_angular_velocity"].weight = 6.0
  return cfg
