"""V035 直立行走 v3：用非饱和的偏航代价治偏航漂移（第二次尝试）。

v1 的缺陷：前进档 wz 漂移 +0.092 rad/s（8 秒拐 42°），转向只做到方向的 62%。

v1 的实际 std 是 **线性 0.20 / 角速度 0.40**（不是基类的 0.5 / 0.707），
所以 0.09 rad/s 的漂移仍能拿 ``exp(-(0.09/0.4)^2) = 0.95``，几乎没有纠正梯度。

v2 同时改了三处（角速度 std 0.35、角速度权重 2.5、线性 std **0.20→0.30 是放宽**、
``body_ang_vel`` 加倍），实测是**退步**：转向从 0.093 掉到 0.027、侧移偏航漂移
反而到 0.126，前进虽然漂移减半（0.092 → 0.051）却开始超速（0.240 vs 命令 0.20）。
三处同时动导致归因不干净，但现象很明确：**非纵向命令下策略退化成站桩**
（转向档支撑数 1.95 ≈ 双足着地，v1 是 1.17）。

本档把 v2 的三处改动**全部回退**，只加一项：

  * ``angular_velocity_huber``: ``angular_velocity_huber_error``，
    权重 −0.30、``delta = 0.25``。

理由：指数型跟踪奖励的梯度在误差超过几个 std 后基本归零，
而本任务的角速度命令只有 ±0.35。Huber 项在 ``delta`` 之外是**恒定**梯度，
既能拉住 0.09 rad/s 的漂移，也能把转向从 62% 往命令上顶。
这个组合在 v031 的 ``E3-YawHuber`` 上验证过（权重 −0.30、delta 0.25）。

单点干预的好处是归因干净：如果转向仍然退化，那就不是 std 尺度的问题。
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.pikachu_v035_biped_walk.env_cfgs import (
  v035_biped_walk_env_cfg,
)


def v035_biped_walk_v3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = v035_biped_walk_env_cfg(play=play)
  cfg.rewards["angular_velocity_huber"] = RewardTermCfg(
    func=mdp.angular_velocity_huber_error,
    weight=-0.30,
    params={"command_name": "twist", "delta": 0.25},
  )
  return cfg
