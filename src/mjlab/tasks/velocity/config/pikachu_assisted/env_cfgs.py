"""Pikachu velocity「学步车」辅助任务环境配置。

基于 ``pikachu_flat_env_cfg``（velocity Pikachu Flat 14dof，已训通能走的最干净
基线）衍生，只加两样东西：

  1. **z 向限位滑轨**：把机器人根部的 freejoint 换成 ``add_z_rail`` 生成的
     ``slide x → slide y → slide z(limited) → ball`` 关节链，把 base 硬性夹在
     ``[z_low, z_high]`` 内。**真关节约束** —— 不施加任何外力，也不增加任何
     碰撞体。x/y/朝向完全自由，track_linear / track_angular 奖励照常可学。
  2. **奖励增强**：适度提高 ``upright`` 权重，引导在滑轨保护下维持竖直。

滑轨在 train 与 play 下都启用 —— 用户需要在 viewer 里看到它（``base_rail``
的 ``debug_vis`` 画上下限两个盘）。要跑「撤掉学步车」的迁移验证，把区间放到
够不着即可：

    PIKACHU_RAIL_Z_LOW=0.0 PIKACHU_RAIL_Z_HIGH=10.0 uv run play ...

不动共享 ``velocity/mdp``、不动现有 ``pikachu`` velocity 任务。
"""

from __future__ import annotations

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.velocity.config.pikachu.env_cfgs import pikachu_flat_env_cfg
from mjlab.utils.spec import add_z_rail

from .assisted_mdp import base_rail

# 滑轨区间(m，世界 z)。围绕 velocity 站高 ~0.19 取一个宽区间，给蹲起/迈步留足
# 余量，只在真要沉下去时才兜住。**z_low 必须低于 reset 高度(~0.19)**，否则每个
# episode 都从"顶在下限上"的穿透瞬态开始。
RAIL_Z_LOW = 0.11
RAIL_Z_HIGH = 0.27

# upright 奖励增强权重（基线为 1.0）。滑轨负责"别沉下去"，upright 负责"教它
# 站直"，两者正交：滑轨不提供任何姿态力矩，所以不存在靠约束躺赢的可能。
UPRIGHT_WEIGHT = 1.5


def _rail_height(env_var: str, default: float) -> float:
  """读滑轨高度，允许环境变量覆盖（用于扫描实验）。

  为什么不做成 CLI 参数：``spec_fn`` 在**任务注册时**就闭包捕获了高度，而 CLI
  覆盖落在 ``load_env_cfg`` 返回的深拷贝上（见 registry.py），闭包看不到副本
  —— 写成 ``--env.events.base_rail.params.z_low`` 只会改到可视化，改不动已经
  编译进模型的关节限位，静默跑出错误的实验。所以参数只能从注册时就存在的
  进程级输入读，环境变量是唯一不重编代码的入口。
  """
  raw = os.environ.get(env_var)
  return default if raw is None else float(raw)


def pikachu_assisted_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """创建 Pikachu「学步车」辅助 velocity 配置（训练 / play）。"""
  cfg = pikachu_flat_env_cfg(play=play)

  z_low = _rail_height("PIKACHU_RAIL_Z_LOW", RAIL_Z_LOW)
  z_high = _rail_height("PIKACHU_RAIL_Z_HIGH", RAIL_Z_HIGH)

  # 滑轨替换的是**实体根部**的 freejoint，所以必须包 EntityCfg.spec_fn
  # （场景层改不动已建好的实体 spec）。
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.spec_fn = add_z_rail(robot_cfg.spec_fn, z_low, z_high)

  # 运行时 term：本身不施力也不碰撞，只承载 debug_vis（viewer 里画上下限）。
  cfg.events["base_rail"] = EventTermCfg(
    func=base_rail,
    mode="reset",
    params={"z_low": z_low, "z_high": z_high},
  )

  cfg.rewards["upright"].weight = UPRIGHT_WEIGHT

  return cfg
