"""Pikachu velocity「学步车」硬约束 —— base 的 z 向限位滑轨（真关节约束）。

机制
====
把机器人根部的 freejoint 换成一根**上下限位的滑轨**。滑轨由 ``add_z_rail``
（``mjlab.utils.spec``）在实体 spec 里插一条关节链，自 world 往下::

  slide x → slide y → slide z(limited: z_low..z_high) → ball → base_link

这条链与它替换掉的 freejoint **逐位等价**：同样 7 qpos / 6 dof，qpos 顺序
``x, y, z, qw, qx, qy, qz``，dof 顺序线速度（世界系）在前、角速度（体系）在后。
所以下游的索引、观测、奖励、传感器看到的东西与原来一模一样；唯一区别是 z 被
关节限位硬性夹在 ``[z_low, z_high]`` 内。

为什么必须是关节链：freejoint 本身不接受 ``limited``/``range``（schema 直接拒绝
"unrecognized attribute"），而一个 body 也不能同时挂 freejoint 和 slide
（"more than 6 dofs in body"）。滑轨是 MuJoCo 原生唯一能给浮动基座限高的做法。

约束语义
========
- **不施加任何外力**：约束由求解器的关节限位约束力实现。base 在区间内时该力
  恒为 0，不是弹簧，也就不存在"增益/参考高度不匹配 → 发散 NaN"。
- **不增加任何碰撞体**：滑轨链上的 body 没有 geom，不参与任何接触。足-地接触、
  raycast 传感器、碰撞掩码全部不受影响。
- **只锁 z**：x/y 平移与三轴转动完全自由 —— base 仍可被驱动到任意水平位置，
  仍可倾倒。竖直能力由 ``upright`` 奖励（soft）负责教，滑轨只负责"别沉下去"。

取值约束：``z_low`` 必须**低于 reset 高度**（``init_state.pos[2]``，约 0.19 m）。
高于它会让每个 episode 都从"顶在下限上"的穿透瞬态开始。

可视化
======
滑轨是纯关节限位，没有 geom 可看。``base_rail.debug_vis`` 在 viewer 里画两个
半透明圆盘（红=下限，青=上限）标出 z 区间 —— **纯视觉元素，不带任何碰撞属性**。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mjlab.utils.spec import RAIL_Z_INDEX

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# 自检容差：滑轨限位在编译期写死，参数只用于可视化与自检，两者必须一致。
RANGE_TOL = 1e-6
# 可视化圆盘半径 / 厚度(m)。
DISC_RADIUS = 0.25
DISC_THICKNESS = 0.004


class base_rail:
  """滑轨的运行时载体：**不施加任何力、不参与碰撞**，只承载可视化与自检。

  约束本身是 ``add_z_rail`` 写进 spec 的关节限位，在物理层实现，本 term 的
  ``__call__`` 是空操作。之所以仍注册成事件 term，是因为 viewer 只从
  ``EventManager``（及 sensor/command）收集 ``debug_vis``，而这里需要在界面上
  把上下限标出来。
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._env = env
    self._z_low = float(cfg.params["z_low"])
    self._z_high = float(cfg.params["z_high"])
    self._verify_rail(env)

  def _verify_rail(self, env: ManagerBasedRlEnv) -> None:
    """核对编译进模型的滑轨是 7 qpos / 6 dof，且 z 限位与本 term 参数一致。

    不一致说明改了 ``RAIL_Z_*`` 常量却没同步参数，直接报错而不是静默跑错实验。
    """
    indexing = env.scene["robot"].indexing
    n_q = len(indexing.free_joint_q_adr)
    n_v = len(indexing.free_joint_v_adr)
    if n_q != 7 or n_v != 6:
      raise ValueError(
        f"base_rail expects a rail root (7 qpos / 6 dof), got {n_q} / {n_v}. "
        "Did the robot EntityCfg.spec_fn get wrapped with add_z_rail?"
      )
    model = env.sim.mj_model
    z_jnt = int(model.dof_jntid[int(indexing.free_joint_v_adr[RAIL_Z_INDEX])])
    if not model.jnt_limited[z_jnt]:
      raise ValueError("rail z joint is not limited; add_z_rail did not run")
    got = (float(model.jnt_range[z_jnt][0]), float(model.jnt_range[z_jnt][1]))
    want = (self._z_low, self._z_high)
    if abs(got[0] - want[0]) > RANGE_TOL or abs(got[1] - want[1]) > RANGE_TOL:
      raise ValueError(
        f"rail z range {got} does not match the configured {want}; update the "
        "RAIL_Z_* constants in env_cfgs.py and this term's params together."
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: object = None,
    **kwargs,
  ) -> None:
    del env, env_ids, kwargs  # 约束在关节限位层，运行时无事可做。

  def reset(self, env_ids: object) -> None:
    del env_ids  # 无跨 episode 状态。

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    """在 base 的水平位置画 z_low（红盘）/ z_high（青盘），标出滑轨区间。"""
    env = self._env
    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return
    base_xy = env.scene["robot"].data.root_link_pos_w[:, :2].cpu().numpy()
    for i in env_indices:
      x, y = base_xy[i]
      for z, color, label in (
        (self._z_low, (0.9, 0.2, 0.2, 0.45), "rail z_low"),
        (self._z_high, (0.2, 0.8, 0.9, 0.45), "rail z_high"),
      ):
        start = np.asarray([x, y, z], dtype=np.float64)
        visualizer.add_cylinder(
          start=start,
          end=start + np.asarray([0.0, 0.0, DISC_THICKNESS]),
          radius=DISC_RADIUS,
          color=color,
          label=f"{label}={z:.3f}",
        )
