"""速度追踪遥测指标：命令 vs 实际 vs 误差。

**为什么单独一个文件**：``scripts/plot_logs/metrics_csv_dashboard.html`` 的
``PREFIXES`` 只认关节/基座那几族列（``joint_position_rad/`` … ``joint_power_w/``），
「命令速度跟得准不准」这条主线它**没有**。而速度任务的核心体检恰恰是这一条
（报告 §3.4 已经证明 ``Episode_Reward/track_*`` 因为 Jensen 不等式不能当进度
指标用）。所以这里把命令/实际/误差三组逐分量补进 metrics，让
``record_metrics_cli`` 写出的 CSV 自带这条主线。

**列名刻意用 ``vel/`` 前缀**：不匹配 dashboard 的任何 ``prefix``，因此它加载
这些 CSV 时会把这几列原样忽略（``for name of headers: if(!name.startsWith(prefix)) continue``），
不会干扰关节分类，也不会撞上 ``joint_*`` 的左/右侧识别。

**坐标系**：命令与实测都在 **base 系**（``vel_command_b`` / ``root_link_lin_vel_b``），
与 ``UniformVelocityCommand._update_metrics`` 的口径逐字一致 —— 误差可以直接相减，
不需要任何旋转。这一点必须保持：heading_command 打开时命令第 3 维是
「朝向误差 × 刚度」压出来的角速度，只有放在 base 系才和 ``ang_vel_b[:, 2]`` 可比。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# (轴名, 分量下标, 单位)。0/1 = base 系线速度 x/y，2 = base 系偏航角速度。
_AXES: tuple[tuple[str, int, str], ...] = (
  ("x", 0, "m/s"),
  ("y", 1, "m/s"),
  ("yaw", 2, "rad/s"),
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _command_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """命令张量 [B, 3]（base 系）。缺失时抛错而不是静默给 0。"""
  command = env.command_manager.get_command(command_name)
  if command is None:
    raise ValueError(
      f"metrics 需要命令项 {command_name!r}，但 command_manager 里没有。"
      f" 现有：{list(env.command_manager.active_terms)}"
    )
  return command


def command_velocity(
  env: ManagerBasedRlEnv, command_name: str = "twist", axis: int = 0
) -> torch.Tensor:
  """命令速度的单个分量 [B]。"""
  return _command_b(env, command_name)[:, axis]


def base_velocity(
  env: ManagerBasedRlEnv,
  axis: int = 0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """实测基座速度的单个分量 [B]（base 系）。

  ``axis`` 0/1 取 ``root_link_lin_vel_b``，2 取 ``root_link_ang_vel_b`` ——
  与命令的第 3 维（偏航角速度）对齐。
  """
  asset = env.scene[asset_cfg.name]
  if axis < 2:
    return asset.data.root_link_lin_vel_b[:, axis]
  return asset.data.root_link_ang_vel_b[:, axis - 2]


def velocity_error(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  axis: int = 0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """命令 − 实测，**带符号** [B]。

  刻意不取绝对值：符号里带着「追不上（负）/冲过头（正）」，而均值归零的
  绝对值会把这个信息抹掉。要看幅度在分析脚本里对 CSV 取 abs 即可。
  """
  return command_velocity(env, command_name, axis) - base_velocity(env, axis, asset_cfg)


def velocity_error_norm(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """平面线速度误差的模 ‖cmd_xy − act_xy‖ [B]，与 ``error_vel_xy`` 同口径。"""
  actual_xy = torch.stack(
    [base_velocity(env, 0, asset_cfg), base_velocity(env, 1, asset_cfg)], dim=1
  )
  return torch.norm(_command_b(env, command_name)[:, :2] - actual_xy, dim=1)


def base_position_w(
  env: ManagerBasedRlEnv,
  axis: int = 0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """基座在**世界系**的位置分量 [B]（``axis`` 0=x, 1=y）。

  为什么要它：本项目 §3.4 的教训是「``Episode_Reward/track_*`` 不能当进度指标，
  判走没走必须用独立于奖励的物理量」。base 系速度积分出来的是「相对身体的里程」，
  只有直线档才等于净位移；世界系坐标首尾相减是**无条件**成立的净位移。

  ⚠️ 但世界系净位移**只在机器人不转身时**才等于 base 系前进 —— 未受命令的 yaw
  可以跑到 0.3 rad/s，8 s 就是 149°，位移方向会随之外偏。所以必须同时看
  :func:`base_yaw_w`，判「世界位移能不能当净位移用」。
  """
  return env.scene[asset_cfg.name].data.root_link_pos_w[:, axis]


def base_yaw_w(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """基座世界系偏航角 [B]（rad，单帧值域 ±π）。

  用 wxyz 四元数的标准 yaw 公式直接算，不引入 ``rpy`` 依赖。
  CSV 里逐帧读出来后**要在分析脚本里 unwrap**（相邻帧差分取 ``atan2(sin, cos)``
  再累加），否则 ±π 跳变会污染「净转角」。
  """
  q = env.scene[asset_cfg.name].data.root_link_quat_w  # wxyz
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def is_standing_env(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  threshold: float = 0.1,
) -> torch.Tensor:
  """该 env 当前是否「命令≈0」[B]（1.0/0.0）。

  与 ``feet_lateral_distance`` 的命令门控**同一判据**（``‖cmd_xy‖ + |cmd_yaw|``
  < ``threshold``），所以 CSV 里的这一列就是门控的逐帧快照 —— 站姿开度指标
  该不该被这一帧计入，看它即可。
  """
  command = _command_b(env, command_name)
  total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return (total < threshold).float()


def add_velocity_tracking_metrics(
  cfg: ManagerBasedRlEnvCfg,
  command_name: str = "twist",
) -> None:
  """把「命令 / 实际 / 误差」三组逐分量指标挂到 ``cfg.metrics``。

  列名（``vel/`` 前缀，dashboard 会忽略，分析脚本会读）::

      vel/cmd_x  vel/cmd_y  vel/cmd_yaw     命令（base 系）
      vel/act_x  vel/act_y  vel/act_yaw     实测（base 系）
      vel/err_x  vel/err_y  vel/err_yaw     带符号误差 = cmd − act
      vel/err_xy_norm                       平面线速度误差模
      vel/pos_x_w  vel/pos_y_w              世界系基座坐标（首尾相减 = 净位移）
      vel/standing                          命令≈0 标志（1/0），即门控快照
  """
  for axis_name, axis, _unit in _AXES:
    cfg.metrics[f"vel/cmd_{axis_name}"] = MetricsTermCfg(
      func=command_velocity,
      params={"command_name": command_name, "axis": axis},
      reduce="mean",
    )
    cfg.metrics[f"vel/act_{axis_name}"] = MetricsTermCfg(
      func=base_velocity,
      params={"axis": axis},
      reduce="mean",
    )
    cfg.metrics[f"vel/err_{axis_name}"] = MetricsTermCfg(
      func=velocity_error,
      params={"command_name": command_name, "axis": axis},
      reduce="mean",
    )

  cfg.metrics["vel/err_xy_norm"] = MetricsTermCfg(
    func=velocity_error_norm,
    params={"command_name": command_name},
    reduce="mean",
  )
  # 世界系坐标用 "last"：episode 摘要报的是终点位置（首尾相减即净位移）；
  # 逐帧 CSV 拿到的仍是当帧真实坐标，与 reduce 无关。
  for axis_name, axis in (("x", 0), ("y", 1)):
    cfg.metrics[f"vel/pos_{axis_name}_w"] = MetricsTermCfg(
      func=base_position_w,
      params={"axis": axis},
      reduce="last",
    )
  cfg.metrics["vel/yaw_w"] = MetricsTermCfg(func=base_yaw_w, reduce="last")
  cfg.metrics["vel/standing"] = MetricsTermCfg(
    func=is_standing_env,
    params={"command_name": command_name},
    reduce="mean",
  )
