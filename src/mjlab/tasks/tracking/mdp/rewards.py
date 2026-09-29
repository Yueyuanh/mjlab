from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def track_command_linear_velocity(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """机器人根速度(世界系)对**命令**的跟踪 —— 与参考无关,因此**无相位噪声**。

  为什么必须有这一项(而不是只靠 ``motion_global_anchor_position_error_exp``):
  位置误差是速度误差的**积分**,而 ``exp(-d²/σ²)`` 是饱和函数 —— 学生一旦落后
  0.5 m 以上,该项就掉到 0.05 以下,**梯度趋近于零,抬权重也没用**(实测:
  σ=0.3 时学生值 0.049,权重从 0.5 抬到 3.0 只是把 0.049 乘 3)。

  而命令速度是**常量**(每个 episode 固定,见 ``VelocityMatchMotionCommand``),
  与步态相位无关。静止机器人在 0.19 m/s 的命令下误差恒为 0.19,σ=0.25 时:
  值 ``exp(-0.19²/0.0625)=0.56``,而**在静止点的梯度达 2×0.19/0.0625×0.56=3.4
  每 (m/s)** —— 比位置项在 σ=0.5 下的 1.4 大一倍多,且没有相位噪声把它吃掉。

  **注意比的是世界系** ``root_link_lin_vel_w``:本库 npz 的 ``command`` 存的是
  世界系稳态均速(逐条核对过,如 ``0.185_0_0.2`` 命令 [0.091,0.135,0.196] 对
  世界系实测 [0.0909,0.1348]),而 mjlab 速度任务的 ``track_linear_velocity``
  用的是体系 ``root_link_lin_vel_b`` —— 对有转向的 clip 两者不等价,不能照搬。

  z 分量按惯例要求为 0(命令不含 vz)。
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = env.scene["robot"].data.root_link_lin_vel_w
  error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  error = error + torch.square(actual[:, 2])
  return torch.exp(-error / std**2)


def track_command_angular_velocity(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """机器人偏航角速度(世界系 z)对**命令** wz 的跟踪。理由同上。

  静止机器人在 wz=0.366 的原地转命令下误差恒为 0.366,σ=0.4 时值 0.43、
  梯度 2×0.366/0.16×0.43=1.97 每 (rad/s)。xy 分量要求为 0(机器人保持直立)。
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = env.scene["robot"].data.root_link_ang_vel_w
  error = torch.square(command[:, 2] - actual[:, 2])
  error = error + torch.sum(torch.square(actual[:, :2]), dim=1)
  return torch.exp(-error / std**2)
