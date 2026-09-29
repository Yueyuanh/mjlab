from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .multi_motion import MultiMotionCommand


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def motion_reference(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """参考动作在该时刻的关节位置+速度 ``(n_joints*2,)``。

  与 ``MotionCommand.command`` 的区别:后者在多动作/速度命令场景下被重写成
  速度命令(见 ``velocity_match.VelocityMatchMotionCommand``),而本函数**始终**
  返回参考本身。蒸馏时教师的观测组必须用它 —— 教师训练时这一项就是参考的
  ``joint_pos``+``joint_vel``,维度与顺序必须逐位对齐,否则教师的观测归一化
  统计量会对不上。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return torch.cat([command.joint_pos, command.joint_vel], dim=1)


def motion_gait_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """参考步态的相位 ``(cos, sin)``,一个周期走完 2π。

  只给学生用。学生观测里没有参考动作/位姿,而它要模仿的教师是相位锁定的,
  缺了这个节拍就只能靠自激振荡,实测会塌缩成单方向行为。静态 clip 输出常数。
  """
  command = cast("MultiMotionCommand", env.command_manager.get_term(command_name))
  return command.gait_phase_obs
