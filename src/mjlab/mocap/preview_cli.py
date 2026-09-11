"""CLI：在交互 viewer 中预览关键帧动作（调试用）。

用法示例（在 mjlab 目录下，用 ``uv run``）：

.. code-block:: sh

  uv run python -m mjlab.mocap.preview_cli \
      src/mjlab/mocap/configs/transition_down.json

  # PD 模式 + 循环播放 + 满速（bool 参数需显式值）
  uv run python -m mjlab.mocap.preview_cli \
      src/mjlab/mocap/configs/transition_down.json --mode pd --loop True --fast True

交互：空格暂停/继续，关闭窗口退出。
"""

from __future__ import annotations

import sys
from typing import Literal

import tyro

import mjlab
from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_constants import (
  get_pikachu_27dof_mocap_robot_cfg,
)
from mjlab.mocap.preview import preview_keyframes


def main(
  keyframes: str,
  fps: float = 30.0,
  easing: Literal["linear", "ease_in_out"] = "linear",
  mode: Literal["kinematic", "pd"] = "kinematic",
  loop: bool = False,
  fast: bool = False,
  device: str = "cpu",
  physics_dt: float = 0.005,
  write_root: bool = False,
  show_com: bool = True,
  robot: Literal["flat14", "27dof"] = "flat14",
  pd_gain_multiplier: float = 1.0,
) -> None:
  """预览 Pikachu 关键帧动作（交互式 viewer 调试）。

  Args:
    keyframes: 关键帧 JSON 文件路径（格式同 record_cli；可用示例
      ``mocap/configs/transition_down.json``）。
    fps: 预览帧率（Hz）。默认 30，显式传入时覆盖 JSON 里的 fps。
    easing: 插值缓动（'linear' 或 'ease_in_out'）。
    mode: 播放模式：'kinematic'（目标姿态）或 'pd'（PD 执行，与录制
      一致）。
    loop: 播完是否循环播放。
    fast: 满速播放（不做实时节流）。
    device: 仿真设备（cpu / cuda:0）。
    physics_dt: 物理仿真步长（秒），仅 pd 模式有意义。
    write_root: pd 模式是否把 root 运动学写入（钉在插值轨迹上）。
      默认 False，与录制一致：root 由浮动基座动力学自由演化。
    show_com: 是否叠加显示整机质心 COM 及其地面投影。默认 True。
    robot: 用哪个机器人预览：``"flat14"``（默认，14dof）或 ``"27dof"``。
      27dof 时未在关键帧里指定的关节（头/耳/尾/臂/腿）默认取 0 弧度，
      无需把所有 27 个都写进 ``joint_pos``。
    pd_gain_multiplier: 执行器 Kp/Kd 统一放大倍率。默认 1.0；pd 模式觉得
      跟随太软可调大（如 1.5~3）让动作更"硬"。只影响这次预览。
  """
  robot_cfg = get_pikachu_27dof_mocap_robot_cfg() if robot == "27dof" else None
  preview_keyframes(
    keyframes,
    fps=fps,
    easing=easing,
    mode=mode,
    loop=loop,
    fast=fast,
    device=device,
    physics_dt=physics_dt,
    write_root=write_root,
    show_com=show_com,
    robot_cfg=robot_cfg,
    pd_gain_multiplier=pd_gain_multiplier,
  )


def _normalize_argv(argv: list[str]) -> list[str]:
  """兼容位置参数形式：``preview_cli cfg.json [opts]``。

  tyro 默认只接受 ``--keyframes`` 命名参数；当命令开头的第一个非选项
  参数是 keyframes 时，映射成命名参数，使位置参数和命名参数用法都可用。
  """
  args = list(argv)
  if args and not args[0].startswith("-"):
    args = ["--keyframes", args.pop(0)] + args
  return args


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS, args=_normalize_argv(sys.argv[1:]))
