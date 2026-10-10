"""CLI：在交互 viewer 中预览关键帧动作（调试用）。

用法示例（在 mjlab 目录下，用 ``uv run``）：

.. code-block:: sh

  # viser（默认）：浏览器 3D 预览，GUI 有进度条 / 切 json / 参数热加载 / 重置，
  # 无需重开进程。
  uv run python -m mjlab.mocap.preview_cli \
      src/mjlab/mocap/configs/transition_down.json --mode pd --loop True --fast True

  # native：原 launch_passive 桌面 viewer（空格暂停/继续，关闭窗口退出）。
  uv run python -m mjlab.mocap.preview_cli \
      src/mjlab/mocap/configs/transition_down.json --backend native --mode pd --loop True

viser 后端：浏览器打开打印的 URL；改 easing/fps/mode/json 即时热生效，Reset 回初始。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import tyro

import mjlab
from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_constants import (
  get_pikachu_27dof_mocap_robot_cfg,
)
from mjlab.mocap.preview import preview_keyframes

_JSON_DIR_DEFAULT = str(Path(__file__).resolve().parent / "configs")


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
  loop_reset: bool = True,
  backend: Literal["viser", "native"] = "viser",
  json_dir: str | None = None,
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
    loop_reset: 循环播放时每轮结束把机器人运动学**恢复回初始位姿**（含
      root pos/rot，即轨迹首帧）再从头播放。**默认 True**；只要
      ``loop=True`` 就生效——对 pd 模式下 root 由动力学自由演化的动作
      （如倒地/自起）尤其有用，否则虽然关节角经回退段回到初始，但 base
      仍然偏离。True 时不附加"末帧→首帧"的回退插值段（轨迹即动作本身）；
      需要平滑倒放回起的循环可显式 ``--loop-reset False``。
    backend: 预览后端。``"viser"``（默认，浏览器 3D + 交互面板，无需重开
      进程）或 ``"native"``（原 launch_passive 桌面窗口，空格暂停）。
    json_dir: viser 后端 GUI "JSON" 下拉列出的 json 目录；默认
      ``mocap/configs/``。当前 keyframes 路径自动并入下拉选项。
  """
  robot_cfg = get_pikachu_27dof_mocap_robot_cfg() if robot == "27dof" else None
  if backend == "viser":
    from mjlab.mocap.preview_viser import preview_viser

    preview_viser(
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
      loop_reset=loop_reset,
      json_dir=json_dir or _JSON_DIR_DEFAULT,
    )
  else:
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
      loop_reset=loop_reset,
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
