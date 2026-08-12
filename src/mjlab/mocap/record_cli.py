"""CLI：从 JSON 关键帧文件录制 Pikachu 动作 CSV。

用法示例（在 mjlab 目录下，用 ``uv run``）：

.. code-block:: sh

  uv run python -m mjlab.mocap.record_cli \
      src/mjlab/mocap/configs/transition_down.json \
      /tmp/transition_down.csv

录制出的 CSV 与 ``scripts/csv_to_npz.py`` 输入格式完全对齐（21 列，
root_quat 为 xyzw 序），可直接用 csv_to_npz 细化成 50fps 的 motion.npz。
"""

from __future__ import annotations

import sys
from typing import Literal

import tyro

import mjlab
from mjlab.mocap.recorder import load_keyframes_json, record_to_csv


def main(
  keyframes: str,
  out_csv: str,
  fps: float = 30.0,
  easing: Literal["linear", "ease_in_out"] = "linear",
  device: str = "cpu",
  physics_dt: float = 0.005,
  write_root: bool = False,
) -> None:
  """录制 Pikachu 动作 CSV（PD 打点 + 插值）。

  Args:
    keyframes: 关键帧 JSON 文件路径（格式见
      ``recorder.load_keyframes_json``；可用示例
      ``mocap/configs/transition_down.json``）。
    out_csv: 输出 CSV 路径（21 列，与 csv_to_npz.py 输入对齐）。
    fps: 目标录制帧率（Hz）。实际帧率以打印值为准，传给 csv_to_npz
      的 ``input_fps`` 用它。
    easing: 关键帧插值缓动（'linear' 或 'ease_in_out'）。
    device: 仿真设备（cpu / cuda:0）。
    physics_dt: 物理仿真步长（秒）。
    write_root: 是否把 root 运动学写入（钉在插值轨迹上）。默认 False，
      root 由浮动基座动力学 + 关节 PD 自由演化（JSON 只需配置关节角）。
  """
  kf_list, cfg = load_keyframes_json(keyframes)
  # CLI 显式参数优先于 JSON 里的 fps / easing。
  if fps != 30.0:
    cfg.fps = fps
  if easing != "linear":
    cfg.easing = easing

  record_to_csv(
    kf_list,
    out_csv,
    fps=cfg.fps,
    easing=cfg.easing,
    device=device,
    physics_dt=physics_dt,
    write_root=write_root,
    verbose=True,
  )


def _normalize_argv(argv: list[str]) -> list[str]:
  """兼容位置参数形式：``record_cli cfg.json out.csv [opts]``。

  tyro 默认只接受 ``--keyframes`` / ``--out-csv`` 命名参数。当命令开头
  的前两个非选项参数是 keyframes / out_csv 时，把它们映射成命名参数，
  使 README 里的位置参数用法和标准命名参数用法都可用。
  """
  args = list(argv)
  if (
    args
    and not args[0].startswith("-")
    and len(args) >= 2
    and not args[1].startswith("-")
  ):
    args = ["--keyframes", args.pop(0), "--out-csv", args.pop(0)] + args
  return args


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS, args=_normalize_argv(sys.argv[1:]))
