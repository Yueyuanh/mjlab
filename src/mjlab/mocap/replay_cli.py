"""``mjlab.mocap.replay_cli`` 的命令行入口：回放录出的 CSV/npz 动作。"""

from __future__ import annotations

from typing import Literal

import tyro

import mjlab
from mjlab.mocap.replay import load_motion, replay_motion


def main(
  input_file: str,
  mode: Literal["kinematic", "pd"] = "kinematic",
  input_fps: float | None = None,
  loop: bool = False,
  fast: bool = False,
  device: str = "cpu",
  physics_dt: float = 0.005,
  write_root: bool = False,
  show_com: bool = True,
) -> None:
  """在交互 viewer 中回放录出的参考动作（CSV/npz）。

  Args:
    input_file: ``*.csv``（mocap 录制格式）或 ``*.npz``
      （csv_to_npz_pikachu 输出）。按后缀自动判别。
    mode: ``"kinematic"``（默认，直接显示参考轨迹，npz 只能用它）或
      ``"pd"``（仅对 CSV 录制产物有意义：近似重放录制时的 PD 动力学
      过程；对 npz 会被强制 kinematic）。
    input_fps: 仅 CSV 需要——录制的**实际**帧率（录制结束会打印，例如
      28.57）。npz 自带 fps，忽略本参数。
    loop: 播完是否循环。
    fast: 满速播放（不做实时节流）。pd 模式建议配合使用。
    device: 仿真设备，``"cpu"`` 或 ``"cuda:0"``。
    physics_dt: 物理仿真步长（秒），仅 pd 模式有意义。
    write_root: pd 模式是否把录出的 root 姿态写回（钉在录制轨迹上）。
      默认 False，与录制默认一致。
    show_com: 是否叠加显示整机质心 COM 及其地面投影。默认 True。
  """
  motion = load_motion(input_file, input_fps)
  replay_motion(
    motion,
    mode=mode,
    loop=loop,
    fast=fast,
    device=device,
    physics_dt=physics_dt,
    write_root=write_root,
    show_com=show_com,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
