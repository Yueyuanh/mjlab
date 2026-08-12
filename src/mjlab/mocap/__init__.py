"""Pikachu 数据录制：MuJoCo 仿真环境（PD 打点 + 插值 → CSV）。

模块组成：
  - ``mocap_env``：加载 Pikachu + 平面地形的仿真环境，提供 PD 打点、
    状态读取等底层接口。
  - ``keyframes``：关键帧（打点）定义与插值（lerp/slerp，可配置
    fps / 时长 / easing）。
  - ``recorder``：录制管线，把关键帧插值 → PD 执行 → 写 CSV（与
    ``scripts/csv_to_npz.py`` 输入格式完全对齐）。
  - ``record_cli``：从 JSON 关键帧文件录制的命令行入口。
  - ``preview``：交互式 viewer 预览关键帧动作（调试姿势与过渡），
    支持 kinematic（目标姿态）与 pd（PD 执行）两种模式。
  - ``preview_cli``：预览的命令行入口。

典型用法：

.. code-block:: python

  from mjlab.mocap import PoseKeyframe, record_to_csv

  record_to_csv(
    [
      PoseKeyframe(joint_pos={...}, root_pos=(0, 0, 0.14),
                   root_quat=(0, 0, 0, 1), duration=1.5),
      PoseKeyframe(joint_pos={...}, root_pos=(0, 0, 0.075),
                   root_quat=(0, 0.7071, 0, 0.7071)),
    ],
    "/tmp/transition.csv",
    fps=30.0,
  )
"""

from mjlab.mocap.keyframes import (
  InterpConfig,
  PoseKeyframe,
  Trajectory,
  build_trajectory,
)
from mjlab.mocap.mocap_env import MocapEnv
from mjlab.mocap.preview import preview_keyframes
from mjlab.mocap.recorder import (
  RecordingResult,
  load_keyframes_json,
  record_to_csv,
)

__all__ = [
  "InterpConfig",
  "PoseKeyframe",
  "Trajectory",
  "build_trajectory",
  "MocapEnv",
  "preview_keyframes",
  "RecordingResult",
  "load_keyframes_json",
  "record_to_csv",
]
