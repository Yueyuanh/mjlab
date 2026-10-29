# R08_T6 720P 视频

标称 `weight6.3` 机构，`model_64996.pt`，每段 5 秒、50 FPS、1280×720。
所有文件已用 `ffprobe` 验证为 250 帧、5.000 秒。

| 文件 | 命令 | 视角 | SHA256 前 12 位 |
|---|---|---|---|
| `stand__front__r08-t6-final-nominal.mp4` | 静止 | 正面 | `dbb5439ca072` |
| `fwd__side__r08-t6-final-nominal.mp4` | vx=+0.40 | 侧面 | `4737989263ab` |
| `fwd_fast__side__r08-t6-final-nominal.mp4` | vx=+0.50 | 侧面 | `17f50e9c7595` |
| `back__side__r08-t6-final-nominal.mp4` | vx=−0.30 | 侧面 | `7fc734e64d39` |
| `lat__front__r08-t6-final-nominal.mp4` | vy=+0.25 | 正面 | `e87a82500f04` |
| `turn__three_quarter__r08-t6-final-nominal.mp4` | wz=+0.30 | 3/4 | `5fb21f5a7932` |

录像命令统一使用 `--nominal --device cpu --width 1280 --height 720`；当前主机
NVIDIA 驱动不可用，使用 EGL 软件离屏渲染，因此只影响生成速度，不改变仿真步长
或视频帧率。

