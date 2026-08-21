# mjlab.mocap — Pikachu 数据录制（PD 打点 + 插值 → CSV）

在 MuJoCo 仿真中加载 Pikachu，用 **PD 打点 + 关键帧插值**的方式录制
tracking 任务所需的参考动作数据，输出与
`scripts/csv_to_npz.py` 输入格式**完全对齐**的 CSV。

## 目录结构

```
src/mjlab/mocap/
├── mocap_env.py      # MocapEnv：加载 Pikachu + 平面地形，PD 打点 / 状态读写 / 预览渲染
├── keyframes.py      # PoseKeyframe + 关键帧插值（lerp/slerp，可配置 fps/时长/easing）
├── recorder.py       # record_to_csv：关键帧 → 插值 → PD 执行 → CSV（接口对齐）
├── record_cli.py     # CLI：从 JSON 关键帧文件录制
├── preview.py        # 交互式 viewer 预览关键帧动作（调试姿势/过渡）
├── preview_cli.py    # 预览的 CLI 入口
├── replay.py         # 回放录出的 CSV/npz 动作（检查录制结果）
├── replay_cli.py     # 回放的 CLI 入口
├── configs/          # 示例关键帧 + NOTES.md（有意超限项说明）
│                     #   biped_squat_stand.json = 纯关节角示例（root 自由演化）
└── README.md
```

## CSV 接口约定（与 csv_to_npz.py 严格一致）

每行一帧，**21 列**，无表头，无单位注释（位置米、角度弧度）：

```
列 0–2   root_pos      (x, y, z)           base_link 世界系位置
列 3–6   root_quat     (x, y, z, w)        ⚠️ xyzw 序！脚本加载后转 wxyz
列 7+    dof_pos       14 个关节角         joint_names 序（= XML 定义序）
```

`dof` 列顺序 = `robot.joint_names`，即：

```
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle,
left_arm_pitch, left_arm_roll, right_arm_pitch, right_arm_roll
```

（XML 关节名带 `_joint` 后缀。）项目已提供 Pikachu 专用转换脚本
`scripts/csv_to_npz_pikachu.py`，直接用上面这个列表逐列对齐，**无需手动
替换**；若用通用版 `csv_to_npz.py`，则需把它的 `joint_names` 换成这个列表
（见 `tasks/tracking/TRACKING_FLOW.md` §3.3）。

## 快速开始

### CLI（推荐）

```sh
# 在 mjlab 目录下，用 uv run
uv run python -m mjlab.mocap.record_cli \
    src/mjlab/mocap/configs/transition_down.json \
    src/mjlab/mocap/tmp/transition_down.csv

# 自定义参数：30fps、ease_in_out 缓动、CPU
uv run python -m mjlab.mocap.record_cli \
    src/mjlab/mocap/configs/transition_down.json \
    src/mjlab/mocap/tmp/transition_down.csv \
    --fps 30.0 --easing ease_in_out --device cpu
```

> 位置参数（`config.json out.csv`）与命名参数（`--keyframes ... --out-csv ...`）
> 均可；布尔参数（如 `--write-root`）需显式值：`--write-root True`。
>
> **默认只需配置关节角**：JSON 里的 `root_pos` / `root_quat` 可省略，root
> 由浮动基座动力学 + 关节 PD 自由演化（见「PD 打点原理」）。纯关节角示例：
> `configs/biped_squat_stand.json`。

### 可视化预览（调试关键帧）

录制前先在交互式 MuJoCo viewer 里播放动作，确认姿势与过渡符合预期：

```sh
# 默认 kinematic 模式：播放插值目标姿态（快，适合调关键帧）
uv run python -m mjlab.mocap.preview_cli \
    src/mjlab/mocap/configs/transition_down.json

# pd 模式：PD 打点执行（与录制一致，更真实）；循环 + 满速
uv run python -m mjlab.mocap.preview_cli \
    src/mjlab/mocap/configs/transition_down.json \
    --mode pd --loop True --fast True
```

- 两种模式：`kinematic`（默认，直接显示插值目标姿态）与 `pd`（warp
  动力学 PD 执行，与录制结果一致）。
- **COM 投影叠加**（默认开启）：实时显示整机质心 **绿点**、质心到地面
  的竖直 **黄线**、地面投影 **红点**——用于判断机器人是否稳定（投影是否
  落在支撑多边形内），纯关节角模式下调站姿/蹲姿尤其有用。关闭：
  `--show-com False`。
- 交互：**空格** 暂停/继续；鼠标旋转/缩放视角；关闭窗口退出。
- 布尔参数需显式值：`--loop True` / `--fast True` / `--show-com False`。
- 需要本地图形界面（`DISPLAY`/`WAYLAND_DISPLAY`）；无显示环境会报错。

### 回放录制结果（replay）

`preview_cli` 只能预览**关键帧 JSON**；要看**录出来的实际动作**
（CSV 或细化后的 npz），用 `replay_cli`：

```sh
# 回放录出的 CSV（--input-fps 传实际帧率，录制结束会打印，通常 28.57）
uv run python -m mjlab.mocap.replay_cli \
    src/mjlab/mocap/tmp/transition_up.csv --input-fps 28.57

# 回放细化后的 npz（fps 自带，无需 --input-fps）
uv run python -m mjlab.mocap.replay_cli \
    src/mjlab/mocap/npz/Pikachu-Transition-Up.npz --loop True --fast True
```

- 按后缀自动判别：`*.csv`（mocap 录制格式，需 `--input-fps`）或
  `*.npz`（`csv_to_npz_pikachu` 输出，帧率自带）。
- **npz 是纯运动学参考序列**（`csv_to_npz` 细化的动作参考，供 tracking
  训练消费，与动力学无关），回放它就是按帧显示参考轨迹，**只能用
  `kinematic`**；传 `--mode pd` 会被强制回退。`pd` 模式仅对 **CSV 录制
  产物**有意义（近似重放录制时的 PD 动力学过程）——但更标准的做法是在
  录制前用 `preview_cli --mode pd` 验证关键帧，录出的 CSV 直接 kinematic
  查看即可。
- 复用 COM 投影叠加、空格暂停、循环/满速等交互（见上）。

### Python API

```python
from mjlab.mocap import PoseKeyframe, record_to_csv

record_to_csv(
    [
        PoseKeyframe(
            duration=1.2,
            root_pos=(0.0, 0.0, 0.14),
            root_quat=(0.0, 0.0, 0.0, 1.0),          # xyzw 序
            joint_pos={
                "left_hip_pitch_joint": -0.3, "left_knee_joint": -0.5,
                # ... 其余 12 个关节（见 configs/transition_down.json）
            },
        ),
        PoseKeyframe(
            duration=1.5,
            root_pos=(0.0, 0.0, 0.075),
            root_quat=(0.0, 0.7071, 0.0, 0.7071),   # 前俯 90°（xyzw）
            joint_pos={"left_hip_pitch_joint": -0.5, "left_knee_joint": -0.8, ...},
        ),
    ],
    "/tmp/transition_down.csv",
    fps=30.0,
)
```

录制完成后打印：帧数 / 实际帧率 / 时长 / `root_z` 范围 / CSV 路径，并校验
关节角是否超出软限位（超限会打 `[WARN]`）。

## 关键帧 JSON 格式

```json
{
  "fps": 30.0,
  "easing": "linear",
  "keyframes": [
    {
      "duration": 1.2,
      "root_pos": [0.0, 0.0, 0.14],
      "root_quat": [0.0, 0.0, 0.0, 1.0],
      "joint_pos": {
        "left_hip_pitch_joint": -0.3,
        "left_knee_joint": -0.5,
        "...": 0.0
      }
    }
  ]
}
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `fps` | float | 目标录制帧率（Hz），即 CSV 的 `input_fps` |
| `easing` | string | 插值缓动：`linear`（默认）或 `ease_in_out` |
| `keyframes[].duration` | float | 到**下一个**关键帧的过渡时长（秒） |
| `keyframes[].time` | float | 绝对时间戳（秒）；只要有一个关键帧用 `time`，整段就按绝对时间解析，`duration` 失效 |
| `keyframes[].root_pos` | [x,y,z] | base_link 世界系位置（米）。**可省略**，省略则 root 由浮动基动力学演化（默认模式） |
| `keyframes[].root_quat` | [x,y,z,w] | base_link 世界系姿态，**xyzw 序**。**可省略**，省略则 root 由浮动基动力学演化；未省略时与 `root_pos` 一起作为初始/引导姿态 |
| `keyframes[].joint_pos` | dict | 关节角（弧度），键为 XML 关节名；未列出的关节沿用上一关键帧的值 |

## 插值参数

- **`fps`**：录制采样率。实际帧率 = `1 / (round(1/(fps*physics_dt)) * physics_dt)`，
  会因物理步长取整略有偏差，以打印值为准，把它传给 `csv_to_npz.py` 的
  `input_fps`（`output_fps` 保持 50，匹配 tracking env 的 `step_dt=0.02s`）。
- **`easing`**：代码/CLI 默认 `linear`（与 `csv_to_npz` 的 lerp 一致）；
  `ease_in_out`（smoothstep）段两端速度为零，过渡更柔和——对"蹲→趴"这类
  起止静止的动作建议使用，`configs/` 示例已用它。
- **`physics_dt`**：物理仿真步长，默认 0.005s（与 tracking env 一致）。
- **四元数插值**：始终使用 slerp（与 csv_to_npz 的 `quat_slerp` 一致）。

## PD 打点原理

Pikachu 是浮动基机器人（root 无执行器），录制时：

1. **关节**：把插值得到的关节角写入 ctrl（MuJoCo position actuator 内部做
   PD），让机器人真实动力学响应若干步，然后读取**实际**关节角写入 CSV——
   录出的是"电机 PD 能做到的角度"，带平滑的动力学响应。
2. **root（默认，`write_root=False`）**：**不写回**，由浮动基座动力学 +
   接触 + 关节 PD 自由演化。因此 **JSON 只需配置关节角**，root 的位置/姿态
   是机器人真实动力学响应（可能站不稳/倾倒，这正是动力学数据）。
3. **root（可选，`write_root=True`）**：把关键帧插值的 root 姿态每子步写回，
   钉在设定的过渡路径上（旧 guided 模式）——仅用于调试关键帧姿态，或录制
   需要 root 大幅姿态变化（如四足前俯过渡）且动力学无法自然完成的动作。

因此 CSV 中：`dof` 列来自 PD 实际执行（动力学），`root` 列默认也是动力学
真实状态（浮动基自由演化）；两者结合是 tracking 训练标准的参考动作形式。

> ⚠️ 关节与执行器命名：写 ctrl 时槽位顺序是 `robot.actuator_names`
> （J4310 的髋 pitch / 膝排在最前），与 `joint_names` 不同；本模块内部
> 已自动重排，用户无需关心，只需在关键帧里用 XML 关节名（`joint_names`）。

## 完整工作流：CSV → npz → 训练

```sh
# 1. 录制（本模块）
uv run python -m mjlab.mocap.record_cli \
    src/mjlab/mocap/configs/transition_down.json \
    /tmp/transition_down.csv --fps 30.0

# 2. 用 csv_to_npz_pikachu.py 细化到 50fps 并生成 motion.npz
#    （Pikachu 专用版，joint_names 已适配 14 关节；默认不上传 wandb，
#     需要上传时加 --wandb）
#    npz 默认输出到项目内 src/mjlab/mocap/npz/{output_name}.npz
#    （目录自动创建，每个 --output-name 一个文件互不覆盖）；
#    自定义路径用 --output-file <path>。
uv run python -m mjlab.scripts.csv_to_npz_pikachu \
    --input-file src/mjlab/mocap/tmp/transition_up.csv \
    --output-name Pikachu-Transition-Down \
    --input-fps 28.57 --output-fps 50 --device cpu

```

> ⚠️ **没有"用 play 预览 npz"这回事（对 Pikachu）**：
> `play --agent zero --motion-file X.npz` 只在 **tracking 任务**下有效——
> play.py 靠 `env_cfg.commands["motion"]`（`MotionCommandCfg`）判断并加载
> 参考动作，而 Pikachu tracking env 尚未落地（tracking 任务目前只有 G1：
> `Mjlab-Tracking-Flat-Unitree-G1`）。改成 `Mjlab-Velocity-Flat-Pikachu`
> 也没用：velocity 没有 motion 命令，`--motion-file` 会被静默忽略。
> 回放 Pikachu 动作用本模块的 `replay_cli`（CSV/npz，见「回放录制结果」）；
> 调关键帧用 `preview_cli`（预览**关键帧 JSON**，见「可视化预览」）。

```sh
# 3. 回放录出的动作（本模块，见「回放录制结果（replay）」）
uv run python -m mjlab.mocap.replay_cli \
    src/mjlab/mocap/tmp/transition_up.csv --input-fps 28.57

# 或回放细化后的 npz（纯运动学参考，kinematic 直接显示；pd 对它无意义）
uv run python -m mjlab.mocap.replay_cli   --input-file  src/mjlab/mocap/npz/Pikachu-Transition-Up.npz
```

## Metrics 日志（对比 RL 控制器 vs 单点位插值控制）

除上述 **21 列运动学参考 CSV**（给 tracking 训练消费）外，mocap 录制还可以
**额外**产出一份与 RL tracking 阶段 `play ... --viewer viser` 录制出的
`metrics_*.csv` **同格式**的电机输出日志。两者并存：运动学 CSV 给训练，
metrics 日志给离线对比「RL policy 控制器 vs 单点位插值控制」下**电机输出
（关节侧 `qfrc_actuator` 扭矩等）是否一致**，保存到
`src/mjlab/mocap/logs/`，可进 plot_logs dashboard 分析。

### 怎么录

在 `record_cli` 加 `--metrics-log <目录>`（建议 `src/mjlab/mocap/logs`）；
`--metrics-source <标识>` 写入 `network` 列与文件名，缺省取关键帧文件名。

```sh
# 录一段 transition_down，同时产出一份 metrics 日志
uv run python -m mjlab.mocap.record_cli \
    src/mjlab/mocap/configs/transition_down.json \
    src/mjlab/mocap/tmp/transition_down.csv \
    --fps 30.0 \
    --metrics-log src/mjlab/mocap/logs \
    --metrics-source Pikachu-Mocap-TransitionDown__pd

# 产物：mocap/logs/metrics_Pikachu-Mocap-TransitionDown__pd_<stamp>_with_target.csv
```

Python API 对应入口是 `record_metrics_to_csv`（参数语义与 `record_to_csv`
一致，只是按帧额外采样电机输出）：

```python
from mjlab.mocap.metrics_logger import record_metrics_to_csv

record_metrics_to_csv(
  keyframes, "src/mjlab/mocap/logs", "Pikachu-Mocap-X__pd", fps=30.0
)
```

### 列格式（对齐 RL `metrics_*.csv`）

表头前 5 列固定：`sample, sim_step, time_s, env_idx, network`。随后是
指标列（与 RL `add_pikachu_telemetry_metrics` 同序、同前缀）：

- `base_height_m`
- 逐关节（14 关节 × 6 指标）：`joint_position_rad/<j>`（实测）、
  `joint_target_rad/<j>`（= position actuator 的 **PD 设定值**，`data.ctrl`
  按 actuator 槽位序重排回 joint 序；与 RL 侧 `add_pikachu_telemetry_metrics`
  新加的 `joint_target_rad` 同口径，可逐关节对比下发的目标角）、
  `joint_velocity_rad_s/<j>`、`joint_torque_nm/<j>`（= 关节侧
  `qfrc_actuator`，N·m）、`joint_torque_ratio/<j>`（= |扭矩|/effort_limit）、
  `joint_velocity_ratio/<j>`（= |角速度|/velocity_limit）
- `actuator_spec_violation`（该帧任一关节扭矩或角速度超电机 spec → 1）

比例的分母（effort/velocity 限位）复用 `pikachu_metrics._actuator_spec` 这个
单一来源，与 RL 那份完全同口径，可直接并排对比。

### 在 plot_logs dashboard 看

```sh
uv run python scripts/plot_logs/serve.py   # 然后浏览器打开打印的 URL
```

`serve.py` 的 metrics 扫描现已包含 `src/mjlab/mocap/logs`，dashboard 会自动
把 mocap/logs 下的 `metrics_*.csv` 与 RL 的 metrics 放进同一关节/扭矩分类；
选择对应条目即可左右脚/左右关节对比、绘制扭矩/位置/速度曲线。

### 与 RL 样本的可比性 & 差异

**逐字节对齐**：表头结构、5 个关节分类前缀、各列名、`metrics_*_<stamp>.csv`
命名 → dashboard 把它们归入同一分类、可分别加载对比。

**不会对齐 / 需注意**：
- **无 `foot_slip_velocity_m_s` / `foot_slide_fraction` / `action_acc`**：
  mocap 没有接触传感器和 action manager，这三列省略（RL 样本里有）。
- **时间轴**：`time_s = 帧间隔 × steps_per_frame × physics_dt`（mocap 是
  物理步推进，通常 0.005s）；RL 用的是 `env.step_dt`=0.02s 控制周期。两者
  duration 都正确，但细腻度不同——dashboard 的 x 轴是 `time_s`，属正常差异。
- **`network` 列**：填 `--metrics-source`（标识 mocap + 动作名），RL 那份
  填的是 `{task_id}__{checkpoint}`。
- **`sim_step`**：mocap 为累计物理步计数；RL 为 env 的物理步计数器，语义对齐。

## 注意事项

- **关键帧数值微调**：`configs/` 里的过渡中间态是粗设占位。建议在 viewer
  里先校准 `QUAD_KEYFRAME`（`pikachu_constants.py`）手脚着地、躯干不陷地，
  再把数值抄进 JSON；中途帧按手感增删。
- **root_quat 必须 xyzw**：写错成 wxyz 会在 `csv_to_npz` 加载后姿态错乱。
- **关节角限位**：超软限位的角度会被录制时 MuJoCo clamp（硬限位），或
  在训练 reset 时被 clip，导致参考与真实不一致。录制后会 `[WARN]` 提示。
  `configs/transition_down.json` 的 ankle=-0.7 / arm_roll=0.0 是**有意**
  超限的校准值，详见 [configs/NOTES.md](configs/NOTES.md)。
- **髋 roll 限位很窄**（硬限位约 ±0.26 rad）。示例的四足过渡在 PD 执行时
  髋 roll 会短暂超限约 2°（ease_in_out 下），属过渡瞬态；若需要更严格的
  参考数据，在 viewer 里给 QUAD 姿态调一个髋 roll 小外展目标即可消除。
- **root_z**：下蹲/趴地数据应平滑过渡（z 约 0.075–0.14），跳变会触发
  tracking 的 `bad_anchor_pos_z_only` 终止条件。
- **默认不写 root**：root 由浮动基座动力学自由演化，JSON 只需配置关节角。
  对需要 root 大幅姿态变化的动作（如四足前俯过渡），机器人可能倾倒——
  此时要么调关节角让过渡更平稳，要么用 `--write-root True` 把 root 钉在
  插值轨迹上（root 列运动学、dof 列动力学，见「PD 打点原理」）。
