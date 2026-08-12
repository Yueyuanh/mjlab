# Pikachu 状态转换 · Tracking 任务

> 让 Pikachu 学会"站↔趴"过渡段（biped↔quad）的 **motion mimic** 任务。
> 策略直接模仿一段参考动作，而不是靠奖励函数试错。
> 属于三模型架构的过渡模型部分，见
> [TRANSITION_ARCHITECTURE.md](../../TRANSITION_ARCHITECTURE.md) §3.3 与 §4。

## 任务列表

| 任务 ID | 方向 | 参考动作 | 长度 @50fps |
|---|---|---|---|
| `Mjlab-Tracking-Flat-Pikachu-Down` | biped → quad（趴下） | `Pikachu-Transition-Down.npz` | 316 帧 ≈ 6.3 s |
| `Mjlab-Tracking-Flat-Pikachu-Up` | quad → biped（站起） | `Pikachu-Transition-Up.npz` | 499 帧 ≈ 10 s |

两个方向动作模式完全不同，拆成两个独立模型（两个 checkpoint），不要拼成一段。
任务可用 `uv run list-envs` 查看。

## 目录结构

```
config/pikachu_transition/
├── __init__.py     # 注册 Down / Up 两个任务（runner 为 MotionTrackingOnPolicyRunner）
├── env_cfgs.py     # Pikachu tracking 环境配置（复用 make_tracking_env_cfg()）
├── rl_cfg.py       # PPO runner 配置（512/256/128 + ELU，复用 G1 tracking 超参）
└── README.md
```

## 数据来源（mjlab.mocap）

参考动作数据由 [`mjlab.mocap`](../../../../mocap/README.md) 录制并转换：

```
关键帧 JSON（configs/transition_down|up.json）
   → record_cli 录制（PD 打点 + 插值）→ tmp/transition_down|up.csv
   → csv_to_npz_pikachu.py（重采样到 50fps + 仿真 playback 校验）
   → mocap/npz/Pikachu-Transition-Down|Up.npz   ← tracking 训练消费
```

- **帧率约定**：tracking env 是 `timestep=0.005, decimation=4` → `step_dt=0.02s`
  = **50fps**，npz 已是 50fps，与 `MotionCommand` 逐帧推进一致。
- **body/joint 序**：npz 存的是全部 17 个 body（`body_link_pos_w` 列序）与全部
  14 个关节（XML 定义序），训练端按 `body_names` 过滤。body 索引 0 = `base_link`
  （freejoint root body）。
- **追踪 body**（`env_cfgs.py::BODY_NAMES`，11 个）：`base_link` + 双腿
  hip/knee/ankle + 双手 `arm_pitch/arm_yaw`。`arm_yaw_link` 是手（四足撑地点），
  起立/趴下时手的位置与着地时机是核心，必须追踪。

### 数据说明

- **关节限位**：`pikachu_constants.py` 中 `soft_joint_pos_limit_factor=1`，即
  软限位 = 硬限位，训练 reset 时 `_write_reference_state_to_sim` 的 clip 基本是
  无操作，参考与录制值一致。个别帧会把 hip_pitch 压到硬限位（−2.90 / +2.90，
  深蹲/趴下姿态）、arm_roll 停在 0.00（左右硬限位边界），属过渡动作的正常极值。
- **root_z 平滑**：趴下数据 base_link 从直立 ~0.18 过渡到前俯 ~0.16，应平滑；
  若出现跳变会触发 `bad_anchor_pos_z_only` 终止，改关键帧时注意。
- 重新录制 / 重新转换数据后，直接覆盖 `mocap/npz/` 下的同名 npz 即可，任务配置
  默认 `motion_file` 已指向该路径，无需改动。

## 用 play 回放 npz 验证数据集

录制并转换出 npz 后，先**用 play 回放确认参考动作正确**，再开始训练。play 对
tracking 任务要求显式 `--motion-file`（play.py 只认 CLI 参数，不读配置里的默认
motion_file）。

### 推荐：viser 网页端（ghost + 逐帧检查）

```sh
uv run play Mjlab-Tracking-Flat-Pikachu-Down --agent zero \
    --motion-file src/mjlab/mocap/npz/Pikachu-Transition-Down.npz \
    --viewer viser --num_envs 1

```

浏览器打开 viser 页面后：

1. **实时播放**：半透明 **ghost**（参考姿态）从第 0 帧开始按 50fps 推进；
   **实心机器人**的初始帧由 `sampling_mode` 决定：默认 `adaptive` 随机取帧，
   传 `--sampling-mode start` 可固定从参考首帧起步。重点看 ghost 的姿态序列
   是否符合预期（趴下：直立 → 屈膝 → 手触地 → 前俯四点支撑）。
2. **逐帧检查**：点 **Pause** 暂停 → 出现 **Frame** 滑块 → 拖到任意帧 →
   点 **Start Here**，仿真会 `reset_to_frame` 瞬移到该参考帧，可放大查看
   手脚着地、躯干姿态、有无穿地。
3. 换 `--agent random` 可看参考动作在随机动作下的对抗效果（ghost 始终是参考）。

常用附加参数：

| 参数 | 作用 |
|---|---|
| `--no-terminations` | 关闭终止条件，避免检查中途因偏离参考被重置 |
| `--sampling-mode start` | 固定每个 episode 从参考首帧起步（默认 `adaptive` 随机） |
| `--num_envs N` | 多环境并排对比 |
| `--agent zero / random` | 不加载策略；`trained` 则需 `--checkpoint-file` |
| `--device cpu` | 无 GPU 时用 CPU |
| `--motion-file <path>` | 指定要回放的 npz（必填） |

### 备选：mocap 的 replay_cli（纯运动学，更快）

只想快速看参考轨迹本身、不需要 sim 动力学时，用
[`mjlab.mocap.replay_cli`](../../../../mocap/README.md)（kinematic 模式按帧显示）：

```sh
uv run python -m mjlab.mocap.replay_cli \
    src/mjlab/mocap/npz/Pikachu-Transition-Down.npz --loop True --fast True
```

> 注意：`replay_cli` 对 npz 只能 `kinematic` 显示（npz 是纯运动学参考序列）；
> `play`（dummy agent）则带仿真动力学，两者侧重点不同。

## 训练

```sh
# Down（趴下）。默认 motion_file 已指向 mocap/npz/Pikachu-Transition-Down.npz，
# 无需再传 --env.commands.motion.motion-file（train.py 检测到文件存在直接使用）。
uv run train Mjlab-Tracking-Flat-Pikachu-Down --env.scene.num_envs=4096

# Up（站起）
uv run train Mjlab-Tracking-Flat-Pikachu-Up --env.scene.num_envs=4096

# 冒烟：小 batch 快速确认奖励上升、参考不 collapse
uv run train Mjlab-Tracking-Flat-Pikachu-Down --env.scene.num_envs=1024 --agent.max_iterations=500   --env.commands.motion.sampling-mode start

# start 

```

### 临时固定首帧训练（调试用）

默认 `sampling_mode="adaptive"`，每个 episode 随机取参考帧起步。想快速开始
训练、让每个 episode 固定从参考首帧起播时，加 `--env.commands.motion.sampling-mode start`
（保留默认 reset 扰动）：

```sh
uv run train Mjlab-Tracking-Flat-Pikachu-Down \
    --env.scene.num_envs=4096 \
    --env.commands.motion.sampling-mode start
```

> 这是**临时调试手段**。训练进入正轨后，删掉
> `--env.commands.motion.sampling-mode start` 即可切回默认随机（`adaptive`）。

若想用自己录的数据替换默认 npz，可直接覆盖 `mocap/npz/` 下的文件，或用 CLI
显式指定：

```sh
uv run train Mjlab-Tracking-Flat-Pikachu-Up \
    --env.commands.motion.motion-file /path/to/your.npz \
    --env.scene.num_envs=4096
```

## 训练回放

```
uv run play Mjlab-Tracking-Flat-Pikachu-Up \
    --motion-file src/mjlab/mocap/npz/Pikachu-Transition-Up.npz \
    --checkpoint-file logs/rsl_rl/pikachu_tracking/2026-08-12_17-42-26_trail-pikachu-transition-up-test/model_499.pt \
    --env.scene.num_envs=1 \
    --viewer viser 


uv run play Mjlab-Tracking-Flat-Pikachu-Down \
    --motion-file src/mjlab/mocap/npz/Pikachu-Transition-Down.npz \
    --checkpoint-file logs/rsl_rl/pikachu_tracking/2026-08-12_17-42-26_trail-pikachu-transition-up-test/model_499.pt \

    --viewer viser 

```

## 评估（trained 策略）

```sh
# 网页端观察（ghost 显示参考、frames 显示坐标系，可拖动 scrubber）
uv run play Mjlab-Tracking-Flat-Pikachu-Down --viewer viser \
    --checkpoint-file logs/rsl_rl/pikachu_tracking/.../model_xxx.pt \
    --motion-file src/mjlab/mocap/npz/Pikachu-Transition-Down.npz
```

上面的 play 用本地 checkpoint 做**定性**观察。要**定量**指标（MPKPE /
R-MPKPE / joint_vel_error / ee_* / success_rate），用 `evaluate.py`。注意它
**只支持 W&B 上的 run**：第一个参数是任务 ID（必填），`--wandb-run-path` 填
`entity/project/run_id`（W&B 网页上该 run 地址 URL 的后三段）：

```sh
uv run python src/mjlab/tasks/tracking/scripts/evaluate.py \
    Mjlab-Tracking-Flat-Pikachu-Down \
    --wandb-run-path <entity>/<project>/<run_id>


# https://wandb.ai/tup_yyh/mjlab/runs/b7w8gngg?nw=nwuser3117144922
uv run python src/mjlab/tasks/tracking/scripts/evaluate.py \
    Mjlab-Tracking-Flat-Pikachu-Down \
    --wandb-run-path tup_yyh/mjlab/b7w8gngg \
    --motion-file src/mjlab/mocap/npz/Pikachu-Transition-Down.npz \
    --num_envs 512 \
    --output-file src/mjlab/mocap/eval/pikachu_down_eval.json
```

`evaluate.py` 从该 W&B run 拉两样东西，缺一不可：

- **参考动作**：run 的 `used_artifacts()` 里 `type == "motions"` 的 artifact
  （`motion.npz`）。只有训练是用 `--registry-name <entity>/motions/<name>` 起的
  （数据先经 `csv_to_npz ... --wandb` 上传成 motions artifact），runner 才会
  `use_artifact`；仅传本地 `--env.commands.motion.motion-file` 训练的 run 没有
  这个 artifact，会报 `No motion artifact found in the run.`。
- **checkpoint**：取 run 里最新的 `model_*.pt`（或 `--wandb-checkpoint-name`
  指定），下载到 `logs/rsl_rl/pikachu_tracking/wandb_checkpoints/<run_id>/`。

可选参数：

| 参数 | 说明 |
|---|---|
| `--wandb-checkpoint-name model_4000.pt` | 指定加载某个 checkpoint，默认最新 |
| `--num_envs 1024` | 并行评估的 episode 数（默认 1024） |
| `--device cuda:0` | 设备，默认 CUDA |
| `--output-file /tmp/eval.json` | 把指标存成 JSON |
| `--log-root logs/rsl_rl` | checkpoint 本地缓存根目录 |

> 注意：evaluate 用**训练 cfg**（`play=False`），且保留传感器噪声
> （`enable_corruption=True`），`sampling_mode="start"` 每个 episode 从参考首帧
> 开始；成功 = 跑完整段 clip 且没被终止条件掐掉（`success_rate`）。

### 六个指标含义

| 指标 | 单位 | 含义 | 实现（`mdp/metrics.py`） |
|---|---|---|---|
| `success_rate` | — | **成功率**：跑完整段 clip 且没被终止条件提前掐掉的比例 | 成功 = `truncated & ~terminated`（episode 因播完而结束，而非偏离参考被终止） |
| `mpkpe` | 米 | **全局坐标系**下所有追踪 body 的平均位置误差（**含**全局平移与朝向漂移） | `‖body_pos_w − robot_body_pos_w‖`，对追踪 body 取平均 |
| `r_mpkpe` | 米 | **相对 root** 的 body 位置误差：参考被重锚定到机器人当前 root 的位置+朝向后计算，去掉全局漂移，只剩纯姿态误差（训练 reward 优化的量） | `‖body_pos_relative_w − robot_body_pos_w‖` |
| `joint_vel_error` | rad/s | 关节**角速度**误差的 RMS（跨所有关节），衡量动作速度/节奏贴合度 | `sqrt(mean((joint_vel − robot_joint_vel)²))` |
| `ee_pos_error` | 米 | **末端**（手+脚 4 个 body：左右 ankle + 左右 `arm_yaw_link`）位置误差，相对 root | `body_pos_relative_w` 对 ee body 取平均 |
| `ee_ori_error` | rad | **末端姿态**误差（四元数差角），相对 root | `quat_error_magnitude` 对 ee body 取平均 |

怎么读这组数：

- `mpkpe` 与 `r_mpkpe` 之差 ≈ **全局漂移**贡献的误差；`r_mpkpe` 才是动作本身的贴合度。
- 分工：`success_rate` 看"崩不崩"，`mpkpe` / `r_mpkpe` 看"动作贴不贴"，`ee_*` 单独看手脚着地质量，`joint_vel_error` 看速度节奏。
- 经验参考：`r_mpkpe` < 0.05 m 不错、< 0.03 m 很好；`success_rate` > 0.9 基本不崩。

## 关键配置项（env_cfgs.py）

| 项 | 值 | 说明 |
|---|---|---|
| `anchor_body_name` | `base_link` | freejoint root body，作锚点 |
| `body_names` | 11 个（锚+腿+手） | 追踪 body，见上文 |
| `motion_file` | `mocap/npz/Pikachu-Transition-{Down,Up}.npz` | 按方向自动选 |
| `sampling_mode` | `adaptive`（默认） | 过渡是单段非周期动作，若收敛差可试 `uniform` |
| `pose_range` / `velocity_range` | 默认抖动 | 过大 reset 会穿地，play 时置空 |
| `terminations.ee_body_pos` | 手+脚 4 个 body | 末端脱离参考 z 提前终止 |
| 碰撞 | `FULL_COLLISION_WITHOUT_SELF` | 手 `arm_yaw_link` condim=3+摩擦，四足撑地不打滑 |

### 碰撞体切换（capsule ↔ lite）

Pikachu 有两套碰撞体 XML，机器人实体与脚底摩擦随机化必须使用同一套：

- **capsule（默认）**：`pikachu_constants.py` → `Pikachu_V025_flat_14dof_capsule.xml`，
  碰撞 geom 稀疏（base/hip_roll/knee/arm_yaw），脚底为 G1 风格 foot1~7 胶囊。
- **lite**：`pikachu_constants_lite.py` → `Pikachu_V025_flat_14dof_lite.xml`，
  原始全身碰撞体（每个 link 一个 `_collision` geom），脚底为 `ankle_link`
  自身网格。

对比两种碰撞体对训练/部署的影响时，把本文件顶部 import 来源整体换成
`pikachu_constants_lite`（两模块导出同名，仅脚底 geom 不同）：

```python
from mjlab.asset_zoo.robots.pikachu.pikachu_constants_lite import (
  PIKACHU_FOOT_GEOM_NAMES,
  get_pikachu_robot_cfg,
)
```

改这一处即可——机器人实体（`get_pikachu_robot_cfg`）和 `foot_friction` 事件
（`PIKACHU_FOOT_GEOM_NAMES`）会自动跟随。velocity 任务的
`tasks/velocity/config/pikachu/env_cfgs.py` 同理。

## 与三模型架构的关系

```
BIPED（velocity 行走）──指令"四足"──▶ Down（本任务）──完成判定──▶ QUAD（四足爬行）
     ▲                                                                    │
     └──────────────────────────────── Up（本任务）──指令"双足"────────────┘
```

本任务负责**过渡段**（站↔趴，一次性非周期动作）；稳态行走 / 爬行分别由
velocity / transition 任务训练。切换时的连续性要求（clip 首帧=当前稳态姿态、
动作通道 14 维对齐、观测通道差异）见 TRANSITION_ARCHITECTURE.md §5.3。

## 相关文档

- [TRANSITION_ARCHITECTURE.md](../../TRANSITION_ARCHITECTURE.md) — 三模型架构
- [TRACKING_FLOW.md](../../TRACKING_FLOW.md) — tracking 任务数据流与调参
- [mjlab.mocap README](../../../../mocap/README.md) — 参考动作录制
