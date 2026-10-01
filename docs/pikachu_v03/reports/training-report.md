# Pikachu v03 velocity 训练报告

> 模型：`pikachu_v03_base_series`（并联四杆膝，14 关节 / 10 驱动）
> 任务：`Mjlab-Velocity-Flat-Pikachu-V03`
> 分支：`task/pikachu-v03-locomotion`
> 日期：2026-10-01

**一句话**：v03 的 velocity 任务从零建起来能跑通、能站住，但第一版训练
（r01）学出来的是**站着不动**——原因是奖励设计里有两个独立缺陷。修复后
（r02）策略开始真正迈步。

---

## 1. 交付物

| 类别 | 路径 |
|---|---|
| 训练用 MJCF（含 meshes） | `src/mjlab/asset_zoo/robots/pikachu/xmls/mjcf_v03/` |
| 机器人常量与 keyframe | `src/mjlab/asset_zoo/robots/pikachu/pikachu_v03_constants.py` |
| 任务配置 | `src/mjlab/tasks/velocity/config/pikachu_v03/` |
| 训练看板（文本） | `scripts/v03_train_dashboard.py` |
| 单行监控 | `scripts/v03_watch.py` |
| 定量评测 | `scripts/eval_v03_velocity.py` |
| 720P 无头录像 | `scripts/record_v03_video.py` |
| 工作日志 | [../04-worklog.md](../04-worklog.md) |

## 2. 环境约束（所有决策都受它影响）

| 项 | 值 | 后果 |
|---|---|---|
| 内存 | 14 GB（可用 ~3-6 GB，**swap 已满**） | 同一进程建两个 env 会 OOM；训练期间起第二个仿真进程会换页到 8 秒/帧 |
| 磁盘 | 468 GB，**93% 已用，剩 ~34 GB** | 需持续盯 |
| GPU | RTX 4060 Laptop 8 GB | 训练占 ~1.5-2.4 GB，非瓶颈 |

## 3. 方法

### 3.1 模型接入

源模型 `pikachu_urdf/pikachu_v03_base_series/`（含 meshes）→ mjlab 训练用
MJCF。改造 6 处，详见 [../01-model-conversion.md](../01-model-conversion.md)：

1. 加 `freejoint`（源模型 base 无自由关节）
2. 统一 geom 命名（脚底 5 条胶囊 → `^(left|right)_foot[1-5]_collision$`）
3. 补足部 `site` + 碰撞胶囊
4. 删 scene include
5. **删 `<actuator>`** —— mjlab 的 `_add_actuators()` 只追加不清理，
   留着会让 `nu` 翻倍
6. 补 `<sensor>` 块，**imu site 必须是单位四元数**

验收：`nq=21 nv=20 nu=0 njnt=15 nbody=16 ngeom=40`。

### 3.2 静平衡 keyframe

「脚在髋下」≠「静平衡」。v03 质心在 base 原点后方约 2 cm，零动作会先仰到
-0.167 rad 再被 PD 拉回，停在非零平衡点。用固定点迭代把脚心挪到质心正下方，
得到 `KNEES_BENT_KEYFRAME`（base_z = 0.26728，CoM x = -0.02683）。
详见 [../02-stand-keyframe.md](../02-stand-keyframe.md)。

> ⚠️ **左腿不是整体镜像**。实测 `knee_L/knee_R = +1.0000`、
> `knee_mid_L/knee_mid_R = -1.0000`、`ankle_L/ankle_R = -1.0000` ——
> 只有 ankle / knee_up / knee_mid 反号。写 keyframe 前必须先打印
> `jnt_axis` / `jnt_range`，按直觉写会错。

### 3.3 任务设计

10 个驱动关节：`hip_pitch / hip_roll / hip_yaw / knee_up / ankle` × 2。
被动的 `knee` / `knee_mid` **不参与**动作、姿态正则和超规格统计 ——
它们的值是四杆约束决定的，惩罚它们等于惩罚机构学。

动作缩放按 v03 真实限位逐关节定标（V025 的 0.35 rad 会把
`hip_roll`/`hip_yaw` 直接顶到 ±0.26 限位上）：

| 关节 | 缩放 | 限位 |
|---|---|---|
| hip_pitch | 0.50 | ±2.45 |
| knee_up | 0.40 | -1.22 ~ +0.52 |
| hip_roll | 0.20 | ±0.26 |
| hip_yaw | 0.20 | ±0.26 |
| ankle | 0.35 | -2.36 ~ +0.79 |

命令范围：`lin_vel_x (-0.4, 0.5)`、`lin_vel_y (-0.35, 0.35)`、
`ang_vel_z (-0.3, 0.3)`；`rel_forward/backward/lateral_envs = 0.25`。

### 3.4 怎么判断「在走」还是「站着不动」

**`Episode_Reward/track_*` 这个数本身会骗人，不能当进度指标。**

读实现（`tasks/velocity/mdp/rewards.py`）：

```python
lin_vel_error = |cmd_xy − actual_xy|² + actual_vz²
reward = exp(-lin_vel_error / std²)        # weight 2.0
```

奖励是 `exp(−误差²)` 的**逐帧均值**，而 `error_vel_xy` 指标是**时均误差**。
`exp(−x²)` 在 |x| > 1/√2 区间是凸的，由 Jensen 不等式，
「一半时间跟得准、一半时间跟得烂」也能刷出高分：

| 场景 | mean\|err\| | mean(exp(−err²/0.04)) |
|---|---|---|
| 恒定 err = 0.17 | 0.17 | 0.50 |
| err 在 0.05 / 0.90 间各半 | **0.475** | **≈0.47** |

所以 `track_linear_velocity = 1.00` 对应的**等效**误差只有 ~0.17，
而 `error_vel_xy` 报的是 0.42。两个数不矛盾，但**只有后者诚实**。

反过来，这个公式**精确验证了 r01 的站桩诊断**：站桩时 err ≈ mean|cmd| = 0.29、
默认 std = 0.5，

```
2 · exp(−0.29² / 0.25) = 1.43
```

而 **r01 实测 `track_linear_velocity` = 1.392 ~ 1.453** —— 分毫不差。
这就是「站桩白拿 72% 跟踪分」的精确版本。

**所以判断必须靠独立于奖励的物理量**：把命令固定成档位，跑固定步数，
量 base 在**世界系的净位移**除以时间（`scripts/eval_v03_velocity.py`）。
日常盯训练则只看 `Metrics/twist/error_vel_xy` 的**趋势**。

> ⚠️ 连 `error_vel_xy` 本身也有坑：它的归一化分母是**命令窗口最大长度**
> （`resampling_time_range[1] / step_dt = 8.0 / 0.02 = 400` 步），
> **不是实际步数**。所以 episode 短于 400 步时这个指标被系统性低估
> （iter 0 时 `ep_len = 23` → 低估 17 倍）。它只在 `ep_len > 400` 之后才诚实，
> 且跨 run 比较时若 `ep_len` 差得多就不可比。
>
> **最终结论一律以净位移评测为准，不用任何训练期指标。**

> ⚠️ 用净位移而不是瞬时速度：步态瞬时速度有周期性起伏，均值可信单帧不可信。
>
> ⚠️ **不要用「接触状态翻转次数」当步频**：胶囊脚底在软接触下 `found` 会
> 抖动，站着不动也能翻出 3~4 Hz 的假步频（实测踩过）。改用
> **单脚支撑占比** —— 真迈步稳定在 0.3~0.6，站桩接近 0。

## 4. r01：第一版学出「站着不动」

### 4.1 现象

配置：track std 用默认值（lin 0.5 / ang 0.707），抬脚奖励只有
`alternating_feet_air_time`。训练 500 迭代后逐档评测：

| 档位 | 命令 vx | 命令 vy | 命令 wz | 实测 vx | 实测 vy | 实测 wz | 跟踪比 | roll° | pitch° | 终止 | 单脚支撑 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stand | 0.00 | 0.00 | 0.00 | 0.001 | 0.000 | 0.001 | — | -0.6 | 3.0 | 0 | 0.00 |
| fwd | **0.40** | 0.00 | 0.00 | **0.001** | 0.000 | 0.001 | **0.00** | -0.6 | 3.0 | 0 | 0.00 |
| back | **-0.30** | 0.00 | 0.00 | **0.001** | 0.000 | 0.001 | **0.00** | -0.6 | 3.0 | 0 | 0.00 |
| lat | 0.00 | **0.25** | 0.00 | 0.001 | **0.000** | 0.001 | **0.00** | -0.6 | 3.0 | 0 | 0.00 |
| turn | 0.00 | 0.00 | **0.30** | 0.001 | 0.000 | **0.001** | — | -0.6 | 3.0 | 0 | 0.00 |

**五个档位实测速度全是 0.001 m/s、单脚支撑占比 0.00 —— 一次脚都没抬过。**
姿态却完美（roll -0.6°、pitch 3.0°、零终止、`episode_length` 顶满 1000）。

训练曲线上的对应信号：`error_vel_xy` 平在 0.68~0.73 **不降**，
`air_time = +0.0016`。

### 4.2 根因：两个**互相独立**的缺陷

**缺陷 1 —— 跟踪奖励的 std 没随命令范围缩放。**

`make_velocity_env_cfg` 的默认 std（lin `√0.25 = 0.5`、ang `√0.5 = 0.707`）
是配 G1 那种 ±3 m/s 大命令的。v03 命令收窄到 lin ±0.5 之后：

```
站桩误差 mean|cmd| ≈ 0.29 m/s
exp(-0.29² / 0.5²) = 0.72     →  站桩白拿 72% 的跟踪分
```

梯度压力消失，策略落向「站着不动」这个鞍点。

> 这不是新坑 —— AMP 侧踩过一模一样的（记忆 `amp-track-std-must-match-command`）。

**缺陷 2 —— `alternating_feet_air_time` 在「从不抬脚」时是死梯度。**

读 `tasks/velocity/mdp/rewards.py`：它只在**落地瞬间**
（`first_contact & has_other_support`）给分。策略一次脚都不抬 →
没有落地事件 → 该项恒为 0、**梯度也为 0**。光靠它推不出第一步。

同文件的 `feet_air_time` 是**每步**给分（`threshold_min < current_air_time
< threshold_max` 就计分），有活梯度。

### 4.3 修复

1. `track_linear_velocity.std`: 0.5 → **0.20**（站桩只剩 0.13 分）
   `track_angular_velocity.std`: 0.707 → **0.25**（站桩 0.70）
2. 抬脚奖励拆成两项，**缺一不可**：
   - `air_time` = `feet_air_time`，weight **1.0**，每步给分 → 脱离站桩鞍点
   - `alternating_air_time` = `alternating_feet_air_time`，weight **3.0**，
     只在落地给分 → 把步态收成真交替步（它不能靠双脚一起腾空刷分）
3. `pose.std_walking` 放宽：`hip_pitch`/`knee_up` 0.5、`ankle` 0.45 ——
   原值把正常迈步摆幅当误差罚，策略为保 pose 分不敢抬腿；
   `hip_roll`/`hip_yaw` 保持 0.15 收着。

> ⚠️ r01 的进程是**启动时**读的 `env_cfgs.py`，改文件对它无效，必须重启。

## 5. r02：修复后的训练

### 5.1 修复生效的证据

| 指标 | r01 @232（旧配置） | r02 @211（新配置） |
|---|---|---|
| `Metrics/twist/error_vel_xy` | 0.676 **平** | 0.592 → 0.516 → **0.456 降** |
| `Episode_Reward/air_time` | **+0.0016** | **+0.4537**（×280） |
| `Episode_Reward/alternating_air_time` | ~0 | +0.0836 |
| `Episode_Termination/fell_over` | 0.00 | 0.95 |
| `Train/mean_episode_length` | 1000（顶满） | 782 |

r01 是「站得完美但一动不动」，r02 是「真的在动、偶尔摔」——
后者才是学习中的形态。`air_time` 从 0.0016 涨到 0.45 直接证明死梯度被救活了。

<!-- RESULTS_PLACEHOLDER -->

## 6. 工程教训

### 6.1 训练期间绝不能再起第二个仿真进程

想把 r01 的 `model_500.pt` 录成「修复前」对照素材，于是在训练还在跑时启动了
录像脚本。结果：

```
[record_v03] ok stand__r01-before.mp4  300 帧  用时 2393.3s
```

**8 秒/帧**。同一套代码单独基准测是 **render 12.4 ms/帧、step 20.2 ms/帧**，
差 **600 倍**。同一时间训练从 0.72 s/迭代 掉到 **30 分钟只走 6 个迭代**。

不是 GPU 争用（GPU 只用 34%、2.1 GB / 8 GB），是**内存**：`free -g` 显示
14 GB 里已用 11 GB、可用 3 GB，**swap 已满**。杀掉录像进程后训练立刻恢复
（45 秒 54 个迭代）。

**规矩**：录像、评测这类要再建仿真环境的活，一律等训练停下来再做。

### 6.2 `play.py` 无头录不出像

`play.py` 最后把控制权交给 `viewer/base.py` 的 `run(num_steps=None)`，
那是**无限循环**：

- `DISPLAY` 存在 → auto 选 `NativeMujocoViewer` → 在桌面上真开一个 GUI 窗口
- `DISPLAY` 清空 → auto 选 viser → 起 web 服务但**停在暂停态**，仿真不推进
  （`step_trigger=lambda step: step == 0` 的录像因此永远不触发）

两条路都录不出东西。改走离线渲染（`scripts/record_v03_video.py`）：
自己 `env.render()` 取帧 + `imageio` 写 mp4。

### 6.3 tyro 的 bool 参数要显式给值

`--video` 会报 `Missing value for argument ('--video',)`，必须写 `--video True`。

### 6.4 命令固定的正确做法

不要去 poke `cmd.vel_command_b`，而是**在构造 env 之前**把 `twist` 的
ranges 收成一个点（`lin_vel_x=(vx, vx)`），并把所有 `rel_*_envs` 归零。
这样 `_resample_command` 每次抽出来就是同一个值，观测里的 command 项也
天然自洽。

> ⚠️ 两个必须同时做的细节：
> 1. `rel_forward_envs` 必须归零 —— `_resample_command` 对 forward env 会把
>    x 钳到 `min=0.3`，留着它后退档会被钳成正的。
> 2. `heading_command=False` 时 `ranges.heading` 必须设成 `None`，
>    否则 `UniformVelocityCommand.__init__` 直接抛 `ValueError`。
