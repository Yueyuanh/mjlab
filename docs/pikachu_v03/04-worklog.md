# 04 · 工作日志

> 按时间**倒序**。每条记「做了什么 / 为什么 / 结论」。

---

## 2026-10-01 · 资产接入 + 静平衡标定（完成）

### 目标
把 `pikachu_v03_base_series` 接入 mjlab，新建 velocity 任务，训到稳定行走。

### 环境约束（先摸清楚，后面所有决策都受它影响）
- 内存 14 GB，可用约 5~6 GB，**swap 已满** → 同一进程里建两个 mjlab 环境
  会 OOM 被 SIGKILL（退出码 137）。
- 磁盘 468 GB **已用 93%，剩 35 GB** → 需要持续盯。
- GPU 8 GB。

### 完成项

**1. 分支**
```bash
git checkout -b task/pikachu-v03-locomotion
```

**2. 训练 MJCF 生成** — `scripts/build_v03_train_xml.py`
源模型（`pikachu_urdf/pikachu_v03_base_series/`，含 meshes）→
`src/mjlab/asset_zoo/robots/pikachu/xmls/mjcf_v03/`。
改造 6 处（freejoint / geom 命名 / 足部 site+胶囊 / 删 scene include /
删 actuator / 补 sensor），详见 [01-model-conversion.md](01-model-conversion.md)。

验收：`nq=21 nv=20 nu=0 njnt=15 nbody=16 ngeom=40`，足部胶囊 10 个。

**3. 双腿独立解 + 静平衡固定点迭代** —
`scripts/probe_v03_both_legs.py`、`scripts/refine_v03_keyframe.py`
产出 `KNEES_BENT_KEYFRAME`，详见 [02-stand-keyframe.md](02-stand-keyframe.md)。
顺带实测确认了**左腿不是整体镜像**（只有 ankle / knee_up / knee_mid 反号）。

**4. velocity 任务** — `src/mjlab/tasks/velocity/config/pikachu_v03/`
注册 `Mjlab-Velocity-Flat-Pikachu-V03` / `Mjlab-Velocity-Rough-Pikachu-V03`
（`uv run list-envs` 确认为 #24 / #25）。
详见 [03-task-design.md](03-task-design.md)。

**5. `pikachu_metrics.py` 参数化** — 支持 v03 的 10 关节表与电机分组，
14dof 任务走默认值不受影响。

**6. 冒烟测试** — `scripts/smoke_v03_velocity.py`（分 `--quiet` / `--noisy`
两个进程跑，绕开 OOM）

| 检查项 | 结果 |
|---|---|
| 动作维度 | 10 ✓ |
| 关节数 | 14 ✓ |
| 执行器数 | 10 ✓ |
| 被动关节 | `knee` / `knee_mid` ×2 ✓ |
| IMU 帧一致性 | 偏差 **0.000e+00** ✓ |
| 观测空间 | `actor Box(N,50)` / `critic Box(N,62)` |
| **阶段 A 零动作开环站立** | base_z 0.26728→0.26731，\|v\|→0.00000，6 s 无终止 **✓ 通过** |
| 阶段 B 完整扰动 | 最低 0.177 m，从未跌破半高，无终止 |

### 关键发现（都是踩过的坑）

1. **左腿不是整体镜像**。按直觉写 `left_ankle = -0.5148` 是错的，
   实测是 `+0.5048`。写 keyframe 前必须先打印 `jnt_axis` / `jnt_range`。
2. **源模型必须删 `<actuator>`**，否则 `nu` 翻倍（mjlab 只追加不清理）。
3. **必须补 `<sensor>` 块**，且 **imu site 必须是单位四元数**
   （14dof 资产那个 107° 的 quat 不要照抄）。
4. **「脚在髋下」≠「静平衡」**。v03 质心在 base 原点后方 2 cm，
   零动作会先仰到 -0.167 rad 再被 PD 拉回，停在非零平衡点。
   必须做固定点迭代把脚心挪到质心正下方。
5. **`push_robot` 与 `reset_base` 的 z 随机化会污染开环测试** ——
   冒烟测试的 quiet 模式要把它们弹掉。

### 开训

```bash
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-Pikachu-V03 \
  --env.scene.num-envs 2048 --agent.run-name v03-flat-r01 --video True
```

- run 目录：`logs/rsl_rl/pikachu_v03_velocity/2026-10-01_12-17-25_v03-flat-r01`
- 训练 commit：`a1ae500`
- 吞吐 ~0.72 s/迭代（warmup 后），50 Hz 控制
- `--video True` 每 500 迭代录一段 200 步的训练录像
  （`VideoRecorder` 默认分辨率，不是最终的 720P 成品）

**踩坑**：tyro 的 bool 参数要显式给值 —— `--video` 会报
`Missing value for argument ('--video',)`，必须写 `--video True`。

### 训练早期观察（iter 271）

| iter | mean_reward | episode_length | track_lin | track_ang | error_vel_xy | fell_over |
|---|---|---|---|---|---|---|
| 0 | -1.91 | 23.3 | 0.011 | 0.001 | 0.013 | 7.05 |
| 116 | +44.3 | 897.4 | 1.475 | 0.542 | 0.639 | 0.00 |
| 232 | +84.9 | **1000.0** | 1.453 | 1.484 | 0.676 | 0.00 |
| 271 | +87.0 | **1000.0** | 1.392 | 1.533 | 0.731 | 0.00 |

- **好消息**：iter 232 起 `episode_length` 顶到 1000（20 s 上限），
  `fell_over = 0` —— 站得非常稳，环境设计（静平衡 keyframe）奏效。
- **待观察**：`error_vel_xy` 平在 0.68~0.73 **不降**。`track_linear_velocity`
  卡在 1.4/2.0 不动，既可能是「走得不错」，也可能是「站着不动拿到的基线分」。
  这是已知的局部最优形态，需要继续盯着 `error_vel_xy` 的趋势 ——
  若再跑 1000~2000 迭代仍平，就要动奖励权重（`pose` 降权 / track 升权）。

### 录像管线

`play.py` 无头必挂（末尾交给 `viewer.run()` 无限循环：有 DISPLAY 开 GUI 窗口
等关闭，无 DISPLAY 走 viser 停在暂停态不推进仿真）。改走离线渲染：
`scripts/record_v03_video.py`，自己 `env.render()` 取帧 + imageio 写 mp4，
命令通过在**构造 env 前**把 twist 的 ranges 收成点来固定。

### 下一步
盯 `error_vel_xy` 趋势；收敛后跑 `record_v03_video.py` 出 720P 成品。

---

## 2026-10-01 · r01 定性：**策略在站桩**，找到两个独立缺陷（已修，待复跑验证）

### 怎么定性的

看板上的 `track_linear_velocity = 1.4/2.0` 是**二义**的：命令幅度平均
~0.25 m/s 时，「站着不动」的跟踪误差本身就只有 0.25 左右，exp 项天然能拿到
0.7+。光看奖励分不出「走得不错」和「站着不动」。

所以新写了 `scripts/eval_v03_velocity.py`：把命令**固定成档位**，跑 8 s，
量 base 在**世界系的净位移**除以时间，与命令对比。

> ⚠️ 用净位移而不是瞬时速度 —— 步态瞬时速度有周期性起伏，均值可信单帧不可信
> （记忆 `amp-v4-behavior-direction`）。

对 `model_500.pt`（r01 中途存档）逐档评测：

| 档位 | 命令 vx | 命令 vy | 命令 wz | 实测 vx | 实测 vy | 实测 wz | 跟踪比 | roll° | pitch° | 终止 | 单脚支撑 | 平均脚数 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stand | 0.00 | 0.00 | 0.00 | 0.001 | 0.000 | 0.001 | — | -0.6 | 3.0 | 0 | 0.00 | 7.00 |
| fwd | **0.40** | 0.00 | 0.00 | **0.001** | 0.000 | 0.001 | **0.00** | -0.6 | 3.0 | 0 | 0.00 | 7.03 |
| back | **-0.30** | 0.00 | 0.00 | **0.001** | 0.000 | 0.001 | **0.00** | -0.6 | 3.0 | 0 | 0.00 | 6.99 |
| lat | 0.00 | **0.25** | 0.00 | 0.001 | **0.000** | 0.001 | **0.00** | -0.6 | 3.0 | 0 | 0.00 | 6.99 |
| turn | 0.00 | 0.00 | **0.30** | 0.001 | 0.000 | **0.001** | — | -0.6 | 3.0 | 0 | 0.00 | 7.01 |

**五个档位实测速度全是 0.001 m/s，单脚支撑占比 0.00 —— 一次脚都没抬过。**
姿态倒是完美（roll -0.6°、pitch 3.0°、零终止）。

> 📌 **迈步判据用「单脚支撑占比」，不要用「接触翻转次数」当步频**：
> 胶囊脚底在软接触下 `found` 会抖动，站着不动也能翻出 3~4 Hz 的假步频
> （实测踩过）。单脚支撑占比没这个毛病：真迈步稳定在 0.3~0.6，站桩接近 0。

### 根因：两个**互相独立**的缺陷

**缺陷 1 —— track 奖励的 std 没随命令范围缩放。**

`make_velocity_env_cfg` 给的默认 std（lin `√0.25 = 0.5`、ang `√0.5 = 0.707`）
是配 G1 那种 ±3 m/s 大命令的。v03 的命令收窄到 lin ±0.5 / ang ±0.3 之后：

```
站桩误差 mean|cmd| ≈ 0.29 m/s
exp(-0.29² / 0.5²) = 0.72     →  站桩白拿 72% 的跟踪分
```

梯度压力直接消失，策略落向「站着不动」这个鞍点。

> 这**不是新坑** —— AMP 侧踩过一模一样的，见记忆
> `amp-track-std-must-match-command`（Pikachu 取 lin 0.12 / ang 0.25）。

**缺陷 2 —— `alternating_feet_air_time` 在「从不抬脚」时是**死梯度**。**

翻它的实现（`tasks/velocity/mdp/rewards.py`）：它只在**落地瞬间**
（`first_contact & has_other_support`）给分。策略一次脚都不抬 → 没有落地事件
→ 该项恒为 0，**梯度也为 0**。光靠它推不出第一步（r01 实测 `air_time`
只有 +0.0016）。同文件的 `feet_air_time` 是**每步**给分（
`threshold_min < current_air_time < threshold_max` 就计分），有活梯度。

### 修复（`env_cfgs.py`）

1. `track_linear_velocity.std`: 0.5 → **0.20**；
   `track_angular_velocity.std`: 0.707 → **0.25**
   （收到命令上限量级：lin 站桩只剩 0.13 分，ang 站桩 0.70）。
2. 抬脚奖励拆成两项，**缺一不可**：
   - `air_time` = `feet_air_time`（每步给分，weight 1.0）→ 把策略推离站桩鞍点
   - `alternating_air_time` = `alternating_feet_air_time`（落地质量，weight 3.0）
     → 它不能靠「双脚一起腾空」刷分，负责把步态收成真交替步
3. `pose.std_walking` 放宽：`hip_pitch`/`knee_up` 0.5、`ankle` 0.45
   （原值把正常迈步摆幅当误差罚，策略为保 pose 分不敢抬腿）；
   `hip_roll`/`hip_yaw` 保持 0.15 收着。

### 复跑

r01 的进程是**启动时**读的 `env_cfgs.py`，改文件对它无效，必须重启。

```bash
kill 901627
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-Pikachu-V03 \
  --env.scene.num-envs 2048 --agent.run-name v03-flat-r02 --video True
```

**验收标准**（跑够 500 迭代后看）：
- `Metrics/twist/error_vel_xy` 明显下降（r01 平在 0.68~0.73）
- `Episode_Reward/air_time` 不再是 ~0
- 复跑 `eval_v03_velocity.py`，fwd 档跟踪比 > 0.5、单脚支撑 > 0.2

### r02 前 211 迭代：修复生效（形态和 r01 完全不同）

| 指标 | r01 @232（旧配置） | r02 @211（新配置） |
|---|---|---|
| `Metrics/twist/error_vel_xy` | 0.676 **平** | 0.592→0.516→**0.456 下降** |
| `Episode_Reward/air_time` | **+0.0016** | **+0.4537**（×280） |
| `Episode_Reward/alternating_air_time` | ~0 | +0.0836 |
| `Episode_Termination/fell_over` | 0.00 | 0.95 |
| `Train/mean_episode_length` | 1000（顶满） | 782 |

r01 是「站得完美但一动不动」，r02 是「真的在动、偶尔摔」—— 后者才是
学习中的形态。`air_time` 从 0.0016 涨到 0.45 直接证明缺陷 2 的死梯度被救活了。

到 iter 1540 继续往好的方向走：

| iter | ep_len | mean_reward | err_xy | err_yaw | air_time | alt_air | track_lin | fell_over |
|---|---|---|---|---|---|---|---|---|
| 211 | 782 | +28.5 | 0.456 | 2.21 | +0.454 | +0.084 | 0.487 | 0.95 |
| 1540 | **969** | **+48.3** | 0.420 | **2.10** | **+0.555** | **+0.127** | 1.00 | **0.05** |

`fell_over` 从 0.95 掉到 **0.05**、`episode_length` 969/1000 —— 步态在稳定下来；
`error_vel_yaw` 也**开始回落**（2.35 → 2.10），说明暂时不必动 ang std。

**待观察的隐患**：`error_vel_yaw` 从 0.058 涨到 **2.21 且仍在涨**，
而 `track_angular_velocity` 只有 +0.0077 —— std=0.25 下
`exp(-(2.21/0.25)²) ≈ e⁻⁷⁸ ≈ 0`，这一项已经**饱和到零梯度**。
若再跑 2000 迭代仍不回落，要把 ang std 放宽（例如 0.4）或改线性跟踪项。

### ⚠️ 重要：`Episode_Reward/track_*` **本身会骗人**，别拿它当进度指标

读实现（`tasks/velocity/mdp/rewards.py`）：

```python
lin_vel_error = |cmd_xy − actual_xy|² + actual_vz²
reward = exp(-lin_vel_error / std²)        # weight 2.0
```

奖励是 `exp(−误差²)` 的**逐帧均值**，而 `error_vel_xy` 是**时均误差**。
`exp(−x²)` 在 |x| > 1/√2 区间是凸的，由 Jensen 不等式，
「一半时间跟得准、一半时间跟得烂」也能刷出高分：

| 场景 | mean\|err\| | mean(exp(−err²/0.04)) |
|---|---|---|
| 恒定 err = 0.17 | 0.17 | 0.50 |
| err 在 0.05 / 0.90 间各半 | **0.475** | **≈0.47** |

**所以 `track_linear_velocity = 1.00` 对应的等效误差只有 ~0.17，
而 `error_vel_xy` 报的是 0.42。两个数不矛盾，但只有后者诚实。**

反过来，这个公式**反过来验证了 r01 的站桩诊断**：
站桩时 err ≈ mean\|cmd\| = 0.29、默认 std = 0.5

```
2 · exp(−0.29² / 0.25) = 1.43
```

而 **r01 实测 `track_linear_velocity` = 1.392 ~ 1.453** —— 分毫不差。
这就是「站桩白拿 72% 跟踪分」那句话的精确版本。

**纪律**：判断「在走还是在站」，只看 `Metrics/twist/error_vel_xy` 的**趋势**
和 `eval_v03_velocity.py` 的**净位移**，不看奖励分。

### ⚠️ 重大教训：**训练期间绝不能再起第二个仿真进程**

想把 r01 的 `model_500.pt` 录成「修复前」对照素材，于是训练还在跑时启动了
`record_v03_video.py`。结果：

```
[record_v03] ok stand__r01-before.mp4  300 帧  用时 2393.3s
```

**8 秒/帧**。而单独基准测同一套代码是 **render 12.4 ms/帧、step 20.2 ms/帧**，
差 **600 倍**。同一时间训练从 0.72 s/迭代 掉到 **30 分钟只走 6 个迭代**。

不是 GPU 争用（GPU 只用了 34%、2.1 GB / 8 GB），是**内存**：
`free -g` 显示 14 GB 里已用 11 GB、可用 3 GB，**swap 已满** —— 训练那 2048 个
环境几乎吃光内存，第二个进程一进来就开始疯狂换页。

杀掉录像进程后训练立刻恢复（45 秒走 54 个迭代 = 0.83 s/迭代）。

**规矩**：录像、评测这类要再建仿真环境的活，一律等训练**停下来**再做。
