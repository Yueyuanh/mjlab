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

---

## 2026-10-01 · ⚠️ 抓到一个极隐蔽的 bug：命令固定**根本没生效**

### 症状

修好配置后停训练、跑 `eval_v03_velocity.py` 评 `model_2500.pt`，五个档位
输出**一模一样**：

```
stand    0.00  0.00  0.00 | -0.062  0.033  0.232 |  nan  5.2  -0.9  0  0.15
fwd      0.40  0.00  0.00 | -0.062  0.033  0.232 | 0.18  5.2  -0.9  0  0.15
back    -0.30  0.00  0.00 | -0.062  0.033  0.232 | 0.23  5.2  -0.9  0  0.15
lat      0.00  0.25  0.00 | -0.062  0.033  0.232 | 0.28  5.2  -0.9  0  0.15
turn     0.00  0.00  0.30 | -0.062  0.033  0.232 |  nan  5.2  -0.9  0  0.15
```

**这个形态极具误导性** —— 看起来像「策略完全忽略命令」，实际上是我把命令
写进去的那一步失败了。

### 排查路径（值得记，因为每一步的结论都曾看起来"没问题"）

1. 先怀疑 `ranges` 是只读属性 → 打印 `type(term.cfg.ranges)`，是普通
   dataclass，赋值能读回来 ✓
2. 怀疑 env 拿的不是我改的那个对象 → 打印 `id()`，`term.cfg is t` → **True** ✓
3. 怀疑 `_resample_command` 被子类重写 → 打印 `__qualname__`，就是
   `UniformVelocityCommand._resample_command`，且全文只用 `self.cfg.ranges` ✓
4. **打桩**：包一层 `_resample_command`，在**调用时刻**打印 `self.cfg.ranges`

```
打桩前 term.cfg.ranges.lin_vel_x = (0.4, 0.4)
>>> INSIDE self.cfg.ranges.lin_vel_x = (-0.4, 0.5)   ← 变回默认值了！
```

### 根因

`cfg.curriculum["command_vel"]`（`tasks/velocity/mdp/curriculums.py::commands_vel`）
**每一步都用 stage 配置覆写 `cfg.ranges.*`**：

```python
for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]   # ← 直接写
```

stage 0 的 `step=0` **恒满足**，所以在第一次 `_resample_command` 之前，
我构造 env 前设的 `(vx, vx)` 就被改回 `(-0.4, 0.5)`。
五个档位于是全都用同一个默认范围抽样（seed 固定 → 同一个命令
`[0.072, -0.121, -0.072]`），结果当然一模一样。

### 修复

`eval_v03_velocity.py` 和 `record_v03_video.py` 里都要加：

```python
cfg.curriculum.pop("command_vel", None)
```

修完验证：

```
设定 vx=+0.40 -> env 命令 = [0.4, 0.0, 0.0]
设定 vx=-0.30 -> env 命令 = [-0.3, 0.0, 0.0]
设定 vx=+0.00 -> env 命令 = [0.0, 0.0, 0.0]
```

### 连带影响

**r01 那张「五档全 0.001 m/s」的表，命令列是错的** —— 那五档实际跑的都是
同一个命令 `[0.072, -0.121, -0.072]`（vx = +0.072，不是我写的 0.40）。
但**结论不受影响**：机器人被命令前进 0.072 m/s，实测 0.001 m/s，
依然是「一动不动」。只是「跟踪比 0.00」的分母得改成 0.072。
修复后已重跑 r01 基线，见下一节。

> **教训**：脚本里「构造 env 前改 cfg」这种做法，必须确认**没有任何 manager
> 会在运行期回写同一个 cfg 字段**。课程项就是这样一个回写者，而且它写的是
> 默认值，所以症状表现为「我设的值被静默丢弃」而不是报错。

---

## 2026-10-01 · ✅ r02 @ 2500 评测：**策略在走**（修复成功）

命令固定修好后重跑评测。**判据加了 base 系时均速度与方向余弦** ——
原来只看世界系净位移，被机器人自身的偏航搞歪了（见下）。

### r02 @ iter 2500（新配置）

| 档位 | 命令 vx,vy,wz | base_vx | base_vy | base_wz | 跟踪比 | 方向cos | roll° | pitch° | 终止 | 单脚支撑 |
|---|---|---|---|---|---|---|---|---|---|---|
| stand | 0, 0, 0 | 0.002 | -0.001 | 0.008 | — | — | 1.9 | 3.3 | 0 | 0.00 |
| **fwd** | 0.40, 0, 0 | **0.415** | 0.002 | 0.330 | **1.04** | **1.00** | -0.4 | 0.5 | 0 | **0.76** |
| **back** | -0.30, 0, 0 | **-0.304** | -0.030 | 0.217 | **1.02** | **1.00** | -2.3 | 3.7 | 0 | 0.25 |
| lat | 0, 0.25, 0 | -0.004 | 0.102 | 0.273 | 0.41 | 1.00 | 4.0 | 2.3 | 0 | 0.10 |
| turn | 0, 0, 0.30 | 0.001 | 0.000 | **0.005** | — | — | 1.7 | 3.3 | 0 | 0.01 |

### r01 @ 500（基线，命令修正后重测）

五档 base 系速度全在 ±0.003 以内、单脚支撑 **0.00** —— 命令修正后
「一动不动」的结论**依然成立**（且现在命令是真的 0.40 而不是 0.072）。

### 结论

- **前进 104%、后退 102%、方向余弦 1.00、零终止** —— 真在走，且方向正确。
- **侧移只有 41%**、**原地转向完全不会**（命令 0.30 → 实际 0.005）。
- **行走时存在未被命令的偏航 0.2~0.33 rad/s** —— 机器人走的是弧线不是直线。

### ⚠️ 世界系净位移会骗人（加 base 系列的原因）

`fwd` 档的世界系净位移是 **-0.221**（看着像倒着走），但 base 系是 **+0.415**。
原因：机器人一边前进一边以 0.33 rad/s 偏航，8 秒转 149°，世界系位移自然绕回来。

> **教训**：机器人会转身时，「世界系净位移」不再等于「base 系前进」。
> 必须同时看 **base 系时均速度**（与命令同系，可直接比）和**方向余弦**
> （+1 走对、-1 反向、0 垂直）。步态瞬时速度有周期性起伏，但 8 s 的均值
> 不受影响 —— 这才是「用均值不用单帧」的正确用法。

### 为什么不会转向（诊断）

1. **命令分布把 yaw 饿死了**：三个方向桶
   （`rel_forward/backward/lateral_envs = 0.25` 各一份）在
   `_resample_command` 里**都把 `ang_vel_z` 强制置 0**。
   ⚠️ 这里一开始我算错了，写成「只剩 25% 的环境能见到 yaw」——
   **真实值只有 6.79%**，因为还要再乘两项
   （`is_heading_env &= ~directional` 和 standing 清零整个命令）。
   精确推算见本文档下方「⚠️ 修正」一节。
2. **`pose` 奖励在罚转向**：`hip_yaw` 的 `std_walking` 设的是 0.15，
   而 `hip_yaw` 的机械限位只有 **±0.26** —— 转向必须动 `hip_yaw`，
   一动就被 pose 当误差罚，策略自然选择不动。
   （`hip_roll` 同理，限位 ±0.26 而 std 只有 0.15，这也是侧移只有 41% 的原因。）
3. **`track_angular_velocity` 的 std=0.25 偏窄**：yaw 误差落在
   0.2~0.6 区间时 `exp(-err²/0.0625)` 已经很小、梯度接近 0。
   对比 s=0.25 与 s=0.40 在 err=0.5 处的梯度：-0.29 vs -1.31。

> 顺带确认：抬脚奖励对纯转向命令是**激活**的 ——
> `feet_air_time` 的门控用 `norm(lin) + |ang|`，纯转向 0.30 > 阈值 0.15。

### r03 计划

针对上面三条，同时改：

| 项 | r02 | r03 |
|---|---|---|
| `track_angular_velocity.std` | 0.25 | **0.40** |
| `rel_forward_envs` | 0.25 | **0.20** |
| `rel_backward_envs` | 0.25 | **0.15** |
| `rel_lateral_envs` | 0.25 | **0.15** |
| `pose.std_walking[hip_yaw]` | 0.15 | **0.30** |
| `pose.std_walking[hip_roll]` | 0.15 | **0.25** |

方向桶总量 0.75 → 0.50，见过 yaw 命令的环境 6.79% → **13.49%**（见下方修正一节）。

---

## r03：给「转向」让路

### 启动

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
  uv run train Mjlab-Velocity-Flat-Pikachu-V03 \
  --env.scene.num-envs 2048 --agent.run-name v03-flat-r03 --video True
```

- run 目录：`logs/rsl_rl/pikachu_v03_velocity/2026-10-01_14-11-08_v03-flat-r03`
- 真实 python PID **963580**（`$!` 拿到的是 bash 包装器 963575，别用）
- 启动后 `0.71 s/iter`，2048 envs，ETA ≈ 7 h

### 为什么改这三处

r02@2500 的评测（命令固定修好之后）显示：

| 档位 | 命令 | base_vx | base_vy | base_wz | 跟踪比 | 方向cos | 单脚支撑 |
|---|---|---|---|---|---|---|---|
| stand | 0,0,0 | 0.002 | -0.001 | 0.008 | — | — | 0.00 |
| fwd | 0.40,0,0 | **0.415** | 0.002 | 0.330 | **1.04** | **1.00** | **0.76** |
| back | -0.30,0,0 | **-0.304** | -0.030 | 0.217 | **1.02** | **1.00** | 0.25 |
| lat | 0,0.25,0 | -0.004 | 0.102 | 0.273 | 0.41 | 1.00 | 0.10 |
| turn | 0,0,0.30 | 0.001 | 0.000 | **0.005** | — | — | 0.01 |

前进/后退已经 100% 跟踪、方向余弦满分、单脚支撑 0.76 —— **「会不会走」这一问已经有答案：会走**。
剩下三个缺陷，按危害排序：

1. **完全不会转向**（命令 0.30 → 实测 0.005，1.7%）。
2. **行走时带 0.2~0.33 rad/s 的未受命令偏航** —— 前进档每 8 s 白转 149°。
3. 侧移只有 41%。

三条根因各自独立，r03 一次全改：

| 项 | r02 | r03 | 依据 |
|---|---|---|---|
| `track_angular_velocity.std` | 0.25 | **0.40** | yaw 误差实测落在 0.2~0.6；std=0.25 时 `exp(-err²/0.0625)` 在该区间梯度已趋零（err=0.5 处 -0.29），0.40 时是 -1.31 |
| `rel_forward_envs` | 0.25 | **0.20** | 三个方向桶在 `_resample_command` 里**都把 `ang_vel_z` 置 0**，桶占比 = 见不到 yaw 命令的环境占比 |
| `rel_backward_envs` | 0.25 | **0.15** | 同上。前进/后退已 100% 跟踪，不需要那么宽覆盖 |
| `rel_lateral_envs` | 0.25 | **0.15** | 同上 |
| `pose.std_walking[hip_yaw]` | 0.15 | **0.30** | v03 的 `hip_yaw` 机械限位只有 **±0.26**，而转向必须动它 —— 给 0.15 等于「一动就罚」 |
| `pose.std_walking[hip_roll]` | 0.15 | **0.25** | 同理，侧移必须动 `hip_roll` |

方向桶总量 0.75 → 0.50：见过转向命令的环境 6.79% → 13.49%
（**不是** 25% → 50%，见下方修正一节）。

### 已排除的嫌疑

`feet_air_time` 的 docstring 让人怀疑「纯转向命令会不会推不出抬脚奖励」：
它的门控是 `total_command = norm(lin) + |ang| > command_threshold(0.15)`。
读了源码：纯转向命令 `0 + 0.30 = 0.30 > 0.15`，**门是开的**。不是阻碍项。

### r03 的验收标准（跑够 2500 迭代后重跑 eval）

- `turn` 档 `base_wz` 明显离开 0（目标 > 0.15，r02 是 0.005）
- `fwd` / `back` 档的 `base_wz` 从 0.33 / 0.22 降下来（目标 < 0.15）
- `fwd` 跟踪比不要掉（r02 = 1.04，允许因让位给转向而降到 0.8 以上）
- 三档 `单脚支撑` 保持 > 0.2

### r03 vs r02 同期对比（iter 50~250）

单看 r03 的绝对数值会误判 —— `err_yaw` 在 iter 100 冲到 2.1 看着像「转向越练越差」，
但 r02 同期也是 2.37。**必须同期对比**：

| 指标 @iter | r02 | r03 |
|---|---|---|
| `error_vel_xy` | 0.459 | **0.464** |
| `error_vel_yaw` | 2.367 | **2.193** |
| `Episode_Reward/air_time` | 0.458 | **0.458** |
| `mean_episode_length` | 849 | 800 |
| `Episode_Termination/fell_over` | **0.71** | 1.43 |

结论：**没有回归**，r03 基本贴着 r02 的轨迹走，`err_yaw` 还略好一点。

唯一要盯的是 `fell_over` 高了一倍（1.43 vs 0.71）。猜测是 `hip_yaw` / `hip_roll`
的 pose std 放宽（0.15 → 0.30 / 0.25）把步态放松了，策略在探索期更敢乱动。
**这不是立刻要改的事** —— 早期跌倒率本来就震荡（r02 在 iter 100 是 1.88、
iter 150 才降到 1.25），到 2500 迭代看终评。若终评跌倒率仍显著高于 r02，
再考虑给这两项加回一点约束，或者用 `pose.std_running`（目前 hip_yaw/roll = 0.2）
在高速段兜底。

### ⚠️ 修正：能见到 yaw 命令的环境不是 25%，是 **6.79%**

前面写「r02 用 0.25×3 = 0.75，只剩 25% 的环境练转向」—— **这个数是错的**。
`1 − 桶总量` 只是第一步，还要连乘两件事：

```python
self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs  # 0.3
...
directional = is_forward | is_backward | is_lateral
self.is_heading_env[env_ids] &= ~directional        # ← 方向桶被踢出 heading
...
# _update_command 的最后一步：
self.vel_command_b[standing_env_ids, :] = 0.0       # ← standing 把整个命令清零
```

所以：

```
P(见过 yaw) = [ rel_heading_envs × (1 − 桶总量) ] × (1 − rel_standing_envs)
```

**只有 heading 环境会拿到非零的 `ang_vel_z`** —— 方向桶把它置 0，
standing 环境每步把整个命令清零，两个都轮不到。

| 轮次 | 桶总量 | heading | standing | 见过 yaw 的环境 |
|---|---|---|---|---|
| r01 / r02 | 0.75 | 0.3 | 0.1 | **6.79%**（139 / 2048） |
| r03 | 0.50 | 0.3 | 0.1 | **13.49%**（276 / 2048） |
| r04 预案 | 0.50 | **0.6** | **0.05** | **28.55%**（585 / 2048） |

（用 2×10⁶ 次采样模拟确认，不是纸面估算。）

**这条把「完全不会转」解释圆了**：`track_angular_velocity` 权重 2.0，
但只有 6.79% 的环境因它产生非零梯度 → 在平均奖励里的最大贡献
`2.0 × 0.068 = 0.14`，而 `track_linear_velocity` 是**全部**环境贡献的 2.0。
**信号弱 15 倍**，策略忽略 yaw 是理性的 —— 不是「学不会」，是「不值得学」。

r03 把暴露面翻倍到 13.49%。同时它还有两个独立杠杆（ang std 0.25→0.40
使 err=0.5 处梯度 ×4.5；pose 不再罚 hip_yaw），三者相乘。
**若 r03 终评转向仍不达标，r04 直接动暴露面**（上面第三行）。

### 又排除两个嫌疑

| 嫌疑 | 结论 |
|---|---|
| `body_ang_vel`（权重 −0.05）会不会罚转向？ | **不会**。`rewards.py::body_angular_velocity_penalty` 里 `ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.` 明确排除 z 轴 |
| `angular_momentum`（权重 −0.02）罚全向量模长、含 z，会不会罚转向？ | **量级可忽略**。转身时 `L_z ≈ I·ω ≈ 0.02 × 0.3 = 0.006`，`L² × 0.02 ≈ 7e-7`，相对 `track_angular` 的 2.0 是噪声 |
