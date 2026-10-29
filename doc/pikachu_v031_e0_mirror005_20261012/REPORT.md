# Pikachu V031 E0-Mirror005 训练评估报告

**对象**：`logs/rsl_rl/pikachu_v031_e0_mirror005/2026-10-12_00-11-23_mirror005-s42`<br>
**任务**：`Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror005`<br>
**checkpoint**：`model_49999.pt`<br>
**训练条件**：4096 env、50,000 iterations、seed 42、`mirror_loss_coeff=0.05`<br>
**日期**：2026-10-12

## 结论摘要

这次实验解决了 E0 中最直观的左右 `hip_roll` **平均幅值不对称**，但没有得到
健康、可部署的步态。

1. 在 0.4 m/s、nominal、确定性策略的 8 秒回放中，左右 `hip_roll` 绝对均值从
   E0 的 `0.1632 / 0.0711 rad` 变为 `0.10154 / 0.10156 rad`；两侧差值由
   **0.09208 rad 降至 0.000016 rad**，改善超过 99.98%。摆幅比为 0.849，也在
   预设的 0.8–1.25 区间内。
2. 直行能力保留：0.4 m/s 命令下实际前向速度为 **0.416 m/s**，8 秒内没有终止。
3. **偏航问题没有解决。** 零 yaw 命令下 yaw-rate RMS 为 **1.049 rad/s**，比
   E0 最终模型的 0.883 rad/s 还高约 18.8%；8 秒航向变化为 −22.1°。平均 yaw
   接近零只是左右快速摆动相互抵消，不能作为稳定证据。
4. **无界动作问题变严重。** 左右 `hip_roll` 目标越过 ±0.26 rad 的比例为
   38.7% / 40.8%，raw action 中 `|a|>1` 的比例为 36.9%，最大值 3.32。该模型
   把 E0 的“非对称饱和”变成了“对称饱和”。
5. +0.3 rad/s 原地转向命令下，实际平均 yaw 只有 −0.0002 rad/s，8 秒只转动
   +2.1°，说明转向命令基本被忽略。
6. `model_500` 已经把 `hip_roll` 幅值差降到 0.0038 rad；继续训练至 50k 没有
   改善 yaw，目标越界率反而从 26.3% 上升到 39.7%。因此**不建议继续训练
   Mirror010（0.10）**。当前瓶颈不再是镜像系数，而是动作分布/目标边界和偏航奖励。

综合判定：`mirror_loss_coeff=0.05` 对左右平均幅值对称是有效的，而且强度已经足够；
但本 checkpoint **不通过部署验收**。

## 评测口径

- actor 使用确定性推理，单环境，seed 42，每项 8 秒、400 samples；统计跳过开头
  0.5 秒瞬态。
- `nominal` 关闭 startup 摩擦/编码器/base COM 随机化、观测噪声和周期 push。
- `startup DR` 只恢复启动随机化；`train conditions` 同时恢复启动随机化、观测噪声
  和周期 push。
- 正式回放在 CPU 上执行，原因是评测时 GPU 正在运行另一个训练任务，避免改变其
  状态。混沌步态的单次航向角对 seed、设备和相位敏感，因此结论优先看 yaw RMS、
  越界率和多条件一致性，而不是只看一次净航向角。
- E0 对照沿用原报告的同 seed、0.4 m/s nominal 数据；它与本次 CPU 回放的精确
  浮点轨迹可能有细小设备差异。

## 数据、曲线与视频

### 1. 训练全程

![训练曲线](plots/training_curves.png)

运行完整结束，包含 `model_0.pt`、每 500 轮 checkpoint 到 `model_49500.pt`，以及
最终 `model_49999.pt`，共 101 个 `.pt`；训练视频共 100 段。训练用时约 11 小时
38 分钟。

最后 1000 轮均值：

| 指标 | 数值 |
|---|---:|
| mean reward | 55.935 |
| mean episode length | 996.69 / 1000 steps |
| policy mean std | 0.2681 |
| mirror loss | 0.0000877 |
| 左 / 右 hip_roll 实际最大绝对角 | 0.26218 / 0.26227 rad |
| 左 / 右 hip_roll 目标最大绝对角 | 0.92676 / 0.92483 rad |
| `dof_pos_limits` episode reward | −0.000489 |
| `torque_soft_limit` episode reward | −0.006273 |
| `track_linear_velocity` episode reward | 1.3456 |
| `track_angular_velocity` episode reward | 0.1607 |

mirror loss 已收敛，左右实际最大角也完全接近；但左右目标最大值都约为硬限位的
3.56 倍。`dof_pos_limits` 几乎为零，只说明实际关节被物理限位挡住，不能证明目标
健康。

[训练筛选标量 CSV](training_selected_scalars.csv)

### 2. 最终模型 0.4 m/s 直行

![最终直行诊断](plots/final_rollout_diagnostics.png)

| 指标 | model_49999 nominal |
|---|---:|
| 实际 vx / vy | 0.4161 / 0.0037 m/s |
| yaw-rate 均值 / RMS | −0.0010 / **1.0486 rad/s** |
| 8 秒航向变化 | **−22.13°** |
| 左 / 右 hip_roll 有符号均值 | +0.02206 / −0.01227 rad |
| 左 / 右 hip_roll 绝对均值 | 0.10154 / 0.10156 rad |
| 绝对均值差 | **0.000016 rad** |
| 左 / 右摆幅 | 0.3131 / 0.3688 rad |
| 摆幅比（左/右） | 0.8490 |
| 左 / 右目标最大绝对值 | 0.5643 / 0.5745 rad |
| 左 / 右目标越 ±0.26 rad | **38.67% / 40.80%** |
| raw action 最大绝对值 | 3.3199 |
| raw action `|a|>1` | **36.88%** |

视频：

- [侧视，0.4 m/s nominal，6 秒](videos/fwd__side__model49999_nominal.mp4)
- [正视，0.4 m/s nominal，6 秒](videos/fwd__front__model49999_nominal.mp4)

两段视频均为 H.264、1280×720、50 fps、300 帧。逐帧 CSV 位于
[`metrics/final_nominal/`](metrics/final_nominal/)。

### 3. 与原 E0 的直接对比

![E0 与 Mirror005 对比](plots/e0_vs_mirror005.png)

| 0.4 m/s nominal 指标 | E0 m29999 | Mirror005 m49999 | 判定 |
|---|---:|---:|---|
| 实际 vx | 0.4215 | 0.4161 | 基本保持（−1.3%） |
| yaw RMS | **0.8830** | **1.0486** | 变差 18.8% |
| 航向变化 | −76.66° | −22.13° | 净值改善，但仍失败且相位敏感 |
| hip_roll 绝对均值 左/右 | 0.1632 / 0.0711 | 0.10154 / 0.10156 | 明显改善 |
| hip_roll 绝对均值差 | **0.09208** | **0.000016** | 改善 >99.98% |
| hip_roll 摆幅比 | 0.987 | 0.849 | 均通过 |
| 目标越界率 左/右 | 20.8% / 32.5% | 38.7% / 40.8% | 对称但整体更差 |
| raw action `|a|>1` | 30.83% | 36.88% | 变差 |

因此 Mirror005 确实改掉了 E0 的左右 DC/幅值偏置，但不是通过减小动作完成，而是
让两侧采用相近的高幅度饱和控制。

### 4. checkpoint 演化

![checkpoint 演化](plots/checkpoint_progression.png)

全部为 seed 42、0.4 m/s nominal：

| checkpoint | 实际 vx | hip_roll 绝对均值差 | yaw RMS | 航向变化 | 平均目标越界率 | raw `|a|>1` |
|---|---:|---:|---:|---:|---:|---:|
| m500 | 0.4010 | 0.00381 | 1.053 | −0.99° | 26.27% | 30.83% |
| m1000 | 0.3905 | 0.01110 | 0.967 | +13.25° | 24.00% | 26.99% |
| m5000 | 0.3977 | 0.00067 | 1.179 | +6.07° | 42.27% | 29.52% |
| m10000 | 0.3982 | 0.00040 | 1.029 | −17.25° | 48.80% | 31.87% |
| m30000 | 0.4071 | 0.00363 | 1.147 | −12.10° | 41.60% | 32.21% |
| m49999 | 0.4161 | 0.00002 | 1.049 | −22.13° | 39.73% | 36.88% |

没有任何抽查 checkpoint 达到 `yaw RMS <0.25 rad/s`。m500 已满足速度和左右幅值
对称门槛，本应在早期评估时因为 yaw RMS 失败而停止，而不是继续消耗完整 50k 预算。

### 5. 最终模型命令覆盖

![最终命令覆盖](plots/command_suite.png)

| 命令 | 实际主速度 | yaw RMS / tracking RMS | 航向变化 | hip_roll 幅值差 | 平均目标越界率 |
|---|---:|---:|---:|---:|---:|
| stand | vx 0.0004 | 0.0001 | −0.02° | 0.00002 | 0% |
| fwd +0.4 m/s | vx 0.4161 | 1.0486 | −22.13° | 0.00002 | 39.73% |
| fwd +0.5 m/s | vx 0.5079 | 0.9181 | +14.61° | 0.00128 | 51.73% |
| back −0.3 m/s | vx −0.3151 | 0.8106 | −7.84° | 0.00015 | 49.47% |
| lateral +0.25 m/s | vy 0.2384 | 1.0736 | −40.93° | 0.00305 | 54.00% |
| turn +0.3 rad/s | wz −0.0002 | 0.3002 | +2.12° | 0.00348 | 0% |

静止表现健康，但一旦进入平移步态就出现约 0.8–1.1 rad/s 的 yaw 摆动；侧移的
raw action 最大值达到 4.23。转向测试表明策略不是“转向略有误差”，而是几乎没有
执行 yaw 命令。

### 6. nominal、DR 与训练条件

| 条件，0.4 m/s | 实际 vx | lateral RMS | yaw RMS | 航向变化 | hip_roll 幅值差 | 平均目标越界率 |
|---|---:|---:|---:|---:|---:|---:|
| nominal | 0.4161 | 0.0430 | 1.0486 | −22.13° | 0.00002 | 39.73% |
| startup DR | 0.4245 | 0.0531 | 0.9646 | +101.03° | 0.00541 | 39.60% |
| train conditions | 0.4193 | 0.1081 | 1.0601 | +0.18° | 0.00357 | 46.40% |

三种条件均完成 8 秒，且左右幅值差仍小，说明镜像效果不是只存在于 nominal。
但是 yaw RMS 始终接近 1 rad/s；`train conditions` 的 +0.18° 净航向变化只是正负
摆动在这次窗口内抵消，不能解读为偏航已修复。

用户已有的 0.2 m/s 手动记录也给出相同结论：8.58 秒、实际 vx 0.192 m/s、
hip_roll 幅值差 0.0124 rad，但 yaw RMS 0.923 rad/s、航向变化 +124.94°。这进一步
说明净航向变化具有相位/条件敏感性，而高 yaw RMS 是稳定复现的问题。

## 验收结果

| 指标 | 门槛 | Mirror005 m49999 | 结果 |
|---|---:|---:|---|
| 直行实际 vx | ≥0.30 m/s | 0.416 m/s | 通过 |
| hip_roll 绝对均值差 | <0.03 rad | 0.000016 rad | 通过 |
| hip_roll p2p 比 | 0.8–1.25 | 0.849 | 通过 |
| 8 秒 heading drift | <10° | 22.13° | **失败** |
| 零 yaw 命令 yaw RMS | <0.25 rad/s | 1.049 rad/s | **失败** |
| 处理目标越 ±0.26 rad | 0% | 39.73% | **失败** |
| raw action `|a|>1` | ≤10%（诊断门槛） | 36.88% | **失败** |
| +0.3 rad/s 转向跟踪 | 应有同向明显响应 | −0.0002 rad/s | **失败** |

最终结论：**左右对称子目标通过，整体步态和控制安全性不通过。**

## 因果解释与配置混杂

### 可以确认的内容

- 本 checkpoint 的策略输出在 nominal、startup DR 和训练条件下都表现出很小的
  左右 hip_roll 幅值差。
- mirror loss 已收敛，且对称效果从早期 checkpoint 开始就存在。
- 模型仍依赖无界动作和越界位置目标，yaw 摆动与转向失效没有被 mirror loss 修复。

### 不能把 E0 对比解释成纯 mirror-loss 消融

该 run 保存的 `git/mjlab.diff` 显示，训练时除加入 `mirror_loss_coeff=0.05` 外，
`pikachu_v031_train.xml` 还同时把左侧 `hip_yaw_link` 质量从 **0.323365 kg** 改为
**0.333676 kg**，与右侧一致。原 E0 报告检查的是旧的左右质量不等 XML。

因此 E0 → Mirror005 至少包含两个变化：

1. 镜像数据增强和 mirror loss；
2. 左右 `hip_yaw_link` 质量对称化。

现有结果足以判定“当前 Mirror005 模型输出对称”，但不能严格量化其中多少改善由
mirror loss 单独贡献。需要在**同一份当前 XML、同一 seed 和同一评测设备**下补跑
一个无 symmetry 的配对 E0，才是干净的因果对照。

## 改进方向

### P0：停止 Mirror010，先修动作输出边界

0.05 在 m500 已经达到对称门槛，继续增大到 0.10 不会直接解决 yaw 或动作越界。
下一轮应只改变动作分布/边界，优先采用有正确 log-prob 修正的
**tanh-squashed Gaussian**，让策略训练时就知道动作有界；如果框架暂时不支持，
则使用从训练开始生效的 `clip_actions=1.0`，并降低初始/允许的策略 std 与 entropy。

不要把 clip 只作为 m49999 的部署端补丁：旧 E0 消融已经证明，现有饱和策略被临时
裁剪后会摔倒，必须重新训练。同时记录 raw action p99/max、`|a|>1` 比例和 processed
target 越界率。

### P1：动作健康后再处理 yaw

- 保留 `mirror_loss_coeff=0.05`，不要先加大系数。
- 为 `wz_cmd - wz_actual` 加小权重非饱和 Huber 项，首轮建议 `−0.05`，若确定性
  回放仍无改善再试 `−0.10`；不要直接跳到大权重。
- 检查训练命令采样中正/负 yaw 命令的覆盖率，并在每 500 轮固定跑
  `turn +0.3/-0.3 rad/s`，避免只看训练的平均 angular reward。
- 保留现有指数 tracking reward 负责零附近精度，Huber 负责大误差区间的恢复梯度。

### P1：补干净配对对照和多 seed

建议最小矩阵：

| 实验 | XML | mirror | 有界动作 | yaw Huber | 目的 |
|---|---|---:|---:|---:|---|
| B0-paired | 当前质量对称 XML | 关 | 关 | 关 | 隔离 XML 改动 |
| A1-bounded | 当前质量对称 XML | 0.05 | 开 | 关 | 验证动作健康与可行走性 |
| A2-yaw005 | 同 A1 | 0.05 | 开 | −0.05 | 修复 yaw |
| A3-yaw010 | 同 A1 | 0.05 | 开 | −0.10 | 仅在 A2 不足时运行 |

先用 seed 42 做因果筛选，通过后再跑 42/43/44。每 500 轮自动进行确定性回放，满足
以下条件才继续长训：`vx≥0.30`、幅值差 `<0.03`、yaw RMS `<0.25`、处理目标越界
`0%`、raw `|a|>1≤10%`，并且正负转向命令都有同向响应。

## 运行完整性与复现信息

### 产物检查

- checkpoint：101 个，`model_0.pt` 到每 500 轮的 `model_49500.pt`，加
  `model_49999.pt`。
- 训练视频：100 个，间隔 12,000 policy steps，对应每 500 iterations。
- 正式评测：6 种 nominal 命令、3 种环境条件、6 个训练进度点，全部 CSV 均为
  400 samples，未发生 8 秒内终止。
- 最终视频：侧视和正视均已用 `ffprobe` 验证为 1280×720、50 fps、300 帧。

关键文件 SHA-256：

```text
222c63f5f861d01a3e61dbad4215649ab9d2c936e796a3fc59ca7ddac0b4e8a8  model_49999.pt
6e84f3eebcb0143f6d28b29cd53518f27b27aa585002dc0a977d2667b457121e  params/agent.yaml
ac5a862a0b7baab3089cf3e70b7227c6a4e7c74c7f625a0c590ef7f88a7c5e5f  params/env.yaml
d68be1ade235303ccc2230dedbfa286051834d1d6728e3453eddf4fd41507866  pikachu_v031_train.xml
2471de897280b990cdd13fc5d50486766708b1aa6d5fecd4a49457c6bf7844fb  env_cfgs.py
3a7587d111d709c6a0e62f77d3b9b4cd5e94c7973b7cf924b686ff0a8d957218  rl_cfg.py
```

run 记录的代码提交为 `c061be4`。完整训练命令：

```bash
MUJOCO_GL=egl uv run train \
  Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror005 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 50000 \
  --agent.save-interval 500 \
  --agent.seed 42 \
  --agent.run-name mirror005-s42 \
  --agent.logger tensorboard \
  --video True \
  --video-length 200
```

`--video-interval` 留空时，训练器自动使用
`500 × 24 = 12000 policy steps`，因此视频与 500-iteration checkpoint 对齐。

交互式播放最终模型：

```bash
uv run play Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror005 \
  --checkpoint-file \
  logs/rsl_rl/pikachu_v031_e0_mirror005/2026-10-12_00-11-23_mirror005-s42/model_49999.pt \
  --viewer velocity-joystick
```

CPU 上复现正式 nominal 命令组：

```bash
CUDA_VISIBLE_DEVICES= uv run python scripts/record_v03_metrics.py \
  --task Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror005 \
  --checkpoint \
  logs/rsl_rl/pikachu_v031_e0_mirror005/2026-10-12_00-11-23_mirror005-s42/model_49999.pt \
  --out doc/pikachu_v031_e0_mirror005_20261012/metrics/final_nominal \
  --tag final_nominal_s42 \
  --clips stand,fwd,fwd_fast,back,lat,turn \
  --seconds 8 --seed 42 --device cpu --nominal
```

重新生成全部摘要和曲线：

```bash
uv run python doc/pikachu_v031_e0_mirror005_20261012/analyze_run.py
```

机器可读结果：

- [完整 JSON 摘要](summary.json)
- [全部 rollout 汇总 CSV](rollout_summary.csv)
- [逐帧回放数据目录](metrics/)
- [全部图目录](plots/)
- [视频目录](videos/)
