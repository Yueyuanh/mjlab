# Pikachu v03 · R07 训练改进方案

## 1. 重新评估结论

基线固定为：

```text
logs/rsl_rl/pikachu_v03_velocity/
  2026-10-09_01-48-29_v03-flat-r06b/model_59997.pt
```

下表全部来自 `metrics_v03_*_r06-best.csv` 的物理量；速度统计丢掉前 12.5%
起步瞬态，不使用 `Episode_Reward/track_*` 代替真实速度。

| 项目 | R06 实测 | 判断 |
|---|---:|---|
| 前进 0.40 m/s | 0.40，跟踪比 1.01 | 好 |
| 前进 0.50 m/s | 0.48，跟踪比 0.97 | 好 |
| 后退 −0.30 m/s | −0.24，跟踪比 0.78 | 需提高 |
| 侧移 +0.25 m/s | +0.09，跟踪比 0.34 | 明显不足 |
| 静止速度误差 | 0.001 m/s | 速度静止，但姿态不合格 |
| 静止关节姿态 RMSE | 0.183 rad / 10.5° | 偏离初始姿态明显 |
| 静止脚间距 | 15.3 cm（初始 20.0 cm） | 仍偏窄 |
| 行走平均脚间距 | 11.0 cm；前进仅 9.3–9.4 cm | 过窄 |
| 峰值扭矩占用 | 1.00 | 已触及电机上限 |

静止姿态并不是“小幅漂移”：左 hip roll −15.2°、左 hip yaw −15.1°、右
hip yaw +15.0°，都接近 ±15°机械硬限位；左 knee_up 也偏了 +14.1°。
现有指数型 `pose` 奖励在这个误差区间已接近 0，恢复梯度也随之消失。

### 小碎步的量化

对 8 秒 CSV 中 `left_knee_up_joint` 去掉前 1 秒后做 0.5–6 Hz 主频分析；
单腿一个周期对应左右两次交替落脚，因此总步频取 `2 × 单腿主频`：

| 档位 | 单腿主频 | 估计步频 | 实测沿命令速度 | 每步推进 |
|---|---:|---:|---:|---:|
| 前进 0.40 | 3.29 Hz | 6.57 步/s | 0.40 m/s | 6.1 cm |
| 前进 0.50 | 3.43 Hz | 6.86 步/s | 0.48 m/s | 7.0 cm |
| 后退 −0.30 | 2.14 Hz | 4.29 步/s | 0.24 m/s | 5.6 cm |
| 侧移 +0.25 | 2.14 Hz | 4.29 步/s | 0.09 m/s | 2.1 cm |

结论：R06 的前进速度很好，但主要靠约 6.6–6.9 步/s 的高步频实现，确实是
小碎步；“动作平滑”不能替代步频/步距约束。

## 2. R07 的改动

独立任务 ID：`Mjlab-Velocity-Flat-Pikachu-V03-R07`。R06 原任务与配置保留，
可随时做回归对照。R07 不改变 observation、action 或网络结构，可从 R06
checkpoint 热启动。

| 目标 | R07 机制 | 关键参数 |
|---|---|---|
| 大误差下仍追速度 | 在原指数奖励旁加无饱和 Huber 代价 | weight −0.25，delta 0.20 m/s |
| 静止保持初始姿态 | 命令门控的全驱动关节二次姿态代价 | weight −0.25；roll/yaw std 0.08 rad，其余 0.12 |
| 静止高度一致 | 去掉 0.25 m 蹲姿目标，改为初始静平衡高度 | 0.26728 m |
| 行走脚不要太近 | 只在“双脚同时支撑”时施加最小宽度 hinge | 下限 0.14 m，std 0.04，weight −0.20 |
| 不跑小碎步 | 对腾空不足的受支撑落脚直接惩罚 | 最短 0.18 s，weight −2.0 |
| 更长摆腿 | 交替落脚目标从 0.22 s 调到 0.28 s | std 0.10；dense air-time 权重 1.0→0.5 |
| 保留实机余量 | 只惩罚超过电机规格 80% 的部分 | weight −0.25 |
| 静止项更快收敛 | standing env 比例从 5% 提至 10% | yaw 暴露面仍有 22.5% |

两个设计边界：

1. 行走步宽项不是“全周期追 20 cm”。摆腿经过身体下方时允许自然收窄，
   只有双支撑且小于 14 cm 才受罚，避免重演 R05 的僵硬原地踏步。
2. 短步项只惩罚有另一只脚稳定支撑的落脚，双脚同时落地不算，防止把跳跃
   误当成大步。

## 3. 训练流程

### 阶段 A：3k 迭代安全探针

```bash
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-Pikachu-V03-R07 \
  --env.scene.num-envs 2048 \
  --agent.run-name v03-flat-r07-probe \
  --agent.resume True \
  --agent.load-run 2026-10-09_01-48-29_v03-flat-r06b \
  --agent.load-checkpoint model_59997.pt \
  --agent.max_iterations 3000
```

加载 actor、critic、optimizer 和 iteration，保留 R06 末期自适应降到约 1e-4 的
学习率，避免用 1e-3 一次打散已有步态。每 500 iteration 已自动保存 checkpoint。

3k 探针继续训练的硬门槛：

- 前进 0.40 跟踪比 ≥ 0.90，方向余弦 ≥ 0.98；
- 8 秒各档零摔倒；
- 静止关节 RMSE < 0.10 rad，且相对 R06 明显下降；
- 前进步频 < 5.5 步/s、每步推进 > 7 cm；
- 走路双支撑脚宽中位数 ≥ 12 cm；
- 峰值扭矩占用不高于 1.00。

任一门槛失败，不进入长训。按以下顺序定位：先将
`walking_stance_width.weight` −0.20→−0.10；再将
`short_step_air_time.weight` −2.0→−1.0；最后才动速度 Huber。静止姿态项不与
走路 env 重叠，不应先削它。

### 阶段 B：追加 9k 迭代

从探针 run 的最后 checkpoint 续训，`--agent.max_iterations 9000`。不要把
`max_iterations` 写成总迭代号；resume 时它表示本次追加量。

建议在相对 R07 起点的 +3k / +6k / +12k 三个点统一录制：

```bash
uv run python scripts/record_v03_metrics.py \
  --task Mjlab-Velocity-Flat-Pikachu-V03-R07 \
  --checkpoint <checkpoint> \
  --out docs/pikachu_v03/metrics --tag r07-<iter> --seconds 8

uv run python scripts/eval_v03_velocity.py \
  --task Mjlab-Velocity-Flat-Pikachu-V03-R07 \
  --checkpoint <checkpoint> --seconds 8
```

更新后的 `eval_v03_velocity.py` 会额外报告关节主频估计的步频、平均腾空时间和
每步推进，不再只看速度与单脚支撑占比。

## 4. 最终验收标准

| 类别 | 通过线 |
|---|---|
| 稳定性 | 所有固定档 20 s 零终止；训练 `mean_episode_length` ≥ 995/1000 |
| 前进 | 0.40/0.50 m/s 跟踪比 0.95–1.05，方向余弦 ≥ 0.98 |
| 后退 | −0.30 m/s 跟踪比 ≥ 0.85，方向余弦 ≥ 0.98 |
| 侧移 | +0.25 m/s 跟踪比 ≥ 0.50，方向余弦 ≥ 0.95 |
| 静止姿态 | 10 驱动关节 RMSE ≤ 0.06 rad；单关节最大均值偏差 ≤ 0.12 rad |
| 静止几何 | 高度 0.26728±0.01 m；脚宽 18–22 cm；速度模 < 0.02 m/s |
| 行走步宽 | 双支撑脚宽 P10 ≥ 0.14 m；禁止用全周期均值替代 |
| 步态形态 | 前进步频 3.0–4.5 步/s；0.40 m/s 时每步推进 ≥ 0.09 m |
| 实机余量 | 峰值扭矩占用 ≤ 0.90；超规格帧占比 0 |
| 平滑性 | `action_acc` 不高于 R06；支撑相滑移 ≤ 0.08 m/s |

若侧移仍低于 0.50，但 hip roll 已持续接近 ±0.26 rad，则判定为机械限位主导，
不再通过加奖励硬推；此时应缩小侧移命令范围或修改机构，而不是牺牲前进步态。

## 5. 当前交付状态

- R07 独立任务已注册，reward manager 可正常构建；CPU 2-env 零动作一步 smoke
  通过，reward 全部 finite。
- 新增奖励均有单元测试；velocity reward/task 定向测试为 25 passed、2 skipped。
- 当前执行环境无 CUDA 设备，所以本次没有伪称 R07 已完成 GPU 长训；下一步是按
  阶段 A 启动 3k 探针，并用上面的物理门槛决定是否续训。
