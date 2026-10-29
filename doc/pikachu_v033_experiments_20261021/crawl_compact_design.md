# V033 四足紧凑姿态设计

新任务：`Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E2`。保留 Crawl、Crawl-E1 和双足任务的配置、动作及物理资产。实验目录：`logs/rsl_rl/pikachu_v033/crawl_e2/compact_20261021`。

## 诊断与目标

以现有 Crawl-E1 `2026-10-20_22-54-12_full-4096x20000/model_3500.pt` 为基线，seed 42、8 环境、12 秒、前进指令 0.15 m/s 的确定性测试结果：实际前进均值 0.1446 m/s，未摔倒；左右肘绝对角度 95 分位为右 49.19°、左 68.69°；同侧手脚末端平均 0.3352/0.3424 m。

E1 只有 arm_roll/arm_yaw 的 ±5° 软窗口，elbow 未独立约束，也没有同侧手脚距离目标。新任务增加：

- elbow：相对标称零位的 ±8° 免惩罚窗口，超出部分除以 8° 后平方，双肘平均，权重 -0.20。
- 同侧末端距离：采用 STL 支撑点 `right_hand/right_foot` 与 `left_hand/left_foot` 的三维距离，目标软区间 0.10–0.25 m，超出区间的距离除以 0.08 m 后平方，双侧平均，权重 -0.40。下界用于避免奖励驱动末端重合；该距离本身不是完整自碰撞保证。
- 保留速度跟踪、四点接触时序、摆动高度、防滑和躯干姿态等原奖励。保留全部 18 个动作，不锁关节、不裁剪策略均值或实际状态，不修改物理限位及碰撞模型。
- 微调噪声初始 std=0.20，范围 0.04–0.25，entropy=0.001，初始学习率 1e-4，自适应 KL 调节。

软窗口并非硬范围保证；最终应报告真实肘角与实际末端距离分布。距离默认按左右各自同侧理解，非交叉配对，也非任意手脚的最近邻。

## 小实验与正式训练

第一阶段：256 环境 × 400 轮，从上述 E1 actor/critic 初始化并重置探索噪声。

相同测试条件下：实际前进均值 0.1338 m/s、零摔倒；左右肘绝对角度 95 分位降到 34.26°/33.16°；同侧末端均距降到 0.2339/0.2505 m。代价是速度误差增加，以及三点以上支撑比例从 0.621 降至 0.442，需要继续训练和验收。

第二阶段：从 pilot 的 `model_399.pt` actor/critic 暖启动，再训练 **5000 轮**，256 环境，每轮每环境 24 个控制步，总计 30,720,000 transitions。优化器和 iteration 计数重新初始化；本次 5000 不包括前面的 400 轮 pilot，也不代表从随机权重训练。

训练使用单 GPU、CPU 2 线程、nice=10，串行运行仿真工作。每轮监控 Linux MemAvailable，低于 2048 MiB 时保存 checkpoint 并退出。正式训练已完成。归档目录为 `training/final_256x5000/`，状态在 `metadata/complete.json` 和 `metadata/progress.json`，最终模型在 `checkpoints/model_4999.pt`。

最终需要按相同条件复测前进，并补测静站、倒退、侧移、转向及第二个随机种子；保存前/侧视频。训练完成并通过检查前，不将中间结果称为最终优化效果。

## 验证与复现

`tests/test_pikachu_v033_crawl_e2.py` 检查同侧配对、刚体变换不变性、任务隔离和原动作/机器人配置保持一致。与 Crawl、Crawl-E1、command 测试合计 21 项通过；新增配置及分析脚本通过 Ruff 和定向 ty 检查。

原训练入口如下。重现实验时请选择不存在的新输出目录；worker 运行时使用原生目录结构，退出后再按 README 分类归档：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 \
uv run python scripts/train_v033_e1.py \
  --task Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E2 \
  --out logs/rsl_rl/pikachu_v033/crawl_e2/compact_20261021/training/final_256x5000 \
  --num-envs 256 --iterations 5000 \
  --actor-init logs/rsl_rl/pikachu_v033/crawl_e2/compact_20261021/training/pilot_256x400/checkpoints/model_399.pt \
  --load-critic --save-interval 500 --min-available-memory-mib 2048
```

相关源文件的副本和 SHA256 在实验目录的 `provenance/` 中；训练的有效配置在每个 `training/<run>/configs/` 中。基线与 pilot 的真实测量分别位于 `evaluation/baseline/metrics/forward_metrics.json` 和 `evaluation/pilot/metrics/forward_metrics.json`。

转换任务的调研、失败案例与后续实现建议见同目录 `transition_research.md`；它与此次四足 5000 轮训练分开记录，不把 PD 试验称为训练成功的转换策略。
