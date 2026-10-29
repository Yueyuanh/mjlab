# V034 续训至 5000 轮

[返回目录导航](../README.md)

2026-10-23 按要求启动双足、四足续训。两者均从已完成的 300 轮权重恢复，**各追加 4700 轮，总计 5000 轮**。128 个并行环境，保持已有任务的环境、奖励、网络和动作设置。每完成 500 轮，保存 checkpoint、独立 ONNX，并用该 checkpoint 在单独评估进程中录制 5 秒固定前进命令视频（640×480，25 fps）。视频附带该段速度与跌倒指标 JSON。

## 正在运行的任务

输出目录：

```text
logs/rsl_rl/pikachu_v034/continue_5000_20261023_133227/
```

这是脱离交互终端的后台进程，启动 PID 为 `752687`。单张 RTX 4060 上顺序运行 **biped → crawl**。进度由实际执行进程持续更新，不能把本文件或队列配置当成训练完成的证据：

- [队列状态](../../../logs/rsl_rl/pikachu_v034/continue_5000_20261023_133227/queue.json)
- [双足实际进度](../../../logs/rsl_rl/pikachu_v034/continue_5000_20261023_133227/biped/progress.json)
- 四足开始后创建 `crawl/progress.json`。
- 双足、四足控制台日志分别为 `biped_train.log`、`crawl_train.log`。

每个任务目录中的文件布局：

```text
biped/ 或 crawl/
  params/env.yaml
  params/agent.yaml
  progress.json
  events.out.tfevents.*
  model_500.pt
  model_500.onnx
  model_1000.pt
  ...
  model_5000.pt
  model_5000.onnx
  videos/model_500.mp4
  videos/model_500.json
  videos/model_500.log
  ...
  videos/model_5000.mp4
  final_nominal.json
  final_randomized.json
```

完成一个任务的 5000 轮训练后，自动进行 16 环境 × 10 秒的五档标称及随机化评估，再开始另一个任务。`queue.json` 只有在两项训练和最终评估全部成功后才写为 `complete`。训练、录像或评估失败会写入 `failed` 和错误原因，失败时不静默跳过视频。

## 轮数与视频一致性

RSL-RL 的旧 `model_299.pt` 内部 `iter=299`，表示已经完成 300 次更新。续训入口把下一次循环索引设为 300，避免重复计算上一轮。新文件名使用**完成轮数**，例如 `model_500.pt` 内部仍按框架约定记录 `iter=499`，另在 `infos.completed_iterations` 写入 500。5000 轮最终文件内部 `iter=4999`。

每个保存点在 PPO 更新完成后触发录像，录像进程加载刚保存的权重，并使用单独环境。不会重置训练环境，也不会用录像更新策略。普通训练脚本按本次运行 policy-step 触发录像的机制不用于本次续训，因此续训起点 300 不会把录像错移至 800、1300 等位置。

新入口：`scripts/continue_v034_training.py`。恢复时保留优化器、actor/critic、观测归一化和 common_step_counter。输出目录带代码快照和资产哈希。后台进程可在关闭终端后继续运行，但系统关机或重启会终止计算。

查看实时日志：

```bash
cd /home/finnox/Pikachu/PikachuRobot/pikachu_playground/mjlab
tail -f logs/rsl_rl/pikachu_v034/continue_5000_20261023_133227/biped_train.log
```

如果进程被系统中断，从最后已保存的 checkpoint 恢复单个任务到一个**新的输出目录**：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/continue_v034_training.py \
  --worker-mode biped --checkpoint /path/to/model_1000.pt \
  --output logs/rsl_rl/pikachu_v034/recovery_new \
  --target-iterations 5000 --save-interval 500 --num-envs 128 --video-seconds 5
```

`--target-iterations` 在这个入口中明确表示总轮数；与普通 `mjlab.scripts.train` 的追加轮数参数不同。当前任务正在后台运行时不应重复启动同样的续训。
