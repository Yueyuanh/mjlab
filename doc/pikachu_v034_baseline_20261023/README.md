# Pikachu v034 训练与评估资料

本目录归档 2026-10-23 的 **300 轮基础训练结果**，并提供后续 **5000 轮续训**的运行入口。基线结果与续训结果分开查看；后台进度和新权重以 `logs/rsl_rl/pikachu_v034/` 中的实际输出为准。

## 常用入口

| 要查看的内容 | 入口 |
| --- | --- |
| 模型接入、任务配置、训练与复现命令 | [基础训练说明](reports/baseline.md) |
| 300 轮策略的定量评估与限制 | [评估报告](reports/evaluation_report.md) |
| 5000 轮续训、每 500 轮模型与视频、实时进度 | [续训说明](reports/continuation_5000.md) |
| 300 轮 checkpoint、ONNX、哈希与评估文件映射 | [运行清单](data/runs.json) |
| 双足 / 四足前进命令回放 | [双足视频](media/clips/biped_forward.mp4) · [四足视频](media/clips/crawl_forward.mp4) |
| 初始姿态 / 训练曲线 / 策略截图 | [姿态](media/figures/poses.png) · [曲线](media/figures/training_curves.png) · [截图](media/figures/policy_review.png) |

## 目录结构

```text
README.md                       总导航
reports/                        基线、评估、续训说明
data/                           评估 JSON、训练统计、运行清单
configs/
  biped/{agent,env}.yaml         双足训练配置
  crawl/{agent,env}.yaml         四足训练配置
media/
  figures/                      姿态、训练曲线、回放截图
  clips/                        两段 300 轮策略回放视频
provenance/
  source_hashes.json            交付时的代码与资产哈希
  source_snapshot/              300 轮基线交付时的代码快照
```

## 数据命名

- `biped_*` 为双足，`crawl_*` 为四足。
- `*_zero.json` 为零动作 PD 对照；`*_nominal.json` 为标称评估；`*_randomized.json` 为训练域随机化评估。这三组均为 16 环境 × 10 秒。
- `*_video_eval.json` 对应 `media/clips/` 内的 5 秒、单环境视频，不与正式评估混用。
- `training_summary.json` 为两项 300 轮训练的曲线统计。
- `data/runs.json` 内的评估文件名相对于 `data/`；checkpoint 和 ONNX 路径保留为实际绝对路径。

`configs/` 和 `provenance/` 是历史记录；当前可执行代码位于仓库的 `scripts/`、`src/`。快照内的历史路径保留原样，以便核对哈希。

后续材料按同类目录存放；新的训练权重、日志和每 500 轮视频继续保存在各自的运行目录，避免与此处的 300 轮基线归档混淆。
