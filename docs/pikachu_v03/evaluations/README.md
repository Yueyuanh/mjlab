# Pikachu v03 模型批次评估索引

从 R08_T6 起，每个训练 run 使用与日志目录完全相同的独立目录：

```text
evaluations/<run-name>/
├── README.md       # 该批次的主评估报告
├── manifest.json   # checkpoint、资产、评估条件与推荐结论
├── metrics/        # 固定命令逐帧 CSV；nominal 与 DR 分开标识
└── videos/         # 经 ffprobe 校验的 720P 定性录像及清单
```

## 批次

| 批次 | 主 checkpoint | 结论 | 报告 |
|---|---|---|---|
| `2026-10-09_17-32-05_v03-flat-r08-probe_t6` | `model_64996.pt` | 纵向追踪和步态周期明显改善；质量/惯量、3 N·m 横滚能力与目标步宽互相矛盾，暂不建议部署 | [R08_T6 评估](2026-10-09_17-32-05_v03-flat-r08-probe_t6/README.md) |

旧的 R04–R06 产物仍在上一级 `metrics/`、`videos/`、`reports/`，保持已有链接不变；
后续新批次不再把文件混入这些公共目录。

