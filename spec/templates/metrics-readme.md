# 固定速度评估数据

Task ID：`<Task ID>`
checkpoint：`../checkpoints/<filename>`
SHA-256：`<sha256>`
控制频率：`50 Hz`
每档时长：`10 s`

## 命令速度

| case | 有效范围来源 | 计算 | 实际命令 |
| --- | --- | --- | --- |
| forward 80% | `../config/env.yaml:<field>` | `0.8 * forward_max` | `<+0.00 m/s>` |
| backward 80% | `../config/env.yaml:<field>` | `0.8 * backward_min` | `<-0.00 m/s>` |

坐标系：`<body | heading>`。`vx/vy` 单位为 m/s，`wz` 单位为 rad/s；目标和实测使用同一坐标系。

## 文件

- `metrics_forward_vx_<speed>mps_p80_10s_50hz_with_target.{csv,json,mp4}`
- `metrics_backward_vx_<speed>mps_p80_10s_50hz_with_target.{csv,json,mp4}`

若存在 `.recording.json`，它与同 stem CSV/MP4 记录样本到视频帧的同步关系。

## 录制命令

粘贴本 run 实际执行的两条完整命令，包括 Task ID、checkpoint、速度、seed、时长、视角和输出 stem。

```bash
uv run python scripts/review_v033_policy.py <完整参数>
```

## 查看

在 mjlab 仓库根执行：

```bash
uv run python scripts/plot_logs/serve.py
```

打开命令输出的 Metrics dashboard 地址。dashboard 会递归发现 `metrics_*.csv`，并自动关联同 stem
MP4 和 `.recording.json`。

## 字段说明

列出本次 CSV/JSON 的字段、单位、聚合方式、warmup/reset 处理和已知限制。不要只描述图表；原始数据
应足以重算报告中的核心结论。
