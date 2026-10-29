# <run ID> - <一句话目的>

状态：`running | completed | interrupted | invalid | evaluated | selected`
阶段：`1-sanity | 2-smoke | 3-pilot | 4-full`
日期：`YY-MM-DD`
Task ID：`<Task ID>`

## 简要总结

用 3-5 句话说明本次唯一主要变量、是否正常结束、核心结果、相对 baseline 的变化和最重要的限制。
没有独立评估时只能写 `completed`，不能写 `selected` 或“最终模型”。

## 身份与结果

| 项目 | 内容 |
| --- | --- |
| run | `E01-R01-26-10-29` |
| seed / num envs / iterations |  |
| 代码 commit / dirty |  |
| 父 checkpoint / SHA-256 |  |
| 最终 checkpoint / SHA-256 |  |
| env / agent config | [`env.yaml`](config/env.yaml) / [`agent.yaml`](config/agent.yaml) |
| manifest | [`run.json`](config/run.json) |
| 外部 logger |  |

核心指标：

| case | command | tracking / survival / task metric | 结论 |
| --- | --- | --- | --- |
| forward 80% |  |  |  |
| backward 80% |  |  |  |

## 训练曲线

![训练曲线](evaluation/training_plots/training-summary.png)

说明曲线中选择 checkpoint 的依据；训练曲线仅表示优化过程，不替代独立评估。

## 最终效果

<video controls width="960" poster="report/figures/final-video-poster.jpg">
  <source src="evaluation/video/final-three-quarter-720p.mp4" type="video/mp4">
</video>

[无法内嵌播放时打开最终视频](evaluation/video/final-three-quarter-720p.mp4)

- 固定前进 80%：[CSV](metrics/metrics_forward_vx_p0p00mps_p80_10s_50hz_with_target.csv) /
  [视频](metrics/metrics_forward_vx_p0p00mps_p80_10s_50hz_with_target.mp4)
- 固定后退 80%：[CSV](metrics/metrics_backward_vx_m0p00mps_p80_10s_50hz_with_target.csv) /
  [视频](metrics/metrics_backward_vx_m0p00mps_p80_10s_50hz_with_target.mp4)

将占位速度替换为 `env.yaml` 计算出的实际命令速度。

## 复现命令

### 训练

```bash
uv run <完整训练入口> \
  <Task ID 和全部非默认参数> \
  <run 输出目录参数>
```

### Play

```bash
uv run <play 入口> \
  <Task ID> \
  --checkpoint checkpoints/<最终 checkpoint>
```

### 固定速度指标与同步视频

```bash
# 前进：将 <forward-80> 替换为 0.8 * forward_max。
uv run python scripts/review_v033_policy.py \
  --task <Task ID> \
  --checkpoint checkpoints/<最终 checkpoint> \
  --out metrics \
  --name metrics_forward_vx_<encoded-speed>mps_p80_10s_50hz_with_target \
  --clip forward \
  --command-x <forward-80> \
  --seconds 10 \
  --video \
  --view three_quarter

# 后退：将 <backward-80> 替换为 0.8 * backward_min，保留负号。
uv run python scripts/review_v033_policy.py \
  --task <Task ID> \
  --checkpoint checkpoints/<最终 checkpoint> \
  --out metrics \
  --name metrics_backward_vx_<encoded-speed>mps_p80_10s_50hz_with_target \
  --clip backward \
  --command-x <backward-80> \
  --seconds 10 \
  --video \
  --view three_quarter
```

当前任务不是 Pikachu v033 时，替换为提供同等参数和产物的通用录制接口。执行前以 `--help` 核对
当前 CLI。

### 最终 720p 视频

```bash
uv run python scripts/review_v033_policy.py \
  --task <Task ID> \
  --checkpoint checkpoints/<最终 checkpoint> \
  --out evaluation/video \
  --name final-three-quarter-720p \
  --clip forward \
  --command-x <forward-80> \
  --seconds 10 \
  --num-envs 1 \
  --seed 42 \
  --video \
  --view three_quarter
```

该接口当前在实现中固定输出 1280x720、50 FPS，并叠加右下角速度曲线；不要填写它不支持的
`--width`、`--height` 或 `--fps` 参数。若任务需要其他录制入口，必须给出该入口的实际完整参数。

## 已知问题与下一步

- 失败或长尾行为：
- 未验证条件：
- 仿真到实机风险：
- 下一次实验：

## 规范例外

无。若有，记录偏离条目、原因、影响、替代证据和补齐计划。
