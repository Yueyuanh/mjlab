# <模型/实验> 评估报告

状态：`failed | simulation-candidate | hardware-review-passed | deployable`
评估日期：`YYYY-MM-DD`

## 结论

用一段话说明是否通过预设门槛、相对 baseline 的变化、最重要的失败模式和当前可交付等级。

## 评估对象

| 项目 | baseline | candidate |
| --- | --- | --- |
| Task ID |  |  |
| checkpoint |  |  |
| SHA-256 |  |  |
| 代码 commit/diff |  |  |
| env/agent config |  |  |

## 协议

- 推理模式：
- seed / 环境数 / 时长 / warmup / control dt：
- commands / 初始状态 / 终止处理：
- nominal 与 randomized 设置：
- 指标定义、单位、坐标系和聚合：
- renderer / 相机 / 分辨率 / FPS：`MuJoCo / three-quarter / 1280x720 / 50`
- 速度 overlay：`右下角；vx/vy/wz；目标虚线、实测实线`
- full 自动评估：`forward 80%, 10 s, 50 Hz` / `backward 80%, 10 s, 50 Hz`
- 可比性说明：

## 结果

| case | 指标 | baseline | candidate | 门槛 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
|  |  |  |  |  |  |

同时报告波动或分位数、跌倒/失败次数和最差 case；不要只列总体均值。

## 行为与失败案例

- 可重复观察：
- 失败或长尾：
- 视频不能说明、仍需数值验证的事项：

## 证据

- 协议与汇总 JSON：
- 原始 CSV/JSON 与同步视频：`metrics/`
- 最终视频与 render metadata：`evaluation/video/`
- 训练曲线（仅作优化过程参考）：`evaluation/training_plots/`
- artifact manifest / SHA-256：

## 归因、风险与下一步

- 本次变量能支持的结论：
- 无法单独归因的联动改动：
- 仿真到实机风险与未验证项：
- 建议的下一次单变量实验：
