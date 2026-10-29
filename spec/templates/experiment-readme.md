# <实验名称>

状态：`planned | running | interrupted | trained | evaluated | selected | invalid`
日期：`YYYY-MM-DD`
负责人/Agent：`<name>`

## 目标与假设

- 目标：
- 要验证的主要假设：
- 明确不在本次范围内的事项：

## 对照与变量

- Task ID：
- baseline checkpoint 与 SHA-256：
- candidate 初始化 checkpoint 与 SHA-256：
- 本次唯一主要变量：
- 共同保持不变的设置：

## 验收标准

- 必须达到的数值门槛：
- 不得退化的指标：
- 失败/提前停止条件：
- 实机安全审核要求（若适用）：

## 运行计划

| 阶段 | run | 环境数 x 迭代 | seed | 状态 | 进入下一阶段的条件 |
| --- | --- | --- | --- | --- | --- |
| `1-sanity` | `E01-R01-YY-MM-DD` |  |  | planned |  |
| `2-smoke` |  |  |  | planned |  |
| `3-pilot` |  |  |  | planned |  |
| `4-full` |  |  |  | planned |  |

## 产物索引

- 最终 run：`4-full/<run>/`
- 有效配置：`4-full/<run>/config/{env.yaml,agent.yaml}`
- run manifest：`4-full/<run>/config/run.json`
- 最终 checkpoint 与 SHA-256：
- 原始指标：`4-full/<run>/metrics/`
- 最终视频：`4-full/<run>/evaluation/video/`
- 报告：`4-full/<run>/report/html/`
- run 总结与复现命令：`4-full/<run>/README.md`

## 结论与限制

训练前保持本节为空；评估后记录通过/未通过、失败案例、归因限制和下一步。没有独立评估时，
状态只能写 `trained`，不能写 `selected` 或“最终模型”。

## 规范例外

无。若有，逐项记录偏离的规范、原因、影响、替代验证和补齐计划。
