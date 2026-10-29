# Pikachu V031 B1-BetaBounded 训练评估报告

> 状态：配置与 smoke/PD 基线已完成；正式 4096 × 50,000 训练待当前 GPU 上的
> T6 任务结束后自动启动。本文件将在 m500/m1000 门槛与最终评估完成后补齐曲线、
> 视频、对照表和部署结论。

## 当前结论

- Beta actor、PPO、KL、镜像数据增强和 mirror loss 已通过 64 env × 2 iterations
  smoke；未出现 NaN。
- actor 输出头为 `[2, 10]` 的 alpha/beta 参数，确定性和随机动作天然位于
  `[-1,1]`；runner clip 只作为冗余保护。
- 当前 PD 的无接触 70-case 响应没有扭矩饱和，左右响应一致，暂不修改 PD。
- 正式结论必须等待负载行走指标，尤其是 ankle 扭矩余量、实际/目标跟踪误差、
  是否贴 Beta 边界以及六命令覆盖。

完整设计、验收门槛及复现命令见 [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)。
