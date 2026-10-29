# R08_T6 metrics

每个 CSV 为单环境固定命令、8 秒、50 Hz 的逐帧物理遥测。

- `r08-t6-final-nominal`：关闭 startup 摩擦、编码器和 base-COM 随机化；用于结构
  参数与倒立摆分析。
- `r08-t6-final`：保留训练 startup DR，固定 seed 42；用于单样本鲁棒性检查。
- 两个 `*_index.json` 保存 checkpoint、seed、档位、步数和终止状态。

报告统计丢弃前 1 秒起步瞬态。步频由左 `knee_up` 的 0.5–6 Hz 主频估计，总交替
步频等于单腿主频的两倍。

