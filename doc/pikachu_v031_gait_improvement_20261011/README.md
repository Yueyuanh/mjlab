# Pikachu V031 步态偏置改进实验

本目录承接 `doc/pikachu_v031_velocity_20261010/REPORT.md` 的诊断结论，执行
E0→E4 单因素链。设计报告：

- [E0 基线](E0_BASELINE.md)
- [E0-Mirror 低权重镜像实验](E0_MIRROR.md)
- [E1 动作边界](E1_ACTION_CLIP.md)
- [E2 对称学习](E2_SYMMETRY.md)
- [E3 yaw Huber](E3_YAW_HUBER.md)
- [E4 扰动课程](E4_CURRICULUM.md)
- [最终数据与结论](FINAL_REPORT.md)

原设计保留在 [experiment_plan.json](manifests/experiment_plan.json) 与
[run_sequential.sh](run_sequential.sh)。本轮实际完成 E1–E3 m1000 因果筛选和
E4 m5000 全课程覆盖；运行路径、事件文件、检查点哈希在
[run_results.json](manifests/run_results.json)。

结果入口：

- [训练与回放曲线](plots/)
- [66 组汇总 JSON](analysis_summary.json) /
  [汇总 CSV](rollout_summary.csv)
- [逐步遥测](metrics/)
- [6 段 720P 对比视频](videos/)
- [可重复评测脚本](evaluate_results.sh)

所有变体均使用独立 task ID 与日志目录，没有覆盖旧运行。最终判定是本轮新模型
尚不可部署：关节越限和静态对称性改善，但确定性策略退化为站立，且 E4 满推扰
阶段发生训练断崖；详见最终报告。

后续的 E0-Mirror 实验回到可行走的 E0，只加入低权重镜像约束，不继承
E1--E4 的 action clip、yaw Huber 或扰动课程。
