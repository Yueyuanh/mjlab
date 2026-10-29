# Pikachu V033 E3 四足动作与柔和转换实验

日期：2026-10-23。

## 设计

`Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E3` 从 E2 热启动，训练采样显式保留
40% 前进、30% 后退，并只对前进重点增加足底姿态、步态接触和欠速修正，避免
修前进时遗忘已经较好的后退动作。

`Mjlab-Transfer-Flat-Pikachu-V033-E3` 将二值姿态跳变改为零端速度平滑目标：
趴下 3.5 秒、四足起立 5 秒、指令驻留 14–18 秒。手部接触按阶段释放，加入动作
二阶差分、关节速度和竖直速度成本。训练允许最多 2.5 秒躯干接触恢复，严格评估
仍将任何躯干接触记为失败。

## 四足结果

Crawl-E3 `model_599.pt` 完成 512 环境 × 600 轮，共 7,372,800 transitions。
nominal/randomized 各 20 条 8 秒 rollout 均无跌倒。与同协议 E2 对比：

| 条件 | E2 速度 | E3 速度 | E2 双脚平均倾角 | E3 双脚平均倾角 |
| --- | ---: | ---: | ---: | ---: |
| nominal +0.15 m/s | 0.140 | 0.139 | 5.02° | 3.37° |
| randomized +0.15 m/s | 0.143 | 0.141 | 5.09° | 3.76° |
| nominal −0.15 m/s | −0.130 | −0.135 | 3.77° | 3.31° |
| randomized −0.15 m/s | −0.127 | −0.133 | 4.25° | 3.86° |

模型 SHA-256 为
`9bb5ab0d3d2741801c07065cc1d986459374f836b644f42e7acadd1e5386f734`。
它被标记为仿真候选，不是实机部署模型：高速前进滑移代理略有上升，仍需录像和
实机前安全审核。

现已补录 `model_599.pt` 的 nominal 最终评价视频，包含站立、+0.15 m/s 前进和
−0.15 m/s 后退三个 8 秒片段。视频位于
`logs/rsl_rl/pikachu_v033/crawl_e3/forward_refinement_20261023/evaluation/`
`final_model_599/videos/crawl_e3_model_599_final_evaluation_mujoco.mp4`，使用 mjlab
`OffscreenRenderer` 与 MuJoCo 从真实策略 rollout 生成。

## 转换结果

共执行一次主动停止的无效 pilot、一次 1200 轮 pilot 和一次带恢复窗口的 1600 轮
pilot，并对 8 个有效 checkpoint 做 16 环境、10 秒/段、禁用自动 reset 的四种
端点及连续循环评估。所有模型四足起立均为 0/16，完整循环均为 0/16。

恢复窗口让训练轨迹显著变长，部分模型能将躯干转到接近直立，但最终高度仍低约
0.24 m，属于低位姿态而非站起；Down 也发生退化。因此本轮没有转换模型晋升，
也没有写入 `pre_train/`。后续应使用经动力学筛选的中间状态 reset 课程或独立 Up
策略，继续用严格无 reset 循环验收。

本机实验产物已经按任务拆分：四足位于
`logs/rsl_rl/pikachu_v033/crawl_e3/forward_refinement_20261023/`，姿态转换位于
`logs/rsl_rl/pikachu_v033/transfer_e1/gentle_transition_20261023/`。原混合目录
`e3_autonomous_20261023/` 已移除；逐文件旧路径、新路径、大小和 SHA-256 记录在
两个实验各自的 `metadata/artifact_migration.json` 中。

## 同步与 Git 状态

本机和 `finnox@10.20.13.246` 均切换到
`feat/pikachu-v033-e3-20261023`。只同步 16 个 E3 相关源码、测试与文档文件，
不删除远端文件、不传训练日志或模型；远端被覆盖文件备份在
`.sync/backups/20261023T025928Z-e3/`。同步后内容复查为 0 差异，远端相关测试
15 passed，`ty check` 通过。

两端在本轮开始前已有大量相互依赖的未提交 V033 与其他实验修改，因此没有把
无法与既有基线安全分离的内容强行提交，也没有 push 一个不可复现的 commit。
