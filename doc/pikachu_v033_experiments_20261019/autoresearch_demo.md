# autoresearch流程体验：Pikachu RL适配

参考：https://github.com/karpathy/autoresearch 。上游README和program.md原文已下载并留存在本轮审计目录的`autoresearch_upstream/`。
本次运行的是迁移到PPO的有界流程演示：由我预先选3个学习率候选，脚本执行短训、固定评估、比较、记录和Git回退。原版的GPT训练程序未运行。

流程：`recipe.json`单点修改 → 候选Git提交 → 训练 → 固定评估 → 写`results.tsv` → 保留或回退。

## 已实际运行的结果

| 候选 | 学习率 | 验证得分，越低越好 | 决策 | 候选提交 |
|---|---:|---:|---|---|
| baseline | 0.0003 | 1.199156 | keep | `31ad193` |
| 降低学习率 | 0.0001 | 2.347469 | discard，已回退 | `7a30379` |
| 提高学习率 | 0.003 | 17504.398500 | discard，已回退 | `d92dbe5` |

高学习率候选在固定验证中大量跌倒，得分包含每次跌倒的100分成本；原始验证JSON可检查误差与跌倒分别贡献多少。
最终recipe仓库HEAD为baseline `31ad193`，所有候选分支`trial-0/1/2`保留；失败候选的日志和checkpoint未删除。
三次总耗时约191秒，每项耗时包含启动、30轮训练、三个命令评估。

## 固定规则

- 任务：直立E1；64环境 × 30轮 × 24步 = 每次46,080条转移；完整18维动作和512/256/128网络。
- 所有候选：训练seed=42，相同`pilot-v2-robust-standing/model_499.pt` actor-only初始化；critic/optimizer/iteration/curriculum全新。
- 只调整学习率，adaptive调度保留。正式4096×20000任务随机初始化。
- 验证seed=123，8环境 × 6秒，分别测试stand、forward(.4m/s)、turn(.3rad/s)。nominal场景，关闭startup摩擦/COM/encoder随机化；这是调参验证集，不是独立最终测试集。
- score = 三种命令各自`mean(MAE / [0.4,0.4,0.3])`之和 + `100 × falls_total`；比当前最优低超过0.001才保留。
- 同预算采用固定转移数；上游采用固定5分钟wall-clock。这里没有比较不同网络吞吐的能力。
- 单seed、极短训练只用于体验流程，不能据此认定3e-4是正式长训最优值；正式初始学习率仍为1e-3。

## 自己操作一次

从mjlab根目录：

```bash
# 看实际决策日志
cat logs/rsl_rl/pikachu_v033/_audit/e1_full_20261020/research/data/autoresearch_demo/results.tsv

# 看所有候选与回退后的HEAD；独立仓库不影响mjlab工作分支
git -C logs/rsl_rl/pikachu_v033/_audit/e1_full_20261020/autoresearch_demo/recipe_repo log --all --oneline --decorate
git -C logs/rsl_rl/pikachu_v033/_audit/e1_full_20261020/autoresearch_demo/recipe_repo diff trial-0 trial-2 -- recipe.json
cat logs/rsl_rl/pikachu_v033/_audit/e1_full_20261020/research/autoresearch_demo/recipe_repo/recipe.json
```

需要重复体验时，在GPU空闲后使用新的输出目录（脚本拒绝覆盖已有实验）：

```bash
uv run --no-sync python scripts/demo_v033_autoresearch.py \
  --out logs/rsl_rl/pikachu_v033/_audit/autoresearch_demo_repeat \
  --actor-init logs/rsl_rl/pikachu_v033/biped_e1/diagnostic_archive_20261022/training/2026-10-20_22-13-28_pilot-v2-robust-standing/checkpoints/model_499.pt
```

实验规则副本在demo仓库`program.md`；实际候选列表与评分执行在`scripts/demo_v033_autoresearch.py`。
脚本不会将修改自动带入正式任务。进一步的自主搜索需要明确可改参数、固定训练预算、多seed确认和独立最终测试。
