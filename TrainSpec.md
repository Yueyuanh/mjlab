# 本地标准调参 / 训练测试流程

## 网页实验启动台（推荐）

在仓库根目录运行：

```sh
uv run start_server
```

脚本会打开仅监听本机的网页，自动发现注册任务、`logs/` 和 `pre_train/`
下每个 run 的最高编号 checkpoint，以及 `src/mjlab/mocap/npz/` 下的动作。
Tracking 动作优先从 checkpoint 的 `params/env.yaml` 精确恢复，找不到时才根据
任务和 run 名匹配，并在启动前要求确认。网页可生成/复制 train、play、response
和 W&B tracking evaluate 命令，也可直接启动、查看实时日志和停止任务。

默认情况下，只要检测到本仓库已有 train/play/response/evaluate 进程，或所选 GPU
上已有计算进程，就会拒绝新启动；确实需要并行时可显式勾选“忽略冲突”。服务器
退出不会结束已经启动的实验，实验输出同时保存在 `.start_server/jobs/`。

网页启动 Viser 回放时会自动分配独立端口。任务卡中的“打开 Viser”会跳转到
当前浏览器主机对应的 Viser 地址；远程机器上运行时，先点击“复制 SSH 转发”，
在本地终端运行生成的命令，再点击链接。SSH 目标默认是
`当前用户@机器主机名`，可在网页中修改，也可启动时指定：

```sh
uv run start_server --ssh-target yons@your-server
```

已退出但未被父进程回收的 zombie、空壳 `uv` / `torchrunx` launcher 和
`nvidia-smi` 的 `[No data]` 行会标为历史残留，不再冒充活跃任务或阻止启动。

不自动打开浏览器或改端口：

```sh
uv run start_server --no-open --port 8011
```

① 环境 sanity check（改完 MDP/奖励/机器人后必做）

cd /home/finnox/Pikachu/PikachuRobot/pikachu_playground/mjlab
uv run list-envs --keyword Velocity        # 看有哪些任务
uv run play Mjlab-Velocity-Flat-Unitree-G1 --agent zero     # 零动作，验证 reset/MDP 不炸
uv run play Mjlab-Velocity-Flat-Unitree-G1 --agent random   # 随机动作，验证奖励/终止有反应
② 小规模试训（先验证收敛，再放大）

uv run train Mjlab-Velocity-Flat-Unitree-G1 \
    --env.scene.num-envs 2048 \
    --agent.max-iterations 500 \
    --agent.run-name trial-01

uv run train Mjlab-Velocity-Flat-Pikachu \
    --env.scene.num-envs 2048 \
    --agent.max-iterations 500 \
    --agent.run-name trial-01


uv run train Mjlab-Velocity-Flat-Pikachu \
    --env.scene.num-envs 2048 \
    --agent.max-iterations 3000 \
    --agent.run-name trial-hip-pitch-3507
    
③ 调参（tyro 点路径覆盖，全配置可覆盖）

# 先查可用 flag
uv run train Mjlab-Velocity-Flat-Unitree-G1 --help | grep learning-rate

uv run train Mjlab-Velocity-Flat-Unitree-G1 \
    --env.scene.num-envs 4096 \
    --env.decimation 2 \
    --agent.max-iterations 30000 \
    --agent.algorithm.learning-rate 3e-4 \
    --agent.algorithm.clip-param 0.2 \
    --agent.algorithm.num-mini-batches 4 \
    --agent.actor.hidden-dims "(512,256,128)" \
    --agent.actor.obs-normalization True
三个关键语法（文档里明确标注）：

连字符分段 + 点路径：全字段用点路径覆盖（如 --env.scene.num-envs，不是 --num-envs，也不是 --num_envs）；
布尔必须显式给值：--agent.resume True，不能只写 --agent.resume（为兼容 W&B sweep）；
全字段可覆盖：--env.* 覆盖环境，--agent.* 覆盖 PPO/网络。
④ 断点续训 + 评估 + 视频

# 续训：自动找 experiment 下最近的 run 目录，载入最高编号 checkpoint
# --agent.max-iterations 是"从断点起再加跑多少轮"
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
    --agent.resume True \
    --agent.load-run 2025-.* \
    --agent.load-checkpoint model_5000.* \
    --agent.max-iterations 1000

# 训练中录视频（rank0，每 2000 步录一段）
uv run train Mjlab-Velocity-Flat-Unitree-G1 --video

# 日志器：默认 wandb（需已登录，wandb login）。在线看曲线/存超参。
#   若需纯本地，可显式 --agent.logger tensorboard（log 仍会落盘到
#   logs/rsl_rl/<exp>/<ts>/events.*，用 tensorboard --logdir 查看）。

# 评估
uv run play Mjlab-Velocity-Flat-Unitree-G1 \
    --checkpoint-file logs/rsl_rl/g1_velocity/<timestamp>/model_5000.pt
# 或从 W&B
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path <entity>/mjlab/<run-id>
产物结构

logs/rsl_rl/{experiment_name}/{timestamp}/
    model_{iter}.pt      # checkpoint（默认每 save_interval 存，并上传 W&B）
    params/env.yaml      # 本次完整环境配置（含命令行覆盖）
    params/agent.yaml    # 本次完整训练配置
    videos/train/        # --video 时的训练视频
代码改动后的规范动作
训练启动后（参数已确认）：立即提交 git commit，锁定代码版本（训练时不要改参数，
因为 params/env.yaml 是启动时的快照，中途改代码会造成"回放用新代码"的错位）。
回放（play）前：先 git 对齐代码版本 —— 确认当前代码与训练时的 commit 一致，
必要时 git checkout <训练时的commit>，再 play，保证环境构建与训练完全一致。
每次 commit 前：make check（format + lint + type）。
PR 前：make test。
用户可见改动：在 docs/source/changelog.rst 的 "Upcoming version" 加条目。


## w&b
# velocity
uv run train Mjlab-Velocity-Flat-Pikachu \
    --agent.wandb-project pikachu-velocity \
    --agent.wandb-tags velocity \
    --env.scene.num_envs=4096

# transition down / up
uv run train Mjlab-Tracking-Flat-Pikachu-Down \
    --agent.wandb-project pikachu-transition \
    --agent.wandb-tags transition \
    --env.scene.num_envs=4096

## GPU

  --gpu-ids "[0, 1]" \

## 浏览器界面 可以用摇杆发送指令

uv run play Mjlab-Velocity-Flat-Pikachu \
    --checkpoint-file logs/rsl_rl/pikachu_velocity/2026-08-11_12-37-14_trial-hip-pitch-3507/model_4999.pt \
    --viewer viser

# 所有 Pikachu 任务的 Viser Metrics 页都可逐关节勾选实时曲线，也可用
# Joint plots 下的 Position / Velocity / Torque 一键显示同类信号。单位分别为
# rad、rad/s、N·m；TensorBoard 的 Episode_Metrics 记录每回合最大绝对值。
# CSV data 中先勾选曲线，再点 Record selected / Stop & save；CSV 默认保存在
# checkpoint 所在目录。Saved CSV + Open saved CSV 可直接打开服务器文件，
# Upload CSV 可从当前浏览器选文件，Return to live 恢复实时曲线。
# 加载 CSV 后可勾选需要的曲线并点 Save selected plots + comparisons。图片保存到
# CSV 同级的 images/<CSV文件名>/，关节数据再按 joint_position / joint_velocity /
# joint_torque 和 left / right 分类；同时生成单侧总览、同类型总览和 1列×7行左右对比图。
# 独立 CSV 分析页：scripts/plot_logs/metrics_csv_dashboard.html
# 推荐启动：python -m http.server 8000 -d scripts/plot_logs
# 然后打开：http://localhost:8000/metrics_csv_dashboard.html

uv run play Mjlab-Tracking-Flat-Pikachu-Down --checkpoint-file logs/rsl_rl/pikachu_tracking/2026-08-12_17-05-14_trail-pikachu-transition-down-test/mod
el_499.pt  --motion-file src/mjlab/mocap/npz/Pikachu-Transition-Down.npz   --viewer viser


uv run play Mjlab-Tracking-Flat-Pikachu-Up --checkpoint-file logs/rsl_rl/pikachu_tracking/2026-08-12_17-05-14_trail-pikachu-transition-up-test/model_499.pt  --motion-file src/mjlab/mocap/npz/Pikachu-Transition-Up.npz   --viewer viser



## tracking train

uv run train Mjlab-Tracking-Flat-Pikachu-Walk \
    --gpu-ids "[0, 1]" \
    --env.scene.num_envs=4096 \
    --agent.max_iterations=5000 \
    --env.commands.motion.motion-file src/mjlab/mocap/npz/pikachu_walk_stand_npz/Pikachu-rmg-stand-0.2.npz \
    --agent.run-name trial-pikachu-walk-stand-forward-dual

## tracking 回放

```
起立

uv run play Mjlab-Tracking-Flat-Pikachu-Up \
--checkpoint-file pre_train/rsl_rl/pikachu_tracking/2026-08-12_20-03-22_trial-pikachu-transition-up-dual/model_4999.pt \
--motion-file src/mjlab/mocap/npz/Pikachu-Transition-Up.npz \
--viewer viser

趴下
uv run play Mjlab-Tracking-Flat-Pikachu-Down \
--checkpoint-file pre_train/rsl_rl/pikachu_tracking/2026-08-12_20-55-27_trial-pikachu-transition-down-dual/model_4999.pt \
--motion-file src/mjlab/mocap/npz/Pikachu-Transition-Down.npz   \
--viewer viser

前进
uv run play Mjlab-Tracking-Flat-Pikachu-Walk   \
--checkpoint-file pre_train/rsl_rl/pikachu_tracking/2026-08-13_17-27-46_trial-pikachu-walk-forward-dual/model_4999.pt \
 --motion-file  src/mjlab/mocap/npz/pikachu_walk_npz/pikachu_walk_0.185_0.0_0.0.npz     --viewer viser

后退
uv run play Mjlab-Tracking-Flat-Pikachu-Walk  \
--checkpoint-file pre_train/rsl_rl/pikachu_tracking/2026-08-13_18-28-36_trial-pikachu-walk-back-dual/model_4999.pt \
--motion-file src/mjlab/mocap/npz/pikachu_walk_npz/pikachu_walk_-0.185_0.0_0.0.npz  \
--viewer viser

```

## tracking npz 验证
uv run play Mjlab-Tracking-Flat-Pikachu-Up --agent zero --motion-file src/mjlab/mocap/npz/Pikachu-rmg-walk-0.185.npz --no-terminations True


上面的 play 用本地 checkpoint 做**定性**观察。要**定量**指标（MPKPE /
R-MPKPE / joint_vel_error / ee_* / success_rate），用 `evaluate.py`。注意它
**只支持 W&B 上的 run**：第一个参数是任务 ID（必填），`--wandb-run-path` 填
`entity/project/run_id`（W&B 网页上该 run 地址 URL 的后三段）：

```sh
uv run python src/mjlab/tasks/tracking/scripts/evaluate.py \
    Mjlab-Tracking-Flat-Pikachu-Down \
    --wandb-run-path <entity>/<project>/<run_id>


# https://wandb.ai/tup_yyh/mjlab/runs/b7w8gngg?nw=nwuser3117144922
uv run python src/mjlab/tasks/tracking/scripts/evaluate.py \
    Mjlab-Tracking-Flat-Pikachu-Down \
    --wandb-run-path tup_yyh/mjlab/b7w8gngg \
    --motion-file src/mjlab/mocap/npz/Pikachu-Transition-Down.npz \
    --num_envs 512 \
    --output-file src/mjlab/mocap/eval/pikachu_down_eval.json

# 在本地跑，不要在服务器上跑
uv run python src/mjlab/tasks/tracking/scripts/evaluate.py \
    Mjlab-Tracking-Flat-Pikachu-Walk \
    --wandb-run-path tup_yyh/pikachu-walk-forward/5zjp9cj0 \
    --motion-file src/mjlab/mocap/npz/pikachu_walk_npz/pikachu_walk_0.185_0.0_0.0.npz \
    --num_envs 512 \
    --output-file src/mjlab/mocap/eval/pikachu_forward_eval.json


```

# velocity train
uv run train Mjlab-Velocity-Flat-Pikachu-Quad \
  --agent.run-name trial-quad-1 \
  --agent.max_iterations 5000 \
  --env.scene.num_envs 4096


uv run play Mjlab-Tracking-Flat-Pikachu-27dof-Quad-Walk --motion-file src/mjlab/mocap/npz/quad_27dof_quad.npz --checkpoint-file logs/rsl_rl/pikachu_27dof_quad_walk/2026-09-10_20-28-04_quad_tracking/model_4999.pt --viewer viser




# scp rsync

拉到本地play

rsync -avz --progress yons@192.168.0.117:/home/yons/pikachu/mjlab/logs/ /home/finnox/Pikachu/PikachuRobot/pikachu_playground/mjlab/logs/4090-logs





# 4090-dual git 无权限

```
# 1. 启动 SSH Agent
eval "$(ssh-agent -s)"

# 2. 添加密钥
4090-dual

ssh-add ~/.ssh/id_ed25519_company

finnox4090
ssh-add ~/.ssh/id_ed25519_yue_gitlab


# 3. 测试连接
ssh -T git@git.finnoxcorp.com -p 2222

# 4. 如果成功，推送
git push
```

$ uv run list-envs
+---------------------------------------------------------+
|             Available Environments in mjlab             |
+----+----------------------------------------------------+
| #  | Task ID                                            |
+----+----------------------------------------------------+
| 1  | Mjlab-Cartpole-Balance                             |
| 2  | Mjlab-Cartpole-Swingup                             |
| 3  | Mjlab-Lift-Cube-Yam                                |
| 4  | Mjlab-Lift-Cube-Yam-Depth                          |
| 5  | Mjlab-Lift-Cube-Yam-Rgb                            |
| 6  | Mjlab-Multi-Cube-Seg-Yam                           |
| 7  | Mjlab-Tracking-Flat-Pikachu-Down                   |
| 8  | Mjlab-Tracking-Flat-Pikachu-Sl21-Down              |
| 9  | Mjlab-Tracking-Flat-Pikachu-Sl21-Up                |
| 10 | Mjlab-Tracking-Flat-Pikachu-Sl21-Walk              |
| 11 | Mjlab-Tracking-Flat-Pikachu-Sl22-Walk              |
| 12 | Mjlab-Tracking-Flat-Pikachu-Up                     |
| 13 | Mjlab-Tracking-Flat-Pikachu-Walk                   |
| 14 | Mjlab-Tracking-Flat-Pikachu-Walk-Base              |
| 15 | Mjlab-Tracking-Flat-Unitree-G1                     |
| 16 | Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation |
| 17 | Mjlab-Velocity-Flat-Pikachu                        |
| 18 | Mjlab-Velocity-Flat-Pikachu-Quad                   |
| 19 | Mjlab-Velocity-Flat-Pikachu-Quad-Bounding          |
| 20 | Mjlab-Velocity-Flat-Pikachu-Sl22-Quad              |
| 21 | Mjlab-Velocity-Flat-Unitree-G1                     |
| 22 | Mjlab-Velocity-Flat-Unitree-Go1                    |
| 23 | Mjlab-Velocity-Rough-Pikachu                       |
| 24 | Mjlab-Velocity-Rough-Unitree-G1                    |
| 25 | Mjlab-Velocity-Rough-Unitree-Go1                   |
+----+----------------------------------------------------+

