# Pikachu 前进行走 Tracking 任务

`Mjlab-Tracking-Flat-Pikachu-Walk` 使用下面的 50 Hz 动作作为默认参考：

```text
src/mjlab/mocap/npz/pikachu_walk_npz/pikachu_walk_0.185_0.0_0.0.npz
```

任务复用 Pikachu transition tracking 的机器人、动作缩放、11 个追踪 body、
手脚末端终止条件、自碰撞检测、域随机化和 PPO 超参数。训练模式默认使用
`adaptive` 起始帧采样；play 模式关闭观测与 reset 扰动，并从第 0 帧开始。

## 训练

```sh
uv run train Mjlab-Tracking-Flat-Pikachu-Walk \
    --env.scene.num_envs=4096
```

建议先进行短程冒烟训练：

```sh
uv run train Mjlab-Tracking-Flat-Pikachu-Walk \
    --env.scene.num_envs=1024 \
    --agent.max_iterations=500 \
    --agent.run-name trial-pikachu-walk-forward-dual-test

```

双卡训练：
```sh
uv run train Mjlab-Tracking-Flat-Pikachu-Walk \
    --gpu-ids "[0, 1]" \
    --env.scene.num_envs=4096 \
    --agent.max_iterations=5000 \
    --agent.run-name trial-pikachu-walk-forward-dual

```



## 检查参考动作

dummy agent 模式下，`play` 仍要求显式传入 motion 文件：

```sh
uv run play Mjlab-Tracking-Flat-Pikachu-Walk \
    --agent zero \
    --motion-file \
      src/mjlab/mocap/npz/pikachu_walk_npz/pikachu_walk_0.185_0.0_0.0.npz \
    --viewer viser \
    --num_envs 1 \
    --no-terminations
```

## 回放 checkpoint

```sh
uv run play Mjlab-Tracking-Flat-Pikachu-Walk \
    --checkpoint-file logs/rsl_rl/pikachu_tracking/<run>/model_<step>.pt \
    --motion-file \
      src/mjlab/mocap/npz/pikachu_walk_npz/pikachu_walk_0.185_0.0_0.0.npz \
    --viewer viser
```

测试回放：
```
uv run play Mjlab-Tracking-Flat-Pikachu-Walk     --checkpoint-file logs/rsl_rl/pikachu_tracking/2026-08-13_17-18-38/model_499.pt     --motion-file       src/mjlab/mocap/npz/pikachu_walk_npz/pikachu_walk_0.185_0.0_0.0.npz     --viewer viser
```
