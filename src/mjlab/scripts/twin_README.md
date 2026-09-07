# Pikachu 数字孪生 — Python 服务端

把 `../mocap/csv/*.csv`（21 列 = root_pos 3 + root_quat xyzw 4 + 14 关节弧度）按序
经 WebSocket 推给 Unity 数字孪生客户端。

## 启动
```sh
uv run python -m mjlab.scripts.twin_server
#  ws → ws://127.0.0.1:8765/ws
# 网页 → http://127.0.0.1:8000/   （或直接浏览器打开 twin_web.html）
```

## 网页控制台
选 CSV → 载入 → 播放；倍速 / 进度滑条即时生效。网页经同一 `/ws` 发控制命令
（`load / play / pause / seek / speed`），服务端负责按 fps×倍速定节奏、循环播放、
并回推 `state`/进度；Unity 连接只收 `frame`。

## 协议
见 `twin_server.py` 模块 docstring（frame / state / list / 控制命令）与
Unity 端文档 `My project/DIGITAL_TWIN.md`。

## 文件
- `twin_server.py` — 服务端（WS + 可选 HTTP 页）
- `twin_web.html` — 独立网页控制台（可 file:// 打开）