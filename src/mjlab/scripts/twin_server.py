"""Pikachu 数字孪生 WebSocket 服务端。

把 ``mocap/npz`` 里的动作数据（``joint_pos``(T×14) 弧度 + ``body_pos_w/body_quat_w``，
base 取 body 索引 0）按序逐帧经 WS 推送给 Unity 端，达到「Python 播放 mocap 数据
→ Unity 实时驱动 URDF + 皮肤」的数字孪生效果。

- 纯 WebSocket（``websockets``），默认端口 8765：
  - Unity 端作为 WS **客户端**连 ``ws://<host>:8765/ws`` 收 ``frame`` 命令。
  - 浏览器打开同地址的网页：连同一 WS，向服务端发 ``load/play/pause/seek/speed``
    控制命令；服务端负责按 fps × 倍速定播放节奏，进度实时回推网页。
- 顺带在 ``--http`` 端口(默认 8001)提供网页（也可直接浏览器打开
  ``twin_web.html``，它请求 file:// 也能连 ws://127.0.0.1:8765/ws）。

消息协议（均为 JSON 文本帧）：
  服务端 → Unity：　{"cmd":"frame","idx":N,"n":total,"fps":f,"pos":[x,y,z],
                     "quat":[x,y,z,w],   # mj 世界系，xyzw
                     "joints":[...14...]}# dof 列序，弧度
  服务端 → 网页：　{"cmd":"state","file":..,"connected_web":w,"connected_unity":u,
                     "playing":b,"idx":N,"n":total,"speed":s,"fps":f}
  网页 → 服务端：　{"cmd":"load","file":..} / "play" / "pause"
                     / {"cmd":"seek","idx":N} / {"cmd":"speed","value":v}

关节 dof 列序（= PIKACHU_JOINT_NAMES）：
  left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle,
  right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle,
  left_arm_pitch, left_arm_roll, right_arm_pitch, right_arm_roll

arm_roll 90° 对正（bake 进帧内）：npz 的 arm_roll 相对手臂下垂，URDF/皮肤以 90° 抬臂为
装配位，故每帧发前把 right = π/2 − v、left = −π/2 − v（v 为 npz 弧度），其余 12 关节
原样弧度。

用法：
  uv run python -m mjlab.scripts.twin_server          # ws:8765 (+http:8001)
  uv run python -m mjlab.scripts.twin_server --port 9000 --http 8080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
import websockets

# NPZ 目录（server 在 src/mjlab/scripts/ → src/mjlab/mocap/npz，含子目录）
NPZ_DIR = Path(__file__).resolve().parent.parent / "mocap" / "npz"
WEB_HTML = Path(__file__).resolve().parent / "twin_web.html"
DEFAULT_FPS = 30.0

# 14 关节 dof 列序（= PIKACHU_JOINT_NAMES，仅带后缀）：arm_roll 索引 left=11 / right=13
IDX_LEFT_ARM_ROLL = 11
IDX_RIGHT_ARM_ROLL = 13


class TwinState:
  """域名状态数据中心（单例）。数据源 = npz（csv_to_npz_pikachu 输出）。"""

  def __init__(self) -> None:
    self._lock = asyncio.Lock()
    self.file = ""  # 当前选定文件名（NPZ_DIR 下相对路径）
    self.joints: np.ndarray | None = None  # (T, 14) 弧度
    self.root_pos: np.ndarray | None = None  # (T, 3)  base_link 世界位置
    self.root_quat: np.ndarray | None = None  # (T, 4)  base_link 四元数 xyzw
    self.idx = 0  # 当前播放帧号
    self.speed = 1.0  # 倍速
    self.fps = float(DEFAULT_FPS)
    self.playing = False
    self.web: set = set()  # 网页连接集合
    self.unity: set = set()  # Unity 连接集合

  @property
  def total(self) -> int:
    return 0 if self.joints is None else len(self.joints)

  def list_motion(self) -> list[str]:
    if not NPZ_DIR.is_dir():
      return []
    return [str(p.relative_to(NPZ_DIR)) for p in sorted(NPZ_DIR.rglob("*.npz"))]

  def load(self, name: str) -> bool:
    path = NPZ_DIR / name
    if not path.is_file():
      return False
    d = np.load(path, allow_pickle=True)
    self.joints = np.asarray(d["joint_pos"], dtype=float)  # (T,14) 弧度
    body_pos = np.asarray(d["body_pos_w"], dtype=float)  # (T, Nbody, 3)
    body_quat = np.asarray(d["body_quat_w"], dtype=float)  # (T, Nbody, 4) wxyz
    # body 0 = base_link；quat 由 MuJoCo 原生 wxyz 转 xyzw
    self.root_pos = body_pos[:, 0, :3]
    self.root_quat = body_quat[:, 0, [1, 2, 3, 0]]
    self.fps = (
      float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d else float(DEFAULT_FPS)
    )
    self.file = name
    self.idx = 0
    self.playing = False
    return True

  def frame(self, i: int) -> dict:
    """取出第 i 帧的转发载荷（mj base pos / base quat xyzw + 14 关节弧度）。
    arm_roll 偏置：right = 90°-v、left = -90°-v（v 为 npz 弧度），折算到弧度域
    （right=π/2-v，left=-π/2-v）；其余关节原样弧度。"""
    jp = np.array(self.joints[i], dtype=float, copy=True)
    # 容错：纯腿 npz 只有 10 关节（无 arm），按实际长度再对 arm_roll 做 90° 偏置
    if jp.shape[0] > IDX_RIGHT_ARM_ROLL:
      jp[IDX_RIGHT_ARM_ROLL] = np.pi / 2.0 - jp[IDX_RIGHT_ARM_ROLL]
    if jp.shape[0] > IDX_LEFT_ARM_ROLL:
      jp[IDX_LEFT_ARM_ROLL] = -np.pi / 2.0 - jp[IDX_LEFT_ARM_ROLL]
    return {
      "cmd": "frame",
      "idx": int(i),
      "n": int(self.total),
      "fps": float(self.fps),
      "pos": [float(x) for x in self.root_pos[i]],
      "quat": [float(x) for x in self.root_quat[i]],
      "joints": [float(x) for x in jp],
    }

  def state_msg(self) -> dict:
    return {
      "cmd": "state",
      "file": self.file,
      "connected_web": len(self.web),
      "connected_unity": len(self.unity),
      "playing": self.playing,
      "idx": int(self.idx),
      "n": int(self.total),
      "speed": float(self.speed),
      "fps": float(self.fps),
    }


STATE = TwinState()


async def broadcast_to(conns: set, msg: dict) -> None:
  if not conns:
    return
  text = json.dumps(msg)
  dead = []
  for c in list(conns):
    try:
      await c.send(text)
    except Exception:
      dead.append(c)
  for c in dead:
    conns.discard(c)


async def player_loop() -> None:
  """按 fps×倍速 驱动播放。"""
  while True:
    if STATE.playing and STATE.joints is not None and STATE.unity:
      await broadcast_to(STATE.unity, STATE.frame(STATE.idx))
      STATE.idx += 1
      if STATE.idx >= STATE.total:
        STATE.idx = 0  # 循环播放
    else:
      await asyncio.sleep(0.02)
    if STATE.joints is not None:
      # 向网页回推进度（单位步，量小，频率由播放节奏决定）
      await broadcast_to(STATE.web, STATE.state_msg())
    # 播放节奏：每次 sleep 用当前速度
    dt = 1.0 / (STATE.fps * STATE.speed)
    await asyncio.sleep(max(dt, 0.001))


async def handle_ws(websocket) -> None:
  """统一 /ws 端点：网页控制命令 vs Unity 只收 frame。"""
  # 握手后第一条判断身份：网页会先发初始化命令；Unity 不发。
  try:
    # 先用周边空位注册为 Unindyause——改为先按消息冒烟：
    # 我们无法预知身份，用「先收到控制命令是 web，什么都还没收到先标记 unity」。
    # 更稳：网页首条总是 {"cmd":"load" 或 "ping"}，Unity 不做主动下发。这里先都按
    # Unity 注册，收到 web 命令时移入 web 集合。
    STATE.unity.add(websocket)
    async for raw in websocket:
      try:
        msg = json.loads(raw)
      except Exception:
        continue
      cmd = msg.get("cmd", "")
      # 网页控制 → 从 unity 集合挪到 web 集合
      if (
        STATE.unity
        and websocket in STATE.unity
        and cmd in ("load", "play", "pause", "seek", "speed", "list", "ping")
      ):
        STATE.unity.discard(websocket)
        STATE.web.add(websocket)

      if cmd == "list":
        await websocket.send(json.dumps({"cmd": "list", "files": STATE.list_motion()}))
      elif cmd == "load":
        ok = STATE.load(msg.get("file", ""))
        if ok:
          await broadcast_to(STATE.web, STATE.state_msg())
        else:
          await websocket.send(
            json.dumps({"cmd": "error", "text": "NPZ 未找到: " + msg.get("file", "")})
          )
      elif cmd == "play":
        STATE.playing = True
      elif cmd == "pause":
        STATE.playing = False
      elif cmd == "seek":
        n = int(msg.get("idx", 0))
        if STATE.joints is not None:
          STATE.idx = max(0, min(n, STATE.total - 1))
      elif cmd == "speed":
        STATE.speed = max(0.05, float(msg.get("value", 1.0)))
      elif cmd == "ping":
        await websocket.send(json.dumps({"cmd": "pong"}))
      await broadcast_to(STATE.web, STATE.state_msg())
  finally:
    STATE.unity.discard(websocket)
    STATE.web.discard(websocket)
    await broadcast_to(STATE.web, STATE.state_msg())


# ---------------------------------------------------------------- HTTP 页
class _HttpHandler(SimpleHTTPRequestHandler):
  def do_GET(self):
    if self.path in ("/", "/index.html"):
      body = (
        WEB_HTML.read_bytes()
        if WEB_HTML.exists()
        else b"<h1>twin_web.html missing</h1>"
      )
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
    else:
      super().do_GET()


def _run_http(host: str, port: int) -> None:
  try:
    srv = HTTPServer((host, port), _HttpHandler)
  except OSError as e:
    print(
      f"[twin] 警告: HTTP 页端口 {port} 被占用({e})，跳过网页（WS 仍可用）。"
      f"如需网页：停掉占用该端口的进程，或 --http 换端口。"
    )
    return
  print(
    f"[twin] HTTP 页(可访问 http://{host}:{port}/ 或直接打开 twin_web.html): 端口 {port}"
  )
  srv.serve_forever()


async def main() -> None:
  ap = argparse.ArgumentParser(description="Pikachu 数字孪生 WS 服务端")
  ap.add_argument("--port", type=int, default=8765, help="WS 端口")
  ap.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
  ap.add_argument("--http", type=int, default=8001, help="HTTP 页端口(0 关闭)")
  args = ap.parse_args()

  if args.http and args.http > 0:
    threading.Thread(target=_run_http, args=(args.host, args.http), daemon=True).start()

  async with websockets.serve(handle_ws, args.host, args.port):
    print(f"[twin] WebSocket 服务端已启动: ws://{args.host}:{args.port}/ws")
    print(f"[twin] NPZ 目录: {NPZ_DIR}")
    if STATE.list_motion():
      print("[twin] 可用 NPZ: " + ", ".join(STATE.list_motion()))
    else:
      print("[twin] 警告: NPZ 目录无 .npz 文件")
    print("[twin] Unity 插件连接后即可接收 frame 数据。Ctrl+C 退出。")
    await player_loop()


if __name__ == "__main__":
  asyncio.run(main())
