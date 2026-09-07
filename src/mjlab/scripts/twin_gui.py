"""Pikachu 数字孪生本地控制台（tkinter，替代网页 twin_web.html）。

独立本地窗口、不依赖浏览器、不卡。**内嵌 WebSocket 服务端(8765)**：
  - Unity 连 ``ws://127.0.0.1:8765/ws`` 收 ``frame``（关节数据，含 arm_roll 90° 偏置）；
  - 本窗口即控制台：载入 NPZ / 播放 / 暂停 / 进度拖动 / 倍速，直接操作同一份
    ``twin_server.TwinState``，数据流/控制逻辑/映射关系与 twin_server 完全一致。

用法：
  uv run python -m mjlab.scripts.twin_gui          # 起内置 server 8765 + 控制台窗口
  uv run python -m mjlab.scripts.twin_gui --port 9000
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import tkinter as tk
from tkinter import ttk

import websockets

from mjlab.scripts.twin_server import NPZ_DIR, STATE, handle_ws, player_loop

HOST = "127.0.0.1"
PORT = 8765


# ---------------------------------------------------------------- 内嵌服务端
def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
  asyncio.set_event_loop(loop)
  loop.run_forever()


async def _serve(port: int) -> None:
  async with websockets.serve(handle_ws, HOST, port):
    print(f"[twin] WebSocket 服务端已启动: ws://{HOST}:{port}/ws")
    print(f"[twin] NPZ 目录: {NPZ_DIR}  （本窗口为控制台，Unity 连上即收 frame）")
    asyncio.create_task(player_loop())
    await asyncio.Future()  # 常驻


def start_server(port: int) -> threading.Thread:
  loop = asyncio.new_event_loop()
  t = threading.Thread(target=_run_loop, args=(loop,), daemon=True)
  t.start()
  asyncio.run_coroutine_threadsafe(_serve(port), loop)
  return t


# ---------------------------------------------------------------- 控制台 GUI
class TwinConsole(tk.Tk):
  def __init__(self, port: int) -> None:
    super().__init__()
    self.port = port
    self.title("Pikachu 数字孪生控制台")
    self.geometry("560x400")
    self.resizable(False, False)
    self.files: dict[str, str] = {}

    self.var_file = tk.StringVar()
    self.var_state = tk.StringVar(value=f"ws://{HOST}:{port}/ws  · 未加载 npz")
    self.var_idx = tk.IntVar(value=0)
    self.var_speed = tk.DoubleVar(value=1.0)

    self._build()
    self.refresh()

  def _build(self) -> None:
    pad = {"padx": 8, "pady": 4}

    f1 = ttk.LabelFrame(self, text="动作文件 (NPZ)")
    f1.pack(fill="x", padx=8, pady=6)
    self.combo = ttk.Combobox(f1, textvariable=self.var_file, state="readonly")
    self.combo.pack(side="left", fill="x", expand=True, **pad)
    ttk.Button(f1, text="刷新", command=self.refresh).pack(side="left")
    ttk.Button(f1, text="载入", command=self.load).pack(side="left")

    self.state_lbl = ttk.Label(self, textvariable=self.var_state, foreground="#1a6fb5")
    self.state_lbl.pack(fill="x", **pad)

    f2 = ttk.Frame(self)
    f2.pack(fill="x", **pad)
    self.btn_play = ttk.Button(f2, text="▶ 播放", command=self.playpause, width=8)
    self.btn_play.pack(side="left")
    self.btn_pause = ttk.Button(f2, text="❚❚ 暂停", command=self.pause, width=8)
    self.btn_pause.pack(side="left")
    self.var_pos = tk.StringVar(value="帧 - / -")
    ttk.Label(f2, textvariable=self.var_pos, width=14).pack(side="left")

    f3 = ttk.LabelFrame(self, text="进度")
    f3.pack(fill="x", padx=8, pady=4)
    self.scale_prog = ttk.Scale(f3, from_=0, to=1, command=self.seek)
    self.scale_prog.pack(fill="x", padx=8, pady=4)

    f4 = ttk.LabelFrame(self, text="倍速")
    f4.pack(fill="x", padx=8, pady=4)
    ttk.Scale(
      f4, from_=0.05, to=4.0, variable=self.var_speed, command=self.speed_cb
    ).pack(fill="x", padx=8, pady=2)
    self.speed_lbl = ttk.Label(f4, text="1.00×")
    self.speed_lbl.pack(padx=8)

    ttk.Label(
      self,
      text="提示：Unity 连 ws://127.0.0.1:8765/ws；本窗口即控制端。",
      foreground="#666",
    ).pack(side="bottom", pady=6)

  # ---- 动作 ----
  def refresh(self) -> None:
    self.files = {p: p for p in STATE.list_motion()}
    self.combo["values"] = list(self.files.keys())

  def load(self) -> None:
    name = self.var_file.get()
    if not name or name not in self.files:
      return
    ok = STATE.load(self.files[name])
    self.var_state.set(
      f"ws://{HOST}:{self.port}/ws  · 已载入: {name}" if ok else f"载入失败: {name}"
    )
    self.var_idx.set(0)

  def playpause(self) -> None:
    if STATE.joints is None:
      return
    STATE.playing = not STATE.playing

  def pause(self) -> None:
    STATE.playing = False

  def seek(self, v) -> None:  # 拖动进度
    if STATE.joints is None:
      return
    STATE.idx = int(float(v))
    self.var_idx.set(STATE.idx)

  def speed_cb(self, v) -> None:
    STATE.speed = max(0.05, float(v))
    self.speed_lbl.config(text=f"{STATE.speed:.2f}×")

  # ---- 轮询刷新（读共享状态，GIL 下线程安全）----
  def poll(self) -> None:
    j = STATE.joints
    if j is not None:
      total = len(j)
      self.scale_prog.configure(to=total - 1 if total > 1 else 1)
      idx = min(STATE.idx, total - 1)
      self.scale_prog.set(idx)
      self.var_pos.set(f"帧 {idx + 1} / {total}")
    else:
      self.var_pos.set("帧 - / -")
    self.btn_play.config(text="❚❚ 暂停(正在播放)" if STATE.playing else "▶ 播放")
    self.after(120, self.poll)


def main() -> None:
  ap = argparse.ArgumentParser(description="Pikachu 数字孪生本地控制台（tkinter）")
  ap.add_argument("--port", type=int, default=PORT, help="WS 端口")
  args = ap.parse_args()
  start_server(args.port)
  TwinConsole(args.port).mainloop()


if __name__ == "__main__":
  main()
