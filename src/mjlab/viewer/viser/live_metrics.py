"""Opt-in bounded metrics feed, served on loopback without touching the sim thread."""

from __future__ import annotations

import json
import math
import secrets
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import parse_qs, urlparse


class MetricsLivePublisher:
  """Keep at most 15k control samples; browser clients fetch deltas on demand."""

  def __init__(self, output_dir: Path, names: list[str], step_dt: float):
    if not math.isfinite(step_dt) or step_dt <= 0:
      raise ValueError("Invalid control period")
    self.names, self.step_dt = names, step_dt
    self._rows: deque = deque(maxlen=15_000)
    self._lock = Lock()
    self._sequence = 0
    self._epoch = 0
    self._first_step: int | None = None
    self._last_step = -1
    self._identity: tuple[int, str] | None = None
    self._last_sample_at = 0.0
    self.closed = False
    self.token = secrets.token_urlsafe(32)
    publisher = self

    class Handler(BaseHTTPRequestHandler):
      def do_GET(self):  # noqa: N802
        if not secrets.compare_digest(
          self.headers.get("X-Metrics-Token", ""), publisher.token
        ):
          self.send_error(403)
          return
        parsed = urlparse(self.path)
        if parsed.path != "/metrics":
          self.send_error(404)
          return
        query = parse_qs(parsed.query)
        try:
          after = max(0, int(query.get("after", ["0"])[0]))
          window = float(query.get("window", ["10"])[0])
          if not math.isfinite(window) or not 1 <= window <= 300:
            raise ValueError("window must be 1–300 seconds")
          body = json.dumps(publisher.snapshot(after, window), allow_nan=False).encode()
        except ValueError as exc:
          self.send_error(400, str(exc))
          return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
          self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
          pass

      def log_message(self, *args):
        pass

    output_dir.mkdir(parents=True, exist_ok=True)
    self.path = output_dir / f".metrics-live-{secrets.token_hex(12)}.json"
    self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self._server.daemon_threads = True
    self._thread = Thread(target=self._server.serve_forever, daemon=True)
    self._thread.start()
    descriptor = {
      "version": 1,
      "port": self._server.server_port,
      "token": self.token,
      "created": time.time(),
    }
    try:
      self.path.touch(mode=0o600, exist_ok=False)
      self.path.write_text(json.dumps(descriptor), encoding="utf-8")
    except Exception:
      self.close()
      raise

  def append(
    self, step: int, env_idx: int, network: str, values: list[float], reset=False
  ):
    if self.closed:
      return
    clean = [v if math.isfinite(v) else None for v in values]
    if len(clean) != len(self.names):
      raise ValueError("Metrics schema changed; reconnect the live feed")
    with self._lock:
      identity = (env_idx, network)
      if reset or identity != self._identity or step < self._last_step:
        self._rows.clear()
        self._epoch += 1
        self._first_step = None
      self._identity = identity
      if reset:
        self._last_step = -1
        return
      if step == self._last_step:
        return
      self._first_step = step if self._first_step is None else self._first_step
      self._last_step = step
      self._last_sample_at = time.time()
      self._sequence += 1
      self._rows.append(
        [self._sequence, step, (step - self._first_step) * self.step_dt, clean]
      )

  def snapshot(self, after: int = 0, window: float = 10) -> dict:
    with self._lock:
      cutoff = self._rows[-1][2] - window if self._rows else 0
      rows = [r for r in self._rows if r[2] >= cutoff and r[0] > after]
      identity = self._identity or (0, "waiting")
      return {
        "version": 1,
        "epoch": self._epoch,
        "sequence": self._sequence,
        "names": self.names,
        "step_dt": self.step_dt,
        "env_idx": identity[0],
        "network": identity[1],
        "last_sample_at": self._last_sample_at,
        "server_time": time.time(),
        "retention_limit_samples": 15_000,
        "rows": rows,
      }

  def close(self):
    if self.closed:
      return
    self.closed = True
    self._server.shutdown()
    self._server.server_close()
    self._thread.join(timeout=2)
    self.path.unlink(missing_ok=True)
