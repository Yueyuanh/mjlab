"""Serve plot dashboards and expose repository CSV logs for auto-discovery."""

from __future__ import annotations

import argparse
import errno
import json
import math
import re
import shutil
import sys
from http import HTTPStatus
from http.client import HTTPException
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import ProxyHandler, Request, build_opener

REPO_ROOT = Path(__file__).resolve().parents[2]


def discover_csv_files(repo_root: Path, kind: str) -> list[dict[str, Any]]:
  """Return newest-first dashboard CSV metadata below ``repo_root`` roots.

  Metrics are scanned from ``logs/`` (live runs) plus ``pre_train/`` (saved
  runs e.g. ``pre_train/rsl_rl/<task>/<run>/metrics_*.csv``) and
  ``src/mjlab/mocap/logs`` (mocap PD-recording metrics logs).
  """
  logs_root = repo_root / "logs"
  pre_train_root = repo_root / "pre_train"
  mocap_root = repo_root / "src" / "mjlab" / "mocap" / "logs"
  if kind not in {"metrics", "response"}:
    raise ValueError(f"Unsupported CSV kind: {kind}")

  if kind == "metrics":
    roots = {logs_root, pre_train_root, mocap_root}
    paths = [
      path for root in roots if root.is_dir() for path in root.rglob("*metrics_*.csv")
    ]
  else:
    response_root = logs_root / "response"
    paths = response_root.rglob("response.csv") if response_root.is_dir() else ()

  files: list[dict[str, Any]] = []
  for path in paths:
    if not path.is_file():
      continue
    if not path.resolve().is_relative_to(repo_root.resolve()):
      continue
    stat = path.stat()
    relative = path.relative_to(repo_root)
    files.append(
      {
        "name": path.name,
        "path": relative.as_posix(),
        "url": "/" + quote(relative.as_posix(), safe="/"),
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "experiment": path.parent.name if path.parent.parent.name != "rsl_rl" else "",
        "video_url": "/"
        + quote(path.with_suffix(".mp4").relative_to(repo_root).as_posix(), safe="/")
        if path.with_suffix(".mp4").is_file()
        else None,
        "manifest_url": "/"
        + quote(
          path.with_suffix(".recording.json").relative_to(repo_root).as_posix(),
          safe="/",
        )
        if path.with_suffix(".recording.json").is_file()
        else None,
      }
    )
  files.sort(key=lambda item: (item["modified"], item["path"]), reverse=True)
  return files


def live_descriptor(repo_root: Path, source: str) -> tuple[Path, dict]:
  path = (repo_root / source).resolve()
  roots = [
    repo_root / "logs",
    repo_root / "pre_train",
    repo_root / "src/mjlab/mocap/logs",
  ]
  if not any(path.is_relative_to(root.resolve()) for root in roots):
    raise ValueError("Invalid live source path")
  if not re.fullmatch(r"\.metrics-live-[a-f0-9]{24}\.json", path.name):
    raise ValueError("Invalid live descriptor")
  data = json.loads(path.read_text(encoding="utf-8"))
  if (
    data.get("version") != 1
    or not isinstance(data.get("port"), int)
    or not 1 <= data["port"] <= 65535
  ):
    raise ValueError("Invalid live descriptor version/port")
  if not isinstance(data.get("token"), str) or not re.fullmatch(
    r"[\w-]{32,100}", data["token"]
  ):
    raise ValueError("Invalid live token")
  return path, data


def read_live_metrics(repo_root: Path, source: str, after: int, window: float) -> dict:
  _, descriptor = live_descriptor(repo_root, source)
  if not math.isfinite(window) or not 1 <= window <= 300 or after < 0:
    raise ValueError("Invalid cursor/window; window must be 1–300 seconds")
  request = Request(
    f"http://127.0.0.1:{descriptor['port']}/metrics?after={int(after)}&window={window}",
    headers={"X-Metrics-Token": descriptor["token"]},
  )
  # Never use HTTP proxy environment settings for the loopback-only producer.
  with build_opener(ProxyHandler({})).open(request, timeout=2) as response:
    return json.load(response)


def discover_live_sources(repo_root: Path) -> list[dict]:
  sources = []
  for root in (
    repo_root / "logs",
    repo_root / "pre_train",
    repo_root / "src/mjlab/mocap/logs",
  ):
    if not root.is_dir():
      continue
    for path in root.rglob(".metrics-live-*.json"):
      relative = path.relative_to(repo_root).as_posix()
      try:
        data = read_live_metrics(repo_root, relative, 2**63 - 1, 1)
      except (OSError, ValueError):
        continue
      sources.append(
        {
          "id": relative,
          "network": data["network"],
          "env_idx": data["env_idx"],
          "last_sample_at": data["last_sample_at"],
          "step_dt": data["step_dt"],
        }
      )
  return sorted(sources, key=lambda source: source["last_sample_at"], reverse=True)


class DashboardRequestHandler(SimpleHTTPRequestHandler):
  """Static repository server with a small read-only CSV discovery endpoint."""

  repo_root = REPO_ROOT

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, directory=str(self.repo_root), **kwargs)

  def do_GET(self) -> None:  # noqa: N802
    parsed = urlparse(self.path)
    if Path(parsed.path).name.startswith(".metrics-live-"):
      self.send_error(HTTPStatus.FORBIDDEN)
      return
    if parsed.path in ("/api/live-sources", "/api/live-metrics"):
      query = parse_qs(parsed.query)
      try:
        if parsed.path == "/api/live-sources":
          payload = {"sources": discover_live_sources(self.repo_root)}
        else:
          payload = read_live_metrics(
            self.repo_root,
            query.get("source", [""])[0],
            int(query.get("after", ["0"])[0]),
            float(query.get("window", ["10"])[0]),
          )
        self._write_json(payload)
      except ValueError as exc:
        self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
      except OSError as exc:
        self._write_json(
          {"error": "实时源已停止或不可达：" + str(exc)},
          status=HTTPStatus.SERVICE_UNAVAILABLE,
        )
      return
    if parsed.path != "/api/csv-files":
      super().do_GET()
      return

    kind = parse_qs(parsed.query).get("kind", [""])[0]
    try:
      files = discover_csv_files(self.repo_root, kind)
    except ValueError as error:
      self._write_json(
        {"error": str(error)},
        status=HTTPStatus.BAD_REQUEST,
      )
      return
    self._write_json({"kind": kind, "files": files})

  def send_head(self):
    self._range_remaining = None
    path = Path(self.translate_path(urlparse(self.path).path))
    if path.name.startswith(".metrics-live-"):
      self.send_error(HTTPStatus.FORBIDDEN)
      return None
    if path.suffix.lower() != ".mp4" or not self.headers.get("Range"):
      return super().send_head()
    if (
      not path.resolve().is_relative_to(self.repo_root.resolve()) or not path.is_file()
    ):
      self.send_error(HTTPStatus.NOT_FOUND)
      return None
    size = path.stat().st_size
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers["Range"])
    if not match or not any(match.groups()):
      self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
      return None
    a, b = match.groups()
    start = int(a) if a else max(0, size - int(b))
    end = min(size - 1, int(b)) if a and b else size - 1
    if start > end or start >= size:
      self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
      self.send_header("Content-Range", f"bytes */{size}")
      self.send_header("Content-Length", "0")
      self.end_headers()
      return None
    stream = path.open("rb")
    stream.seek(start)
    self._range_remaining = end - start + 1
    self.send_response(HTTPStatus.PARTIAL_CONTENT)
    self.send_header("Content-Type", "video/mp4")
    self.send_header("Accept-Ranges", "bytes")
    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    self.send_header("Content-Length", str(end - start + 1))
    self.end_headers()
    return stream

  def copyfile(self, source, outputfile):
    remaining = self._range_remaining
    if remaining is None:
      shutil.copyfileobj(source, outputfile)
      return
    while remaining > 0:
      data = source.read(min(64 * 1024, remaining))
      if not data:
        break
      outputfile.write(data)
      remaining -= len(data)

  def _write_json(
    self,
    payload: dict[str, Any],
    *,
    status: HTTPStatus = HTTPStatus.OK,
  ) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.end_headers()
    self.wfile.write(body)


def display_host(host: str) -> str:
  """Return a browser-usable host for the URLs printed on startup."""
  if host in {"", "0.0.0.0"}:
    return "127.0.0.1"
  if host == "::":
    return "[::1]"
  return f"[{host}]" if ":" in host else host


def probe_dashboard(host: str, port: int) -> bool:
  """Return True when ``port`` already answers with this dashboard's API."""
  probe_host = "127.0.0.1" if host in {"", "0.0.0.0"} else host
  request = Request(f"http://{probe_host}:{port}/api/csv-files?kind=metrics")
  try:
    # Never use HTTP proxy environment settings for the loopback-only probe.
    with build_opener(ProxyHandler({})).open(request, timeout=1) as response:
      payload = json.load(response)
  except (OSError, ValueError, HTTPException):
    return False
  return (
    isinstance(payload, dict)
    and payload.get("kind") == "metrics"
    and isinstance(payload.get("files"), list)
  )


def busy_port_reason(host: str, port: int) -> str:
  """Describe who holds ``port``, for the startup messages."""
  if probe_dashboard(host, port):
    return "已有一个 dashboard 服务在运行"
  return "被其他程序占用"


def bind_dashboard_server(
  host: str,
  port: int,
  handler: type[SimpleHTTPRequestHandler],
  *,
  max_tries: int = 20,
  strict: bool = False,
) -> ThreadingHTTPServer:
  """Bind the dashboard server, walking forward while ``port`` stays busy.

  A busy port is the normal case when an earlier ``serve.py`` is still running,
  so the default is to keep trying ``port + 1, port + 2, …`` instead of dying
  with ``Address already in use``. ``port=0`` asks the OS for any free port and
  ``strict`` restores the old fail-fast behaviour.
  """
  candidates = range(port, port + max_tries) if port else [0]
  for candidate in candidates:
    try:
      return ThreadingHTTPServer((host, candidate), handler)
    except OSError as error:
      if error.errno not in {errno.EADDRINUSE, errno.EACCES}:
        raise
      if strict:
        raise OSError(
          error.errno,
          f"端口 {candidate} {busy_port_reason(host, candidate)}（--strict-port）",
        ) from error
      if candidate == port:
        print(
          f"端口 {candidate} {busy_port_reason(host, candidate)}，自动向后寻找空闲端口……"
        )
  raise OSError(
    errno.EADDRINUSE,
    f"端口 {port}–{port + max_tries - 1} 均不可用；请释放端口或用 --port 0 让系统分配",
  )


def port_arg(value: str) -> int:
  try:
    port = int(value)
  except ValueError as error:
    raise argparse.ArgumentTypeError(f"端口必须是整数：{value!r}") from error
  if not 0 <= port <= 65535:
    raise argparse.ArgumentTypeError(f"端口必须在 0–65535 之间：{value!r}")
  return port


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Serve plot dashboards with automatic logs/*.csv discovery."
  )
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument(
    "--port",
    type=port_arg,
    default=8000,
    help="监听端口（默认 8000）；被占用时自动改用后续空闲端口，0 表示由系统分配。",
  )
  parser.add_argument(
    "--strict-port",
    action="store_true",
    help="端口被占用时直接报错退出，不自动换端口。",
  )
  args = parser.parse_args()

  try:
    server = bind_dashboard_server(
      args.host,
      args.port,
      DashboardRequestHandler,
      strict=args.strict_port,
    )
  except OSError as error:
    print(f"无法启动 dashboard 服务：{error}", file=sys.stderr)
    raise SystemExit(1) from error

  port = server.server_address[1]
  base = f"http://{display_host(args.host)}:{port}/scripts/plot_logs"
  print(f"Metrics dashboard:  {base}/metrics_csv_dashboard.html")
  print(f"Response dashboard: {base}/response_csv_dashboard.html")
  print("CSV discovery root: " + str(REPO_ROOT / "logs"))
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    server.server_close()


if __name__ == "__main__":
  main()
