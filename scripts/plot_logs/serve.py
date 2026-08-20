"""Serve plot dashboards and expose repository CSV logs for auto-discovery."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

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
      path for root in roots if root.is_dir() for path in root.rglob("metrics_*.csv")
    ]
  else:
    response_root = logs_root / "response"
    paths = response_root.rglob("response.csv") if response_root.is_dir() else ()

  files: list[dict[str, Any]] = []
  for path in paths:
    if not path.is_file():
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
      }
    )
  files.sort(key=lambda item: (item["modified"], item["path"]), reverse=True)
  return files


class DashboardRequestHandler(SimpleHTTPRequestHandler):
  """Static repository server with a small read-only CSV discovery endpoint."""

  repo_root = REPO_ROOT

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, directory=str(self.repo_root), **kwargs)

  def do_GET(self) -> None:  # noqa: N802
    parsed = urlparse(self.path)
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


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Serve plot dashboards with automatic logs/*.csv discovery."
  )
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8000)
  args = parser.parse_args()

  server = ThreadingHTTPServer((args.host, args.port), DashboardRequestHandler)
  base = f"http://{args.host}:{args.port}/scripts/plot_logs"
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
