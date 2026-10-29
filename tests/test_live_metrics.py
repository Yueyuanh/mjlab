"""Bounded live transport, read-only proxy and video HTTP range support."""

import json
import runpy
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from mjlab.viewer.viser.live_metrics import MetricsLivePublisher

SERVER = runpy.run_path("scripts/plot_logs/serve.py")


@pytest.fixture
def live(tmp_path):
  publisher = MetricsLivePublisher(tmp_path / "logs/run", ["metric"], 0.02)
  try:
    yield publisher
  finally:
    publisher.close()


def test_live_buffer_window_delta_and_identity(live):
  for i in range(601):
    live.append(i, 0, "modelA", [float(i)])
  data = live.snapshot(window=10)
  assert data["rows"][0][2] >= 2
  assert len(data["rows"]) <= 501
  assert live.snapshot(after=600)["rows"][0][0] == 601
  live.append(601, 1, "modelB", [float("nan")])
  changed = live.snapshot()
  assert changed["epoch"] > data["epoch"]
  assert changed["network"] == "modelB" and changed["env_idx"] == 1
  assert len(changed["rows"]) == 1 and changed["rows"][0][3] == [None]
  live.append(602, 1, "modelB", [1], reset=True)
  assert live.snapshot()["rows"] == []
  live.append(603, 1, "modelB", [1])
  assert live.snapshot()["rows"][0][2] == 0


def test_live_buffer_has_hard_sample_cap(live):
  for i in range(15_500):
    live.append(i, 0, "m", [1])
  assert len(live._rows) == 15_000


def test_live_proxy_and_token_guards(live, tmp_path):
  live.append(1, 0, "modelA", [3])
  source = live.path.relative_to(tmp_path).as_posix()
  result = SERVER["read_live_metrics"](tmp_path, source, 0, 10)
  assert result["rows"][0][3] == [3]
  assert "token" not in result
  sources = SERVER["discover_live_sources"](tmp_path)
  assert sources[0]["id"] == source and "token" not in sources[0]
  descriptor = json.loads(live.path.read_text())
  with pytest.raises(HTTPError) as exc:
    urlopen(f"http://127.0.0.1:{descriptor['port']}/metrics")
  assert exc.value.code == 403
  for path in ("../private.json", "logs/run/other.json"):
    with pytest.raises(ValueError):
      SERVER["read_live_metrics"](tmp_path, path, 0, 10)
  with pytest.raises(ValueError):
    SERVER["read_live_metrics"](tmp_path, source, 0, float("nan"))


def test_http_video_ranges_and_private_descriptor(live, tmp_path):
  video = tmp_path / "logs/run/experiment/video.mp4"
  video.parent.mkdir()
  video.write_bytes(bytes(range(256)) * 4)
  handler = type(
    "Handler", (SERVER["DashboardRequestHandler"],), {"repo_root": tmp_path}
  )
  server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
  worker = Thread(target=server.serve_forever, daemon=True)
  worker.start()
  base = f"http://127.0.0.1:{server.server_port}"
  try:
    with urlopen(
      Request(base + "/logs/run/experiment/video.mp4", headers={"Range": "bytes=20-29"})
    ) as response:
      assert response.status == 206
      assert response.headers["Content-Range"] == "bytes 20-29/1024"
      assert response.read() == bytes(range(20, 30))
    with urlopen(
      Request(base + "/logs/run/experiment/video.mp4", headers={"Range": "bytes=-10"})
    ) as response:
      assert len(response.read()) == 10
    with pytest.raises(HTTPError) as exc:
      urlopen(
        Request(
          base + "/logs/run/experiment/video.mp4", headers={"Range": "bytes=5000-"}
        )
      )
    assert exc.value.code == 416
    with pytest.raises(HTTPError) as exc:
      urlopen(base + "/" + live.path.relative_to(tmp_path).as_posix())
    assert exc.value.code == 403
    live.append(1, 0, "m", [1])
    query = urlencode({"source": live.path.relative_to(tmp_path).as_posix()})
    with urlopen(base + "/api/live-metrics?" + query) as response:
      assert json.load(response)["rows"][0][3] == [1]
  finally:
    server.shutdown()
    server.server_close()
    worker.join()
