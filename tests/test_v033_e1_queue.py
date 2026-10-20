"""A failed first job must not silently cancel the user's second experiment."""

import hashlib
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_queue_preserves_both_training_failures_and_refuses_overwrite(
  tmp_path, monkeypatch
):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  queue = importlib.import_module("run_v033_e1_queue")
  child = MagicMock()
  child.pid = 123
  child.returncode = 7
  child.poll.return_value = 7
  launch = MagicMock(return_value=child)
  monkeypatch.setattr(queue.subprocess, "Popen", launch)
  audit = tmp_path / "audit"
  audit.mkdir()
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "queue",
      "--audit",
      str(audit),
      "--source",
      str(tmp_path),
      "--root",
      str(tmp_path),
    ],
  )
  queue.main()
  state = json.loads((audit / "status.json").read_text())
  assert launch.call_count == 2
  assert state["status"] == "finished_with_errors"
  assert all(
    experiment["status"] == "training_failed"
    for experiment in state["experiments"].values()
  )
  commands = [
    json.loads(line)["argv"]
    for line in (audit / "commands.jsonl").read_text().splitlines()
  ]
  assert all(command[command.index("--num-envs") + 1] == "4096" for command in commands)
  assert all(
    command[command.index("--iterations") + 1] == "20000" for command in commands
  )
  previous = (audit / "status.json").read_bytes()
  try:
    queue.main()
  except FileExistsError:
    pass
  else:
    raise AssertionError("Existing audit must not be overwritten")
  assert (audit / "status.json").read_bytes() == previous


@pytest.mark.parametrize("fail_review", (False, True))
def test_queue_verifies_artifacts_and_continues_after_review_error(
  tmp_path, monkeypatch, fail_review
):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  queue = importlib.import_module("run_v033_e1_queue")
  launched = []

  def launch(command, **kwargs):
    launched.append(command)
    out = Path(command[command.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    task = command[command.index("--task") + 1]
    child = MagicMock(pid=123, returncode=0)
    if "--iterations" in command:
      checkpoint = out / "model_19999.pt"
      checkpoint.write_bytes(b"test-checkpoint")
      (out / "complete.json").write_text(
        json.dumps({"sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
      )
    else:
      clip = command[command.index("--clip") + 1]
      if fail_review and "Crawl" not in task and clip == "turn":
        child.returncode = 9
      elif "--video" in command:
        view = command[command.index("--view") + 1]
        (out / f"{clip}_{view}.mp4").write_bytes(b"test-video")
      else:
        (out / f"{clip}_metrics.json").write_text(
          json.dumps(
            {
              "task": task,
              "checkpoint": command[command.index("--checkpoint") + 1],
              "command": [0, 0, 0],
              "actual_mean": [0, 0, 0],
              "mae": [0, 0, 0],
              "falls_total": 0,
              "survival_fraction": 1,
            }
          )
        )
    child.poll.return_value = child.returncode
    return child

  monkeypatch.setattr(queue.subprocess, "Popen", launch)
  monkeypatch.setattr(
    queue.subprocess,
    "check_output",
    lambda *a, **kw: json.dumps({"format": {"duration": "8.0"}}),
  )
  audit = tmp_path / "audit"
  audit.mkdir()
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "queue",
      "--audit",
      str(audit),
      "--source",
      str(tmp_path),
      "--root",
      str(tmp_path),
    ],
  )
  queue.main()
  state = json.loads((audit / "status.json").read_text())
  assert len(launched) == 26  # two training jobs plus 24 nominal review jobs
  assert state["experiments"]["crawl_e1"]["status"] == "complete"
  assert state["status"] == ("finished_with_errors" if fail_review else "complete")
  for name, experiment in state["experiments"].items():
    manifest = json.loads(
      (Path(experiment["run"]) / "review/video_manifest.json").read_text()
    )
    assert len(manifest) == (6 if fail_review and name == "biped_e1" else 8)
