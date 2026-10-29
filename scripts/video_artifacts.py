"""Write a videos/artifacts.json for rendered evaluation MP4s.

Combines the review JSON metadata (camera, renderer, checkpoint SHA-256,
velocity overlay) with the MP4 file hash and ffprobe stream facts, so nobody
has to open the video to verify it.

Usage:
  uv run python scripts/video_artifacts.py <videos_dir>
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _probe(path: Path) -> dict:
  try:
    out = subprocess.run(
      [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size",
        "-of",
        "json",
        str(path),
      ],
      capture_output=True,
      text=True,
      check=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    fmt = data["format"]
    return {
      "codec": stream.get("codec_name"),
      "width": stream.get("width"),
      "height": stream.get("height"),
      "fps": stream.get("r_frame_rate"),
      "frames": stream.get("nb_frames"),
      "duration_s": float(fmt.get("duration", 0.0)),
      "size_bytes": int(fmt.get("size", 0)),
    }
  except (subprocess.CalledProcessError, FileNotFoundError, KeyError) as error:
    return {"probe_error": str(error)}


def main() -> None:
  if len(sys.argv) != 2:
    raise SystemExit(__doc__)
  videos = Path(sys.argv[1])
  mp4s = sorted(videos.glob("*.mp4"))
  if not mp4s:
    raise SystemExit(f"no mp4 in {videos}")
  entries = []
  for video in mp4s:
    meta_path = video.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    entries.append(
      {
        "path": str(video.resolve()),
        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "view": meta.get("view"),
        "task": meta.get("task"),
        "checkpoint": meta.get("checkpoint"),
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "renderer": meta.get("renderer"),
        "camera": meta.get("camera"),
        "command": meta.get("command"),
        "velocity_overlay": meta.get("velocity_overlay"),
        "seconds": meta.get("seconds"),
        "num_envs": meta.get("num_envs"),
        **_probe(video),
      }
    )
  out = videos / "artifacts.json"
  out.write_text(json.dumps({"videos": entries}, indent=2) + "\n")
  print(f"wrote {out} ({len(entries)} videos)")
  for entry in entries:
    print(
      f"  {Path(entry['path']).name}: {entry.get('width')}x{entry.get('height')} "
      f"{entry.get('codec')} {entry.get('duration_s')}s "
      f"{entry.get('size_bytes')}B sha={entry['sha256'][:16]}"
    )


if __name__ == "__main__":
  main()
