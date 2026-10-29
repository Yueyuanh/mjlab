"""Join all review clips in order with provenance labels, retaining every frame."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from v033_artifacts import experiment_runs, review_dir


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--experiment", choices=("biped", "crawl"), required=True)
  args = parser.parse_args()
  experiment = experiment_runs()[args.experiment]
  folder = review_dir(experiment, "videos")
  metrics = review_dir(experiment, "metrics")
  output = folder / "review_all.mp4"
  font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
  manifest = []
  frames = 0
  writer = imageio.get_writer(
    output, fps=50, codec="libx264", quality=8, ffmpeg_params=["-threads", "2"]
  )
  try:
    for clip in ("stand", "forward", "backward", "turn"):
      for view in ("side", "front"):
        source = folder / f"{clip}_{view}.mp4"
        info = json.loads((metrics / f"{source.stem}.json").read_text())
        start = frames / 50
        command = info["command"]
        title = (
          f"{args.experiment.upper()} | {clip.upper()} | {view.upper()} | "
          f"vx={command[0]:.2f} vy={command[1]:.2f} m/s  wz={command[2]:.2f} rad/s"
        )
        reader = imageio.get_reader(source, input_params=["-threads", "2"])
        count, _ = imageio_ffmpeg.count_frames_and_secs(str(source))
        assert count is not None
        try:
          for index in range(count):
            frame = reader.get_data(index)
            picture = Image.fromarray(frame)
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, 1280, 68), fill="black")
            draw.text((18, 12), title, font=font, fill="white")
            draw.text(
              (18, 40),
              f"Final {Path(info['checkpoint']).stem} | deterministic policy | no cuts or speed changes",
              font=font,
              fill="#cccccc",
            )
            writer.append_data(np.asarray(picture))
            frames += 1
        finally:
          reader.close()
        manifest.append(
          {
            "source": str(source),
            "start_s": start,
            "end_s": frames / 50,
            "checkpoint": info["checkpoint"],
            "command": command,
          }
        )
  finally:
    writer.close()
  (metrics / "review_all.json").write_text(
    json.dumps(
      {
        "frames": frames,
        "fps": 50,
        "seconds": frames / 50,
        "note": "Every original frame retained; text labels only; use source clips/CSV for analysis.",
        "segments": manifest,
      },
      indent=2,
    )
    + "\n"
  )
  print(output)


if __name__ == "__main__":
  main()
