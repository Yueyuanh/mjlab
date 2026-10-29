"""Export actual TensorBoard curves and CSV without touching training state."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from v033_artifacts import AUDIT, experiment_runs, training_dir

LOGS = AUDIT / "reports/figures"


def main():
  LOGS.mkdir(parents=True, exist_ok=True)
  figure, axes = plt.subplots(2, 2, figsize=(12, 7))
  categories = (
    ("Train/mean_reward", "Mean training reward"),
    ("Train/mean_episode_length", "Episode length (policy steps)"),
    ("Loss/value", "Value loss"),
    ("Policy/mean_std", "Action noise std"),
  )
  rows = []
  for label, folder in experiment_runs().items():
    acc = EventAccumulator(
      str(training_dir(folder, "tensorboard")), size_guidance={"scalars": 0}
    )
    acc.Reload()
    tags = acc.Tags()["scalars"]
    for axis, (tag, title) in zip(axes.flat, categories, strict=True):
      if tag in tags:
        events = acc.Scalars(tag)
        axis.plot(
          [e.step for e in events],
          [e.value for e in events],
          label=label,
          linewidth=0.8,
        )
        rows.extend((label, tag, e.step, e.wall_time, e.value) for e in events)
      axis.set_title(title)
      axis.set_xlabel("iteration")
      axis.grid(alpha=0.2)
  for axis in axes.flat:
    if axis.lines:
      axis.legend()
  figure.tight_layout()
  figure.savefig(LOGS / "training_curves.png", dpi=150)
  plt.close(figure)
  with (LOGS / "training_curves.csv").open("w") as stream:
    writer = csv.writer(stream)
    writer.writerow(("experiment", "tag", "iteration", "wall_time", "value"))
    writer.writerows(rows)


if __name__ == "__main__":
  main()
