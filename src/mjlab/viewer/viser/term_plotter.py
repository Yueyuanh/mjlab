"""Plotting functionality for Viser viewer."""

from __future__ import annotations

import csv
import io
import re
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, TextIO

import numpy as np
import viser
import viser.uplot

_PALETTE = [
  "#1f77b4",  # blue
  "#ff7f0e",  # orange
  "#2ca02c",  # green
  "#d62728",  # red
  "#9467bd",  # purple
  "#8c564b",  # brown
  "#e377c2",  # pink
  "#7f7f7f",  # gray
  "#bcbd22",  # olive
  "#17becf",  # cyan
  "#aec7e8",  # light blue
  "#ffbb78",  # light orange
]

_JOINT_SIGNAL_PREFIXES = {
  "Position": "joint_position_rad/",
  "Velocity": "joint_velocity_rad_s/",
  "Torque": "joint_torque_nm/",
}

_CSV_METADATA_COLUMNS = frozenset(
  {"sample", "sim_step", "time_s", "env_idx", "network"}
)
_MAX_CSV_ROWS = 200_000
_IMAGE_WIDTH = 1920
_IMAGE_HEIGHT = 1080
_PAIR_IMAGE_HEIGHT = 3780
_IMAGE_DPI = 120

_JOINT_IMAGE_GROUPS = {
  "joint_position": ("joint_position_rad/", "Joint position (rad)"),
  "joint_velocity": ("joint_velocity_rad_s/", "Joint velocity (rad/s)"),
  "joint_torque": ("joint_torque_nm/", "Actuator torque (N·m)"),
}
_JOINT_PAIR_ORDER = (
  "hip_pitch",
  "hip_roll",
  "hip_yaw",
  "knee",
  "ankle",
  "arm_pitch",
  "arm_roll",
)


def _color_for(index: int) -> str:
  return _PALETTE[index % len(_PALETTE)]


def _safe_filename_component(value: str) -> str:
  """Return a compact filesystem-safe name for a task/checkpoint label."""
  value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
  return value[:120] or "unknown-network"


@dataclass(frozen=True)
class LoadedMetricsCsv:
  """Validated metric curves loaded from a recorded CSV file."""

  time_s: np.ndarray
  values: dict[str, np.ndarray]
  network: str | None


def load_metrics_csv(
  content: bytes,
  available_terms: Sequence[str],
) -> LoadedMetricsCsv:
  """Parse a recorded metrics CSV and retain terms supported by this task."""
  try:
    text = content.decode("utf-8-sig")
  except UnicodeDecodeError as exc:
    raise ValueError("CSV must be UTF-8 encoded") from exc

  reader = csv.DictReader(io.StringIO(text))
  if reader.fieldnames is None:
    raise ValueError("CSV has no header")
  if "time_s" not in reader.fieldnames:
    raise ValueError("CSV is missing the required time_s column")

  available = set(available_terms)
  metric_names = [
    name
    for name in reader.fieldnames
    if name not in _CSV_METADATA_COLUMNS and name in available
  ]
  if not metric_names:
    raise ValueError("CSV has no metric columns available in the current task")

  time_s: list[float] = []
  values = {name: [] for name in metric_names}
  network: str | None = None
  for row_index, row in enumerate(reader, start=2):
    if len(time_s) >= _MAX_CSV_ROWS:
      raise ValueError(f"CSV exceeds the {_MAX_CSV_ROWS:,}-row display limit")
    try:
      time_s.append(float(row["time_s"] or "nan"))
      for name in metric_names:
        values[name].append(float(row[name] or "nan"))
    except (TypeError, ValueError) as exc:
      raise ValueError(f"CSV contains a non-numeric value on row {row_index}") from exc
    if network is None and row.get("network"):
      network = row["network"]

  if not time_s:
    raise ValueError("CSV has no data rows")
  time_array = np.asarray(time_s, dtype=np.float64)
  if not np.all(np.isfinite(time_array)):
    raise ValueError("CSV time_s values must be finite")
  if np.any(np.diff(time_array) < 0):
    raise ValueError("CSV time_s values must be non-decreasing")

  return LoadedMetricsCsv(
    time_s=time_array,
    values={
      name: np.asarray(term_values, dtype=np.float64)
      for name, term_values in values.items()
    },
    network=network,
  )


def _metric_axis_label(term_name: str) -> str:
  """Return a human-readable y-axis label from a telemetry term name."""
  labels = {
    "joint_position_rad/": "Joint position (rad)",
    "joint_velocity_rad_s/": "Joint velocity (rad/s)",
    "joint_torque_nm/": "Actuator torque (N·m)",
  }
  for prefix, label in labels.items():
    if term_name.startswith(prefix):
      return label
  if term_name == "base_height_m":
    return "Base height (m)"
  return "Value"


def _joint_image_parts(term_name: str) -> tuple[str, str, str] | None:
  """Return ``(signal group, side, joint key)`` for a joint metric term."""
  for group, (prefix, _) in _JOINT_IMAGE_GROUPS.items():
    if not term_name.startswith(prefix):
      continue
    joint_name = term_name.removeprefix(prefix)
    for side in ("left", "right"):
      side_prefix = f"{side}_"
      if joint_name.startswith(side_prefix) and joint_name.endswith("_joint"):
        return group, side, joint_name.removeprefix(side_prefix).removesuffix(
          "_joint"
        )
  return None


def save_metrics_curve_images(
  loaded: LoadedMetricsCsv,
  term_names: Sequence[str],
  csv_parent: Path,
  csv_name: str,
  colors: dict[str, str] | None = None,
) -> list[Path]:
  """Save selected CSV curves as structured individual and comparison PNGs.

  Joint signals are grouped by position/velocity/torque and by left/right side.
  Each group also gets all-joint, per-side, and fixed 1x7 left/right comparison
  figures. Images are rooted at ``<csv_parent>/images/<csv_stem>/``.
  """
  selected = tuple(
    dict.fromkeys(name for name in term_names if name in loaded.values)
  )
  if not selected:
    raise ValueError("Select at least one loaded metric before saving images")

  # Lazy imports keep normal viewer startup light and use the headless Agg canvas.
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from matplotlib.figure import Figure

  csv_stem = _safe_filename_component(Path(csv_name).stem)
  image_dir = csv_parent / "images" / csv_stem
  image_dir.mkdir(parents=True, exist_ok=True)
  output_paths: list[Path] = []

  def _new_figure(height: int = _IMAGE_HEIGHT):
    figure = Figure(
      figsize=(_IMAGE_WIDTH / _IMAGE_DPI, height / _IMAGE_DPI),
      dpi=_IMAGE_DPI,
      layout="constrained",
    )
    FigureCanvasAgg(figure)
    return figure

  def _save_figure(figure, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
      output_path,
      format="png",
      dpi=_IMAGE_DPI,
      facecolor="white",
      metadata={"Title": title, "Software": "mjlab Viser Metrics"},
    )
    figure.clear()
    output_paths.append(output_path)

  def _stats(term_name: str) -> str:
    finite_values = loaded.values[term_name][np.isfinite(loaded.values[term_name])]
    if not finite_values.size:
      return "No finite samples"
    return (
      f"min={np.min(finite_values):.5g}   "
      f"max={np.max(finite_values):.5g}   "
      f"peak |value|={np.max(np.abs(finite_values)):.5g}"
    )

  grouped_terms: dict[str, dict[str, dict[str, str]]] = {}
  for term_name in selected:
    values = loaded.values[term_name]
    joint_parts = _joint_image_parts(term_name)
    if joint_parts is None:
      output_path = image_dir / "other" / f"{_safe_filename_component(term_name)}.png"
    else:
      group, side, joint_key = joint_parts
      grouped_terms.setdefault(group, {}).setdefault(side, {})[joint_key] = term_name
      output_path = image_dir / group / side / f"{joint_key}.png"

    figure = _new_figure()
    axis = figure.subplots()
    axis.plot(
      loaded.time_s,
      values,
      color=(colors or {}).get(term_name, "#1f77b4"),
      linewidth=1.8,
    )
    axis.set_title(term_name, fontsize=18, pad=18)
    axis.set_xlabel("Time (s)", fontsize=14)
    axis.set_ylabel(_metric_axis_label(term_name), fontsize=14)
    axis.grid(visible=True, alpha=0.3)
    axis.tick_params(labelsize=11)
    subtitle = f"Network: {loaded.network or 'unknown'}\n{_stats(term_name)}"
    axis.text(
      0.01,
      0.99,
      subtitle,
      transform=axis.transAxes,
      verticalalignment="top",
      fontsize=11,
      bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    _save_figure(figure, output_path, term_name)

  for group, sides in grouped_terms.items():
    _, y_label = _JOINT_IMAGE_GROUPS[group]
    group_dir = image_dir / group

    # One overview per side keeps seven similarly-oriented joints easy to scan.
    for side in ("left", "right"):
      side_terms = sides.get(side, {})
      if not side_terms:
        continue
      figure = _new_figure()
      axis = figure.subplots()
      for joint_key in _JOINT_PAIR_ORDER:
        term_name = side_terms.get(joint_key)
        if term_name is None:
          continue
        axis.plot(
          loaded.time_s,
          loaded.values[term_name],
          label=joint_key,
          linewidth=1.5,
        )
      axis.set_title(f"{group}: {side} joints", fontsize=18, pad=18)
      axis.set_xlabel("Time (s)", fontsize=14)
      axis.set_ylabel(y_label, fontsize=14)
      axis.grid(visible=True, alpha=0.3)
      axis.legend(ncols=2, fontsize=11)
      _save_figure(
        figure,
        group_dir / side / f"{side}_joints_comparison.png",
        f"{group}: {side} joints",
      )

    # All selected joints of one signal type overlaid in one 1080p figure.
    figure = _new_figure()
    axis = figure.subplots()
    for side, linestyle in (("left", "-"), ("right", "--")):
      for joint_key in _JOINT_PAIR_ORDER:
        term_name = sides.get(side, {}).get(joint_key)
        if term_name is None:
          continue
        axis.plot(
          loaded.time_s,
          loaded.values[term_name],
          label=f"{side[0].upper()} {joint_key}",
          linewidth=1.35,
          linestyle=linestyle,
        )
    axis.set_title(f"{group}: all selected joints", fontsize=18, pad=18)
    axis.set_xlabel("Time (s)", fontsize=14)
    axis.set_ylabel(y_label, fontsize=14)
    axis.grid(visible=True, alpha=0.3)
    axis.legend(ncols=3, fontsize=10)
    _save_figure(
      figure,
      group_dir / "all_joints_comparison.png",
      f"{group}: all selected joints",
    )

    # Fixed 1x7 layout: each row compares the corresponding left/right joint.
    figure = _new_figure(height=_PAIR_IMAGE_HEIGHT)
    axes = figure.subplots(len(_JOINT_PAIR_ORDER), 1, sharex=True)
    figure.suptitle(
      f"{group}: left/right joint comparison\nNetwork: {loaded.network or 'unknown'}",
      fontsize=20,
    )
    for row, joint_key in enumerate(_JOINT_PAIR_ORDER):
      axis = axes[row]
      available_sides = 0
      for side, color in (("left", "#1f77b4"), ("right", "#d62728")):
        term_name = sides.get(side, {}).get(joint_key)
        if term_name is None:
          continue
        available_sides += 1
        finite_values = loaded.values[term_name][
          np.isfinite(loaded.values[term_name])
        ]
        peak = np.max(np.abs(finite_values)) if finite_values.size else np.nan
        axis.plot(
          loaded.time_s,
          loaded.values[term_name],
          label=f"{side.capitalize()} (peak={peak:.5g})",
          color=color,
          linewidth=1.6,
        )
      axis.set_title(joint_key, fontsize=14)
      axis.set_ylabel(y_label, fontsize=10)
      axis.grid(visible=True, alpha=0.3)
      if available_sides:
        axis.legend(loc="upper right", fontsize=10, ncols=2)
      else:
        axis.text(
          0.5,
          0.5,
          "Not selected / unavailable",
          transform=axis.transAxes,
          ha="center",
          va="center",
          color="#7f7f7f",
        )
    axes[-1].set_xlabel("Time (s)", fontsize=14)
    _save_figure(
      figure,
      group_dir / "left_right_comparison_1x7.png",
      f"{group}: left/right 1x7 comparison",
    )
  return output_paths


class MetricsCsvRecorder:
  """Thread-safe writer for the metric terms selected in the Viser UI."""

  def __init__(
    self,
    output_dir: Path,
    source_name: Callable[[], str],
    step_dt: float,
  ) -> None:
    self.output_dir = output_dir
    self._source_name = source_name
    self._step_dt = step_dt
    self._lock = Lock()
    self._file: TextIO | None = None
    self._writer: Any | None = None
    self._term_names: tuple[str, ...] = ()
    self._path: Path | None = None
    self._sample = 0
    self._first_sim_step: int | None = None

  @property
  def active(self) -> bool:
    """Whether a recording file is currently open."""
    return self._file is not None

  @property
  def path(self) -> Path | None:
    """Path of the active or most recently completed recording."""
    return self._path

  def start(self, term_names: Sequence[str]) -> Path:
    """Start a new recording for an immutable snapshot of selected terms."""
    selected = tuple(dict.fromkeys(term_names))
    if not selected:
      raise ValueError("Select at least one metric before recording")
    self.stop()
    self.output_dir.mkdir(parents=True, exist_ok=True)

    source = self._source_name()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"metrics_{_safe_filename_component(source)}_{stamp}"
    path = self.output_dir / f"{stem}.csv"
    suffix = 2
    while path.exists():
      path = self.output_dir / f"{stem}_{suffix}.csv"
      suffix += 1

    record_file = path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(record_file)
    writer.writerow(
      ["sample", "sim_step", "time_s", "env_idx", "network", *selected]
    )
    record_file.flush()
    with self._lock:
      self._file = record_file
      self._writer = writer
      self._term_names = selected
      self._path = path
      self._sample = 0
      self._first_sim_step = None
    return path

  def append(
    self,
    values: dict[str, float],
    env_idx: int,
    sim_step: int,
  ) -> None:
    """Append one environment-control sample if recording is active."""
    with self._lock:
      if self._file is None or self._writer is None:
        return
      if self._first_sim_step is None:
        self._first_sim_step = sim_step
      time_s = (sim_step - self._first_sim_step) * self._step_dt
      self._writer.writerow(
        [
          self._sample,
          sim_step,
          f"{time_s:.9g}",
          env_idx,
          self._source_name(),
          *(values.get(name, np.nan) for name in self._term_names),
        ]
      )
      self._sample += 1
      if self._sample % 25 == 0:
        self._file.flush()

  def stop(self) -> Path | None:
    """Flush and close the active file, returning its path."""
    with self._lock:
      if self._file is None:
        return self._path
      self._file.flush()
      self._file.close()
      self._file = None
      self._writer = None
      self._term_names = ()
      return self._path


@dataclass
class _TermState:
  """Mutable state for a single term."""

  name: str
  color: str
  enabled: bool = False
  history: deque[float] = field(default_factory=lambda: deque(maxlen=300))
  plot: viser.GuiUplotHandle | None = None


class ViserTermPlotter:
  """Handles plotting for the Viser viewer with selective display."""

  def __init__(
    self,
    server: viser.ViserServer,
    term_names: list[str],
    name: str = "Reward",
    history_length: int = 150,
    env_idx: int = 0,
    csv_output_dir: Path | None = None,
    source_name: Callable[[], str] | None = None,
    step_dt: float = 0.02,
  ) -> None:
    """Initialize the plotter.

    Args:
      server: The Viser server instance
      term_names: List of term names to plot
      name: Name prefix for the plots (e.g. "Reward" or "Metric")
      history_length: Number of points to keep in history
      env_idx: Index of the environment being displayed
      csv_output_dir: Enables CSV controls and stores recordings in this directory
      source_name: Returns the current task/checkpoint label for CSV metadata
      step_dt: Environment control period used for the CSV time axis
    """
    self._server = server
    self._name = name
    self._history_length = history_length
    self._loaded_csv: LoadedMetricsCsv | None = None
    self._loaded_csv_source_path: Path | None = None
    self._loaded_csv_name: str | None = None
    self._loaded_csv_generation = 0
    self._last_step_idx: int | None = None
    self._implicit_step_idx = 0
    self._recorder = (
      MetricsCsvRecorder(
        csv_output_dir,
        source_name or (lambda: "unknown-network"),
        step_dt,
      )
      if csv_output_dir is not None
      else None
    )
    self._image_executor = (
      ThreadPoolExecutor(max_workers=1, thread_name_prefix="metrics-image-export")
      if csv_output_dir is not None
      else None
    )
    self._closed = False
    self._csv_file_paths: dict[str, Path] = {}

    # Pre-allocated x-axis array (reused for all plots).
    self._x_array = np.arange(-history_length + 1, 1, dtype=np.float64)

    # Stable color assignment.
    self._terms: dict[str, _TermState] = {}
    for i, tname in enumerate(term_names):
      self._terms[tname] = _TermState(
        name=tname,
        color=_color_for(i),
        history=deque(maxlen=history_length),
      )

    # GUI handles.
    self._checkboxes: dict[str, viser.GuiInputHandle] = {}

    self._empty = np.array([], dtype=np.float64)

    self._env_idx = env_idx

    # Build all GUI elements.
    self._build_selector_gui(term_names)
    if self._recorder is not None:
      self._build_csv_gui()
    self._plots_folder = self._server.gui.add_folder("Plots", expand_by_default=True)

  def _build_selector_gui(self, term_names: list[str]) -> None:
    """Build flat checkboxes with a filter input for term selection."""
    with self._server.gui.add_folder("Select terms", expand_by_default=True):
      self._env_label = self._server.gui.add_markdown(self._env_label_text())

      # Filter input.
      self._filter_input = self._server.gui.add_text(
        "Filter",
        initial_value="",
        hint="Term name must contain this string",
      )

      @self._filter_input.on_update
      def _(_) -> None:
        filter_str = self._filter_input.value.lower()
        for tname in self._terms:
          cb = self._checkboxes[tname]
          cb.visible = filter_str in tname.lower()
        self._sync_plots()

      # Bulk actions.
      bulk = self._server.gui.add_button_group("Select", options=["All", "None"])

      @bulk.on_click
      def _(event) -> None:
        enable = event.target.value == "All"
        for tname, state in self._terms.items():
          cb = self._checkboxes[tname]
          if cb.visible and not cb.disabled:
            state.enabled = enable
            cb.value = enable
        self._sync_plots()

      available_joint_signals = [
        label
        for label, prefix in _JOINT_SIGNAL_PREFIXES.items()
        if any(term_name.startswith(prefix) for term_name in term_names)
      ]
      if available_joint_signals:
        joint_signal = self._server.gui.add_button_group(
          "Joint plots", options=[*available_joint_signals, "None"]
        )

        @joint_signal.on_click
        def _(event) -> None:
          selected_prefix = _JOINT_SIGNAL_PREFIXES.get(event.target.value)
          joint_prefixes = tuple(_JOINT_SIGNAL_PREFIXES.values())
          for term_name, state in self._terms.items():
            if not term_name.startswith(joint_prefixes):
              continue
            cb = self._checkboxes[term_name]
            if cb.disabled:
              continue
            enabled = selected_prefix is not None and term_name.startswith(
              selected_prefix
            )
            state.enabled = enabled
            cb.value = enabled
          self._sync_plots()

      # Flat checkbox list.
      for tname in term_names:
        state = self._terms[tname]
        cb = self._server.gui.add_checkbox(
          tname,
          initial_value=state.enabled,
          hint=f"Color: {state.color}",
        )
        self._checkboxes[tname] = cb

        @cb.on_update
        def _(event, _tname=tname) -> None:
          self._terms[_tname].enabled = event.target.value
          self._sync_plots()

  def _build_csv_gui(self) -> None:
    """Build recording, server-file, and browser-upload controls."""
    assert self._recorder is not None
    recorder = self._recorder
    with self._server.gui.add_folder("CSV data", expand_by_default=True):
      self._data_status = self._server.gui.add_markdown(
        f"Save directory: `{recorder.output_dir}`"
      )
      self._record_start = self._server.gui.add_button(
        "Record selected",
        hint="Record only the currently checked metric terms",
      )
      self._record_stop = self._server.gui.add_button(
        "Stop & save",
        disabled=True,
      )

      @self._record_start.on_click
      def _(_) -> None:
        selected = [state.name for state in self._terms.values() if state.enabled]
        try:
          path = recorder.start(selected)
        except (OSError, ValueError) as exc:
          self._set_data_status(f"Recording failed: {exc}")
          return
        self._record_start.disabled = True
        self._record_stop.disabled = False
        self._return_live.disabled = True
        self._set_data_status(
          f"Recording {len(selected)} terms to `{path.name}`"
        )

      @self._record_stop.on_click
      def _(_) -> None:
        self._finish_recording()

      empty_csv_options: list[str] = ["No CSV files"]
      self._saved_csv_dropdown = self._server.gui.add_dropdown(
        "Saved CSV",
        options=empty_csv_options,
        initial_value=empty_csv_options[0],
        disabled=True,
      )
      self._refresh_csv_button = self._server.gui.add_button("Refresh saved files")
      self._open_csv_button = self._server.gui.add_button(
        "Open saved CSV",
        disabled=True,
      )

      @self._refresh_csv_button.on_click
      def _(_) -> None:
        self._refresh_saved_csvs()

      @self._open_csv_button.on_click
      def _(_) -> None:
        path = self._csv_file_paths.get(self._saved_csv_dropdown.value)
        if path is None:
          self._set_data_status("Select an available saved CSV file")
          return
        try:
          content = path.read_bytes()
        except OSError as exc:
          self._set_data_status(f"Could not read saved CSV: {exc}")
          self._refresh_saved_csvs()
          return
        self._load_csv_content(content, path.name, source_path=path)

      self._upload_csv_button = self._server.gui.add_upload_button(
        "Upload CSV",
        mime_type="text/csv",
        hint="Load a metrics CSV from this browser/device",
      )

      @self._upload_csv_button.on_upload
      def _(event) -> None:
        uploaded = event.target.value
        self._load_csv_content(uploaded.content, uploaded.name)

      self._save_images_button = self._server.gui.add_button(
        "Save selected plots + comparisons",
        disabled=True,
        hint="Save 1080p individual/overview plots and a 1920x3780 left-right plot",
      )

      @self._save_images_button.on_click
      def _(_) -> None:
        self._save_loaded_curve_images()

      self._return_live = self._server.gui.add_button(
        "Return to live",
        disabled=True,
      )

      @self._return_live.on_click
      def _(_) -> None:
        self._leave_loaded_csv()

      self._refresh_saved_csvs()

  def _set_data_status(self, message: str) -> None:
    """Update the CSV panel status line."""
    self._data_status.content = message

  def _refresh_saved_csvs(self, preferred: Path | None = None) -> None:
    """Refresh CSV files found beside the selected model checkpoint."""
    assert self._recorder is not None
    try:
      paths = sorted(
        self._recorder.output_dir.glob("metrics_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
      )
    except OSError as exc:
      self._set_data_status(f"Could not list saved CSV files: {exc}")
      paths = []
    self._csv_file_paths = {path.name: path for path in paths}
    if not paths:
      self._saved_csv_dropdown.options = ["No CSV files"]
      self._saved_csv_dropdown.value = "No CSV files"
      self._saved_csv_dropdown.disabled = True
      self._open_csv_button.disabled = True
      return

    options = [path.name for path in paths]
    self._saved_csv_dropdown.options = options
    preferred_name = preferred.name if preferred is not None else None
    self._saved_csv_dropdown.value = (
      preferred_name if preferred_name in self._csv_file_paths else options[0]
    )
    self._saved_csv_dropdown.disabled = False
    self._open_csv_button.disabled = False

  def _finish_recording(self) -> None:
    """Stop recording, close the file, and expose it in the saved-file picker."""
    assert self._recorder is not None
    was_active = self._recorder.active
    try:
      path = self._recorder.stop()
    except OSError as exc:
      self._set_data_status(f"Could not finish recording: {exc}")
      return
    self._record_start.disabled = self._loaded_csv is not None
    self._record_stop.disabled = True
    if was_active and path is not None:
      self._set_data_status(f"Saved recording to `{path}`")
      self._refresh_saved_csvs(preferred=path)

  def _load_csv_content(
    self,
    content: bytes,
    display_name: str,
    source_path: Path | None = None,
  ) -> None:
    """Validate CSV bytes and switch all available plots to offline mode."""
    assert self._recorder is not None
    if self._recorder.active:
      self._finish_recording()
    try:
      loaded = load_metrics_csv(content, tuple(self._terms))
    except ValueError as exc:
      self._set_data_status(f"Could not load CSV: {exc}")
      return

    self._loaded_csv = loaded
    self._loaded_csv_source_path = source_path
    self._loaded_csv_name = Path(display_name).name
    self._loaded_csv_generation += 1
    for name, state in self._terms.items():
      available = name in loaded.values
      state.enabled = available
      state.history.clear()
      checkbox = self._checkboxes[name]
      checkbox.disabled = not available
      checkbox.value = available
    self._env_label.content = self._env_label_text()
    self._recreate_plots()
    self._record_start.disabled = True
    self._record_stop.disabled = True
    self._save_images_button.disabled = False
    self._return_live.disabled = False
    source = f" from {loaded.network}" if loaded.network else ""
    self._set_data_status(
      f"Loaded {len(loaded.values)} curves / {len(loaded.time_s)} samples"
      f"{source} (`{display_name}`)"
    )

  def _save_loaded_curve_images(self) -> None:
    """Export checked offline curves in the background and report their path."""
    assert self._recorder is not None
    assert self._image_executor is not None
    loaded = self._loaded_csv
    csv_name = self._loaded_csv_name
    if loaded is None or csv_name is None:
      self._set_data_status("Load a CSV before saving curve images")
      return
    selected = [
      name
      for name, state in self._terms.items()
      if state.enabled and name in loaded.values
    ]
    if not selected:
      self._set_data_status("Select at least one loaded curve before saving images")
      return

    csv_parent = (
      self._loaded_csv_source_path.parent
      if self._loaded_csv_source_path is not None
      else self._recorder.output_dir
    )
    colors = {name: self._terms[name].color for name in selected}
    generation = self._loaded_csv_generation
    self._save_images_button.disabled = True
    self._set_data_status(
      f"Saving {len(selected)} curves with grouped comparison PNGs..."
    )
    future = self._image_executor.submit(
      save_metrics_curve_images,
      loaded,
      selected,
      csv_parent,
      csv_name,
      colors,
    )

    def _on_export_done(_future) -> None:
      try:
        paths = _future.result()
      except (ImportError, OSError, RuntimeError, ValueError) as exc:
        if not self._closed and generation == self._loaded_csv_generation:
          self._save_images_button.disabled = False
          self._set_data_status(f"Could not save curve images: {exc}")
        return
      if self._closed or generation != self._loaded_csv_generation:
        return
      self._save_images_button.disabled = False
      self._set_data_status(
        f"Saved {len(paths)} PNGs under `{csv_parent / 'images'}`"
      )

    future.add_done_callback(_on_export_done)

  def _leave_loaded_csv(self) -> None:
    """Return plots from an uploaded recording to live environment values."""
    if self._loaded_csv is None:
      return
    self._loaded_csv = None
    self._loaded_csv_source_path = None
    self._loaded_csv_name = None
    self._loaded_csv_generation += 1
    for name, state in self._terms.items():
      state.history.clear()
      self._checkboxes[name].disabled = False
    self._env_label.content = self._env_label_text()
    self._recreate_plots()
    self._record_start.disabled = False
    self._save_images_button.disabled = True
    self._return_live.disabled = True
    self._set_data_status("Showing live metrics")

  def _recreate_plots(self) -> None:
    """Rebuild plot handles after switching between live and CSV data."""
    for state in self._terms.values():
      if state.plot is not None:
        state.plot.remove()
        state.plot = None
    self._sync_plots()

  def _env_label_text(self) -> str:
    if self._loaded_csv is not None:
      return "<small><em>Showing curves loaded from CSV</em></small>"
    return f"<small><em>Showing terms for environment #{self._env_idx}</em></small>"

  def update_env_idx(self, env_idx: int) -> None:
    """Update the displayed environment index."""
    self._env_idx = env_idx
    self._env_label.content = self._env_label_text()

  def on_source_change(self) -> None:
    """Finish an active recording before the policy/checkpoint changes."""
    if self._recorder is not None and self._recorder.active:
      self._finish_recording()

  def _sync_plots(self) -> None:
    """Create or remove plots to match current selection."""
    for tname, state in self._terms.items():
      cb = self._checkboxes[tname]
      available = self._loaded_csv is None or tname in self._loaded_csv.values
      should_show = state.enabled and cb.visible and available
      if should_show and state.plot is None:
        self._create_plot(state)
      elif not should_show and state.plot is not None:
        state.plot.remove()
        state.plot = None

  def _create_plot(self, state: _TermState) -> None:
    """Lazily create a single-term plot inside the scoped folder."""
    if self._loaded_csv is not None and state.name in self._loaded_csv.values:
      x = self._loaded_csv.time_s
      y = self._loaded_csv.values[state.name]
      x_series_label = "Time (s)"
      x_scale = viser.uplot.Scale(time=False, auto=True)
    else:
      h = state.history
      hist_len = len(h)
      if hist_len > 0:
        x = self._x_array[-hist_len:]
        y = np.fromiter(h, dtype=np.float64, count=hist_len)
      else:
        x = self._empty
        y = self._empty
      x_series_label = "Steps"
      x_scale = viser.uplot.Scale(
        time=False,
        auto=False,
        range=(-self._history_length, 0),
      )

    with self._plots_folder:
      state.plot = self._server.gui.add_uplot(
        data=(x, y),
        series=(
          viser.uplot.Series(label=x_series_label),
          viser.uplot.Series(label=state.name, stroke=state.color, width=2),
        ),
        scales={
          "x": x_scale,
          "y": viser.uplot.Scale(auto=True),
        },
        legend=viser.uplot.Legend(show=True),
        title=state.name,
        aspect=2.0,
        visible=True,
      )

  def update(
    self,
    terms: list[tuple[str, np.ndarray]],
    step_idx: int | None = None,
  ) -> None:
    """Push new data and refresh visible plots."""
    if self._loaded_csv is not None:
      return
    if step_idx is None:
      step_idx = self._implicit_step_idx
      self._implicit_step_idx += 1
    elif step_idx == self._last_step_idx:
      return
    self._last_step_idx = step_idx

    any_enabled = False
    values: dict[str, float] = {}
    for tname, arr in terms:
      state = self._terms.get(tname)
      if state is None:
        continue
      val = float(arr[0])
      values[tname] = val
      if np.isfinite(val):
        state.history.append(val)
      if state.enabled:
        any_enabled = True

    if self._recorder is not None and self._recorder.active:
      self._recorder.append(values, self._env_idx, step_idx)

    if not any_enabled:
      return

    # Update plots.
    for state in self._terms.values():
      if not state.enabled or state.plot is None:
        continue
      h = state.history
      hist_len = len(h)
      if hist_len > 0:
        x = self._x_array[-hist_len:]
        y = np.fromiter(h, dtype=np.float64, count=hist_len)
        state.plot.data = (x, y)

  def clear_histories(self) -> None:
    """Clear all term histories."""
    if self._loaded_csv is not None:
      return
    for state in self._terms.values():
      state.history.clear()
      if state.plot is not None:
        state.plot.data = (self._empty, self._empty)

  def cleanup(self) -> None:
    """Clean up resources."""
    self._closed = True
    if self._recorder is not None:
      self._recorder.stop()
    if self._image_executor is not None:
      self._image_executor.shutdown(wait=True)
    for state in self._terms.values():
      if state.plot is not None:
        state.plot.remove()
    for cb in self._checkboxes.values():
      cb.remove()
    self._terms.clear()
    self._checkboxes.clear()
