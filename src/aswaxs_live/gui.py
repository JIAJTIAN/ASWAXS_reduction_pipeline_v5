"""Legacy GUI launcher for FrameByFrame-ASWAXS live reduction.

Window 0 is the setup/control window. It builds the command for the live reducer
launcher and can launch/stop the reducer.

Window 1 is the acquisition/reduction monitor. It shows reducer stdout/stderr
and tails ``live_events.jsonl`` so the user can see frame, group, and energy
triggers as they happen.

Window 2 is the curve browser/plotter. It reuses ``LiveCurveViewer`` and lets
the user select available 1D output files to plot.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from aswaxs_live.bluesky_queue import append_measurement_done_message
from aswaxs_live.stitcher import (
    StitchedAsaxsSettings,
    clear_stitched_averages,
    paired_detector_analysis_h5s,
    update_live_stitched_averages,
    write_stitched_asaxs_outputs,
)
from aswaxs_live.viewer import LiveCurveViewer
from aswaxs_live.xanos_export import export_analysis_h5_to_xanos_format


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
PLAYGROUND_DIR = PROJECT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "live_gui_run"
DEFAULT_MANIFEST = ""
DEFAULT_EIGER_WATCH_DIR = ""
DEFAULT_PIL300K_WATCH_DIR = ""
DEFAULT_WATCH_DIR = ""
DEFAULT_EXTRACTED_FOLDER_NAME = "Extracted"
DEFAULT_SAMPLE_NAME = ""
DEFAULT_SAXS_PONI = ""
DEFAULT_SAXS_MASK = ""
DEFAULT_WAXS_PONI = ""
DEFAULT_WAXS_MASK = ""
DEFAULT_PONI = DEFAULT_SAXS_PONI
DEFAULT_MASK = DEFAULT_SAXS_MASK
SETTINGS_PATH = PROJECT_DIR / ".aswaxs_live_gui_settings.json"
DEFAULT_KAFKA_BOOTSTRAP = "164.54.169.92:9092"
DEFAULT_KAFKA_TOPIC = "bluesky_aswaxs"
DEFAULT_KAFKA_GROUP_ID = "aswaxs-v5-reduction-bridge"


@dataclass
class PreflightRow:
    sample: str
    detector: str
    data_dir: Path
    file_count: int
    num_energies: int
    num_frames: int
    inferred_groups: int | None
    warning: str | None = None


class NoWheelSpinBox(QtWidgets.QSpinBox):
    """QSpinBox that ignores mouse-wheel changes."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt method name.
        event.ignore()


class NoWheelDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """QDoubleSpinBox that ignores mouse-wheel changes."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt method name.
        event.ignore()


class ProcessMonitorWindow(QtWidgets.QMainWindow):
    """Show reducer text output and structured live event records."""

    def __init__(self, output_dir: Path, title: str = "ASWAXS Live Process Monitor") -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1120, 760)
        self.process: QtCore.QProcess | None = None
        self.output_dir = output_dir
        self.event_log_path = self.output_dir / "live_events.jsonl"
        self._event_offset = 0
        self.expected_frames: int | None = None
        self._dynamic_expected_frames = True
        self._expected_data_dirs: set[str] = set()
        self.frames_reduced = 0
        self.groups_done = 0
        self.energies_done = 0
        self.run_started_at: float | None = None
        self.cpu_info_text = "CPU: 1 process x 1 core"
        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tail_event_log)
        self.timer.start()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        self.setStyleSheet(
            """
            QGroupBox {
                font-weight: 600;
                border: 1px solid #c8cdd4;
                border-radius: 6px;
                margin-top: 10px;
                background: #f8f9fb;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QToolButton {
                font-weight: 600;
                padding: 4px 2px;
                text-align: left;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                min-height: 24px;
            }
            QPushButton {
                min-height: 26px;
                padding-left: 10px;
                padding-right: 10px;
            }
            """
        )
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 6)
        root.setSpacing(8)

        self.status_label = QtWidgets.QLabel("Reducer is not running")
        root.addWidget(self.status_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        root.addWidget(splitter, 1)

        self.event_table = QtWidgets.QTableWidget(0, 7)
        self.event_table.setHorizontalHeaderLabels(
            ["time", "event", "energy", "group", "frame", "sequence", "message/path"]
        )
        header = self.event_table.horizontalHeader()
        header.setStretchLastSection(True)
        for column, width in enumerate([220, 230, 70, 70, 70, 90, 420]):
            self.event_table.setColumnWidth(column, width)
        for column in range(6):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)
        splitter.addWidget(self.event_table)

        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        splitter.addWidget(self.log_edit)
        splitter.setSizes([430, 300])

        progress_panel = QtWidgets.QWidget()
        progress_layout = QtWidgets.QVBoxLayout(progress_panel)
        progress_layout.setContentsMargins(0, 6, 0, 0)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Waiting for frames")
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #9aa4b2;
                border-radius: 6px;
                background: #eef1f5;
                min-height: 18px;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background-color: #2f7ed8;
            }
            """
        )
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QtWidgets.QLabel("Frames 0 | Groups 0 | Energies 0")
        progress_layout.addWidget(self.progress_label)
        root.addWidget(progress_panel)

    def set_output_dir(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.event_log_path = self.output_dir / "live_events.jsonl"
        self.clear_run_display()

    def prepare_event_log(self, *, clear: bool = False) -> None:
        """Create the event log path that the progress table tails."""
        try:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            if clear:
                try:
                    self.event_log_path.unlink()
                except FileNotFoundError:
                    pass
            if not self.event_log_path.exists():
                self.event_log_path.touch()
            self.status_label.setText(f"Watching event log: {self.event_log_path}")
        except OSError as exc:
            self.status_label.setText(f"Cannot prepare event log: {exc}")

    def set_expected_frames(self, expected_frames: int | None) -> None:
        if expected_frames and expected_frames > 0:
            self.expected_frames = expected_frames
            self._dynamic_expected_frames = False
        else:
            if self.frames_reduced == 0:
                self.expected_frames = None
            self._dynamic_expected_frames = True
        self._update_progress()

    def set_cpu_info(self, text: str) -> None:
        self.cpu_info_text = text.strip() or "CPU: 1 process x 1 core"
        self._update_progress()

    def clear_run_display(self) -> None:
        self._event_offset = 0
        self.frames_reduced = 0
        self.groups_done = 0
        self.energies_done = 0
        self.run_started_at = None
        self._expected_data_dirs.clear()
        self.event_table.setRowCount(0)
        self.log_edit.clear()
        self.status_label.setText("Reducer is not running")
        self._update_progress()

    def attach_process(self, process: QtCore.QProcess) -> None:
        self.process = process
        process.readyReadStandardOutput.connect(self.read_stdout)
        process.readyReadStandardError.connect(self.read_stderr)
        process.started.connect(self._process_started)
        process.finished.connect(self._process_finished)

    def _process_started(self) -> None:
        self.status_label.setText("Reducer running")
        self.run_started_at = time.monotonic()
        if self.event_log_path.exists() and self._event_offset == 0:
            try:
                self._event_offset = self.event_log_path.stat().st_size
            except OSError:
                self._event_offset = 0
        self._update_progress()

    def append_log(self, text: str) -> None:
        self.log_edit.appendPlainText(text.rstrip())
        self._maybe_follow_output_line(text)

    def _maybe_follow_output_line(self, text: str) -> None:
        """Follow the reducer's active output folder if it reports one."""
        for line in text.splitlines():
            if not line.startswith("Output:"):
                continue
            output_text = line.split(":", 1)[1].strip()
            if not output_text:
                continue
            output_dir = Path(output_text).expanduser()
            if output_dir == self.output_dir:
                continue
            self.output_dir = output_dir
            self.event_log_path = self.output_dir / "live_events.jsonl"
            self._event_offset = 0
            self.prepare_event_log(clear=False)

    def read_stdout(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.append_log(text)

    def read_stderr(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode(errors="replace")
        if text:
            self.append_log(text)

    def _process_finished(self, code: int, status: QtCore.QProcess.ExitStatus) -> None:
        state = "crashed" if status == QtCore.QProcess.CrashExit else "finished"
        self.status_label.setText(f"Reducer {state} with exit code {code}")
        self.tail_event_log()

    def tail_event_log(self) -> None:
        if not self.event_log_path.exists():
            self.status_label.setText(f"Waiting for event log: {self.event_log_path}")
            return
        try:
            with self.event_log_path.open("r", encoding="utf-8") as handle:
                handle.seek(self._event_offset)
                lines = handle.readlines()
                self._event_offset = handle.tell()
        except OSError as exc:
            self.status_label.setText(f"Cannot read event log: {exc}")
            return
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._add_event_row(event)

    def _add_event_row(self, event: dict[str, object]) -> None:
        row = self.event_table.rowCount()
        self.event_table.insertRow(row)
        message = event.get("message") or event.get("path") or ""
        values = [
            event.get("time", ""),
            event.get("event", ""),
            event.get("energy_index", ""),
            event.get("group_index", ""),
            event.get("frame_index", ""),
            event.get("sequence_index", ""),
            message,
        ]
        for column, value in enumerate(values):
            self.event_table.setItem(row, column, QtWidgets.QTableWidgetItem("" if value is None else str(value)))
        self.event_table.scrollToBottom()
        self._update_progress_from_event(event)

    def _update_progress_from_event(self, event: dict[str, object]) -> None:
        event_name = str(event.get("event", ""))
        self._learn_expected_frames_from_event(event)
        if event_name == "frame_reduced_1d":
            if self.run_started_at is None:
                self.run_started_at = time.monotonic()
            self.frames_reduced += 1
        elif event_name == "group_average_written":
            self.groups_done += 1
        elif event_name in {"energy_batch_asaxs_completed", "energy_batch_saxs_completed"}:
            self.energies_done += 1
        self._update_progress()

    def _learn_expected_frames_from_event(self, event: dict[str, object]) -> None:
        """Use reducer-authored totals for queue runs where preflight is unknown."""
        if not self._dynamic_expected_frames:
            return
        total = self._event_int(event.get("expected_total_frames"))
        if total is None or total <= 0:
            return
        data_dir = str(event.get("data_dir") or event.get("output_dir") or event.get("path") or "")
        event_name = str(event.get("event", ""))
        if event_name in {"measurement_done_received", "auto_groups_inferred"} and data_dir:
            if data_dir in self._expected_data_dirs:
                return
            self._expected_data_dirs.add(data_dir)
            self.expected_frames = (self.expected_frames or 0) + total
            return
        if event_name == "file_assigned_sequence":
            self.expected_frames = max(self.expected_frames or 0, total)

    def _event_int(self, value: object) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _update_progress(self) -> None:
        if self.expected_frames:
            display_total = max(self.expected_frames, self.frames_reduced)
            self.progress_bar.setMaximum(display_total)
            self.progress_bar.setValue(min(self.frames_reduced, display_total))
            percent = 100.0 * min(self.frames_reduced, display_total) / display_total
            self.progress_bar.setFormat(
                f"{self.frames_reduced}/{display_total} frames ({percent:.1f}%) | "
                f"{self._eta_text()}"
            )
        else:
            self.progress_bar.setMaximum(0)
            self.progress_bar.setFormat(f"{self.frames_reduced} frames reduced | {self._elapsed_text()}")
        self.progress_label.setText(
            f"Frames {self.frames_reduced}"
            + (f" / {max(self.expected_frames, self.frames_reduced)}" if self.expected_frames else "")
            + f" | Groups {self.groups_done} | Energies {self.energies_done}"
            + f" | {self.cpu_info_text}"
            + f" | {self._elapsed_text()}"
            + (f" | {self._eta_text()}" if self.expected_frames else "")
        )

    def _elapsed_text(self) -> str:
        if self.run_started_at is None:
            return "elapsed --"
        return f"elapsed {self._format_duration(time.monotonic() - self.run_started_at)}"

    def _eta_text(self) -> str:
        if self.run_started_at is None or not self.expected_frames or self.frames_reduced <= 0:
            return "remaining estimating"
        elapsed = time.monotonic() - self.run_started_at
        rate = self.frames_reduced / elapsed if elapsed > 0 else 0
        if rate <= 0:
            return "remaining estimating"
        remaining_frames = max(0, self.expected_frames - self.frames_reduced)
        return f"remaining {self._format_duration(remaining_frames / rate)}"

    def _format_duration(self, seconds: float) -> str:
        total = max(0, int(round(seconds)))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m {secs:02d}s"
        if minutes:
            return f"{minutes:d}m {secs:02d}s"
        return f"{secs:d}s"


class TabbedToolWindow(QtWidgets.QMainWindow):
    """Small container that keeps related live tools in one tabbed window."""

    def __init__(self, title: str, tabs: dict[str, QtWidgets.QWidget]) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1220, 820)
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        for label, widget in tabs.items():
            self.tabs.addTab(widget, label)

    def show_tab(self, label: str) -> None:
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == label:
                self.tabs.setCurrentIndex(index)
                break
        self.show()
        self.raise_()


class SetupWindow(QtWidgets.QMainWindow):
    """Parameter window for launching and monitoring live reduction."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FrameByFrame-ASWAXS Live Pipeline Setup")
        self.resize(1180, 860)
        self.process: QtCore.QProcess | None = None
        self.bridge_process: QtCore.QProcess | None = None
        self.detector_processes: dict[str, QtCore.QProcess] = {}
        self.stitch_run_started_ns: int | None = None
        self._settings_loaded = False
        self.monitor_window = ProcessMonitorWindow(DEFAULT_OUTPUT_DIR)
        self.bridge_monitor_window = ProcessMonitorWindow(DEFAULT_OUTPUT_DIR, "Kafka Bridge Monitor")
        self.dual_monitor_windows = {
            "pil300k": ProcessMonitorWindow(DEFAULT_OUTPUT_DIR / "Pil300K", "Pil300K Process Monitor"),
            "eig1m": ProcessMonitorWindow(DEFAULT_OUTPUT_DIR / "Eig1M", "Eig1M Process Monitor"),
        }
        self.curve_window = LiveCurveViewer(DEFAULT_OUTPUT_DIR, refresh_ms=1000)
        self.dual_curve_windows = {
            "pil300k": LiveCurveViewer(DEFAULT_OUTPUT_DIR / "Pil300K", refresh_ms=1000),
            "eig1m": LiveCurveViewer(DEFAULT_OUTPUT_DIR / "Eig1M", refresh_ms=1000),
            "stitched": LiveCurveViewer(DEFAULT_OUTPUT_DIR / f"{DEFAULT_SAMPLE_NAME}_analysis.h5", refresh_ms=1000),
        }
        self.dual_curve_windows["stitched"].curve_kind_combo.setCurrentText("h5 stitched averages")
        self.dual_curve_windows["stitched"].follow_latest_check.setChecked(True)
        self.dual_monitor_window = TabbedToolWindow(
            "Detector Process Monitors",
            {
                "Pil300K": self.dual_monitor_windows["pil300k"],
                "Eig1M": self.dual_monitor_windows["eig1m"],
            },
        )
        self.dual_curve_window = TabbedToolWindow(
            "Detector Curves",
            {
                "Pil300K": self.dual_curve_windows["pil300k"],
                "Eig1M": self.dual_curve_windows["eig1m"],
                "Stitched": self.dual_curve_windows["stitched"],
            },
        )
        for window in self.dual_curve_windows.values():
            window.timer.stop()
        self.stitch_timer = QtCore.QTimer(self)
        self.stitch_timer.setInterval(1000)
        self.stitch_timer.timeout.connect(self._update_live_stitched_outputs)
        self._build_ui()
        self._load_settings()
        self._settings_loaded = True
        self._update_source_visibility()
        self.refresh_command()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QtWidgets.QWidget()
        self.form_layout = QtWidgets.QHBoxLayout(form_widget)
        self.form_layout.setContentsMargins(12, 12, 12, 12)
        self.form_layout.setSpacing(12)
        self.left_column = QtWidgets.QVBoxLayout()
        self.right_column = QtWidgets.QVBoxLayout()
        for column in (self.left_column, self.right_column):
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(12)
            column.addStretch(1)
        self.form_layout.addLayout(self.left_column, 1)
        self.form_layout.addLayout(self.right_column, 1)
        self._next_section_column = 0
        scroll.setWidget(form_widget)
        root.addWidget(scroll, 1)

        self._build_source_group()
        self._build_sequence_group()
        self._build_bluesky_queue_group()
        self._build_paths_group()
        self._build_reduction_group()
        self._build_asaxs_group()

        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(8)
        root.addLayout(actions)
        self.start_button = QtWidgets.QPushButton("Start Reducer")
        self.start_button.clicked.connect(self.start_reducer)
        actions.addWidget(self.start_button)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_reducer)
        actions.addWidget(self.stop_button)
        monitor_button = QtWidgets.QPushButton("Show Process Monitor")
        monitor_button.clicked.connect(self.show_monitor)
        actions.addWidget(monitor_button)
        curves_button = QtWidgets.QPushButton("Show Curves")
        curves_button.clicked.connect(self.show_curves)
        actions.addWidget(curves_button)
        open_h5_button = QtWidgets.QPushButton("Open Analysis H5")
        open_h5_button.clicked.connect(self.open_analysis_h5_in_viewer)
        actions.addWidget(open_h5_button)
        export_xanos_button = QtWidgets.QPushButton("Export XAnos Format")
        export_xanos_button.clicked.connect(self.export_xanos_format_from_h5)
        actions.addWidget(export_xanos_button)
        actions.addStretch(1)

        self.statusBar().showMessage("Ready")

    def _add_main_section(self, widget: QtWidgets.QWidget, *, span: int = 1) -> None:
        """Place setup sections in two independent vertical columns."""
        if span >= 2:
            self.left_column.insertWidget(self.left_column.count() - 1, widget)
            return
        target = self.left_column if self._next_section_column == 0 else self.right_column
        target.insertWidget(target.count() - 1, widget)
        self._next_section_column = 1 - self._next_section_column

    def _setup_form(self, form: QtWidgets.QFormLayout) -> QtWidgets.QFormLayout:
        """Use consistent compact spacing for setup forms."""
        form.setContentsMargins(10, 14, 10, 10)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(7)
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        return form

    def _settings_widgets(self) -> dict[str, QtWidgets.QWidget]:
        return {
            "source_mode": self.source_mode_combo,
            "detector_run_mode": self.detector_run_mode_combo,
            "analysis_mode": self.analysis_mode_combo,
            "sample_name": self.sample_name_edit,
            "sample_queue": self.sample_queue_table,
            "write_text_output": self.write_text_output_check,
            "restart_behavior": self.restart_behavior_combo,
            "once": self.once_check,
            "use_measurement_queue": self.use_measurement_queue_check,
            "queue_source": self.queue_source_combo,
            "start_kafka_bridge": self.start_kafka_bridge_check,
            "measurement_queue": self.measurement_queue_edit,
            "kafka_bootstrap_servers": self.kafka_bootstrap_edit,
            "kafka_topic": self.kafka_topic_edit,
            "kafka_group_id": self.kafka_group_id_edit,
            "watch_dir": self.watch_dir_edit,
            "extracted_folder_name": self.extracted_folder_edit,
            "manifest": self.manifest_edit,
            "output_dir": self.output_dir_edit,
            "analysis_h5": self.analysis_h5_edit,
            "poni": self.poni_edit,
            "mask": self.mask_edit,
            "pil300k_poni": self.saxs_poni_edit,
            "pil300k_mask": self.saxs_mask_edit,
            "eig1m_poni": self.waxs_poni_edit,
            "eig1m_mask": self.waxs_mask_edit,
            "pil300k_monitor_key": self.saxs_monitor_key_edit,
            "eig1m_monitor_key": self.waxs_monitor_key_edit,
            "dataset_path": self.dataset_path_edit,
            "detector": self.detector_combo,
            "monitor_key": self.monitor_key_edit,
            "npt": self.npt_spin,
            "jobs": self.jobs_spin,
            "analysis_write_interval_groups": self.analysis_write_interval_spin,
            "unit": self.unit_edit,
            "outlier_zmax": self.outlier_spin,
            "delta_energy_percent": self.delta_energy_spin,
            "pattern": self.pattern_edit,
            "num_energies": self.num_energies_spin,
            "auto_num_groups": self.auto_num_groups_check,
            "num_groups": self.num_groups_spin,
            "num_frames": self.num_frames_spin,
            "limit_energies": self.limit_energies_spin,
            "limit_frames": self.limit_frames_spin,
            "poll_seconds": self.poll_spin,
            "settle_seconds": self.settle_spin,
            "gc_group": self.gc_group_spin,
            "air_group": self.air_group_spin,
            "empty_group": self.empty_group_spin,
            "asaxs_pairs": self.asaxs_pairs_table,
            "gc_reference_file": self.gc_ref_edit,
            "gc_q_min": self.gc_q_min_spin,
            "gc_q_max": self.gc_q_max_spin,
            "capillary_thickness": self.capillary_thickness_spin,
            "gc_thickness": self.gc_thickness_spin,
            "subtract_fluorescence": self.subtract_fluorescence_check,
            "fluorescence_reference": self.fluorescence_reference_combo,
            "fluorescence_level": self.fluorescence_level_spin,
            "fluorescence_q_min": self.fluorescence_q_min_spin,
            "fluorescence_q_max": self.fluorescence_q_max_spin,
        }

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            return
        try:
            settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for key, widget in self._settings_widgets().items():
            if key in settings:
                self._set_widget_value(widget, settings[key])
        legacy_detector_keys = {
            "saxs_watch_dir": self.saxs_watch_dir_edit,
            "waxs_watch_dir": self.waxs_watch_dir_edit,
            "saxs_poni": self.saxs_poni_edit,
            "saxs_mask": self.saxs_mask_edit,
            "waxs_poni": self.waxs_poni_edit,
            "waxs_mask": self.waxs_mask_edit,
            "saxs_monitor_key": self.saxs_monitor_key_edit,
            "waxs_monitor_key": self.waxs_monitor_key_edit,
        }
        for key, widget in legacy_detector_keys.items():
            if key in settings:
                self._set_widget_value(widget, settings[key])
        if "watch_dir" not in settings:
            self._migrate_watch_root_from_legacy_settings(settings)
        if settings.get("detector_run_mode") == "SAXS + WAXS":
            self.detector_run_mode_combo.setCurrentText("Pil300K + Eig1M")
        if settings.get("queue_source") == "single sample":
            self.queue_source_combo.setCurrentText("sample list")
        self._migrate_dual_detector_settings()
        self._migrate_monitor_key_settings()
        self.statusBar().showMessage(f"Loaded previous settings from {SETTINGS_PATH.name}")

    def _save_settings(self) -> None:
        settings = {
            key: self._widget_value(widget)
            for key, widget in self._settings_widgets().items()
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            self.statusBar().showMessage(f"Could not save GUI settings: {exc}")

    def _widget_value(self, widget: QtWidgets.QWidget) -> object:
        if isinstance(widget, QtWidgets.QLineEdit):
            return widget.text()
        if isinstance(widget, QtWidgets.QPlainTextEdit):
            return widget.toPlainText()
        if isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText()
        if isinstance(widget, QtWidgets.QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QtWidgets.QSpinBox | QtWidgets.QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QtWidgets.QTableWidget) and widget is self.asaxs_pairs_table:
            values = []
            for row in range(widget.rowCount()):
                row_values = []
                for column in range(widget.columnCount()):
                    item = widget.item(row, column)
                    row_values.append(item.text().strip() if item is not None else "")
                if any(row_values):
                    values.append(row_values)
            return values
        if isinstance(widget, QtWidgets.QTableWidget):
            values = []
            for row in range(widget.rowCount()):
                item = widget.item(row, 0)
                text = item.text().strip() if item is not None else ""
                if text:
                    values.append(text)
            return values
        return None

    def _set_widget_value(self, widget: QtWidgets.QWidget, value: object) -> None:
        widget.blockSignals(True)
        try:
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.setText("" if value is None else str(value))
            elif isinstance(widget, QtWidgets.QPlainTextEdit):
                widget.setPlainText("" if value is None else str(value))
            elif isinstance(widget, QtWidgets.QComboBox):
                text = "" if value is None else str(value)
                if widget.findText(text) >= 0:
                    widget.setCurrentText(text)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QtWidgets.QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QtWidgets.QTableWidget) and widget is self.asaxs_pairs_table:
                widget.setRowCount(0)
                if isinstance(value, list):
                    for row_value in value:
                        if isinstance(row_value, list):
                            self._append_asaxs_pair_row(*[str(item) for item in row_value[:3]])
                        else:
                            parts = [part.strip() for part in str(row_value).replace(":", ",").split(",")]
                            self._append_asaxs_pair_row(*(parts + ["", ""])[:3])
            elif isinstance(widget, QtWidgets.QTableWidget):
                widget.setRowCount(0)
                if isinstance(value, list):
                    for text in value:
                        self._append_sample_queue_row(str(text))
        except (TypeError, ValueError):
            pass
        finally:
            widget.blockSignals(False)

    def _migrate_dual_detector_settings(self) -> None:
        """Swap older saved GUI defaults to SAXS=Pil300K and WAXS=Eig1M."""
        saxs_text = " ".join([self.saxs_watch_dir_edit.text(), self.saxs_poni_edit.text(), self.saxs_mask_edit.text()])
        waxs_text = " ".join([self.waxs_watch_dir_edit.text(), self.waxs_poni_edit.text(), self.waxs_mask_edit.text()])
        if "Eig1M" not in saxs_text or "Pil300K" not in waxs_text:
            return
        saxs_watch, waxs_watch = self.saxs_watch_dir_edit.text(), self.waxs_watch_dir_edit.text()
        saxs_poni, waxs_poni = self.saxs_poni_edit.text(), self.waxs_poni_edit.text()
        saxs_mask, waxs_mask = self.saxs_mask_edit.text(), self.waxs_mask_edit.text()
        self.saxs_watch_dir_edit.setText(waxs_watch)
        self.waxs_watch_dir_edit.setText(saxs_watch)
        self.saxs_poni_edit.setText(waxs_poni)
        self.waxs_poni_edit.setText(saxs_poni)
        self.saxs_mask_edit.setText(waxs_mask)
        self.waxs_mask_edit.setText(saxs_mask)

    def _migrate_watch_root_from_legacy_settings(self, settings: dict[str, object]) -> None:
        """Use old per-detector watch settings to fill the single acquisition root."""
        for key in ("pil300k_watch_dir", "eig1m_watch_dir", "saxs_watch_dir", "waxs_watch_dir"):
            text = str(settings.get(key, "")).strip()
            if not text:
                continue
            path = Path(text)
            if path.name.lower() in {"pil300k", "eig1m", "saxs", "waxs"} and path.parent.name:
                self.watch_dir_edit.setText(str(path.parent))
            else:
                self.watch_dir_edit.setText(str(path))
            return

    def _migrate_monitor_key_settings(self) -> None:
        """Rename old point-detector monitor keys to current beamline names."""
        replacements = {"SPD": "SPDS", "WPD": "WPDS"}
        for edit in [self.monitor_key_edit, self.saxs_monitor_key_edit, self.waxs_monitor_key_edit]:
            text = edit.text().strip()
            if text in replacements:
                edit.setText(replacements[text])

    def _line(self, value: str = "") -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(value)
        edit.textChanged.connect(self.refresh_command)
        return edit

    def _spin(self, value: int, minimum: int = 0, maximum: int = 1_000_000) -> QtWidgets.QSpinBox:
        spin = NoWheelSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.valueChanged.connect(self.refresh_command)
        return spin

    def _set_spin_without_signal(self, spin: QtWidgets.QSpinBox, value: int) -> None:
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)

    def _double(self, value: float, minimum: float = -1e9, maximum: float = 1e9) -> QtWidgets.QDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(6)
        spin.setValue(value)
        spin.valueChanged.connect(self.refresh_command)
        return spin

    def _combo(self, values: list[str], current: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItems(values)
        combo.setCurrentText(current)
        combo.currentTextChanged.connect(self.refresh_command)
        return combo

    def _browse_button(self, target: QtWidgets.QLineEdit, mode: str, caption: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton("Browse")

        def browse() -> None:
            start = target.text() or str(PROJECT_DIR)
            if mode == "dir":
                path = QtWidgets.QFileDialog.getExistingDirectory(self, caption, start)
            elif mode == "save_file":
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    caption,
                    start,
                    "HDF5 files (*.h5 *.hdf5);;All files (*)",
                )
            elif mode == "hdf5":
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    caption,
                    start,
                    "HDF5 files (*.h5 *.hdf5);;All files (*)",
                )
            else:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption, start, mode)
            if path:
                target.setText(path)

        button.clicked.connect(browse)
        return button

    def _path_row(self, form: QtWidgets.QFormLayout, label: str, edit: QtWidgets.QLineEdit, mode: str) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        layout.addWidget(self._browse_button(edit, mode, f"Select {label}"))
        form.addRow(label, row)
        return row

    def _collapsible_form_group(self, title: str, expanded: bool = False) -> tuple[QtWidgets.QWidget, QtWidgets.QFormLayout]:
        """Create a compact section that hides rarely changed controls."""
        section = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        toggle = QtWidgets.QToolButton()
        toggle.setText(title)
        toggle.setCheckable(True)
        toggle.setChecked(expanded)
        toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        content = QtWidgets.QWidget()
        form = self._setup_form(QtWidgets.QFormLayout(content))
        form.setContentsMargins(16, 6, 8, 8)
        content.setVisible(expanded)

        def set_open(opened: bool) -> None:
            content.setVisible(opened)
            toggle.setArrowType(QtCore.Qt.DownArrow if opened else QtCore.Qt.RightArrow)

        toggle.toggled.connect(set_open)
        layout.addWidget(toggle)
        layout.addWidget(content)
        return section, form

    def _build_source_group(self) -> None:
        box = QtWidgets.QGroupBox("Window 0: Run Mode")
        form = self._setup_form(QtWidgets.QFormLayout(box))
        self.source_mode_combo = self._combo(["watch folder", "manifest replay"], "watch folder")
        self.source_mode_combo.currentTextChanged.connect(self._update_source_visibility)
        form.addRow("Source", self.source_mode_combo)
        self.detector_run_mode_combo = self._combo(["single detector", "Pil300K + Eig1M"], "single detector")
        self.detector_run_mode_combo.currentTextChanged.connect(self._update_source_visibility)
        form.addRow("Detector jobs", self.detector_run_mode_combo)
        self.analysis_mode_combo = self._combo(["asaxs", "saxs"], "asaxs")
        self.analysis_mode_combo.currentTextChanged.connect(self._analysis_mode_changed)
        form.addRow("Analysis mode", self.analysis_mode_combo)
        self.sample_name_edit = self._line(DEFAULT_SAMPLE_NAME)
        form.addRow("Sample name", self.sample_name_edit)
        self.write_text_output_check = QtWidgets.QCheckBox("write legacy .dat curve files")
        self.write_text_output_check.setChecked(False)
        self.write_text_output_check.stateChanged.connect(self.refresh_command)
        form.addRow("Text output", self.write_text_output_check)
        self.restart_behavior_combo = self._combo(["resume", "restart"], "resume")
        form.addRow("Existing output", self.restart_behavior_combo)
        self.once_check = QtWidgets.QCheckBox("process current files once")
        self.once_check.stateChanged.connect(self._update_source_visibility)
        form.addRow("Watcher once", self.once_check)
        self._add_main_section(box)

    def _build_bluesky_queue_group(self) -> None:
        box = QtWidgets.QGroupBox("Sample Task Queue")
        self.bluesky_queue_box = box
        form = self._setup_form(QtWidgets.QFormLayout(box))
        self.use_measurement_queue_check = QtWidgets.QCheckBox("use sample task list")
        self.use_measurement_queue_check.stateChanged.connect(self._update_source_visibility)
        form.addRow("Reducer input", self.use_measurement_queue_check)
        self.queue_source_combo = self._combo(["sample list", "online Kafka"], "sample list")
        self.queue_source_combo.currentTextChanged.connect(self._update_source_visibility)
        form.addRow("Task source", self.queue_source_combo)
        self.sample_queue_widget = QtWidgets.QWidget()
        sample_queue_layout = QtWidgets.QVBoxLayout(self.sample_queue_widget)
        sample_queue_layout.setContentsMargins(0, 0, 0, 0)
        self.sample_queue_table = QtWidgets.QTableWidget(0, 1)
        self.sample_queue_table.setHorizontalHeaderLabels(["Sample name"])
        self.sample_queue_table.horizontalHeader().setStretchLastSection(True)
        self.sample_queue_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.sample_queue_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.sample_queue_table.itemChanged.connect(self.refresh_command)
        self.sample_queue_table.setMinimumHeight(92)
        self.sample_queue_table.setMaximumHeight(130)
        self.sample_queue_table.verticalHeader().setDefaultSectionSize(24)
        sample_queue_layout.addWidget(self.sample_queue_table)
        sample_queue_actions = QtWidgets.QHBoxLayout()
        add_sample_button = QtWidgets.QPushButton("Add")
        add_sample_button.clicked.connect(self._add_sample_queue_row)
        sample_queue_actions.addWidget(add_sample_button)
        remove_sample_button = QtWidgets.QPushButton("Remove")
        remove_sample_button.clicked.connect(self._remove_selected_sample_queue_rows)
        sample_queue_actions.addWidget(remove_sample_button)
        clear_sample_button = QtWidgets.QPushButton("Clear")
        clear_sample_button.clicked.connect(self._clear_sample_queue_rows)
        sample_queue_actions.addWidget(clear_sample_button)
        sample_queue_actions.addStretch(1)
        sample_queue_layout.addLayout(sample_queue_actions)
        form.addRow("Sample list", self.sample_queue_widget)
        default_queue = DEFAULT_OUTPUT_DIR / "measurement_done_queue.jsonl"
        self.measurement_queue_edit = self._line(str(default_queue))
        self.measurement_queue_edit.setReadOnly(True)
        self.measurement_queue_row = self._path_row(form, "Auto task file", self.measurement_queue_edit, "JSONL files (*.jsonl);;JSON files (*.json);;All files (*)")
        kafka_section, kafka_form = self._collapsible_form_group("Online Kafka connection", expanded=False)
        self.start_kafka_bridge_check = QtWidgets.QCheckBox("start Kafka bridge with reducer")
        self.start_kafka_bridge_check.stateChanged.connect(self._update_source_visibility)
        kafka_form.addRow("Kafka bridge", self.start_kafka_bridge_check)
        self.kafka_bootstrap_edit = self._line(DEFAULT_KAFKA_BOOTSTRAP)
        kafka_form.addRow("Bootstrap servers", self.kafka_bootstrap_edit)
        self.kafka_topic_edit = self._line(DEFAULT_KAFKA_TOPIC)
        kafka_form.addRow("Topic", self.kafka_topic_edit)
        self.kafka_group_id_edit = self._line(DEFAULT_KAFKA_GROUP_ID)
        kafka_form.addRow("Group ID", self.kafka_group_id_edit)
        form.addRow(kafka_section)
        self._add_main_section(box)

    def _build_paths_group(self) -> None:
        box = QtWidgets.QGroupBox("Paths")
        form = self._setup_form(QtWidgets.QFormLayout(box))
        self.watch_dir_edit = self._line(DEFAULT_WATCH_DIR)
        self.watch_dir_edit.editingFinished.connect(self._maybe_update_sample_name_from_watch_dir)
        self._path_row(form, "Beamtime date folder", self.watch_dir_edit, "dir")
        self.extracted_folder_edit = self._line(DEFAULT_EXTRACTED_FOLDER_NAME)
        form.addRow("Analysis folder", self.extracted_folder_edit)
        path_details, path_details_form = self._collapsible_form_group("Single-detector and replay paths", expanded=False)
        self.manifest_edit = self._line(DEFAULT_MANIFEST)
        self._path_row(path_details_form, "Manifest", self.manifest_edit, "CSV files (*.csv);;All files (*)")
        self.output_dir_edit = self._line(str(DEFAULT_OUTPUT_DIR))
        self.output_dir_row = self._path_row(path_details_form, "Output directory", self.output_dir_edit, "dir")
        self.analysis_h5_edit = self._line("")
        self.analysis_h5_row = self._path_row(path_details_form, "Analysis HDF5", self.analysis_h5_edit, "save_file")
        self.poni_edit = self._line(str(DEFAULT_PONI))
        self._path_row(path_details_form, "PONI", self.poni_edit, "PONI files (*.poni);;All files (*)")
        self.mask_edit = self._line(str(DEFAULT_MASK))
        self._path_row(path_details_form, "Mask", self.mask_edit, "Mask files (*.msk *.npy *.edf);;HDF5 files (*.h5 *.hdf5);;All files (*)")
        form.addRow(path_details)
        self._add_main_section(box)

        dual_box, dual_form = self._collapsible_form_group("Parallel detector calibration", expanded=False)
        self.dual_detector_box = dual_box
        self.saxs_watch_dir_edit = self._line(DEFAULT_PIL300K_WATCH_DIR)
        self.waxs_watch_dir_edit = self._line(DEFAULT_EIGER_WATCH_DIR)
        self.saxs_poni_edit = self._line(str(DEFAULT_SAXS_PONI))
        self._path_row(dual_form, "Pil300K PONI", self.saxs_poni_edit, "PONI files (*.poni);;All files (*)")
        self.saxs_mask_edit = self._line(str(DEFAULT_SAXS_MASK))
        self._path_row(dual_form, "Pil300K mask", self.saxs_mask_edit, "Mask files (*.msk *.npy *.edf);;HDF5 files (*.h5 *.hdf5);;All files (*)")
        self.saxs_monitor_key_edit = self._line("")
        dual_form.addRow("Pil300K monitor key", self.saxs_monitor_key_edit)
        self.waxs_poni_edit = self._line(str(DEFAULT_WAXS_PONI))
        self._path_row(dual_form, "Eig1M PONI", self.waxs_poni_edit, "PONI files (*.poni);;All files (*)")
        self.waxs_mask_edit = self._line(str(DEFAULT_WAXS_MASK))
        self._path_row(dual_form, "Eig1M mask", self.waxs_mask_edit, "Mask files (*.msk *.npy *.edf);;HDF5 files (*.h5 *.hdf5);;All files (*)")
        self.waxs_monitor_key_edit = self._line("")
        dual_form.addRow("Eig1M monitor key", self.waxs_monitor_key_edit)
        self._add_main_section(dual_box)

    def _build_reduction_group(self) -> None:
        box, form = self._collapsible_form_group("Reduction parameters", expanded=False)
        self.reduction_form = form
        self.dataset_path_edit = self._line("entry/data/data")
        form.addRow("Detector dataset path", self.dataset_path_edit)
        self.detector_combo = self._combo(["auto", "Eig1M", "Pil300K"], "auto")
        form.addRow("Detector override", self.detector_combo)
        self.monitor_key_edit = self._line("")
        form.addRow("Monitor key", self.monitor_key_edit)
        self.npt_spin = self._spin(1000, 1)
        form.addRow("q bins", self.npt_spin)
        self.jobs_spin = self._spin(1, 1, max(1, os.cpu_count() or 1))
        self.jobs_spin.valueChanged.connect(self._update_multicore_info)
        form.addRow("CPU cores", self.jobs_spin)
        self.multicore_info_label = QtWidgets.QLabel()
        self.multicore_info_label.setWordWrap(True)
        self.multicore_info_label.setStyleSheet("color: #555;")
        form.addRow("CPU use", self.multicore_info_label)
        self.analysis_write_interval_spin = self._spin(1, 1, 10000)
        form.addRow("H5 write every N groups", self.analysis_write_interval_spin)
        self.unit_edit = self._line("q_A^-1")
        form.addRow("q unit", self.unit_edit)
        self.outlier_spin = self._double(3.5, 0.0, 100.0)
        form.addRow("Outlier z max", self.outlier_spin)
        self.delta_energy_spin = self._double(0.001, 0.0, 100.0)
        form.addRow("Delta energy %", self.delta_energy_spin)
        self._add_main_section(box)

    def _build_sequence_group(self) -> None:
        box = QtWidgets.QGroupBox("Acquisition Sequence")
        form = self._setup_form(QtWidgets.QFormLayout(box))
        self.sequence_form = form
        self.num_frames_spin = self._spin(1, 1)
        form.addRow("Frames per group", self.num_frames_spin)
        self.num_groups_spin = self._spin(1, 1)
        form.addRow("Groups per energy", self.num_groups_spin)
        self.num_energies_spin = self._spin(1, 0)
        self.num_energies_spin.valueChanged.connect(self._update_multicore_info)
        form.addRow("Number of energies", self.num_energies_spin)
        self.auto_num_groups_check = QtWidgets.QCheckBox("infer from HDF5 file count")
        self.auto_num_groups_check.setChecked(False)
        self.auto_num_groups_check.stateChanged.connect(self._update_source_visibility)
        form.addRow("Auto groups", self.auto_num_groups_check)
        timing_section, timing_form = self._collapsible_form_group("Timing and replay details", expanded=False)
        self.pattern_edit = self._line("*.h5")
        timing_form.addRow("File pattern", self.pattern_edit)
        self.limit_energies_spin = self._spin(0, 0)
        timing_form.addRow("Replay limit energies", self.limit_energies_spin)
        self.limit_frames_spin = self._spin(0, 0)
        timing_form.addRow("Replay limit frames/group", self.limit_frames_spin)
        self.poll_spin = self._double(2.0, 0.1, 3600.0)
        timing_form.addRow("Poll seconds", self.poll_spin)
        self.settle_spin = self._double(2.0, 0.0, 3600.0)
        timing_form.addRow("Settle seconds", self.settle_spin)
        form.addRow(timing_section)
        self._add_main_section(box)

    def _build_asaxs_group(self) -> None:
        box, form = self._collapsible_form_group("ASAXS roles and corrections", expanded=False)
        self.asaxs_options_box = box
        self.gc_group_spin = self._spin(1, 0)
        form.addRow("GC group", self.gc_group_spin)
        self.air_group_spin = self._spin(2, 0)
        form.addRow("Air group", self.air_group_spin)
        self.empty_group_spin = self._spin(3, 0)
        form.addRow("Empty group", self.empty_group_spin)
        self.water_group_spin = self._spin(0, 0)
        self.sample_group_spin = self._spin(0, 0)
        self.asaxs_pairs_widget = QtWidgets.QWidget()
        asaxs_pairs_layout = QtWidgets.QVBoxLayout(self.asaxs_pairs_widget)
        asaxs_pairs_layout.setContentsMargins(0, 0, 0, 0)
        self.asaxs_pairs_table = QtWidgets.QTableWidget(0, 3)
        self.asaxs_pairs_table.setHorizontalHeaderLabels(["Output name", "Sample group", "Solvent group"])
        self.asaxs_pairs_table.horizontalHeader().setStretchLastSection(True)
        self.asaxs_pairs_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.asaxs_pairs_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.asaxs_pairs_table.itemChanged.connect(self.refresh_command)
        self.asaxs_pairs_table.setMinimumHeight(112)
        self.asaxs_pairs_table.setMaximumHeight(160)
        self.asaxs_pairs_table.verticalHeader().setDefaultSectionSize(24)
        self.asaxs_pairs_table.setColumnWidth(0, 150)
        self.asaxs_pairs_table.setColumnWidth(1, 92)
        self.asaxs_pairs_table.setColumnWidth(2, 102)
        asaxs_pairs_layout.addWidget(self.asaxs_pairs_table)
        asaxs_pair_actions = QtWidgets.QHBoxLayout()
        add_pair_button = QtWidgets.QPushButton("Add")
        add_pair_button.clicked.connect(self._add_asaxs_pair_row)
        asaxs_pair_actions.addWidget(add_pair_button)
        remove_pair_button = QtWidgets.QPushButton("Remove")
        remove_pair_button.clicked.connect(self._remove_selected_asaxs_pair_rows)
        asaxs_pair_actions.addWidget(remove_pair_button)
        clear_pair_button = QtWidgets.QPushButton("Clear")
        clear_pair_button.clicked.connect(self._clear_asaxs_pair_rows)
        asaxs_pair_actions.addWidget(clear_pair_button)
        asaxs_pair_actions.addStretch(1)
        asaxs_pairs_layout.addLayout(asaxs_pair_actions)
        form.addRow("Sample/solvent outputs", self.asaxs_pairs_widget)
        self.gc_ref_edit = self._line("")
        self._path_row(form, "GC reference file", self.gc_ref_edit, "Data files (*.dat *.txt *.csv);;All files (*)")
        self.gc_q_min_spin = self._double(0.03, 0.0, 1000.0)
        self.gc_q_max_spin = self._double(0.20, 0.0, 1000.0)
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.gc_q_min_spin)
        layout.addWidget(self.gc_q_max_spin)
        form.addRow("GC q range", row)
        self.capillary_thickness_spin = self._double(0.15, 0.0, 1e9)
        form.addRow("Sample/tube thickness", self.capillary_thickness_spin)
        self.gc_thickness_spin = self._double(0.1055, 0.0, 1e9)
        form.addRow("GC standard thickness", self.gc_thickness_spin)
        self.subtract_fluorescence_check = QtWidgets.QCheckBox("subtract fluorescence")
        self.subtract_fluorescence_check.stateChanged.connect(self.refresh_command)
        form.addRow("Fluorescence", self.subtract_fluorescence_check)
        self.fluorescence_reference_combo = self._combo(["latest", "each"], "latest")
        form.addRow("Fluorescence reference", self.fluorescence_reference_combo)
        self.fluorescence_level_spin = self._double(0.0, -1e12, 1e12)
        form.addRow("Fluorescence fixed level", self.fluorescence_level_spin)
        self.fluorescence_q_min_spin = self._double(0.8, 0.0, 1000.0)
        self.fluorescence_q_max_spin = self._double(1.0, 0.0, 1000.0)
        fluorescence_q_row = QtWidgets.QWidget()
        fluorescence_q_layout = QtWidgets.QHBoxLayout(fluorescence_q_row)
        fluorescence_q_layout.setContentsMargins(0, 0, 0, 0)
        fluorescence_q_layout.addWidget(self.fluorescence_q_min_spin)
        fluorescence_q_layout.addWidget(self.fluorescence_q_max_spin)
        form.addRow("Fluorescence q range", fluorescence_q_row)
        self._add_main_section(box)

    def _analysis_mode_changed(self, _text: str) -> None:
        if self.analysis_mode_combo.currentText() == "saxs" and not self._settings_loaded:
            self._set_spin_without_signal(self.num_energies_spin, 1)
            self._set_spin_without_signal(self.num_groups_spin, 1)
            self._set_spin_without_signal(self.num_frames_spin, 1)
        self._update_source_visibility()

    def _update_source_visibility(self) -> None:
        if self._dual_detector_enabled() and self.source_mode_combo.currentText() != "watch folder":
            self.source_mode_combo.blockSignals(True)
            self.source_mode_combo.setCurrentText("watch folder")
            self.source_mode_combo.blockSignals(False)
        manifest_mode = self.source_mode_combo.currentText() == "manifest replay"
        dual_mode = self._dual_detector_enabled()
        self.use_measurement_queue_check.setEnabled(not manifest_mode)
        queue_mode = self.use_measurement_queue_check.isChecked() and not manifest_mode
        online_mode = queue_mode and self.queue_source_combo.currentText() == "online Kafka"
        sample_list_mode = queue_mode and self.queue_source_combo.currentText() == "sample list"
        effective_once_mode = self.once_check.isChecked() or sample_list_mode
        kafka_fast_mode = online_mode
        if online_mode and not self.start_kafka_bridge_check.isChecked():
            self.start_kafka_bridge_check.blockSignals(True)
            self.start_kafka_bridge_check.setChecked(True)
            self.start_kafka_bridge_check.blockSignals(False)
        bridge_mode = online_mode and self.start_kafka_bridge_check.isChecked()
        if not online_mode and self.start_kafka_bridge_check.isChecked():
            self.start_kafka_bridge_check.blockSignals(True)
            self.start_kafka_bridge_check.setChecked(False)
            self.start_kafka_bridge_check.blockSignals(False)
        saxs_mode = self.analysis_mode_combo.currentText() == "saxs"
        self.manifest_edit.setEnabled(manifest_mode)
        self.watch_dir_edit.setEnabled(not manifest_mode)
        self.sample_name_edit.setEnabled(not queue_mode)
        self.sample_queue_widget.setVisible(sample_list_mode)
        self.queue_source_combo.setEnabled(queue_mode)
        self.measurement_queue_edit.setEnabled(False)
        self._set_form_row_visible(self.measurement_queue_row, False)
        self._set_form_row_visible(self.output_dir_row, not queue_mode)
        self._set_form_row_visible(self.analysis_h5_row, not queue_mode)
        self.start_kafka_bridge_check.setEnabled(online_mode)
        self.kafka_bootstrap_edit.setEnabled(bridge_mode)
        self.kafka_topic_edit.setEnabled(bridge_mode)
        self.kafka_group_id_edit.setEnabled(bridge_mode)
        self.detector_combo.setEnabled(not dual_mode)
        self.detector_combo.setVisible(not dual_mode)
        detector_label = self.reduction_form.labelForField(self.detector_combo)
        if detector_label is not None:
            detector_label.setVisible(not dual_mode)
        self.monitor_key_edit.setEnabled(not dual_mode)
        self.poni_edit.setEnabled(not dual_mode)
        self.mask_edit.setEnabled(not dual_mode)
        self.dual_detector_box.setVisible(dual_mode)
        self.once_check.setEnabled(not manifest_mode and not sample_list_mode)
        self.poll_spin.setEnabled(not manifest_mode and not effective_once_mode and not kafka_fast_mode)
        self.settle_spin.setEnabled(not manifest_mode and not effective_once_mode and not kafka_fast_mode)
        self.num_groups_spin.setEnabled(not self.auto_num_groups_check.isChecked())
        frame_label = self.sequence_form.labelForField(self.num_frames_spin)
        if frame_label is not None:
            frame_label.setText("Frames per group")
        self.asaxs_options_box.setVisible(not saxs_mode)
        self._update_multicore_info()
        self.refresh_command()

    def _update_multicore_info(self) -> None:
        """Show what the CPU-core setting currently means for this GUI run."""
        if not hasattr(self, "multicore_info_label"):
            return
        requested = self.jobs_spin.value()
        cpu_max = max(1, os.cpu_count() or 1)
        detector_processes = 2 if self._dual_detector_enabled() else 1
        if self.source_mode_combo.currentText() == "manifest replay":
            mode_text = "V5 replay currently reduces frames serially in each detector process"
        elif self.use_measurement_queue_check.isChecked():
            mode_text = "queue/watch mode reduces arriving frames serially in each detector process"
        else:
            mode_text = "watch mode reduces arriving frames serially in each detector process"
        extra = ""
        if requested > 1:
            extra = " The extra cores are not used for live frame integration yet."
        self.multicore_info_label.setText(
            f"Requested {requested}/{cpu_max} cores. Effective now: {detector_processes} detector process(es), "
            f"1 integration worker per process; {mode_text}.{extra}"
        )

    def _set_form_row_visible(self, row: QtWidgets.QWidget, visible: bool) -> None:
        row.setVisible(visible)
        parent = row.parentWidget()
        if parent is None:
            return
        layout = parent.layout()
        if isinstance(layout, QtWidgets.QFormLayout):
            label = layout.labelForField(row)
            if label is not None:
                label.setVisible(visible)

    def _set_form_field_visible(self, form: QtWidgets.QFormLayout, field: QtWidgets.QWidget, visible: bool) -> None:
        if hasattr(form, "setRowVisible"):
            form.setRowVisible(field, visible)
            return
        field.setVisible(visible)
        label = form.labelForField(field)
        if label is not None:
            label.setVisible(visible)

    def _dual_detector_enabled(self) -> bool:
        return self.detector_run_mode_combo.currentText() in {"Pil300K + Eig1M", "SAXS + WAXS"}

    def _maybe_update_sample_name_from_watch_dir(self) -> None:
        current = self.sample_name_edit.text().strip()
        if current and current != DEFAULT_SAMPLE_NAME:
            return
        watch_dir = Path(self.watch_dir_edit.text())
        detector_names = {"eig1m", "pil300k", "saxs", "waxs"}
        if watch_dir.name.lower() in detector_names and watch_dir.parent.name:
            self.sample_name_edit.setText(watch_dir.parent.name)
        elif watch_dir.name:
            self.sample_name_edit.setText(watch_dir.name)

    def _detector_watch_dir(self, detector: str) -> Path:
        """Derive a detector-specific input folder from the main acquisition root."""
        root = self._raw_sample_root()
        detector_names = {"pil300k", "eig1m", "saxs", "waxs"}
        if root.name.lower() in detector_names:
            if root.name.lower() == detector.lower():
                return root
            return root.parent / detector
        return root / detector

    def _safe_sample_name(self) -> str:
        sample = self.sample_name_edit.text().strip() or DEFAULT_SAMPLE_NAME
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample).strip("._") or "sample"

    def _safe_name(self, value: str, fallback: str = "sample") -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or fallback

    def _user_root(self) -> Path:
        return Path(self.watch_dir_edit.text()).expanduser()

    def _raw_data_root(self) -> Path:
        return self._user_root()

    def _analysis_root(self) -> Path:
        folder = self.extracted_folder_edit.text().strip() or DEFAULT_EXTRACTED_FOLDER_NAME
        return self._user_root() / folder

    def _raw_sample_root(self) -> Path:
        return self._resolve_raw_sample_root(self.sample_name_edit.text().strip() or self._safe_sample_name())

    def _resolve_raw_sample_root(self, sample_name: str) -> Path:
        """Find a sample folder below the selected beamtime/project root.

        Most beamtime folders are ``root/<sample>/<detector>``, but some
        projects add another organizational layer under the selected main
        folder. The GUI should accept the same sample-list name in both cases
        and find the raw data folder without ever treating ``Extracted`` as raw
        input.
        """
        root = self._raw_data_root().expanduser()
        sample_text = sample_name.strip()
        safe_sample = self._safe_name(sample_text)
        target_names = {sample_text, safe_sample}
        target_names = {name for name in target_names if name}
        if root.name in target_names:
            return root
        for name in target_names:
            direct = root / name
            if direct.exists():
                return direct
        if root.exists():
            extracted_name = (self.extracted_folder_edit.text().strip() or DEFAULT_EXTRACTED_FOLDER_NAME).lower()
            matches: list[Path] = []
            for name in target_names:
                for path in root.rglob(name):
                    if not path.is_dir():
                        continue
                    relative_parts = {part.lower() for part in path.relative_to(root).parts}
                    if extracted_name in relative_parts or "_live_status" in relative_parts:
                        continue
                    matches.append(path)
            if matches:
                matches.sort(key=self._sample_folder_score, reverse=True)
                return matches[0]
        return root / safe_sample

    def _sample_folder_score(self, path: Path) -> tuple[int, int]:
        """Rank likely raw sample folders ahead of unrelated same-name folders."""
        detector_names = {"pil300k", "eig1m", "saxs", "waxs", "spds", "wpds"}
        try:
            child_names = {child.name.lower() for child in path.iterdir() if child.is_dir()}
        except OSError:
            child_names = set()
        detector_score = len(child_names & detector_names)
        try:
            h5_score = sum(1 for _ in path.rglob("*.h5"))
        except OSError:
            h5_score = 0
        return detector_score, h5_score

    def _analysis_sample_root(self) -> Path:
        return self._analysis_root_for_sample(self.sample_name_edit.text().strip() or self._safe_sample_name())

    def _relative_sample_path(self, sample_name: str) -> Path:
        """Return the sample path relative to the selected main folder."""
        raw_root = self._raw_data_root().expanduser().resolve()
        sample_root = self._resolve_raw_sample_root(sample_name).expanduser().resolve()
        try:
            relative = sample_root.relative_to(raw_root)
        except ValueError:
            relative = Path(self._safe_name(sample_name))
        extracted_name = (self.extracted_folder_edit.text().strip() or DEFAULT_EXTRACTED_FOLDER_NAME).lower()
        if any(part.lower() == extracted_name for part in relative.parts):
            return Path(self._safe_name(sample_name))
        return relative

    def _analysis_root_for_sample(self, sample_name: str) -> Path:
        """Mirror the raw sample's relative folder under the Extracted root."""
        return self._analysis_root() / self._relative_sample_path(sample_name)

    def _queue_mode_enabled(self) -> bool:
        return self.use_measurement_queue_check.isChecked() and self.source_mode_combo.currentText() != "manifest replay"

    def _sample_list_mode_enabled(self) -> bool:
        return self._queue_mode_enabled() and self.queue_source_combo.currentText() == "sample list"

    def _online_kafka_mode_enabled(self) -> bool:
        """Return True when the GUI-owned queue is fed by the Kafka bridge."""
        return self._queue_mode_enabled() and self.queue_source_combo.currentText() == "online Kafka"

    def _fresh_session_mode_enabled(self) -> bool:
        """Return True when startup should discard stale live-session records."""
        return self.restart_behavior_combo.currentText() == "restart" or self._online_kafka_mode_enabled()

    def _queue_primary_sample_name(self) -> str:
        """Return the visible first queued sample for local sample-list mode."""
        names = self._sample_names_for_queue()
        return names[0] if names else "queued_samples"

    def _queue_primary_sample_root(self) -> Path:
        return self._analysis_root_for_sample(self._queue_primary_sample_name())

    def _control_output_root(self) -> Path:
        """Return the startup output root used before a queued job is active."""
        if self._sample_list_mode_enabled():
            return self._queue_primary_sample_root()
        if self._queue_mode_enabled():
            return self._analysis_root()
        return self._analysis_sample_root()

    def _combined_analysis_h5_path_for_sample(self, sample_name: str) -> Path:
        """Return the sample-level HDF5 file that stores stitched detector output."""
        safe_sample = self._safe_name(sample_name, "analysis")
        return self._analysis_root_for_sample(sample_name).expanduser().resolve() / f"{safe_sample}_analysis.h5"

    def _combined_analysis_h5_paths_for_current_run(self) -> list[Path]:
        """Return every combined HDF5 file the current GUI run may update."""
        if self._sample_list_mode_enabled():
            names = self._sample_names_for_queue()
            return [self._combined_analysis_h5_path_for_sample(name) for name in names]
        return [self._combined_analysis_h5_path()]

    def _detector_analysis_h5_paths_for_current_run(self, detector: str) -> list[Path] | None:
        """Return detector-specific analysis HDF5 files expected for this run."""
        if self._queue_mode_enabled() and not self._sample_list_mode_enabled():
            return None
        if self._sample_list_mode_enabled():
            samples = self._sample_names_for_queue()
        else:
            samples = [self.sample_name_edit.text().strip() or DEFAULT_SAMPLE_NAME]
        paths: list[Path] = []
        for sample in samples:
            safe_sample = self._safe_name(sample, "analysis")
            root = self._analysis_root_for_sample(sample).expanduser().resolve()
            paths.append(root / detector / f"{safe_sample}_{detector}_analysis.h5")
        return paths

    def _analysis_h5_paths_to_clear_for_restart(self) -> list[Path]:
        """Return analysis HDF5 files that belong to this GUI-started run."""
        paths = list(self._combined_analysis_h5_paths_for_current_run())
        if self._sample_list_mode_enabled():
            for detector in self._detectors_for_manual_queue():
                detector_paths = self._detector_analysis_h5_paths_for_current_run(detector)
                if detector_paths:
                    paths.extend(detector_paths)
        return sorted({path.expanduser().resolve() for path in paths}, key=str)

    def _clear_analysis_h5_paths_for_restart(self) -> bool:
        """Delete old analysis records before the viewer captures its baseline."""
        for path in self._analysis_h5_paths_to_clear_for_restart():
            try:
                self._remove_file_if_present(path)
            except OSError as exc:
                QtWidgets.QMessageBox.warning(self, "Could not clear old analysis HDF5", f"{path}\n\n{exc}")
                return False
        return True

    def _stitched_viewer_path(self) -> Path:
        """Return the path the stitched-curve viewer should scan."""
        if self._queue_mode_enabled():
            return self._analysis_root().expanduser().resolve()
        return self._combined_analysis_h5_path()

    def _append_sample_queue_row(self, sample_name: str = "") -> None:
        row = self.sample_queue_table.rowCount()
        self.sample_queue_table.insertRow(row)
        item = QtWidgets.QTableWidgetItem(sample_name)
        self.sample_queue_table.setItem(row, 0, item)

    def _add_sample_queue_row(self) -> None:
        self._append_sample_queue_row("")
        self.sample_queue_table.editItem(self.sample_queue_table.item(self.sample_queue_table.rowCount() - 1, 0))

    def _remove_selected_sample_queue_rows(self) -> None:
        rows = sorted({index.row() for index in self.sample_queue_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.sample_queue_table.removeRow(row)
        self.refresh_command()

    def _clear_sample_queue_rows(self) -> None:
        self.sample_queue_table.setRowCount(0)
        self.refresh_command()

    def _append_asaxs_pair_row(self, output_name: str = "", sample_group: str = "", solvent_group: str = "") -> None:
        row = self.asaxs_pairs_table.rowCount()
        self.asaxs_pairs_table.insertRow(row)
        for column, value in enumerate([output_name, sample_group, solvent_group]):
            self.asaxs_pairs_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))

    def _add_asaxs_pair_row(self) -> None:
        self._append_asaxs_pair_row("", "", "")
        self.asaxs_pairs_table.editItem(self.asaxs_pairs_table.item(self.asaxs_pairs_table.rowCount() - 1, 0))

    def _remove_selected_asaxs_pair_rows(self) -> None:
        rows = sorted({index.row() for index in self.asaxs_pairs_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.asaxs_pairs_table.removeRow(row)
        self.refresh_command()

    def _clear_asaxs_pair_rows(self) -> None:
        self.asaxs_pairs_table.setRowCount(0)
        self.refresh_command()

    def _sample_names_from_table(self) -> list[str]:
        self._commit_sample_queue_editor()
        names = []
        for row in range(self.sample_queue_table.rowCount()):
            item = self.sample_queue_table.item(row, 0)
            text = item.text().strip() if item is not None else ""
            if text:
                names.append(text)
        return names

    def _sample_names_for_queue(self) -> list[str]:
        """Return manually queued sample names from the table."""
        return self._sample_names_from_table()

    def _commit_sample_queue_editor(self) -> None:
        """Capture text from an active sample-list editor before Start/Preflight.

        When the user types a sample name and immediately clicks Start, the
        temporary cell editor may still own the newest text. Pulling that text
        into the table item here keeps the queue from using stale or empty row
        values.
        """
        focus = QtWidgets.QApplication.focusWidget()
        if focus is None or not self.sample_queue_table.isAncestorOf(focus):
            return
        index = self.sample_queue_table.currentIndex()
        if not index.isValid():
            return
        text_getter = getattr(focus, "text", None)
        if not callable(text_getter):
            return
        item = self.sample_queue_table.item(index.row(), index.column())
        if item is None:
            item = QtWidgets.QTableWidgetItem("")
            self.sample_queue_table.setItem(index.row(), index.column(), item)
        item.setText(str(text_getter()).strip())

    def _commit_asaxs_pair_editor(self) -> None:
        """Capture active ASAXS pair table edits before building commands."""
        focus = QtWidgets.QApplication.focusWidget()
        if focus is None or not self.asaxs_pairs_table.isAncestorOf(focus):
            return
        index = self.asaxs_pairs_table.currentIndex()
        if not index.isValid():
            return
        text_getter = getattr(focus, "text", None)
        if not callable(text_getter):
            return
        item = self.asaxs_pairs_table.item(index.row(), index.column())
        if item is None:
            item = QtWidgets.QTableWidgetItem("")
            self.asaxs_pairs_table.setItem(index.row(), index.column(), item)
        item.setText(str(text_getter()).strip())

    def _detectors_for_manual_queue(self) -> list[str]:
        if self._dual_detector_enabled():
            return ["Pil300K", "Eig1M"]
        detector = self.detector_combo.currentText()
        return [detector if detector != "auto" else "Pil300K"]

    def _preflight_scan_dir_for(self, sample_name: str, detector: str | None = None) -> Path:
        if self._sample_list_mode_enabled():
            sample_root = self._resolve_raw_sample_root(sample_name)
            if detector:
                return sample_root / detector
            return sample_root
        if self._dual_detector_enabled() and detector:
            return self._raw_sample_root() / detector
        if detector:
            return self._raw_sample_root() / detector
        return self._raw_sample_root()

    def _count_h5_files(self, data_dir: Path) -> int:
        pattern = self.pattern_edit.text().strip() or "*.h5"
        if not data_dir.exists():
            return 0
        iterator = data_dir.rglob(pattern) if self.analysis_mode_combo.currentText() == "saxs" and not self._dual_detector_enabled() else data_dir.glob(pattern)
        return sum(1 for path in iterator if path.is_file())

    def _preflight_rows(self) -> list[PreflightRow]:
        if self.source_mode_combo.currentText() != "watch folder":
            return []
        if self._queue_mode_enabled() and self.queue_source_combo.currentText() == "online Kafka":
            return []
        energies = self.num_energies_spin.value()
        frames = self.num_frames_spin.value()
        if energies <= 0 or frames <= 0:
            return []
        samples = self._sample_names_for_queue() if self._sample_list_mode_enabled() else [self._safe_sample_name()]
        detectors = self._detectors_for_manual_queue() if self._sample_list_mode_enabled() else (
            ["Pil300K", "Eig1M"] if self._dual_detector_enabled() else [None]
        )
        rows: list[PreflightRow] = []
        for sample in samples:
            for detector in detectors:
                data_dir = self._preflight_scan_dir_for(sample, detector)
                count = self._count_h5_files(data_dir)
                denominator = energies * frames
                inferred = count // denominator if denominator > 0 and count > 0 and count % denominator == 0 else None
                warning = None
                if not data_dir.exists():
                    warning = "folder missing"
                elif count == 0:
                    warning = "no H5 files"
                elif self.auto_num_groups_check.isChecked() and inferred is None:
                    remainder = count % denominator if denominator > 0 else count
                    warning = (
                        f"{count} files not divisible by energies*frames "
                        f"({energies}*{frames}={denominator}, remainder {remainder}); "
                        "frame number may be wrong"
                    )
                elif (
                    self._sample_list_mode_enabled()
                    and not self.auto_num_groups_check.isChecked()
                    and count > 0
                    and energies > 0
                    and frames > 0
                    and self.num_groups_spin.value() > 0
                ):
                    block_size = energies * self.num_groups_spin.value() * frames
                    complete_blocks = count // block_size if block_size > 0 else 0
                    remainder = count % block_size if block_size > 0 else count
                    if complete_blocks == 0:
                        warning = f"incomplete: needs {block_size} files per set; no complete set will be reduced"
                    elif remainder:
                        warning = (
                            f"incomplete tail: will reduce {complete_blocks * block_size} files "
                            f"from {complete_blocks} complete set(s), ignore {remainder}"
                        )
                rows.append(
                    PreflightRow(
                        sample=sample,
                        detector=detector or self.detector_combo.currentText(),
                        data_dir=data_dir,
                        file_count=count,
                        num_energies=energies,
                        num_frames=frames,
                        inferred_groups=inferred if self.auto_num_groups_check.isChecked() else self.num_groups_spin.value(),
                        warning=warning,
                    )
                )
        return rows

    def _preflight_expected_frames(self, detector: str | None = None) -> int | None:
        rows = self._preflight_rows()
        if detector is not None:
            rows = [row for row in rows if row.detector == detector]
        if rows and all(row.warning is None for row in rows):
            return sum(row.file_count for row in rows)
        return None

    def _preflight_summary_text(self) -> str:
        rows = self._preflight_rows()
        if not rows:
            return "Preflight: waiting for online Kafka jobs, or no local folders to inspect."
        total = sum(row.file_count for row in rows)
        lines = [f"Preflight: {len(rows)} detector job(s), {total} H5 frame(s) visible."]
        for row in rows:
            groups = row.inferred_groups if row.inferred_groups is not None else "?"
            suffix = f"  [{row.warning}]" if row.warning else ""
            lines.append(
                f"{row.sample} / {row.detector}: {row.file_count} files, "
                f"energies={row.num_energies}, groups={groups}, frames/avg={row.num_frames}{suffix}"
            )
        return "\n".join(lines)

    def _confirm_preflight_before_start(self) -> bool:
        text = self._preflight_summary_text()
        rows = self._preflight_rows()
        if not rows:
            return True
        has_warning = any(row.warning for row in rows)
        icon = QtWidgets.QMessageBox.Warning if has_warning else QtWidgets.QMessageBox.Information
        box = QtWidgets.QMessageBox(icon, "Preflight check", text, QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok, self)
        box.setDefaultButton(QtWidgets.QMessageBox.Ok)
        box.setInformativeText("Start reduction with these numbers?")
        return box.exec_() == QtWidgets.QMessageBox.Ok

    def _queue_single_sample_jobs(self) -> None:
        """Rebuild the local JSONL task file from the visible sample table."""
        if (
            not self.use_measurement_queue_check.isChecked()
            or self.queue_source_combo.currentText() != "sample list"
            or self.source_mode_combo.currentText() == "manifest replay"
        ):
            return
        queue_path = self._prepare_measurement_queue_path(clear=True, touch=False)
        detectors = self._detectors_for_manual_queue()
        sample_names = self._sample_names_for_queue()
        if not sample_names:
            raise ValueError("Add at least one sample name to the Sample list before starting.")
        for sample_name in sample_names:
            raw_sample_root = self._resolve_raw_sample_root(sample_name)
            analysis_sample_root = self._analysis_root_for_sample(sample_name)
            for detector in detectors:
                detector_analysis_mode = "saxs" if self._dual_detector_enabled() else self.analysis_mode_combo.currentText()
                append_measurement_done_message(
                    queue_path,
                    sample_name=sample_name,
                    detector=detector,
                    analysis_mode=detector_analysis_mode,
                    measurement_type="manual_single_sample",
                    data_dir=raw_sample_root / detector,
                    output_dir=analysis_sample_root / detector,
                )

    def _measurement_queue_path(self) -> Path:
        """Return the local JSONL queue used between Kafka bridge and reducer."""
        if self._queue_mode_enabled():
            return self._control_output_root().expanduser().resolve() / "measurement_done_queue.jsonl"
        text = self.measurement_queue_edit.text().strip()
        if text:
            return Path(text).expanduser().resolve()
        return self._control_output_root().expanduser().resolve() / "measurement_done_queue.jsonl"

    def _prepare_measurement_queue_path(self, *, clear: bool = False, touch: bool = True) -> Path:
        """Create the internal task JSONL path used by GUI-owned queue modes."""
        queue_path = self._measurement_queue_path()
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        if clear:
            self._remove_file_if_present(queue_path)
        if touch and not queue_path.exists():
            queue_path.touch()
        return queue_path

    def _remove_file_if_present(self, path: Path) -> bool:
        """Remove a stale output sidecar, tolerating network-filesystem races."""
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def _append_arg(self, args: list[str], name: str, value: str | int | float | None) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text:
            args.extend([name, text])

    def command_args(self) -> list[str]:
        args = [str(PROJECT_DIR / "scripts" / "run_reducer.py")]
        if self.source_mode_combo.currentText() == "manifest replay":
            self._append_arg(args, "--manifest", self.manifest_edit.text())
            if self.limit_energies_spin.value() > 0:
                self._append_arg(args, "--limit-energies", self.limit_energies_spin.value())
            if self.limit_frames_spin.value() > 0:
                self._append_arg(args, "--limit-frames-per-group", self.limit_frames_spin.value())
        else:
            if self.use_measurement_queue_check.isChecked():
                self._append_arg(args, "--measurement-done-queue", self._measurement_queue_path())
                args.append("--continuous-queue")
                if self.queue_source_combo.currentText() == "sample list":
                    args.append("--stop-when-queue-drained")
            else:
                self._append_arg(args, "--watch-dir", self._raw_sample_root())
            self._append_arg(args, "--pattern", self.pattern_edit.text())
            if self.num_energies_spin.value() > 0:
                self._append_arg(args, "--num-energies", self.num_energies_spin.value())
            if self.auto_num_groups_check.isChecked():
                args.append("--auto-num-groups")
            else:
                self._append_arg(args, "--num-groups", self.num_groups_spin.value())
            self._append_arg(args, "--num-frames", self.num_frames_spin.value())
            if self.once_check.isChecked() or self._sample_list_mode_enabled():
                args.append("--once")
            elif self._online_kafka_mode_enabled():
                # Kafka measurement_done messages are the finished-file signal.
                # Keep a short queue poll for responsiveness, but do not wait
                # again for per-file size settling after the message arrives.
                self._append_arg(args, "--poll-seconds", 0.25)
                self._append_arg(args, "--settle-seconds", 0.0)
            else:
                self._append_arg(args, "--poll-seconds", self.poll_spin.value())
                self._append_arg(args, "--settle-seconds", self.settle_spin.value())

        live_mode = self.source_mode_combo.currentText() != "manifest replay"
        self._append_arg(args, "--output-dir", self._control_output_root() if live_mode else self.output_dir_edit.text())
        sample_arg = self.sample_name_edit.text()
        if self._sample_list_mode_enabled():
            sample_arg = self._queue_primary_sample_name()
        self._append_arg(args, "--sample-name", sample_arg)
        if not self.use_measurement_queue_check.isChecked():
            self._append_arg(args, "--analysis-h5", self.analysis_h5_edit.text())
        self._append_arg(args, "--analysis-mode", self.analysis_mode_combo.currentText())
        if self.write_text_output_check.isChecked():
            args.append("--write-text-output")
        if self._fresh_session_mode_enabled():
            args.append("--restart")
        self._append_arg(args, "--poni", self.poni_edit.text())
        self._append_arg(args, "--mask", self.mask_edit.text())
        self._append_arg(args, "--dataset-path", self.dataset_path_edit.text())
        self._append_arg(args, "--npt", self.npt_spin.value())
        self._append_arg(args, "--jobs", self.jobs_spin.value())
        self._append_arg(args, "--analysis-write-interval-groups", self.analysis_write_interval_spin.value())
        args.append("--quiet")
        self._append_arg(args, "--unit", self.unit_edit.text())
        self._append_arg(args, "--detector", self.detector_combo.currentText())
        self._append_arg(args, "--monitor-key", self.monitor_key_edit.text())
        self._append_arg(args, "--delta-energy-percent", self.delta_energy_spin.value())
        self._append_arg(args, "--outlier-zmax", self.outlier_spin.value())

        if self.analysis_mode_combo.currentText() == "asaxs":
            self._append_optional_group(args, "--gc-group", self.gc_group_spin.value())
            self._append_optional_group(args, "--air-group", self.air_group_spin.value())
            self._append_optional_group(args, "--empty-group", self.empty_group_spin.value())
            for pair in self._asaxs_pairs_from_table():
                args.extend(["--asaxs-pair", pair])
            self._append_arg(args, "--gc-reference-file", self.gc_ref_edit.text())
            args.extend(["--gc-q-range", str(self.gc_q_min_spin.value()), str(self.gc_q_max_spin.value())])
            if self.capillary_thickness_spin.value() > 0 or self.gc_thickness_spin.value() > 0:
                self._append_arg(args, "--capillary-thickness", self.capillary_thickness_spin.value())
                self._append_arg(args, "--gc-thickness", self.gc_thickness_spin.value())
            if self.subtract_fluorescence_check.isChecked():
                args.append("--subtract-fluorescence")
                self._append_arg(args, "--fluorescence-reference", self.fluorescence_reference_combo.currentText())
                if self.fluorescence_level_spin.value() != 0:
                    self._append_arg(args, "--fluorescence-level", self.fluorescence_level_spin.value())
                args.extend(
                    [
                        "--fluorescence-q-range",
                        str(self.fluorescence_q_min_spin.value()),
                        str(self.fluorescence_q_max_spin.value()),
                    ]
                )
        return args

    def _asaxs_pairs_from_table(self) -> list[str]:
        """Read GUI sample/solvent output rows as NAME:SAMPLE:SOLVENT."""
        self._commit_asaxs_pair_editor()
        pairs: list[str] = []
        for row in range(self.asaxs_pairs_table.rowCount()):
            values = []
            for column in range(3):
                item = self.asaxs_pairs_table.item(row, column)
                values.append(item.text().strip() if item is not None else "")
            name, sample, solvent = values
            if not name and not sample and not solvent:
                continue
            if not name or not sample or not solvent:
                continue
            pairs.append(f"{name}:{sample}:{solvent}")
        return pairs

    def detector_command_args(self, role: str) -> list[str]:
        """Build one reducer command for a parallel detector job."""
        role = role.lower()
        if role not in {"pil300k", "eig1m"}:
            raise ValueError(f"Unknown detector role: {role}")
        args = self.command_args()
        for flag, has_value in [
            ("--manifest", True),
            ("--watch-dir", True),
            ("--measurement-done-queue", True),
            ("--output-dir", True),
            ("--sample-name", True),
            ("--analysis-h5", True),
            ("--analysis-mode", True),
            ("--detector", True),
            ("--monitor-key", True),
            ("--poni", True),
            ("--mask", True),
            ("--gc-group", True),
            ("--air-group", True),
            ("--empty-group", True),
            ("--water-group", True),
            ("--sample-group", True),
            ("--gc-reference-file", True),
            ("--gc-q-range", 2),
            ("--capillary-thickness", True),
            ("--gc-thickness", True),
            ("--subtract-fluorescence", False),
            ("--fluorescence-reference", True),
            ("--fluorescence-level", True),
            ("--fluorescence-q-range", 2),
        ]:
            args = self._remove_arg(args, flag, has_value)

        detector = "Pil300K" if role == "pil300k" else "Eig1M"
        output_dir = self._control_output_root().expanduser().resolve() / detector
        base_sample_name = self._queue_primary_sample_name() if self._sample_list_mode_enabled() else self.sample_name_edit.text().strip() or DEFAULT_SAMPLE_NAME
        sample_name = f"{base_sample_name}_{detector}"
        if role == "pil300k":
            if self.use_measurement_queue_check.isChecked():
                args.extend(["--measurement-done-queue", str(self._measurement_queue_path())])
            else:
                args.extend(["--watch-dir", str(self._detector_watch_dir("Pil300K"))])
            args.extend(
                [
                    "--output-dir",
                    str(output_dir),
                    "--sample-name",
                    sample_name,
                    "--detector",
                    detector,
                    "--poni",
                    self.saxs_poni_edit.text(),
                    "--mask",
                    self.saxs_mask_edit.text(),
                ]
            )
            self._append_arg(args, "--monitor-key", self.saxs_monitor_key_edit.text())
        else:
            if self.use_measurement_queue_check.isChecked():
                args.extend(["--measurement-done-queue", str(self._measurement_queue_path())])
            else:
                args.extend(["--watch-dir", str(self._detector_watch_dir("Eig1M"))])
            args.extend(
                [
                    "--output-dir",
                    str(output_dir),
                    "--sample-name",
                    sample_name,
                    "--detector",
                    detector,
                    "--poni",
                    self.waxs_poni_edit.text(),
                    "--mask",
                    self.waxs_mask_edit.text(),
                ]
            )
            self._append_arg(args, "--monitor-key", self.waxs_monitor_key_edit.text())
        # In paired-detector mode, detector reducers only perform 1D reduction
        # and averaging. ASAXS/background/GC correction is applied once to the
        # stitched SAXS+WAXS curve in the combined analysis HDF5.
        args.extend(["--analysis-mode", "saxs"])
        return args

    def bridge_command_args(self) -> list[str]:
        """Build the optional Kafka-to-JSONL bridge command."""
        args = [str(PROJECT_DIR / "scripts" / "run_kafka_queue_bridge.py")]
        self._append_arg(args, "--queue", self._measurement_queue_path())
        self._append_arg(args, "--data-root", self._raw_data_root())
        self._append_arg(args, "--output-root", self._analysis_root())
        args.extend(["--detector", "Pil300K", "--detector", "Eig1M"])
        self._append_arg(args, "--bootstrap-servers", self.kafka_bootstrap_edit.text().strip() or DEFAULT_KAFKA_BOOTSTRAP)
        topics = [part.strip() for part in re.split(r"[,;]", self.kafka_topic_edit.text()) if part.strip()]
        if not topics:
            topics = [DEFAULT_KAFKA_TOPIC]
        for topic in topics:
            self._append_arg(args, "--topic", topic)
        self._append_arg(args, "--group-id", self.kafka_group_id_edit.text().strip() or DEFAULT_KAFKA_GROUP_ID)
        return args

    def _remove_arg(self, args: list[str], flag: str, has_value: bool | int) -> list[str]:
        cleaned: list[str] = []
        skip_next = 0
        for arg in args:
            if skip_next:
                skip_next -= 1
                continue
            if arg == flag:
                skip_next = int(has_value)
                continue
            cleaned.append(arg)
        return cleaned

    def _append_optional_group(self, args: list[str], flag: str, value: int) -> None:
        if value > 0:
            args.extend([flag, str(value)])

    def refresh_command(self) -> None:
        if not hasattr(self, "command_preview"):
            return
        lines = []
        if (
            self.use_measurement_queue_check.isChecked()
            and self.start_kafka_bridge_check.isChecked()
            and self.source_mode_combo.currentText() != "manifest replay"
        ):
            bridge_args = [sys.executable, *self.bridge_command_args()]
            bridge_command = " ".join(f'"{arg}"' if " " in arg else arg for arg in bridge_args)
            lines.append(f"[Kafka bridge] {bridge_command}")
        if self._dual_detector_enabled():
            for role in ("pil300k", "eig1m"):
                args = [sys.executable, *self.detector_command_args(role)]
                command = " ".join(f'"{arg}"' if " " in arg else arg for arg in args)
                label = "Pil300K" if role == "pil300k" else "Eig1M"
                lines.append(f"[{label}] {command}")
            self.command_preview.setPlainText("\n".join(lines))
        else:
            args = [sys.executable, *self.command_args()]
            reducer_command = " ".join(f'"{arg}"' if " " in arg else arg for arg in args)
            lines.append(reducer_command)
            self.command_preview.setPlainText("\n".join(lines))
        summary = self._preflight_summary_text().splitlines()[0]
        self.statusBar().showMessage(summary)

    def _start_bridge_if_requested(self, output_dir: Path) -> bool:
        """Launch the Kafka bridge when queue mode asks the GUI to own it."""
        if (
            self.source_mode_combo.currentText() == "manifest replay"
            or not self.use_measurement_queue_check.isChecked()
            or not self.start_kafka_bridge_check.isChecked()
        ):
            return True
        if self.bridge_process is not None and self.bridge_process.state() != QtCore.QProcess.NotRunning:
            return True
        self._prepare_measurement_queue_path(clear=False, touch=True)
        self.bridge_monitor_window.set_output_dir(output_dir)
        self.bridge_monitor_window.clear_run_display()
        self.bridge_process = QtCore.QProcess(self)
        self.bridge_process.setProgram(sys.executable)
        self.bridge_process.setArguments(self.bridge_command_args())
        self.bridge_process.setWorkingDirectory(str(PROJECT_DIR))
        self.bridge_monitor_window.attach_process(self.bridge_process)
        self.bridge_monitor_window.show()
        self.bridge_process.start()
        if not self.bridge_process.waitForStarted(3000):
            QtWidgets.QMessageBox.critical(self, "Kafka bridge failed to start", self.bridge_process.errorString())
            return False
        self.bridge_monitor_window.append_log("Kafka bridge started")
        return True

    def start_reducer(self) -> None:
        if self._dual_detector_enabled():
            self.start_dual_reducers()
            return
        if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:
            QtWidgets.QMessageBox.information(self, "Reducer already running", "Stop the current reducer first.")
            return
        if not self._confirm_preflight_before_start():
            return
        self._save_settings()
        fresh_session = self._fresh_session_mode_enabled()
        output_dir = self._control_output_root().expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if fresh_session:
            if self.use_measurement_queue_check.isChecked():
                queue_path = self._prepare_measurement_queue_path(clear=False, touch=False)
                try:
                    self._remove_file_if_present(queue_path)
                except OSError as exc:
                    QtWidgets.QMessageBox.warning(self, "Could not clear old measurement queue", str(exc))
                    return
            event_log_path = output_dir / "live_events.jsonl"
            try:
                self._remove_file_if_present(event_log_path)
            except OSError as exc:
                QtWidgets.QMessageBox.warning(self, "Could not clear old monitor log", str(exc))
                return
        try:
            self._queue_single_sample_jobs()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Sample list is empty", str(exc))
            return
        if fresh_session and not self._clear_analysis_h5_paths_for_restart():
            return
        self.monitor_window.set_output_dir(output_dir)
        self.monitor_window.set_expected_frames(self._expected_monitor_frames())
        self.monitor_window.set_cpu_info(self._monitor_cpu_info(process_count=1))
        self.monitor_window.clear_run_display()
        self.monitor_window.prepare_event_log(clear=fresh_session)
        if self._sample_list_mode_enabled():
            single_allowed = []
            for detector in self._detectors_for_manual_queue():
                detector_paths = self._detector_analysis_h5_paths_for_current_run(detector)
                if detector_paths:
                    single_allowed.extend(detector_paths)
            viewer_output = self._analysis_root().expanduser().resolve()
        else:
            single_allowed = None if self._queue_mode_enabled() else [self._combined_analysis_h5_path()]
            viewer_output = output_dir
        self.curve_window.reset_for_new_run(viewer_output, single_allowed)

        if not self._start_bridge_if_requested(output_dir):
            return
        self.process = QtCore.QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(self.command_args())
        self.process.setWorkingDirectory(str(PLAYGROUND_DIR))
        self.monitor_window.attach_process(self.process)
        self.monitor_window.show()
        self.curve_window.show()
        self.process.start()
        if not self.process.waitForStarted(3000):
            QtWidgets.QMessageBox.critical(self, "Reducer failed to start", self.process.errorString())
            return
        self.statusBar().showMessage("Reducer started")

    def start_dual_reducers(self) -> None:
        if any(process.state() != QtCore.QProcess.NotRunning for process in self.detector_processes.values()):
            QtWidgets.QMessageBox.information(self, "Reducers already running", "Stop the current reducers first.")
            return
        if not self._confirm_preflight_before_start():
            return
        self._save_settings()
        fresh_session = self._fresh_session_mode_enabled()
        self.stitch_run_started_ns = time.time_ns()
        base_output = self._control_output_root().expanduser().resolve()
        base_output.mkdir(parents=True, exist_ok=True)
        self.detector_processes.clear()
        combined_h5_paths = self._combined_analysis_h5_paths_for_current_run()
        if fresh_session:
            legacy_root_stitched = self._analysis_root().expanduser().resolve() / "stitched_analysis.h5"
            for combined_h5 in [*combined_h5_paths, legacy_root_stitched]:
                try:
                    self._remove_file_if_present(combined_h5)
                except OSError as exc:
                    QtWidgets.QMessageBox.warning(self, "Could not clear old combined analysis HDF5", str(exc))
                    return
        if fresh_session and self.use_measurement_queue_check.isChecked():
            queue_path = self._prepare_measurement_queue_path(clear=False, touch=False)
            try:
                self._remove_file_if_present(queue_path)
            except OSError as exc:
                QtWidgets.QMessageBox.warning(self, "Could not clear old measurement queue", str(exc))
                return
        elif self.use_measurement_queue_check.isChecked():
            self._prepare_measurement_queue_path(clear=False, touch=True)
        try:
            self._queue_single_sample_jobs()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Sample list is empty", str(exc))
            return
        if fresh_session and not self._clear_analysis_h5_paths_for_restart():
            return
        try:
            for combined_h5 in combined_h5_paths:
                clear_stitched_averages(combined_h5)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Could not clear old stitched curves", str(exc))
            return
        if not self._start_bridge_if_requested(base_output):
            return
        for role in ("pil300k", "eig1m"):
            detector = "Pil300K" if role == "pil300k" else "Eig1M"
            output_dir = base_output / detector
            viewer_output = self._analysis_root().expanduser().resolve() if self._queue_mode_enabled() else output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            if fresh_session:
                event_log_path = output_dir / "live_events.jsonl"
                try:
                    self._remove_file_if_present(event_log_path)
                except OSError as exc:
                    QtWidgets.QMessageBox.warning(self, f"Could not clear old {detector} monitor log", str(exc))
                    return
            monitor = self.dual_monitor_windows[role]
            viewer = self.dual_curve_windows[role]
            monitor.set_output_dir(output_dir)
            monitor.set_expected_frames(self._expected_monitor_frames(detector))
            monitor.set_cpu_info(self._monitor_cpu_info(process_count=2))
            monitor.clear_run_display()
            monitor.prepare_event_log(clear=fresh_session)
            viewer.reset_for_new_run(
                viewer_output,
                self._detector_analysis_h5_paths_for_current_run(detector),
            )

            process = QtCore.QProcess(self)
            process.setProgram(sys.executable)
            process.setArguments(self.detector_command_args(role))
            process.setWorkingDirectory(str(PLAYGROUND_DIR))
            monitor.attach_process(process)
            process.finished.connect(
                lambda code, status, finished_role=role: self._dual_detector_process_finished(finished_role, code, status)
            )
            process.start()
            self.detector_processes[role] = process
        failed = [role for role, process in self.detector_processes.items() if not process.waitForStarted(3000)]
        if failed:
            QtWidgets.QMessageBox.critical(self, "Reducer failed to start", ", ".join(failed))
            return
        stitched_viewer = self.dual_curve_windows["stitched"]
        stitched_allowed = (
            None
            if self._queue_mode_enabled() and not self._sample_list_mode_enabled()
            else self._combined_analysis_h5_paths_for_current_run()
        )
        stitched_viewer.reset_for_new_run(
            self._stitched_viewer_path(),
            stitched_allowed,
        )
        self.dual_monitor_window.show_tab("Pil300K")
        self.stitch_timer.start()
        self.statusBar().showMessage("Pil300K and Eig1M reducers started; monitors are tabbed in one window")

    def _dual_detector_process_finished(
        self,
        role: str,
        code: int,
        status: QtCore.QProcess.ExitStatus,
    ) -> None:
        """Finalize live stitching once both detector reducers exit."""
        detector = "Pil300K" if role == "pil300k" else "Eig1M"
        state = "crashed" if status == QtCore.QProcess.CrashExit else "finished"
        self.statusBar().showMessage(f"{detector} reducer {state} with exit code {code}")
        if not self.detector_processes:
            return
        if any(process.state() != QtCore.QProcess.NotRunning for process in self.detector_processes.values()):
            return
        QtCore.QTimer.singleShot(500, self._finalize_dual_detector_run)

    def _finalize_dual_detector_run(self) -> None:
        """Run one last stitch pass, then stop the stitch refresh timer."""
        if not self.detector_processes:
            return
        self._update_live_stitched_outputs()
        self._write_final_stitched_asaxs_outputs()
        self.stitch_timer.stop()
        self.statusBar().showMessage("Pil300K and Eig1M reducers finished; stitched output finalized")

    def _monitor_cpu_info(self, process_count: int | None = None) -> str:
        """Short CPU/process summary shown under the monitor progress bar."""
        processes = process_count if process_count is not None else (2 if self._dual_detector_enabled() else 1)
        cores = self.jobs_spin.value()
        total_requested = processes * cores
        cpu_max = max(1, os.cpu_count() or 1)
        plural_process = "process" if processes == 1 else "processes"
        plural_core = "core" if cores == 1 else "cores"
        return f"CPU: {processes} reducer {plural_process} x {cores} requested {plural_core} = {total_requested} requested workers, system {cpu_max} cores"

    def _write_final_stitched_asaxs_outputs(self) -> None:
        """Write ASAXS/GC outputs to the combined stitched files, if requested."""
        if self.analysis_mode_combo.currentText() != "asaxs":
            return
        settings = self._stitched_asaxs_settings()
        wrote_count = 0
        for combined_h5 in self._stitched_asaxs_target_paths():
            try:
                if write_stitched_asaxs_outputs(combined_h5, settings):
                    wrote_count += 1
            except Exception as exc:  # noqa: BLE001 - keep GUI finalization alive.
                self.statusBar().showMessage(f"Stitched ASAXS correction skipped for {combined_h5.name}: {exc}")
        if wrote_count:
            self.statusBar().showMessage(f"Stitched ASAXS/GC correction written for {wrote_count} file(s)")

    def _stitched_asaxs_target_paths(self) -> list[Path]:
        """Return combined analysis H5 files that should receive stitched ASAXS output."""
        if self._sample_list_mode_enabled():
            return [path for path in self._combined_analysis_h5_paths_for_current_run() if path.exists()]
        if self._queue_mode_enabled():
            root = self._analysis_root().expanduser().resolve()
            return sorted(
                path
                for path in root.rglob("*_analysis.h5")
                if path.exists() and "Pil300K" not in path.parts and "Eig1M" not in path.parts
                and (self.stitch_run_started_ns is None or path.stat().st_mtime_ns >= self.stitch_run_started_ns)
            )
        path = self._combined_analysis_h5_path()
        return [path] if path.exists() else []

    def _stitched_asaxs_settings(self) -> StitchedAsaxsSettings:
        """Collect ASAXS parameters for stitched-curve normalization."""
        return StitchedAsaxsSettings(
            num_groups=max(1, self.num_groups_spin.value()),
            sample_group=None,
            air_group=self._optional_group_value(self.air_group_spin.value()),
            empty_group=self._optional_group_value(self.empty_group_spin.value()),
            water_group=None,
            gc_group=self._optional_group_value(self.gc_group_spin.value()),
            gc_reference_file=self.gc_ref_edit.text().strip() or None,
            gc_q_range=(float(self.gc_q_min_spin.value()), float(self.gc_q_max_spin.value())),
            capillary_thickness=self._optional_positive_float(self.capillary_thickness_spin.value()),
            gc_thickness=self._optional_positive_float(self.gc_thickness_spin.value()),
            subtract_fluorescence=self.subtract_fluorescence_check.isChecked(),
            fluorescence_level=self.fluorescence_level_spin.value() if self.fluorescence_level_spin.value() != 0 else None,
            fluorescence_reference=self.fluorescence_reference_combo.currentText(),
            fluorescence_q_range=(float(self.fluorescence_q_min_spin.value()), float(self.fluorescence_q_max_spin.value())),
            asaxs_pairs=tuple(self._asaxs_pairs_from_table()),
        )

    @staticmethod
    def _optional_group_value(value: int) -> int | None:
        return int(value) if value > 0 else None

    @staticmethod
    def _optional_positive_float(value: float) -> float | None:
        return float(value) if value > 0 else None

    def stop_reducer(self) -> None:
        if self._dual_detector_enabled() or self.detector_processes:
            self.stop_dual_reducers()
            self.stop_bridge()
            return
        if self.process is None or self.process.state() == QtCore.QProcess.NotRunning:
            self.stop_bridge()
            return
        self._record_user_stop(self.monitor_window, "single detector")
        self._request_process_stop(self.process, self.monitor_window, "single detector")
        self.stop_bridge()
        self.statusBar().showMessage("Reducer stop requested")

    def stop_dual_reducers(self) -> None:
        stopped = False
        for role, process in self.detector_processes.items():
            if process.state() == QtCore.QProcess.NotRunning:
                continue
            detector = "Pil300K" if role == "pil300k" else "Eig1M"
            self._record_user_stop(self.dual_monitor_windows[role], detector)
            self._request_process_stop(process, self.dual_monitor_windows[role], detector)
            stopped = True
        if stopped:
            self.statusBar().showMessage("Pil300K/Eig1M reducer stop requested")
        self.stitch_timer.stop()

    def stop_bridge(self) -> None:
        if self.bridge_process is None or self.bridge_process.state() == QtCore.QProcess.NotRunning:
            return
        self.bridge_monitor_window.append_log("User stopped Kafka bridge")
        self.bridge_process.terminate()

        def force_kill_if_running() -> None:
            if self.bridge_process is None or self.bridge_process.state() == QtCore.QProcess.NotRunning:
                self.bridge_monitor_window.append_log("Kafka bridge stopped")
                return
            self.bridge_monitor_window.append_log("Kafka bridge did not exit quickly; killing process")
            self.bridge_process.kill()

        QtCore.QTimer.singleShot(1000, force_kill_if_running)

    def _request_process_stop(self, process: QtCore.QProcess, monitor: ProcessMonitorWindow, label: str) -> None:
        """Ask a reducer to stop, then force-kill it shortly after if needed."""
        process.terminate()

        def force_kill_if_running() -> None:
            if process.state() == QtCore.QProcess.NotRunning:
                monitor.append_log(f"{label} reducer stopped")
                return
            monitor.append_log(f"{label} reducer did not exit quickly; killing process")
            process.kill()

        QtCore.QTimer.singleShot(1000, force_kill_if_running)

    def _record_user_stop(self, monitor: ProcessMonitorWindow, label: str) -> None:
        """Write a visible stop marker before terminating a reducer process."""
        message = f"User stopped {label} reducer"
        monitor.append_log(message)
        event = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": "user_stop_requested",
            "message": message,
        }
        try:
            monitor.output_dir.mkdir(parents=True, exist_ok=True)
            with monitor.event_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
        except OSError:
            return
        monitor.tail_event_log()

    def show_monitor(self) -> None:
        if (
            self.bridge_process is not None
            and self.bridge_process.state() != QtCore.QProcess.NotRunning
        ) or (
            self.use_measurement_queue_check.isChecked()
            and self.start_kafka_bridge_check.isChecked()
        ):
            self.bridge_monitor_window.show()
            self.bridge_monitor_window.raise_()
        if self._dual_detector_enabled():
            base_output = self._control_output_root().expanduser().resolve()
            for role, monitor in self.dual_monitor_windows.items():
                detector = "Pil300K" if role == "pil300k" else "Eig1M"
                monitor.set_output_dir(base_output / detector)
                detector = "Pil300K" if role == "pil300k" else "Eig1M"
                monitor.set_expected_frames(self._expected_monitor_frames(detector))
                monitor.set_cpu_info(self._monitor_cpu_info(process_count=2))
            self.dual_monitor_window.show_tab("Pil300K")
            return
        self.monitor_window.set_output_dir(self._control_output_root().expanduser().resolve())
        self.monitor_window.set_expected_frames(self._expected_monitor_frames())
        self.monitor_window.set_cpu_info(self._monitor_cpu_info(process_count=1))
        self.monitor_window.show()
        self.monitor_window.raise_()

    def show_curves(self) -> None:
        if self._dual_detector_enabled():
            base_output = self._control_output_root().expanduser().resolve()
            for role, window in self.dual_curve_windows.items():
                if role == "stitched":
                    window.output_dir_edit.setText(str(self._stitched_viewer_path()))
                else:
                    detector = "Pil300K" if role == "pil300k" else "Eig1M"
                    window.output_dir_edit.setText(str(base_output / detector))
                window._refresh_now()
            self.dual_curve_window.show_tab("Stitched")
            return
        self.curve_window.output_dir_edit.setText(str(self._control_output_root().expanduser().resolve()))
        self.curve_window._refresh_now()
        self.curve_window.show()
        self.curve_window.raise_()

    def open_analysis_h5_in_viewer(self) -> None:
        """Open an existing analysis HDF5 without starting a reducer."""
        path = self._choose_analysis_h5("Open analysis HDF5")
        if path is None:
            return
        self.curve_window.reset_for_new_run(path, [path])
        self.curve_window.output_dir_edit.setText(str(path))
        self.curve_window._refresh_now(update_plot=True)
        self.curve_window.show()
        self.curve_window.raise_()
        self.statusBar().showMessage(f"Loaded analysis H5 in curve viewer: {path.name}")

    def export_xanos_format_from_h5(self) -> None:
        """Export final ASAXS HDF5 curves to the older XAnoS text-file layout."""
        path = self._choose_analysis_h5("Export XAnos format from analysis HDF5")
        if path is None:
            return
        try:
            written = export_analysis_h5_to_xanos_format(path)
        except Exception as exc:  # noqa: BLE001 - present a clean GUI error.
            QtWidgets.QMessageBox.warning(self, "XAnos export failed", str(exc))
            return
        if not written:
            QtWidgets.QMessageBox.information(
                self,
                "No final ASAXS curves",
                "This HDF5 file does not contain /entry/final/corrected_I_q_E yet.",
            )
            return
        dat_count = sum(1 for item in written if item.suffix.lower() == ".dat")
        list_count = sum(1 for item in written if item.name == "xanos_file_list.txt")
        output_dir = written[0].parents[1] if written[0].parent.name != "XAnos format" else written[0].parent
        QtWidgets.QMessageBox.information(
            self,
            "XAnos export complete",
            f"Wrote {dat_count} energy file(s) and {list_count} list file(s) under:\n{output_dir}",
        )
        self.statusBar().showMessage(f"XAnos format exported: {output_dir}")

    def _choose_analysis_h5(self, caption: str) -> Path | None:
        """Select an existing analysis HDF5 file for offline viewing/export."""
        start = str(self._analysis_root().expanduser().resolve())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption,
            start,
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        return Path(path).expanduser().resolve() if path else None

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt method name.
        self._save_settings()
        super().closeEvent(event)

    def _expected_monitor_frames(self, detector: str | None = None) -> int | None:
        if self.source_mode_combo.currentText() == "watch folder":
            preflight_total = self._preflight_expected_frames(detector)
            if preflight_total is not None:
                return preflight_total
            energies = self.num_energies_spin.value()
            if energies <= 0:
                return None
            frames_per_sample = energies * self.num_groups_spin.value() * self.num_frames_spin.value()
            if self.auto_num_groups_check.isChecked():
                return None
            if self._sample_list_mode_enabled():
                sample_count = len(self._sample_names_for_queue())
                return frames_per_sample * sample_count if sample_count > 0 else None
            return frames_per_sample

        # Manifest replay may use limit controls for quick tests. If the energy
        # limit is unknown/zero, leave the monitor in activity mode.
        limit_energies = self.limit_energies_spin.value()
        limit_frames = self.limit_frames_spin.value()
        if limit_energies <= 0:
            return None
        frames_per_group = limit_frames if limit_frames > 0 else self.num_frames_spin.value()
        return limit_energies * self.num_groups_spin.value() * frames_per_group

    def _update_live_stitched_outputs(self) -> None:
        if not self._dual_detector_enabled():
            return
        base_output = (
            self._analysis_root().expanduser().resolve()
            if self._sample_list_mode_enabled()
            else self._control_output_root().expanduser().resolve()
        )
        online_queue_mode = self._queue_mode_enabled() and not self._sample_list_mode_enabled()
        if online_queue_mode:
            base_output = self._analysis_root().expanduser().resolve()
        pil300k_root = base_output if (self._sample_list_mode_enabled() or online_queue_mode) else base_output / "Pil300K"
        eig1m_root = base_output if (self._sample_list_mode_enabled() or online_queue_mode) else base_output / "Eig1M"
        stitched_paths: list[Path] = []
        try:
            if self._sample_list_mode_enabled():
                for sample_name in self._sample_names_for_queue():
                    stitched_h5 = update_live_stitched_averages(
                        pil300k_root,
                        eig1m_root,
                        self._combined_analysis_h5_path_for_sample(sample_name),
                        sample_names=[sample_name],
                        min_mtime_ns=self.stitch_run_started_ns,
                    )
                    if stitched_h5 is not None:
                        stitched_paths.append(stitched_h5)
            elif online_queue_mode:
                for sample_name, pil300k_h5, eig1m_h5 in paired_detector_analysis_h5s(
                    pil300k_root,
                    eig1m_root,
                    min_mtime_ns=self.stitch_run_started_ns,
                ):
                    sample_output_root = self._paired_detector_sample_output_root(pil300k_h5, eig1m_h5)
                    stitched_h5 = update_live_stitched_averages(
                        pil300k_h5.parent,
                        eig1m_h5.parent,
                        sample_output_root / f"{self._safe_name(sample_name, 'analysis')}_analysis.h5",
                        sample_names=[sample_name],
                        min_mtime_ns=self.stitch_run_started_ns,
                    )
                    if stitched_h5 is not None:
                        stitched_paths.append(stitched_h5)
            else:
                stitched_h5 = update_live_stitched_averages(
                    pil300k_root,
                    eig1m_root,
                    self._combined_analysis_h5_path(),
                    min_mtime_ns=self.stitch_run_started_ns,
                )
                if stitched_h5 is not None:
                    stitched_paths.append(stitched_h5)
        except Exception as exc:  # noqa: BLE001 - keep the live GUI alive.
            self.statusBar().showMessage(f"Stitch update skipped: {exc}")
            return
        if stitched_paths:
            viewer = self.dual_curve_windows["stitched"]
            viewer.output_dir_edit.setText(str(self._stitched_viewer_path()))
            viewer.curve_kind_combo.setCurrentText("h5 stitched averages")
            viewer._refresh_now(update_plot=viewer._auto_should_update_plot())

    def _paired_detector_sample_output_root(self, pil300k_h5: Path, eig1m_h5: Path) -> Path:
        """Return the sample output folder for paired online detector H5 files."""
        if pil300k_h5.parent.parent == eig1m_h5.parent.parent:
            return pil300k_h5.parent.parent
        try:
            relative_parent = pil300k_h5.parent.parent.relative_to(self._analysis_root().expanduser().resolve())
        except ValueError:
            return pil300k_h5.parent.parent
        return self._analysis_root().expanduser().resolve() / relative_parent

    def _combined_analysis_h5_path(self) -> Path:
        """Return the one public analysis HDF5 path for the current batch."""
        explicit = "" if self.use_measurement_queue_check.isChecked() else self.analysis_h5_edit.text().strip()
        if explicit:
            return Path(explicit).expanduser().resolve()
        if self._queue_mode_enabled() and self._sample_names_for_queue():
            sample = self._queue_primary_sample_name()
        else:
            sample = self.sample_name_edit.text().strip() or DEFAULT_SAMPLE_NAME
        safe_sample = self._safe_name(sample, "analysis")
        if self._sample_list_mode_enabled():
            return self._combined_analysis_h5_path_for_sample(sample)
        output_root = self._queue_primary_sample_root() if self._sample_list_mode_enabled() else self._control_output_root()
        return output_root.expanduser().resolve() / f"{safe_sample}_analysis.h5"


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = SetupWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
