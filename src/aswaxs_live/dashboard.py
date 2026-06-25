"""ASWAXS v5 dashboard GUI."""

from __future__ import annotations

import os
import json
import re
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from aswaxs_live.h5_tools import H5IqViewerDialog, H5StructureViewerDialog
from aswaxs_live.rack_builder import RackBuilderDialog
from aswaxs_live.task_queue import (
    DEFAULT_QUEUE_PATH,
    AsaxsPair,
    TaskSpec,
    load_queue,
    preflight_task,
    run_task,
    safe_name,
    save_queue,
    scan_detector_files,
    sort_h5_files,
    task_from_json,
    task_to_json,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
BUILDER_SETTINGS_PATH = PROJECT_DIR / "aswaxs_v5_builder_settings.json"
DETECTOR_PROGRESS_RE = re.compile(r"\b(Pil300K|Eig1M)\s+(\d+)/(\d+)\b")


class QueueTable(QtWidgets.QTableWidget):
    row_move_requested = QtCore.pyqtSignal(int, int)
    copy_requested = QtCore.pyqtSignal()
    paste_requested = QtCore.pyqtSignal()
    delete_requested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.locked = False
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.MoveAction)

    def dropEvent(self, event: QtCore.QEvent) -> None:
        if self.locked:
            event.ignore()
            return
        if not isinstance(event, QtGui.QDropEvent):  # type: ignore[name-defined]
            super().dropEvent(event)
            return
        source_rows = self.selectionModel().selectedRows() if self.selectionModel() else []
        if not source_rows:
            event.ignore()
            return
        source = source_rows[0].row()
        target = self.rowAt(event.pos().y())
        indicator = self.dropIndicatorPosition()
        if target < 0:
            target = self.rowCount()
        elif indicator == QtWidgets.QAbstractItemView.BelowItem:
            target += 1
        elif indicator == QtWidgets.QAbstractItemView.OnViewport:
            target = self.rowCount()
        if source < target:
            target -= 1
        if source != target and 0 <= source < self.rowCount() and 0 <= target < self.rowCount():
            self.row_move_requested.emit(source, target)
        event.acceptProposedAction()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 - Qt override name.
        if self.locked:
            event.ignore()
            return
        if event.matches(QtGui.QKeySequence.Copy):
            self.copy_requested.emit()
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.Paste):
            self.paste_requested.emit()
            event.accept()
            return
        if event.key() in {QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace}:
            self.delete_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 - Qt override name.
        if self.locked:
            event.ignore()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 - Qt override name.
        if self.locked:
            event.ignore()
            return
        super().mouseDoubleClickEvent(event)


class TaskRunner(QtCore.QThread):
    message = QtCore.pyqtSignal(str)
    task_progress = QtCore.pyqtSignal(int, float, str)
    task_started = QtCore.pyqtSignal(int)
    task_finished = QtCore.pyqtSignal(int, bool, str)
    task_skipped = QtCore.pyqtSignal(int, str)
    all_done = QtCore.pyqtSignal()

    def __init__(self, tasks: list[TaskSpec], indices: list[int], accepted_indices: list[int], *, run_any_status: bool = False) -> None:
        super().__init__()
        self.tasks = tasks
        self.indices = indices
        self.accepted_indices = set(accepted_indices)
        self.run_any_status = run_any_status
        self.stop_after_current_requested = False
        self.abort_current_requested = False

    def request_stop_after_current(self) -> None:
        self.stop_after_current_requested = True

    def request_stop_current(self) -> None:
        self.abort_current_requested = True
        self.stop_after_current_requested = True

    def run(self) -> None:
        for index in self.indices:
            if self.stop_after_current_requested:
                self.message.emit("Stop requested; queue paused before next task.")
                break
            task = self.tasks[index]
            if index not in self.accepted_indices:
                reason = f"Skipped because status is {task.status}."
                self.message.emit(f"{task.task_name}: {reason}")
                self.task_skipped.emit(index, reason)
                continue
            self.message.emit(f"{task.task_name}: starting queued task.")
            self.task_started.emit(index)
            try:
                run_task(
                    task,
                    self.message.emit,
                    should_stop=lambda: self.abort_current_requested,
                    progress=lambda fraction, label, task_index=index: self.task_progress.emit(task_index, fraction, label),
                )
            except Exception as exc:  # noqa: BLE001 - keep queue UI alive.
                if str(exc) == "Stopped by user":
                    error_text = "Stopped by user"
                    self.message.emit(f"{task.task_name}: stopped by user.")
                else:
                    error_text = traceback.format_exc()
                    self.message.emit(error_text)
                self.task_finished.emit(index, False, error_text)
            else:
                self.task_finished.emit(index, True, "Complete")
        self.all_done.emit()


class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ASWAXS v5")
        self.resize(1480, 900)
        self.tasks: list[TaskSpec] = []
        self.log_messages: list[str] = []
        self.queue_path = DEFAULT_QUEUE_PATH
        self.runner: TaskRunner | None = None
        self.queue_locked = False
        self._queue_selection_mode = QtWidgets.QAbstractItemView.ExtendedSelection
        self.h5_iq_viewer: H5IqViewerDialog | None = None
        self.h5_structure_viewer: H5StructureViewerDialog | None = None
        self.active_run_indices: list[int] = []
        self.editing_index: int | None = None
        self._loading_builder_settings = False
        self._output_dir_manually_overridden = False
        self._build_ui()
        self.load_builder_settings()
        self._load_default_queue()

    def _build_ui(self) -> None:
        self._build_actions()
        self._build_menus()
        self._build_queue_toolbar()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)

        self._build_dashboard_tab()
        self._build_task_builder_tab()

        self.curve_refresh_timer = QtCore.QTimer(self)
        self.curve_refresh_timer.setInterval(5000)
        self.curve_refresh_timer.timeout.connect(self.refresh_current_curves)
        self.curve_refresh_timer.start()

        self.statusBar().showMessage("Ready")

    def _build_actions(self) -> None:
        self.new_queue_action = QtWidgets.QAction("New Queue", self)
        self.new_queue_action.triggered.connect(self.new_queue)
        self.clear_queue_action = QtWidgets.QAction("Clear Queue", self)
        self.clear_queue_action.triggered.connect(self.clear_queue)
        self.open_queue_action = QtWidgets.QAction("Open Queue...", self)
        self.open_queue_action.triggered.connect(self.open_queue)
        self.save_queue_action = QtWidgets.QAction("Save Queue", self)
        self.save_queue_action.triggered.connect(self.save_queue)
        self.save_queue_as_action = QtWidgets.QAction("Save Queue As...", self)
        self.save_queue_as_action.triggered.connect(self.save_queue_as)
        self.new_task_action = QtWidgets.QAction("New Task", self)
        self.new_task_action.triggered.connect(self.clear_builder)
        self.add_folder_action = QtWidgets.QAction("Add Folder", self)
        self.add_folder_action.triggered.connect(self.browse_raw_folder)
        self.add_files_action = QtWidgets.QAction("Choose Raw HDF5 Files", self)
        self.add_files_action.triggered.connect(self.browse_raw_files)
        self.add_to_queue_action = QtWidgets.QAction("Add Task to Queue", self)
        self.add_to_queue_action.triggered.connect(self.add_task_from_builder)
        self.update_task_action = QtWidgets.QAction("Update Selected Task", self)
        self.update_task_action.triggered.connect(self.update_selected_task_from_builder)
        self.edit_task_action = QtWidgets.QAction("Edit Task...", self)
        self.edit_task_action.triggered.connect(self.edit_selected_task)
        self.duplicate_task_action = QtWidgets.QAction("Duplicate Task", self)
        self.duplicate_task_action.setShortcut(QtGui.QKeySequence("Ctrl+D"))
        self.duplicate_task_action.triggered.connect(self.duplicate_selected_task)
        self.remove_task_action = QtWidgets.QAction("Delete Task", self)
        self.remove_task_action.setShortcut(QtGui.QKeySequence.Delete)
        self.remove_task_action.triggered.connect(self.remove_selected_task)
        self.copy_task_action = QtWidgets.QAction("Copy Task", self)
        self.copy_task_action.setShortcut(QtGui.QKeySequence.Copy)
        self.copy_task_action.triggered.connect(self.copy_selected_tasks)
        self.paste_task_action = QtWidgets.QAction("Paste Task", self)
        self.paste_task_action.setShortcut(QtGui.QKeySequence.Paste)
        self.paste_task_action.triggered.connect(self.paste_tasks)
        self.set_status_action = QtWidgets.QAction("Set Status...", self)
        self.set_status_action.triggered.connect(self.set_selected_task_status)
        self.move_up_action = QtWidgets.QAction("Move Up", self)
        self.move_up_action.triggered.connect(lambda: self.move_selected_task(-1))
        self.move_down_action = QtWidgets.QAction("Move Down", self)
        self.move_down_action.triggered.connect(lambda: self.move_selected_task(1))
        self.run_selected_action = QtWidgets.QAction("Run Task", self)
        self.run_selected_action.triggered.connect(self.run_selected)
        self.run_all_action = QtWidgets.QAction("Start Queue", self)
        self.run_all_action.triggered.connect(self.run_all)
        self.stop_current_action = QtWidgets.QAction("Stop Queue", self)
        self.stop_current_action.triggered.connect(self.stop_current_queue)
        self.open_output_action = QtWidgets.QAction("Open Output", self)
        self.open_output_action.triggered.connect(self.open_selected_output)
        self.h5_iq_viewer_action = QtWidgets.QAction("HDF5 I-q Plot Viewer", self)
        self.h5_iq_viewer_action.triggered.connect(self.open_h5_iq_viewer)
        self.h5_structure_viewer_action = QtWidgets.QAction("HDF5 Structure / Metadata Viewer", self)
        self.h5_structure_viewer_action.triggered.connect(self.open_h5_structure_viewer)
    def _build_menus(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addActions([self.new_queue_action, self.clear_queue_action, self.open_queue_action, self.save_queue_action, self.save_queue_as_action])
        file_menu.addSeparator()
        file_menu.addAction(self.open_output_action)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        task_menu = menu.addMenu("Task")
        task_menu.addActions([
            self.new_task_action,
            self.add_to_queue_action,
            self.update_task_action,
            self.edit_task_action,
        ])
        task_menu.addSeparator()
        task_menu.addActions([self.run_selected_action, self.run_all_action])
        task_menu.addActions([self.move_up_action, self.move_down_action])
        task_menu.addSeparator()
        task_menu.addAction(self.stop_current_action)
        task_menu.addSeparator()
        task_menu.addAction(self.copy_task_action)
        task_menu.addAction(self.paste_task_action)
        task_menu.addAction(self.duplicate_task_action)
        task_menu.addSeparator()
        task_menu.addAction(self.set_status_action)
        task_menu.addAction(self.remove_task_action)

        view_menu = menu.addMenu("View")
        for index, label in enumerate(["Dashboard", "Task Builder"]):
            action = QtWidgets.QAction(label, self)
            action.triggered.connect(lambda _checked=False, tab=index: self.tabs.setCurrentIndex(tab))
            view_menu.addAction(action)

        tools_menu = menu.addMenu("Tools")
        tools_menu.addAction(self.h5_iq_viewer_action)
        tools_menu.addAction(self.h5_structure_viewer_action)

        help_menu = menu.addMenu("Help")
        help_menu.addAction("About ASWAXS v5", self.about)

    def _build_queue_toolbar(self) -> None:
        toolbar = self.addToolBar("Queue")
        toolbar.setMovable(False)
        toolbar.addAction(self.run_all_action)
        toolbar.addAction(self.stop_current_action)

    def _build_dashboard_tab(self) -> None:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(main_splitter, 1)

        queue_box = QtWidgets.QGroupBox("Queue")
        queue_layout = QtWidgets.QVBoxLayout(queue_box)
        self.queue_lock_label = QtWidgets.QLabel("")
        self.queue_lock_label.setVisible(False)
        self.queue_lock_label.setStyleSheet("color: #666; font-weight: 600; padding: 2px 4px;")
        queue_layout.addWidget(self.queue_lock_label)

        self.queue_table = QueueTable()
        self.queue_table.setColumnCount(7)
        self.queue_table.setHorizontalHeaderLabels(["Task", "Status", "Files", "Sequence", "Detectors", "ASAXS Pairs", "Output"])
        self.queue_table.setMinimumHeight(170)
        self.queue_table.setWordWrap(False)
        self.queue_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        header = self.queue_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        for column, width in enumerate([280, 90, 150, 110, 210, 160, 420]):
            self.queue_table.setColumnWidth(column, width)
        self.queue_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.queue_table.setSelectionMode(self._queue_selection_mode)
        self.queue_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.queue_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.queue_table.itemSelectionChanged.connect(self._sync_selection_from_table)
        self.queue_table.customContextMenuRequested.connect(self.show_queue_context_menu)
        self.queue_table.row_move_requested.connect(self.move_task_row)
        self.queue_table.copy_requested.connect(self.copy_selected_tasks)
        self.queue_table.paste_requested.connect(self.paste_tasks)
        self.queue_table.delete_requested.connect(self.remove_selected_task)
        queue_layout.addWidget(self.queue_table)
        main_splitter.addWidget(queue_box)

        main_splitter.addWidget(self._build_current_curves_panel())
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 4)
        main_splitter.setSizes([190, 560])

        progress_box = QtWidgets.QGroupBox("Progress")
        progress_layout = QtWidgets.QVBoxLayout(progress_box)
        self.overall_progress = QtWidgets.QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.current_stage_label = QtWidgets.QLabel("Idle")
        self.pil_detector_progress_label = QtWidgets.QLabel("Pil300K: idle")
        self.eig_detector_progress_label = QtWidgets.QLabel("Eig1M: idle")
        detector_progress_row = QtWidgets.QHBoxLayout()
        detector_progress_row.addWidget(self.pil_detector_progress_label)
        detector_progress_row.addWidget(self.eig_detector_progress_label)
        detector_progress_row.addStretch(1)
        progress_layout.addWidget(self.overall_progress)
        progress_layout.addWidget(self.current_stage_label)
        progress_layout.addLayout(detector_progress_row)
        layout.addWidget(progress_box)

        self.tabs.addTab(page, "Dashboard")

    def _build_current_curves_panel(self) -> QtWidgets.QGroupBox:
        curve_box = QtWidgets.QGroupBox("Final Reduced Curves")
        layout = QtWidgets.QVBoxLayout(curve_box)
        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        source_label = QtWidgets.QLabel("Source: final data only")
        source_label.setStyleSheet("color: #444;")
        controls.addWidget(source_label)

        self.curve_max_spin = QtWidgets.QSpinBox()
        self.curve_max_spin.setRange(1, 100)
        self.curve_max_spin.setValue(12)
        self.curve_max_spin.valueChanged.connect(self.refresh_current_curves)
        controls.addWidget(QtWidgets.QLabel("Max"))
        controls.addWidget(self.curve_max_spin)

        self.curve_log_x_check = QtWidgets.QCheckBox("log q")
        self.curve_log_x_check.setChecked(True)
        self.curve_log_x_check.stateChanged.connect(self.refresh_current_curves)
        controls.addWidget(self.curve_log_x_check)

        self.curve_log_y_check = QtWidgets.QCheckBox("log I")
        self.curve_log_y_check.setChecked(True)
        self.curve_log_y_check.stateChanged.connect(self.refresh_current_curves)
        controls.addWidget(self.curve_log_y_check)

        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_current_curves)
        controls.addWidget(refresh)
        controls.addStretch(1)

        self.current_curve_plot = pg.PlotWidget()
        self.current_curve_plot.showGrid(x=True, y=True, alpha=0.25)
        self.current_curve_plot.setLabel("bottom", "q", units="A^-1")
        self.current_curve_plot.setLabel("left", "I(q)", units="a.u.")
        self.current_curve_legend = self.current_curve_plot.addLegend(offset=(8, 8))
        self.current_curve_plot.getPlotItem().setDownsampling(auto=True, mode="peak")
        self.current_curve_plot.getPlotItem().setClipToView(True)
        layout.addWidget(self.current_curve_plot, 1)

        self.current_curve_status = QtWidgets.QLabel("Select a task to show final reduced curves.")
        layout.addWidget(self.current_curve_status)
        return curve_box

    def _build_task_builder_tab(self) -> None:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.task_name_edit = QtWidgets.QLineEdit()
        self.raw_folder_edit = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse Folder")
        browse.clicked.connect(self.browse_raw_folder)
        raw_row = QtWidgets.QHBoxLayout()
        raw_row.addWidget(self.raw_folder_edit, 1)
        raw_row.addWidget(browse)

        self.detector_mode_combo = QtWidgets.QComboBox()
        self.detector_mode_combo.addItem("Both detectors: Pil300K + Eig1M", "both")
        self.detector_mode_combo.addItem("Pil300K only", "pil300k")
        self.detector_mode_combo.addItem("Eig1M only", "eig1m")
        self.reduction_mode_combo = QtWidgets.QComboBox()
        self.reduction_mode_combo.addItem("ASAXS / XAnos", "asaxs")
        self.reduction_mode_combo.addItem("SAXS only", "saxs")

        self.pil_files_edit = QtWidgets.QPlainTextEdit()
        self.pil_files_edit.setPlaceholderText("Choose Pil300K raw .h5 files, one path per line.")
        self.pil_files_edit.setMaximumHeight(92)
        pil_file_buttons = QtWidgets.QHBoxLayout()
        pil_choose = QtWidgets.QPushButton("Choose Pil300K HDF5")
        pil_choose.clicked.connect(lambda: self.browse_detector_files("Pil300K"))
        pil_clear = QtWidgets.QPushButton("Clear Pil300K Files")
        pil_clear.clicked.connect(lambda: self.clear_detector_files("Pil300K"))
        pil_file_buttons.addWidget(pil_choose)
        pil_file_buttons.addWidget(pil_clear)
        pil_file_buttons.addStretch(1)

        self.eig_files_edit = QtWidgets.QPlainTextEdit()
        self.eig_files_edit.setPlaceholderText("Choose Eig1M raw .h5 files, one path per line.")
        self.eig_files_edit.setMaximumHeight(92)
        eig_file_buttons = QtWidgets.QHBoxLayout()
        eig_choose = QtWidgets.QPushButton("Choose Eig1M HDF5")
        eig_choose.clicked.connect(lambda: self.browse_detector_files("Eig1M"))
        eig_clear = QtWidgets.QPushButton("Clear Eig1M Files")
        eig_clear.clicked.connect(lambda: self.clear_detector_files("Eig1M"))
        eig_file_buttons.addWidget(eig_choose)
        eig_file_buttons.addWidget(eig_clear)
        eig_file_buttons.addStretch(1)

        self.output_dir_edit = QtWidgets.QLineEdit()
        output_browse = QtWidgets.QPushButton("Override")
        output_browse.clicked.connect(self.browse_output_dir)
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output_dir_edit, 1)
        output_row.addWidget(output_browse)

        self.pil_count_label = QtWidgets.QLabel("Pil300K: 0")
        self.eig_count_label = QtWidgets.QLabel("Eig1M: 0")
        counts_row = QtWidgets.QHBoxLayout()
        counts_row.addWidget(self.pil_count_label)
        counts_row.addWidget(self.eig_count_label)
        counts_row.addStretch(1)

        self.energy_spin = self._spin(1, 999, 20)
        self.group_spin = self._spin(1, 999, 13)
        self.frame_spin = self._spin(1, 999, 10)
        sequence_row = QtWidgets.QHBoxLayout()
        for label, widget in [("Energies", self.energy_spin), ("Groups", self.group_spin), ("Frames", self.frame_spin)]:
            sequence_row.addWidget(QtWidgets.QLabel(label))
            sequence_row.addWidget(widget)
        sequence_row.addStretch(1)

        self.pil_poni_edit = QtWidgets.QLineEdit()
        self.pil_mask_edit = QtWidgets.QLineEdit()
        self.eig_poni_edit = QtWidgets.QLineEdit()
        self.eig_mask_edit = QtWidgets.QLineEdit()
        pil_poni_row = self._file_browse_row(self.pil_poni_edit, "Choose Pil300K PONI", "PONI files (*.poni);;All files (*)")
        pil_mask_row = self._file_browse_row(self.pil_mask_edit, "Choose Pil300K mask", "Mask files (*.msk *.edf *.npy);;All files (*)")
        eig_poni_row = self._file_browse_row(self.eig_poni_edit, "Choose Eig1M PONI", "PONI files (*.poni);;All files (*)")
        eig_mask_row = self._file_browse_row(self.eig_mask_edit, "Choose Eig1M mask", "Mask files (*.msk *.edf *.npy);;All files (*)")
        self.capillary_spin = self._double_spin(0.0001, 100.0, 0.15)
        self.gc_thickness_spin = self._double_spin(0.0001, 100.0, 0.1055)
        self.gc_group_spin = self._spin(0, 999, 1)
        self.air_group_spin = self._spin(0, 999, 2)
        self.empty_group_spin = self._spin(0, 999, 3)
        cpu_count = max(1, os.cpu_count() or 1)
        self.core_spin = self._spin(1, cpu_count, cpu_count)

        self.pair_table = QtWidgets.QTableWidget(0, 3)
        self.pair_table.setHorizontalHeaderLabels(["Output name", "Sample group", "Solvent group"])
        self.pair_table.horizontalHeader().setStretchLastSection(True)
        pair_buttons = QtWidgets.QHBoxLayout()
        rack_helper = QtWidgets.QPushButton("Visual Rack Builder")
        rack_helper.clicked.connect(self.open_rack_builder)
        add_pair = QtWidgets.QPushButton("Add Pair")
        add_pair.clicked.connect(lambda: self._append_pair_row("", "", ""))
        remove_pair = QtWidgets.QPushButton("Remove Pair")
        remove_pair.clicked.connect(self.remove_selected_pair)
        clear_pair = QtWidgets.QPushButton("Clear Pairs")
        clear_pair.clicked.connect(self.clear_pair_rows)
        pair_buttons.addWidget(rack_helper)
        pair_buttons.addWidget(add_pair)
        pair_buttons.addWidget(remove_pair)
        pair_buttons.addWidget(clear_pair)
        pair_buttons.addStretch(1)

        add_task = QtWidgets.QPushButton("Add Task to Queue")
        add_task.clicked.connect(self.add_task_from_builder)
        self.update_task_button = QtWidgets.QPushButton("Update Selected Task")
        self.update_task_button.clicked.connect(self.update_selected_task_from_builder)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(add_task)
        buttons.addWidget(self.update_task_button)
        buttons.addStretch(1)

        form.addRow("Task name", self.task_name_edit)
        form.addRow("Raw folder fallback", raw_row)
        form.addRow("Detector mode", self.detector_mode_combo)
        form.addRow("Reduction mode", self.reduction_mode_combo)
        form.addRow("Pil300K raw HDF5 files", self.pil_files_edit)
        form.addRow("", pil_file_buttons)
        form.addRow("Eig1M raw HDF5 files", self.eig_files_edit)
        form.addRow("", eig_file_buttons)
        form.addRow("Output folder (auto)", output_row)
        form.addRow("File counts", counts_row)
        form.addRow("Sequence", sequence_row)
        form.addRow("Pil300K PONI", pil_poni_row)
        form.addRow("Pil300K mask", pil_mask_row)
        form.addRow("Eig1M PONI", eig_poni_row)
        form.addRow("Eig1M mask", eig_mask_row)
        form.addRow("Capillary thickness", self.capillary_spin)
        form.addRow("GC thickness", self.gc_thickness_spin)
        form.addRow("GC group", self.gc_group_spin)
        form.addRow("Air group", self.air_group_spin)
        form.addRow("Empty group", self.empty_group_spin)
        form.addRow("CPU cores", self.core_spin)
        self.pair_table_label = QtWidgets.QLabel("ASAXS sample/solvent pairs")
        layout.addWidget(self.pair_table_label)
        layout.addWidget(self.pair_table)
        layout.addLayout(pair_buttons)
        layout.addLayout(buttons)
        layout.addStretch(1)

        self._append_pair_row("10pYb", "5", "4")
        self._append_pair_row("5pYb", "6", "4")
        self._connect_builder_autosave()
        self.tabs.addTab(page, "Task Builder")

    @staticmethod
    def _spin(low: int, high: int, value: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        return spin

    @staticmethod
    def _double_spin(low: float, high: float, value: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(5)
        spin.setValue(value)
        return spin

    def _file_browse_row(self, edit: QtWidgets.QLineEdit, title: str, file_filter: str) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(lambda _checked=False, target=edit, dialog_title=title, filt=file_filter: self.browse_path_file(target, dialog_title, filt))
        layout.addWidget(edit, 1)
        layout.addWidget(browse)
        return row

    def _append_pair_row(self, name: str, sample: str, solvent: str) -> None:
        row = self.pair_table.rowCount()
        self.pair_table.insertRow(row)
        for column, value in enumerate([name, sample, solvent]):
            self.pair_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        self.schedule_save_builder_settings()

    def _connect_builder_autosave(self) -> None:
        for edit in [
            self.task_name_edit,
            self.raw_folder_edit,
            self.output_dir_edit,
            self.pil_poni_edit,
            self.pil_mask_edit,
            self.eig_poni_edit,
            self.eig_mask_edit,
        ]:
            edit.textChanged.connect(self.schedule_save_builder_settings)
        self.detector_mode_combo.currentIndexChanged.connect(self._detector_mode_changed)
        self.reduction_mode_combo.currentIndexChanged.connect(self._reduction_mode_changed)
        self.raw_folder_edit.textChanged.connect(self._raw_folder_changed)
        for edit in [self.pil_files_edit, self.eig_files_edit]:
            edit.textChanged.connect(self._selected_files_changed)
        for spin in [
            self.energy_spin,
            self.group_spin,
            self.frame_spin,
            self.gc_group_spin,
            self.air_group_spin,
            self.empty_group_spin,
            self.core_spin,
        ]:
            spin.valueChanged.connect(self.schedule_save_builder_settings)
        for spin in [self.capillary_spin, self.gc_thickness_spin]:
            spin.valueChanged.connect(self.schedule_save_builder_settings)
        self.pair_table.itemChanged.connect(self.schedule_save_builder_settings)

    def schedule_save_builder_settings(self, *_args: object) -> None:
        if self._loading_builder_settings:
            return
        QtCore.QTimer.singleShot(250, self.save_builder_settings)

    def builder_settings_payload(self) -> dict[str, object]:
        pairs: list[dict[str, str]] = []
        for row in range(self.pair_table.rowCount()):
            pairs.append(
                {
                    "output_name": self._table_text(self.pair_table, row, 0),
                    "sample_group": self._table_text(self.pair_table, row, 1),
                    "solvent_group": self._table_text(self.pair_table, row, 2),
                }
            )
        return {
            "task_name": self.task_name_edit.text(),
            "raw_folder": self.raw_folder_edit.text(),
            "output_dir": self.output_dir_edit.text(),
            "detector_mode": self.detector_mode_combo.currentData(),
            "reduction_mode": self.reduction_mode_combo.currentData(),
            "pil300k_files": self._files_from_edit(self.pil_files_edit),
            "eig1m_files": self._files_from_edit(self.eig_files_edit),
            "num_energies": self.energy_spin.value(),
            "num_groups": self.group_spin.value(),
            "num_frames": self.frame_spin.value(),
            "pil300k_poni": self.pil_poni_edit.text(),
            "pil300k_mask": self.pil_mask_edit.text(),
            "eig1m_poni": self.eig_poni_edit.text(),
            "eig1m_mask": self.eig_mask_edit.text(),
            "capillary_thickness": self.capillary_spin.value(),
            "gc_thickness": self.gc_thickness_spin.value(),
            "gc_group": self.gc_group_spin.value(),
            "air_group": self.air_group_spin.value(),
            "empty_group": self.empty_group_spin.value(),
            "cores": self.core_spin.value(),
            "pairs": pairs,
        }

    @staticmethod
    def _table_text(table: QtWidgets.QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        return item.text() if item else ""

    def save_builder_settings(self) -> None:
        if self._loading_builder_settings:
            return
        BUILDER_SETTINGS_PATH.write_text(json.dumps(self.builder_settings_payload(), indent=2), encoding="utf-8")

    def load_builder_settings(self) -> None:
        if not BUILDER_SETTINGS_PATH.exists():
            return
        try:
            payload = json.loads(BUILDER_SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.log(f"Could not load builder settings: {exc}")
            return
        self.apply_builder_settings(payload)

    def apply_builder_settings(self, payload: dict[str, object]) -> None:
        self._loading_builder_settings = True
        try:
            self.task_name_edit.setText(str(payload.get("task_name", "")))
            self.raw_folder_edit.setText(str(payload.get("raw_folder", "")))
            self.output_dir_edit.setText(str(payload.get("output_dir", "")))
            self._set_detector_mode(str(payload.get("detector_mode", "both")))
            self._set_reduction_mode(str(payload.get("reduction_mode", "asaxs")))
            self._set_files_edit(self.pil_files_edit, payload.get("pil300k_files", []))
            self._set_files_edit(self.eig_files_edit, payload.get("eig1m_files", []))
            self.energy_spin.setValue(int(payload.get("num_energies", self.energy_spin.value())))
            self.group_spin.setValue(int(payload.get("num_groups", self.group_spin.value())))
            self.frame_spin.setValue(int(payload.get("num_frames", self.frame_spin.value())))
            self.pil_poni_edit.setText(str(payload.get("pil300k_poni", self.pil_poni_edit.text())))
            self.pil_mask_edit.setText(str(payload.get("pil300k_mask", self.pil_mask_edit.text())))
            self.eig_poni_edit.setText(str(payload.get("eig1m_poni", self.eig_poni_edit.text())))
            self.eig_mask_edit.setText(str(payload.get("eig1m_mask", self.eig_mask_edit.text())))
            self.capillary_spin.setValue(float(payload.get("capillary_thickness", self.capillary_spin.value())))
            self.gc_thickness_spin.setValue(float(payload.get("gc_thickness", self.gc_thickness_spin.value())))
            self.gc_group_spin.setValue(int(payload.get("gc_group", self.gc_group_spin.value())))
            self.air_group_spin.setValue(int(payload.get("air_group", self.air_group_spin.value())))
            self.empty_group_spin.setValue(int(payload.get("empty_group", self.empty_group_spin.value())))
            saved_cores = int(payload.get("cores", self.core_spin.value()))
            if saved_cores <= 1 and max(1, os.cpu_count() or 1) > 1:
                saved_cores = max(1, os.cpu_count() or 1)
            self.core_spin.setValue(saved_cores)
            pairs = payload.get("pairs")
            if isinstance(pairs, list):
                self.pair_table.setRowCount(0)
                for pair in pairs:
                    if not isinstance(pair, dict):
                        continue
                    self._append_pair_row(
                        str(pair.get("output_name", "")),
                        str(pair.get("sample_group", "")),
                        str(pair.get("solvent_group", "")),
                    )
        finally:
            self._loading_builder_settings = False
        if not self.output_dir_edit.text().strip():
            self._update_auto_output_dir(force=True)
        self.scan_builder_folder()

    def remove_selected_pair(self) -> None:
        rows = sorted({index.row() for index in self.pair_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.pair_table.removeRow(row)
        self.schedule_save_builder_settings()

    def clear_pair_rows(self) -> None:
        self.pair_table.setRowCount(0)
        self.schedule_save_builder_settings()

    def _set_detector_mode(self, mode: str) -> None:
        index = self.detector_mode_combo.findData(mode if mode in {"both", "pil300k", "eig1m"} else "both")
        self.detector_mode_combo.setCurrentIndex(max(0, index))

    def _detector_mode_changed(self) -> None:
        self.scan_builder_folder()
        self.schedule_save_builder_settings()

    def _set_reduction_mode(self, mode: str) -> None:
        index = self.reduction_mode_combo.findData(mode if mode in {"asaxs", "saxs"} else "asaxs")
        self.reduction_mode_combo.setCurrentIndex(max(0, index))

    def _reduction_mode_changed(self) -> None:
        is_asaxs = self.reduction_mode_combo.currentData() == "asaxs"
        self.pair_table.setEnabled(True)
        self.pair_table_label.setText(
            "ASAXS sample/solvent pairs" if is_asaxs else "SAXS XAnos output name (first row used; groups ignored)"
        )
        self.schedule_save_builder_settings()

    def open_rack_builder(self) -> None:
        dialog = RackBuilderDialog(
            self,
            group_count=self.group_spin.value(),
            gc_group=self._optional_group(self.gc_group_spin.value()),
            air_group=self._optional_group(self.air_group_spin.value()),
            empty_group=self._optional_group(self.empty_group_spin.value()),
            pairs=self._current_pair_rows(),
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        result = dialog.result_payload()
        self.gc_group_spin.setValue(result.gc_group or 0)
        self.air_group_spin.setValue(result.air_group or 0)
        self.empty_group_spin.setValue(result.empty_group or 0)
        self.pair_table.setRowCount(0)
        for output_name, sample_group, solvent_group in result.pairs:
            self._append_pair_row(output_name, str(sample_group), str(solvent_group))
        self.save_builder_settings()

    def _current_pair_rows(self) -> list[tuple[str, int, int]]:
        pairs: list[tuple[str, int, int]] = []
        for row in range(self.pair_table.rowCount()):
            name = self._table_text(self.pair_table, row, 0).strip()
            sample = self._table_text(self.pair_table, row, 1).strip()
            solvent = self._table_text(self.pair_table, row, 2).strip()
            if not name and not sample and not solvent:
                continue
            if sample.isdigit() and solvent.isdigit():
                pairs.append((name or f"sample_{sample}", int(sample), int(solvent)))
        return pairs

    def browse_raw_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose raw sample folder", self.raw_folder_edit.text() or str(Path.home()))
        if not folder:
            return
        self.raw_folder_edit.setText(folder)
        path = Path(folder)
        if not self.task_name_edit.text().strip():
            self.task_name_edit.setText(path.name)
        self._update_auto_output_dir(force=True, sample_root=path)
        self.scan_builder_folder()
        self.save_builder_settings()

    def browse_raw_files(self) -> None:
        self.browse_detector_files("Pil300K")
        self.browse_detector_files("Eig1M")

    def browse_detector_files(self, detector: str) -> None:
        edit = self.pil_files_edit if detector == "Pil300K" else self.eig_files_edit
        start = self._detector_file_start_dir(detector)
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, f"Choose {detector} raw HDF5 files", str(start), "HDF5 files (*.h5 *.hdf5);;All files (*)")
        if not files:
            return
        sorted_files = [str(path) for path in sort_h5_files(files)]
        edit.setPlainText("\n".join(sorted_files))
        if not self.raw_folder_edit.text().strip():
            parent = self._common_parent(sorted_files)
            if parent is not None:
                self.raw_folder_edit.setText(str(parent.parent if parent.name in {"Pil300K", "Eig1M"} else parent))
        if not self.task_name_edit.text().strip():
            folder = self._common_parent(sorted_files)
            if folder is not None:
                self.task_name_edit.setText(folder.parent.name if folder.name in {"Pil300K", "Eig1M"} else folder.name)
        self._update_auto_output_dir(force=True, sample_root=self._sample_root_from_files(sorted_files))
        self.scan_builder_folder()
        self.save_builder_settings()

    def clear_detector_files(self, detector: str) -> None:
        edit = self.pil_files_edit if detector == "Pil300K" else self.eig_files_edit
        edit.clear()
        self.scan_builder_folder()
        self.save_builder_settings()

    def _detector_file_start_dir(self, detector: str) -> Path:
        files = self._files_from_edit(self.pil_files_edit if detector == "Pil300K" else self.eig_files_edit)
        if files:
            return Path(files[0]).parent
        raw = self.raw_folder_edit.text().strip()
        if raw:
            detector_dir = Path(raw) / detector
            return detector_dir if detector_dir.exists() else Path(raw)
        return Path.home()

    @staticmethod
    def _common_parent(files: list[str]) -> Path | None:
        if not files:
            return None
        try:
            common = Path(os.path.commonpath(files))
        except ValueError:
            common = Path(files[0]).parent
        return common.parent if common.suffix.lower() in {".h5", ".hdf5"} else common

    def browse_output_dir(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_dir_edit.text() or str(Path.home()))
        if folder:
            self._output_dir_manually_overridden = True
            self.output_dir_edit.setText(folder)
            self.save_builder_settings()

    def _update_auto_output_dir(self, *, force: bool = False, sample_root: Path | None = None) -> None:
        if self._output_dir_manually_overridden and not force:
            return
        sample_root = sample_root or self._builder_sample_root()
        if sample_root is None:
            return
        sample_root = sample_root.expanduser()
        self.output_dir_edit.setText(str(sample_root.parent / "Extracted" / sample_root.name))
        self._output_dir_manually_overridden = False

    def _builder_sample_root(self) -> Path | None:
        pil_files = self._files_from_edit(self.pil_files_edit)
        eig_files = self._files_from_edit(self.eig_files_edit)
        file_root = self._sample_root_from_files(pil_files or eig_files)
        if file_root is not None:
            return file_root
        raw = self.raw_folder_edit.text().strip()
        return Path(raw).expanduser() if raw else None

    @staticmethod
    def _sample_root_from_files(files: list[str]) -> Path | None:
        folder = DashboardWindow._common_parent(files)
        if folder is None:
            return None
        return folder.parent if folder.name in {"Pil300K", "Eig1M"} else folder

    def browse_path_file(self, edit: QtWidgets.QLineEdit, title: str, file_filter: str) -> None:
        current = edit.text().strip()
        start = str(Path(current).expanduser().parent) if current else str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, start, file_filter)
        if not path:
            return
        edit.setText(path)
        self.save_builder_settings()

    def scan_builder_folder(self) -> None:
        pil_files = self._files_from_edit(self.pil_files_edit)
        eig_files = self._files_from_edit(self.eig_files_edit)
        if pil_files or eig_files:
            pil, eig = len(pil_files), len(eig_files)
        else:
            raw_folder = self.raw_folder_edit.text().strip()
            if not raw_folder:
                pil, eig = 0, 0
            else:
                pil, eig = scan_detector_files(Path(raw_folder), "*.h5")
        source = "selected" if pil_files or eig_files else "folder"
        self.pil_count_label.setText(f"Pil300K: {pil} ({source})")
        self.eig_count_label.setText(f"Eig1M: {eig} ({source})")

    def _selected_files_changed(self) -> None:
        if not self._loading_builder_settings:
            self._update_auto_output_dir(force=True)
        self.scan_builder_folder()
        self.schedule_save_builder_settings()

    def _raw_folder_changed(self) -> None:
        if not self._loading_builder_settings:
            raw = self.raw_folder_edit.text().strip()
            self._update_auto_output_dir(sample_root=Path(raw) if raw else None)

    @staticmethod
    def _files_from_edit(edit: QtWidgets.QPlainTextEdit) -> list[str]:
        return [line.strip() for line in edit.toPlainText().splitlines() if line.strip()]

    @staticmethod
    def _set_files_edit(edit: QtWidgets.QPlainTextEdit, files: object) -> None:
        if isinstance(files, list):
            edit.setPlainText("\n".join(str(path) for path in files if str(path).strip()))
        else:
            edit.clear()

    def builder_task(self) -> TaskSpec:
        self.scan_builder_folder()
        if not self.output_dir_edit.text().strip():
            self._update_auto_output_dir(force=True)
        pairs: list[AsaxsPair] = []
        is_asaxs = self.reduction_mode_combo.currentData() == "asaxs"
        for row in range(self.pair_table.rowCount()):
            name_item = self.pair_table.item(row, 0)
            sample_item = self.pair_table.item(row, 1)
            solvent_item = self.pair_table.item(row, 2)
            name = name_item.text().strip() if name_item else ""
            sample = sample_item.text().strip() if sample_item else ""
            solvent = solvent_item.text().strip() if solvent_item else ""
            if not name and not sample and not solvent:
                continue
            if not is_asaxs:
                if name:
                    pairs.append(AsaxsPair(name, 0, 0))
                continue
            if not sample.isdigit() or not solvent.isdigit():
                raise ValueError(f"ASAXS pair row {row + 1} needs numeric sample and solvent groups.")
            pairs.append(AsaxsPair(name or f"sample_{sample}", int(sample), int(solvent)))
        pil_files = self._files_from_edit(self.pil_files_edit)
        eig_files = self._files_from_edit(self.eig_files_edit)
        if pil_files or eig_files:
            pil, eig = len(pil_files), len(eig_files)
        else:
            pil, eig = scan_detector_files(Path(self.raw_folder_edit.text()), "*.h5")
        return TaskSpec(
            task_name=self.task_name_edit.text().strip() or safe_name(Path(self.raw_folder_edit.text()).name),
            raw_folder=self.raw_folder_edit.text().strip(),
            output_dir=self.output_dir_edit.text().strip(),
            num_energies=self.energy_spin.value(),
            num_groups=self.group_spin.value(),
            num_frames=self.frame_spin.value(),
            pil300k_poni=self.pil_poni_edit.text().strip(),
            pil300k_mask=self.pil_mask_edit.text().strip(),
            eig1m_poni=self.eig_poni_edit.text().strip(),
            eig1m_mask=self.eig_mask_edit.text().strip(),
            pil300k_files=pil_files,
            eig1m_files=eig_files,
            detector_mode=str(self.detector_mode_combo.currentData() or "both"),
            reduction_mode=str(self.reduction_mode_combo.currentData() or "asaxs"),
            capillary_thickness=self.capillary_spin.value(),
            gc_thickness=self.gc_thickness_spin.value(),
            gc_group=self._optional_group(self.gc_group_spin.value()),
            air_group=self._optional_group(self.air_group_spin.value()),
            empty_group=self._optional_group(self.empty_group_spin.value()),
            cores=self.core_spin.value(),
            asaxs_pairs=pairs,
            pil300k_count=pil,
            eig1m_count=eig,
        )

    @staticmethod
    def _optional_group(value: int) -> int | None:
        return value if value > 0 else None

    def add_task_from_builder(self) -> None:
        import copy

        try:
            task = self.builder_task()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Task is incomplete", str(exc))
            return
        self.save_builder_settings()
        ok, message = preflight_task(task)
        task.status = "Ready" if ok else "Needs Attention"
        task.message = message
        self.tasks.append(copy.deepcopy(task))
        self.editing_index = None
        new_index = len(self.tasks) - 1
        self.refresh_queue(select_row=new_index)
        self.queue_table.scrollToItem(self.queue_table.item(new_index, 0))
        self.log(f"Added task: {task.task_name} ({message})")
        self.tabs.setCurrentIndex(0)
        if ok:
            self.statusBar().showMessage(f"Added task to queue: {task.task_name}")
        else:
            self.statusBar().showMessage(f"Task needs attention: {message}")
            self._show_validation_failed(task.task_name, message)

    def update_selected_task_from_builder(self) -> None:
        index = self.editing_index if self.editing_index is not None else self.selected_index()
        if index is None or not (0 <= index < len(self.tasks)):
            QtWidgets.QMessageBox.information(self, "No task selected", "Select a queue task to update.")
            return
        try:
            task = self.builder_task()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Task is incomplete", str(exc))
            return
        self.save_builder_settings()
        ok, message = preflight_task(task)
        task.status = "Ready" if ok else "Needs Attention"
        task.message = message
        self.tasks[index] = task
        self.editing_index = None
        self.refresh_queue(select_row=index)
        self.log(f"Updated task: {task.task_name} ({message})")
        self.tabs.setCurrentIndex(0)
        if not ok:
            self.statusBar().showMessage(f"Task needs attention: {message}")
            self._show_validation_failed(task.task_name, message)

    def clear_builder(self) -> None:
        self.editing_index = None
        self.reset_builder_to_defaults()
        self.tabs.setCurrentIndex(1)

    def reset_builder_to_defaults(self) -> None:
        self._loading_builder_settings = True
        try:
            self.task_name_edit.clear()
            self.raw_folder_edit.clear()
            self.pil_files_edit.clear()
            self.eig_files_edit.clear()
            self.output_dir_edit.clear()
            self._output_dir_manually_overridden = False
            self.energy_spin.setValue(20)
            self.group_spin.setValue(13)
            self.frame_spin.setValue(10)
            self.pil_poni_edit.clear()
            self.pil_mask_edit.clear()
            self.eig_poni_edit.clear()
            self.eig_mask_edit.clear()
            self.capillary_spin.setValue(0.15)
            self.gc_thickness_spin.setValue(0.1055)
            self.gc_group_spin.setValue(1)
            self.air_group_spin.setValue(2)
            self.empty_group_spin.setValue(3)
            self.core_spin.setValue(max(1, os.cpu_count() or 1))
            self.detector_mode_combo.setCurrentIndex(0)
            self.reduction_mode_combo.setCurrentIndex(0)
            self.pair_table.setRowCount(0)
            self._append_pair_row("10pYb", "5", "4")
            self._append_pair_row("5pYb", "6", "4")
            self.pil_count_label.setText("Pil300K: 0")
            self.eig_count_label.setText("Eig1M: 0")
        finally:
            self._loading_builder_settings = False
        self.save_builder_settings()

    def refresh_queue(self, select_row: int | None = None, select_rows: list[int] | None = None) -> None:
        if select_rows is None and select_row is None:
            select_rows = self.selected_indices()
        self.queue_table.setRowCount(len(self.tasks))
        for row, task in enumerate(self.tasks):
            self._update_queue_row(row)
        if select_rows and self.tasks:
            self.queue_table.clearSelection()
            selection_model = self.queue_table.selectionModel()
            selected_rows = sorted({max(0, min(value, len(self.tasks) - 1)) for value in select_rows})
            for row in selected_rows:
                if selection_model is None:
                    self.queue_table.selectRow(row)
                else:
                    top_left = self.queue_table.model().index(row, 0)
                    bottom_right = self.queue_table.model().index(row, self.queue_table.columnCount() - 1)
                    selection = QtCore.QItemSelection(top_left, bottom_right)
                    selection_model.select(selection, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
            if selected_rows and selection_model is not None:
                selection_model.setCurrentIndex(
                    self.queue_table.model().index(selected_rows[0], 0),
                    QtCore.QItemSelectionModel.NoUpdate,
                )
            return
        if select_row is not None and self.tasks:
            select_row = max(0, min(select_row, len(self.tasks) - 1))
            self.queue_table.selectRow(select_row)

    def set_queue_locked(self, locked: bool, reason: str = "") -> None:
        self.queue_locked = locked
        self.queue_table.locked = locked
        self.queue_table.setAcceptDrops(not locked)
        self.queue_table.setDragEnabled(not locked)
        self.queue_table.setSelectionMode(self._queue_selection_mode)
        self.queue_table.setContextMenuPolicy(QtCore.Qt.NoContextMenu if locked else QtCore.Qt.CustomContextMenu)
        for action in [
            self.new_queue_action,
            self.clear_queue_action,
            self.open_queue_action,
            self.add_to_queue_action,
            self.update_task_action,
            self.edit_task_action,
            self.copy_task_action,
            self.duplicate_task_action,
            self.remove_task_action,
            self.paste_task_action,
            self.set_status_action,
            self.move_up_action,
            self.move_down_action,
            self.run_selected_action,
            self.run_all_action,
        ]:
            action.setEnabled(not locked)
        cursor = QtCore.Qt.ForbiddenCursor if locked else QtCore.Qt.ArrowCursor
        self.queue_table.setCursor(cursor)
        self.queue_table.viewport().setCursor(cursor)
        self.queue_lock_label.setText(reason or "Queue locked while running. Use Stop Queue to interrupt.")
        self.queue_lock_label.setVisible(locked)
        self.queue_table.setStyleSheet(
            """
            QTableWidget {
                background: #eeeeee;
                color: #777777;
                gridline-color: #d0d0d0;
            }
            QTableWidget::item {
                background: #eeeeee;
                color: #777777;
            }
            QHeaderView::section {
                background: #e0e0e0;
                color: #777777;
            }
            """
            if locked
            else ""
        )

    def _queue_mutation_blocked(self, action: str) -> bool:
        if not self.queue_locked:
            return False
        self._queue_command_notice(f"Queue is locked while running; cannot {action}.")
        return True

    def _queue_row_values(self, task: TaskSpec) -> list[str]:
        return [
            task.task_name,
            task.status,
            f"{task.pil300k_count} Pil + {task.eig1m_count} Eig",
            task.sequence_label,
            f"{task.detector_label} ({task.source_label})",
            task.pair_label,
            str(task.output_path),
        ]

    def _update_queue_row(self, row: int) -> None:
        if not (0 <= row < len(self.tasks)) or row >= self.queue_table.rowCount():
            return
        task = self.tasks[row]
        tooltip = self._task_tooltip(task)
        for col, value in enumerate(self._queue_row_values(task)):
            item = self.queue_table.item(row, col)
            if item is None:
                item = QtWidgets.QTableWidgetItem()
                self.queue_table.setItem(row, col, item)
            item.setText(value)
            item.setToolTip(tooltip)
            if self.queue_locked:
                item.setBackground(QtGui.QColor("#eeeeee"))
                item.setForeground(QtGui.QColor("#777777"))
            else:
                self._style_queue_item(item, task.status)

    def selected_index(self) -> int | None:
        rows = self.selected_indices()
        if rows:
            return rows[0]
        return 0 if self.tasks else None

    def selected_indices(self) -> list[int]:
        rows = self.queue_table.selectionModel().selectedRows() if self.queue_table.selectionModel() else []
        values = sorted({index.row() for index in rows if 0 <= index.row() < len(self.tasks)})
        return values

    def _sync_selection_from_table(self) -> None:
        self.refresh_current_curves()

    def _style_queue_item(self, item: QtWidgets.QTableWidgetItem, status: str) -> None:
        palette = {
            "Ready": (QtGui.QColor("#f7f7f7"), QtGui.QColor("#111111")),
            "Queued": (QtGui.QColor("#fff4cc"), QtGui.QColor("#111111")),
            "Running": (QtGui.QColor("#cfe8ff"), QtGui.QColor("#000000")),
            "Done": (QtGui.QColor("#dff3df"), QtGui.QColor("#0b3d0b")),
            "Failed": (QtGui.QColor("#ffd9d9"), QtGui.QColor("#6b0000")),
            "Stopped": (QtGui.QColor("#eadfff"), QtGui.QColor("#2b145f")),
            "Skipped": (QtGui.QColor("#e6e6e6"), QtGui.QColor("#333333")),
            "Needs Attention": (QtGui.QColor("#ffe1b8"), QtGui.QColor("#5a2b00")),
        }
        background, foreground = palette.get(status, (QtGui.QColor("#ffffff"), QtGui.QColor("#111111")))
        item.setBackground(background)
        item.setForeground(foreground)
        if status in {"Running", "Failed", "Needs Attention"}:
            font = item.font()
            font.setBold(True)
            item.setFont(font)

    def _task_tooltip(self, task: TaskSpec) -> str:
        return "\n".join([
            f"Task: {task.task_name}",
            f"Status: {task.status}",
            f"Message: {task.message or '-'}",
            f"Detector mode: {task.detector_label}",
            f"Reduction mode: {'ASAXS / XAnos' if task.is_asaxs_mode() else 'SAXS only'}",
            f"Source: {task.source_label}",
            f"Raw folder fallback: {task.raw_folder}",
            f"Selected Pil300K files: {len(task.pil300k_files)}",
            f"Selected Eig1M files: {len(task.eig1m_files)}",
            f"Output: {task.output_dir}",
            f"Pil300K files: {task.pil300k_count}",
            f"Eig1M files: {task.eig1m_count}",
            f"Sequence: {task.sequence_label}",
            f"PONI/mask: {task.detector_label} configured",
            f"Thickness: capillary={task.capillary_thickness}, GC={task.gc_thickness}",
            f"Groups: GC={task.gc_group}, air={task.air_group}, empty={task.empty_group}",
            f"Pairs: {task.pair_label}",
        ])

    def refresh_current_curves(self) -> None:
        if not hasattr(self, "current_curve_plot"):
            return
        self.current_curve_plot.clear()
        self.current_curve_legend.clear()
        self.current_curve_plot.setLogMode(x=self.curve_log_x_check.isChecked(), y=self.curve_log_y_check.isChecked())
        self.current_curve_plot.setLabel("bottom", "q", units="A^-1")
        self.current_curve_plot.setLabel("left", "I(q)", units="a.u.")
        index = self.selected_index()
        if index is None:
            self.current_curve_plot.setTitle("No task selected")
            self.current_curve_status.setText("Select a task to show final reduced curves.")
            return
        task = self.tasks[index]
        try:
            curves, source_label, analysis_h5 = self._current_curve_payloads(task)
        except (OSError, RuntimeError, ValueError) as exc:
            self.current_curve_plot.setTitle("Could not load curves")
            self.current_curve_status.setText(str(exc))
            return
        if not curves:
            self.current_curve_plot.setTitle("Waiting for final reduced curves")
            self.current_curve_status.setText(f"No final reduced curves found yet for {task.task_name}.")
            return
        plotted = 0
        for label, q, intensity in curves:
            q = np.asarray(q, dtype=float).reshape(-1)
            intensity = np.asarray(intensity, dtype=float).reshape(-1)
            mask = np.isfinite(q) & np.isfinite(intensity)
            if self.curve_log_x_check.isChecked():
                mask &= q > 0
            if self.curve_log_y_check.isChecked():
                mask &= intensity > 0
            if np.count_nonzero(mask) < 2:
                continue
            pen = pg.mkPen(pg.intColor(plotted, hues=max(8, len(curves))), width=1.4)
            self.current_curve_plot.plot(q[mask], intensity[mask], pen=pen, name=label)
            plotted += 1
        title = f"{task.task_name} - {source_label}"
        self.current_curve_plot.setTitle(title)
        self.current_curve_status.setText(f"{plotted}/{len(curves)} curves shown from {analysis_h5}")

    def _current_curve_payloads(self, task: TaskSpec) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], str, Path]:
        analysis_h5 = self._find_task_analysis_h5(task)
        if analysis_h5 is None:
            return [], "No analysis HDF5", task.combined_h5_path()
        with h5py.File(analysis_h5, "r") as handle:
            curves = self._read_final_curve_payloads(handle, include_saxs_final=task.is_saxs_mode())
            source_label = "Final SAXS" if task.is_saxs_mode() else "Final ASAXS"
            return curves[: self.curve_max_spin.value()], source_label, analysis_h5

    def _find_task_analysis_h5(self, task: TaskSpec) -> Path | None:
        expected = task.combined_h5_path()
        if expected.exists():
            return expected
        candidates = sorted(task.output_path.glob("*_analysis.h5"), key=lambda path: path.stat().st_mtime_ns if path.exists() else 0)
        return candidates[-1] if candidates else None

    def _read_final_curve_payloads(self, handle: h5py.File, include_saxs_final: bool = False) -> list[tuple[str, np.ndarray, np.ndarray]]:
        curves: list[tuple[str, np.ndarray, np.ndarray]] = []
        named = handle.get("/entry/asaxs_outputs")
        if isinstance(named, h5py.Group):
            for output_name in sorted(named):
                group = named[output_name].get("corrected_I_q_E")
                curves.extend(self._rows_from_q_i_group(group, output_name))
        if curves:
            return curves
        group = handle.get("/entry/final/corrected_I_q_E")
        curves = self._rows_from_q_i_group(group, "final")
        if curves or not include_saxs_final:
            return curves
        return self._read_stitched_curve_payloads(handle)

    def _read_stitched_curve_payloads(self, handle: h5py.File) -> list[tuple[str, np.ndarray, np.ndarray]]:
        root = handle.get("/entry/stitched_averages/curves")
        if not isinstance(root, h5py.Group):
            return []
        curves: list[tuple[str, np.ndarray, np.ndarray]] = []
        for name in sorted(root):
            group = root[name]
            if not isinstance(group, h5py.Group) or "q" not in group or "I" not in group:
                continue
            energy = group.attrs.get("energy_kev", np.nan)
            energy_label = f" {float(energy):.4f} keV" if np.isfinite(energy) else ""
            curves.append((f"{name}{energy_label}", np.asarray(group["q"][()], dtype=float), np.asarray(group["I"][()], dtype=float)))
        if len(curves) > self.curve_max_spin.value():
            return curves[-self.curve_max_spin.value() :]
        return curves

    def _read_detector_reduction_payloads(self, handle: h5py.File) -> list[tuple[str, np.ndarray, np.ndarray]]:
        group = handle.get("/entry/process_01_reduction/data")
        if not isinstance(group, h5py.Group) or "q" not in group or "I" not in group:
            return []
        q = np.asarray(group["q"][()], dtype=float)
        intensity = np.asarray(group["I"][()], dtype=float)
        energy = np.asarray(group["energy"][()], dtype=float) if "energy" in group else np.full((intensity.shape[0] if intensity.ndim > 1 else 1,), np.nan)
        group_index = np.asarray(group["group_index"][()], dtype=int) if "group_index" in group else np.arange(1, intensity.shape[0] + 1)
        if intensity.ndim == 1:
            intensity = intensity.reshape(1, -1)
        curves: list[tuple[str, np.ndarray, np.ndarray]] = []
        for row in range(intensity.shape[0]):
            q_row = self._q_for_curve_row(q, row)
            energy_value = float(energy[row]) if row < energy.size else float("nan")
            group_value = int(group_index[row]) if row < group_index.size else row + 1
            energy_label = f", {energy_value:.4f} keV" if np.isfinite(energy_value) else ""
            curves.append((f"group {group_value:02d}{energy_label}", q_row, intensity[row]))
        return curves

    def _rows_from_q_i_group(self, group: h5py.Group | None, label_prefix: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
        if not isinstance(group, h5py.Group) or "q" not in group or "I" not in group:
            return []
        q = np.asarray(group["q"][()], dtype=float)
        intensity = np.asarray(group["I"][()], dtype=float)
        energy = np.asarray(group["energy"][()], dtype=float) if "energy" in group else np.full((intensity.shape[0] if intensity.ndim > 1 else 1,), np.nan)
        if intensity.ndim == 1:
            intensity = intensity.reshape(1, -1)
        rows: list[tuple[str, np.ndarray, np.ndarray]] = []
        for row in range(intensity.shape[0]):
            q_row = self._q_for_curve_row(q, row)
            energy_value = float(energy[row]) if row < energy.size else float("nan")
            energy_label = f" {energy_value:.4f} keV" if np.isfinite(energy_value) else f" row {row + 1:03d}"
            rows.append((f"{label_prefix}{energy_label}", q_row, intensity[row]))
        return rows

    @staticmethod
    def _q_for_curve_row(q: np.ndarray, row: int) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.ndim > 1:
            return q[row] if row < q.shape[0] else q[0]
        return q

    def new_queue(self) -> None:
        self.tasks.clear()
        self.refresh_queue()
        self.log("New queue created.")

    def clear_queue(self) -> None:
        if self.runner is not None and self.runner.isRunning():
            self._queue_command_notice("Stop the current queue before clearing it.")
            return
        if not self.tasks:
            self._queue_command_notice("Queue is already empty.")
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Clear Queue",
            "Remove all tasks from the queue?\n\nThis does not delete raw data or output files.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.tasks.clear()
        self.active_run_indices = []
        self.editing_index = None
        self.refresh_queue()
        self._reset_task_progress()
        self.current_stage_label.setText("Idle")
        self.log("Queue cleared.")

    def _load_default_queue(self) -> None:
        if self.queue_path.exists():
            self.tasks = load_queue(self.queue_path)
            self._reset_stale_running_tasks()
            self.refresh_queue()

    def open_queue(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open queue", str(PROJECT_DIR), "JSON files (*.json)")
        if path:
            self.queue_path = Path(path)
            self.tasks = load_queue(self.queue_path)
            self._reset_stale_running_tasks()
            self.refresh_queue()

    def _reset_stale_running_tasks(self) -> None:
        for task in self.tasks:
            if task.status == "Running":
                task.status = "Ready"
                task.message = "Reset from stale Running state after queue reload"

    def save_queue(self) -> None:
        save_queue(self.queue_path, self.tasks)
        self.statusBar().showMessage(f"Queue saved: {self.queue_path}")

    def save_queue_as(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save queue as", str(self.queue_path), "JSON files (*.json)")
        if path:
            self.queue_path = Path(path)
            self.save_queue()

    def show_queue_context_menu(self, position: QtCore.QPoint) -> None:
        row = self.queue_table.rowAt(position.y())
        selected = self.selected_indices()
        if row >= 0 and row not in selected:
            self.queue_table.selectRow(row)
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.edit_task_action)
        menu.addSeparator()
        menu.addAction(self.run_selected_action)
        menu.addAction(self.stop_current_action)
        menu.addSeparator()
        menu.addAction(self.copy_task_action)
        menu.addAction(self.paste_task_action)
        menu.addAction(self.duplicate_task_action)
        menu.addSeparator()
        menu.addAction(self.move_up_action)
        menu.addAction(self.move_down_action)
        menu.addAction(self.open_output_action)
        menu.addSeparator()
        menu.addAction(self.set_status_action)
        menu.addAction(self.remove_task_action)
        menu.exec_(self.queue_table.viewport().mapToGlobal(position))

    def edit_selected_task(self) -> None:
        index = self.selected_index()
        if index is None:
            return
        self.editing_index = index
        self.populate_builder_from_task(self.tasks[index])
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage(f"Editing queue task: {self.tasks[index].task_name}")

    def populate_builder_from_task(self, task: TaskSpec) -> None:
        self._loading_builder_settings = True
        try:
            self.task_name_edit.setText(task.task_name)
            self.raw_folder_edit.setText(task.raw_folder)
            self.output_dir_edit.setText(task.output_dir)
            self._output_dir_manually_overridden = bool(task.output_dir)
            self._set_detector_mode(task.detector_mode)
            self._set_reduction_mode(task.reduction_mode)
            self.pil_files_edit.setPlainText("\n".join(task.pil300k_files))
            self.eig_files_edit.setPlainText("\n".join(task.eig1m_files))
            self.energy_spin.setValue(task.num_energies)
            self.group_spin.setValue(task.num_groups)
            self.frame_spin.setValue(task.num_frames)
            self.pil_poni_edit.setText(task.pil300k_poni)
            self.pil_mask_edit.setText(task.pil300k_mask)
            self.eig_poni_edit.setText(task.eig1m_poni)
            self.eig_mask_edit.setText(task.eig1m_mask)
            self.capillary_spin.setValue(task.capillary_thickness)
            self.gc_thickness_spin.setValue(task.gc_thickness)
            self.gc_group_spin.setValue(task.gc_group or 0)
            self.air_group_spin.setValue(task.air_group or 0)
            self.empty_group_spin.setValue(task.empty_group or 0)
            self.core_spin.setValue(task.cores)
            self.pair_table.setRowCount(0)
            for pair in task.asaxs_pairs:
                self._append_pair_row(pair.output_name, str(pair.sample_group), str(pair.solvent_group))
            self.pil_count_label.setText(f"Pil300K: {task.pil300k_count}")
            self.eig_count_label.setText(f"Eig1M: {task.eig1m_count}")
        finally:
            self._loading_builder_settings = False
        self.save_builder_settings()

    def duplicate_selected_task(self) -> None:
        rows = self.selected_indices()
        if not rows:
            return
        if self._queue_mutation_blocked("duplicate tasks"):
            return
        if self._queue_is_running():
            self._queue_command_notice("Cannot duplicate tasks while the queue is running.")
            return
        copied = [task_from_json(task_to_json(self.tasks[index])) for index in rows]
        insert_at = rows[-1] + 1
        new_rows: list[int] = []
        existing_names = {task.task_name for task in self.tasks}
        for offset, task in enumerate(copied):
            task.task_name = self._unique_task_name(f"{task.task_name}_copy", existing_names)
            task.status = "Ready"
            task.message = "Duplicated from selected task"
            existing_names.add(task.task_name)
            self.tasks.insert(insert_at + offset, task)
            new_rows.append(insert_at + offset)
        self.refresh_queue(select_rows=new_rows)
        self.log(f"Duplicated {len(new_rows)} task(s).")

    def copy_selected_tasks(self) -> None:
        rows = self.selected_indices()
        if not rows:
            self._queue_command_notice("No queue task is selected.")
            return
        payload = {
            "format": "ASWAXS v5 queue tasks",
            "version": 1,
            "tasks": [task_to_json(self.tasks[index]) for index in rows],
        }
        QtWidgets.QApplication.clipboard().setText(json.dumps(payload, indent=2))
        self._queue_command_notice(f"Copied {len(rows)} task(s) to clipboard.")

    def paste_tasks(self) -> None:
        if self._queue_mutation_blocked("paste tasks"):
            return
        if self._queue_is_running():
            self._queue_command_notice("Cannot paste tasks while the queue is running.")
            return
        text = QtWidgets.QApplication.clipboard().text().strip()
        if not text:
            self._queue_command_notice("Clipboard is empty.")
            return
        try:
            payload = json.loads(text)
            raw_tasks = payload.get("tasks") if isinstance(payload, dict) else payload
            if not isinstance(raw_tasks, list):
                raise ValueError("clipboard does not contain ASWAXS task JSON")
            tasks = [task_from_json(item) for item in raw_tasks if isinstance(item, dict)]
        except Exception as exc:  # noqa: BLE001 - show GUI-friendly status.
            self._queue_command_notice(f"Clipboard does not contain valid ASWAXS queue task data: {exc}")
            return
        if not tasks:
            self._queue_command_notice("Clipboard has no ASWAXS tasks to paste.")
            return
        selected = self.selected_indices()
        insert_at = (selected[-1] + 1) if selected else len(self.tasks)
        existing_names = {task.task_name for task in self.tasks}
        pasted_rows: list[int] = []
        for offset, task in enumerate(tasks):
            task.task_name = self._unique_task_name(task.task_name, existing_names)
            task.status = "Ready"
            task.message = "Pasted from clipboard"
            existing_names.add(task.task_name)
            self.tasks.insert(insert_at + offset, task)
            pasted_rows.append(insert_at + offset)
        self.refresh_queue(select_rows=pasted_rows)
        self.log(f"Pasted {len(pasted_rows)} task(s).")

    def set_selected_task_status(self) -> None:
        rows = self.selected_indices()
        if not rows:
            return
        if self._queue_mutation_blocked("change task status"):
            return
        if self.runner is not None and self.runner.isRunning() and any(self.tasks[index].status == "Running" for index in rows):
            self._queue_command_notice("Cannot change the status of a currently running task.")
            return
        statuses = ["Ready", "Needs Attention", "Done", "Failed", "Stopped", "Skipped"]
        current = self.tasks[rows[0]].status if self.tasks[rows[0]].status in statuses else "Ready"
        label = self.tasks[rows[0]].task_name if len(rows) == 1 else f"{len(rows)} selected tasks"
        status, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Set Task Status",
            f"Set status for:\n{label}",
            statuses,
            statuses.index(current),
            False,
        )
        if not ok or not status:
            return
        for index in rows:
            self.tasks[index].status = status
            self.tasks[index].message = "Status set manually" if status == "Ready" else f"Status manually set to {status}"
            self._update_queue_row(index)
        self.statusBar().showMessage(f"{len(rows)} task(s): status set to {status}")
        self.log(f"{len(rows)} task(s): status manually set to {status}")

    def remove_selected_task(self) -> None:
        rows = self.selected_indices()
        if not rows:
            return
        if self._queue_mutation_blocked("delete tasks"):
            return
        if self._queue_is_running():
            self._queue_command_notice("Cannot delete tasks while the queue is running.")
            return
        for index in rows:
            if self.tasks[index].status == "Running":
                self._queue_command_notice("Cannot delete the currently running task.")
                return
        first = rows[0]
        for index in reversed(rows):
            self.tasks.pop(index)
        if self.editing_index in rows:
            self.editing_index = None
        elif self.editing_index is not None:
            self.editing_index -= sum(1 for row in rows if row < self.editing_index)
        self.refresh_queue(select_row=first)
        self.log(f"Deleted {len(rows)} task(s).")

    def move_selected_task(self, direction: int) -> None:
        if self._queue_mutation_blocked("move tasks"):
            return
        index = self.selected_index()
        if index is None:
            return
        self.move_task_row(index, index + direction)

    def move_task_row(self, source: int, target: int) -> None:
        if self._queue_mutation_blocked("move tasks"):
            return
        if not (0 <= source < len(self.tasks)) or not (0 <= target < len(self.tasks)) or source == target:
            return
        task = self.tasks.pop(source)
        self.tasks.insert(target, task)
        if self.editing_index == source:
            self.editing_index = target
        elif self.editing_index is not None:
            low, high = sorted((source, target))
            if low <= self.editing_index <= high:
                self.editing_index += -1 if source < target else 1
        self.refresh_queue(select_row=target)
        self.log(f"Moved task '{task.task_name}' to row {target + 1}.")

    def run_selected(self) -> None:
        rows = self.selected_indices()
        if not rows:
            self._queue_command_notice("No task is selected.")
            return
        self._start_runner(rows, skip_invalid=False, run_any_status=True)

    def run_all(self) -> None:
        if not self.tasks:
            self._queue_command_notice("Queue is empty.")
            return
        self._start_runner(list(range(len(self.tasks))), skip_invalid=True, run_any_status=False)

    def _start_runner(self, indices: list[int], *, skip_invalid: bool, run_any_status: bool) -> None:
        if self.runner is not None and self.runner.isRunning():
            self._queue_command_notice("Queue is already running.")
            return
        if not indices:
            self._queue_command_notice("No runnable tasks were requested.")
            return
        runnable = [index for index in indices if 0 <= index < len(self.tasks)]
        if not runnable:
            self._queue_command_notice("No runnable tasks were found.")
            return
        queue_candidates: list[int] = []
        failed_messages: list[str] = []
        for index in runnable:
            if not run_any_status and self.tasks[index].status not in {"Ready", "Queued"}:
                continue
            ok, message = preflight_task(self.tasks[index])
            if ok:
                queue_candidates.append(index)
            elif run_any_status:
                self._show_validation_failed("Cannot start invalid task", f"{self.tasks[index].task_name}: {message}")
                self._queue_command_notice("Task validation failed. Fix the task settings before running.")
                return
            else:
                self.tasks[index].status = "Needs Attention"
                self.tasks[index].message = message
                failed_messages.append(f"{self.tasks[index].task_name}: {message}")
        if not queue_candidates:
            if failed_messages:
                self.refresh_queue(select_row=runnable[0])
                self.log("Skipped invalid queue task(s): " + " | ".join(failed_messages))
            self._queue_command_notice("No Ready tasks to run. Use Run Task to rerun a selected completed task.")
            return
        if failed_messages:
            self.log("Skipped invalid queue task(s): " + " | ".join(failed_messages))
        for index in queue_candidates:
            self.tasks[index].status = "Queued"
            self.tasks[index].message = "Waiting to restart"
        self.active_run_indices = queue_candidates
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        self.overall_progress.setFormat("Waiting for current task")
        self._reset_detector_progress_labels("Queued")
        self.refresh_queue(select_row=queue_candidates[0])
        self.set_queue_locked(True, "Queue locked while running. Stop Queue is still available.")
        self.runner = TaskRunner(self.tasks, runnable, queue_candidates, run_any_status=run_any_status)
        self.runner.message.connect(self.log)
        self.runner.task_progress.connect(self._task_progress)
        self.runner.task_started.connect(self._task_started)
        self.runner.task_finished.connect(self._task_finished)
        self.runner.task_skipped.connect(self._task_skipped)
        self.runner.all_done.connect(self._all_done)
        self.runner.start()
        self.tabs.setCurrentIndex(0)
        self._queue_command_notice(f"Started {len(queue_candidates)} queued task(s).")

    def stop_current_queue(self) -> None:
        if self.runner is None or not self.runner.isRunning():
            self._queue_command_notice("No queue is currently running.")
            return
        self.runner.request_stop_current()
        self.current_stage_label.setText("Stopping current task...")
        self._reset_detector_progress_labels("Stopping")
        self._queue_command_notice("Stop requested for current queue task.")

    def _queue_is_running(self) -> bool:
        return self.runner is not None and self.runner.isRunning()

    def _unique_task_name(self, base: str, existing_names: set[str]) -> str:
        clean_base = str(base).strip() or "Task"
        if clean_base not in existing_names:
            return clean_base
        suffix = 2
        while f"{clean_base}_{suffix}" in existing_names:
            suffix += 1
        return f"{clean_base}_{suffix}"

    def _queue_command_notice(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self.log(message)

    def _show_validation_failed(self, title: str, message: str) -> None:
        details = "\n".join(f"- {part.strip()}" for part in message.split(";") if part.strip())
        QtWidgets.QMessageBox.warning(self, "Task validation failed", f"{title}\n\n{details or message}")

    def _task_started(self, index: int) -> None:
        self.tasks[index].status = "Running"
        self.tasks[index].message = "Restarting from scratch"
        self.current_stage_label.setText(f"Running {self.tasks[index].task_name}")
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        self._reset_detector_progress_labels("0/" + str(self.tasks[index].expected_files_per_detector) + " frames", self.tasks[index])
        self._set_task_progress_value(0.01, "Starting")
        self.refresh_queue(select_row=index)
        self.refresh_current_curves()

    def _task_progress(self, index: int, fraction: float, label: str) -> None:
        if not (0 <= index < len(self.tasks)):
            return
        main_label = self._update_detector_progress_labels(label)
        self.tasks[index].message = main_label
        self.current_stage_label.setText(f"{self.tasks[index].task_name}: {main_label}")
        self._set_task_progress_value(fraction, main_label)
        self._update_queue_row(index)

    def _task_finished(self, index: int, ok: bool, message: str) -> None:
        self.tasks[index].status = "Done" if ok else ("Stopped" if message == "Stopped by user" else "Failed")
        self.tasks[index].message = message
        if ok:
            self._set_detector_progress_complete(self.tasks[index])
            self._set_task_progress_value(1.0, "Complete")
        elif message != "Stopped by user":
            self._show_task_failed(index, message)
        self.refresh_queue(select_row=self._next_active_or_finished_row(index))
        self.refresh_current_curves()

    def _show_task_failed(self, index: int, message: str) -> None:
        task_name = self.tasks[index].task_name if 0 <= index < len(self.tasks) else "Task"
        self.current_stage_label.setText(f"{task_name}: Failed")
        self.statusBar().showMessage(f"{task_name} failed. See message dialog and task tooltip.")
        details = message.strip() or "Unknown error"
        if len(details) > 6000:
            details = details[-6000:]
            details = "... earlier traceback omitted ...\n" + details
        QtWidgets.QMessageBox.critical(self, "Task failed", f"{task_name}\n\n{details}")

    def _task_skipped(self, index: int, message: str) -> None:
        if not (0 <= index < len(self.tasks)):
            return
        if self.tasks[index].status in {"Ready", "Queued"}:
            self.tasks[index].status = "Needs Attention" if message.startswith("Validation failed") else "Skipped"
        self.tasks[index].message = message
        self._update_queue_row(index)

    def _next_active_or_finished_row(self, fallback: int) -> int:
        for index in self.active_run_indices:
            if 0 <= index < len(self.tasks) and self.tasks[index].status in {"Running", "Queued"}:
                return index
        return max(0, min(fallback, len(self.tasks) - 1)) if self.tasks else 0

    def _set_task_progress_value(self, fraction: float, label: str) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        value = int(fraction * 100)
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(value)
        self.overall_progress.setFormat(f"%p% - {label}")

    def _all_done(self) -> None:
        self.set_queue_locked(False)
        self.overall_progress.setRange(0, 100)
        if any(0 <= index < len(self.tasks) and self.tasks[index].status == "Done" for index in self.active_run_indices):
            self._set_task_progress_value(1.0, "Queue complete")
        else:
            self._reset_task_progress()
        self.current_stage_label.setText("Idle")
        self.save_queue()
        self.refresh_queue(select_row=self._next_active_or_finished_row(self.selected_index() or 0))
        self.refresh_current_curves()

    def _reset_task_progress(self) -> None:
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        self.overall_progress.setFormat("%p%")
        self._reset_detector_progress_labels("Idle")

    def _reset_detector_progress_labels(self, text: str = "idle", task: TaskSpec | None = None) -> None:
        active = set(task.active_detectors()) if task is not None else {"Pil300K", "Eig1M"}
        self.pil_detector_progress_label.setText(f"Pil300K: {text if 'Pil300K' in active else 'not used'}")
        self.eig_detector_progress_label.setText(f"Eig1M: {text if 'Eig1M' in active else 'not used'}")

    def _set_detector_progress_complete(self, task: TaskSpec) -> None:
        expected = task.expected_files_per_detector
        active = set(task.active_detectors())
        self.pil_detector_progress_label.setText(
            f"Pil300K: {expected}/{expected} frames complete" if "Pil300K" in active else "Pil300K: not used"
        )
        self.eig_detector_progress_label.setText(
            f"Eig1M: {expected}/{expected} frames complete" if "Eig1M" in active else "Eig1M: not used"
        )

    def _update_detector_progress_labels(self, label: str) -> str:
        for detector, done_text, total_text in DETECTOR_PROGRESS_RE.findall(label):
            done = int(done_text)
            total = int(total_text)
            percent = int(done / max(1, total) * 100)
            state = "complete" if done >= total else "running"
            text = f"{detector}: {done}/{total} frames ({percent}%) {state}"
            if detector == "Pil300K":
                self.pil_detector_progress_label.setText(text)
            elif detector == "Eig1M":
                self.eig_detector_progress_label.setText(text)
        return re.sub(r"\s*\((?=[^)]*(?:Pil300K|Eig1M))[^)]*\)\s*$", "", label).strip()

    def open_selected_output(self) -> None:
        index = self.selected_index()
        if index is None:
            return
        path = self.tasks[index].output_path
        path.mkdir(parents=True, exist_ok=True)
        QtCore.QUrl.fromLocalFile(str(path))
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]

    def open_h5_iq_viewer(self) -> None:
        if self.h5_iq_viewer is None:
            self.h5_iq_viewer = H5IqViewerDialog(self)
        index = self.selected_index()
        if index is not None:
            self.h5_iq_viewer.open_file(self.tasks[index].output_path)
        else:
            self.h5_iq_viewer.show()
            self.h5_iq_viewer.raise_()
            self.h5_iq_viewer.activateWindow()

    def open_h5_structure_viewer(self) -> None:
        if self.h5_structure_viewer is None:
            self.h5_structure_viewer = H5StructureViewerDialog(self)
        path = self._selected_task_analysis_h5()
        if path is not None:
            self.h5_structure_viewer.open_file(path)
        else:
            self.h5_structure_viewer.show()
            self.h5_structure_viewer.raise_()
            self.h5_structure_viewer.activateWindow()

    def _selected_task_analysis_h5(self) -> Path | None:
        index = self.selected_index()
        if index is None:
            return None
        return self._find_task_analysis_h5(self.tasks[index])

    def about(self) -> None:
        QtWidgets.QMessageBox.information(self, "About ASWAXS v5", "ASWAXS v5 task-queue dashboard.")

    def log(self, message: str) -> None:
        self.log_messages.append(message)
        self.log_messages = self.log_messages[-1000:]
        self.statusBar().showMessage(message[:180])

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt override name.
        self.save_builder_settings()
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = DashboardWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
