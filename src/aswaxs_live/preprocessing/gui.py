"""Calibration GUI for HDF5 inspection, EDF export, PONI setup, and masks."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5 import QtWidgets

from .io_utils import DEFAULT_DATASET_PATH, load_hdf5_image
from .processing import (
    export_image_as_edf,
    export_mask_as_edf,
    launch_pyfai_calib2,
    launch_pyfai_drawmask,
    load_mask,
    load_mask_from_edf,
    load_poni_summary,
    save_mask,
)


class PreprocessingWindow(QtWidgets.QMainWindow):
    """Interactive pyFAI setup window; scientific reduction happens elsewhere."""
    def __init__(self, initial_path: str | None = None):
        super().__init__()
        self.setWindowTitle("ASWAXS pyFAI Setup")
        self.resize(1680, 1020)

        self.current_file: Path | None = None
        self.current_image: np.ndarray | None = None
        self.current_dataset_path = DEFAULT_DATASET_PATH
        self.current_metadata: dict[str, object] = {}
        self.current_mask: np.ndarray | None = None
        self.current_mask_path: Path | None = None
        self.current_poni_path: Path | None = None
        self.current_edf_path: Path | None = None
        self.current_mask_edf_path: Path | None = None
        self._calib2_process = None
        self._drawmask_process = None

        self._build_ui()
        if initial_path:
            self.load_hdf5_file(initial_path)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter()
        root.addWidget(splitter)

        left = QtWidgets.QScrollArea()
        left.setWidgetResizable(True)
        left_content = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_content)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(10)
        left.setWidget(left_content)
        left.setMinimumWidth(390)
        left.setMaximumWidth(560)
        splitter.addWidget(left)

        figure_widget = QtWidgets.QWidget()
        figure_layout = QtWidgets.QVBoxLayout(figure_widget)
        figure_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(12, 9))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.image_ax = self.figure.add_subplot(111)
        figure_layout.addWidget(self.toolbar)
        figure_layout.addWidget(self.canvas)
        splitter.addWidget(figure_widget)
        splitter.setSizes([450, 1200])

        self.tabs = QtWidgets.QTabWidget()
        left_layout.addWidget(self.tabs)
        self.calibration_tab = QtWidgets.QWidget()
        self.mask_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.calibration_tab, "Calibration")
        self.tabs.addTab(self.mask_tab, "Mask")

        self.calibration_layout = QtWidgets.QVBoxLayout(self.calibration_tab)
        self.calibration_layout.setContentsMargins(0, 0, 0, 0)
        self.calibration_layout.setSpacing(10)
        self.mask_layout = QtWidgets.QVBoxLayout(self.mask_tab)
        self.mask_layout.setContentsMargins(0, 0, 0, 0)
        self.mask_layout.setSpacing(10)

        self._build_load_group()
        self._build_calibration_group()
        self._build_mask_group()

        self.status_label = QtWidgets.QLabel("Load an HDF5 file to begin.")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)

    def _build_load_group(self) -> None:
        box = QtWidgets.QGroupBox("1. Load HDF5")
        form = QtWidgets.QFormLayout(box)

        self.file_path_edit = QtWidgets.QLineEdit()
        self.dataset_path_edit = QtWidgets.QLineEdit(DEFAULT_DATASET_PATH)

        browse_button = QtWidgets.QPushButton("Browse HDF5")
        browse_button.clicked.connect(self._browse_hdf5)
        load_button = QtWidgets.QPushButton("Load File")
        load_button.clicked.connect(lambda: self.load_hdf5_file(self.file_path_edit.text().strip()))

        self.file_info_label = QtWidgets.QLabel("No file loaded.")
        self.file_info_label.setWordWrap(True)
        self.detector_info_label = QtWidgets.QLabel("Detector: -")
        self.detector_info_label.setWordWrap(True)
        self.metadata_label = QtWidgets.QLabel("Metadata: -")
        self.metadata_label.setWordWrap(True)
        self.detector_position_label = QtWidgets.QLabel("Detector position: -")
        self.detector_position_label.setWordWrap(True)

        form.addRow("HDF5 path", self.file_path_edit)
        form.addRow("Dataset path", self.dataset_path_edit)
        form.addRow("", browse_button)
        form.addRow("", load_button)
        form.addRow("Loaded file", self.file_info_label)
        form.addRow("Detector", self.detector_info_label)
        form.addRow("Metadata", self.metadata_label)
        form.addRow("Position", self.detector_position_label)
        self.calibration_layout.addWidget(box)

    def _build_calibration_group(self) -> None:
        box = QtWidgets.QGroupBox("2. Calibration In pyFAI")
        form = QtWidgets.QFormLayout(box)

        self.calibrant_edit = QtWidgets.QLineEdit("AgBh")
        self.calibration_note_label = QtWidgets.QLabel(
            "Workflow: load HDF5, export EDF bridge, open pyFAI-calib2, save .poni, then load the .poni back here."
        )
        self.calibration_note_label.setWordWrap(True)
        self.edf_path_label = QtWidgets.QLabel("EDF bridge: -")
        self.edf_path_label.setWordWrap(True)
        self.calib2_prefill_label = QtWidgets.QLabel("pyFAI prefill: detector=-, energy=-, distance=-")
        self.calib2_prefill_label.setWordWrap(True)
        self.poni_path_label = QtWidgets.QLabel("PONI: -")
        self.poni_path_label.setWordWrap(True)
        self.poni_summary_label = QtWidgets.QLabel("Calibration summary: -")
        self.poni_summary_label.setWordWrap(True)

        export_button = QtWidgets.QPushButton("Export EDF Bridge")
        export_button.clicked.connect(self._export_current_edf)
        calib2_button = QtWidgets.QPushButton("Launch pyFAI-calib2")
        calib2_button.clicked.connect(self._launch_calib2)
        load_poni_button = QtWidgets.QPushButton("Load PONI")
        load_poni_button.clicked.connect(self._load_poni_dialog)

        form.addRow("Calibrant", self.calibrant_edit)
        form.addRow("", self.calibration_note_label)
        form.addRow("", export_button)
        form.addRow("", calib2_button)
        form.addRow("", load_poni_button)
        form.addRow("EDF bridge", self.edf_path_label)
        form.addRow("pyFAI prefill", self.calib2_prefill_label)
        form.addRow("Loaded PONI", self.poni_path_label)
        form.addRow("PONI summary", self.poni_summary_label)
        self.calibration_layout.addWidget(box)

    def _build_mask_group(self) -> None:
        box = QtWidgets.QGroupBox("3. Mask In pyFAI")
        form = QtWidgets.QFormLayout(box)

        self.mask_note_label = QtWidgets.QLabel(
            "Workflow: export EDF bridge for mask authoring, open pyFAI-drawmask, save the mask, then import it here."
        )
        self.mask_note_label.setWordWrap(True)
        self.mask_path_label = QtWidgets.QLabel("Mask: -")
        self.mask_path_label.setWordWrap(True)
        self.mask_stats_label = QtWidgets.QLabel("Mask pixels: -")
        self.mask_stats_label.setWordWrap(True)

        export_mask_edf_button = QtWidgets.QPushButton("Export Mask EDF Bridge")
        export_mask_edf_button.clicked.connect(self._export_mask_edf_bridge)
        drawmask_button = QtWidgets.QPushButton("Launch pyFAI-drawmask")
        drawmask_button.clicked.connect(self._launch_drawmask)
        import_mask_button = QtWidgets.QPushButton("Import pyFAI Mask EDF")
        import_mask_button.clicked.connect(self._import_mask_edf_dialog)
        save_mask_button = QtWidgets.QPushButton("Save Mask As .npy")
        save_mask_button.clicked.connect(self._save_mask_dialog)
        load_mask_button = QtWidgets.QPushButton("Load Mask .npy")
        load_mask_button.clicked.connect(self._load_mask_dialog)

        form.addRow("", self.mask_note_label)
        form.addRow("", export_mask_edf_button)
        form.addRow("", drawmask_button)
        form.addRow("", import_mask_button)
        form.addRow("", save_mask_button)
        form.addRow("", load_mask_button)
        form.addRow("Loaded mask", self.mask_path_label)
        form.addRow("Mask summary", self.mask_stats_label)
        self.mask_layout.addWidget(box)

    def _browse_hdf5(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open HDF5", "", "HDF5 files (*.h5 *.hdf5)")
        if path:
            self.file_path_edit.setText(path)

    def load_hdf5_file(self, path: str) -> None:
        if not path:
            self.status_label.setText("Choose an HDF5 file first.")
            return
        dataset_path = self.dataset_path_edit.text().strip() or DEFAULT_DATASET_PATH
        try:
            loaded = load_hdf5_image(path, dataset_path=dataset_path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load HDF5: {exc}")
            return

        self.current_file = loaded.path
        self.current_image = loaded.image
        self.current_dataset_path = loaded.dataset_path
        self.current_metadata = loaded.metadata
        self.current_mask = None
        self.current_mask_path = None
        self.current_poni_path = None
        self.current_edf_path = None
        self.current_mask_edf_path = None

        self.file_path_edit.setText(str(loaded.path))
        self.dataset_path_edit.setText(loaded.dataset_path)
        self.file_info_label.setText(f"{loaded.path.name} | shape={loaded.image.shape} | dtype={loaded.image.dtype}")
        detector = loaded.metadata.get("detector_name") or "unknown"
        self.detector_info_label.setText(str(detector))
        distance_source = loaded.metadata.get("distance_source")
        self.metadata_label.setText(
            f"energy_keV={loaded.metadata.get('mono_energy_keV')}, "
            f"SAXS_distance_m={loaded.metadata.get('distance_m')} ({distance_source}), "
            f"SPDS_point_detector={loaded.metadata.get('saxs_point_detector')}"
        )
        self.detector_position_label.setText(self._format_detector_positions(loaded.metadata))
        self.edf_path_label.setText("EDF bridge: -")
        self.poni_path_label.setText("PONI: -")
        self.poni_summary_label.setText("Calibration summary: -")
        self.mask_path_label.setText("Mask: -")
        self.mask_stats_label.setText("Mask pixels: -")
        detector_text = loaded.metadata.get("detector_name") or "-"
        energy_text = loaded.metadata.get("mono_energy_keV")
        distance_text = loaded.metadata.get("distance_mm")
        self.calib2_prefill_label.setText(
            f"detector={detector_text}, energy_keV={energy_text if energy_text is not None else '-'}, "
            f"distance_mm={distance_text if distance_text is not None else '-'}"
        )
        self._update_image_view()
        self.status_label.setText("HDF5 loaded. Next step: export EDF bridge and open pyFAI-calib2.")

    def _format_detector_positions(self, metadata: dict[str, object]) -> str:
        saxs = metadata.get("saxs_detector_position_mm") or metadata.get("saxs_detector_position") or {}
        waxs = metadata.get("waxs_detector_position_with_units") or metadata.get("waxs_detector_position") or {}

        def format_position(values: object) -> str:
            if not isinstance(values, dict) or not values:
                return "-"
            return ", ".join(f"{key}={value}" for key, value in values.items())

        return (
            "SAXS stage: "
            + format_position(saxs)
            + "\nWAXS detector: "
            + format_position(waxs)
            + "\nNote: linear motor readings are in mm. SAXS area detector distance is fixed near 5.6 m; "
            + "SPDS is the SAXS point detector; WD_Z is WAXS distance; WD_RY is WAXS rotation about y."
        )

    def _default_edf_path(self) -> Path:
        assert self.current_file is not None
        return self.current_file.with_suffix(".pyfai_setup.edf")

    def _default_mask_edf_path(self) -> Path:
        assert self.current_file is not None
        return self.current_file.with_suffix(".pyfai_mask_input.edf")

    def _export_current_edf(self) -> None:
        if self.current_image is None or self.current_file is None:
            self.status_label.setText("Load an HDF5 file before exporting EDF.")
            return
        path = self._default_edf_path()
        export_image_as_edf(path, self.current_image)
        self.current_edf_path = path
        self.edf_path_label.setText(str(path))
        self.status_label.setText(f"Exported EDF bridge to {path.name}.")

    def _launch_calib2(self) -> None:
        if self.current_image is None or self.current_file is None:
            self.status_label.setText("Load an HDF5 file before launching pyFAI-calib2.")
            return
        if self.current_edf_path is None or not self.current_edf_path.exists():
            self._export_current_edf()
        mask_path = None
        if self.current_mask is not None and self.current_file is not None:
            mask_path = self.current_file.with_suffix(".pyfai_mask.edf")
            export_mask_as_edf(mask_path, self.current_mask)
        try:
            self._calib2_process = launch_pyfai_calib2(
                edf_path=self.current_edf_path,
                calibrant=self.calibrant_edit.text().strip() or "AgBh",
                detector_name=self.current_metadata.get("detector_name"),
                energy_kev=self._metadata_float("mono_energy_keV"),
                distance_m=self._metadata_distance_m(),
                mask_path=mask_path,
            )
        except Exception as exc:
            self.status_label.setText(f"Failed to launch pyFAI-calib2: {exc}")
            return
        self.status_label.setText(
            f"pyFAI-calib2 launched on {self.current_edf_path.name} with detector/energy/distance prefilled when available. "
            f"Save the .poni from pyFAI, then load it here."
        )

    def _load_poni_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load PONI", "", "PONI file (*.poni)")
        if not path:
            return
        self.current_poni_path = Path(path).expanduser().resolve()
        try:
            summary = load_poni_summary(self.current_poni_path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load PONI: {exc}")
            return
        self.poni_path_label.setText(str(self.current_poni_path))
        self.poni_summary_label.setText(summary)
        self.status_label.setText("PONI loaded. You can now continue with mask authoring.")

    def _export_mask_edf_bridge(self) -> None:
        if self.current_image is None or self.current_file is None:
            self.status_label.setText("Load an HDF5 file before exporting a mask EDF bridge.")
            return
        path = self._default_mask_edf_path()
        export_image_as_edf(path, self.current_image)
        self.current_mask_edf_path = path
        self.status_label.setText(f"Exported mask EDF bridge to {path.name}.")

    def _launch_drawmask(self) -> None:
        if self.current_image is None or self.current_file is None:
            self.status_label.setText("Load an HDF5 file before launching pyFAI-drawmask.")
            return
        if self.current_mask_edf_path is None or not self.current_mask_edf_path.exists():
            self._export_mask_edf_bridge()
        try:
            self._drawmask_process = launch_pyfai_drawmask(self.current_mask_edf_path)
        except Exception as exc:
            self.status_label.setText(f"Failed to launch pyFAI-drawmask: {exc}")
            return
        self.status_label.setText(
            f"pyFAI-drawmask launched on {self.current_mask_edf_path.name}. Save the mask EDF in pyFAI, then import it here."
        )

    def _import_mask_edf_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import pyFAI Mask EDF", "", "EDF files (*.edf)")
        if not path:
            return
        try:
            mask = load_mask_from_edf(path)
        except Exception as exc:
            self.status_label.setText(f"Failed to import mask EDF: {exc}")
            return
        self.current_mask = mask
        self.current_mask_path = Path(path).expanduser().resolve()
        self.mask_path_label.setText(str(self.current_mask_path))
        self.mask_stats_label.setText(f"Mask pixels: {int(mask.sum())}")
        self._update_image_view()
        self.status_label.setText("Imported mask EDF.")

    def _save_mask_dialog(self) -> None:
        if self.current_mask is None:
            self.status_label.setText("Import or load a mask before saving it.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Mask", "", "NumPy mask (*.npy)")
        if not path:
            return
        save_mask(path, self.current_mask)
        self.current_mask_path = Path(path).expanduser().resolve()
        self.mask_path_label.setText(str(self.current_mask_path))
        self.status_label.setText(f"Saved mask to {path}.")

    def _load_mask_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Mask", "", "NumPy mask (*.npy)")
        if not path:
            return
        try:
            mask = load_mask(path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load mask: {exc}")
            return
        self.current_mask = mask
        self.current_mask_path = Path(path).expanduser().resolve()
        self.mask_path_label.setText(str(self.current_mask_path))
        self.mask_stats_label.setText(f"Mask pixels: {int(mask.sum())}")
        self._update_image_view()
        self.status_label.setText("Mask loaded.")

    def _update_image_view(self) -> None:
        self.image_ax.clear()
        if self.current_image is None:
            self.canvas.draw_idle()
            return
        self.image_ax.imshow(self.current_image, cmap="magma", origin="lower")
        if self.current_mask is not None:
            overlay = np.ma.masked_where(~self.current_mask.astype(bool), self.current_mask)
            self.image_ax.imshow(overlay, cmap="cool", origin="lower", alpha=0.35)
        self.image_ax.set_title("Detector View")
        self.image_ax.set_xlabel("column")
        self.image_ax.set_ylabel("row")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _metadata_float(self, key: str) -> float | None:
        value = self.current_metadata.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _metadata_distance_m(self) -> float | None:
        distance_m = self._metadata_float("distance_m")
        if distance_m is not None:
            return distance_m
        distance_mm = self._metadata_float("distance_mm")
        if distance_mm is None:
            return None
        return distance_mm * 1e-3


def run_app(initial_path: str | None = None) -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = PreprocessingWindow(initial_path=initial_path)
    window.show()
    return app.exec_()
