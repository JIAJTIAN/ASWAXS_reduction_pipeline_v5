"""Small read-only HDF5 inspection tools for the v5 dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets


H5_FILTER = "HDF5 files (*.h5 *.hdf5);;All files (*)"
MAX_CURVE_ROWS_PER_GROUP = 250
MAX_TREE_ITEMS = 6000
MAX_ERROR_BAR_POINTS = 450
MIN_Q_GRID_POINTS = 8
Q_DATASET_NAMES = {"q", "Q", "q_A^-1", "q_invA", "q_nm^-1"}
METADATA_GROUP_NAMES = {
    "metadata",
    "parameters",
    "normalization_factors",
    "subtraction_map",
    "detector_summary",
    "monitor_summary",
    "sample_summary",
    "scan_summary",
    "data_reference_summary",
}


@dataclass(frozen=True)
class H5CurveRecord:
    label: str
    h5_path: str | None
    group_path: str
    q_path: str
    i_path: str
    y_name: str
    sigma_path: str | None
    row: int | None


class H5IqViewerDialog(QtWidgets.QDialog):
    """Read-only HDF5 q-I curve viewer."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("HDF5 I-q Plot Viewer")
        self.resize(1180, 760)
        self.curves: list[H5CurveRecord] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self.browse_file)
        browse_folder = QtWidgets.QPushButton("Browse Folder")
        browse_folder.clicked.connect(self.browse_folder)
        load = QtWidgets.QPushButton("Load")
        load.clicked.connect(self.load_file)
        path_row.addWidget(QtWidgets.QLabel("HDF5 file/folder"))
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        path_row.addWidget(browse_folder)
        path_row.addWidget(load)
        root.addLayout(path_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        controls = QtWidgets.QHBoxLayout()
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter curves, e.g. final, stitched, normalized, group_05")
        self.filter_edit.textChanged.connect(self._refill_curve_list)
        self.log_q_check = QtWidgets.QCheckBox("log q")
        self.log_q_check.setChecked(True)
        self.log_i_check = QtWidgets.QCheckBox("log I")
        self.log_i_check.setChecked(True)
        self.error_check = QtWidgets.QCheckBox("errors")
        self.error_check.setChecked(True)
        self.max_curves_spin = QtWidgets.QSpinBox()
        self.max_curves_spin.setRange(1, 200)
        self.max_curves_spin.setValue(20)
        plot_button = QtWidgets.QPushButton("Plot Selected")
        plot_button.clicked.connect(self.plot_selected)
        controls.addWidget(QtWidgets.QLabel("Filter"))
        controls.addWidget(self.filter_edit, 1)
        controls.addWidget(self.log_q_check)
        controls.addWidget(self.log_i_check)
        controls.addWidget(self.error_check)
        controls.addWidget(QtWidgets.QLabel("Max"))
        controls.addWidget(self.max_curves_spin)
        controls.addWidget(plot_button)
        left_layout.addLayout(controls)

        self.curve_list = QtWidgets.QListWidget()
        self.curve_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.curve_list.itemDoubleClicked.connect(lambda _item: self.plot_selected())
        left_layout.addWidget(self.curve_list, 1)
        self.status_label = QtWidgets.QLabel("Choose an analysis HDF5 file.")
        left_layout.addWidget(self.status_label)
        splitter.addWidget(left)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "q")
        self.plot.setLabel("left", "I")
        self.legend = self.plot.addLegend(offset=(8, 8))
        self.plot.getPlotItem().setDownsampling(auto=True, mode="peak")
        self.plot.getPlotItem().setClipToView(True)
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

    def open_file(self, path: Path | str) -> None:
        self.path_edit.setText(str(path))
        self.load_file()
        self.show()
        self.raise_()
        self.activateWindow()

    def browse_file(self) -> None:
        start = Path(self.path_edit.text()).parent if self.path_edit.text().strip() else Path.home()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open HDF5 for I-q plotting", str(start), H5_FILTER)
        if path:
            self.path_edit.setText(path)
            self.load_file()

    def browse_folder(self) -> None:
        start = Path(self.path_edit.text()) if self.path_edit.text().strip() else Path.home()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open result folder for I-q plotting", str(start))
        if folder:
            self.path_edit.setText(folder)
            self.load_file()

    def load_file(self) -> None:
        path = Path(self.path_edit.text().strip())
        self.curve_list.clear()
        self.curves = []
        if not path.exists():
            self.status_label.setText("HDF5 file/folder does not exist.")
            return
        try:
            h5_paths = _h5_paths_from_file_or_folder(path)
            for h5_path in h5_paths:
                with h5py.File(h5_path, "r") as handle:
                    source_label = _source_label(path, h5_path) if len(h5_paths) > 1 else ""
                    self.curves.extend(discover_iq_curves(handle, source_path=h5_path, source_label=source_label))
        except Exception as exc:  # noqa: BLE001 - show GUI-friendly error.
            self.status_label.setText(f"Could not read HDF5: {exc}")
            return
        self._refill_curve_list(select_first=True)
        source_text = f" in {len(h5_paths)} HDF5 file(s)" if path.is_dir() else ""
        self.status_label.setText(f"Found {len(self.curves)} plottable q-data rows{source_text}.")
        self.plot_selected()

    def _refill_curve_list(self, select_first: bool = False) -> None:
        needle = self.filter_edit.text().strip().lower()
        self.curve_list.clear()
        for index, curve in enumerate(self.curves):
            if needle and needle not in curve.label.lower() and needle not in curve.group_path.lower():
                continue
            item = QtWidgets.QListWidgetItem(curve.label)
            item.setData(QtCore.Qt.UserRole, index)
            self.curve_list.addItem(item)
        if select_first:
            for row in range(min(self.curve_list.count(), self.max_curves_spin.value())):
                self.curve_list.item(row).setSelected(True)

    def plot_selected(self) -> None:
        self.plot.clear()
        self.legend.clear()
        self.plot.setLogMode(x=self.log_q_check.isChecked(), y=self.log_i_check.isChecked())
        path = Path(self.path_edit.text().strip())
        selected = [self.curve_list.item(index.row()).data(QtCore.Qt.UserRole) for index in self.curve_list.selectedIndexes()]
        selected = selected[: self.max_curves_spin.value()]
        if not selected:
            self.status_label.setText("Select one or more curves to plot.")
            return
        plotted = 0
        try:
            for record_index in selected:
                record = self.curves[record_index]
                h5_path = Path(record.h5_path) if record.h5_path else path
                with h5py.File(h5_path, "r") as handle:
                    q, intensity, sigma = read_curve_row(handle, record)
                    q = np.asarray(q, dtype=float).reshape(-1)
                    intensity = np.asarray(intensity, dtype=float).reshape(-1)
                    n = min(q.size, intensity.size)
                    q = q[:n]
                    intensity = intensity[:n]
                    if sigma is not None:
                        sigma = np.asarray(sigma, dtype=float).reshape(-1)[:n]
                    mask = np.isfinite(q) & np.isfinite(intensity)
                    if self.log_q_check.isChecked():
                        mask &= q > 0
                    if self.log_i_check.isChecked():
                        mask &= intensity > 0
                    if np.count_nonzero(mask) < 2:
                        continue
                    pen = pg.mkPen(pg.intColor(plotted, hues=max(8, len(selected))), width=1.4)
                    self.plot.plot(q[mask], intensity[mask], pen=pen, name=record.label)
                    if self.error_check.isChecked() and sigma is not None:
                        self._plot_error_bars(q, intensity, sigma, mask, pen)
                    plotted += 1
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Could not plot selected curves: {exc}")
            return
        self.status_label.setText(f"Plotted {plotted}/{len(selected)} selected curves.")

    def _plot_error_bars(self, q: np.ndarray, intensity: np.ndarray, sigma: np.ndarray, mask: np.ndarray, pen: pg.mkPen) -> None:
        sigma_mask = mask & np.isfinite(sigma) & (sigma >= 0)
        if np.count_nonzero(sigma_mask) < 2:
            return
        x = q[sigma_mask]
        y = intensity[sigma_mask]
        err = sigma[sigma_mask]
        if x.size > MAX_ERROR_BAR_POINTS:
            indices = np.linspace(0, x.size - 1, MAX_ERROR_BAR_POINTS).astype(int)
            x = x[indices]
            y = y[indices]
            err = err[indices]
        self.plot.addItem(pg.ErrorBarItem(x=x, y=y, top=err, bottom=err, beam=0.0, pen=pen))


class H5StructureViewerDialog(QtWidgets.QDialog):
    """Read-only HDF5 structure and metadata browser."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("HDF5 Structure / Metadata Viewer")
        self.resize(1180, 760)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self.browse_file)
        load = QtWidgets.QPushButton("Load")
        load.clicked.connect(self.load_file)
        path_row.addWidget(QtWidgets.QLabel("HDF5 file"))
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        path_row.addWidget(load)
        root.addLayout(path_row)

        search_row = QtWidgets.QHBoxLayout()
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Search structure/metadata")
        self.search_edit.textChanged.connect(self._apply_tree_filter)
        search_row.addWidget(QtWidgets.QLabel("Search"))
        search_row.addWidget(self.search_edit, 1)
        root.addLayout(search_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Name", "Kind", "Shape", "Dtype", "Preview / attrs"])
        self.tree.header().setStretchLastSection(True)
        self.tree.itemSelectionChanged.connect(self._show_selected_details)
        splitter.addWidget(self.tree)

        right = QtWidgets.QTabWidget()
        self.details_text = QtWidgets.QPlainTextEdit()
        self.details_text.setReadOnly(True)
        self.metadata_text = QtWidgets.QPlainTextEdit()
        self.metadata_text.setReadOnly(True)
        right.addTab(self.details_text, "Selected Item")
        right.addTab(self.metadata_text, "Metadata Summary")
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.status_label = QtWidgets.QLabel("Choose an HDF5 file.")
        root.addWidget(self.status_label)

    def open_file(self, path: Path | str) -> None:
        self.path_edit.setText(str(path))
        self.load_file()
        self.show()
        self.raise_()
        self.activateWindow()

    def browse_file(self) -> None:
        start = Path(self.path_edit.text()).parent if self.path_edit.text().strip() else Path.home()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open HDF5 structure viewer", str(start), H5_FILTER)
        if path:
            self.path_edit.setText(path)
            self.load_file()

    def load_file(self) -> None:
        path = Path(self.path_edit.text().strip())
        self.tree.clear()
        self.details_text.clear()
        self.metadata_text.clear()
        if not path.exists():
            self.status_label.setText("HDF5 file does not exist.")
            return
        counter = {"items": 0, "truncated": False}
        try:
            with h5py.File(path, "r") as handle:
                root = QtWidgets.QTreeWidgetItem(["/", "file", "", "", _attrs_preview(handle)])
                root.setData(0, QtCore.Qt.UserRole, "/")
                self.tree.addTopLevelItem(root)
                add_h5_children(root, handle, counter)
                root.setExpanded(True)
                self.metadata_text.setPlainText(metadata_summary(handle))
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Could not read HDF5: {exc}")
            return
        suffix = " Tree truncated for speed." if counter["truncated"] else ""
        self.status_label.setText(f"Loaded {counter['items']} HDF5 items from {path}.{suffix}")
        self._apply_tree_filter()

    def _show_selected_details(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            self.details_text.clear()
            return
        item = items[0]
        path = item.data(0, QtCore.Qt.UserRole) or item.text(0)
        lines = [
            f"Path: {path}",
            f"Kind: {item.text(1)}",
            f"Shape: {item.text(2)}",
            f"Dtype: {item.text(3)}",
            "",
            item.text(4),
        ]
        self.details_text.setPlainText("\n".join(line for line in lines if line is not None))

    def _apply_tree_filter(self) -> None:
        needle = self.search_edit.text().strip().lower()

        def apply(item: QtWidgets.QTreeWidgetItem) -> bool:
            text = " ".join(item.text(column) for column in range(item.columnCount())).lower()
            child_match = False
            for child_index in range(item.childCount()):
                child_match = apply(item.child(child_index)) or child_match
            matched = not needle or needle in text or child_match
            item.setHidden(not matched)
            if needle and child_match:
                item.setExpanded(True)
            return matched

        for index in range(self.tree.topLevelItemCount()):
            apply(self.tree.topLevelItem(index))


def discover_iq_curves(handle: h5py.File, source_path: Path | None = None, source_label: str = "") -> list[H5CurveRecord]:
    records: list[H5CurveRecord] = []

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Group):
            return
        q_obj = _q_dataset_for_group(obj)
        if q_obj is None:
            return
        if q_obj.ndim == 0 or q_obj.ndim > 2:
            return
        candidates = _curve_y_datasets(obj, q_obj)
        energy = obj.get("energy")
        for y_name, i_obj in candidates:
            rows = _curve_row_count(q_obj, i_obj)
            rows = min(rows, MAX_CURVE_ROWS_PER_GROUP)
            sigma_path = _matching_sigma_path(obj, y_name, i_obj.shape)
            for row in range(rows):
                row_index = row if _dataset_has_curve_rows(i_obj) else None
                label = _curve_label(obj, y_name, row_index, energy)
                if source_label:
                    label = f"{source_label} | {label}"
                records.append(
                    H5CurveRecord(
                        label=label,
                        h5_path=str(source_path) if source_path is not None else None,
                        group_path=obj.name,
                        q_path=q_obj.name,
                        i_path=i_obj.name,
                        y_name=y_name,
                        sigma_path=sigma_path,
                        row=row_index,
                    )
                )

    handle.visititems(visitor)
    return sorted(records, key=lambda record: (_stage_sort_key(record.group_path, record.y_name), record.label))


def read_curve_row(handle: h5py.File, record: H5CurveRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    q = np.asarray(handle[record.q_path][()], dtype=float)
    intensity = np.asarray(handle[record.i_path][()], dtype=float)
    sigma = np.asarray(handle[record.sigma_path][()], dtype=float) if record.sigma_path else None
    if record.row is not None:
        intensity = intensity[record.row]
        if sigma is not None and sigma.ndim == 2:
            sigma = sigma[record.row]
        if q.ndim == 2:
            q = q[record.row if record.row < q.shape[0] else 0]
    return q, intensity, sigma


def add_h5_children(parent: QtWidgets.QTreeWidgetItem, group: h5py.Group | h5py.File, counter: dict[str, Any]) -> None:
    for name in sorted(group.keys()):
        if counter["items"] >= MAX_TREE_ITEMS:
            counter["truncated"] = True
            return
        child = group[name]
        counter["items"] += 1
        if isinstance(child, h5py.Group):
            item = QtWidgets.QTreeWidgetItem([name, "group", "", "", _attrs_preview(child)])
            item.setData(0, QtCore.Qt.UserRole, child.name)
            parent.addChild(item)
            add_h5_children(item, child, counter)
        elif isinstance(child, h5py.Dataset):
            item = QtWidgets.QTreeWidgetItem([name, "dataset", str(child.shape), str(child.dtype), _dataset_preview(child)])
            item.setData(0, QtCore.Qt.UserRole, child.name)
            parent.addChild(item)
            attrs = _attrs_preview(child)
            if attrs:
                attrs_item = QtWidgets.QTreeWidgetItem(["@attrs", "attrs", "", "", attrs])
                attrs_item.setData(0, QtCore.Qt.UserRole, child.name + "/@attrs")
                item.addChild(attrs_item)


def metadata_summary(handle: h5py.File) -> str:
    sections: list[str] = []

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Group):
            return
        base_name = name.rsplit("/", 1)[-1]
        if base_name not in METADATA_GROUP_NAMES and not name.endswith("original_metadata"):
            return
        lines = _metadata_group_lines(obj)
        if lines:
            sections.append(f"[/{name}]\n" + "\n".join(lines))

    handle.visititems(visitor)
    return "\n\n".join(sections) if sections else "No metadata/parameter groups found."


def _metadata_group_lines(group: h5py.Group, limit: int = 80) -> list[str]:
    lines: list[str] = []
    for key in sorted(group.keys()):
        child = group[key]
        if isinstance(child, h5py.Dataset):
            lines.append(f"{key}: {_dataset_metadata_value(child)}")
        elif isinstance(child, h5py.Group):
            scalar_children = [
                subkey
                for subkey in sorted(child.keys())
                if isinstance(child[subkey], h5py.Dataset) and (child[subkey].shape == () or child[subkey].size <= 8)
            ]
            if scalar_children:
                lines.append(f"{key}/: " + "; ".join(f"{subkey}={_dataset_metadata_value(child[subkey])}" for subkey in scalar_children[:8]))
        if len(lines) >= limit:
            lines.append("...")
            break
    return lines


def _dataset_metadata_value(dataset: h5py.Dataset) -> str:
    try:
        if dataset.shape == () or dataset.size <= 8:
            return _value_preview(dataset[()])
    except Exception:  # noqa: BLE001
        pass
    return f"shape={dataset.shape}, dtype={dataset.dtype}"


def _curve_label(group: h5py.Group, y_name: str, row: int | None, energy: h5py.Dataset | None) -> str:
    base = _friendly_group_label(group.name)
    y_label = y_name if y_name != "I" else "I"
    if row is None:
        return f"{base} | {y_label}"
    energy_label = ""
    if isinstance(energy, h5py.Dataset) and energy.ndim <= 1 and row < energy.shape[0]:
        try:
            energy_value = float(energy[row])
            if np.isfinite(energy_value):
                energy_label = f" {energy_value:.4f} keV"
        except (TypeError, ValueError):
            energy_label = ""
    return f"{base} | {y_label} row {row + 1:03d}{energy_label}"


def _friendly_group_label(path: str) -> str:
    text = path.strip("/")
    replacements = {
        "entry/asaxs_outputs/": "ASAXS final / ",
        "entry/final/corrected_I_q_E": "Legacy final",
        "entry/process_01_reduction/data": "Detector group averages",
        "entry/process_02_background_subtraction/data": "Background corrected",
        "entry/process_03_glassy_carbon_normalization/data": "GC normalized",
        "entry/stitched_averages/curves/": "Stitched / ",
    }
    for old, new in replacements.items():
        if text.startswith(old):
            return text.replace(old, new, 1)
    return "/" + text


def _is_numeric_dataset(dataset: h5py.Dataset) -> bool:
    return np.issubdtype(dataset.dtype, np.number)


def _h5_paths_from_file_or_folder(path: Path) -> list[Path]:
    if path.is_file():
        paths = [path]
        for detector in ("Pil300K", "Eig1M"):
            detector_dir = path.parent / detector
            if detector_dir.exists() and detector_dir.is_dir():
                paths.extend(sorted(detector_dir.glob("*_analysis.h5")))
                paths.extend(sorted(detector_dir.glob("*_analysis.hdf5")))
        return _unique_paths(paths)
    patterns = ("*_analysis.h5", "*_analysis.hdf5", "analysis.h5", "analysis.hdf5")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(path.rglob(pattern)))
    return _unique_paths(paths)


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for h5_path in paths:
        resolved = h5_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(h5_path)
    return unique


def _source_label(root: Path, h5_path: Path) -> str:
    try:
        relative = h5_path.relative_to(root)
    except ValueError:
        relative = h5_path.name
    text = str(relative).replace("\\", "/")
    return text[:-3] if text.endswith(".h5") else text


def _curve_y_datasets(group: h5py.Group, q_dataset: h5py.Dataset) -> list[tuple[str, h5py.Dataset]]:
    candidates: list[tuple[str, h5py.Dataset]] = []
    for name in sorted(group.keys(), key=_y_dataset_sort_key):
        if group[name].name == q_dataset.name:
            continue
        child = group[name]
        if not isinstance(child, h5py.Dataset) or not _is_numeric_dataset(child):
            continue
        if not _is_curve_compatible(q_dataset, child):
            continue
        candidates.append((name, child))
    return candidates


def _q_dataset_for_group(group: h5py.Group) -> h5py.Dataset | None:
    for name in Q_DATASET_NAMES:
        child = group.get(name)
        if isinstance(child, h5py.Dataset) and _is_numeric_dataset(child) and _looks_like_q_grid(child):
            return child
    for name, child in group.items():
        if not isinstance(child, h5py.Dataset) or not _is_numeric_dataset(child) or not _looks_like_q_grid(child):
            continue
        lowered = name.lower()
        if lowered in {"q", "q_values", "q_grid"} or lowered.startswith(("q_inv", "q_a", "q_nm")):
            return child
    return None


def _looks_like_q_grid(dataset: h5py.Dataset) -> bool:
    return 1 <= dataset.ndim <= 2 and int(dataset.size) >= MIN_Q_GRID_POINTS


def _is_curve_compatible(q_dataset: h5py.Dataset, y_dataset: h5py.Dataset) -> bool:
    if y_dataset.ndim == 0 or y_dataset.ndim > 2:
        return False
    q_shape = q_dataset.shape
    y_shape = y_dataset.shape
    if q_dataset.ndim == 1 and y_dataset.ndim == 1:
        return q_shape[0] == y_shape[0]
    if q_dataset.ndim == 2 and y_dataset.ndim == 2:
        return q_shape == y_shape
    if q_dataset.ndim == 2 and y_dataset.ndim == 1:
        return q_shape[1] == y_shape[0]
    if q_dataset.ndim == 1 and y_dataset.ndim == 2:
        return q_shape[0] == y_shape[1]
    return False


def _curve_row_count(q_dataset: h5py.Dataset, y_dataset: h5py.Dataset) -> int:
    if _dataset_has_curve_rows(y_dataset):
        return int(y_dataset.shape[0])
    if q_dataset.ndim == 2 and y_dataset.ndim == 1:
        return 1
    return 1


def _dataset_has_curve_rows(dataset: h5py.Dataset) -> bool:
    return dataset.ndim == 2


def _matching_sigma_path(group: h5py.Group, y_name: str, y_shape: tuple[int, ...]) -> str | None:
    candidates = []
    if y_name == "I":
        candidates.extend(["sigma_I", "I_error", "error"])
    elif y_name.startswith("I_"):
        candidates.append("sigma_" + y_name[2:])
    elif y_name.startswith("intensity"):
        candidates.append(y_name.replace("intensity", "sigma", 1))
    for candidate in candidates:
        child = group.get(candidate)
        if isinstance(child, h5py.Dataset) and _is_numeric_dataset(child) and child.shape == y_shape:
            return child.name
    return None


def _y_dataset_sort_key(name: str) -> tuple[int, str]:
    if name == "I":
        return (0, name)
    if name.startswith("I_"):
        return (1, name)
    if name.startswith("sigma"):
        return (2, name)
    return (3, name)


def _stage_sort_key(path: str, y_name: str) -> tuple[int, str, str]:
    order = [
        "asaxs_outputs",
        "/final/",
        "process_03_glassy_carbon_normalization",
        "process_02_background_subtraction",
        "stitched_averages",
        "process_01_reduction",
    ]
    rank = next((index for index, token in enumerate(order) if token in path), len(order))
    return (rank, path, y_name)


def _attrs_preview(obj: h5py.Group | h5py.Dataset | h5py.File) -> str:
    parts: list[str] = []
    for key in sorted(obj.attrs.keys()):
        parts.append(f"{key}={_value_preview(obj.attrs[key])}")
        if len(parts) >= 8:
            parts.append("...")
            break
    return "; ".join(parts)


def _dataset_preview(dataset: h5py.Dataset) -> str:
    attrs = _attrs_preview(dataset)
    if dataset.size == 0:
        return attrs
    preview = ""
    if dataset.shape == () or dataset.size <= 8:
        try:
            preview = _value_preview(dataset[()])
        except Exception:  # noqa: BLE001
            preview = ""
    if attrs and preview:
        return f"{preview}; attrs: {attrs}"
    return preview or attrs


def _value_preview(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    array = np.asarray(value)
    if array.shape == ():
        scalar = array.item()
        if isinstance(scalar, bytes):
            return scalar.decode("utf-8", errors="replace")
        return str(scalar)
    flat = array.reshape(-1)
    shown = ", ".join(_value_preview(item) for item in flat[:4])
    suffix = ", ..." if flat.size > 4 else ""
    return f"[{shown}{suffix}]"
