"""Frame-resolved SAXS stability metrics and derived-HDF5 persistence."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class FrameStabilitySettings:
    q_min: float | None = None
    q_max: float | None = None
    low_q_min: float | None = None
    low_q_max: float | None = None
    reference_mode: str = "first"
    good_drift_fraction: float = 0.02
    max_drift_fraction: float = 0.05
    good_chi2: float = 1.5
    max_chi2: float = 3.0
    cormap_alpha: float = 0.01
    consecutive_failures: int = 3


@dataclass(frozen=True)
class FrameSeries:
    q: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    frame_index: np.ndarray
    sequence_index: np.ndarray
    energy_index: np.ndarray
    group_index: np.ndarray
    energy_kev: np.ndarray
    monitor_value: np.ndarray
    source_path: list[str]
    existing_status: list[str]


@dataclass(frozen=True)
class FrameStabilityResult:
    frame_index: np.ndarray
    q_common: np.ndarray
    intensity_common: np.ndarray
    sigma_common: np.ndarray
    relative_intensity: np.ndarray
    invariant_ratio: np.ndarray
    low_q_ratio: np.ndarray
    reduced_chi2: np.ndarray
    cormap_p: np.ndarray
    longest_run: np.ndarray
    rg: np.ndarray
    i0: np.ndarray
    peak_q: np.ndarray
    peak_fwhm: np.ndarray
    labels: list[str]
    recommended: np.ndarray
    first_failure_frame: int | None
    damage_onset_frame: int | None
    q_range: tuple[float, float]
    low_q_range: tuple[float, float]


@dataclass(frozen=True)
class FrameSourceItem:
    sequence_index: int
    energy_index: int
    group_index: int
    frame_index: int
    path: Path


@dataclass(frozen=True)
class FrameSourceSeries:
    label: str
    detector: str
    items: tuple[FrameSourceItem, ...]
    poni_path: Path
    mask_path: Path
    dataset_path: str
    monitor_key: str
    npt: int
    unit: str


def analyze_frame_series(series: FrameSeries, settings: FrameStabilitySettings | None = None) -> FrameStabilityResult:
    settings = settings or FrameStabilitySettings()
    order = np.argsort(np.asarray(series.frame_index, dtype=int), kind="stable")
    frame_index = np.asarray(series.frame_index, dtype=int)[order]
    intensity = _as_frame_rows(series.intensity)[order]
    sigma = _as_frame_rows(series.sigma)[order]
    q_rows = _q_rows(series.q, intensity.shape[0])[order]
    if intensity.shape[0] == 0:
        raise ValueError("Frame series is empty.")

    q_common, q_range = _common_q_grid(q_rows, settings.q_min, settings.q_max)
    intensity_common = np.vstack([_interp_row(q_common, q, row) for q, row in zip(q_rows, intensity)])
    sigma_common = np.vstack([_interp_row(q_common, q, row) for q, row in zip(q_rows, sigma)])
    low_q_range = _low_q_range(q_common, settings.low_q_min, settings.low_q_max)
    low_mask = (q_common >= low_q_range[0]) & (q_common <= low_q_range[1])

    invariant = np.asarray([_trapz(q_common**2 * row, q_common) for row in intensity_common], dtype=float)
    low_q = np.asarray([np.nanmean(row[low_mask]) for row in intensity_common], dtype=float)
    invariant_ratio = _safe_ratio(invariant, invariant[0])
    low_q_ratio = _safe_ratio(low_q, low_q[0])

    reduced_chi2 = np.full(frame_index.size, np.nan, dtype=float)
    cormap_p = np.full(frame_index.size, np.nan, dtype=float)
    longest_run = np.zeros(frame_index.size, dtype=int)
    relative = np.full_like(intensity_common, np.nan, dtype=float)
    reference_first = intensity_common[0]
    relative[:] = _safe_array_ratio(intensity_common, reference_first)
    for row in range(frame_index.size):
        reference_row = 0 if settings.reference_mode == "first" or row == 0 else row - 1
        reduced_chi2[row] = reduced_chi_square(
            intensity_common[row],
            intensity_common[reference_row],
            sigma_common[row],
            sigma_common[reference_row],
        )
        cormap_p[row], longest_run[row] = cormap_p_value(intensity_common[row], intensity_common[reference_row])

    rg = np.full(frame_index.size, np.nan, dtype=float)
    i0 = np.full(frame_index.size, np.nan, dtype=float)
    peak_q = np.full(frame_index.size, np.nan, dtype=float)
    peak_fwhm = np.full(frame_index.size, np.nan, dtype=float)
    for row, values in enumerate(intensity_common):
        rg[row], i0[row] = guinier_estimate(q_common[low_mask], values[low_mask])
        peak_q[row], peak_fwhm[row] = peak_metrics(q_common, values)

    labels = ["Good"]
    bad = np.zeros(frame_index.size, dtype=bool)
    for row in range(1, frame_index.size):
        invariant_drift = abs(invariant_ratio[row] - 1.0)
        low_q_drift = abs(low_q_ratio[row] - 1.0)
        cormap_failed = np.isfinite(cormap_p[row]) and cormap_p[row] < settings.cormap_alpha
        chi_bad = np.isfinite(reduced_chi2[row]) and reduced_chi2[row] > settings.max_chi2
        drift_bad = invariant_drift > settings.max_drift_fraction or low_q_drift > settings.max_drift_fraction
        statistically_bad = cormap_failed and chi_bad and max(invariant_drift, low_q_drift) > settings.good_drift_fraction
        bad[row] = drift_bad or statistically_bad
        if bad[row]:
            labels.append("Bad")
            continue
        good = (
            invariant_drift <= settings.good_drift_fraction
            and low_q_drift <= settings.good_drift_fraction
            and (not np.isfinite(reduced_chi2[row]) or reduced_chi2[row] <= settings.good_chi2)
        )
        labels.append("Good" if good else "Acceptable")

    first_failure_pos = next((index for index, value in enumerate(bad) if value), None)
    onset_pos = _first_consecutive_true(bad, settings.consecutive_failures)
    recommended = np.ones(frame_index.size, dtype=bool)
    if first_failure_pos is not None:
        recommended[first_failure_pos:] = False
    return FrameStabilityResult(
        frame_index=frame_index,
        q_common=q_common,
        intensity_common=intensity_common,
        sigma_common=sigma_common,
        relative_intensity=relative,
        invariant_ratio=invariant_ratio,
        low_q_ratio=low_q_ratio,
        reduced_chi2=reduced_chi2,
        cormap_p=cormap_p,
        longest_run=longest_run,
        rg=rg,
        i0=i0,
        peak_q=peak_q,
        peak_fwhm=peak_fwhm,
        labels=labels,
        recommended=recommended,
        first_failure_frame=int(frame_index[first_failure_pos]) if first_failure_pos is not None else None,
        damage_onset_frame=int(frame_index[onset_pos]) if onset_pos is not None else None,
        q_range=q_range,
        low_q_range=low_q_range,
    )


def reduced_chi_square(
    values: np.ndarray,
    reference: np.ndarray,
    sigma: np.ndarray,
    reference_sigma: np.ndarray,
) -> float:
    denominator = np.square(sigma) + np.square(reference_sigma)
    mask = np.isfinite(values) & np.isfinite(reference) & np.isfinite(denominator) & (denominator > 0)
    count = int(np.count_nonzero(mask))
    if count < 2:
        return float("nan")
    return float(np.sum(np.square(values[mask] - reference[mask]) / denominator[mask]) / (count - 1))


def cormap_p_value(values: np.ndarray, reference: np.ndarray) -> tuple[float, int]:
    difference = np.asarray(values, dtype=float) - np.asarray(reference, dtype=float)
    signs = np.sign(difference[np.isfinite(difference)])
    signs = signs[signs != 0]
    if signs.size == 0:
        return 1.0, 0
    longest = _longest_equal_run(signs)
    return _longest_run_tail_probability(int(signs.size), longest), longest


def guinier_estimate(q: np.ndarray, intensity: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(q) & np.isfinite(intensity) & (q > 0) & (intensity > 0)
    if np.count_nonzero(mask) < 5:
        return float("nan"), float("nan")
    x = np.square(q[mask])
    y = np.log(intensity[mask])
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or slope >= 0:
        return float("nan"), float(np.exp(intercept)) if np.isfinite(intercept) else float("nan")
    return float(np.sqrt(-3.0 * slope)), float(np.exp(intercept))


def peak_metrics(q: np.ndarray, intensity: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(q) & np.isfinite(intensity)
    if np.count_nonzero(mask) < 3:
        return float("nan"), float("nan")
    x = q[mask]
    y = intensity[mask]
    peak = int(np.nanargmax(y))
    peak_q = float(x[peak])
    baseline = float(np.nanmin(y))
    half = baseline + (float(y[peak]) - baseline) / 2.0
    left = np.where(y[: peak + 1] <= half)[0]
    right = np.where(y[peak:] <= half)[0]
    if left.size == 0 or right.size == 0:
        return peak_q, float("nan")
    return peak_q, float(x[peak + right[0]] - x[left[-1]])


def discover_frame_source_series(analysis_h5: Path) -> dict[str, FrameSourceSeries]:
    """Discover post-reduction frame series from analysis-HDF5 provenance."""
    referenced_detector_files: set[Path] = set()
    with h5py.File(analysis_h5, "r") as handle:
        processes: dict[str, h5py.Group] = {}

        def visitor(_name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if not isinstance(obj, h5py.Group):
                return
            basename = obj.name.rsplit("/", 1)[-1]
            if not basename.startswith("process_01_reduction") or "metadata" not in obj or "parameters" not in obj:
                return
            parent = obj.parent.name
            previous = processes.get(parent)
            if previous is None or obj.name > previous.name:
                processes[parent] = obj

        handle.visititems(visitor)
        discovered: dict[str, FrameSourceSeries] = {}
        for parent, process in processes.items():
            metadata = process["metadata"]
            parameters = process["parameters"]
            manifest_path = Path(_h5_text(metadata, "input_h5_file", "")).expanduser()
            items = _manifest_items(manifest_path) if manifest_path.is_file() else _items_from_data_reference(process)
            if not items:
                continue
            poni_path = Path(_h5_text(parameters, "poni_file", "")).expanduser()
            mask_path = Path(_h5_text(parameters, "mask_file", "")).expanduser()
            dataset_path = _h5_text(metadata, "input_data_path", "entry/data/data")
            normalization = _h5_text(parameters, "normalization_method", "")
            detector = _h5_text(parameters, "detector", "") or _detector_prefix(parent).strip(" |") or analysis_h5.parent.name
            monitor_key = normalization.split("monitor:", 1)[-1] if "monitor:" in normalization else ("SPDS" if "Pil" in detector else "WPDS")
            npt = int(_h5_scalar(parameters, "n_q_bins", 1000))
            unit = _h5_text(parameters, "q_unit", "q_A^-1")
            prefix = f"{detector} | " if detector else ""
            for energy_index, group_index in sorted({(item.energy_index, item.group_index) for item in items}):
                selected = tuple(
                    item for item in items if item.energy_index == energy_index and item.group_index == group_index
                )
                label = f"{prefix}E{energy_index:03d} G{group_index:03d}"
                discovered[label] = FrameSourceSeries(
                    label=label,
                    detector=detector,
                    items=selected,
                    poni_path=poni_path,
                    mask_path=mask_path,
                    dataset_path=dataset_path,
                    monitor_key=monitor_key,
                    npt=npt,
                    unit=unit,
                )
        if not discovered:
            stitched = handle.get("/entry/stitched_averages/curves")
            if isinstance(stitched, h5py.Group):
                for curve in stitched.values():
                    if not isinstance(curve, h5py.Group):
                        continue
                    for name in ("low_q_analysis_h5", "high_q_analysis_h5"):
                        value = curve.attrs.get(name)
                        if value:
                            text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
                            referenced_detector_files.add(Path(text).expanduser())
    for referenced in sorted(referenced_detector_files):
        try:
            resolved = referenced.resolve()
            if resolved == analysis_h5.resolve():
                continue
            for label, source in discover_frame_source_series(resolved).items():
                discovered.setdefault(label, source)
        except (OSError, RuntimeError, ValueError):
            continue
    if not discovered:
        for detector in ("Pil300K", "Eig1M"):
            detector_dir = analysis_h5.parent / detector
            if not detector_dir.is_dir():
                continue
            for detector_h5 in sorted(detector_dir.glob("*_analysis.h5")):
                try:
                    for label, source in discover_frame_source_series(detector_h5).items():
                        discovered.setdefault(label, source)
                except (OSError, RuntimeError, ValueError):
                    continue
    return discovered


def reduce_source_series(source: FrameSourceSeries, progress_queue: object | None = None) -> FrameSeries:
    """Re-integrate one provenance-selected series without writing raw or analysis HDF5."""
    from aswaxs_live.core.reduce_aswaxs_sequence import ManifestItem, reduce_manifest_frames

    missing = [path for path in (source.poni_path, source.mask_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing reduction file(s): " + ", ".join(str(path) for path in missing))
    manifest_items = [
        ManifestItem(item.sequence_index, item.energy_index, item.group_index, item.frame_index, item.path)
        for item in source.items
    ]
    args = argparse.Namespace(
        poni=str(source.poni_path),
        mask=str(source.mask_path),
        dataset_path=source.dataset_path,
        npt=source.npt,
        unit=source.unit,
        delta_energy_percent=1e-3,
    )
    curves = reduce_manifest_frames(
        manifest_items,
        args,
        source.monitor_key,
        progress_label=source.label,
        progress_queue=progress_queue,
    )
    return FrameSeries(
        q=_stack_rows([curve.q for curve in curves]),
        intensity=_stack_rows([curve.normalized_intensity for curve in curves]),
        sigma=_stack_rows([curve.normalized_error for curve in curves]),
        frame_index=np.asarray([curve.item.frame_index for curve in curves], dtype=int),
        sequence_index=np.asarray([curve.item.sequence_index for curve in curves], dtype=int),
        energy_index=np.asarray([curve.item.energy_index for curve in curves], dtype=int),
        group_index=np.asarray([curve.item.group_index for curve in curves], dtype=int),
        energy_kev=np.asarray([np.nan if curve.energy_kev is None else curve.energy_kev for curve in curves]),
        monitor_value=np.asarray([curve.monitor_value for curve in curves], dtype=float),
        source_path=[str(curve.item.path) for curve in curves],
        existing_status=["provenance_reduced"] * len(curves),
    )


def _manifest_items(path: Path) -> list[FrameSourceItem]:
    items: list[FrameSourceItem] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                items.append(
                    FrameSourceItem(
                        sequence_index=int(row["sequence_index"]),
                        energy_index=int(row["energy_index"]),
                        group_index=int(row["group_index"]),
                        frame_index=int(row["frame_index"]),
                        path=Path(row["hdf5_path"]).expanduser(),
                    )
                )
    except (KeyError, OSError, TypeError, ValueError):
        return []
    return items


def _items_from_data_reference(process: h5py.Group) -> list[FrameSourceItem]:
    parent = process.parent
    if "data_reference" not in parent or "data_file" not in parent["data_reference"] or "data" not in process:
        return []
    raw_paths = [Path(value).expanduser() for value in _h5_text_list(parent["data_reference"]["data_file"])]
    data = process["data"]
    if "energy_index" not in data or "group_index" not in data:
        return []
    pairs = list(zip(np.asarray(data["energy_index"][()], dtype=int), np.asarray(data["group_index"][()], dtype=int)))
    if not pairs or len(raw_paths) % len(pairs) != 0:
        return []
    frames_per_group = len(raw_paths) // len(pairs)
    items: list[FrameSourceItem] = []
    for sequence, raw_path in enumerate(raw_paths, start=1):
        pair_index, frame_offset = divmod(sequence - 1, frames_per_group)
        energy_index, group_index = pairs[pair_index]
        items.append(FrameSourceItem(sequence, int(energy_index), int(group_index), frame_offset + 1, raw_path))
    return items


def _common_q_grid(q_rows: np.ndarray, q_min: float | None, q_max: float | None) -> tuple[np.ndarray, tuple[float, float]]:
    lows = [float(np.nanmin(row[np.isfinite(row)])) for row in q_rows if np.any(np.isfinite(row))]
    highs = [float(np.nanmax(row[np.isfinite(row)])) for row in q_rows if np.any(np.isfinite(row))]
    if not lows or not highs:
        raise ValueError("Frame series has no finite q values.")
    low = max(lows) if q_min is None else max(max(lows), float(q_min))
    high = min(highs) if q_max is None else min(min(highs), float(q_max))
    reference = np.asarray(q_rows[0], dtype=float)
    mask = np.isfinite(reference) & (reference >= low) & (reference <= high)
    if np.count_nonzero(mask) < 8:
        raise ValueError(f"Selected q range {low:g}-{high:g} contains fewer than 8 common points.")
    return reference[mask], (low, high)


def _low_q_range(q: np.ndarray, low: float | None, high: float | None) -> tuple[float, float]:
    q_low = float(np.nanmin(q))
    q_high = float(np.nanmax(q))
    default_high = q_low + 0.1 * (q_high - q_low)
    selected_low = q_low if low is None else max(q_low, float(low))
    selected_high = default_high if high is None else min(q_high, float(high))
    if selected_high <= selected_low or np.count_nonzero((q >= selected_low) & (q <= selected_high)) < 3:
        selected_low, selected_high = q_low, float(q[min(q.size - 1, max(2, q.size // 10))])
    return selected_low, selected_high


def _interp_row(q_target: np.ndarray, q_source: np.ndarray, values: np.ndarray) -> np.ndarray:
    mask = np.isfinite(q_source) & np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return np.full_like(q_target, np.nan)
    order = np.argsort(q_source[mask])
    return np.interp(q_target, q_source[mask][order], values[mask][order], left=np.nan, right=np.nan)


def _longest_equal_run(signs: np.ndarray) -> int:
    longest = current = 1
    for index in range(1, signs.size):
        current = current + 1 if signs[index] == signs[index - 1] else 1
        longest = max(longest, current)
    return int(longest)


def _longest_run_tail_probability(n: int, observed: int) -> float:
    if n <= 0 or observed <= 0:
        return 1.0
    probabilities = np.zeros(observed, dtype=float)
    if observed > 1:
        probabilities[1] = 1.0
    for _ in range(1, n):
        updated = np.zeros_like(probabilities)
        updated[1] += 0.5 * np.sum(probabilities)
        if observed > 2:
            updated[2:] += 0.5 * probabilities[1:-1]
        probabilities = updated
    return float(np.clip(1.0 - np.sum(probabilities), 0.0, 1.0))


def _first_consecutive_true(values: np.ndarray, count: int) -> int | None:
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if value else 0
        if run >= max(1, count):
            return index - run + 1
    return None


def _safe_ratio(values: np.ndarray, reference: float) -> np.ndarray:
    if not np.isfinite(reference) or abs(reference) < 1e-30:
        return np.full_like(values, np.nan, dtype=float)
    return np.asarray(values, dtype=float) / reference


def _safe_array_ratio(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    np.divide(values, reference, out=result, where=np.isfinite(reference) & (np.abs(reference) > 1e-30))
    return result


def _trapz(values: np.ndarray, q: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(q)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    x = q[mask]
    y = values[mask]
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5))


def _as_frame_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array.reshape(1, -1) if array.ndim == 1 else array


def _q_rows(q: np.ndarray, rows: int) -> np.ndarray:
    q_array = np.asarray(q, dtype=float)
    if q_array.ndim == 1:
        return np.repeat(q_array.reshape(1, -1), rows, axis=0)
    return q_array


def _stack_rows(rows: list[np.ndarray]) -> np.ndarray:
    width = max(np.asarray(row).size for row in rows)
    stacked = np.full((len(rows), width), np.nan, dtype=float)
    for index, row in enumerate(rows):
        values = np.asarray(row, dtype=float).reshape(-1)
        stacked[index, : values.size] = values
    return stacked


def _h5_scalar(group: h5py.Group, name: str, default: object) -> object:
    if name not in group or not isinstance(group[name], h5py.Dataset):
        return default
    value = group[name][()]
    array = np.asarray(value)
    if array.size == 0:
        return default
    scalar = array.reshape(-1)[0]
    return scalar.item() if isinstance(scalar, np.generic) else scalar


def _h5_text(group: h5py.Group, name: str, default: str) -> str:
    value = _h5_scalar(group, name, default)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _h5_text_list(dataset: h5py.Dataset) -> list[str]:
    values = np.asarray(dataset[()]).reshape(-1)
    return [value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value) for value in values]


def _detector_prefix(path: str) -> str:
    parts = path.strip("/").split("/")
    for detector in ("Pil300K", "Eig1M"):
        if detector in parts:
            return f"{detector} | "
    return ""
