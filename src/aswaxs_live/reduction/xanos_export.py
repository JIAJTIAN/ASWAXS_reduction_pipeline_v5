"""Export final ASAXS curves from analysis HDF5 to XAnoS-style text files."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import h5py
import numpy as np


XANOS_FOLDER_NAME = "XAnos format"
XANOS_FILE_LIST_NAME = "xanos_file_list.txt"


def export_analysis_h5_to_xanos_format(
    analysis_h5_path: str | Path,
    saxs_output_name: str | None = None,
    force_saxs: bool = False,
) -> list[Path]:
    """Write one XAnoS-compatible final sample curve per energy.

    The analysis HDF5 remains the authoritative processing record. This helper
    creates the compatibility text files expected by the older XAnoS component
    workflow:

    ``XAnos format/energy_001_sample_final.dat``
    ``XAnos format/energy_002_sample_final.dat``
    ``XAnos format/xanos_file_list.txt``
    """
    path = Path(analysis_h5_path).expanduser().resolve()
    if not path.exists():
        return []
    with h5py.File(path, "r") as handle:
        if force_saxs:
            fallback = _saxs_xanos_payload(handle, path, output_name=saxs_output_name)
            if fallback is None:
                return []
            output_name, q, intensity, sigma, energy, h5_data_path, header_rows = fallback
            use_energy_prefix = False
        else:
            use_energy_prefix = True
            named_root = handle.get("/entry/asaxs_outputs")
            if named_root is not None:
                named_payloads = []
                for output_name in sorted(named_root):
                    group = named_root[output_name].get("corrected_I_q_E")
                    if group is None or "q" not in group or "I" not in group:
                        continue
                    rows = int(group["I"].shape[0]) if group["I"].ndim > 1 else 1
                    named_payloads.append(
                        (
                            output_name,
                            np.asarray(group["q"][()], dtype=float),
                            np.asarray(group["I"][()], dtype=float),
                            np.asarray(group["sigma_I"][()], dtype=float) if "sigma_I" in group else None,
                            np.asarray(group["energy"][()], dtype=float) if "energy" in group else None,
                            _xanos_header_rows(handle, group, rows),
                        )
                    )
                if named_payloads:
                    written: list[Path] = []
                    for output_name, q, intensity, sigma, energy, header_rows in named_payloads:
                        written.extend(
                            _write_xanos_payload(
                                path,
                                output_name,
                                q,
                                intensity,
                                sigma,
                                energy,
                                f"/entry/asaxs_outputs/{output_name}/corrected_I_q_E",
                                header_rows,
                                use_energy_prefix=True,
                            )
                        )
                    return written

            group = handle.get("/entry/final/corrected_I_q_E")
            if group is not None and "q" in group and "I" in group:
                q = np.asarray(group["q"][()], dtype=float)
                intensity = np.asarray(group["I"][()], dtype=float)
                sigma = np.asarray(group["sigma_I"][()], dtype=float) if "sigma_I" in group else np.full_like(intensity, np.nan)
                energy = np.asarray(group["energy"][()], dtype=float) if "energy" in group else np.full((intensity.shape[0],), np.nan)
                rows = int(intensity.shape[0]) if intensity.ndim > 1 else 1
                header_rows = _xanos_header_rows(handle, group, rows)
                output_name = "sample"
                h5_data_path = "/entry/final/corrected_I_q_E"
                if _all_missing_energy(energy):
                    energy = _fallback_energy_values(handle, intensity.shape[0])
            else:
                fallback = _saxs_xanos_payload(handle, path, output_name=saxs_output_name)
                if fallback is None:
                    return []
                output_name, q, intensity, sigma, energy, h5_data_path, header_rows = fallback
                use_energy_prefix = False

    if intensity.ndim == 1:
        intensity = intensity.reshape(1, -1)
    if sigma.ndim == 1:
        sigma = sigma.reshape(1, -1)
    if energy.ndim == 0:
        energy = energy.reshape(1)

    return _write_xanos_payload(
        path,
        output_name,
        q,
        intensity,
        sigma,
        energy,
        h5_data_path,
        header_rows,
        use_energy_prefix=use_energy_prefix,
    )


def _write_xanos_payload(
    analysis_h5: Path,
    output_name: str,
    q: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray | None,
    energy: np.ndarray | None,
    h5_data_path: str,
    header_rows: list[dict[str, float | str | None]] | None = None,
    use_energy_prefix: bool = True,
) -> list[Path]:
    output_name = _safe_output_name(output_name)
    if sigma is None:
        sigma = np.full_like(intensity, np.nan)
    if energy is None:
        energy = np.full((intensity.shape[0],), np.nan)
    if intensity.ndim == 1:
        intensity = intensity.reshape(1, -1)
    if sigma.ndim == 1:
        sigma = sigma.reshape(1, -1)
    if energy.ndim == 0:
        energy = energy.reshape(1)
    if not np.any(np.isfinite(sigma)):
        raise RuntimeError(_sigma_failure_message(analysis_h5))
    header_rows = header_rows or [{} for _row in range(intensity.shape[0])]

    output_dir = analysis_h5.parent / XANOS_FOLDER_NAME / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for row, curve in enumerate(intensity, start=1):
        sigma_row = sigma[row - 1] if row - 1 < sigma.shape[0] else np.full_like(curve, np.nan)
        q_row = q[row - 1] if np.asarray(q).ndim > 1 and row - 1 < q.shape[0] else q
        q_row, curve, sigma_row = _finite_curve_rows(q_row, curve, sigma_row)
        if q_row.size == 0:
            continue
        energy_value = float(energy[row - 1]) if row - 1 < energy.size else float("nan")
        header_info = header_rows[row - 1] if row - 1 < len(header_rows) else {}
        cf = _header_float(header_info.get("CF"), 1.0)
        thickness = _header_float(header_info.get("Thickness"), 1.0)
        xrf_bkg = _header_float(header_info.get("xrf_bkg"), 0.0)
        out_path = output_dir / _xanos_curve_filename(output_name, row, intensity.shape[0], use_energy_prefix)
        write_xanos_curve_file(
            out_path,
            q_row,
            curve,
            sigma_row,
            analysis_h5=analysis_h5,
            h5_data_path=h5_data_path,
            output_name=output_name,
            energy_index=row,
            energy_kev=energy_value,
            cf=cf,
            thickness=thickness,
            xrf_bkg=xrf_bkg,
        )
        written.append(out_path)

    list_path = output_dir / XANOS_FILE_LIST_NAME
    with list_path.open("w", encoding="utf-8") as handle:
        for out_path in written:
            handle.write(str(out_path) + "\n")
    written.append(list_path)
    return written


def _xanos_curve_filename(output_name: str, row: int, total_rows: int, use_energy_prefix: bool) -> str:
    if use_energy_prefix:
        return f"energy_{row:03d}_{output_name}_final.dat"
    if total_rows <= 1:
        return f"{output_name}_final.dat"
    return f"{output_name}_{row:03d}_final.dat"


def write_xanos_curve_file(
    path: str | Path,
    q: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray | None,
    *,
    analysis_h5: str | Path,
    h5_data_path: str,
    output_name: str,
    energy_index: int = 1,
    energy_kev: float = np.nan,
    cf: float = 1.0,
    thickness: float = 1.0,
    xrf_bkg: float = 0.0,
    metadata_extra: dict[str, object] | None = None,
) -> Path:
    """Write one curve using the exact header and columns used by XAnoS export."""
    destination = Path(path)
    sigma_values = np.full_like(np.asarray(intensity, dtype=float), np.nan) if sigma is None else sigma
    q_values, intensity_values, sigma_values = _finite_curve_rows(q, intensity, sigma_values)
    if q_values.size == 0:
        raise RuntimeError("Cannot export XAnoS .dat: the curve has no finite q and intensity values.")
    if not np.any(np.isfinite(sigma_values)):
        raise RuntimeError("Cannot export XAnoS .dat: sigma_I contains no finite error values.")

    energy_value = _header_float(energy_kev, np.nan)
    cf_value = _header_float(cf, 1.0)
    thickness_value = _header_float(thickness, 1.0)
    xrf_value = _header_float(xrf_bkg, 0.0)
    metadata = {
        "analysis_h5": str(analysis_h5),
        "h5_data_path": h5_data_path,
        "output_name": _safe_output_name(output_name),
        "energy_index": int(energy_index),
        "energy_kev": energy_value if np.isfinite(energy_value) else None,
        "format": "XAnoS-compatible reduced I-q curve",
        "CF": cf_value,
        "Thickness": thickness_value,
        "xrf_bkg": xrf_value,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    header = "\n".join(
        [
            "Reduced per-energy I-q curve exported from analysis HDF5",
            f"Energy={energy_value:.9f}" if np.isfinite(energy_value) else "Energy=nan",
            f"CF={cf_value:.12g}",
            f"Thickness={thickness_value:.12g}",
            f"xrf_bkg={xrf_value:.12g}",
            "metadata_json=" + json.dumps(metadata, sort_keys=True),
            "col_names=['Q (inv Angs)','Int','Int_err']",
            "columns=q I_final I_final_err",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        destination,
        np.column_stack([q_values, intensity_values, sigma_values]),
        header=header,
        comments="#",
    )
    return destination


def xanos_curve_header_info(
    handle: h5py.File,
    group: h5py.Group,
    row: int | None,
) -> tuple[float, dict[str, float | str | None]]:
    """Return the energy and XAnoS header values for one HDF5 curve row."""
    requested_row = max(0, int(row or 0))
    rows = int(group["I"].shape[0]) if "I" in group and group["I"].ndim > 1 else requested_row + 1
    row_index = min(requested_row, rows - 1)
    header_info = _xanos_header_rows(handle, group, rows)[row_index]
    energy = np.nan
    energy_dataset = group.get("energy")
    if isinstance(energy_dataset, h5py.Dataset):
        values = np.asarray(energy_dataset[()], dtype=float).reshape(-1)
        if row_index < values.size:
            energy = float(values[row_index])
    if not np.isfinite(energy):
        energy = _curve_attr_float(group, "energy_kev", np.nan)
    if not np.isfinite(energy):
        fallback = _fallback_energy_values(handle, rows)
        if row_index < fallback.size:
            energy = float(fallback[row_index])
    return energy, header_info


def _sigma_failure_message(analysis_h5: Path) -> str:
    details = _group_summary_sigma_details(analysis_h5.parent)
    message = (
        f"Cannot export XAnos files for {analysis_h5}: sigma_I contains no finite error values. "
        "No XAnos .dat files were written. The reducer should propagate raw counting uncertainty "
        "for each integrated frame, so this usually means the input/output HDF5 is from an older run "
        "or the raw uncertainty propagation failed."
    )
    if details:
        message += f" {details}"
    return message


def _saxs_xanos_payload(
    handle: h5py.File,
    analysis_h5: Path,
    output_name: str | None = None,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, list[dict[str, float | str | None]]] | None:
    stitched = _stitched_saxs_payload(handle, analysis_h5, output_name=output_name)
    if stitched is not None:
        return stitched
    return _detector_reduction_saxs_payload(handle, analysis_h5, output_name=output_name)


def _stitched_saxs_payload(
    handle: h5py.File,
    analysis_h5: Path,
    output_name: str | None = None,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, list[dict[str, float | str | None]]] | None:
    curves = handle.get("/entry/stitched_averages/curves")
    if curves is None:
        return None
    q_rows: list[np.ndarray] = []
    intensity_rows: list[np.ndarray] = []
    sigma_rows: list[np.ndarray] = []
    energy_rows: list[float] = []
    header_rows: list[dict[str, float | str | None]] = []
    for name in sorted(curves):
        curve = curves[name]
        if not isinstance(curve, h5py.Group) or "q" not in curve or "I" not in curve:
            continue
        q = np.asarray(curve["q"][()], dtype=float)
        intensity = np.asarray(curve["I"][()], dtype=float)
        sigma = np.asarray(curve["sigma_I"][()], dtype=float) if "sigma_I" in curve else np.full_like(intensity, np.nan)
        q, intensity, sigma = _finite_curve_rows(q, intensity, sigma)
        if q.size == 0:
            continue
        q_rows.append(q)
        intensity_rows.append(intensity)
        sigma_rows.append(sigma)
        energy_rows.append(_curve_attr_float(curve, "energy_kev", np.nan))
        header_rows.append({"CF": 1.0, "Thickness": 1.0, "xrf_bkg": 0.0})
    if not q_rows:
        return None
    output_name = output_name or _analysis_output_name(analysis_h5)
    return (
        output_name,
        _stack_rows(q_rows),
        _stack_rows(intensity_rows),
        _stack_rows(sigma_rows),
        np.asarray(energy_rows, dtype=float),
        "/entry/stitched_averages/curves",
        header_rows,
    )


def _detector_reduction_saxs_payload(
    handle: h5py.File,
    analysis_h5: Path,
    output_name: str | None = None,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, list[dict[str, float | str | None]]] | None:
    group = handle.get("/entry/process_01_reduction/data")
    if group is None or "q" not in group or "I" not in group:
        return None
    intensity = np.asarray(group["I"][()], dtype=float)
    q = np.asarray(group["q"][()], dtype=float)
    sigma = np.asarray(group["sigma_I"][()], dtype=float) if "sigma_I" in group else np.full_like(intensity, np.nan)
    rows = int(intensity.shape[0]) if intensity.ndim > 1 else 1
    energy = np.asarray(group["energy"][()], dtype=float) if "energy" in group else np.full((rows,), np.nan)
    return (
        output_name or _analysis_output_name(analysis_h5),
        q,
        intensity,
        sigma,
        energy,
        "/entry/process_01_reduction/data",
        [{"CF": 1.0, "Thickness": 1.0, "xrf_bkg": 0.0} for _row in range(rows)],
    )


def _analysis_output_name(analysis_h5: Path) -> str:
    name = analysis_h5.stem
    if name.endswith("_analysis"):
        name = name[: -len("_analysis")]
    return name or "sample"


def _curve_attr_float(group: h5py.Group, name: str, default: float) -> float:
    try:
        value = float(group.attrs.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _stack_rows(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=float)
    min_width = min(row.size for row in rows)
    if min_width <= 0:
        return np.empty((len(rows), 0), dtype=float)
    return np.vstack([np.asarray(row, dtype=float).reshape(-1)[:min_width] for row in rows])


def _group_summary_sigma_details(output_dir: Path) -> str:
    summaries: list[str] = []
    for detector in ("Pil300K", "Eig1M"):
        path = output_dir / detector / "group_summary.csv"
        if not path.exists():
            continue
        kept_counts: list[int] = []
        frame_counts: list[int] = []
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    kept_counts.append(int(float(row.get("kept_count", "0") or 0)))
                    frame_counts.append(int(float(row.get("frame_count", "0") or 0)))
        except (OSError, ValueError):
            continue
        if not kept_counts:
            continue
        low_kept = sum(1 for value in kept_counts if value < 2)
        summaries.append(
            f"{detector} group_summary: frame_count {min(frame_counts)}-{max(frame_counts)}, "
            f"kept_count {min(kept_counts)}-{max(kept_counts)}, {low_kept}/{len(kept_counts)} groups kept fewer than 2 frames"
        )
    return "; ".join(summaries)


def _xanos_header_rows(handle: h5py.File, group: h5py.Group, rows: int) -> list[dict[str, float | str | None]]:
    cf = _dataset_or_default(group, "xanos_calibration_factor", rows, np.nan)
    thickness = _dataset_or_default(group, "sample_thickness", rows, np.nan)
    xrf_bkg = _dataset_or_default(group, "fluorescence_background", rows, 0.0)
    if not np.any(np.isfinite(cf)):
        cf = _normalization_factor_dataset(handle, "xanos_calibration_factor", rows, np.nan)
    if not np.any(np.isfinite(thickness)):
        thickness = _normalization_factor_dataset(handle, "sample_thickness", rows, np.nan)
    if not np.any(np.isfinite(thickness)):
        thickness = np.full(rows, _metadata_scalar_float(group.parent.get("metadata"), "sample_thickness", 1.0), dtype=float)
    cf = np.where(np.isfinite(cf), cf, 1.0)
    thickness = np.where(np.isfinite(thickness), thickness, 1.0)
    xrf_bkg = np.where(np.isfinite(xrf_bkg), xrf_bkg, 0.0)
    return [{"CF": float(cf[row]), "Thickness": float(thickness[row]), "xrf_bkg": float(xrf_bkg[row])} for row in range(rows)]


def _dataset_or_default(group: h5py.Group, name: str, rows: int, default: float) -> np.ndarray:
    if name not in group:
        return np.full(rows, default, dtype=float)
    values = np.asarray(group[name][()], dtype=float).reshape(-1)
    return _fit_numeric_length(values, rows, default)


def _normalization_factor_dataset(handle: h5py.File, name: str, rows: int, default: float) -> np.ndarray:
    path = f"/entry/process_03_glassy_carbon_normalization/normalization_factors/{name}"
    if path not in handle:
        return np.full(rows, default, dtype=float)
    return _fit_numeric_length(np.asarray(handle[path][()], dtype=float).reshape(-1), rows, default)


def _metadata_scalar_float(group: h5py.Group | None, name: str, default: float) -> float:
    if group is None or name not in group:
        return default
    try:
        value = group[name][()]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _fit_numeric_length(values: np.ndarray, rows: int, default: float) -> np.ndarray:
    fitted = np.full(rows, default, dtype=float)
    if values.size:
        fitted[: min(rows, values.size)] = values[:rows]
    return fitted


def _header_float(value: float | str | None, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _safe_output_name(name: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return cleaned.strip("._") or "sample"


def _finite_curve_rows(q: np.ndarray, intensity: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=float).reshape(-1)
    intensity = np.asarray(intensity, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    width = min(q.size, intensity.size, sigma.size)
    q = q[:width]
    intensity = intensity[:width]
    sigma = sigma[:width]
    keep = np.isfinite(q) & np.isfinite(intensity)
    return q[keep], intensity[keep], sigma[keep]


def _all_missing_energy(values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=float).reshape(-1)
    return values.size == 0 or not np.any(np.isfinite(values))


def _fallback_energy_values(handle: h5py.File, rows: int) -> np.ndarray:
    """Recover energy values from other analysis branches in older HDF5 files."""
    for path in (
        "/entry/process_03_glassy_carbon_normalization/data/energy",
        "/entry/process_03_glassy_carbon_normalization/normalization_factors/energy",
        "/entry/process_02_background_subtraction/data/energy",
        "/entry/process_02_background_subtraction/subtraction_map/energy",
        "/entry/process_01_reduction/data/energy",
    ):
        if path in handle:
            values = np.asarray(handle[path][()], dtype=float).reshape(-1)
            if np.any(np.isfinite(values)):
                return _fit_energy_length(values, rows)
    curve_values = _stitched_curve_energy_values(handle)
    if np.any(np.isfinite(curve_values)):
        return _fit_energy_length(curve_values, rows)
    return np.full((rows,), np.nan, dtype=float)


def _stitched_curve_energy_values(handle: h5py.File) -> np.ndarray:
    curves = handle.get("/entry/stitched_averages/curves")
    if curves is None:
        return np.asarray([], dtype=float)
    values: list[float] = []
    for name in sorted(curves):
        curve = curves[name]
        try:
            value = float(curve.attrs.get("energy_kev", np.nan))
        except (TypeError, ValueError):
            value = np.nan
        values.append(value)
    return np.asarray(values, dtype=float)


def _fit_energy_length(values: np.ndarray, rows: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size >= rows:
        return values[:rows]
    fitted = np.full((rows,), np.nan, dtype=float)
    fitted[: values.size] = values
    return fitted
