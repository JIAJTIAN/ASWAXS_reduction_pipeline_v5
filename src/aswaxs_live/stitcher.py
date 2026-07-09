"""Live HDF5 stitching helpers for paired detector reductions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from aswaxs_live.core.analysis_h5 import (
    write_background_subtraction_to_analysis_h5,
    write_glassy_carbon_normalization_to_analysis_h5,
)
from aswaxs_live.xanos_export import export_analysis_h5_to_xanos_format


DEFAULT_OVERLAP_Q_MAX = 0.20
EDGE_SCALE_POINTS = 40
MIN_EDGE_SCALE_POINTS = 5


@dataclass
class ReductionRows:
    detector: str
    path: Path
    q: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    energy: np.ndarray
    energy_index: np.ndarray
    group_index: np.ndarray
    mtime_ns: int
    size: int


@dataclass
class StitchedAsaxsSettings:
    """Settings needed to apply ASAXS corrections after detector stitching."""

    num_groups: int
    sample_group: int | None
    air_group: int | None
    empty_group: int | None
    water_group: int | None
    gc_group: int | None
    gc_reference_file: str | None
    gc_q_range: tuple[float, float]
    capillary_thickness: float | None
    gc_thickness: float | None
    subtract_fluorescence: bool
    fluorescence_level: float | None
    fluorescence_reference: str
    fluorescence_q_range: tuple[float, float]
    asaxs_pairs: tuple[str, ...] = ()


def find_analysis_h5(output_dir: Path) -> Path | None:
    candidates = sorted(
        [*output_dir.glob("*_analysis.h5"), *output_dir.glob("*_analysis.hdf5")],
        key=lambda path: path.stat().st_mtime_ns if path.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_detector_analysis_h5s(output_root: Path, detector: str) -> dict[str, Path]:
    """Find detector analysis files keyed by sample name.

    The live sample-list layout stores detector files as
    ``Extracted/<sample>/<detector>/<sample>_<detector>_analysis.h5``.  Older
    single-sample runs may still store files directly inside the detector
    folder, so this scan accepts both layouts.
    """
    if not output_root.exists():
        return {}
    suffixes = (f"_{detector}_analysis.h5", f"_{detector}_analysis.hdf5")
    matches: dict[str, Path] = {}
    for path in output_root.rglob("*_analysis.h5"):
        name = path.name
        lower_name = name.lower()
        if "_corrupt_" in lower_name or "_old_" in lower_name or "should_not_be_used" in lower_name:
            continue
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        sample = name
        for suffix in suffixes:
            if sample.endswith(suffix):
                sample = sample[: -len(suffix)]
                break
        previous = matches.get(sample)
        if previous is None or path.stat().st_mtime_ns >= previous.stat().st_mtime_ns:
            matches[sample] = path
    return matches


def clear_stitched_averages(combined_h5_path: Path) -> None:
    """Remove derived stitched curves from a combined analysis HDF5.

    The stitched branch is rebuilt from the detector analysis H5 files, so it is
    safe to clear at the start of a new live run. This prevents the viewer from
    showing curves produced by an older sample queue.
    """
    if not combined_h5_path.exists():
        return
    with h5py.File(combined_h5_path, "a") as handle:
        if "/entry/stitched_averages" in handle:
            del handle["/entry/stitched_averages"]
        handle.flush()


def write_stitched_asaxs_outputs(combined_h5_path: Path, settings: StitchedAsaxsSettings) -> bool:
    """Apply ASAXS/background/GC correction to already stitched group curves.

    The paired detector reducers intentionally stop at 1D/group-average output.
    This function is the stitched-level correction step: SAXS and WAXS are
    stitched first, then GC/background normalization is applied to the stitched
    I(q) rows in the combined analysis HDF5.
    """
    averages = read_stitched_group_averages(combined_h5_path, settings.num_groups)
    if not averages or (settings.sample_group is None and not settings.asaxs_pairs):
        return False

    # Local import avoids making the live reducer depend on the stitcher.
    from aswaxs_live.reducer import build_final_outputs_for_h5, load_reduction_core  # pylint: disable=import-outside-toplevel

    args = SimpleNamespace(
        sample_group=settings.sample_group,
        air_group=settings.air_group,
        empty_group=settings.empty_group,
        water_group=settings.water_group,
        gc_group=settings.gc_group,
        gc_reference_file=settings.gc_reference_file,
        gc_q_range=list(settings.gc_q_range),
        capillary_thickness=settings.capillary_thickness,
        gc_thickness=settings.gc_thickness,
        subtract_fluorescence=settings.subtract_fluorescence,
        fluorescence_level=settings.fluorescence_level,
        fluorescence_reference=settings.fluorescence_reference,
        fluorescence_q_range=list(settings.fluorescence_q_range),
        asaxs_pair=list(settings.asaxs_pairs),
        asaxs_extraction_plan=None,
        asaxs_output_name="sample",
        write_text_output=False,
    )
    core = load_reduction_core()
    final_outputs = build_final_outputs_for_h5(core, averages, args, combined_h5_path.parent)
    if not final_outputs:
        return False
    _replace_stitched_asaxs_outputs(combined_h5_path)
    _write_stitched_process_outputs(combined_h5_path, final_outputs, args)
    return True


def read_stitched_group_averages(combined_h5_path: Path, num_groups: int) -> list[object]:
    """Convert stitched HDF5 curves into reducer-compatible GroupAverage rows."""
    from aswaxs_live.core.reduce_aswaxs_sequence import GroupAverage  # pylint: disable=import-outside-toplevel

    if num_groups < 1 or not combined_h5_path.exists():
        return []
    rows: list[object] = []
    with h5py.File(combined_h5_path, "r") as handle:
        curves = handle.get("/entry/stitched_averages/curves")
        if curves is None:
            return []
        for index, name in enumerate(sorted(curves), start=1):
            curve = curves[name]
            if "q" not in curve or "I" not in curve:
                continue
            q = np.asarray(curve["q"][()], dtype=float)
            intensity = np.asarray(curve["I"][()], dtype=float)
            sigma = np.asarray(curve["sigma_I"][()], dtype=float) if "sigma_I" in curve else np.full_like(intensity, np.nan)
            row_index = int(curve.attrs.get("row_index", index))
            zero_based = max(0, row_index - 1)
            energy_index = int(curve.attrs.get("energy_index", zero_based // num_groups + 1))
            group_index = int(curve.attrs.get("group_index", zero_based % num_groups + 1))
            energy_kev = _optional_float_attr(curve.attrs.get("energy_kev"))
            if energy_kev is None and "energy" in curve:
                energy_values = np.asarray(curve["energy"][()], dtype=float).reshape(-1)
                energy_kev = float(energy_values[0]) if energy_values.size and np.isfinite(energy_values[0]) else None
            rows.append(
                GroupAverage(
                    energy_index=energy_index,
                    group_index=group_index,
                    q=q,
                    energy_kev=energy_kev,
                    avg_intensity=intensity,
                    avg_error=sigma,
                    frame_count=1,
                    kept_count=1,
                    dropped_count=0,
                    kept_sequence_indices=[row_index],
                    dropped_sequence_indices=[],
                    avg_total_intensity=float(np.trapz(intensity[np.isfinite(intensity)], q[np.isfinite(intensity)]))
                    if np.count_nonzero(np.isfinite(q) & np.isfinite(intensity)) >= 2
                    else np.nan,
                    avg_monitor_value=1.0,
                )
            )
    return sorted(rows, key=lambda item: (item.energy_index, item.group_index))


def read_detector_group_averages(detector_h5_path: Path, detector: str) -> list[object]:
    """Convert one detector analysis HDF5 into reducer-compatible GroupAverage rows."""
    from aswaxs_live.core.reduce_aswaxs_sequence import GroupAverage  # pylint: disable=import-outside-toplevel

    rows = read_reduction_rows(detector_h5_path, detector)
    if rows is None:
        return []
    averages: list[object] = []
    for row in range(rows.intensity.shape[0]):
        q = q_for_reduction_row(rows.q, row)
        intensity = np.asarray(rows.intensity[row], dtype=float)
        sigma = np.asarray(rows.sigma[row], dtype=float)
        finite = np.isfinite(q) & np.isfinite(intensity)
        averages.append(
            GroupAverage(
                energy_index=_row_int(rows.energy_index, row, 1),
                group_index=_row_int(rows.group_index, row, 1),
                q=q,
                energy_kev=_row_float(rows.energy, row, np.nan),
                avg_intensity=intensity,
                avg_error=sigma,
                frame_count=1,
                kept_count=1,
                dropped_count=0,
                kept_sequence_indices=[row + 1],
                dropped_sequence_indices=[],
                avg_total_intensity=float(np.trapz(intensity[finite], q[finite])) if np.count_nonzero(finite) >= 2 else np.nan,
                avg_monitor_value=1.0,
            )
        )
    return sorted(averages, key=lambda item: (item.energy_index, item.group_index))


def _replace_stitched_asaxs_outputs(combined_h5_path: Path) -> None:
    """Remove old stitched-level correction branches before rewriting them."""
    with h5py.File(combined_h5_path, "a") as handle:
        entry = handle.require_group("entry")
        for name in list(entry):
            if name.startswith("process_02_background_subtraction") or name.startswith("process_03_glassy_carbon_normalization"):
                del entry[name]
        if "final" in entry:
            del entry["final"]
        if "asaxs_outputs" in entry:
            del entry["asaxs_outputs"]


def _write_stitched_process_outputs(combined_h5_path: Path, final_outputs: list[object], args: SimpleNamespace) -> None:
    from aswaxs_live.core.reduce_aswaxs_sequence import (  # pylint: disable=import-outside-toplevel
        _write_named_asaxs_outputs,
        stack_ragged_rows,
    )

    _write_named_asaxs_outputs(combined_h5_path, final_outputs)
    primary_name = getattr(final_outputs[0], "output_name", "sample")
    primary_outputs = [item for item in final_outputs if getattr(item, "output_name", "sample") == primary_name]
    final_q = stack_ragged_rows([item.q for item in primary_outputs])
    energies = np.asarray([np.nan if item.energy_kev is None else item.energy_kev for item in primary_outputs], dtype=float)
    final_i = stack_ragged_rows([item.I for item in primary_outputs])
    final_sigma = stack_ragged_rows([item.sigma_I for item in primary_outputs])
    component_lookup = [_component_dict(item.component_names, item.component_columns) for item in primary_outputs]
    corrected_data = {
        "q": final_q,
        "energy": energies,
        "I_sample_corrected": final_i,
        "sigma_sample_corrected": final_sigma,
        "I_gc_corrected": _stack_component(component_lookup, "I_gc_minus_air"),
        "sigma_gc_corrected": _stack_component(component_lookup, "I_gc_minus_air_err"),
    }
    subtraction_metadata = {
        "input_h5_file": str(combined_h5_path),
        "input_data_path": "/entry/stitched_averages/curves",
        "output_h5_file": str(combined_h5_path),
        "output_data_path": "/entry/process_02_background_subtraction/data",
        "notes": "stitched detector curves corrected after SAXS/WAXS stitching",
    }
    subtraction_parameters = {
        "gc_background": "air" if args.air_group is not None else "unknown",
        "sample_background": "empty_cell/solvent" if args.empty_group or args.water_group else "unknown",
        "scale_by_I0": True,
        "scale_by_transmission": False,
        "scale_by_exposure_time": False,
        "subtraction_formula": "stitched sample-empty; stitched water-empty; stitched gc-air",
        "solvent_scale_factor": 1.0,
        "empty_cell_scale_factor": 1.0,
    }
    subtraction_map = {
        "energy": energies,
        "sample_id": primary_outputs[0].metadata.get("sample_group", args.sample_group),
        "air_id": args.air_group if args.air_group is not None else "unknown",
        "glassy_carbon_id": args.gc_group if args.gc_group is not None else "unknown",
        "empty_cell_id": args.empty_group if args.empty_group is not None else "unknown",
        "solvent_id": primary_outputs[0].metadata.get("water_group", args.water_group),
        "output_name": primary_name,
    }
    write_background_subtraction_to_analysis_h5(combined_h5_path, corrected_data, subtraction_metadata, subtraction_parameters, subtraction_map)
    if args.gc_group is not None:
        normalized_data = {
            "q": final_q,
            "energy": energies,
            "I_sample_normalized": final_i,
            "sigma_sample_normalized": final_sigma,
            "I_gc_normalized": _stack_component(component_lookup, "I_gc_minus_air"),
            "sigma_gc_normalized": _stack_component(component_lookup, "I_gc_minus_air_err"),
        }
        normalization_parameters = {
            "gc_reference_file": str(args.gc_reference_file) if args.gc_reference_file else "NIST_SRM3600_builtin",
            "gc_reference_file_hash": "unknown",
            "reference_units": "differential_scattering_cross_section",
            "q_range_requested": list(args.gc_q_range),
            "q_range_used": "per-energy in /normalization_factors/q_min_used and q_max_used",
            "scale_method": "XAnoS_CF_from_integrated_area_ratio_after_stitching_recorded_not_applied",
            "absolute_scale": False,
            "uncertainty_propagation": "GC scale uncertainty not propagated because CF is recorded but not applied",
        }
        normalization_factors = {
            "energy": energies,
            "scale_factor": np.asarray([item.metadata.get("absolute_scale_factor", np.nan) for item in primary_outputs], dtype=float),
            "xanos_calibration_factor": np.asarray([item.metadata.get("xanos_calibration_factor", np.nan) for item in primary_outputs], dtype=float),
            "gc_reference_over_measured_factor": np.asarray([item.metadata.get("gc_reference_over_measured_factor", np.nan) for item in primary_outputs], dtype=float),
            "sample_thickness": np.asarray([item.metadata.get("sample_thickness", np.nan) for item in primary_outputs], dtype=float),
            "gc_standard_thickness": np.asarray([item.metadata.get("gc_standard_thickness", np.nan) for item in primary_outputs], dtype=float),
            "scale_uncertainty": np.full_like(energies, np.nan),
            "q_min_used": np.asarray([item.metadata.get("gc_q_min_used", np.nan) for item in primary_outputs], dtype=float),
            "q_max_used": np.asarray([item.metadata.get("gc_q_max_used", np.nan) for item in primary_outputs], dtype=float),
            "scale_status": [item.metadata.get("gc_scale_status", "unknown") for item in primary_outputs],
            "scale_factor_basis": "scale_factor = XAnoS_CF / sample_thickness recorded for downstream software; exported final I is not scaled",
        }
        normalization_metadata = {
            "input_h5_file": str(combined_h5_path),
            "input_data_path": "/entry/process_02_background_subtraction/data",
            "output_h5_file": str(combined_h5_path),
            "output_data_path": "/entry/process_03_glassy_carbon_normalization/data",
            "notes": "XAnoS CF recorded from stitched GC curve; final exported I is not CF/thickness scaled",
        }
        write_glassy_carbon_normalization_to_analysis_h5(combined_h5_path, normalized_data, normalization_metadata, normalization_parameters, normalization_factors)
    _write_stitched_final_group(combined_h5_path, final_q, energies, final_i, final_sigma)
    written = export_analysis_h5_to_xanos_format(combined_h5_path)
    dat_files = [path for path in written if path.suffix.lower() == ".dat"]
    if not dat_files:
        raise RuntimeError(f"XAnos export wrote no .dat files for {combined_h5_path}")


def _write_stitched_final_group(combined_h5_path: Path, q: np.ndarray, energies: np.ndarray, intensity: np.ndarray, sigma: np.ndarray) -> None:
    with h5py.File(combined_h5_path, "a") as handle:
        entry = handle.require_group("entry")
        final = entry.require_group("final")
        if "corrected_I_q_E" in final:
            del final["corrected_I_q_E"]
        group = final.create_group("corrected_I_q_E")
        group.attrs["NX_class"] = "NXdata"
        group.attrs["signal"] = "I"
        group.attrs["axes"] = np.asarray(["energy", "q"], dtype=h5py.string_dtype("utf-8"))
        group.create_dataset("q", data=q)
        group.create_dataset("energy", data=energies)
        group.create_dataset("I", data=intensity)
        group.create_dataset("sigma_I", data=sigma)


def _component_dict(names: list[str], columns: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(column) for name, column in zip(names, columns)}


def _stack_component(component_lookup: list[dict[str, np.ndarray]], name: str) -> np.ndarray:
    from aswaxs_live.core.reduce_aswaxs_sequence import stack_ragged_rows  # pylint: disable=import-outside-toplevel

    if not component_lookup:
        return np.empty((0, 0))
    first_q = next(iter(component_lookup[0].values()))
    values = [item.get(name, np.full_like(first_q, np.nan, dtype=float)) for item in component_lookup]
    return stack_ragged_rows(values)


def update_live_stitched_averages(
    pil300k_output_dir: Path,
    eig1m_output_dir: Path,
    combined_h5_path: Path | None = None,
    overlap_q_max: float = DEFAULT_OVERLAP_Q_MAX,
    sample_names: list[str] | None = None,
    min_mtime_ns: int | None = None,
) -> Path | None:
    """Copy detector analysis records and stitch matching rows.

    The two detector reducers keep writing their own private HDF5 files during a
    live run. This coordinator function is the single writer for the combined
    batch file, so the public analysis HDF5 stays organized as one record:
    ``/entry/Pil300K``, ``/entry/Eig1M``, and ``/entry/stitched_averages``.
    """
    pairs = paired_detector_analysis_h5s(
        pil300k_output_dir,
        eig1m_output_dir,
        sample_names=sample_names,
        min_mtime_ns=min_mtime_ns,
    )
    if not pairs:
        return None

    target_h5 = combined_h5_path or pairs[-1][1]
    if target_h5.exists() and not _has_pending_stitched_rows(target_h5, pairs):
        return None
    target_h5.parent.mkdir(parents=True, exist_ok=True)
    wrote_any = False
    try:
        with h5py.File(target_h5, "a") as handle:
            entry = handle.require_group("entry")
            group = entry.require_group("stitched_averages")
            group.attrs["NX_class"] = "NXprocess"
            group.attrs["process_stage"] = "detector_stitching"
            group.attrs["overlap_q_max"] = float(overlap_q_max)
            curves = group.require_group("curves")
            curves.attrs["NX_class"] = "NXdata"
            curves.attrs["signal"] = "I"
            curves.attrs["axes"] = "q"
            for sample_name, pil300k_h5, eig1m_h5 in pairs:
                try:
                    pil300k = read_reduction_rows(pil300k_h5, "Pil300K")
                    eig1m = read_reduction_rows(eig1m_h5, "Eig1M")
                except (OSError, RuntimeError):
                    continue
                if pil300k is None or eig1m is None:
                    continue
                low_q_rows, high_q_rows = choose_stitch_order(pil300k, eig1m)
                n_rows = min(low_q_rows.intensity.shape[0], high_q_rows.intensity.shape[0])
                if n_rows <= 0:
                    continue
                for row in range(n_rows):
                    name = f"{sanitize_h5_name(sample_name)}_curve_{row + 1:03d}"
                    if name in curves and _stitched_curve_is_current(curves[name], low_q_rows, high_q_rows, row):
                        continue
                    try:
                        stitched, source_detector, scale, q_min, q_max, join_q, n_overlap, low_q_points, high_q_points = stitch_one_row(
                            low_q_rows,
                            high_q_rows,
                            row,
                            overlap_q_max,
                        )
                    except ValueError:
                        continue
                    if name in curves:
                        del curves[name]
                    curve = curves.create_group(name)
                    curve.create_dataset("q", data=stitched[:, 0])
                    curve.create_dataset("I", data=stitched[:, 1])
                    curve.create_dataset("sigma_I", data=stitched[:, 2])
                    curve.create_dataset("source_detector", data=source_detector.astype(h5py.string_dtype("utf-8")))
                    curve.attrs["NX_class"] = "NXdata"
                    curve.attrs["signal"] = "I"
                    curve.attrs["axes"] = "q"
                    curve.attrs["sample_name"] = sample_name
                    curve.attrs["row_index"] = row + 1
                    curve.attrs["energy_kev"] = _row_float(low_q_rows.energy, row, _row_float(high_q_rows.energy, row, np.nan))
                    curve.attrs["energy_index"] = _row_int(low_q_rows.energy_index, row, row + 1)
                    curve.attrs["group_index"] = _row_int(low_q_rows.group_index, row, row + 1)
                    curve.attrs["low_q_detector"] = low_q_rows.detector
                    curve.attrs["high_q_detector"] = high_q_rows.detector
                    curve.attrs["low_q_analysis_h5"] = str(low_q_rows.path)
                    curve.attrs["high_q_analysis_h5"] = str(high_q_rows.path)
                    curve.attrs["low_q_row_index"] = int(row)
                    curve.attrs["high_q_row_index"] = int(row)
                    curve.attrs["high_q_scale_factor"] = float(scale)
                    curve.attrs["overlap_q_min"] = float(q_min)
                    curve.attrs["overlap_q_max"] = float(q_max)
                    curve.attrs["join_q"] = float(join_q)
                    curve.attrs["n_overlap_points"] = int(n_overlap)
                    curve.attrs["scale_method"] = "gap_edge_extrapolation" if n_overlap == 0 else "overlap_median_ratio"
                    curve.attrs["low_q_points"] = int(low_q_points)
                    curve.attrs["high_q_points"] = int(high_q_points)
                    source_low_q = q_for_reduction_row(low_q_rows.q, row)
                    source_high_q = q_for_reduction_row(high_q_rows.q, row)
                    curve.attrs["source_low_q_points"] = int(source_low_q.size)
                    curve.attrs["source_high_q_points"] = int(source_high_q.size)
                    curve.attrs["source_low_q_min"] = float(np.nanmin(source_low_q))
                    curve.attrs["source_low_q_max"] = float(np.nanmax(source_low_q))
                    curve.attrs["source_high_q_min"] = float(np.nanmin(source_high_q))
                    curve.attrs["source_high_q_max"] = float(np.nanmax(source_high_q))
                    wrote_any = True
            handle.flush()
    except (OSError, RuntimeError):
        return None
    return target_h5 if wrote_any else None


def _stitched_curve_is_current(curve: h5py.Group, low_q_rows: ReductionRows, high_q_rows: ReductionRows, row: int) -> bool:
    """Return True when an existing stitched row already represents this pair.

    HDF5 files do not automatically shrink when datasets are deleted and
    recreated. The live GUI calls the stitcher periodically, so unchanged
    derived rows must be left in place instead of being rewritten every tick.
    """
    try:
        if (
            "q" not in curve
            or "I" not in curve
            or "sigma_I" not in curve
            or str(curve.attrs.get("low_q_analysis_h5", "")) != str(low_q_rows.path)
            or str(curve.attrs.get("high_q_analysis_h5", "")) != str(high_q_rows.path)
            or int(curve.attrs.get("low_q_row_index", -1)) != row
            or int(curve.attrs.get("high_q_row_index", -1)) != row
            or "scale_method" not in curve.attrs
        ):
            return False
        low_q_axis = q_for_reduction_row(low_q_rows.q, row)
        high_q_axis = q_for_reduction_row(high_q_rows.q, row)
        return (
            int(curve.attrs.get("source_low_q_points", -1)) == low_q_axis.size
            and int(curve.attrs.get("source_high_q_points", -1)) == high_q_axis.size
            and np.isclose(float(curve.attrs.get("source_low_q_min", np.nan)), float(np.nanmin(low_q_axis)))
            and np.isclose(float(curve.attrs.get("source_low_q_max", np.nan)), float(np.nanmax(low_q_axis)))
            and np.isclose(float(curve.attrs.get("source_high_q_min", np.nan)), float(np.nanmin(high_q_axis)))
            and np.isclose(float(curve.attrs.get("source_high_q_max", np.nan)), float(np.nanmax(high_q_axis)))
        )
    except (TypeError, ValueError):
        return False


def _has_pending_stitched_rows(target_h5: Path, pairs: list[tuple[str, Path, Path]]) -> bool:
    """Check read-only whether the combined file needs new stitched rows."""
    try:
        with h5py.File(target_h5, "r") as handle:
            curves = handle.get("/entry/stitched_averages/curves")
            if curves is None:
                return True
            for sample_name, pil300k_h5, eig1m_h5 in pairs:
                pil300k = read_reduction_rows(pil300k_h5, "Pil300K")
                eig1m = read_reduction_rows(eig1m_h5, "Eig1M")
                if pil300k is None or eig1m is None:
                    continue
                low_q_rows, high_q_rows = choose_stitch_order(pil300k, eig1m)
                n_rows = min(low_q_rows.intensity.shape[0], high_q_rows.intensity.shape[0])
                for row in range(n_rows):
                    name = f"{sanitize_h5_name(sample_name)}_curve_{row + 1:03d}"
                    if name not in curves or not _stitched_curve_is_current(curves[name], low_q_rows, high_q_rows, row):
                        return True
    except (OSError, RuntimeError):
        return True
    return False


def paired_detector_analysis_h5s(
    pil300k_output_dir: Path,
    eig1m_output_dir: Path,
    sample_names: list[str] | None = None,
    min_mtime_ns: int | None = None,
) -> list[tuple[str, Path, Path]]:
    pil300k_by_sample = find_detector_analysis_h5s(pil300k_output_dir, "Pil300K")
    eig1m_by_sample = find_detector_analysis_h5s(eig1m_output_dir, "Eig1M")
    shared_samples = sorted(set(pil300k_by_sample) & set(eig1m_by_sample))
    if sample_names is not None:
        allowed = {sanitize_h5_name(sample) for sample in sample_names}
        shared_samples = [sample for sample in shared_samples if sanitize_h5_name(sample) in allowed]
    if min_mtime_ns is not None:
        shared_samples = [
            sample
            for sample in shared_samples
            if pil300k_by_sample[sample].stat().st_mtime_ns >= min_mtime_ns
            and eig1m_by_sample[sample].stat().st_mtime_ns >= min_mtime_ns
        ]
    if shared_samples:
        return [(sample, pil300k_by_sample[sample], eig1m_by_sample[sample]) for sample in shared_samples]
    if sample_names is not None:
        return []

    pil300k_h5 = find_analysis_h5(pil300k_output_dir)
    eig1m_h5 = find_analysis_h5(eig1m_output_dir)
    if pil300k_h5 is None or eig1m_h5 is None:
        return []
    return [(sample_name_from_detector_h5(pil300k_h5, "Pil300K"), pil300k_h5, eig1m_h5)]


def sample_name_from_detector_h5(path: Path, detector: str) -> str:
    suffix = f"_{detector}_analysis"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def sanitize_h5_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._")
    return safe or "sample"


def _copy_detector_entry(source_h5: Path, combined_entry: h5py.Group, detector: str) -> None:
    """Replace one detector branch in the combined analysis file.

    Copying the detector branch keeps the combined file self-contained while
    avoiding concurrent writes from the two live reducer processes.
    """
    if detector in combined_entry:
        del combined_entry[detector]
    detector_group = combined_entry.create_group(detector)
    detector_group.attrs["NX_class"] = "NXcollection"
    detector_group.attrs["detector"] = detector
    detector_group.attrs["source_analysis_h5"] = str(source_h5)
    with h5py.File(source_h5, "r") as source:
        if "entry" not in source:
            return
        for name in source["entry"]:
            if name == "stitched_averages":
                continue
            source.copy(source["entry"][name], detector_group, name=name)


def read_reduction_rows(path: Path, detector: str) -> ReductionRows | None:
    if not path.exists():
        return None
    stat = path.stat()
    with h5py.File(path, "r") as handle:
        data_path = latest_reduction_data_path(handle)
        if data_path is None:
            return None
        if f"{data_path}/q" not in handle or f"{data_path}/I" not in handle:
            return None
        q = np.asarray(handle[f"{data_path}/q"][()], dtype=float)
        intensity = np.asarray(handle[f"{data_path}/I"][()], dtype=float)
        sigma = np.asarray(handle[f"{data_path}/sigma_I"][()], dtype=float) if f"{data_path}/sigma_I" in handle else np.full_like(intensity, np.nan)
        energy_path = f"{data_path}/energy"
        energy_index_path = f"{data_path}/energy_index"
        group_index_path = f"{data_path}/group_index"
    if intensity.ndim == 1:
        intensity = intensity.reshape(1, -1)
    if sigma.ndim == 1:
        sigma = sigma.reshape(1, -1)
    with h5py.File(path, "r") as handle:
        n_rows = int(intensity.shape[0])
        energy = _read_1d_or_default(handle, energy_path, n_rows, np.nan, float)
        energy_index = _read_1d_or_default(handle, energy_index_path, n_rows, 0, int)
        group_index = _read_1d_or_default(handle, group_index_path, n_rows, 0, int)
    return ReductionRows(
        detector=detector,
        path=path,
        q=q,
        intensity=intensity,
        sigma=sigma,
        energy=energy,
        energy_index=energy_index,
        group_index=group_index,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )


def latest_reduction_data_path(handle: h5py.File) -> str | None:
    """Return the newest process_01_reduction data path in an analysis HDF5."""
    if "entry" not in handle:
        return None
    entry = handle["entry"]
    names = [name for name in entry if name == "process_01_reduction" or name.startswith("process_01_reduction_v")]
    if not names:
        return None

    def version_key(name: str) -> int:
        if name == "process_01_reduction":
            return 1
        try:
            return int(name.rsplit("_v", 1)[1])
        except (IndexError, ValueError):
            return 1

    return f"/entry/{sorted(names, key=version_key)[-1]}/data"


def _read_1d_or_default(handle: h5py.File, dataset_path: str, rows: int, default: float | int, dtype) -> np.ndarray:
    """Read a per-row dataset, padding older analysis files with defaults."""
    if dataset_path not in handle:
        return np.full(rows, default, dtype=dtype)
    values = np.asarray(handle[dataset_path][()], dtype=dtype).reshape(-1)
    if values.size >= rows:
        return values[:rows]
    padded = np.full(rows, default, dtype=dtype)
    padded[: values.size] = values
    return padded


def _row_float(values: np.ndarray, row: int, default: float) -> float:
    try:
        value = float(values[row])
    except (IndexError, TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _row_int(values: np.ndarray, row: int, default: int) -> int:
    try:
        value = int(values[row])
    except (IndexError, TypeError, ValueError):
        return default
    return value if value > 0 else default


def _optional_float_attr(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def choose_stitch_order(first: ReductionRows, second: ReductionRows) -> tuple[ReductionRows, ReductionRows]:
    """Choose low-q and high-q detector roles from the actual q coverage."""
    first_min = float(np.nanmin(first.q))
    second_min = float(np.nanmin(second.q))
    if not np.isclose(first_min, second_min):
        return (first, second) if first_min < second_min else (second, first)
    first_max = float(np.nanmax(first.q))
    second_max = float(np.nanmax(second.q))
    return (first, second) if first_max <= second_max else (second, first)


def stitch_one_row(low_q: ReductionRows, high_q: ReductionRows, row: int, overlap_q_max: float) -> tuple[np.ndarray, np.ndarray, float, float, float, float, int, int, int]:
    low_q_axis = q_for_reduction_row(low_q.q, row)
    high_q_axis = q_for_reduction_row(high_q.q, row)
    low_q_data = np.column_stack([low_q_axis, low_q.intensity[row], low_q.sigma[row]])
    high_q_data = np.column_stack([high_q_axis, high_q.intensity[row], high_q.sigma[row]])
    scale, q_min, q_max, n_overlap = scale_high_q_to_low_q(low_q_data, high_q_data, overlap_q_max)
    high_q_scaled = high_q_data.copy()
    high_q_scaled[:, 1:] *= scale
    join_q = 0.5 * (q_min + q_max)
    low_q_part = low_q_data[low_q_data[:, 0] <= join_q]
    high_q_part = high_q_scaled[high_q_scaled[:, 0] > join_q]
    if low_q_part.size == 0 or high_q_part.size == 0:
        raise ValueError("Stitch split removed one detector contribution.")
    stitched = np.vstack([low_q_part, high_q_part])
    source_detector = np.concatenate(
        [
            np.full(low_q_part.shape[0], low_q.detector, dtype=object),
            np.full(high_q_part.shape[0], high_q.detector, dtype=object),
        ]
    )
    order = np.argsort(stitched[:, 0])
    return stitched[order], source_detector[order], scale, q_min, q_max, join_q, n_overlap, low_q_part.shape[0], high_q_part.shape[0]


def q_for_reduction_row(q: np.ndarray, row: int) -> np.ndarray:
    """Return the q grid for one reduction row.

    PyFAI q changes with wavelength, so ASAXS energy rows may have distinct q
    grids. Older files may still store q as one shared 1D axis; newer reduction
    records can store q as row x q_bin.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        return q
    if q.ndim != 2:
        raise ValueError(f"Expected q to be 1D or 2D; found shape {q.shape}.")
    if row < q.shape[0]:
        return q[row]
    return q[0]


def scale_high_q_to_low_q(low_q: np.ndarray, high_q: np.ndarray, overlap_q_max: float) -> tuple[float, float, float, int]:
    q_low = max(float(np.nanmin(low_q[:, 0])), float(np.nanmin(high_q[:, 0])))
    detector_overlap_q_high = min(float(np.nanmax(low_q[:, 0])), float(np.nanmax(high_q[:, 0])))
    q_high = min(detector_overlap_q_high, float(overlap_q_max))
    if q_high < q_low:
        q_high = detector_overlap_q_high
    overlap = high_q[(high_q[:, 0] >= q_low) & (high_q[:, 0] <= q_high) & (high_q[:, 1] > 0)]
    if overlap.shape[0] >= 3:
        low_q_positive = low_q[(low_q[:, 0] > 0) & (low_q[:, 1] > 0)]
        low_q_at_high_q = np.exp(np.interp(np.log(overlap[:, 0]), np.log(low_q_positive[:, 0]), np.log(low_q_positive[:, 1])))
        ratios = low_q_at_high_q / overlap[:, 1]
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        if ratios.size >= 3:
            return float(np.median(ratios)), float(np.nanmin(overlap[:, 0])), float(np.nanmax(overlap[:, 0])), int(ratios.size)
    return estimate_gap_scale_high_q_to_low_q(low_q, high_q)


def estimate_gap_scale_high_q_to_low_q(low_q: np.ndarray, high_q: np.ndarray) -> tuple[float, float, float, int]:
    """Estimate detector scale when q coverage has a gap instead of overlap.

    The estimate uses the high-q edge of the low-q detector and the low-q edge of
    the high-q detector.  Each edge is fit as log(I) = a + b log(q), then both
    fits are evaluated at the geometric midpoint of the gap.
    """
    low_edge = _positive_sorted_curve(low_q)
    high_edge = _positive_sorted_curve(high_q)
    if low_edge.shape[0] < MIN_EDGE_SCALE_POINTS or high_edge.shape[0] < MIN_EDGE_SCALE_POINTS:
        raise ValueError("Too few positive edge points for gap scaling.")
    low_q_max = float(np.nanmax(low_edge[:, 0]))
    high_q_min = float(np.nanmin(high_edge[:, 0]))
    if not np.isfinite(low_q_max) or not np.isfinite(high_q_min) or low_q_max <= 0 or high_q_min <= 0:
        raise ValueError("Invalid q edges for gap scaling.")
    if low_q_max >= high_q_min:
        raise ValueError("Too few valid overlap ratios for detector stitching.")
    q_join = float(np.sqrt(low_q_max * high_q_min))
    low_window = low_edge[-min(EDGE_SCALE_POINTS, low_edge.shape[0]) :]
    high_window = high_edge[: min(EDGE_SCALE_POINTS, high_edge.shape[0])]
    low_estimate = _loglog_edge_estimate(low_window, q_join)
    high_estimate = _loglog_edge_estimate(high_window, q_join)
    if not np.isfinite(low_estimate) or not np.isfinite(high_estimate) or high_estimate <= 0:
        raise ValueError("Could not estimate detector scale across q gap.")
    return float(low_estimate / high_estimate), low_q_max, high_q_min, 0


def _positive_sorted_curve(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=float)
    mask = np.isfinite(curve[:, 0]) & np.isfinite(curve[:, 1]) & (curve[:, 0] > 0) & (curve[:, 1] > 0)
    positive = curve[mask]
    if positive.size == 0:
        return positive.reshape(0, curve.shape[1])
    return positive[np.argsort(positive[:, 0])]


def _loglog_edge_estimate(window: np.ndarray, q_value: float) -> float:
    if window.shape[0] < MIN_EDGE_SCALE_POINTS:
        raise ValueError("Too few edge points for log-log fit.")
    x = np.log(window[:, 0])
    y = np.log(window[:, 1])
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("Invalid edge points for log-log fit.")
    if np.nanmax(x) - np.nanmin(x) <= 1e-12:
        return float(np.exp(np.nanmedian(y)))
    slope, intercept = np.polyfit(x, y, 1)
    estimate = np.exp(intercept + slope * np.log(q_value))
    return float(estimate)
