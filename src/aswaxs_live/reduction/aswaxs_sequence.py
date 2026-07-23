"""Reduce a structured ASAXS energy/group/frame HDF5 sequence.

This is the main multi-frame workflow: it maps source files into a manifest,
integrates each frame, rejects bad repeats, monitor-normalizes, averages by
energy/group, applies background/GC corrections, and writes analysis HDF5
provenance without modifying the source HDF5 files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import os
import queue
import re
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from pyFAI import load as load_poni

from aswaxs_live.reduction.frame_qc import (
    analyze_frame_series,
    cleanup_frame_stability_shards,
    frame_series_from_curves,
    write_frame_stability_results,
    write_frame_stability_shard,
)

from aswaxs_live.reduction.analysis_h5 import (
    create_analysis_h5_from_data,
    file_sha256,
    write_background_subtraction_to_analysis_h5,
    write_glassy_carbon_normalization_to_analysis_h5,
    write_reduction_to_analysis_h5,
)
from .sequence import (
    build_sequence_map,
    collect_files,
    expected_count,
    remove_sequence_indices,
    resolve_sequence_files,
    write_manifest,
)
from .pipeline import (
    DEFAULT_DATASET_PATH,
    DEFAULT_FLUORESCENCE_Q_RANGE,
    DEFAULT_GC_Q_RANGE,
    DEFAULT_NPT,
    _integrated_intensity,
    _load_mask,
    _load_reference_curve,
    energy_kev_to_wavelength_m,
    estimate_constant_fluorescence,
    poni_geometry_metadata,
)

PROJECT_DIR = Path(__file__).resolve().parents[1] / "core"

_GROUP_WORKER_ARGS: argparse.Namespace | None = None
_GROUP_WORKER_MONITOR_KEY = ""
_GROUP_WORKER_AI: object | None = None
_GROUP_WORKER_MASK: np.ndarray | None = None
_GROUP_WORKER_QC_DIR: Path | None = None


NDATTR_PREFIX = "entry/instrument/NDAttributes"


@dataclass
class ManifestItem:
    sequence_index: int
    energy_index: int
    group_index: int
    frame_index: int
    path: Path


@dataclass
class FrameCurve:
    item: ManifestItem
    energy_kev: float | None
    monitor_value: float
    q: np.ndarray
    intensity: np.ndarray
    intensity_error: np.ndarray
    total_intensity: float
    normalized_intensity: np.ndarray
    normalized_error: np.ndarray
    timing_seconds: dict[str, float] | None = None
    image_shape: tuple[int, ...] | None = None
    source_file_bytes: int | None = None


@dataclass
class GroupAverage:
    energy_index: int
    group_index: int
    q: np.ndarray
    energy_kev: float | None
    avg_intensity: np.ndarray
    avg_error: np.ndarray
    frame_count: int
    kept_count: int
    dropped_count: int
    kept_sequence_indices: list[int]
    dropped_sequence_indices: list[int]
    avg_total_intensity: float
    avg_monitor_value: float
    frame_qc: object | None = None
    frame_qc_status: str = "complete"
    frame_qc_shard: str | None = None
    frame_qc_group: str | None = None


@dataclass
class FinalRecord:
    energy_index: int
    energy_kev: float | None
    q: np.ndarray
    final_before_fluorescence: np.ndarray
    final_error: np.ndarray
    component_columns: list[np.ndarray]
    component_names: list[str]
    metadata: dict[str, object]


@dataclass
class FinalOutput:
    path: Path
    component_path: Path
    output_name: str
    energy_index: int
    energy_kev: float | None
    q: np.ndarray
    I: np.ndarray
    sigma_I: np.ndarray
    component_names: list[str]
    component_columns: list[np.ndarray]
    metadata: dict[str, object]


@dataclass
class ExtractionRecipe:
    output_name: str
    sample_group: int
    air_group: int | None
    empty_group: int | None
    water_group: int | None
    gc_group: int | None


def stabilize_energy_kev(
    energy_kev: float | None,
    previous_energy_kev: float | None,
    delta_percent: float,
) -> float | None:
    """Suppress tiny monochromator readback jitter within one energy point."""
    if energy_kev is None:
        return previous_energy_kev
    if previous_energy_kev is None:
        return energy_kev
    if delta_percent < 0:
        raise ValueError("delta energy percent must be non-negative.")
    if previous_energy_kev == 0:
        return energy_kev
    percent_diff = abs(energy_kev - previous_energy_kev) / abs(previous_energy_kev) * 100.0
    if percent_diff <= delta_percent:
        return previous_energy_kev
    return energy_kev


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reduce an ASWAXS sequence manifest with frame rejection, monitor normalization, averaging, air subtraction, and GC scaling."
    )
    parser.add_argument("--manifest", help="Existing sequence manifest, or output manifest path when using --data-dir.")
    parser.add_argument("--data-dir", help="Directory containing the continuous HDF5 sequence. If provided, a manifest is created first.")
    parser.add_argument("--pattern", default="*.h5", help="HDF5 filename pattern when using --data-dir. Default: *.h5")
    parser.add_argument("--num-energies", type=int, help="Number of energies in the sequence.")
    parser.add_argument("--num-groups", type=int, help="Number of groups per energy.")
    parser.add_argument("--num-frames", type=int, help="Number of repeated frames per group.")
    parser.add_argument("--skip-files", type=int, default=0, help="Number of leading sorted files to ignore before sequence mapping.")
    parser.add_argument(
        "--skip-sequence-indices",
        nargs="*",
        type=int,
        default=[],
        help="One-based sorted file positions to skip after --skip-files, for known beamdown/repeated measurements.",
    )
    parser.add_argument(
        "--allow-extra-files",
        action="store_true",
        help="Allow more files than expected and use the first complete sequence.",
    )
    parser.add_argument(
        "--resume-mode",
        choices=("strict", "first", "last"),
        default="strict",
        help="How to handle extra files when creating a manifest. Default: strict.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not ask for beamdown indices interactively while creating a manifest.",
    )
    parser.add_argument("--poni", required=True, help="Calibration .poni file.")
    parser.add_argument("--mask", required=True, help="Mask file (.npy or EDF-readable).")
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs" / "sequence_reduction_output"), help="Output directory.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes for dynamic energy/group reduction jobs. Default: 1",
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="HDF5 detector image dataset path.")
    parser.add_argument("--npt", type=int, default=DEFAULT_NPT, help="Number of q bins.")
    parser.add_argument("--unit", default="q_A^-1", help="pyFAI radial unit.")
    parser.add_argument(
        "--delta-energy-percent",
        type=float,
        default=1e-3,
        help="Accept consecutive Mono_Energy fluctuations within this percent as the same previous energy. Default: 1e-3",
    )
    parser.add_argument(
        "--detector",
        choices=("auto", "Pil300K", "Eig1M"),
        default="auto",
        help="Detector used to choose default monitor normalization.",
    )
    parser.add_argument(
        "--monitor-key",
        help="Override monitor normalization key. Defaults: Pil300K -> SPDS, Eig1M -> WPDS.",
    )
    parser.add_argument("--outlier-zmax", type=float, default=3.5, help="MAD modified-z threshold for bad-frame rejection.")
    parser.add_argument("--sample-group", type=int, help="Group index for sample.")
    parser.add_argument("--asaxs-output-name", default="sample", help="Output name for the single sample-group workflow.")
    parser.add_argument(
        "--asaxs-pair",
        action="append",
        default=[],
        metavar="NAME:SAMPLE_GROUP:SOLVENT_GROUP",
        help=(
            "Named ASAXS extraction. May be repeated. GC/air/empty groups are shared; "
            "sample and solvent are specified per output, for example Pt_mesh:5:4."
        ),
    )
    parser.add_argument(
        "--asaxs-extraction-plan",
        help=(
            "Optional CSV or JSON extraction table for multiple named sample outputs. "
            "Columns/keys: output_name,sample_group,water_group. Shared air/empty/GC groups still come from the GUI/CLI."
        ),
    )
    parser.add_argument("--air-group", type=int, help="Group index for air/background.")
    parser.add_argument("--empty-group", type=int, help="Group index for empty cell/capillary.")
    parser.add_argument("--water-group", type=int, help="Group index for water reference.")
    parser.add_argument("--gc-group", type=int, help="Group index for glassy carbon.")
    parser.add_argument(
        "--gc-reference-file",
        help="Optional two-column q,I glassy carbon reference curve. Default: built-in NIST SRM 3600 values.",
    )
    parser.add_argument(
        "--gc-q-range",
        nargs=2,
        type=float,
        default=DEFAULT_GC_Q_RANGE,
        metavar=("QMIN", "QMAX"),
        help="q range used to match glassy carbon to reference. Default: 0.03 0.20",
    )
    parser.add_argument(
        "--capillary-thickness",
        type=float,
        help="XAnoS sample/tube thickness. Stored in headers; downstream XAnoS applies CF/thickness.",
    )
    parser.add_argument(
        "--gc-thickness",
        type=float,
        help="XAnoS glassy-carbon standard thickness. CF is fitted as reference = CF * measured_GC / gc_thickness.",
    )
    parser.add_argument("--subtract-fluorescence", action="store_true", help="Subtract constant fluorescence background.")
    parser.add_argument(
        "--fluorescence-q-range",
        nargs=2,
        type=float,
        default=DEFAULT_FLUORESCENCE_Q_RANGE,
        metavar=("QMIN", "QMAX"),
        help="q range used to estimate fluorescence. Default: 0.16 0.20",
    )
    parser.add_argument("--fluorescence-level", type=float, help="Fixed fluorescence level.")
    parser.add_argument(
        "--fluorescence-reference",
        choices=("latest", "each"),
        default="latest",
        help="Use the latest energy curve as one shared fluorescence background, or estimate each curve separately. Default: latest.",
    )
    parser.add_argument(
        "--analysis-h5",
        help="Analysis HDF5 output path. Default: output-dir/analysis.h5.",
    )
    return parser


def read_manifest(path: Path) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence_index", "energy_index", "group_index", "frame_index", "hdf5_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            items.append(
                ManifestItem(
                    sequence_index=int(row["sequence_index"]),
                    energy_index=int(row["energy_index"]),
                    group_index=int(row["group_index"]),
                    frame_index=int(row["frame_index"]),
                    path=Path(row["hdf5_path"]).expanduser().resolve(),
                )
            )
    if not items:
        raise ValueError("Manifest has no sequence rows.")
    return items


def asaxs_group_roles(args: argparse.Namespace) -> dict[int, list[str]]:
    """Map configured ASAXS group numbers to their scientific roles."""
    roles: dict[int, list[str]] = {}
    for attr, role in [
        ("sample_group", "sample"),
        ("air_group", "air"),
        ("empty_group", "empty_cell"),
        ("water_group", "water"),
        ("gc_group", "glassy_carbon"),
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            roles.setdefault(int(value), []).append(role)
    return roles


def asaxs_role_for_group(group_index: int, args: argparse.Namespace) -> str:
    """Return the configured ASAXS role name for a group index."""
    roles = asaxs_group_roles(args).get(int(group_index), [])
    return "+".join(roles) if roles else "unassigned"


def validate_asaxs_group_roles(args: argparse.Namespace) -> None:
    """Stop ASAXS reduction when different scientific roles share a group."""
    roles = asaxs_group_roles(args)
    collisions = {group: names for group, names in roles.items() if len(names) > 1}
    if collisions:
        details = "; ".join(f"group {group}: {', '.join(names)}" for group, names in sorted(collisions.items()))
        raise ValueError(
            "ASAXS role groups overlap. Each role must use a separate group, otherwise "
            f"averages can mix sample/background/GC data. Overlap: {details}"
        )


def safe_output_name(name: str) -> str:
    """Return a filesystem/HDF5 friendly output name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return cleaned.strip("._") or "sample"


def extraction_recipes(args: argparse.Namespace) -> list[ExtractionRecipe]:
    """Return named ASAXS sample/solvent extraction recipes.

    GC, air, and empty groups are shared run-level settings. Each recipe chooses
    one sample group and one solvent/water group and gives the result a stable
    output name for analysis HDF5 and XAnos export.
    """
    recipes: list[ExtractionRecipe] = []
    plan_path = getattr(args, "asaxs_extraction_plan", None)
    if plan_path:
        recipes.extend(_read_extraction_plan(Path(plan_path).expanduser()))
    for text in getattr(args, "asaxs_pair", []) or []:
        recipes.append(_parse_extraction_pair(text))
    if not recipes and getattr(args, "sample_group", None) is not None:
        recipes.append(
            ExtractionRecipe(
                output_name=safe_output_name(getattr(args, "asaxs_output_name", "sample")),
                sample_group=int(args.sample_group),
                air_group=args.air_group,
                empty_group=args.empty_group,
                water_group=args.water_group,
                gc_group=args.gc_group,
            )
        )
    shared: list[ExtractionRecipe] = []
    seen_names: set[str] = set()
    for recipe in recipes:
        name = safe_output_name(recipe.output_name)
        if name in seen_names:
            raise ValueError(f"Duplicate ASAXS output name: {name}")
        seen_names.add(name)
        shared_groups = {
            "air": args.air_group,
            "empty": args.empty_group,
            "gc": args.gc_group,
        }
        for role, group in shared_groups.items():
            if group in {None, 0}:
                continue
            if int(recipe.sample_group) == int(group):
                raise ValueError(f"ASAXS output {name}: sample group {recipe.sample_group} overlaps shared {role} group.")
            if recipe.water_group not in {None, 0} and int(recipe.water_group) == int(group):
                raise ValueError(f"ASAXS output {name}: solvent group {recipe.water_group} overlaps shared {role} group.")
        if recipe.water_group not in {None, 0} and int(recipe.water_group) == int(recipe.sample_group):
            raise ValueError(f"ASAXS output {name}: sample and solvent groups are the same.")
        shared.append(
            ExtractionRecipe(
                output_name=name,
                sample_group=int(recipe.sample_group),
                air_group=args.air_group,
                empty_group=args.empty_group,
                water_group=None if recipe.water_group in {None, 0} else int(recipe.water_group),
                gc_group=args.gc_group,
            )
        )
    return shared


def _parse_extraction_pair(text: str) -> ExtractionRecipe:
    parts = [part.strip() for part in str(text).replace(",", ":").split(":")]
    if len(parts) != 3:
        raise ValueError("--asaxs-pair must use NAME:SAMPLE_GROUP:SOLVENT_GROUP")
    name, sample_group, solvent_group = parts
    return ExtractionRecipe(
        output_name=safe_output_name(name),
        sample_group=int(sample_group),
        air_group=None,
        empty_group=None,
        water_group=int(solvent_group) if solvent_group else None,
        gc_group=None,
    )


def _read_extraction_plan(path: Path) -> list[ExtractionRecipe]:
    if not path.exists():
        raise FileNotFoundError(f"Missing ASAXS extraction plan: {path}")
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, Mapping):
            rows = rows.get("extractions", [])
        return [_recipe_from_mapping(row) for row in rows]
    with path.open(newline="", encoding="utf-8") as handle:
        return [_recipe_from_mapping(row) for row in csv.DictReader(handle)]


def _recipe_from_mapping(row: object) -> ExtractionRecipe:
    if not isinstance(row, dict):
        raise ValueError("Each extraction plan entry must be a mapping/object.")
    name = row.get("output_name") or row.get("name") or row.get("sample_name")
    sample_group = row.get("sample_group")
    water_group = row.get("water_group", row.get("solvent_group"))
    if name in {None, ""} or sample_group in {None, ""}:
        raise ValueError("Extraction plan rows require output_name and sample_group.")
    return ExtractionRecipe(
        output_name=safe_output_name(str(name)),
        sample_group=int(sample_group),
        air_group=None,
        empty_group=None,
        water_group=None if water_group in {None, ""} else int(water_group),
        gc_group=None,
    )


def create_manifest_from_sequence(args: argparse.Namespace, output_dir: Path) -> Path:
    missing = [
        name
        for name in ("num_energies", "num_groups", "num_frames")
        if getattr(args, name) is None
    ]
    if missing:
        options = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(f"When using --data-dir, also provide: {options}")
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    validate_asaxs_group_roles(args)

    expected = expected_count(args.num_energies, args.num_groups, args.num_frames)
    raw_files = collect_files(data_dir, args.pattern, args.skip_files)
    sequence_files, skip_indices = resolve_sequence_files(
        raw_files=raw_files,
        expected=expected,
        initial_skip_indices=args.skip_sequence_indices,
        allow_extra_files=args.allow_extra_files,
        resume_mode=args.resume_mode,
        no_prompt=args.no_prompt,
    )
    sequence_items = build_sequence_map(sequence_files, args.num_energies, args.num_groups, args.num_frames)
    manifest_path = Path(args.manifest).expanduser() if args.manifest else output_dir / "sequence_manifest.csv"
    manifest = write_manifest(sequence_items, manifest_path)

    print("ASWAXS sequence validated.")
    print(f"Data directory: {data_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Expected files: {expected}")
    print(f"Actual files found after leading skip: {len(raw_files)}")
    if skip_indices:
        print(f"Skipped sequence indices: {', '.join(str(index) for index in skip_indices)}")
    print(f"Actual files after beamdown skips: {len(remove_sequence_indices(raw_files, skip_indices))}")
    print(f"Actual files used: {len(sequence_files)}")
    print(f"Resume mode: {args.resume_mode}")
    print(f"Energies: {args.num_energies}")
    print(f"Groups per energy: {args.num_groups}")
    print(f"Frames per group: {args.num_frames}")
    print(f"Manifest: {manifest}")
    return manifest


def resolve_manifest(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.data_dir:
        return create_manifest_from_sequence(args, output_dir)
    if not args.manifest:
        raise ValueError("Provide either --manifest or --data-dir with --num-energies, --num-groups, and --num-frames.")
    return Path(args.manifest).expanduser().resolve()


def infer_detector(items: list[ManifestItem], requested: str) -> str:
    if requested != "auto":
        return requested
    text = " ".join(str(item.path) for item in items[: min(len(items), 10)])
    if "Pil300K" in text:
        return "Pil300K"
    if "Eig1M" in text:
        return "Eig1M"
    raise ValueError("Could not infer detector from manifest paths. Use --detector Pil300K or --detector Eig1M.")


def default_monitor_key(detector: str) -> str:
    if detector == "Pil300K":
        return "SPDS"
    if detector == "Eig1M":
        return "WPDS"
    raise ValueError(f"Unsupported detector for monitor normalization: {detector}")


def read_ndattr_scalar(path: Path, key: str) -> float | None:
    with h5py.File(path, "r") as handle:
        return read_ndattr_scalar_from_handle(handle, key)


def read_ndattr_scalar_from_handle(handle: h5py.File, key: str) -> float | None:
    key = str(key).strip().lstrip("/")
    candidates = [key] if "/" in key else []
    candidates.append(f"{NDATTR_PREFIX}/{key}")
    dataset_path = next((candidate for candidate in candidates if candidate in handle), None)
    if dataset_path is None:
        return None
    value = np.asarray(handle[dataset_path][()])
    if value.size == 0:
        return None
    return float(value.reshape(-1)[0])


def read_hdf5_image_from_handle(handle: h5py.File, path: Path, dataset_path: str, frame: int | None) -> np.ndarray:
    """Read a 2D image or average/select from a 3D frame stack from an open HDF5 file."""
    if dataset_path not in handle:
        raise KeyError(f"Dataset '{dataset_path}' was not found in {path}")
    data = np.asarray(handle[dataset_path][()], dtype=np.float32)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        if frame is None:
            return np.nanmean(data, axis=0).astype(np.float32)
        return np.asarray(data[frame], dtype=np.float32)
    raise ValueError(f"Expected a 2D image or 3D frame stack in {path}; found shape {data.shape}")


def read_frame_inputs(path: Path, dataset_path: str, monitor_key: str) -> tuple[float | None, np.ndarray, float | None, dict[str, float]]:
    timings: dict[str, float] = {}
    open_start = time.perf_counter()
    with h5py.File(path, "r") as handle:
        timings["h5_open"] = time.perf_counter() - open_start
        energy_start = time.perf_counter()
        raw_energy_kev = read_ndattr_scalar_from_handle(handle, "Mono_Energy")
        timings["read_energy"] = time.perf_counter() - energy_start
        image_start = time.perf_counter()
        image = read_hdf5_image_from_handle(handle, path, dataset_path, frame=None)
        timings["read_image"] = time.perf_counter() - image_start
        monitor_start = time.perf_counter()
        monitor_value = read_ndattr_scalar_from_handle(handle, monitor_key)
        timings["read_monitor"] = time.perf_counter() - monitor_start
    return raw_energy_kev, image, monitor_value, timings


def reject_outliers(total_intensities: np.ndarray, zmax: float) -> np.ndarray:
    finite = np.isfinite(total_intensities)
    if np.count_nonzero(finite) == 0:
        return np.zeros_like(total_intensities, dtype=bool)
    median = np.median(total_intensities[finite])
    mad = np.median(np.abs(total_intensities[finite] - median))
    if mad == 0:
        return finite
    modified_z = np.full_like(total_intensities, np.inf, dtype=float)
    modified_z[finite] = 0.6745 * (total_intensities[finite] - median) / mad
    return np.abs(modified_z) <= zmax


def resample_to_q(q_target: np.ndarray, q_source: np.ndarray, values: np.ndarray) -> np.ndarray:
    mask = np.isfinite(q_source) & np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return np.full_like(q_target, np.nan, dtype=float)
    return np.interp(q_target, q_source[mask], values[mask], left=np.nan, right=np.nan)


def q_grids_match(first: np.ndarray, second: np.ndarray, rtol: float = 1e-6, atol: float = 1e-12) -> bool:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    return first.shape == second.shape and np.allclose(first, second, rtol=rtol, atol=atol, equal_nan=True)


def values_on_q(q_target: np.ndarray, q_source: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) if q_grids_match(q_target, q_source) else resample_to_q(q_target, q_source, values)


def stack_ragged_rows(rows: list[np.ndarray], fill_value: float = np.nan) -> np.ndarray:
    """Stack per-energy arrays while preserving row-specific q grids."""
    if not rows:
        return np.empty((0, 0), dtype=float)
    arrays = [np.asarray(row, dtype=float).reshape(-1) for row in rows]
    width = max(array.size for array in arrays)
    stacked = np.full((len(arrays), width), fill_value, dtype=float)
    for row_index, array in enumerate(arrays):
        stacked[row_index, : array.size] = array
    return stacked


def reduce_manifest_frames(
    items: list[ManifestItem],
    args: argparse.Namespace,
    monitor_key: str,
    progress_label: str | None = None,
    progress_queue: object | None = None,
    image_callback: object | None = None,
    integrator: object | None = None,
    loaded_mask: np.ndarray | None = None,
) -> list[FrameCurve]:
    """Integrate every manifest item and keep per-frame values for later QC."""
    curves: list[FrameCurve] = []
    ai = integrator if integrator is not None else load_poni(str(Path(args.poni).expanduser().resolve()))
    mask = loaded_mask if loaded_mask is not None else _load_mask(Path(args.mask).expanduser().resolve())
    previous_energy_kev: float | None = None
    for idx, item in enumerate(items, start=1):
        frame_start = time.perf_counter()
        raw_energy_kev, image, monitor_value, input_timings = read_frame_inputs(item.path, args.dataset_path, monitor_key)
        if callable(image_callback):
            image_callback(item, image)
        energy_kev = stabilize_energy_kev(raw_energy_kev, previous_energy_kev, args.delta_energy_percent)
        if energy_kev is not None:
            ai.wavelength = energy_kev_to_wavelength_m(energy_kev)
            previous_energy_kev = energy_kev
        integrate_start = time.perf_counter()
        q, intensity, intensity_error = integrate_image_with_counting_error(ai, image, mask, args.npt, args.unit)
        integrate_seconds = time.perf_counter() - integrate_start
        if monitor_value is None:
            raise KeyError(f"Monitor key {monitor_key} not found in {item.path}")
        if monitor_value == 0:
            raise ValueError(f"Monitor key {monitor_key} is zero in {item.path}")
        total_intensity = _integrated_intensity(q, intensity)
        total_seconds = time.perf_counter() - frame_start
        curves.append(
            FrameCurve(
                item=item,
                energy_kev=energy_kev,
                monitor_value=monitor_value,
                q=q,
                intensity=intensity,
                intensity_error=intensity_error,
                total_intensity=total_intensity,
                normalized_intensity=intensity / monitor_value,
                normalized_error=intensity_error / abs(monitor_value),
                timing_seconds={
                    "total": total_seconds,
                    "h5_open": input_timings["h5_open"],
                    "read_energy": input_timings["read_energy"],
                    "read_image": input_timings["read_image"],
                    "integrate": integrate_seconds,
                    "read_monitor": input_timings["read_monitor"],
                    "h5_input_total": sum(input_timings.values()),
                },
                image_shape=tuple(int(part) for part in image.shape),
                source_file_bytes=None,
            )
        )
        if idx == 1 or idx % 50 == 0 or idx == len(items):
            prefix = f"[{progress_label}] " if progress_label else ""
            message = (
                f"{prefix}Reduced frame {idx}/{len(items)}: {item.path.name} "
                f"(total={total_seconds:.3f}s, h5={sum(input_timings.values()):.3f}s, "
                f"read={input_timings['read_image']:.3f}s, integrate={integrate_seconds:.3f}s)"
            )
            if progress_queue is not None:
                progress_queue.put(message)
            else:
                print(message)
    summary = reduction_timing_summary(curves)
    if summary:
        prefix = f"[{progress_label}] " if progress_label else ""
        if progress_queue is not None:
            progress_queue.put(prefix + summary)
        else:
            print(prefix + summary)
    return curves


def integrate_image_with_counting_error(ai, image: np.ndarray, mask: np.ndarray, npt: int, unit: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate one image and propagate raw counting uncertainty into I(q)."""
    if image.shape != mask.shape:
        raise ValueError(f"Image shape {image.shape} does not match mask shape {mask.shape}")
    variance = np.abs(np.asarray(image, dtype=np.float32))
    result = ai.integrate1d(image, npt, mask=mask, unit=unit, variance=variance, error_model="poisson")
    q = np.asarray(result.radial, dtype=float)
    intensity = np.asarray(result.intensity, dtype=float)
    sigma = np.asarray(getattr(result, "sigma", np.full_like(intensity, np.nan)), dtype=float)
    if sigma.shape != intensity.shape or not np.any(np.isfinite(sigma)):
        sigma = np.sqrt(np.clip(np.abs(intensity), 0.0, None))
    return q, intensity, sigma


def reduction_timing_summary(curves: list[FrameCurve]) -> str:
    if not curves:
        return ""
    keys = ["h5_open", "read_energy", "read_image", "read_monitor", "h5_input_total", "integrate", "total"]
    sums = {key: 0.0 for key in keys}
    for curve in curves:
        timing = curve.timing_seconds or {}
        for key in keys:
            sums[key] += float(timing.get(key, 0.0) or 0.0)
    total = max(sums["total"], 1e-12)
    h5_total = sums["h5_input_total"]
    parts = [
        f"frames={len(curves)}",
        f"total={sums['total']:.2f}s",
        f"h5_input={h5_total:.2f}s ({h5_total / total * 100:.1f}%)",
        f"integrate={sums['integrate']:.2f}s ({sums['integrate'] / total * 100:.1f}%)",
        f"h5_open={sums['h5_open']:.2f}s",
        f"image_read={sums['read_image']:.2f}s",
        f"metadata_read={(sums['read_energy'] + sums['read_monitor']):.2f}s",
    ]
    return "Timing summary: " + ", ".join(parts)


def initialize_group_worker(args_dict: dict[str, object], monitor_key: str, qc_dir: str) -> None:
    """Load immutable calibration once in each group worker process."""
    global _GROUP_WORKER_ARGS, _GROUP_WORKER_MONITOR_KEY, _GROUP_WORKER_AI, _GROUP_WORKER_MASK, _GROUP_WORKER_QC_DIR
    _GROUP_WORKER_ARGS = argparse.Namespace(**args_dict)
    _GROUP_WORKER_MONITOR_KEY = monitor_key
    _GROUP_WORKER_AI = load_poni(str(Path(_GROUP_WORKER_ARGS.poni).expanduser().resolve()))
    _GROUP_WORKER_MASK = _load_mask(Path(_GROUP_WORKER_ARGS.mask).expanduser().resolve())
    _GROUP_WORKER_QC_DIR = Path(qc_dir)


def reduce_group_worker(
    energy_index: int,
    group_index: int,
    items: list[ManifestItem],
    progress_queue: object | None,
) -> GroupAverage:
    """Reduce, QC, and average one energy/group using worker-cached calibration."""
    if _GROUP_WORKER_ARGS is None or _GROUP_WORKER_AI is None or _GROUP_WORKER_MASK is None:
        raise RuntimeError("Reduction worker was not initialized.")
    label = f"E{energy_index:03d} G{group_index:03d}"
    curves = reduce_manifest_frames(
        items,
        _GROUP_WORKER_ARGS,
        _GROUP_WORKER_MONITOR_KEY,
        progress_label=label,
        progress_queue=progress_queue,
        integrator=_GROUP_WORKER_AI,
        loaded_mask=_GROUP_WORKER_MASK,
    )
    averages = average_groups(curves, _GROUP_WORKER_ARGS.outlier_zmax)
    if len(averages) != 1:
        raise RuntimeError(f"Expected one group average for {label}; found {len(averages)}")
    average = averages[0]
    if average.frame_qc is not None:
        if _GROUP_WORKER_QC_DIR is None:
            raise RuntimeError("QC temporary directory was not initialized.")
        shard_path = _GROUP_WORKER_QC_DIR / f"worker_{os.getpid()}.h5"
        average.frame_qc_group = write_frame_stability_shard(
            shard_path,
            str(getattr(_GROUP_WORKER_ARGS, "detector", "")),
            average,
            average.frame_qc,
        )
        average.frame_qc_shard = str(shard_path)
        average.frame_qc = None
    if progress_queue is not None:
        progress_queue.put(f"[{label}] Group average complete")
    return average


def drain_progress_queue(progress_queue: object | None) -> None:
    if progress_queue is None:
        return
    while True:
        try:
            message = progress_queue.get_nowait()
        except queue.Empty:
            break
        else:
            print(message)


def reduce_manifest_frames_parallel(
    items: list[ManifestItem],
    args: argparse.Namespace,
    monitor_key: str,
) -> list[GroupAverage]:
    grouped_items: dict[tuple[int, int], list[ManifestItem]] = {}
    for item in items:
        grouped_items.setdefault((item.energy_index, item.group_index), []).append(item)
    group_jobs = [
        (energy_index, group_index, sorted(group_items, key=lambda item: item.frame_index))
        for (energy_index, group_index), group_items in sorted(grouped_items.items())
    ]
    max_workers = max(1, os.cpu_count() or 1)
    jobs = max(1, min(args.jobs, len(group_jobs), max_workers))
    if jobs == 1:
        curves = reduce_manifest_frames(items, args, monitor_key)
        return average_groups(curves, args.outlier_zmax)

    print(
        f"Parallel group reduction enabled: workers={jobs}, cpu_max={max_workers}, "
        f"energy/group jobs={len(group_jobs)}"
    )

    args_dict = vars(args).copy()
    averages: list[GroupAverage] = []
    qc_temp_dir = Path(tempfile.mkdtemp(prefix="framebyframe_qc_"))
    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=jobs,
                initializer=initialize_group_worker,
                initargs=(args_dict, monitor_key, str(qc_temp_dir)),
            ) as executor:
                future_map = {
                    executor.submit(
                        reduce_group_worker,
                        energy_index,
                        group_index,
                        group_items,
                        progress_queue,
                    ): (energy_index, group_index)
                    for energy_index, group_index, group_items in group_jobs
                }
                pending = set(future_map)
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=0.2,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    drain_progress_queue(progress_queue)
                    for future in done:
                        energy_index, group_index = future_map[future]
                        averages.append(future.result())
                        print(f"Completed E{energy_index:03d} G{group_index:03d}")
                drain_progress_queue(progress_queue)
        except Exception:
            shutil.rmtree(qc_temp_dir, ignore_errors=True)
            raise
    if not any(average.frame_qc_shard for average in averages):
        shutil.rmtree(qc_temp_dir, ignore_errors=True)
    return sorted(averages, key=lambda avg: (avg.energy_index, avg.group_index))


def average_groups(curves: list[FrameCurve], zmax: float) -> list[GroupAverage]:
    """Reject repeated-frame outliers and average accepted frames by group."""
    grouped: dict[tuple[int, int], list[FrameCurve]] = {}
    for curve in curves:
        grouped.setdefault((curve.item.energy_index, curve.item.group_index), []).append(curve)

    averages: list[GroupAverage] = []
    for (energy_index, group_index), group_curves in sorted(grouped.items()):
        frame_qc = analyze_frame_series(frame_series_from_curves(group_curves)) if len(group_curves) > 1 else None
        frame_qc_status = "complete" if frame_qc is not None else "not_applicable_single_frame"
        totals = np.asarray([curve.total_intensity for curve in group_curves], dtype=float)
        keep = reject_outliers(totals, zmax)
        if np.count_nonzero(keep) == 0:
            raise ValueError(f"All frames rejected for energy {energy_index}, group {group_index}")
        kept_curves = [curve for curve, keep_one in zip(group_curves, keep) if keep_one]
        q_ref = kept_curves[0].q
        kept_stack = np.vstack([resample_to_q(q_ref, curve.q, curve.normalized_intensity) for curve in kept_curves])
        kept_error_stack = np.vstack([resample_to_q(q_ref, curve.q, curve.normalized_error) for curve in kept_curves])
        avg_intensity = np.nanmean(kept_stack, axis=0)
        raw_error = np.sqrt(np.nansum(kept_error_stack**2, axis=0)) / max(1, len(kept_curves))
        if len(kept_curves) > 1:
            avg_error = np.nanstd(kept_stack, axis=0, ddof=1)
            avg_error = np.where(np.isfinite(avg_error), avg_error, raw_error)
        else:
            avg_error = raw_error
        energy_values = [curve.energy_kev for curve in kept_curves if curve.energy_kev is not None]
        monitor_values = [curve.monitor_value for curve in kept_curves]
        averages.append(
            GroupAverage(
                energy_index=energy_index,
                group_index=group_index,
                q=q_ref,
                energy_kev=float(np.mean(energy_values)) if energy_values else None,
                avg_intensity=avg_intensity,
                avg_error=avg_error,
                frame_count=len(group_curves),
                kept_count=len(kept_curves),
                dropped_count=len(group_curves) - len(kept_curves),
                kept_sequence_indices=[curve.item.sequence_index for curve in kept_curves],
                dropped_sequence_indices=[
                    curve.item.sequence_index for curve, keep_one in zip(group_curves, keep) if not keep_one
                ],
                avg_total_intensity=_integrated_intensity(q_ref, avg_intensity),
                avg_monitor_value=float(np.mean(monitor_values)),
                frame_qc=frame_qc,
                frame_qc_status=frame_qc_status,
            )
        )
    return averages


def group_lookup(averages: list[GroupAverage]) -> dict[tuple[int, int], GroupAverage]:
    return {(avg.energy_index, avg.group_index): avg for avg in averages}


def energy_header_lines(energy_kev: float | None) -> list[str]:
    if energy_kev is None:
        return []
    return [
        f"Energy={energy_kev}",
        f"Wavelength={12.398419843320026 / energy_kev}",
    ]


def write_group_average(
    avg: GroupAverage,
    output_dir: Path,
    geometry_metadata: Mapping[str, object] | None = None,
) -> Path:
    path = output_dir / "groups" / f"energy_{avg.energy_index:03d}_group_{avg.group_index:02d}_avg.dat"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "energy_index": avg.energy_index,
        "group_index": avg.group_index,
        "energy_kev": avg.energy_kev,
        "frame_count": avg.frame_count,
        "kept_count": avg.kept_count,
        "dropped_count": avg.dropped_count,
        "kept_sequence_indices": avg.kept_sequence_indices,
        "dropped_sequence_indices": avg.dropped_sequence_indices,
        "avg_monitor_value": avg.avg_monitor_value,
    }
    if geometry_metadata:
        metadata.update(geometry_metadata)
    header = "\n".join(
        [
            "ASWAXS group average after bad-frame rejection and monitor normalization",
            *energy_header_lines(avg.energy_kev),
            "metadata_json=" + json.dumps(metadata, sort_keys=True),
            "columns=q I_avg_monitor_normalized I_sigma_frame_std",
        ]
    )
    np.savetxt(path, np.column_stack([avg.q, avg.avg_intensity, avg.avg_error]), header=header, comments="#")
    return path


def write_summary(
    averages: list[GroupAverage],
    output_dir: Path,
    monitor_key: str,
    detector: str,
    args: argparse.Namespace | None = None,
) -> Path:
    path = output_dir / "group_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "energy_index",
                "group_index",
                "energy_kev",
                "frame_count",
                "kept_count",
                "dropped_count",
                "kept_sequence_indices",
                "dropped_sequence_indices",
                "avg_total_intensity_monitor_normalized",
                "avg_monitor_value",
                "monitor_key",
                "detector",
                "asaxs_role",
            ]
        )
        for avg in averages:
            role = asaxs_role_for_group(avg.group_index, args) if args is not None else "unknown"
            writer.writerow(
                [
                    avg.energy_index,
                    avg.group_index,
                    avg.energy_kev,
                    avg.frame_count,
                    avg.kept_count,
                    avg.dropped_count,
                    " ".join(str(value) for value in avg.kept_sequence_indices),
                    " ".join(str(value) for value in avg.dropped_sequence_indices),
                    avg.avg_total_intensity,
                    avg.avg_monitor_value,
                    monitor_key,
                    detector,
                    role,
                ]
            )
    return path


def compute_gc_scale(
    gc_curve: np.ndarray,
    q: np.ndarray,
    q_range: tuple[float, float],
    reference_file: Path | None,
    standard_thickness: float | None = None,
) -> tuple[float, float, float, float, float, float, str]:
    """Return the XAnoS-style GC calibration factor.

    XAnoS fits the calibration factor as:

        reference_I(q) = CF * measured_GC_I(q) / standard_thickness

    so CF = reference/measured * standard_thickness.  The final exported sample
    curve stores this CF in metadata/headers, but leaves the intensity unscaled
    so downstream XAnoS software can apply the calibration.
    """
    reference_q, reference_i = _load_reference_curve(reference_file)
    standard_thickness = 1.0 if standard_thickness is None else float(standard_thickness)
    if standard_thickness <= 0:
        raise ValueError("Glassy carbon standard thickness must be positive.")
    qmin, qmax = q_range
    finite = np.isfinite(q) & np.isfinite(gc_curve)
    reference_low = float(np.nanmin(reference_q))
    reference_high = float(np.nanmax(reference_q))
    effective_qmin = max(float(qmin), reference_low)
    effective_qmax = min(float(qmax), reference_high)
    mask = finite & (q >= effective_qmin) & (q <= effective_qmax)
    status = "requested_range"
    if np.count_nonzero(mask) < 2:
        # WAXS/Eig1M may start above the SAXS GC fitting range. Try any overlap
        # with the reference before giving up on absolute GC scaling.
        effective_qmin = max(float(np.nanmin(q[finite])) if np.any(finite) else np.nan, reference_low)
        effective_qmax = min(float(np.nanmax(q[finite])) if np.any(finite) else np.nan, reference_high)
        mask = finite & (q >= effective_qmin) & (q <= effective_qmax)
        status = "auto_overlap_range"
    if np.count_nonzero(mask) < 2:
        return 1.0, 1.0, np.nan, np.nan, np.nan, np.nan, "skipped_no_reference_overlap"
    q_window = q[mask]
    gc_window = gc_curve[mask]
    reference_window = np.interp(q_window, reference_q, reference_i)
    measured_area = _integrated_intensity(q_window, gc_window)
    reference_area = _integrated_intensity(q_window, reference_window)
    if not np.isfinite(measured_area) or measured_area <= 0:
        return 1.0, 1.0, measured_area, reference_area, float(np.nanmin(q_window)), float(np.nanmax(q_window)), "skipped_invalid_measured_area"
    reference_over_measured = float(reference_area / measured_area)
    calibration_factor = reference_over_measured * standard_thickness
    return (
        calibration_factor,
        reference_over_measured,
        measured_area,
        reference_area,
        float(np.nanmin(q_window)),
        float(np.nanmax(q_window)),
        status,
    )


def quadrature(*errors: np.ndarray) -> np.ndarray:
    finite_errors = [np.asarray(error, dtype=float) for error in errors if error is not None]
    if not finite_errors:
        raise ValueError("No error arrays were provided for quadrature.")
    variance = np.zeros_like(finite_errors[0], dtype=float)
    for error in finite_errors:
        variance = variance + np.square(error)
    return np.sqrt(variance)


def xanos_sample_thickness(args: argparse.Namespace) -> float:
    """Return sample/tube thickness used by XAnoS absolute scaling."""
    thickness = args.capillary_thickness
    if thickness is None:
        return 0.15
    if thickness <= 0:
        raise ValueError("Sample/capillary thickness must be positive.")
    return float(thickness)


def xanos_gc_standard_thickness(args: argparse.Namespace) -> float:
    """Return glassy-carbon standard thickness used while fitting CF."""
    thickness = args.gc_thickness
    if thickness is None:
        return 0.1055
    if thickness <= 0:
        raise ValueError("Glassy carbon standard thickness must be positive.")
    return float(thickness)


def build_final_record(
    energy_index: int,
    lookup: dict[tuple[int, int], GroupAverage],
    args: argparse.Namespace,
    gc_reference_file: Path | None,
    geometry_metadata: Mapping[str, object] | None = None,
) -> FinalRecord:
    """Build one corrected sample curve for an energy from averaged role groups."""
    sample = lookup.get((energy_index, args.sample_group))
    if sample is None:
        raise ValueError(f"Missing sample group {args.sample_group} at energy {energy_index}")
    q = sample.q
    sample_curve = sample.avg_intensity
    sample_error = sample.avg_error

    air_curve = None
    air_error = None
    if args.air_group is not None:
        air = lookup.get((energy_index, args.air_group))
        if air is None:
            raise ValueError(f"Missing air group {args.air_group} at energy {energy_index}")
        air_curve = values_on_q(q, air.q, air.avg_intensity)
        air_error = values_on_q(q, air.q, air.avg_error)

    empty_curve = None
    empty_error = None
    if args.empty_group is not None:
        empty = lookup.get((energy_index, args.empty_group))
        if empty is None:
            raise ValueError(f"Missing empty group {args.empty_group} at energy {energy_index}")
        empty_curve = values_on_q(q, empty.q, empty.avg_intensity)
        empty_error = values_on_q(q, empty.q, empty.avg_error)

    sample_background_curve = np.zeros_like(sample_curve)
    sample_background_error_terms: list[np.ndarray] = []
    sample_background_names: list[str] = []
    if empty_curve is not None:
        sample_background_curve = sample_background_curve + empty_curve
        sample_background_error_terms.append(empty_error)
        sample_background_names.append("empty")
    sample_background_error = (
        quadrature(*sample_background_error_terms) if sample_background_error_terms else np.zeros_like(sample_curve)
    )
    sample_minus_background = sample_curve - sample_background_curve
    sample_minus_background_error = quadrature(sample_error, sample_background_error)

    water_curve = None
    water_error = None
    water_minus_background = None
    water_minus_background_error = None
    sample_minus_water = sample_minus_background
    sample_minus_water_error = sample_minus_background_error
    if args.water_group is not None:
        water = lookup.get((energy_index, args.water_group))
        if water is None:
            raise ValueError(f"Missing water group {args.water_group} at energy {energy_index}")
        water_curve = values_on_q(q, water.q, water.avg_intensity)
        water_error = values_on_q(q, water.q, water.avg_error)
        water_minus_background = water_curve - sample_background_curve
        water_minus_background_error = quadrature(water_error, sample_background_error)
        sample_minus_water = sample_minus_background - water_minus_background
        sample_minus_water_error = quadrature(sample_minus_background_error, water_minus_background_error)

    gc_background_curve = np.zeros_like(sample_curve)
    gc_background_error_terms: list[np.ndarray] = []
    gc_background_names: list[str] = []
    if air_curve is not None:
        gc_background_curve = gc_background_curve + air_curve
        gc_background_error_terms.append(air_error)
        gc_background_names.append("air")
    gc_background_error = quadrature(*gc_background_error_terms) if gc_background_error_terms else np.zeros_like(sample_curve)
    gc_curve = None
    gc_error = None
    gc_minus_background = None
    gc_minus_background_error = None
    measured_area = None
    reference_area = None
    gc_q_min_used = np.nan
    gc_q_max_used = np.nan
    gc_scale_status = "not_requested"
    reference_over_measured = None
    gc_calibration_factor = None
    sample_thickness = xanos_sample_thickness(args)
    gc_standard_thickness = xanos_gc_standard_thickness(args)
    absolute_scale_factor = None
    if args.gc_group is not None:
        gc = lookup.get((energy_index, args.gc_group))
        if gc is None:
            raise ValueError(f"Missing glassy carbon group {args.gc_group} at energy {energy_index}")
        gc_curve = values_on_q(q, gc.q, gc.avg_intensity)
        gc_error = values_on_q(q, gc.q, gc.avg_error)
        gc_minus_background = gc_curve - gc_background_curve
        gc_minus_background_error = quadrature(gc_error, gc_background_error)
        (
            gc_calibration_factor,
            reference_over_measured,
            measured_area,
            reference_area,
            gc_q_min_used,
            gc_q_max_used,
            gc_scale_status,
        ) = compute_gc_scale(
            gc_minus_background,
            q,
            tuple(args.gc_q_range),
            gc_reference_file,
            gc_standard_thickness,
        )
        absolute_scale_factor = gc_calibration_factor / sample_thickness
    else:
        gc_calibration_factor = None

    final_before_fluorescence = sample_minus_water
    final_error = sample_minus_water_error

    component_columns = [q, sample_curve, sample_error]
    component_names = ["q", "I_sample_avg_norm", "I_sample_err"]
    if air_curve is not None:
        component_columns.extend([air_curve, air_error])
        component_names.extend(["I_air_avg_norm", "I_air_err"])
    if empty_curve is not None:
        component_columns.extend([empty_curve, empty_error])
        component_names.extend(["I_empty_avg_norm", "I_empty_err"])
    if water_curve is not None:
        component_columns.extend([water_curve, water_error])
        component_names.extend(["I_water_avg_norm", "I_water_err"])
    if sample_background_names:
        component_columns.extend([sample_background_curve, sample_background_error])
        component_names.extend(["I_sample_background", "I_sample_background_err"])
    component_columns.extend([sample_minus_background, sample_minus_background_error])
    component_names.extend(["I_sample_minus_empty", "I_sample_minus_empty_err"])
    if water_minus_background is not None:
        component_columns.extend([water_minus_background, water_minus_background_error])
        component_names.extend(["I_water_minus_empty", "I_water_minus_empty_err"])
        component_columns.extend([sample_minus_water, sample_minus_water_error])
        component_names.extend(["I_sample_minus_water_after_empty", "I_sample_minus_water_after_empty_err"])
    if gc_curve is not None:
        if gc_background_names:
            component_columns.extend([gc_background_curve, gc_background_error])
            component_names.extend(["I_gc_background", "I_gc_background_err"])
        component_columns.extend([gc_curve, gc_error, gc_minus_background, gc_minus_background_error])
        component_names.extend(["I_gc_avg_norm", "I_gc_err", "I_gc_minus_air", "I_gc_minus_air_err"])
    metadata = {
        "energy_index": energy_index,
        "energy_kev": sample.energy_kev,
        "sample_group": args.sample_group,
        "air_group": args.air_group,
        "empty_group": args.empty_group,
        "water_group": args.water_group,
        "gc_group": args.gc_group,
        "sample_background_terms": sample_background_names,
        "gc_background_terms": gc_background_names,
        "correction_order": "sample-empty; water-empty; sample_corrected-water_corrected; gc-air used only to compute XAnoS CF; final I is not CF/thickness scaled",
        "gc_scale_factor": reference_over_measured,
        "gc_reference_over_measured_factor": reference_over_measured,
        "gc_calibration_factor": gc_calibration_factor,
        "xanos_calibration_factor": gc_calibration_factor,
        "sample_thickness": sample_thickness,
        "capillary_thickness": args.capillary_thickness,
        "gc_standard_thickness": gc_standard_thickness,
        "gc_thickness": args.gc_thickness,
        "sample_thickness_normalization_factor": 1.0 / sample_thickness if sample_thickness else None,
        "thickness_normalization_factor": 1.0 / sample_thickness if sample_thickness else None,
        "combined_scale_factor": absolute_scale_factor,
        "absolute_scale_factor": absolute_scale_factor,
        "xanos_scale_formula": "CF stored for downstream XAnoS; exported I is unscaled corrected intensity. Downstream may use I_absolute = I_exported * CF / sample_thickness.",
        "gc_measured_area": measured_area,
        "gc_reference_area": reference_area,
        "gc_q_min_used": gc_q_min_used,
        "gc_q_max_used": gc_q_max_used,
        "gc_scale_status": gc_scale_status,
        "error_model": "raw Poisson/counting uncertainty propagated through pyFAI integration and monitor normalization; frame scatter used when multiple accepted frames are available; subtraction errors by quadrature; GC scale uncertainty not included",
    }
    if geometry_metadata:
        metadata.update(geometry_metadata)
    return FinalRecord(
        energy_index=energy_index,
        energy_kev=sample.energy_kev,
        q=q,
        final_before_fluorescence=final_before_fluorescence,
        final_error=final_error,
        component_columns=component_columns,
        component_names=component_names,
        metadata=metadata,
    )


def _args_for_recipe(args: argparse.Namespace, recipe: ExtractionRecipe) -> argparse.Namespace:
    recipe_args = argparse.Namespace(**vars(args))
    recipe_args.sample_group = recipe.sample_group
    recipe_args.water_group = recipe.water_group
    recipe_args.air_group = recipe.air_group
    recipe_args.empty_group = recipe.empty_group
    recipe_args.gc_group = recipe.gc_group
    recipe_args.asaxs_output_name = recipe.output_name
    return recipe_args


def final_outputs_for_recipe(
    averages: list[GroupAverage],
    args: argparse.Namespace,
    output_dir: Path,
    recipe: ExtractionRecipe,
    *,
    write_text: bool,
    geometry_metadata: Mapping[str, object] | None = None,
) -> list[FinalOutput]:
    recipe_args = _args_for_recipe(args, recipe)
    if recipe_args.sample_group is None:
        return []
    lookup = group_lookup(averages)
    energy_indices = sorted({avg.energy_index for avg in averages})
    outputs: list[FinalOutput] = []
    gc_reference_file = Path(recipe_args.gc_reference_file).expanduser().resolve() if recipe_args.gc_reference_file else None
    records = [
        build_final_record(energy_index, lookup, recipe_args, gc_reference_file, geometry_metadata)
        for energy_index in energy_indices
    ]

    fluorescence_level = None
    fluorescence_reference_energy_index = None
    fluorescence_reference_energy_kev = None
    if recipe_args.subtract_fluorescence:
        if recipe_args.fluorescence_level is not None:
            fluorescence_level = recipe_args.fluorescence_level
        elif recipe_args.fluorescence_reference == "latest":
            reference = records[-1]
            fluorescence_level = estimate_constant_fluorescence(
                reference.q,
                reference.final_before_fluorescence,
                tuple(recipe_args.fluorescence_q_range),
            )
            fluorescence_reference_energy_index = reference.energy_index
            fluorescence_reference_energy_kev = reference.energy_kev
        print(
            "Fluorescence subtraction: "
            f"output={recipe.output_name}, reference={recipe_args.fluorescence_reference}, q_range={tuple(recipe_args.fluorescence_q_range)}, "
            f"level={fluorescence_level}"
        )

    for record in records:
        if recipe_args.subtract_fluorescence:
            if recipe_args.fluorescence_reference == "each" and recipe_args.fluorescence_level is None:
                fluorescence_level_one = estimate_constant_fluorescence(
                    record.q,
                    record.final_before_fluorescence,
                    tuple(recipe_args.fluorescence_q_range),
                )
                fluorescence_reference_energy_index_one = record.energy_index
                fluorescence_reference_energy_kev_one = record.energy_kev
            else:
                fluorescence_level_one = fluorescence_level
                fluorescence_reference_energy_index_one = fluorescence_reference_energy_index
                fluorescence_reference_energy_kev_one = fluorescence_reference_energy_kev
            final_curve = record.final_before_fluorescence - fluorescence_level_one
        else:
            fluorescence_level_one = None
            fluorescence_reference_energy_index_one = None
            fluorescence_reference_energy_kev_one = None
            final_curve = record.final_before_fluorescence

        component_columns = list(record.component_columns)
        component_names = list(record.component_names)
        if recipe_args.subtract_fluorescence:
            component_columns.append(np.full_like(record.q, fluorescence_level_one, dtype=float))
            component_names.append("I_fluorescence_background")
        component_columns.extend([final_curve, record.final_error])
        component_names.extend(["I_final", "I_final_err"])

        metadata = dict(record.metadata)
        metadata.update(
            {
                "output_name": recipe.output_name,
                "fluorescence_background": fluorescence_level_one,
                "fluorescence_q_range": list(recipe_args.fluorescence_q_range),
                "fluorescence_reference": recipe_args.fluorescence_reference if recipe_args.subtract_fluorescence else None,
                "fluorescence_reference_energy_index": fluorescence_reference_energy_index_one,
                "fluorescence_reference_energy_kev": fluorescence_reference_energy_kev_one,
            }
        )

        path = output_dir / "final" / recipe.output_name / f"energy_{record.energy_index:03d}_{recipe.output_name}_final.dat"
        component_path = output_dir / "components" / recipe.output_name / f"energy_{record.energy_index:03d}_{recipe.output_name}_components.dat"
        if write_text:
            path.parent.mkdir(parents=True, exist_ok=True)
            header = "\n".join(
                [
                    f"ASWAXS final per-energy sample curve: {recipe.output_name}",
                    *energy_header_lines(record.energy_kev),
                    "metadata_json=" + json.dumps(metadata, sort_keys=True),
                    "columns=q I_final I_final_err",
                ]
            )
            np.savetxt(path, np.column_stack([record.q, final_curve, record.final_error]), header=header, comments="#")
            component_path.parent.mkdir(parents=True, exist_ok=True)
            component_header = "\n".join(
                [
                    f"ASWAXS final per-energy sample curve with intermediate components: {recipe.output_name}",
                    *energy_header_lines(record.energy_kev),
                    "metadata_json=" + json.dumps(metadata, sort_keys=True),
                    "columns=" + " ".join(component_names),
                ]
            )
            np.savetxt(component_path, np.column_stack(component_columns), header=component_header, comments="#")
        outputs.append(
            FinalOutput(
                path=path,
                component_path=component_path,
                output_name=recipe.output_name,
                energy_index=record.energy_index,
                energy_kev=record.energy_kev,
                q=record.q,
                I=final_curve,
                sigma_I=record.final_error,
                component_names=component_names,
                component_columns=component_columns,
                metadata=metadata,
            )
        )
    return outputs


def write_final_sample_outputs(
    averages: list[GroupAverage],
    args: argparse.Namespace,
    output_dir: Path,
    geometry_metadata: Mapping[str, object] | None = None,
) -> list[FinalOutput]:
    outputs: list[FinalOutput] = []
    for recipe in extraction_recipes(args):
        outputs.extend(
            final_outputs_for_recipe(
                averages,
                args,
                output_dir,
                recipe,
                write_text=True,
                geometry_metadata=geometry_metadata,
            )
        )
    return outputs


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cpu_max = max(1, os.cpu_count() or 1)
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1.")
    if args.jobs > cpu_max:
        print(f"Requested jobs={args.jobs} exceeds system CPU count {cpu_max}; using {cpu_max}.")
        args.jobs = cpu_max

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_manifest(args, output_dir)
    args.manifest = str(manifest_path)
    items = read_manifest(manifest_path)
    validate_asaxs_group_roles(args)
    detector = infer_detector(items, args.detector)
    monitor_key = args.monitor_key or default_monitor_key(detector)

    print("ASWAXS sequence reduction")
    print(f"Manifest: {manifest_path}")
    print(f"Detector: {detector}")
    print(f"Monitor normalization key: {monitor_key}")
    print(f"Jobs: {args.jobs}")
    print("Order: integrate frames -> reject bad frames by total intensity -> normalize kept frames -> average groups")

    geometry_metadata = poni_geometry_metadata(args.poni)
    averages = reduce_manifest_frames_parallel(items, args, monitor_key)
    try:
        for avg in averages:
            write_group_average(avg, output_dir, geometry_metadata)
        summary_path = write_summary(averages, output_dir, monitor_key, detector, args)
        final_outputs = write_final_sample_outputs(averages, args, output_dir, geometry_metadata)
        analysis_path = Path(args.analysis_h5).expanduser().resolve() if args.analysis_h5 else output_dir / "analysis.h5"
        _write_sequence_analysis_h5(
            analysis_path=analysis_path,
            manifest_path=manifest_path,
            items=items,
            averages=averages,
            final_outputs=final_outputs,
            args=args,
            monitor_key=monitor_key,
            detector=detector,
            summary_path=summary_path,
            geometry_metadata=geometry_metadata,
        )
    finally:
        cleanup_frame_stability_shards(averages)

    print(f"Wrote group summary: {summary_path}")
    print(f"Wrote {len(averages)} group-average files.")
    print(f"Wrote analysis HDF5: {analysis_path}")
    if final_outputs:
        print(f"Wrote {len(final_outputs)} final sample files.")
    else:
        print("No final sample files written because --sample-group was not provided.")
    return 0


def _write_sequence_analysis_h5(
    analysis_path: Path,
    manifest_path: Path,
    items: list[ManifestItem],
    averages: list[GroupAverage],
    final_outputs: list[FinalOutput],
    args: argparse.Namespace,
    monitor_key: str,
    detector: str,
    summary_path: Path,
    geometry_metadata: Mapping[str, object] | None = None,
) -> None:
    """Mirror the sequence reduction products into the structured analysis HDF5."""
    raw_paths = [item.path for item in items]
    data_reference_metadata = {
        "data_detector_path": args.dataset_path,
        "source_frame_indices": [item.sequence_index for item in items],
        "source_frame_count": len(items),
        "notes": f"ASWAXS sequence from manifest {manifest_path}; source data files opened read-only",
    }
    needs_data_reference = not Path(analysis_path).exists()
    if not needs_data_reference:
        with h5py.File(analysis_path, "r") as handle:
            needs_data_reference = "/entry/data_reference" not in handle
    if needs_data_reference:
        create_analysis_h5_from_data(raw_paths, analysis_path, data_reference_metadata=data_reference_metadata)

    q = stack_ragged_rows([avg.q for avg in averages]) if averages else np.asarray([])
    reduction_i = stack_ragged_rows([avg.avg_intensity for avg in averages]) if averages else np.empty((0, 0))
    reduction_sigma = stack_ragged_rows([avg.avg_error for avg in averages]) if averages else np.empty((0, 0))
    frame_log = _frame_filter_log_from_averages(averages, items, args.outlier_zmax)
    reduction_metadata = {
        "input_h5_file": str(manifest_path),
        "input_data_path": args.dataset_path,
        "output_h5_file": str(analysis_path),
        "output_data_path": "/entry/process_01_reduction/data",
        "energy": np.asarray([np.nan if avg.energy_kev is None else avg.energy_kev for avg in averages], dtype=float),
        "energy_index": np.asarray([avg.energy_index for avg in averages], dtype=int),
        "group_index": np.asarray([avg.group_index for avg in averages], dtype=int),
        "n_total_frames": len(items),
        "n_accepted_frames": int(sum(avg.kept_count for avg in averages)),
        "n_rejected_frames": int(sum(avg.dropped_count for avg in averages)),
        "notes": "sequence reduction: integrate frames, monitor normalize, reject outliers, average by energy/group",
    }
    if geometry_metadata:
        reduction_metadata.update(geometry_metadata)
    reduction_parameters = {
        "poni_file": str(args.poni),
        "poni_file_hash": file_sha256(args.poni),
        "mask_file": str(args.mask),
        "mask_file_hash": file_sha256(args.mask),
        "q_unit": args.unit,
        "q_min": float(np.nanmin(q)) if q.size else "unknown",
        "q_max": float(np.nanmax(q)) if q.size else "unknown",
        "n_q_bins": args.npt,
        "integration_method": "pyFAI.integrate1d",
        "normalization_method": f"monitor:{monitor_key}",
        "dark_subtraction": False,
        "flatfield_correction": False,
        "solid_angle_correction": "pyFAI_default",
        "polarization_correction": "unknown",
        "error_model": "raw Poisson/counting uncertainty propagated through pyFAI integration and monitor normalization; frame scatter used when multiple accepted frames are available",
        "detector": detector,
        "outlier_zmax": args.outlier_zmax,
    }
    reduction_process_path = write_reduction_to_analysis_h5(
        analysis_path,
        raw_paths[0],
        q,
        reduction_i,
        reduction_sigma,
        reduction_metadata,
        reduction_parameters,
        frame_filter_log=frame_log,
    )
    write_frame_stability_results(analysis_path, reduction_process_path, detector, averages)

    if not final_outputs:
        return

    _write_named_asaxs_outputs(analysis_path, final_outputs)
    _write_legacy_final_group(analysis_path, final_outputs)
    primary_name = final_outputs[0].output_name
    primary_outputs = [item for item in final_outputs if item.output_name == primary_name]
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
        "input_h5_file": str(analysis_path),
        "input_data_path": "/entry/process_01_reduction/data",
        "output_h5_file": str(analysis_path),
        "output_data_path": "/entry/process_02_background_subtraction/data",
        "notes": f"group summary CSV retained at {summary_path}",
    }
    subtraction_parameters = {
        "gc_background": "air" if args.air_group is not None else "unknown",
        "sample_background": "empty_cell/solvent" if args.empty_group or args.water_group else "unknown",
        "scale_by_I0": True,
        "scale_by_transmission": False,
        "scale_by_exposure_time": False,
        "subtraction_formula": "sample-empty; water-empty; sample_corrected-water_corrected; gc-air",
        "solvent_scale_factor": 1.0,
        "empty_cell_scale_factor": 1.0,
    }
    subtraction_map = {
        "energy": energies,
        "sample_id": primary_outputs[0].metadata.get("sample_group", args.sample_group),
        "air_id": args.air_group if args.air_group is not None else "unknown",
        "glassy_carbon_id": args.gc_group if args.gc_group is not None else "unknown",
        "empty_cell_id": args.empty_group if args.empty_group is not None else "unknown",
        "solvent_id": primary_outputs[0].metadata.get("water_group", args.water_group) if primary_outputs else "unknown",
        "output_name": primary_name,
    }
    write_background_subtraction_to_analysis_h5(
        analysis_path,
        corrected_data,
        subtraction_metadata,
        subtraction_parameters,
        subtraction_map,
    )

    if args.gc_group is None:
        return

    normalized_data = {
        "q": final_q,
        "energy": energies,
        "I_sample_normalized": final_i,
        "sigma_sample_normalized": final_sigma,
        "I_gc_normalized": _stack_component(component_lookup, "I_gc_minus_air"),
        "sigma_gc_normalized": _stack_component(component_lookup, "I_gc_minus_air_err"),
    }
    normalization_metadata = {
        "input_h5_file": str(analysis_path),
        "input_data_path": "/entry/process_02_background_subtraction/data",
        "output_h5_file": str(analysis_path),
        "output_data_path": "/entry/process_03_glassy_carbon_normalization/data",
    }
    normalization_parameters = {
        "gc_reference_file": str(args.gc_reference_file) if args.gc_reference_file else "NIST_SRM3600_builtin",
        "gc_reference_file_hash": file_sha256(args.gc_reference_file) if args.gc_reference_file else "builtin",
        "reference_units": "differential_scattering_cross_section",
        "q_range_requested": list(args.gc_q_range),
        "q_range_used": "per-energy in /normalization_factors/q_min_used and q_max_used",
        "scale_method": "XAnoS_CF_from_integrated_area_ratio_recorded_not_applied",
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
    write_glassy_carbon_normalization_to_analysis_h5(
        analysis_path,
        normalized_data,
        normalization_metadata,
        normalization_parameters,
        normalization_factors,
    )


def _frame_filter_log_from_averages(
    averages: list[GroupAverage],
    items: list[ManifestItem],
    outlier_zmax: float,
) -> dict[str, np.ndarray]:
    accepted_sequences = {
        sequence_index
        for average in averages
        for sequence_index in average.kept_sequence_indices
    }
    rejected_sequences = {
        sequence_index
        for average in averages
        for sequence_index in average.dropped_sequence_indices
    }
    average_metric = {
        sequence_index: average.avg_total_intensity
        for average in averages
        for sequence_index in average.kept_sequence_indices
    }
    sequence_index: list[int] = []
    energy_index: list[int] = []
    group_index: list[int] = []
    frame_index: list[int] = []
    source_file: list[str] = []
    accepted: list[bool] = []
    reason: list[str] = []
    metric: list[float] = []
    low: list[float] = []
    high: list[float] = []
    for item in sorted(items, key=lambda value: value.sequence_index):
        is_accepted = item.sequence_index in accepted_sequences
        is_rejected = item.sequence_index in rejected_sequences
        sequence_index.append(item.sequence_index)
        energy_index.append(item.energy_index)
        group_index.append(item.group_index)
        frame_index.append(item.frame_index)
        source_file.append(str(item.path))
        accepted.append(is_accepted)
        reason.append("total_intensity_outlier" if is_rejected else "")
        metric.append(average_metric.get(item.sequence_index, np.nan))
        low.append(np.nan)
        high.append(outlier_zmax)
    return {
        "sequence_index": np.asarray(sequence_index, dtype=int),
        "energy_index": np.asarray(energy_index, dtype=int),
        "group_index": np.asarray(group_index, dtype=int),
        "frame_index": np.asarray(frame_index, dtype=int),
        "source_file": np.asarray(source_file),
        "accepted": np.asarray(accepted, dtype=bool),
        "rejection_reason": np.asarray(reason),
        "metric_value": np.asarray(metric, dtype=float),
        "threshold_low": np.asarray(low, dtype=float),
        "threshold_high": np.asarray(high, dtype=float),
    }


def _write_named_asaxs_outputs(analysis_path: Path, final_outputs: list[FinalOutput]) -> None:
    """Store multiple named sample/solvent extractions in one analysis HDF5."""
    grouped: dict[str, list[FinalOutput]] = {}
    for item in final_outputs:
        grouped.setdefault(item.output_name, []).append(item)
    with h5py.File(analysis_path, "a") as handle:
        entry = handle.require_group("entry")
        if "asaxs_outputs" in entry:
            del entry["asaxs_outputs"]
        root = entry.create_group("asaxs_outputs")
        for output_name, outputs in sorted(grouped.items()):
            outputs = sorted(outputs, key=lambda item: item.energy_index)
            group = root.create_group(safe_output_name(output_name))
            group.attrs["NX_class"] = "NXcollection"
            group.attrs["output_name"] = output_name
            data = group.create_group("corrected_I_q_E")
            data.attrs["NX_class"] = "NXdata"
            data.attrs["signal"] = "I"
            data.attrs["axes"] = np.asarray(["energy", "q"], dtype=h5py.string_dtype("utf-8"))
            data.create_dataset("q", data=stack_ragged_rows([item.q for item in outputs]))
            data.create_dataset("energy", data=np.asarray([np.nan if item.energy_kev is None else item.energy_kev for item in outputs], dtype=float))
            data.create_dataset("energy_index", data=np.asarray([item.energy_index for item in outputs], dtype=int))
            data.create_dataset("I", data=stack_ragged_rows([item.I for item in outputs]))
            data.create_dataset("sigma_I", data=stack_ragged_rows([item.sigma_I for item in outputs]))
            data.create_dataset("sample_group", data=np.asarray([item.metadata.get("sample_group", -1) for item in outputs], dtype=int))
            data.create_dataset("solvent_group", data=np.asarray([item.metadata.get("water_group", -1) or -1 for item in outputs], dtype=int))
            data.create_dataset("xanos_calibration_factor", data=np.asarray([item.metadata.get("xanos_calibration_factor", np.nan) for item in outputs], dtype=float))
            data.create_dataset("sample_thickness", data=np.asarray([item.metadata.get("sample_thickness", np.nan) for item in outputs], dtype=float))
            data.create_dataset("fluorescence_background", data=np.asarray([item.metadata.get("fluorescence_background", 0.0) or 0.0 for item in outputs], dtype=float))
            metadata = group.create_group("metadata")
            for key in (
                "output_name",
                "sample_group",
                "water_group",
                "empty_group",
                "air_group",
                "gc_group",
                "sample_thickness",
                "gc_standard_thickness",
                "xanos_calibration_factor",
                "absolute_scale_factor",
                "fluorescence_background",
                "xanos_scale_formula",
            ):
                value = outputs[0].metadata.get(key, "unknown")
                if value is None:
                    value = "unknown"
                metadata.create_dataset(key, data=str(value), dtype=h5py.string_dtype("utf-8"))


def _write_legacy_final_group(analysis_path: Path, final_outputs: list[FinalOutput]) -> None:
    """Mirror the first named output to /entry/final for older viewers/tools."""
    if not final_outputs:
        return
    primary_name = final_outputs[0].output_name
    outputs = [item for item in final_outputs if item.output_name == primary_name]
    outputs = sorted(outputs, key=lambda item: item.energy_index)
    with h5py.File(analysis_path, "a") as handle:
        entry = handle.require_group("entry")
        if "final" in entry:
            del entry["final"]
        final = entry.create_group("final")
        group = final.create_group("corrected_I_q_E")
        group.attrs["NX_class"] = "NXdata"
        group.attrs["signal"] = "I"
        group.attrs["axes"] = np.asarray(["energy", "q"], dtype=h5py.string_dtype("utf-8"))
        group.attrs["output_name"] = primary_name
        group.create_dataset("q", data=stack_ragged_rows([item.q for item in outputs]))
        group.create_dataset("energy", data=np.asarray([np.nan if item.energy_kev is None else item.energy_kev for item in outputs], dtype=float))
        group.create_dataset("I", data=stack_ragged_rows([item.I for item in outputs]))
        group.create_dataset("sigma_I", data=stack_ragged_rows([item.sigma_I for item in outputs]))


def _component_dict(names: list[str], columns: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(column) for name, column in zip(names, columns)}


def _stack_component(rows: list[dict[str, np.ndarray]], name: str) -> np.ndarray:
    if not rows:
        return np.asarray([])
    fallback = np.full_like(rows[0].get("q", np.asarray([])), np.nan, dtype=float)
    return stack_ragged_rows([row.get(name, fallback) for row in rows])


if __name__ == "__main__":
    raise SystemExit(main())

