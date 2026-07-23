"""Live-style ASWAXS pipeline V5.

The current production reducer works as a batch program: read a sequence, reduce
all frames, average every group, then write final per-energy products. This
version keeps that science code and changes only the orchestration. A manifest
can be replayed one row at a time, a detector folder can be watched directly, or
a Bluesky-assisted JSONL queue can wake the same folder watcher:

    frame arrives -> 1D reduction -> group average -> per-energy correction

The raw HDF5 files are opened read-only. Reduction state, provenance, and live
history are written to the analysis HDF5 file and live_events.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import io
import json
import os
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
import time

import h5py
import numpy as np

from aswaxs_live.workflows.bluesky_queue import ReductionJobQueueReader
from aswaxs_live.paths import PACKAGE_DIR, PLAYGROUND_DIR, PROJECT_DIR
from aswaxs_live.reduction.xanos_export import export_analysis_h5_to_xanos_format

HDF5_RUNTIME_ERROR_MARKERS = (
    "hdf5",
    "bad heap",
    "heap free list",
    "addr overflow",
    "unable to",
    "link iteration failed",
    "object header",
)


def data_file_sort_key(path: Path) -> tuple[tuple[int, ...], list[object], str]:
    """Sort acquisition HDF5 files by the numeric counter in the filename."""
    parts = re.split(r"(\d+)", path.name)
    numbers = tuple(int(part) for part in parts if part.isdigit())
    natural = [int(part) if part.isdigit() else part.lower() for part in parts]
    return numbers, natural, path.name.lower()


def is_hdf5_access_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in HDF5_RUNTIME_ERROR_MARKERS)


def quarantine_hdf5(path: Path, reason: BaseException | str) -> Path | None:
    """Move a damaged analysis HDF5 aside so resume can continue with a fresh file."""
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = path.with_name(f"{path.stem}_corrupt_{stamp}{path.suffix}")
    counter = 2
    while target.exists():
        target = path.with_name(f"{path.stem}_corrupt_{stamp}_{counter:02d}{path.suffix}")
        counter += 1
    try:
        path.replace(target)
    except OSError as exc:
        print(f"Warning: could not quarantine damaged analysis HDF5 {path}: {exc}")
        return None
    print(f"Warning: moved damaged analysis HDF5 aside: {target}")
    print(f"Warning: original HDF5 access error was: {reason}")
    return target


def validate_existing_analysis_h5(path: Path) -> bool:
    """Return False when an existing analysis HDF5 has broken link metadata."""
    if not path.exists():
        return True
    try:
        with h5py.File(path, "r") as handle:
            handle.visit(lambda _name: None)
    except (OSError, RuntimeError) as exc:
        if is_hdf5_access_error(exc):
            quarantine_hdf5(path, exc)
            return False
        raise
    return True


@contextmanager
def open_h5_retry(path: Path, mode: str, attempts: int = 60, delay_seconds: float = 0.1):
    """Open an HDF5 file, retrying transient Windows/HDF5 file locks."""
    last_error: BaseException | None = None
    handle = None
    for attempt in range(attempts):
        try:
            handle = h5py.File(path, mode)
            break
        except (OSError, RuntimeError) as exc:
            if not is_hdf5_access_error(exc):
                raise
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(delay_seconds)
    if handle is None and last_error is not None:
        raise last_error
    if handle is None:
        raise OSError(f"Could not open HDF5 file: {path}")
    try:
        yield handle
    finally:
        handle.close()


def run_h5_write_retry(action, attempts: int = 60, delay_seconds: float = 0.1):
    """Run a write action that may internally open an HDF5 file."""
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return action()
        except (OSError, RuntimeError) as exc:
            if not is_hdf5_access_error(exc):
                raise
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    return None


@dataclass(frozen=True)
class LiveEvent:
    """One audit record describing what the live scheduler just did."""

    time: str
    event: str
    energy_index: int | None = None
    group_index: int | None = None
    frame_index: int | None = None
    sequence_index: int | None = None
    path: str | None = None
    message: str | None = None
    uid: str | None = None
    scan_id: int | str | None = None
    sample_name: str | None = None
    detector: str | None = None
    data_dir: str | None = None
    output_dir: str | None = None
    expected_total_frames: int | None = None
    num_energies: int | None = None
    num_groups: int | None = None
    num_frames: int | None = None
    reduce_total_seconds: float | None = None
    read_energy_seconds: float | None = None
    read_image_seconds: float | None = None
    integrate_seconds: float | None = None
    read_monitor_seconds: float | None = None
    h5_write_seconds: float | None = None
    source_file_mb: float | None = None
    image_rows: int | None = None
    image_cols: int | None = None


DETECTOR_ALIASES = {
    "pil300k": "Pil300K",
    "saxs": "Pil300K",
    "spds": "Pil300K",
    "eig1m": "Eig1M",
    "waxs": "Eig1M",
    "wpds": "Eig1M",
}


def normalize_detector_name(detector: str | None) -> str | None:
    """Map beamline aliases to canonical detector names before comparisons."""
    if detector is None:
        return None
    text = str(detector).strip()
    if not text:
        return None
    return DETECTOR_ALIASES.get(text.lower(), text)


@dataclass(frozen=True)
class SequencePosition:
    """Scientific meaning assigned to one file by acquisition order."""

    sequence_index: int
    energy_index: int
    group_index: int
    frame_index: int


class SequenceAssigner:
    """Map arriving files into energy -> group -> frame order.

    This is the watcher equivalent of the offline manifest. It assumes the
    acquisition writes files in this nested order:

        energy 1, group 1, frame 1
        energy 1, group 1, frame 2
        ...
        energy 1, group 2, frame 1
        ...
        energy 2, group 1, frame 1
    """

    def __init__(self, num_groups: int, num_frames: int, num_energies: int | None = None) -> None:
        if num_groups < 1:
            raise ValueError("--num-groups must be at least 1.")
        if num_frames < 1:
            raise ValueError("--num-frames must be at least 1.")
        if num_energies is not None and num_energies < 1:
            raise ValueError("--num-energies must be at least 1 when provided.")
        self.num_groups = num_groups
        self.num_frames = num_frames
        self.num_energies = num_energies
        self._next_sequence_index = 1

    @property
    def expected_total(self) -> int | None:
        if self.num_energies is None:
            return None
        return self.num_energies * self.num_groups * self.num_frames

    @property
    def assigned_count(self) -> int:
        return self._next_sequence_index - 1

    def is_complete(self) -> bool:
        return self.expected_total is not None and self.assigned_count >= self.expected_total

    def advance_to_sequence_index(self, next_sequence_index: int) -> None:
        """Resume assignment after frames already present in the analysis HDF5."""
        if next_sequence_index < 1:
            raise ValueError("next_sequence_index must be at least 1.")
        self._next_sequence_index = max(self._next_sequence_index, next_sequence_index)

    def next_position(self) -> SequencePosition:
        sequence_index = self._next_sequence_index
        if self.is_complete():
            raise StopIteration("All expected acquisition files have already been assigned.")

        zero_based = sequence_index - 1
        frames_per_energy = self.num_groups * self.num_frames
        energy_index = zero_based // frames_per_energy + 1
        group_index = (zero_based % frames_per_energy) // self.num_frames + 1
        frame_index = zero_based % self.num_frames + 1
        self._next_sequence_index += 1
        return SequencePosition(sequence_index, energy_index, group_index, frame_index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FrameByFrame-ASWAXS live orchestration from replay, folder watch, or Bluesky-assisted queue.",
    )
    parser.add_argument("--manifest", help="Existing sequence_manifest.csv to replay.")
    parser.add_argument("--watch-dir", help="Directory receiving live HDF5 files from acquisition.")
    parser.add_argument(
        "--frame-done-queue",
        default=None,
        help=(
            "Deprecated alias for --measurement-done-queue."
        ),
    )
    parser.add_argument(
        "--measurement-done-queue",
        default=None,
        help=(
            "JSONL reduction job queue written from Bluesky/Kafka measurement_done messages. "
            "Each job supplies uid, scan_id, sample_name, detector, and data_dir."
        ),
    )
    parser.add_argument(
        "--continuous-queue",
        action="store_true",
        help="Queue mode: keep running and reset for the next queued sample after one batch completes.",
    )
    parser.add_argument(
        "--stop-when-queue-drained",
        action="store_true",
        help="Queue mode: exit after all currently queued jobs for this detector finish.",
    )
    parser.add_argument(
        "--sample-name",
        default=None,
        help="Sample/run name used for the default analysis HDF5 filename.",
    )
    parser.add_argument("--pattern", default="*.h5", help="Watcher input filename pattern. Default: *.h5")
    parser.add_argument(
        "--recursive-watch",
        action="store_true",
        help="Watcher/job mode: scan subfolders too. V3 does not enable this automatically.",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Watcher polling interval.")
    parser.add_argument("--settle-seconds", type=float, default=2.0, help="File-size stability wait.")
    parser.add_argument("--once", action="store_true", help="Watcher: process current files once, then exit.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing analysis HDF5/event log when present. Default: resume.",
    )
    parser.add_argument(
        "--restart",
        dest="resume",
        action="store_false",
        help="Restart from scratch: overwrite the analysis HDF5 and replace the live event log.",
    )
    parser.add_argument("--num-energies", type=int, default=None, help="Watcher: expected number of energies.")
    parser.add_argument("--num-groups", type=int, default=None, help="Watcher: expected groups per energy.")
    parser.add_argument(
        "--auto-num-groups",
        action="store_true",
        help=(
            "Infer groups per energy from the number of HDF5 files in each watch/job folder. "
            "Requires --num-energies and --num-frames; best for offline finished data or measurement_done queue jobs."
        ),
    )
    parser.add_argument("--num-frames", type=int, default=None, help="Watcher: expected frames per group.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "outputs" / "live_v5_demo"),
        help="Output directory for analysis HDF5 and live_events.jsonl.",
    )
    parser.add_argument(
        "--analysis-h5",
        default=None,
        help="Analysis HDF5 output path. Default: output-dir/<sample_name>_analysis.h5.",
    )
    parser.add_argument(
        "--analysis-mode",
        default="asaxs",
        choices=["asaxs", "saxs"],
        help="asaxs: run background/GC/final correction after each energy. saxs: stop at 1D + group averages.",
    )
    parser.add_argument(
        "--write-text-output",
        action="store_true",
        help="Also write legacy .dat curve files. Default is HDF5-only for reduction curves.",
    )
    parser.add_argument("--poni", required=True, help="pyFAI PONI calibration file.")
    parser.add_argument("--mask", required=True, help="Detector mask file.")
    parser.add_argument("--dataset-path", default="entry/data/data", help="Detector image dataset inside each HDF5 file.")
    parser.add_argument("--npt", type=int, default=1000, help="Number of q bins for 1D integration.")
    parser.add_argument("--jobs", type=int, default=1, help="Worker/core count requested for reduction stages. Default: 1.")
    parser.add_argument(
        "--analysis-write-interval-groups",
        type=int,
        default=1,
        help="Live mode: rewrite the full structured analysis HDF5 after this many new group averages. Default: 1.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce terminal chatter; live_events.jsonl remains the detailed log.")
    parser.add_argument("--unit", default="q_A^-1", help="pyFAI radial unit.")
    parser.add_argument("--detector", default="auto", choices=["auto", "Pil300K", "Eig1M"])
    parser.add_argument("--monitor-key", default=None, help="NDAttribute monitor key. Default depends on detector.")
    parser.add_argument("--delta-energy-percent", type=float, default=1e-3)
    parser.add_argument("--outlier-zmax", type=float, default=3.5)
    parser.add_argument("--gc-group", type=int, default=None)
    parser.add_argument("--air-group", type=int, default=None)
    parser.add_argument("--empty-group", type=int, default=None)
    parser.add_argument("--water-group", type=int, default=None)
    parser.add_argument("--sample-group", type=int, default=None)
    parser.add_argument("--asaxs-output-name", default="sample")
    parser.add_argument("--asaxs-pair", action="append", default=[])
    parser.add_argument("--asaxs-extraction-plan", default=None)
    parser.add_argument("--gc-reference-file", default=None)
    parser.add_argument("--gc-q-range", nargs=2, type=float, default=[0.03, 0.20])
    parser.add_argument("--capillary-thickness", type=float, default=None)
    parser.add_argument("--gc-thickness", type=float, default=None)
    parser.add_argument("--subtract-fluorescence", action="store_true")
    parser.add_argument("--fluorescence-level", type=float, default=None)
    parser.add_argument("--fluorescence-reference", default="latest", choices=["latest", "each"])
    parser.add_argument("--fluorescence-q-range", nargs=2, type=float, default=[0.8, 1.0])
    parser.add_argument("--limit-energies", type=int, default=None, help="Use only the first N energies for a quick test.")
    parser.add_argument(
        "--limit-frames-per-group",
        type=int,
        default=None,
        help="Use only the first N frames per energy/group for a quick test.",
    )
    return parser


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_event(handle, event: LiveEvent) -> None:
    """Append a machine-readable scheduler event and flush for live tailing."""
    if hasattr(handle, "name"):
        try:
            Path(handle.name).parent.mkdir(parents=True, exist_ok=True)
        except (OSError, TypeError, ValueError):
            pass
    handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
    handle.flush()


@contextmanager
def open_event_log(path: Path, mode: str):
    """Open a live event log, creating parents and retrying network races."""
    path = Path(path)
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open(mode, encoding="utf-8") as handle:
                yield handle
            return
        except FileNotFoundError as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(0.2)
    if last_error is not None:
        raise last_error


def file_is_ready(path: Path, dataset_path: str, settle_seconds: float) -> tuple[bool, str]:
    """Return True only when the detector file is stable and readable.

    The raw HDF5 file is opened read-only here. That is the same rule the
    reducer uses, and it avoids touching a file while the detector is still
    writing it.
    """
    if not path.exists() or not path.is_file():
        return False, "missing"
    first_size = path.stat().st_size
    if first_size <= 0:
        return False, "empty"
    if settle_seconds > 0:
        time.sleep(settle_seconds)
        if not path.exists() or path.stat().st_size != first_size:
            return False, "size-changing"
    try:
        with h5py.File(path, "r") as handle:
            if dataset_path not in handle:
                return False, f"missing dataset {dataset_path}"
            dataset = handle[dataset_path]
            if dataset.size == 0:
                return False, f"empty dataset {dataset_path}"
    except OSError as exc:
        return False, f"hdf5 not readable yet: {exc}"
    return True, "ready"


def watch_candidates(scan_dir: Path, pattern: str, recursive: bool, output_dir: Path) -> list[Path]:
    """Return candidate raw HDF5 files without following the output folder.

    SAXS mode can watch a broad sample/project folder recursively. This helper
    keeps generated analysis HDF5 files out of the input list when users place
    outputs under or near the watched tree.
    """
    iterator = scan_dir.rglob(pattern) if recursive else scan_dir.glob(pattern)
    output_root = output_dir.expanduser().resolve()
    candidates: list[Path] = []
    for path in iterator:
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(output_root)
            continue
        except ValueError:
            candidates.append(resolved)
    return sorted(candidates, key=data_file_sort_key)


def infer_num_groups_from_folder(
    scan_dir: Path,
    args: argparse.Namespace,
    output_dir: Path,
    recursive: bool,
) -> int | None:
    """Infer group count from a completed detector folder.

    The sequence model is energy -> group -> frame, so:
    file_count = num_energies * num_groups * num_frames.
    This is intentionally conservative; if the count is not divisible, the
    reducer keeps waiting in live/queue mode instead of guessing.
    """
    if getattr(args, "num_groups", None):
        return int(args.num_groups)
    if not getattr(args, "auto_num_groups", False):
        return None
    if args.num_energies is None or args.num_frames is None:
        raise ValueError("--auto-num-groups requires --num-energies and --num-frames.")
    denominator = int(args.num_energies) * int(args.num_frames)
    if denominator <= 0:
        raise ValueError("--auto-num-groups requires positive --num-energies and --num-frames.")
    file_count = len(watch_candidates(scan_dir, args.pattern, recursive, output_dir))
    if file_count == 0:
        return None
    if file_count % denominator != 0:
        return None
    inferred = file_count // denominator
    return inferred if inferred > 0 else None


def auto_group_wait_message(scan_dir: Path, args: argparse.Namespace, output_dir: Path, recursive: bool) -> str:
    """Explain why auto group inference cannot proceed yet."""
    file_count = len(watch_candidates(scan_dir, args.pattern, recursive, output_dir))
    energies = int(args.num_energies) if args.num_energies is not None else 0
    frames = int(args.num_frames) if args.num_frames is not None else 0
    denominator = energies * frames
    if file_count == 0:
        return "waiting for HDF5 files"
    if denominator <= 0:
        return "cannot infer groups because energies or frames/group is not positive"
    remainder = file_count % denominator
    return (
        f"{file_count} HDF5 files is not divisible by energies * frames/group "
        f"({energies} * {frames} = {denominator}, remainder {remainder}). "
        "The frame number or energy count may be wrong, or extra/incomplete files are present."
    )


def offline_complete_block_plan(args: argparse.Namespace, file_count: int) -> tuple[int | None, int, int, int]:
    """Return the file limit for complete offline blocks.

    In offline ``--once`` mode a folder may contain complete repeated sets plus
    a tail from an aborted/incomplete set. The configured sequence
    ``num_energies * num_groups * num_frames`` defines one complete block. Only
    whole blocks are reduced; leftover files are ignored for a future rerun.
    """
    if not getattr(args, "once", False):
        return None, 0, 0, 0
    if args.num_energies is None or args.num_groups is None or args.num_frames is None:
        return None, 0, 0, 0
    block_size = int(args.num_energies) * int(args.num_groups) * int(args.num_frames)
    if block_size <= 0:
        return None, 0, 0, 0
    complete_blocks = file_count // block_size
    remainder = file_count % block_size
    return complete_blocks * block_size, block_size, complete_blocks, remainder


def inferred_sample_name(args: argparse.Namespace) -> str:
    """Pick a useful sample name for output filenames.

    In live mode the watched directory often points at a detector folder such as
    ``.../<sample>/Eig1M``. In that case the parent folder is the sample name.
    """
    if args.sample_name:
        return sanitize_name(args.sample_name)
    if args.watch_dir:
        watch_dir = Path(args.watch_dir)
        detector_names = {"eig1m", "pil300k", "saxs", "waxs"}
        if watch_dir.name.lower() in detector_names and watch_dir.parent.name:
            return sanitize_name(watch_dir.parent.name)
        if watch_dir.name:
            return sanitize_name(watch_dir.name)
    if getattr(args, "measurement_done_queue", None) or getattr(args, "frame_done_queue", None):
        return "bluesky_live"
    if args.manifest:
        return sanitize_name(Path(args.manifest).stem.replace("sequence_manifest", "analysis"))
    return "analysis"


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "analysis"


def default_analysis_h5_path(args: argparse.Namespace, output_dir: Path) -> Path:
    name = inferred_sample_name(args)
    if name.lower().endswith("_analysis"):
        filename = f"{name}.h5"
    else:
        filename = f"{name}_analysis.h5"
    return output_dir / filename


def queue_job_runtime_args(args: argparse.Namespace, job: object, requested_detector: str | None) -> argparse.Namespace:
    """Clone reducer args with the queued sample/detector as the output identity."""
    job_args = argparse.Namespace(**vars(args))
    if getattr(args, "measurement_done_queue", None) or getattr(args, "frame_done_queue", None):
        job_args.settle_seconds = 0.0
        job_args.poll_seconds = min(float(getattr(job_args, "poll_seconds", 0.25)), 0.25)
    detector = normalize_detector_name(getattr(job, "detector", None)) or requested_detector
    if detector == "auto" or detector is None:
        detector = normalize_detector_name(Path(getattr(job, "data_dir")).name) or detector
    sample_name = getattr(job, "sample_name", None) or Path(getattr(job, "data_dir")).parent.name
    safe_sample = sanitize_name(str(sample_name))
    if detector in {"Pil300K", "Eig1M"}:
        job_args.sample_name = f"{safe_sample}_{detector}"
        job_args.detector = detector
    else:
        job_args.sample_name = safe_sample
    analysis_mode = getattr(job, "analysis_mode", None)
    if analysis_mode in {"saxs", "asaxs"}:
        job_args.analysis_mode = analysis_mode
    for field in ("num_energies", "num_groups", "num_frames"):
        value = getattr(job, field, None)
        if value is not None:
            setattr(job_args, field, int(value))
    job_args.analysis_h5 = None
    return job_args


def queue_job_sequence_message(base_args: argparse.Namespace, job_args: argparse.Namespace, job: object) -> str:
    """Describe the effective sequence values for a queued measurement."""
    parts = []
    for field, label in [
        ("num_frames", "frames/group"),
        ("num_groups", "groups/energy"),
        ("num_energies", "energies"),
    ]:
        value = getattr(job_args, field, None)
        source = "message" if getattr(job, field, None) is not None else "GUI"
        if getattr(job, field, None) is not None and getattr(base_args, field, None) != value:
            source = "message override"
        parts.append(f"{label}={value} ({source})")
    return "Sequence settings: " + ", ".join(parts)


def expected_total_frames_for_args(args: argparse.Namespace) -> int | None:
    """Return the planned detector-frame count when the sequence is finite."""
    if args.num_energies is None or args.num_groups is None or args.num_frames is None:
        return None
    return int(args.num_energies) * int(args.num_groups) * int(args.num_frames)


def multicore_status_message(args: argparse.Namespace) -> str:
    """Describe the current V5 worker behavior for the process monitor."""
    requested = max(1, int(getattr(args, "jobs", 1) or 1))
    cpu_max = max(1, os.cpu_count() or 1)
    effective = 1
    detail = (
        "V5 live/replay currently reduces frames serially inside each detector reducer; "
        "paired Pil300K+Eig1M runs as two separate reducer processes. "
        "Per-frame timing is recorded in live_events.jsonl and the analysis HDF5 frame table."
    )
    if requested > effective:
        detail += " Extra requested cores are not used for live frame integration yet."
    return f"CPU workers: requested={requested}, system={cpu_max}, effective_this_process={effective}. {detail}"


def queue_job_output_dir(args: argparse.Namespace, job: object, startup_output_dir: Path, requested_detector: str | None) -> Path:
    """Return the output folder for one queued sample/detector job.

    Older or external queue messages may omit ``output_dir``. In sample-list
    dual-detector mode the reducer startup output folder points at the first
    sample, so falling back to it can put the final queued sample in the wrong
    folder. Derive ``Extracted/<sample>/<detector>`` from the job identity
    instead.
    """
    explicit = getattr(job, "output_dir", None)
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    detector = normalize_detector_name(getattr(job, "detector", None)) or requested_detector
    if detector == "auto" or detector is None:
        detector = normalize_detector_name(Path(getattr(job, "data_dir")).name) or "auto"
    sample_name = getattr(job, "sample_name", None) or Path(getattr(job, "data_dir")).parent.name
    sample_folder = sanitize_name(str(sample_name))

    root = startup_output_dir.expanduser().resolve()
    if normalize_detector_name(root.name) in {"Pil300K", "Eig1M"}:
        root = root.parent
    if root.parent.name.lower() in {"extracted", "outputs"} and root.name != sample_folder:
        root = root.parent
    if root.name == sample_folder:
        root = root.parent
    if detector in {"Pil300K", "Eig1M"}:
        return root / sample_folder / detector
    return root / sample_folder


def analysis_h5_path_for_args(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.analysis_h5:
        return Path(args.analysis_h5).expanduser().resolve()
    return default_analysis_h5_path(args, output_dir)


def prepare_output_records_for_run(args: argparse.Namespace, output_dir: Path) -> Path:
    """Apply resume/restart behavior before the reducer writes any records.

    Resume keeps the existing analysis HDF5 and appends to the event log.
    Restart removes the old analysis record and old run sidecar records first,
    so the new run cannot mix with stale HDF5 rows or stale GUI monitor lines.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = analysis_h5_path_for_args(args, output_dir)
    if not args.resume:
        for path in [
            analysis_path,
            output_dir / "live_events.jsonl",
            output_dir / "live_replay_manifest.csv",
            output_dir / "live_sequence_manifest.csv",
            output_dir / "group_summary.csv",
        ]:
            if remove_file_if_present(path):
                print(f"Restart: removed old run record: {path}")
    else:
        validate_existing_analysis_h5(analysis_path)
    return analysis_path


def remove_file_if_present(path: Path) -> bool:
    """Delete a stale output file; missing files are already clean."""
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def append_frame_curve_to_analysis_h5(
    analysis_path: Path,
    curve: object,
    monitor_key: str,
) -> int:
    """Append one live single-frame 1D curve to the analysis HDF5 file."""
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    item = curve.item
    q = np.asarray(curve.q, dtype=float)
    intensity = np.asarray(curve.normalized_intensity, dtype=float)
    sigma = np.asarray(curve.normalized_error, dtype=float)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    timing = getattr(curve, "timing_seconds", None) or {}
    image_shape = tuple(getattr(curve, "image_shape", None) or ())
    source_file_bytes = getattr(curve, "source_file_bytes", None)

    with open_h5_retry(analysis_path, "a") as handle:
        entry = handle.require_group("entry")
        realtime = entry.require_group("realtime")
        process = realtime.require_group("process_01_reduction")
        process.attrs["NX_class"] = "NXprocess"
        frames = process.require_group("frames")
        frames.attrs["NX_class"] = "NXdata"
        frames.attrs["signal"] = "I_frame_q"
        frames.attrs["axes"] = np.asarray(["frame_number", "q"], dtype="S")
        frames.attrs["notes"] = "Single-frame 1D curves appended as files are reduced live."

        if "q" not in frames:
            frames.create_dataset("q", data=q)

        n_rows = frames["I_frame_q"].shape[0] if "I_frame_q" in frames else 0
        if "I_frame_q" not in frames:
            _create_extendable_dataset(frames, "I_frame_q", intensity.dtype, (0, intensity.size), (None, intensity.size))
            _create_extendable_dataset(frames, "sigma_frame_q", sigma.dtype, (0, sigma.size), (None, sigma.size))
            for name, dtype in [
                ("sequence_index", "i8"),
                ("energy_index", "i8"),
                ("group_index", "i8"),
                ("frame_index", "i8"),
            ]:
                _create_extendable_dataset(frames, name, np.dtype(dtype), (0,), (None,))
            for name in ["energy_kev", "monitor_value", "total_intensity"]:
                _create_extendable_dataset(frames, name, np.dtype("f8"), (0,), (None,))
            _create_extendable_dataset(frames, "source_file", string_dtype, (0,), (None,))
            _create_extendable_dataset(frames, "monitor_key", string_dtype, (0,), (None,))
            _create_extendable_dataset(frames, "qc_status", string_dtype, (0,), (None,))
            frames["qc_status"].attrs["pending_group_qc"] = (
                "Frame was reduced to 1D, but its energy/group has not reached "
                "the averaging trigger yet."
            )
            frames["qc_status"].attrs["accepted"] = (
                "Frame was kept by the group-average outlier filter."
            )
            frames["qc_status"].attrs["rejected_total_intensity"] = (
                "Frame was dropped from the group average by the total-intensity "
                "outlier filter."
            )
        for name in [
            "reduce_total_seconds",
            "read_energy_seconds",
            "read_image_seconds",
            "integrate_seconds",
            "read_monitor_seconds",
            "source_file_mb",
        ]:
            _ensure_extendable_1d_dataset(frames, name, np.dtype("f8"), n_rows, np.nan)
        for name in ["image_rows", "image_cols"]:
            _ensure_extendable_1d_dataset(frames, name, np.dtype("i8"), n_rows, -1)
        if "q_frame_q" in frames:
            _append_row(frames["q_frame_q"], q)
        elif frames["q"].shape == q.shape and not np.allclose(frames["q"][()], q, rtol=1e-7, atol=1e-12):
            existing_q = np.asarray(frames["q"][()], dtype=float)
            _create_extendable_dataset(frames, "q_frame_q", q.dtype, (0, q.size), (None, q.size))
            for _ in range(n_rows):
                _append_row(frames["q_frame_q"], existing_q)
            _append_row(frames["q_frame_q"], q)

        _append_row(frames["I_frame_q"], intensity)
        _append_row(frames["sigma_frame_q"], sigma)
        _append_scalar(frames["sequence_index"], item.sequence_index)
        _append_scalar(frames["energy_index"], item.energy_index)
        _append_scalar(frames["group_index"], item.group_index)
        _append_scalar(frames["frame_index"], item.frame_index)
        _append_scalar(frames["energy_kev"], np.nan if curve.energy_kev is None else curve.energy_kev)
        _append_scalar(frames["monitor_value"], curve.monitor_value)
        _append_scalar(frames["total_intensity"], curve.total_intensity)
        _append_scalar(frames["source_file"], str(item.path))
        _append_scalar(frames["monitor_key"], monitor_key)
        _append_scalar(frames["qc_status"], "pending_group_qc")
        _append_scalar(frames["reduce_total_seconds"], float(timing.get("total", np.nan)))
        _append_scalar(frames["read_energy_seconds"], float(timing.get("read_energy", np.nan)))
        _append_scalar(frames["read_image_seconds"], float(timing.get("read_image", np.nan)))
        _append_scalar(frames["integrate_seconds"], float(timing.get("integrate", np.nan)))
        _append_scalar(frames["read_monitor_seconds"], float(timing.get("read_monitor", np.nan)))
        _append_scalar(frames["source_file_mb"], np.nan if source_file_bytes is None else float(source_file_bytes) / 1_000_000.0)
        _append_scalar(frames["image_rows"], int(image_shape[0]) if len(image_shape) >= 1 else -1)
        _append_scalar(frames["image_cols"], int(image_shape[1]) if len(image_shape) >= 2 else -1)
        handle.flush()
    return n_rows


def append_frame_curve_to_analysis_h5_safe(
    analysis_path: Path,
    curve: object,
    monitor_key: str,
) -> int:
    """Append a live frame, recovering once from a damaged existing HDF5 file."""
    try:
        return append_frame_curve_to_analysis_h5(analysis_path, curve, monitor_key)
    except (OSError, RuntimeError) as exc:
        if not is_hdf5_access_error(exc):
            raise
        quarantined = quarantine_hdf5(analysis_path, exc)
        if quarantined is None:
            raise
        return append_frame_curve_to_analysis_h5(analysis_path, curve, monitor_key)


def update_frame_qc_status_in_analysis_h5(analysis_path: Path, avg: object) -> None:
    """Mark live frame QC status after group averaging.

    Status meanings:
    pending_group_qc: 1D frame exists, but the group-average trigger has not run.
    accepted: frame was kept by the group-average outlier filter.
    rejected_total_intensity: frame was dropped by the total-intensity filter.
    """
    if not analysis_path.exists():
        return
    frames_path = "/entry/realtime/process_01_reduction/frames"
    try:
        with open_h5_retry(analysis_path, "a") as handle:
            if frames_path not in handle:
                return
            frames = handle[frames_path]
            if "sequence_index" not in frames or "qc_status" not in frames:
                return
            sequence_indices = frames["sequence_index"][()]
            accepted = set(avg.kept_sequence_indices)
            rejected = set(avg.dropped_sequence_indices)
            qc = frames["qc_status"]
            for row, sequence_index in enumerate(sequence_indices):
                if int(sequence_index) in accepted:
                    qc[row] = "accepted"
                elif int(sequence_index) in rejected:
                    qc[row] = "rejected_total_intensity"
            handle.flush()
    except (OSError, RuntimeError) as exc:
        if not is_hdf5_access_error(exc):
            raise
        print(f"Warning: skipped live-frame QC update for damaged HDF5 {analysis_path}: {exc}")


def existing_live_frame_paths_and_sequences(analysis_path: Path) -> tuple[set[Path], set[int], int]:
    """Return already-written source files and sequence indices from live HDF5 rows."""
    if not analysis_path.exists():
        return set(), set(), 0
    frames_path = "/entry/realtime/process_01_reduction/frames"
    try:
        with open_h5_retry(analysis_path, "r") as handle:
            if frames_path not in handle:
                return set(), set(), 0
            frames = handle[frames_path]
            source_values = frames["source_file"][()] if "source_file" in frames else []
            sequence_values = frames["sequence_index"][()] if "sequence_index" in frames else []
    except (OSError, RuntimeError) as exc:
        if is_hdf5_access_error(exc):
            quarantine_hdf5(analysis_path, exc)
            return set(), set(), 0
        raise
    paths = {Path(_decode_h5_text(value)).expanduser().resolve() for value in source_values if _decode_h5_text(value)}
    sequences = {int(value) for value in sequence_values}
    return paths, sequences, max(sequences) if sequences else 0


def read_live_frame_items_from_analysis_h5(analysis_path: Path, core) -> list[object]:
    """Read lightweight manifest items for frames already present in analysis HDF5."""
    if not analysis_path.exists():
        return []
    frames_path = "/entry/realtime/process_01_reduction/frames"
    try:
        with open_h5_retry(analysis_path, "r") as handle:
            if frames_path not in handle:
                return []
            frames = handle[frames_path]
            required = ["sequence_index", "energy_index", "group_index", "frame_index", "source_file"]
            if any(name not in frames for name in required):
                return []
            items = [
                core.ManifestItem(
                    sequence_index=int(sequence_index),
                    energy_index=int(energy_index),
                    group_index=int(group_index),
                    frame_index=int(frame_index),
                    path=Path(_decode_h5_text(source_file)).expanduser().resolve(),
                )
                for sequence_index, energy_index, group_index, frame_index, source_file in zip(
                    frames["sequence_index"][()],
                    frames["energy_index"][()],
                    frames["group_index"][()],
                    frames["frame_index"][()],
                    frames["source_file"][()],
                )
            ]
    except (OSError, RuntimeError) as exc:
        if is_hdf5_access_error(exc):
            quarantine_hdf5(analysis_path, exc)
            return []
        raise
    return sorted(items, key=lambda item: item.sequence_index)


def resume_items_match_sequence(items: list[object], args: argparse.Namespace) -> bool:
    """Return True when old live rows are compatible with current sequence settings."""
    if not items:
        return True
    if any(item.group_index < 1 or item.group_index > args.num_groups for item in items):
        return False
    if any(item.frame_index < 1 or item.frame_index > args.num_frames for item in items):
        return False
    if args.num_energies is not None and any(item.energy_index < 1 or item.energy_index > args.num_energies for item in items):
        return False
    expected_total = args.num_energies * args.num_groups * args.num_frames if args.num_energies is not None else None
    if expected_total is not None and max(item.sequence_index for item in items) > expected_total:
        return False
    return True


def compatible_resume_items_or_quarantine(analysis_path: Path, core, args: argparse.Namespace) -> list[object]:
    """Read resume rows, moving old analysis aside when its sequence shape is stale."""
    items = read_live_frame_items_from_analysis_h5(analysis_path, core) if args.resume else []
    if items and not resume_items_match_sequence(items, args):
        quarantine_hdf5(
            analysis_path,
            (
                f"resume sequence mismatch for requested "
                f"energies={args.num_energies}, groups={args.num_groups}, frames={args.num_frames}"
            ),
        )
        return []
    return items


def restore_state_from_analysis_h5(state: "LivePipelineState") -> set[Path]:
    """Restore processed frame state from the live single-frame table.

    This makes watcher restarts continue after the last written frame instead of
    assigning old files again. It also rebuilds unfinished groups so the group
    average can still use frames that were reduced before the restart.
    """
    if not state.args.resume or not state.analysis_path.exists():
        return set()
    frames_path = "/entry/realtime/process_01_reduction/frames"
    try:
        with open_h5_retry(state.analysis_path, "r") as handle:
            if frames_path not in handle:
                return set()
            frames = handle[frames_path]
            required = ["q", "I_frame_q", "sequence_index", "energy_index", "group_index", "frame_index", "source_file"]
            if any(name not in frames for name in required):
                return set()
            q = np.asarray(frames["q"][()], dtype=float)
            normalized = np.asarray(frames["I_frame_q"][()], dtype=float)
            sequence_indices = np.asarray(frames["sequence_index"][()], dtype=int)
            energy_indices = np.asarray(frames["energy_index"][()], dtype=int)
            group_indices = np.asarray(frames["group_index"][()], dtype=int)
            frame_indices = np.asarray(frames["frame_index"][()], dtype=int)
            source_files = [_decode_h5_text(value) for value in frames["source_file"][()]]
            energy_kev = np.asarray(frames["energy_kev"][()] if "energy_kev" in frames else np.full(len(sequence_indices), np.nan), dtype=float)
            monitor_values = np.asarray(frames["monitor_value"][()] if "monitor_value" in frames else np.ones(len(sequence_indices)), dtype=float)
            total_values = np.asarray(frames["total_intensity"][()] if "total_intensity" in frames else np.full(len(sequence_indices), np.nan), dtype=float)
    except (OSError, RuntimeError) as exc:
        if is_hdf5_access_error(exc):
            quarantine_hdf5(state.analysis_path, exc)
            return set()
        raise

    restored_curves: list[object] = []
    processed_paths: set[Path] = set()
    for row, sequence_index in enumerate(sequence_indices):
        source_path = Path(source_files[row]).expanduser().resolve()
        processed_paths.add(source_path)
        item = state.core.ManifestItem(
            sequence_index=int(sequence_index),
            energy_index=int(energy_indices[row]),
            group_index=int(group_indices[row]),
            frame_index=int(frame_indices[row]),
            path=source_path,
        )
        monitor_value = float(monitor_values[row]) if np.isfinite(monitor_values[row]) else 1.0
        curve = state.core.FrameCurve(
            item=item,
            energy_kev=None if not np.isfinite(energy_kev[row]) else float(energy_kev[row]),
            monitor_value=monitor_value,
            q=q,
            intensity=normalized[row] * monitor_value,
            total_intensity=float(total_values[row]) if np.isfinite(total_values[row]) else float(np.nan),
            normalized_intensity=normalized[row],
        )
        restored_curves.append(curve)

    state.items = sorted([curve.item for curve in restored_curves], key=lambda item: item.sequence_index)
    grouped: dict[tuple[int, int], list[object]] = defaultdict(list)
    for curve in restored_curves:
        grouped[(curve.item.energy_index, curve.item.group_index)].append(curve)

    for key, curves in grouped.items():
        expected = state.expected_frame_counts.get(key)
        if expected is not None and len(curves) >= expected:
            averages = state.core.average_groups(curves, state.runtime_args.outlier_zmax)
            if averages:
                state.completed_averages[key] = averages[0]
                update_frame_qc_status_in_analysis_h5(state.analysis_path, averages[0])
        else:
            state.pending_group_curves[key].extend(sorted(curves, key=lambda curve: curve.item.sequence_index))

    for energy_index, groups in state.expected_energy_groups.items():
        energy_key_set = {(energy_index, group) for group in groups}
        if energy_key_set and energy_key_set.issubset(state.completed_averages):
            state.completed_energies.add(energy_index)
            if state.args.analysis_mode == "asaxs":
                energy_averages = [state.completed_averages[group_key] for group_key in sorted(energy_key_set)]
                state.completed_final_outputs.extend(
                    build_final_outputs_for_h5(state.core, energy_averages, state.runtime_args, state.output_dir)
                )

    state.last_analysis_path = state.analysis_path
    state.analysis_dirty = False
    return processed_paths


def _decode_h5_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _create_extendable_dataset(group: h5py.Group, name: str, dtype, shape: tuple[int, ...], maxshape: tuple[object, ...]) -> None:
    group.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype, chunks=True)


def _ensure_extendable_1d_dataset(
    group: h5py.Group,
    name: str,
    dtype,
    rows: int,
    fill_value: object,
) -> None:
    if name in group:
        return
    dataset = group.create_dataset(name, shape=(rows,), maxshape=(None,), dtype=dtype, chunks=True)
    if rows:
        dataset[...] = fill_value


def _append_row(dataset: h5py.Dataset, values: np.ndarray) -> None:
    row = dataset.shape[0]
    dataset.resize((row + 1, *dataset.shape[1:]))
    dataset[row, ...] = values


def _append_scalar(dataset: h5py.Dataset, value: object) -> None:
    row = dataset.shape[0]
    dataset.resize((row + 1,))
    dataset[row] = value


def filter_manifest_items(items: list[object], limit_energies: int | None, limit_frames: int | None) -> list[object]:
    """Apply demo-size limits while preserving acquisition order."""
    selected_energy_indices = sorted({item.energy_index for item in items})
    if limit_energies is not None:
        selected_energy_indices = selected_energy_indices[:limit_energies]
    selected_energy_set = set(selected_energy_indices)

    frame_counts: dict[tuple[int, int], int] = defaultdict(int)
    filtered: list[object] = []
    for item in sorted(items, key=lambda value: value.sequence_index):
        if item.energy_index not in selected_energy_set:
            continue
        key = (item.energy_index, item.group_index)
        if limit_frames is not None and frame_counts[key] >= limit_frames:
            continue
        frame_counts[key] += 1
        filtered.append(item)
    return filtered


def expected_counts_by_group(items: Iterable[object]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for item in items:
        counts[(item.energy_index, item.group_index)] += 1
    return dict(counts)


def expected_groups_by_energy(items: Iterable[object]) -> dict[int, set[int]]:
    groups: dict[int, set[int]] = defaultdict(set)
    for item in items:
        groups[item.energy_index].add(item.group_index)
    return dict(groups)


def write_live_manifest(items: list[object], path: Path, args: argparse.Namespace | None = None, core: object | None = None) -> Path:
    """Save the exact replay subset so the demo can be reproduced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_index", "energy_index", "group_index", "frame_index", "asaxs_role", "hdf5_path"])
        for item in items:
            role = "unknown"
            if args is not None and core is not None and hasattr(core, "asaxs_role_for_group"):
                role = core.asaxs_role_for_group(item.group_index, args)
            writer.writerow([item.sequence_index, item.energy_index, item.group_index, item.frame_index, role, item.path])
    return path


def make_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the small args namespace expected by the existing reducer functions."""
    return argparse.Namespace(
        poni=args.poni,
        mask=args.mask,
        dataset_path=args.dataset_path,
        npt=args.npt,
        unit=args.unit,
        delta_energy_percent=args.delta_energy_percent,
        outlier_zmax=args.outlier_zmax,
        gc_group=args.gc_group,
        air_group=args.air_group,
        empty_group=args.empty_group,
        water_group=args.water_group,
        sample_group=args.sample_group,
        asaxs_output_name=args.asaxs_output_name,
        asaxs_pair=list(args.asaxs_pair or []),
        asaxs_extraction_plan=args.asaxs_extraction_plan,
        gc_reference_file=args.gc_reference_file,
        gc_q_range=args.gc_q_range,
        capillary_thickness=args.capillary_thickness,
        gc_thickness=args.gc_thickness,
        subtract_fluorescence=args.subtract_fluorescence,
        fluorescence_level=args.fluorescence_level,
        fluorescence_reference=args.fluorescence_reference,
        fluorescence_q_range=args.fluorescence_q_range,
        write_text_output=args.write_text_output,
    )


def load_reduction_core() -> argparse.Namespace:
    """Import the bundled reduction functions used by the live scheduler."""
    from aswaxs_live.reduction.aswaxs_sequence import (  # pylint: disable=import-outside-toplevel
        FinalOutput,
        FrameCurve,
        ManifestItem,
        _write_sequence_analysis_h5,
        average_groups,
        asaxs_role_for_group,
        build_final_record,
        default_monitor_key,
        estimate_constant_fluorescence,
        extraction_recipes,
        final_outputs_for_recipe,
        infer_detector,
        poni_geometry_metadata,
        read_manifest,
        reduce_manifest_frames,
        validate_asaxs_group_roles,
        write_final_sample_outputs,
        write_group_average,
        write_summary,
    )

    return argparse.Namespace(
        FinalOutput=FinalOutput,
        FrameCurve=FrameCurve,
        ManifestItem=ManifestItem,
        _write_sequence_analysis_h5=_write_sequence_analysis_h5,
        average_groups=average_groups,
        asaxs_role_for_group=asaxs_role_for_group,
        build_final_record=build_final_record,
        default_monitor_key=default_monitor_key,
        estimate_constant_fluorescence=estimate_constant_fluorescence,
        extraction_recipes=extraction_recipes,
        final_outputs_for_recipe=final_outputs_for_recipe,
        infer_detector=infer_detector,
        poni_geometry_metadata=poni_geometry_metadata,
        read_manifest=read_manifest,
        reduce_manifest_frames=reduce_manifest_frames,
        validate_asaxs_group_roles=validate_asaxs_group_roles,
        write_final_sample_outputs=write_final_sample_outputs,
        write_group_average=write_group_average,
        write_summary=write_summary,
    )


def build_final_outputs_for_h5(
    core,
    averages: list[object],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[object]:
    """Compute final ASAXS outputs without writing legacy text files."""
    if args.write_text_output:
        return core.write_final_sample_outputs(averages, args, output_dir)
    outputs: list[object] = []
    for recipe in core.extraction_recipes(args):
        outputs.extend(core.final_outputs_for_recipe(averages, args, output_dir, recipe, write_text=False))
    return outputs


class LivePipelineState:
    """Stateful trigger engine shared by manifest replay and folder watching."""

    def __init__(
        self,
        args: argparse.Namespace,
        output_dir: Path,
        event_log,
        core,
        expected_frame_counts: dict[tuple[int, int], int],
        expected_energy_groups: dict[int, set[int]],
        monitor_key: str,
        detector: str,
    ) -> None:
        self.args = args
        self.output_dir = output_dir
        self.event_log = event_log
        self.core = core
        self.expected_frame_counts = expected_frame_counts
        self.expected_energy_groups = expected_energy_groups
        self.monitor_key = monitor_key
        self.detector = detector
        self.runtime_args = make_runtime_args(args)
        self.geometry_metadata = self.core.poni_geometry_metadata(args.poni)
        self.pending_group_curves: dict[tuple[int, int], list[object]] = defaultdict(list)
        self.completed_averages: dict[tuple[int, int], object] = {}
        self.completed_energies: set[int] = set()
        self.completed_final_outputs: list[object] = []
        self.items: list[object] = []
        self.analysis_path = analysis_h5_path_for_args(self.args, self.output_dir)
        self.analysis_dirty = False
        self.groups_since_analysis_write = 0
        self.last_analysis_path: Path | None = None

    def process_item(self, item: object) -> list[object]:
        self.items.append(item)
        self.analysis_dirty = True
        key = (item.energy_index, item.group_index)
        write_event(
            self.event_log,
            LiveEvent(
                time=now_iso(),
                event="frame_arrived",
                energy_index=item.energy_index,
                group_index=item.group_index,
                frame_index=item.frame_index,
                sequence_index=item.sequence_index,
                path=str(item.path),
            ),
        )

        if getattr(self.args, "quiet", False):
            with contextlib.redirect_stdout(io.StringIO()):
                curves = self.core.reduce_manifest_frames(
                    [item],
                    self.runtime_args,
                    self.monitor_key,
                    image_callback=getattr(self, "image_callback", None),
                )
        else:
            curves = self.core.reduce_manifest_frames(
                [item],
                self.runtime_args,
                self.monitor_key,
                image_callback=getattr(self, "image_callback", None),
            )
        h5_write_start = time.perf_counter()
        frame_rows = [append_frame_curve_to_analysis_h5_safe(self.analysis_path, curve, self.monitor_key) for curve in curves]
        h5_write_seconds = time.perf_counter() - h5_write_start
        self.pending_group_curves[key].extend(curves)
        frame_message = f"{len(self.pending_group_curves[key])}/{self.expected_frame_counts[key]} frames ready for group"
        if frame_rows:
            frame_message += f"; live H5 row {frame_rows[-1] + 1}"
        timing = getattr(curves[-1], "timing_seconds", None) if curves else None
        image_shape = tuple(getattr(curves[-1], "image_shape", None) or ()) if curves else ()
        source_file_bytes = getattr(curves[-1], "source_file_bytes", None) if curves else None
        write_event(
            self.event_log,
            LiveEvent(
                time=now_iso(),
                event="frame_reduced_1d",
                energy_index=item.energy_index,
                group_index=item.group_index,
                frame_index=item.frame_index,
                sequence_index=item.sequence_index,
                path=str(self.analysis_path),
                message=frame_message,
                reduce_total_seconds=None if not timing else timing.get("total"),
                read_energy_seconds=None if not timing else timing.get("read_energy"),
                read_image_seconds=None if not timing else timing.get("read_image"),
                integrate_seconds=None if not timing else timing.get("integrate"),
                read_monitor_seconds=None if not timing else timing.get("read_monitor"),
                h5_write_seconds=h5_write_seconds,
                source_file_mb=None if source_file_bytes is None else float(source_file_bytes) / 1_000_000.0,
                image_rows=int(image_shape[0]) if len(image_shape) >= 1 else None,
                image_cols=int(image_shape[1]) if len(image_shape) >= 2 else None,
            ),
        )

        if len(self.pending_group_curves[key]) == self.expected_frame_counts[key]:
            self._complete_group(key)
        self._complete_energy_if_ready(item.energy_index)
        return curves

    def _complete_group(self, key: tuple[int, int]) -> None:
        averages = self.core.average_groups(self.pending_group_curves[key], self.runtime_args.outlier_zmax)
        if len(averages) != 1:
            raise RuntimeError(f"Expected one completed average for {key}, got {len(averages)}")
        avg = averages[0]
        self.completed_averages[key] = avg
        self.pending_group_curves.pop(key, None)
        avg_path = self.core.write_group_average(avg, self.output_dir) if self.args.write_text_output else None
        update_frame_qc_status_in_analysis_h5(self.analysis_path, avg)
        self.analysis_dirty = True
        self.groups_since_analysis_write += 1
        interval = max(1, int(getattr(self.args, "analysis_write_interval_groups", 10)))
        wrote_analysis = False
        if self.groups_since_analysis_write >= interval:
            wrote_analysis = self.write_analysis_h5(force=True) is not None
        write_event(
            self.event_log,
            LiveEvent(
                time=now_iso(),
                event="group_average_written",
                energy_index=avg.energy_index,
                group_index=avg.group_index,
                path=str(avg_path) if avg_path else None,
                message=f"kept={avg.kept_count}, dropped={avg.dropped_count}; analysis H5 {'updated' if wrote_analysis else 'deferred'}",
            ),
        )

    def _complete_energy_if_ready(self, energy_index: int) -> None:
        energy_key_set = {(energy_index, group) for group in self.expected_energy_groups[energy_index]}
        if energy_index in self.completed_energies or not energy_key_set.issubset(self.completed_averages):
            return
        self.completed_energies.add(energy_index)
        energy_averages = [self.completed_averages[group_key] for group_key in sorted(energy_key_set)]
        if self.args.analysis_mode == "saxs":
            self.core.write_summary(
                list(self.completed_averages.values()),
                self.output_dir,
                self.monitor_key,
                self.detector,
                self.runtime_args,
            )
            self.write_analysis_h5(force=True)
            write_event(
                self.event_log,
                LiveEvent(
                    time=now_iso(),
                    event="energy_batch_saxs_completed",
                    energy_index=energy_index,
                    message=f"{len(energy_averages)} groups averaged; ASAXS correction skipped",
                ),
            )
            return

        write_event(
            self.event_log,
            LiveEvent(
                time=now_iso(),
                event="energy_batch_asaxs_started",
                energy_index=energy_index,
                message=f"{len(energy_averages)} groups ready",
            ),
        )
        final_outputs = build_final_outputs_for_h5(self.core, energy_averages, self.runtime_args, self.output_dir)
        self.completed_final_outputs.extend(final_outputs)
        self.analysis_dirty = True
        self.core.write_summary(
            list(self.completed_averages.values()),
            self.output_dir,
            self.monitor_key,
            self.detector,
            self.runtime_args,
        )
        self.write_analysis_h5(force=True)
        write_event(
            self.event_log,
            LiveEvent(
                time=now_iso(),
                event="energy_batch_asaxs_completed",
                energy_index=energy_index,
                message=f"final_outputs={len(final_outputs)}",
            ),
        )

    def force_complete_pending(self, reason: str) -> None:
        """Finalize groups that have frames when a queued job is declared done.

        Kafka ``measurement_done`` or abort-style messages mean the acquisition
        side is finished even if fewer frames arrived than the nominal GUI
        sequence settings. Average the frames that actually arrived and shrink
        the energy completion set to those completed groups.
        """
        completed_now: list[tuple[int, int]] = []
        for key in sorted(self.pending_group_curves):
            if key in self.completed_averages or not self.pending_group_curves[key]:
                continue
            self._complete_group(key)
            completed_now.append(key)
        if not completed_now:
            self.write_analysis_h5(force=True)
            return
        completed_by_energy: dict[int, set[int]] = defaultdict(set)
        for energy_index, group_index in self.completed_averages:
            completed_by_energy[int(energy_index)].add(int(group_index))
        for energy_index, groups in completed_by_energy.items():
            self.expected_energy_groups[energy_index] = set(groups)
            self._complete_energy_if_ready(energy_index)
        write_event(
            self.event_log,
            LiveEvent(
                time=now_iso(),
                event="queue_partial_groups_finalized",
                message=f"{len(completed_now)} partial group(s) finalized because {reason}",
            ),
        )

    def write_analysis_h5(self, *, force: bool = False) -> Path | None:
        if not self.completed_averages:
            return None
        if not force and not self.analysis_dirty and self.last_analysis_path is not None:
            return self.last_analysis_path
        manifest_path = write_live_manifest(self.items, self.output_dir / "live_sequence_manifest.csv", self.runtime_args, self.core)
        summary_path = self.core.write_summary(
            list(self.completed_averages.values()),
            self.output_dir,
            self.monitor_key,
            self.detector,
            self.runtime_args,
        )
        analysis_path = self.analysis_path
        try:
            run_h5_write_retry(
                lambda: self.core._write_sequence_analysis_h5(
                    analysis_path=analysis_path,
                    manifest_path=manifest_path,
                    items=self.items,
                    averages=list(self.completed_averages.values()),
                    final_outputs=self.completed_final_outputs,
                    args=self.runtime_args,
                    monitor_key=self.monitor_key,
                    detector=self.detector,
                    summary_path=summary_path,
                    geometry_metadata=self.geometry_metadata,
                )
            )
            if getattr(self.args, "export_xanos", True):
                export_analysis_h5_to_xanos_format(analysis_path)
        except (OSError, RuntimeError) as exc:
            if not is_hdf5_access_error(exc):
                raise
            quarantined = quarantine_hdf5(analysis_path, exc)
            if quarantined is None:
                raise
            run_h5_write_retry(
                lambda: self.core._write_sequence_analysis_h5(
                    analysis_path=analysis_path,
                    manifest_path=manifest_path,
                    items=self.items,
                    averages=list(self.completed_averages.values()),
                    final_outputs=self.completed_final_outputs,
                    args=self.runtime_args,
                    monitor_key=self.monitor_key,
                    detector=self.detector,
                    summary_path=summary_path,
                    geometry_metadata=self.geometry_metadata,
                )
            )
        self.analysis_dirty = False
        self.groups_since_analysis_write = 0
        self.last_analysis_path = analysis_path
        return analysis_path


def replay_live_pipeline(args: argparse.Namespace) -> int:
    core = load_reduction_core()
    if args.analysis_mode == "asaxs":
        core.validate_asaxs_group_roles(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = prepare_output_records_for_run(args, output_dir)
    event_log_path = output_dir / "live_events.jsonl"
    manifest_path = Path(args.manifest).expanduser().resolve()

    all_items = core.read_manifest(manifest_path)
    items = filter_manifest_items(all_items, args.limit_energies, args.limit_frames_per_group)
    if not items:
        raise ValueError("No manifest rows remain after applying demo limits.")

    detector = core.infer_detector(items, args.detector)
    monitor_key = args.monitor_key or core.default_monitor_key(detector)
    replay_manifest_path = write_live_manifest(items, output_dir / "live_replay_manifest.csv", args, core)

    expected_frame_counts = expected_counts_by_group(items)
    expected_energy_groups = expected_groups_by_energy(items)

    print("ASWAXS live V5 replay")
    print(f"Replay manifest: {replay_manifest_path}")
    print(f"Detector: {detector}")
    print(f"Monitor normalization key: {monitor_key}")
    print(f"Frames in manifest: {len(all_items)}")
    print(f"Frames to replay: {len(items)}")
    print(multicore_status_message(args))
    if len(items) != len(all_items):
        print("Replay limits are active; only replayed frames are written to the live single-frame HDF5 table.")
    print(f"Output: {output_dir}")
    print(f"Run behavior: {'resume existing analysis HDF5' if args.resume else 'restart and overwrite analysis HDF5'}")

    log_mode = "a" if args.resume and event_log_path.exists() else "w"
    with open_event_log(event_log_path, log_mode) as event_log:
        state = LivePipelineState(
            args=args,
            output_dir=output_dir,
            event_log=event_log,
            core=core,
            expected_frame_counts=expected_frame_counts,
            expected_energy_groups=expected_energy_groups,
            monitor_key=monitor_key,
            detector=detector,
        )
        restored_paths = restore_state_from_analysis_h5(state)
        restored_sequences = {item.sequence_index for item in state.items}
        if restored_paths or restored_sequences:
            print(f"Resume: restored {len(restored_sequences)} previously reduced frame rows from {state.analysis_path}")
        remaining_items = [
            item
            for item in sorted(items, key=lambda value: value.sequence_index)
            if item.sequence_index not in restored_sequences and item.path.expanduser().resolve() not in restored_paths
        ]
        if len(remaining_items) != len(items):
            print(f"Resume: skipping {len(items) - len(remaining_items)} already reduced manifest frames")
        for item in remaining_items:
            state.process_item(item)
        analysis_path = state.write_analysis_h5(force=True) or analysis_path

    print(f"Wrote live event log: {event_log_path}")
    print(f"Wrote analysis HDF5: {analysis_path}")
    print(f"Completed group averages: {len(state.completed_averages)}")
    print(f"Completed energy batches: {len(state.completed_energies)}")
    return 0


def watch_live_pipeline(args: argparse.Namespace) -> int:
    """Watch acquisition folders and assign sequence meaning by file order."""
    if args.num_frames is None:
        raise ValueError("Watcher mode requires --num-frames.")
    if args.num_groups is None and not args.auto_num_groups:
        raise ValueError("Watcher mode requires --num-groups, or use --auto-num-groups with --num-energies and --num-frames.")

    core = load_reduction_core()
    if args.analysis_mode == "asaxs":
        core.validate_asaxs_group_roles(args)
    watch_dir = Path(args.watch_dir).expanduser().resolve() if args.watch_dir else None
    queue_path = args.measurement_done_queue or args.frame_done_queue
    queue_reader = ReductionJobQueueReader(queue_path) if queue_path else None
    if queue_reader is not None:
        args.settle_seconds = 0.0
        args.poll_seconds = min(float(args.poll_seconds), 0.25)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if watch_dir is not None and not watch_dir.exists():
        raise FileNotFoundError(f"Missing watch directory: {watch_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = prepare_output_records_for_run(args, output_dir)
    recursive_watch = args.recursive_watch
    if watch_dir is not None and args.num_groups is None:
        inferred_groups = infer_num_groups_from_folder(watch_dir, args, output_dir, recursive_watch)
        if inferred_groups is None:
            raise ValueError("Could not infer groups yet; file count is empty or not divisible by num_energies * num_frames.")
        args.num_groups = inferred_groups
        print(f"Auto groups from files: {args.num_groups}")

    bootstrap_num_groups = args.num_groups if args.num_groups is not None else 1
    assigner = SequenceAssigner(
        num_groups=bootstrap_num_groups,
        num_frames=args.num_frames,
        num_energies=args.num_energies,
    )
    expected_total = assigner.expected_total
    expected_group_set = set(range(1, bootstrap_num_groups + 1))
    processed_paths: set[Path] = set()
    detector: str | None = None
    monitor_key: str | None = None
    state: LivePipelineState | None = None
    event_log_path = output_dir / "live_events.jsonl"

    # In queue mode the active sample/output folder is not known until a queue
    # job arrives. Resume per queued scan directory below so stale rows from a
    # different sample cannot become the starting point for this run.
    resume_items = [] if queue_reader is not None else compatible_resume_items_or_quarantine(analysis_path, core, args)
    if resume_items:
        detector = core.infer_detector(resume_items, args.detector)
        monitor_key = args.monitor_key or core.default_monitor_key(detector)
        expected_frame_counts = {
            (item.energy_index, item.group_index): args.num_frames
            for item in resume_items
        }
        if args.num_energies is not None:
            expected_energy_groups = {
                energy_index: expected_group_set
                for energy_index in range(1, args.num_energies + 1)
            }
        else:
            expected_energy_groups = {
                energy_index: expected_group_set
                for energy_index in sorted({item.energy_index for item in resume_items})
            }
        state = LivePipelineState(
            args=args,
            output_dir=output_dir,
            event_log=None,
            core=core,
            expected_frame_counts=expected_frame_counts,
            expected_energy_groups=expected_energy_groups,
            monitor_key=monitor_key,
            detector=detector,
        )
        processed_paths = restore_state_from_analysis_h5(state)
        if resume_items:
            assigner.advance_to_sequence_index(max(item.sequence_index for item in resume_items) + 1)

    print("ASWAXS live V5 watcher")
    if queue_reader is not None:
        print(f"Bluesky measurement_done queue: {queue_reader.path}")
        print("Queue role: receive completed-measurement reduction jobs with detector data_dir.")
    if watch_dir is not None:
        print(f"Watching: {watch_dir}")
    print(f"Pattern: {args.pattern}")
    requested_detector = normalize_detector_name(args.detector)
    print(f"Recursive watch: {'yes' if recursive_watch else 'no'}")
    group_rule = args.num_groups if args.num_groups is not None else "auto from file count"
    print(f"Sequence rule: energy -> group -> frame, groups={group_rule}, frames/group={args.num_frames}")
    print(f"Timing: poll_seconds={args.poll_seconds}, settle_seconds={args.settle_seconds}")
    print(multicore_status_message(args))
    if args.num_energies is not None:
        print(f"Expected total files: {expected_total}")
    print(f"Output: {output_dir}")
    print(f"Run behavior: {'resume existing analysis HDF5' if args.resume else 'restart and overwrite analysis HDF5'}")
    if state is not None:
        print(f"Resume: restored {len(processed_paths)} previously reduced frame rows from {analysis_path}")
        print(f"Resume: next sequence index is {assigner.assigned_count + 1}")

    log_mode = "a" if args.resume and event_log_path.exists() else "w"
    with open_event_log(event_log_path, log_mode) as event_log:
        if state is not None:
            state.event_log = event_log
        active_scan_dirs: list[Path] = []
        active_scan_dir_set: set[Path] = set()
        output_dirs_by_scan_dir: dict[Path, Path] = {}
        args_by_scan_dir: dict[Path, argparse.Namespace] = {}
        finalize_on_idle_by_scan_dir: dict[Path, str] = {}
        current_scan_dir: Path | None = None
        saw_matching_queue_job = False
        total_processed_paths = 0
        total_completed_averages = 0
        total_completed_energies = 0
        last_completed_analysis_path: Path | None = None

        def reset_assigner_for_args(run_args: argparse.Namespace) -> None:
            nonlocal assigner, expected_group_set
            run_num_groups = run_args.num_groups if run_args.num_groups is not None else 1
            assigner = SequenceAssigner(
                num_groups=run_num_groups,
                num_frames=run_args.num_frames,
                num_energies=run_args.num_energies,
            )
            expected_group_set = set(range(1, run_num_groups + 1))

        def add_active_scan_dir(scan_dir: Path) -> None:
            if scan_dir in active_scan_dir_set:
                return
            active_scan_dirs.append(scan_dir)
            active_scan_dir_set.add(scan_dir)

        if watch_dir is not None:
            add_active_scan_dir(watch_dir)

        def reset_completed_queue_batch(reason: str = "expected sequence completed") -> None:
            nonlocal assigner, processed_paths, detector, monitor_key, state, current_scan_dir
            nonlocal total_processed_paths, total_completed_averages, total_completed_energies, last_completed_analysis_path
            nonlocal expected_group_set
            if state is not None:
                state.force_complete_pending(reason)
                last_completed_analysis_path = state.write_analysis_h5(force=True)
                total_processed_paths += len(processed_paths)
                total_completed_averages += len(state.completed_averages)
                total_completed_energies += len(state.completed_energies)
            if current_scan_dir is not None:
                if current_scan_dir in active_scan_dir_set:
                    active_scan_dir_set.remove(current_scan_dir)
                active_scan_dirs[:] = [scan_dir for scan_dir in active_scan_dirs if scan_dir != current_scan_dir]
                output_dirs_by_scan_dir.pop(current_scan_dir, None)
                args_by_scan_dir.pop(current_scan_dir, None)
                finalize_on_idle_by_scan_dir.pop(current_scan_dir, None)
            write_event(
                event_log,
                LiveEvent(
                    time=now_iso(),
                    event="queue_batch_complete",
                    data_dir=str(current_scan_dir) if current_scan_dir is not None else None,
                    message=f"Completed queued detector batch ({reason}); waiting for the next queued sample.",
                ),
            )
            reset_assigner_for_args(args)
            processed_paths = set()
            detector = None
            monitor_key = None
            state = None
            current_scan_dir = None

        while True:
            ready_count_this_poll = 0
            if queue_reader is not None:
                for job in queue_reader.poll():
                    job_detector = normalize_detector_name(job.detector)
                    if (
                        requested_detector != "auto"
                        and job_detector is not None
                        and job_detector != requested_detector
                    ):
                        write_event(
                            event_log,
                            LiveEvent(
                                time=now_iso(),
                                event="measurement_done_ignored",
                                message="Queue job belongs to another detector reducer; skipping it here.",
                                uid=job.uid,
                                scan_id=job.scan_id,
                                sample_name=job.sample_name,
                                detector=job.detector,
                                data_dir=str(job.data_dir),
                                output_dir=str(job.output_dir) if job.output_dir is not None else None,
                            ),
                        )
                        continue
                    saw_matching_queue_job = True
                    add_active_scan_dir(job.data_dir)
                    job_output_dir = queue_job_output_dir(args, job, output_dir, requested_detector)
                    job_output_dir.mkdir(parents=True, exist_ok=True)
                    output_dirs_by_scan_dir[job.data_dir] = job_output_dir
                    job_runtime_args = queue_job_runtime_args(args, job, requested_detector)
                    args_by_scan_dir[job.data_dir] = job_runtime_args
                    if job.event in {"measurement_done", "measurement_aborted", "measurement_stopped", "frame_done"}:
                        finalize_on_idle_by_scan_dir[job.data_dir] = job.event
                    write_event(
                        event_log,
                        LiveEvent(
                            time=now_iso(),
                            event="measurement_done_received",
                            message=(
                                "Bluesky/Kafka measurement_done job queued this detector directory for reduction. "
                                + queue_job_sequence_message(args, job_runtime_args, job)
                            ),
                            uid=job.uid,
                            scan_id=job.scan_id,
                            sample_name=job.sample_name,
                            detector=job.detector,
                            data_dir=str(job.data_dir),
                            output_dir=str(job_output_dir),
                            expected_total_frames=expected_total_frames_for_args(job_runtime_args),
                            num_energies=job_runtime_args.num_energies,
                            num_groups=job_runtime_args.num_groups,
                            num_frames=job_runtime_args.num_frames,
                        ),
                    )

            for scan_dir in list(active_scan_dirs):
                if assigner.is_complete():
                    break
                current_scan_dir = scan_dir
                if not scan_dir.exists():
                    write_event(
                        event_log,
                        LiveEvent(
                            time=now_iso(),
                            event="scan_dir_waiting",
                            data_dir=str(scan_dir),
                            message="directory does not exist yet",
                        ),
                    )
                    break
                scan_output_dir = output_dirs_by_scan_dir.get(scan_dir, output_dir)
                scan_args = args_by_scan_dir.get(scan_dir, args)
                scan_output_dir.mkdir(parents=True, exist_ok=True)
                if scan_args.num_groups is None:
                    inferred_groups = infer_num_groups_from_folder(scan_dir, scan_args, scan_output_dir, recursive_watch)
                    if inferred_groups is None:
                        write_event(
                            event_log,
                            LiveEvent(
                                time=now_iso(),
                                event="auto_groups_waiting",
                                data_dir=str(scan_dir),
                                message=auto_group_wait_message(scan_dir, scan_args, scan_output_dir, recursive_watch),
                            ),
                        )
                        continue
                    scan_args.num_groups = inferred_groups
                    write_event(
                        event_log,
                        LiveEvent(
                            time=now_iso(),
                            event="auto_groups_inferred",
                            data_dir=str(scan_dir),
                            message=f"inferred {inferred_groups} groups per energy from HDF5 file count",
                            expected_total_frames=expected_total_frames_for_args(scan_args),
                            num_energies=scan_args.num_energies,
                            num_groups=scan_args.num_groups,
                            num_frames=scan_args.num_frames,
                        ),
                    )
                if state is None:
                    candidates = watch_candidates(scan_dir, scan_args.pattern, recursive_watch, scan_output_dir)
                    file_limit, block_size, complete_blocks, remainder = offline_complete_block_plan(scan_args, len(candidates))
                    if file_limit is not None:
                        if complete_blocks == 0:
                            write_event(
                                event_log,
                                LiveEvent(
                                    time=now_iso(),
                                    event="offline_incomplete_set_skipped",
                                    data_dir=str(scan_dir),
                                    message=(
                                        f"{len(candidates)} HDF5 file(s) found, but one complete block requires "
                                        f"{block_size}; no files reduced for this queued folder."
                                    ),
                                ),
                            )
                            reset_completed_queue_batch("offline folder has no complete block")
                            break
                        if complete_blocks > 1:
                            scan_args.num_energies = int(scan_args.num_energies) * complete_blocks
                        if remainder:
                            write_event(
                                event_log,
                                LiveEvent(
                                    time=now_iso(),
                                    event="offline_incomplete_tail_ignored",
                                    data_dir=str(scan_dir),
                                    message=(
                                        f"{len(candidates)} HDF5 file(s) found; reducing {file_limit} file(s) "
                                        f"from {complete_blocks} complete block(s), ignoring {remainder} tail file(s)."
                                    ),
                                ),
                            )
                    reset_assigner_for_args(scan_args)
                    if scan_output_dir.resolve() != output_dir.resolve():
                        prepare_output_records_for_run(scan_args, scan_output_dir)
                    scan_analysis_path = analysis_h5_path_for_args(scan_args, scan_output_dir)
                    scan_resume_items = compatible_resume_items_or_quarantine(scan_analysis_path, core, scan_args)
                    if scan_resume_items:
                        detector = core.infer_detector(scan_resume_items, scan_args.detector)
                        monitor_key = scan_args.monitor_key or core.default_monitor_key(detector)
                        expected_frame_counts = {
                            (item.energy_index, item.group_index): scan_args.num_frames
                            for item in scan_resume_items
                        }
                        if scan_args.num_energies is not None:
                            expected_energy_groups = {
                                energy_index: expected_group_set
                                for energy_index in range(1, scan_args.num_energies + 1)
                            }
                        else:
                            expected_energy_groups = {
                                energy_index: expected_group_set
                                for energy_index in sorted({item.energy_index for item in scan_resume_items})
                            }
                        state = LivePipelineState(
                            args=scan_args,
                            output_dir=scan_output_dir,
                            event_log=event_log,
                            core=core,
                            expected_frame_counts=expected_frame_counts,
                            expected_energy_groups=expected_energy_groups,
                            monitor_key=monitor_key,
                            detector=detector,
                        )
                        processed_paths = restore_state_from_analysis_h5(state)
                        if scan_resume_items:
                            assigner.advance_to_sequence_index(max(item.sequence_index for item in scan_resume_items) + 1)
                        print(f"Resume: restored {len(processed_paths)} previously reduced frame rows from {scan_analysis_path}")
                        print(f"Resume: next sequence index is {assigner.assigned_count + 1}")
                        if assigner.is_complete() and queue_reader is not None and args.continuous_queue:
                            reset_completed_queue_batch("resume state already matches expected sequence")
                            break
                candidates = watch_candidates(scan_dir, scan_args.pattern, recursive_watch, scan_output_dir)
                file_limit, _block_size, _complete_blocks, _remainder = offline_complete_block_plan(scan_args, len(candidates))
                if file_limit is not None:
                    candidates = candidates[:file_limit]
                waiting_files_this_scan = False
                for h5_path in candidates:
                    if assigner.is_complete():
                        break
                    h5_path = h5_path.resolve()
                    if h5_path in processed_paths:
                        continue
                    settle_seconds = 0.0 if args.once else args.settle_seconds
                    ready, reason = file_is_ready(h5_path, scan_args.dataset_path, settle_seconds)
                    if not ready:
                        waiting_files_this_scan = True
                        write_event(
                            event_log,
                            LiveEvent(
                                time=now_iso(),
                                event="file_waiting",
                                path=str(h5_path),
                                message=reason,
                                data_dir=str(scan_dir),
                            ),
                        )
                        continue

                    try:
                        position = assigner.next_position()
                    except StopIteration:
                        break

                    item = core.ManifestItem(
                        sequence_index=position.sequence_index,
                        energy_index=position.energy_index,
                        group_index=position.group_index,
                        frame_index=position.frame_index,
                        path=h5_path,
                    )
                    if detector is None:
                        detector = core.infer_detector([item], scan_args.detector)
                        monitor_key = scan_args.monitor_key or core.default_monitor_key(detector)
                        state = LivePipelineState(
                            args=scan_args,
                            output_dir=scan_output_dir,
                            event_log=event_log,
                            core=core,
                            expected_frame_counts={},
                            expected_energy_groups={},
                            monitor_key=monitor_key,
                            detector=detector,
                        )
                        print(f"Detector: {detector}")
                        print(f"Monitor normalization key: {monitor_key}")

                    assert state is not None
                    state.expected_frame_counts[(position.energy_index, position.group_index)] = scan_args.num_frames
                    state.expected_energy_groups[position.energy_index] = expected_group_set
                    write_event(
                        event_log,
                        LiveEvent(
                            time=now_iso(),
                            event="file_assigned_sequence",
                            energy_index=position.energy_index,
                            group_index=position.group_index,
                            frame_index=position.frame_index,
                            sequence_index=position.sequence_index,
                            path=str(h5_path),
                            data_dir=str(scan_dir),
                            expected_total_frames=assigner.expected_total,
                            num_energies=scan_args.num_energies,
                            num_groups=scan_args.num_groups,
                            num_frames=scan_args.num_frames,
                        ),
                    )
                    state.process_item(item)
                    processed_paths.add(h5_path)
                    ready_count_this_poll += 1
                    if assigner.is_complete() and queue_reader is not None and args.continuous_queue:
                        reset_completed_queue_batch("expected sequence completed")
                        break
                if (
                    queue_reader is not None
                    and args.continuous_queue
                    and state is not None
                    and scan_dir in finalize_on_idle_by_scan_dir
                    and not waiting_files_this_scan
                    and current_scan_dir == scan_dir
                ):
                    reset_completed_queue_batch(f"{finalize_on_idle_by_scan_dir.get(scan_dir, 'measurement_done')} message drained ready files")
                    break

            if assigner.is_complete():
                if queue_reader is not None and args.continuous_queue:
                    reset_completed_queue_batch("expected sequence completed")
                    continue
                break
            if (
                queue_reader is not None
                and args.stop_when_queue_drained
                and saw_matching_queue_job
                and not active_scan_dirs
                and state is None
            ):
                write_event(
                    event_log,
                    LiveEvent(
                        time=now_iso(),
                        event="queue_drained",
                        message="All queued sample-list jobs for this detector are complete; stopping reducer.",
                    ),
                )
                break
            if args.once:
                break
            if ready_count_this_poll == 0:
                time.sleep(max(0.1, args.poll_seconds))

        if state is not None:
            analysis_path = state.write_analysis_h5(force=True)
            total_processed_paths += len(processed_paths)
            total_completed_averages += len(state.completed_averages)
            total_completed_energies += len(state.completed_energies)
        else:
            analysis_path = last_completed_analysis_path

    print(f"Wrote live event log: {event_log_path}")
    if analysis_path is not None:
        print(f"Wrote analysis HDF5: {analysis_path}")
    if total_processed_paths == 0:
        print("No HDF5 files were processed.")
    else:
        print(f"Processed files: {total_processed_paths}")
        print(f"Completed group averages: {total_completed_averages}")
        print(f"Completed energy batches: {total_completed_energies}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    live_sources = [bool(args.watch_dir), bool(args.frame_done_queue), bool(args.measurement_done_queue)]
    if args.manifest and any(live_sources):
        raise ValueError("Use --manifest for replay or live watch/queue mode, not both.")
    if sum(live_sources) == 0 and not args.manifest:
        raise ValueError("Provide --manifest, --watch-dir, or --frame-done-queue.")
    if args.watch_dir or args.frame_done_queue or args.measurement_done_queue:
        return watch_live_pipeline(args)
    if args.manifest:
        return replay_live_pipeline(args)
    raise ValueError("No reducer mode selected.")


if __name__ == "__main__":
    raise SystemExit(main())

