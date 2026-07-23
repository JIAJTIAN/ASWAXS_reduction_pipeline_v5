"""Task definitions shared by the dashboard and reduction workflows."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


def safe_name(value: str) -> str:
    """Return a filesystem-safe analysis name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "analysis"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@dataclass
class AsaxsPair:
    output_name: str
    sample_group: int
    solvent_group: int

    def cli_value(self) -> str:
        return f"{self.output_name}:{self.sample_group}:{self.solvent_group}"


@dataclass
class TaskSpec:
    task_name: str
    raw_folder: str
    output_dir: str
    num_energies: int
    num_groups: int
    num_frames: int
    pil300k_poni: str
    pil300k_mask: str
    eig1m_poni: str
    eig1m_mask: str
    pil300k_files: list[str] = field(default_factory=list)
    eig1m_files: list[str] = field(default_factory=list)
    detector_mode: str = "both"
    reduction_mode: str = "asaxs"
    capillary_thickness: float = 0.15
    gc_thickness: float = 0.1055
    gc_group: int | None = 1
    air_group: int | None = 2
    empty_group: int | None = 3
    cores: int = max(1, os.cpu_count() or 1)
    pattern: str = "*.h5"
    dataset_path: str = "entry/data/data"
    pil300k_monitor_key: str = "SPDS"
    eig1m_monitor_key: str = "WPDS"
    npt: int = 1000
    unit: str = "q_A^-1"
    asaxs_pairs: list[AsaxsPair] = field(default_factory=list)
    pil300k_count: int = 0
    eig1m_count: int = 0
    analysis_h5_path: str = ""
    status: str = "Ready"
    message: str = ""
    last_run_seconds: float | None = None
    last_run_finished_at: str = ""

    @property
    def raw_path(self) -> Path:
        return Path(self.raw_folder).expanduser().resolve()

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    @property
    def expected_files_per_detector(self) -> int:
        return int(self.num_energies) * int(self.num_groups) * int(self.num_frames)

    @property
    def sequence_label(self) -> str:
        return f"{self.num_energies}E {self.num_groups}G {self.num_frames}F"

    @property
    def pair_label(self) -> str:
        return ", ".join(pair.output_name for pair in self.asaxs_pairs) or "-"

    def detector_dir(self, detector: str) -> Path:
        return self.raw_path / detector

    def detector_files(self, detector: str) -> list[Path]:
        values = self.pil300k_files if detector == "Pil300K" else self.eig1m_files
        return [Path(value).expanduser().resolve() for value in values]

    def raw_source_roots(self) -> list[Path]:
        roots: list[Path] = []
        if self.raw_folder:
            roots.append(self.raw_path)
        for detector in ("Pil300K", "Eig1M"):
            for path in self.detector_files(detector):
                parent = path.parent
                roots.append(parent.parent if parent.name in {"Pil300K", "Eig1M"} else parent)
        unique: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root)
            if key not in seen:
                unique.append(root)
                seen.add(key)
        return unique

    def has_selected_files(self, detector: str | None = None) -> bool:
        if detector == "Pil300K":
            return bool(self.pil300k_files)
        if detector == "Eig1M":
            return bool(self.eig1m_files)
        return bool(self.pil300k_files or self.eig1m_files)

    @property
    def source_label(self) -> str:
        return "selected HDF5 files" if self.has_selected_files() else "detector folders"

    def detector_output_dir(self, detector: str) -> Path:
        return self.output_path / detector

    def detector_analysis_h5_path(self, detector: str) -> Path:
        return self.detector_output_dir(detector) / f"{safe_name(self.task_name)}_{detector}_analysis.h5"

    def combined_h5_path(self) -> Path:
        return self.output_path / f"{safe_name(self.task_name)}_analysis.h5"

    def active_detectors(self) -> tuple[str, ...]:
        mode = str(self.detector_mode or "both")
        if mode == "pil300k":
            return ("Pil300K",)
        if mode == "eig1m":
            return ("Eig1M",)
        return ("Pil300K", "Eig1M")

    @property
    def detector_label(self) -> str:
        return " + ".join(self.active_detectors())

    def is_asaxs_mode(self) -> bool:
        return str(self.reduction_mode or "asaxs") == "asaxs"

    def is_saxs_mode(self) -> bool:
        return str(self.reduction_mode or "asaxs") == "saxs"

    def xanos_output_name(self) -> str:
        for pair in self.asaxs_pairs:
            name = str(pair.output_name).strip()
            if name:
                return name
        return ""

    @property
    def last_run_label(self) -> str:
        if self.last_run_seconds is None:
            return "not recorded"
        return _format_duration(self.last_run_seconds)

