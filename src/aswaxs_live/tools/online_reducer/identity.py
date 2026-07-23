from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np


TITLE_KEYS = (
    "experiment_title",
    "experiment_name",
    "proposal_title",
    "experiment",
    "title",
    "scan_title",
    "sample_name",
    "sample",
    "TIFFImageDescription",
)
EXPERIMENT_UID_KEYS = ("experiment_uid", "proposal_uid", "proposal_id", "run_uid", "uid", "item_uid")
RUN_UID_KEYS = ("run_uid", "uid", "item_uid")
MEASUREMENT_UID_KEYS = ("measurement_uid", "measurement_id", "point_uid", "run_uid", "uid")
SCAN_ID_KEYS = ("scan_id", "scan", "scan_number")
NDATTR_ROOTS = ("entry/instrument/NDAttributes", "entry/metadata", "metadata")


@dataclass(frozen=True)
class ExperimentIdentity:
    title: str
    experiment_uid: str
    run_uid: str
    measurement_uid: str
    scan_id: str
    raw_experiment_root: Path
    canonical_output_root: Path
    identity_source: str

    @property
    def safe_title(self) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.title.strip()).strip("._")
        return cleaned or "online_experiment"

    @property
    def storage_name(self) -> str:
        return f"{self.safe_title}_{self.experiment_uid[:8]}"

    @property
    def analysis_uid(self) -> str:
        return hashlib.sha256(f"FrameByFrame:{self.experiment_uid}".encode("utf-8")).hexdigest()

    def metadata(self) -> dict[str, str]:
        return {
            "experiment_title": self.title,
            "experiment_uid": self.experiment_uid,
            "run_uid": self.run_uid,
            "measurement_uid": self.measurement_uid,
            "scan_id": self.scan_id,
            "analysis_uid": self.analysis_uid,
            "raw_experiment_root": str(self.raw_experiment_root),
            "canonical_output_root": str(self.canonical_output_root),
            "identity_source": self.identity_source,
        }


def resolve_experiment_identity(
    image_path: Path,
    payload: dict[str, Any] | None,
    *,
    detector: str,
    fallback_title: str = "",
) -> ExperimentIdentity:
    path = Path(image_path).expanduser().resolve()
    payload = dict(payload or {})
    raw_root = experiment_root_for_image(path, detector)
    h5_values = _identity_values_from_h5(path)

    title = _first_payload_value(payload, TITLE_KEYS)
    source = "ZMQ payload"
    if not title:
        title = _first_mapping_value(h5_values, TITLE_KEYS)
        source = "raw HDF5 metadata"
    if not title:
        configured = str(fallback_title or "").strip()
        title = raw_root.name if not configured or configured == "online_aswaxs" else configured
        source = "raw experiment folder" if title == raw_root.name else "online setup fallback"

    experiment_uid = _first_payload_value(payload, EXPERIMENT_UID_KEYS) or _first_mapping_value(h5_values, EXPERIMENT_UID_KEYS)
    if not experiment_uid:
        experiment_uid = hashlib.sha256(str(raw_root).encode("utf-8")).hexdigest()
    run_uid = _first_payload_value(payload, RUN_UID_KEYS) or _first_mapping_value(h5_values, RUN_UID_KEYS) or experiment_uid
    measurement_uid = (
        _first_payload_value(payload, MEASUREMENT_UID_KEYS)
        or _first_mapping_value(h5_values, MEASUREMENT_UID_KEYS)
        or run_uid
    )
    scan_id = _first_payload_value(payload, SCAN_ID_KEYS) or _first_mapping_value(h5_values, SCAN_ID_KEYS) or ""
    return ExperimentIdentity(
        title=str(title),
        experiment_uid=str(experiment_uid),
        run_uid=str(run_uid),
        measurement_uid=str(measurement_uid),
        scan_id=str(scan_id),
        raw_experiment_root=raw_root,
        canonical_output_root=canonical_analysis_root(raw_root),
        identity_source=source,
    )


def experiment_root_for_image(image_path: Path, detector: str) -> Path:
    parent = Path(image_path).parent
    detector_names = {str(detector).lower(), "pil300k", "eig1m", "saxs", "waxs"}
    return parent.parent if parent.name.lower() in detector_names else parent


def canonical_analysis_root(raw_experiment_root: Path) -> Path:
    raw_root = Path(raw_experiment_root).expanduser().resolve()
    return raw_root.parent / "Extracted" / raw_root.name


def _first_payload_value(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    mappings = [payload]
    for container in ("metadata", "start", "run", "document"):
        nested = payload.get(container)
        if isinstance(nested, dict):
            mappings.append(nested)
    for mapping in mappings:
        value = _first_mapping_value(mapping, keys)
        if value:
            return value
    return ""


def _first_mapping_value(values: dict[str, Any], keys: tuple[str, ...]) -> str:
    lowered = {str(key).lower(): value for key, value in values.items()}
    for key in keys:
        value = values.get(key, lowered.get(key.lower()))
        text = _scalar_text(value)
        if text:
            return text
    return ""


def _identity_values_from_h5(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    keys = set(TITLE_KEYS + EXPERIMENT_UID_KEYS + RUN_UID_KEYS + MEASUREMENT_UID_KEYS + SCAN_ID_KEYS)
    try:
        with h5py.File(path, "r") as handle:
            for root in NDATTR_ROOTS:
                group = handle.get(root)
                if not isinstance(group, h5py.Group):
                    continue
                children = {str(name).lower(): name for name in group}
                attributes = {str(name).lower(): name for name in group.attrs}
                for key in keys:
                    actual = children.get(key.lower())
                    if actual is not None and isinstance(group[actual], h5py.Dataset):
                        values[key] = group[actual][()]
                    elif key.lower() in attributes:
                        values[key] = group.attrs[attributes[key.lower()]]
            if "entry/title" in handle and not _first_mapping_value(values, TITLE_KEYS):
                values["title"] = handle["entry/title"][()]
    except (OSError, KeyError, ValueError):
        return {}
    return values


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    array = np.asarray(value)
    if array.size == 0:
        return ""
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8", errors="replace")
    return str(scalar).strip()
