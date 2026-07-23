"""HDF5 image and metadata loading helpers for the calibration GUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from pyFAI.detectors import ALL_DETECTORS, detector_factory


DEFAULT_DATASET_PATH = "entry/data/data"
DEFAULT_SAXS_DISTANCE_M = 5.6
PATH_KEYS = (
    "path",
    "file",
    "filename",
    "filepath",
    "file_path",
    "image",
    "image_path",
    "data_path",
    "hdf5_path",
)
NDATTR_PREFIX = "entry/instrument/NDAttributes"
MOTOR_KEYS = (
    "SD_X",
    "SD_Y",
    "SPDS",
    "WD_X",
    "WD_Y",
    "WD_Z",
    "WD_RY",
)


@dataclass
class LoadedImage:
    path: Path
    dataset_path: str
    image: np.ndarray
    metadata: dict[str, Any]


def _normalize_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _find_path_candidate(payload: Any) -> str | None:
    if isinstance(payload, str):
        text = payload.strip()
        if text.lower().endswith((".h5", ".hdf5")):
            return text
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return None
        return _find_path_candidate(decoded)

    if isinstance(payload, dict):
        for key in PATH_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.lower().endswith((".h5", ".hdf5")):
                return value
        for value in payload.values():
            candidate = _find_path_candidate(value)
            if candidate:
                return candidate
        return None

    if isinstance(payload, list):
        for item in payload:
            candidate = _find_path_candidate(item)
            if candidate:
                return candidate
    return None


def extract_file_path_from_kafka_message(message_text: str) -> Path:
    """Find the HDF5 path embedded in a Bluesky/Kafka-style JSON message."""
    candidate = _find_path_candidate(message_text)
    if not candidate:
        raise ValueError("Could not find an HDF5 file path in the Kafka message payload.")
    return _normalize_path(candidate)


def _read_scalar(file_handle: h5py.File, key: str) -> float | str | None:
    if key not in file_handle:
        return None
    value = file_handle[key][()]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.flat[0]
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return str(value)
    return value


def _read_ndattr(file_handle: h5py.File, key: str) -> float | str | None:
    return _read_scalar(file_handle, f"{NDATTR_PREFIX}/{key}")


def _compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _with_mm_suffix(values: dict[str, Any]) -> dict[str, Any]:
    return {f"{key}_mm": value for key, value in values.items() if value is not None}


def _waxs_position_with_units(file_handle: h5py.File) -> dict[str, Any]:
    return _compact_dict(
        {
            "WD_X_mm": _read_ndattr(file_handle, "WD_X"),
            "WD_Y_mm": _read_ndattr(file_handle, "WD_Y"),
            "WD_Z_mm": _read_ndattr(file_handle, "WD_Z"),
            "WD_RY": _read_ndattr(file_handle, "WD_RY"),
        }
    )


def load_hdf5_image(path: str | Path, dataset_path: str = DEFAULT_DATASET_PATH) -> LoadedImage:
    """Load an image for GUI display and collect common detector metadata."""
    resolved = _normalize_path(str(path))
    with h5py.File(resolved, "r") as handle:
        image = handle[dataset_path][()].astype(np.float32)
        detector_name = None
        for candidate in ALL_DETECTORS.keys():
            try:
                detector = detector_factory(candidate)
            except Exception:
                continue
            if getattr(detector, "max_shape", None) == tuple(image.shape):
                detector_name = type(detector).__name__
                break
        metadata = {
            "mono_energy_keV": _read_ndattr(handle, "Mono_Energy"),
            "distance_m": DEFAULT_SAXS_DISTANCE_M,
            "distance_mm": DEFAULT_SAXS_DISTANCE_M * 1000.0,
            "distance_source": "fixed_saxs_geometry",
            "saxs_point_detector": _read_ndattr(handle, "SPDS"),
            "pixel_size_um": _read_ndattr(handle, "Sx"),
            "pixel_size_y_um": _read_ndattr(handle, "Sy"),
            "title": _read_ndattr(handle, "TIFFImageDescription"),
            "detector_name": detector_name,
            "saxs_detector_position": _compact_dict(
                {
                    "SD_X": _read_ndattr(handle, "SD_X"),
                    "SD_Y": _read_ndattr(handle, "SD_Y"),
                }
            ),
            "saxs_detector_position_mm": _with_mm_suffix(
                {
                    "SD_X": _read_ndattr(handle, "SD_X"),
                    "SD_Y": _read_ndattr(handle, "SD_Y"),
                }
            ),
            "waxs_detector_position": _compact_dict(
                {
                    "WD_X": _read_ndattr(handle, "WD_X"),
                    "WD_Y": _read_ndattr(handle, "WD_Y"),
                    "WD_Z": _read_ndattr(handle, "WD_Z"),
                    "WD_RY": _read_ndattr(handle, "WD_RY"),
                }
            ),
            "waxs_detector_position_with_units": _waxs_position_with_units(handle),
            "waxs_distance": _read_ndattr(handle, "WD_Z"),
            "waxs_distance_mm": _read_ndattr(handle, "WD_Z"),
            "waxs_distance_source": "WD_Z",
            "motor_readings": _compact_dict({key: _read_ndattr(handle, key) for key in MOTOR_KEYS}),
        }
    return LoadedImage(path=resolved, dataset_path=dataset_path, image=image, metadata=metadata)
