from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py

from .identity import ExperimentIdentity


CHECKPOINT_SCHEMA_VERSION = "1.0"


def parameter_signature(values: dict[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_experiment_metadata(analysis_h5: Path, identity: ExperimentIdentity) -> None:
    with h5py.File(analysis_h5, "a") as handle:
        entry = handle.require_group("entry")
        _set_value(entry, "title", identity.title)
        experiment = entry.require_group("experiment")
        experiment.attrs["NX_class"] = "NXcollection"
        for key, value in identity.metadata().items():
            _set_value(experiment, key, value)


def write_stage_checkpoint(
    analysis_h5: Path,
    stage: str,
    *,
    identity: ExperimentIdentity,
    status: str,
    output_group_path: str,
    expected_items: int,
    written_items: int,
    parameters: dict[str, Any],
    input_checkpoint_ids: list[str] | None = None,
    validation_message: str = "",
) -> str:
    if status not in {"partial", "complete", "invalid"}:
        raise ValueError(f"Unsupported checkpoint status: {status}")
    signature = parameter_signature(parameters)
    checkpoint_uid = hashlib.sha256(
        f"{identity.analysis_uid}:{stage}:{signature}".encode("utf-8")
    ).hexdigest()
    with h5py.File(analysis_h5, "a") as handle:
        checkpoints = handle.require_group("entry").require_group("checkpoints")
        checkpoints.attrs["NX_class"] = "NXcollection"
        checkpoints.attrs["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
        group = checkpoints.require_group(stage)
        group.attrs["NX_class"] = "NXprocess"
        values = {
            "stage": stage,
            "status": status,
            "checkpoint_uid": checkpoint_uid,
            "analysis_uid": identity.analysis_uid,
            "experiment_uid": identity.experiment_uid,
            "output_group_path": output_group_path,
            "expected_items": int(expected_items),
            "written_items": int(written_items),
            "parameter_signature": signature,
            "input_checkpoint_ids_json": json.dumps(input_checkpoint_ids or []),
            "validation_status": "pending_reader_validation" if status == "complete" else "pending",
            "validation_message": validation_message,
        }
        for key, value in values.items():
            _set_value(group, key, value)
        parameters_group = group.require_group("parameters")
        for key in list(parameters_group):
            del parameters_group[key]
        for key, value in parameters.items():
            _set_value(parameters_group, key, value)
    return checkpoint_uid


def read_checkpoint_uid(analysis_h5: Path, stage: str) -> str:
    try:
        with h5py.File(analysis_h5, "r") as handle:
            value = handle[f"/entry/checkpoints/{stage}/checkpoint_uid"][()]
    except (OSError, KeyError):
        return ""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _set_value(group: h5py.Group, name: str, value: Any) -> None:
    if name in group:
        del group[name]
    if isinstance(value, str):
        group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))
    else:
        group.create_dataset(name, data=value)
