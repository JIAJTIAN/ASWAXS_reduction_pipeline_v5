"""Small queue helpers for Bluesky-assisted live reduction.

The beamline side should publish one ``measurement_done`` JSON object when a
measurement is complete. The reducer does not trust the message as data; it uses
the message as a reduction job, then scans the specified detector folder and
runs the same read-only HDF5 checks used by the folder watcher.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReductionJob:
    """One Bluesky measurement queued for SAXS/ASAXS reduction."""

    event: str
    uid: str | None
    scan_id: int | str | None
    sample_name: str | None
    detector: str | None
    data_dir: Path
    output_dir: Path | None
    analysis_mode: str | None
    measurement_type: str | None
    num_energies: int | None
    num_groups: int | None
    num_frames: int | None
    raw: dict[str, Any]


class ReductionJobQueueReader:
    """Tail a JSONL reduction-job queue file without deleting messages."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._offset = 0

    def poll(self) -> list[ReductionJob]:
        """Return newly appended valid reduction jobs."""
        if not self.path.exists():
            return []
        messages: list[ReductionJob] = []
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self._offset)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = parse_reduction_job(payload)
                if message is not None:
                    messages.append(message)
            self._offset = handle.tell()
        return messages


def parse_reduction_job(payload: dict[str, Any]) -> ReductionJob | None:
    """Convert a JSON object into a reduction job when possible.

    ``measurement_done`` is the preferred v3 event. ``measurement_aborted`` and
    ``measurement_stopped`` are accepted as finite jobs too, so the reducer can
    conclude whatever frames arrived before acquisition stopped. ``frame_done``
    is accepted as a compatibility alias for early v3 tests. ``sample_active``
    is used by the Bluesky/Kafka bridge when it only knows the current sample
    name and the reducer should derive and watch the detector folder.
    """
    event = str(payload.get("event") or "").strip()
    if event not in {"measurement_done", "measurement_aborted", "measurement_stopped", "frame_done", "sample_active", "sample_started"}:
        return None
    data_dir = payload.get("data_dir")
    if not data_dir:
        return None
    return ReductionJob(
        event=event,
        uid=_optional_text(payload.get("uid")),
        scan_id=payload.get("scan_id"),
        sample_name=_optional_text(payload.get("sample_name")),
        detector=_optional_text(payload.get("detector")),
        data_dir=Path(str(data_dir)).expanduser().resolve(),
        output_dir=_optional_path(payload.get("output_dir")),
        analysis_mode=_optional_text(payload.get("analysis_mode")),
        measurement_type=_optional_text(payload.get("measurement_type")),
        num_energies=_optional_int(payload.get("num_energies")),
        num_groups=_optional_int(payload.get("num_groups")),
        num_frames=_optional_int(payload.get("num_frames")),
        raw=payload,
    )


def append_measurement_done_message(
    queue_path: str | Path,
    *,
    event: str = "measurement_done",
    uid: str | None = None,
    scan_id: int | str | None = None,
    sample_name: str | None = None,
    detector: str | None = None,
    analysis_mode: str | None = "saxs",
    measurement_type: str | None = None,
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    num_energies: int | None = None,
    num_groups: int | None = None,
    num_frames: int | None = None,
) -> Path:
    """Append a test/beamline-style measurement_done record to a JSONL queue."""
    if event not in {"measurement_done", "measurement_aborted", "measurement_stopped", "frame_done", "sample_active", "sample_started"}:
        raise ValueError(f"Unsupported reduction queue event: {event}")
    path = Path(queue_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": event,
        "uid": uid,
        "scan_id": scan_id,
        "sample_name": sample_name,
        "detector": detector,
        "analysis_mode": analysis_mode,
        "measurement_type": measurement_type,
        "data_dir": str(Path(data_dir).expanduser().resolve()),
    }
    if num_energies is not None:
        payload["num_energies"] = int(num_energies)
    if num_groups is not None:
        payload["num_groups"] = int(num_groups)
    if num_frames is not None:
        payload["num_frames"] = int(num_frames)
    if output_dir is not None:
        payload["output_dir"] = str(Path(output_dir).expanduser().resolve())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


FrameDoneMessage = ReductionJob
FrameDoneQueueReader = ReductionJobQueueReader


def parse_frame_done_message(payload: dict[str, Any]) -> ReductionJob | None:
    """Backward-compatible parser name for early v3 tests."""
    return parse_reduction_job(payload)


def append_frame_done_message(
    queue_path: str | Path,
    *,
    uid: str | None = None,
    scan_id: int | str | None = None,
    sample_name: str | None = None,
    detector: str | None = None,
    data_dir: str | Path,
) -> Path:
    """Backward-compatible writer for early v3 tests."""
    return append_measurement_done_message(
        queue_path,
        uid=uid,
        scan_id=scan_id,
        sample_name=sample_name,
        detector=detector,
        data_dir=data_dir,
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_path(value: Any) -> Path | None:
    text = _optional_text(value)
    if text is None:
        return None
    return Path(text).expanduser().resolve()
