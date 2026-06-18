"""Append a test Bluesky/Kafka-style measurement_done job to a JSONL queue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aswaxs_live.bluesky_queue import append_measurement_done_message  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append one measurement_done reduction job to a v3 JSONL queue.")
    parser.add_argument("--queue", required=True, help="Path to measurement_done_queue.jsonl.")
    parser.add_argument("--data-dir", required=True, help="Detector data directory to reduce.")
    parser.add_argument("--output-dir", default=None, help="Detector analysis output directory.")
    parser.add_argument("--uid", default=None, help="Bluesky run UID.")
    parser.add_argument("--scan-id", default=None, help="Bluesky scan_id.")
    parser.add_argument("--sample-name", default=None, help="Sample/run name.")
    parser.add_argument("--detector", default=None, help="Detector name, for example Pil300K or Eig1M.")
    parser.add_argument("--analysis-mode", default="saxs", choices=["saxs", "asaxs"])
    parser.add_argument("--measurement-type", default="normal_saxs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = append_measurement_done_message(
        args.queue,
        uid=args.uid,
        scan_id=args.scan_id,
        sample_name=args.sample_name,
        detector=args.detector,
        analysis_mode=args.analysis_mode,
        measurement_type=args.measurement_type,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print(f"Appended measurement_done reduction job to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
