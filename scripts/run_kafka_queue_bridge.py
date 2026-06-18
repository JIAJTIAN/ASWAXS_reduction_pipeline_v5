"""Bridge Bluesky/Kafka measurement_done messages into the v5 reducer queue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aswaxs_live.kafka_bridge import replay_jsonl_messages, run_bluesky_kafka_bridge  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge Bluesky/Kafka measurement_done messages to v3 JSONL jobs.")
    parser.add_argument("--queue", required=True, help="Reducer measurement_done_queue.jsonl path.")
    parser.add_argument("--bootstrap-servers", help="Kafka bootstrap servers, for example host:9092.")
    parser.add_argument("--topic", action="append", default=[], help="Kafka topic. Repeat for multiple topics.")
    parser.add_argument("--group-id", default="aswaxs-v3-reduction-bridge")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Main acquisition root. Bluesky start docs with sampleName queue data-root/sampleName/detector.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Main analysis root. Queue jobs carry output-root/sampleName/detector when provided.",
    )
    parser.add_argument(
        "--detector",
        action="append",
        default=[],
        help="Detector folder name to derive from sampleName. Repeat for multiple detectors.",
    )
    parser.add_argument(
        "--replay-jsonl",
        default=None,
        help="Optional local JSONL file of Kafka-like messages to convert once instead of opening Kafka.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    detectors = args.detector or ["Pil300K", "Eig1M"]
    if args.replay_jsonl:
        count = replay_jsonl_messages(
            args.queue,
            args.replay_jsonl,
            data_root=args.data_root,
            output_root=args.output_root,
            detectors=detectors,
        )
        print(f"Queued {count} measurement_done jobs from {args.replay_jsonl}")
        return 0
    if not args.bootstrap_servers or not args.topic:
        raise ValueError("Provide --bootstrap-servers and at least one --topic, or use --replay-jsonl.")
    run_bluesky_kafka_bridge(
        bootstrap_servers=args.bootstrap_servers,
        topics=args.topic,
        queue_path=args.queue,
        data_root=args.data_root,
        output_root=args.output_root,
        detectors=detectors,
        group_id=args.group_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
