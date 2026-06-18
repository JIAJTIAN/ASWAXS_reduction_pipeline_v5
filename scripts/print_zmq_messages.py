"""Print live Bluesky/ZeroMQ messages for connection testing.

This is a diagnostic script. It does not reduce data and does not write analysis
files. It subscribes to a ZMQ endpoint, prints each multipart message, and tries
common decoders so we can see what the beamline server is actually sending.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from typing import Any

import zmq


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print live ZMQ messages from a Bluesky-style stream.")
    parser.add_argument("--address", default="tcp://127.0.0.1:5578", help="ZMQ SUB endpoint to connect to.")
    parser.add_argument("--topic", default="", help="Subscription topic prefix. Empty subscribes to everything.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="Stop if no message arrives in this time.")
    parser.add_argument("--max-bytes", type=int, default=1200, help="Maximum bytes to print for one message part.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    socket.setsockopt(zmq.RCVTIMEO, int(max(args.timeout_seconds, 0.1) * 1000))
    socket.connect(args.address)

    print(f"Listening on {args.address}")
    print(f"Topic prefix: {args.topic!r}")
    print(f"Timeout: {args.timeout_seconds:g} s without messages")
    count = 0
    try:
        while True:
            try:
                parts = socket.recv_multipart()
            except zmq.Again:
                print("No ZMQ messages received before timeout.")
                return 1 if count == 0 else 0
            count += 1
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] message {count}, parts={len(parts)}")
            for index, part in enumerate(parts):
                print(f"  part[{index}] bytes={len(part)}")
                decoded = decode_part(part)
                print_limited(decoded, args.max_bytes, prefix="    ")
    finally:
        socket.close(linger=0)


def decode_part(part: bytes) -> Any:
    """Try likely encodings without assuming the beamline transport format."""
    for decoder in (decode_json, decode_pickle, decode_utf8):
        value = decoder(part)
        if value is not None:
            return value
    return {"binary_hex": part[:120].hex(), "note": "unrecognized binary payload"}


def decode_json(part: bytes) -> Any | None:
    try:
        return json.loads(part.decode("utf-8"))
    except Exception:
        return None


def decode_pickle(part: bytes) -> Any | None:
    try:
        return pickle.loads(part)
    except Exception:
        return None


def decode_utf8(part: bytes) -> str | None:
    try:
        return part.decode("utf-8")
    except Exception:
        return None


def print_limited(value: Any, max_bytes: int, prefix: str = "") -> None:
    text = value if isinstance(value, str) else repr(value)
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) > max_bytes:
        text = encoded[:max_bytes].decode("utf-8", errors="replace") + " ... <truncated>"
    for line in text.splitlines() or [""]:
        print(prefix + line)


if __name__ == "__main__":
    raise SystemExit(main())
