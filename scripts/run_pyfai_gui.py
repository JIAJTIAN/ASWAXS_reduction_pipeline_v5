"""Launch the HDF5-to-pyFAI calibration and mask GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aswaxs_live.preprocessing.gui import run_app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="HDF5-to-pyFAI setup GUI for ASWAXS calibration and mask authoring.")
    parser.add_argument("--file", help="Optional HDF5 file to load on startup.")
    args = parser.parse_args()
    return run_app(args.file)


if __name__ == "__main__":
    raise SystemExit(main())
