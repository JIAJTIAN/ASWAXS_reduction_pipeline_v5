"""Source-tree launcher for the FrameByFrame online ZMQ reducer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aswaxs_live.tools.online_reducer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
