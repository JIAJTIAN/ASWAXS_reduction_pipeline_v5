"""Run the ASWAXS live GUI without installing the project."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5 import QtWidgets


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if __name__ == "__main__":
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from aswaxs_live.dashboard import main  # noqa: E402

    app.setApplicationName("ASWAXS v5")
    raise SystemExit(main())
