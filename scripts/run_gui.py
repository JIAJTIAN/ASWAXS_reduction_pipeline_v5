"""Run the ASWAXS live GUI without installing the project."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _quiet_server_glx_warning() -> None:
    """Suppress noisy Qt/GLX warnings seen on remote Linux displays."""
    rule = "qt.glx=false"
    existing = os.environ.get("QT_LOGGING_RULES", "").strip()
    if not existing:
        os.environ["QT_LOGGING_RULES"] = rule
    elif rule not in existing:
        os.environ["QT_LOGGING_RULES"] = f"{existing};{rule}"


_quiet_server_glx_warning()

from PyQt5 import QtWidgets  # noqa: E402


if __name__ == "__main__":
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from aswaxs_live.dashboard import main  # noqa: E402

    app.setApplicationName("ASWAXS v5")
    raise SystemExit(main())
