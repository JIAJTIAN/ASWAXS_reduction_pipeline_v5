"""Application entry point shared by installed and source-tree launchers."""

from __future__ import annotations

import sys

from aswaxs_live.qt_runtime import suppress_glx_warning


def _quiet_server_glx_warning() -> None:
    """Suppress noisy Qt/GLX warnings seen on remote Linux displays."""
    suppress_glx_warning()


def main() -> int:
    """Start the FrameByFrame-ASWAXS dashboard."""
    _quiet_server_glx_warning()

    from PyQt5 import QtWidgets

    from aswaxs_live.dashboard import DashboardWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("FrameByFrame-ASWAXS")
    app.setOrganizationName("ChemMatCARS")
    window = DashboardWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
