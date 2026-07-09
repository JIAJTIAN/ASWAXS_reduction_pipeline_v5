"""Bridge the v5 dashboard to the existing XAnoS component GUI."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from types import ModuleType

from PyQt5 import QtCore, QtWidgets

from aswaxs_live.ui_theme import apply_tool_theme, fit_window_to_available_screen


PROJECT_DIR = Path(__file__).resolve().parents[2]
PLATFORM_NAME = os.name
DEFAULT_XANOS_COMPONENTS_PATH = Path(
    os.environ.get(
        "ASWAXS_XANOS_COMPONENTS",
        r"C:\Users\jiajtian\Documents\Playground\XAnoS\XAnoS_Components.py",
    )
)


class XAnoSBridgeError(RuntimeError):
    """Raised when the XAnoS component GUI cannot be loaded."""


_xanos_module: ModuleType | None = None


def resolve_xanos_components_path(path: str | Path | None = None) -> Path:
    candidates = [Path(path).expanduser()] if path else _default_xanos_candidates()
    for candidate in candidates:
        script_path = candidate.resolve()
        if script_path.exists():
            return script_path
    checked = "\n".join(str(candidate) for candidate in candidates)
    if path:
        raise XAnoSBridgeError(
            "Cannot find the XAnoS component GUI.\n\n"
            f"Expected script:\n{Path(path).expanduser()}"
        )
    raise XAnoSBridgeError(
        "Cannot find the XAnoS component GUI.\n\n"
        f"Checked:\n{checked}\n\n"
        "Set ASWAXS_XANOS_COMPONENTS to the XAnoS_Components.py path if it lives somewhere else."
    )


def _default_xanos_candidates() -> list[Path]:
    env_path = os.environ.get("ASWAXS_XANOS_COMPONENTS")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            DEFAULT_XANOS_COMPONENTS_PATH,
            PROJECT_DIR.parent / "XAnoS" / "XAnoS_Components.py",
            PROJECT_DIR.parent.parent / "XAnoS" / "XAnoS_Components.py",
            Path.cwd().parent / "XAnoS" / "XAnoS_Components.py",
            Path.cwd().parent.parent / "XAnoS" / "XAnoS_Components.py",
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not _candidate_is_compatible(candidate):
            continue
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _candidate_is_compatible(candidate: Path) -> bool:
    """Return False for paths that are invalid on the current platform."""
    if PLATFORM_NAME != "nt" and re.match(r"^[A-Za-z]:[\\/]", str(candidate)):
        return False
    return True


def open_xanos_components_window(
    data_files: list[str | Path] | None = None,
    *,
    script_path: str | Path | None = None,
) -> QtWidgets.QWidget:
    """Open XAnoS_Components and optionally preload XAnoS-format .dat files."""
    module = _load_xanos_module(resolve_xanos_components_path(script_path))
    widget_class = getattr(module, "XAnoS_Components", None)
    if widget_class is None:
        raise XAnoSBridgeError("The XAnoS script does not define XAnoS_Components.")

    try:
        import pyqtgraph as pg

        pg.setConfigOptions(background="w", foreground="k")
    except Exception:
        pass

    window = widget_class()
    window.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
    window.setWindowTitle("XAnoS Components - ASAXS Component Extraction")
    apply_tool_theme(window)
    fit_window_to_available_screen(window, (1500, 940), minimum=(900, 640), margin=80)

    files = [str(Path(path)) for path in data_files or [] if str(path).strip()]
    if files:
        window.import_data(files)
        _select_all_imported_data(window)
    window.show()
    window.raise_()
    window.activateWindow()
    return window


def _select_all_imported_data(window: QtWidgets.QWidget) -> None:
    data_list = getattr(window, "dataListWidget", None)
    if data_list is None:
        return
    try:
        data_list.blockSignals(True)
        data_list.selectAll()
        data_list.blockSignals(False)
        data_selection_changed = getattr(window, "dataSelectionChanged", None)
        if callable(data_selection_changed):
            data_selection_changed()
    except Exception:
        try:
            data_list.blockSignals(False)
        except Exception:
            pass


def _load_xanos_module(script_path: Path) -> ModuleType:
    global _xanos_module
    if _xanos_module is not None:
        return _xanos_module

    module_dir = str(script_path.parent)
    added_path = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        added_path = True
    try:
        spec = importlib.util.spec_from_file_location("aswaxs_external_xanos_components", script_path)
        if spec is None or spec.loader is None:
            raise XAnoSBridgeError(f"Cannot load XAnoS script:\n{script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _xanos_module = module
        return module
    except XAnoSBridgeError:
        raise
    except Exception as exc:  # noqa: BLE001 - report optional external GUI failures.
        raise XAnoSBridgeError(f"Could not start XAnoS Components from:\n{script_path}\n\n{exc}") from exc
    finally:
        if added_path:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass
