"""Shared helpers for non-invasive links to external scientific GUIs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from aswaxs_live.paths import PROJECT_DIR

PLATFORM_NAME = os.name


class ExternalToolError(RuntimeError):
    """Raised when an optional external tool cannot be found or started."""


def compatible_candidates(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if PLATFORM_NAME != "nt" and re.match(r"^[A-Za-z]:[\\/]", str(path)):
            continue
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def resolve_script(
    *,
    explicit: str | Path | None,
    candidates: list[Path],
    tool_name: str,
    env_var: str,
) -> Path:
    search_paths = [Path(explicit).expanduser()] if explicit else compatible_candidates(candidates)
    for candidate in search_paths:
        script_path = candidate.resolve()
        if script_path.exists() and script_path.is_file():
            return script_path
    checked = "\n".join(str(candidate) for candidate in search_paths)
    if explicit:
        raise ExternalToolError(
            f"Cannot find {tool_name}.\n\n"
            f"Expected script:\n{Path(explicit).expanduser()}"
        )
    raise ExternalToolError(
        f"Cannot find {tool_name}.\n\n"
        f"Checked:\n{checked}\n\n"
        f"Set {env_var} to the tool's launcher script if it lives somewhere else."
    )


def launch_python_script(script_path: str | Path, *, extra_args: list[str] | None = None) -> subprocess.Popen[str]:
    script = Path(script_path).expanduser().resolve()
    env = os.environ.copy()
    root = str(script.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([root, existing_pythonpath]) if existing_pythonpath else root
    command = [sys.executable, str(script), *(extra_args or [])]
    return subprocess.Popen(command, cwd=root, env=env, text=True)
