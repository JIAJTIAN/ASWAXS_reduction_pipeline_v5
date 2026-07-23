"""Qt runtime configuration shared by GUI launchers and child processes."""

from __future__ import annotations

import os
from collections.abc import MutableMapping


def suppress_glx_warning(environment: MutableMapping[str, str] | None = None) -> None:
    """Disable the noisy qt.glx category without replacing other Qt rules."""
    target = environment if environment is not None else os.environ
    rule = "qt.glx=false"
    existing = target.get("QT_LOGGING_RULES", "").strip()
    if not existing:
        target["QT_LOGGING_RULES"] = rule
    elif rule not in existing.split(";"):
        target["QT_LOGGING_RULES"] = f"{existing};{rule}"
