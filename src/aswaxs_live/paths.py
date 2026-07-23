"""Stable project paths that do not depend on a module's package depth."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SOURCE_DIR.parent
PLAYGROUND_DIR = PROJECT_DIR.parent

