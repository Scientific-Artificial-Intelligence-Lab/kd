"""Helper utilities for KD visualization package."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_output_path(filename: str, save_dir: Optional[Path]) -> Path:
    """Return a filesystem path for saving visual artifacts."""

    base = save_dir if save_dir is not None else Path.cwd()
    path = base / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
