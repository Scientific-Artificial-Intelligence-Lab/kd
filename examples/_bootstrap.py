"""Small helper to make examples runnable without installing the package.

Usage from an example script::

    from _bootstrap import ensure_project_root_on_syspath
    PROJECT_ROOT = ensure_project_root_on_syspath()

This allows running examples via ``python examples/xxx.py`` directly,
without requiring ``pip install -e .`` beforehand.
"""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_project_root_on_syspath() -> Path:
    """Ensure the repository root is present on ``sys.path``.

    Returns:
        Path: The resolved project root directory.
    """
    project_root = Path(__file__).resolve().parent.parent
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.append(root_str)
    return project_root


__all__ = ["ensure_project_root_on_syspath"]

