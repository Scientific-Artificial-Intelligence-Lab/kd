"""Styling configuration for KD visualization façade."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

DEFAULT_STYLE: Dict[str, Any] = {
    'font.size': 12,
    'figure.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}


@dataclass
class VizConfig:
    style: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_STYLE))
    save_dir: Optional[Path] = None
    backend: Optional[str] = None


_CONFIG = VizConfig()


def configure(
    *,
    style: Optional[Dict[str, Any]] = None,
    save_dir: Optional[Path] = None,
    backend: Optional[str] = None,
) -> VizConfig:
    if style is not None:
        _CONFIG.style = dict(DEFAULT_STYLE)
        _CONFIG.style.update(style)
    if save_dir is not None:
        _CONFIG.save_dir = Path(save_dir)
    if backend is not None:
        matplotlib.use(backend, force=True)
        _CONFIG.backend = backend
    return _CONFIG


def get_config() -> VizConfig:
    return _CONFIG


@contextmanager
def style_context(extra_style: Optional[Dict[str, Any]] = None):
    original = matplotlib.rcParams.copy()
    try:
        combined = dict(_CONFIG.style)
        if extra_style:
            combined.update(extra_style)
        matplotlib.rcParams.update(combined)
        yield
    finally:
        matplotlib.rcParams.update(original)
