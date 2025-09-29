"""Adapter registry façade wiring model-specific implementations."""

from __future__ import annotations

from ._adapters import DLGAVizAdapter, DSCVVizAdapter
from .registry import register_adapter

__all__ = [
    'DLGAVizAdapter',
    'DSCVVizAdapter',
    'register_default_adapters',
]


def register_default_adapters() -> None:
    try:
        from kd.model.kd_dlga import KD_DLGA  # type: ignore
    except Exception:  # pragma: no cover
        KD_DLGA = None  # type: ignore
    else:
        register_adapter(KD_DLGA, DLGAVizAdapter())

    try:
        from kd.model.kd_dscv import KD_DSCV  # type: ignore
    except Exception:  # pragma: no cover
        KD_DSCV = None  # type: ignore
    else:
        register_adapter(KD_DSCV, DSCVVizAdapter())
