"""Adapter registry façade wiring model-specific implementations."""

from __future__ import annotations

from ._adapters import DLGAVizAdapter, DSCVVizAdapter, PySRVizAdapter, SGAVizAdapter
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

    try:
        from kd.model.kd_sga import KD_SGA  # type: ignore
    except Exception:  # pragma: no cover
        KD_SGA = None  # type: ignore
    else:
        register_adapter(KD_SGA, SGAVizAdapter())

    # PySR 是可选依赖：只有在 kd.model.kd_pysr 能成功导入时才注册 adapter
    try:
        from kd.model.kd_pysr import KD_PySR  # type: ignore
    except Exception:  # pragma: no cover
        KD_PySR = None  # type: ignore
    else:
        register_adapter(KD_PySR, PySRVizAdapter())
