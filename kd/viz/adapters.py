"""Adapter registry façade wiring model-specific implementations."""

from __future__ import annotations

from ._adapters import DLGAVizAdapter, DiscoverRegressionVizAdapter, DiscoverVizAdapter, EqGPTVizAdapter, PySRVizAdapter, SGAVizAdapter
from .registry import register_adapter

__all__ = [
    'DLGAVizAdapter',
    'DiscoverRegressionVizAdapter',
    'DiscoverVizAdapter',
    'EqGPTVizAdapter',
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
        from kd.model.kd_discover import KD_Discover  # type: ignore
    except Exception:  # pragma: no cover
        KD_Discover = None  # type: ignore
    else:
        register_adapter(KD_Discover, DiscoverVizAdapter())

    try:
        from kd.model.kd_discover_regression import KD_Discover_Regression  # type: ignore
    except Exception:  # pragma: no cover
        KD_Discover_Regression = None  # type: ignore
    else:
        register_adapter(KD_Discover_Regression, DiscoverRegressionVizAdapter())

    try:
        from kd.model.kd_sga import KD_SGA  # type: ignore
    except Exception:  # pragma: no cover
        KD_SGA = None  # type: ignore
    else:
        register_adapter(KD_SGA, SGAVizAdapter())

    # PySR adapter is registered lazily to avoid importing juliacall at
    # package load time (torch + juliacall SIGABRT crash).
    # See: https://github.com/pytorch/pytorch/issues/78829
    from .registry import register_lazy_adapter
    register_lazy_adapter("KD_PySR", _resolve_pysr)

    # EqGPT adapter is registered lazily to avoid importing torch at
    # package load time (EqGPT depends on torch).
    register_lazy_adapter("KD_EqGPT", _resolve_eqgpt)


def _resolve_pysr():
    """Lazily resolve KD_PySR class and its adapter."""
    try:
        from kd.model.kd_pysr import KD_PySR  # type: ignore
    except Exception:  # pragma: no cover
        return None, None
    return KD_PySR, PySRVizAdapter()


def _resolve_eqgpt():
    """Lazily resolve KD_EqGPT class and its adapter."""
    try:
        from kd.model.kd_eqgpt import KD_EqGPT  # type: ignore
    except Exception:  # pragma: no cover
        return None, None
    return KD_EqGPT, EqGPTVizAdapter()
