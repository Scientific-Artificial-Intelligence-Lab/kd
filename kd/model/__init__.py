"""Model implementations for PDE discovery."""

__all__ = [
    'DLGA',
    'KD_DLGA',
    'KD_Discover',
    'KD_Discover_SPR',
    'KD_Discover_Regression',
    'KD_SGA',
    'KD_PySR',
]


def __getattr__(name: str):
    if name == 'DLGA':
        from .dlga import DLGA
        return DLGA
    if name == 'KD_DLGA':
        from .kd_dlga import KD_DLGA
        return KD_DLGA
    if name == 'KD_Discover':
        from .kd_discover import KD_Discover
        return KD_Discover
    if name == 'KD_Discover_SPR':
        from .kd_discover import KD_Discover_SPR
        return KD_Discover_SPR
    if name == 'KD_Discover_Regression':
        from .kd_discover_regression import KD_Discover_Regression
        return KD_Discover_Regression
    if name == 'KD_SGA':
        from .kd_sga import KD_SGA
        return KD_SGA
    if name == 'KD_PySR':
        from .kd_pysr import KD_PySR
        return KD_PySR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
