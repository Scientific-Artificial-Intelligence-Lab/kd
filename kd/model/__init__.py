"""Model implementations for PDE discovery."""

from .dlga import DLGA
from .kd_dlga import KD_DLGA
from .kd_dscv import KD_DSCV, KD_DSCV_SPR
from .kd_sga import KD_SGA

# KD_PySR uses Julia via juliacall, which crashes if torch is already loaded
# (https://github.com/pytorch/pytorch/issues/78829).
# Lazy import to avoid triggering Julia init on `import kd.model`.

__all__ = ['DLGA', 'KD_DLGA', 'KD_DSCV', 'KD_DSCV_SPR', 'KD_SGA', 'KD_PySR']


def __getattr__(name: str):
    if name == 'KD_PySR':
        from .kd_pysr import KD_PySR
        return KD_PySR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
