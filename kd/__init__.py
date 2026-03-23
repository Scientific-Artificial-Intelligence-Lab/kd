"""KD (Knowledge Discovery) package for PDE discovery.

This package provides tools for discovering governing equations of PDEs
using deep learning and symbolic regression approaches.
"""

__version__ = '1.0.0'

__pkg_name__ = 'KD'

from .dataset import load_burgers_equation, load_mat_file

_submodules = [
    'dataset',
    'model',
    'viz'
]

_model_names = [
    "DLGA",
    "KD_DLGA",
    "KD_Discover",
    "KD_Discover_SPR",
    "KD_Discover_Regression",
    "KD_SGA",
    "KD_PySR",
    "KD_EqGPT",
]

__all__ = _submodules + [
    "load_burgers_equation",
    "load_mat_file",
    "load_kdv_equation",
    "viz",
] + _model_names


def __getattr__(name: str):
    if name == "viz":
        from . import viz as _viz
        return _viz
    if name in _model_names:
        from . import model as _model
        return getattr(_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
