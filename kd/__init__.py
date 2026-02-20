"""KD (Knowledge Discovery) package for PDE discovery.

This package provides tools for discovering governing equations of PDEs
using deep learning and symbolic regression approaches.
"""

__version__ = '1.0.0'

__pkg_name__ = 'KD'

from .dataset import load_burgers_equation, load_mat_file
from . import viz

_submodules = [
    'dataset',
    'model',
    'viz'
]

__all__ = _submodules + [
    "load_burgers_equation",
    "load_mat_file",
    "load_kdv_equation", 
    "DLGA",
    "KD_DLGA",
    "KD_DSCV",
    "KD_DSCV_Pinn",
    "viz"
]