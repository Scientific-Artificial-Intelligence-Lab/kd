"""KD (Knowledge Discovery) package for PDE discovery.

This package provides tools for discovering governing equations of PDEs
using deep learning and symbolic regression approaches.
"""

__version__ = '0.1.0'

__pkg_name__ = 'KD'

from . import base
from . import data
from . import model
from . import utils

__all__ = ['base', 'data', 'model', 'utils']