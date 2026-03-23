"""Model-specific visualization adapters."""

from .dlga import DLGAVizAdapter
from .discover import DiscoverVizAdapter
from .discover_regression import DiscoverRegressionVizAdapter
from .eqgpt import EqGPTVizAdapter
from .sga import SGAVizAdapter
from .pysr import PySRVizAdapter

__all__ = [
    'DLGAVizAdapter',
    'DiscoverRegressionVizAdapter',
    'DiscoverVizAdapter',
    'EqGPTVizAdapter',
    'SGAVizAdapter',
    'PySRVizAdapter',
]
