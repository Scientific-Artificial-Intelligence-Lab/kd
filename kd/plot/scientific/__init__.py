"""Scientific plotting module for visualization."""

from kd.plot.scientific.metadata_plane import MetadataValuePlane
from .residual import ResidualAnalysis
from .comparison import ComparisonAnalysis
from .equation import TermsHeatmap, TermsAnalysis, calculate_metadata, calculate_equation_residual

__all__ = [
    'MetadataValuePlane',
    'ResidualAnalysis',
    'ComparisonAnalysis',
    'TermsHeatmap',
    'TermsAnalysis',
    'calculate_metadata',
    'calculate_equation_residual',
]
