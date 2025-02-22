"""Residual analysis plotting functionality.

This module provides classes for creating various types of residual plots,
including heatmaps, time slice comparisons, and histograms.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List, Tuple

from ..core.base import BasePlot, CompositePlot


class ResidualHeatmap(BasePlot):
    """Heatmap visualization of residuals."""
    
    def _validate_data(self, u_pred: np.ndarray, u_exact: np.ndarray) -> bool:
        """Validate input data shapes match."""
        if u_pred.shape != u_exact.shape:
            raise ValueError(f"Shape mismatch: u_pred {u_pred.shape} != u_exact {u_exact.shape}")
        return True
    
    def plot(self, u_pred: np.ndarray, u_exact: np.ndarray, 
            x: np.ndarray, t: np.ndarray):
        """Create residual heatmap.
        
        Args:
            u_pred: Predicted solution array (nt x nx)
            u_exact: Exact solution array (nt x nx)
            x: Spatial coordinates
            t: Time coordinates
        """
        self._validate_data(u_pred, u_exact)
        residual = u_pred - u_exact
        
        self.fig, self.axes = plt.subplots(figsize=self.figsize)
        im = self.axes.imshow(residual,
                            extent=[t.min(), t.max(), x.min(), x.max()],
                            aspect='auto',
                            origin='lower',
                            cmap='RdBu')
        
        # Add colorbar
        plt.colorbar(im, ax=self.axes)
        
        # Configure axes
        self.axes.set_xlabel('t')
        self.axes.set_ylabel('x')
        self.axes.set_title('Residual u_pred - u_exact')


class ResidualTimeslice(BasePlot):
    """Time slice comparison of residuals."""
    
    def plot(self, u_pred: np.ndarray, u_exact: np.ndarray,
            x: np.ndarray, t: np.ndarray,
            slice_times: List[float] = [0.25, 0.5, 0.75]):
        """Create time slice residual plots.
        
        Args:
            u_pred: Predicted solution array (nt x nx)
            u_exact: Exact solution array (nt x nx)
            x: Spatial coordinates
            t: Time coordinates
            slice_times: List of time points to plot
        """
        self._validate_data(u_pred, u_exact)
        residual = u_pred - u_exact
        
        self.fig, self.axes = plt.subplots(figsize=self.figsize)
        
        for t_slice in slice_times:
            t_idx = int(t_slice * len(t))
            self.axes.plot(x, residual[t_idx], 
                         label=f't = {t_slice:.2f}')
        
        self.axes.set_xlabel('x')
        self.axes.set_ylabel('Residual')
        self.axes.grid(True, linestyle=':')
        self.axes.legend()


class ResidualHistogram(BasePlot):
    """Histogram of residual distribution."""
    
    def plot(self, u_pred: np.ndarray, u_exact: np.ndarray,
            bins: int = 50):
        """Create residual histogram.
        
        Args:
            u_pred: Predicted solution array
            u_exact: Exact solution array
            bins: Number of histogram bins
        """
        self._validate_data(u_pred, u_exact)
        residual = u_pred - u_exact
        
        self.fig, self.axes = plt.subplots(figsize=self.figsize)
        
        self.axes.hist(residual.flatten(), bins=bins, 
                      density=True, alpha=0.7)
        self.axes.set_xlabel('Residual')
        self.axes.set_ylabel('Density')
        self.axes.grid(True, linestyle=':')


class ResidualAnalysis(CompositePlot):
    """Comprehensive residual analysis with multiple plot types."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create figure with GridSpec
        self.fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2)
        
        # Create axes
        self.axes = {
            'heatmap': self.fig.add_subplot(gs[0, :]),
            'timeslice': self.fig.add_subplot(gs[1, 0]),
            'histogram': self.fig.add_subplot(gs[1, 1])
        }
        
    def plot(self, u_pred: np.ndarray, u_exact: np.ndarray,
            x: np.ndarray, t: np.ndarray,
            slice_times: List[float] = [0.25, 0.5, 0.75]):
        """Create comprehensive residual analysis plots.
        
        Args:
            u_pred: Predicted solution array (nt x nx)
            u_exact: Exact solution array (nt x nx)
            x: Spatial coordinates
            t: Time coordinates
            slice_times: List of time points to plot
        """
        residual = u_pred - u_exact
        
        # Heatmap
        im = self.axes['heatmap'].imshow(residual,
                                       extent=[t.min(), t.max(), x.min(), x.max()],
                                       aspect='auto',
                                       origin='lower',
                                       cmap='RdBu')
        plt.colorbar(im, ax=self.axes['heatmap'])
        self.axes['heatmap'].set_title('Residual Distribution')
        
        # Time slices
        for t_slice in slice_times:
            t_idx = int(t_slice * len(t))
            self.axes['timeslice'].plot(x, residual[t_idx],
                                      label=f't = {t_slice:.2f}')
        self.axes['timeslice'].set_title('Residual at Different Times')
        self.axes['timeslice'].legend()
        
        # Histogram
        self.axes['histogram'].hist(residual.flatten(), bins=50,
                                  density=True, alpha=0.7)
        self.axes['histogram'].set_title('Residual Histogram')
        
        # Adjust layout
        self.fig.tight_layout() 