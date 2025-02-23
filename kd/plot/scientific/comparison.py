"""Comparison plotting functionality.

This module provides classes for creating various types of comparison plots,
including scatter plots with error bars, 45-degree line plots, and
comprehensive comparison analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional
import os

from ..core.base import BasePlot, CompositePlot


class ComparisonScatter(BasePlot):
    """Scatter plot with error bars for noisy data comparison."""
    
    def _validate_data(self, y_true: np.ndarray, y_pred: np.ndarray) -> bool:
        """Validate input data shapes match."""
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")
        return True
    
    def plot(self, y_true: np.ndarray, y_pred: np.ndarray,
            error_bars: bool = True,
            confidence_interval: float = 0.95):
        """Create scatter plot with optional error bars.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            error_bars: Whether to show error bars
            confidence_interval: Confidence interval for error bars
        """
        self._validate_data(y_true, y_pred)
        
        # Create figure
        self.fig, self.axes = plt.subplots(figsize=self.figsize)
        
        if error_bars:
            # Calculate error bars
            error = np.abs(y_true - y_pred)
            ci = stats.t.interval(confidence_interval, len(error)-1,
                                loc=np.mean(error), scale=stats.sem(error))
            yerr = ci[1] - np.mean(error)
            
            # Plot with error bars
            self.axes.errorbar(y_true, y_pred, yerr=yerr,
                             fmt='o', alpha=0.5, capsize=2)
        else:
            # Simple scatter plot
            self.axes.scatter(y_true, y_pred, alpha=0.5)
        
        # Add 45-degree line
        lims = [
            np.min([self.axes.get_xlim(), self.axes.get_ylim()]),
            np.max([self.axes.get_xlim(), self.axes.get_ylim()])
        ]
        self.axes.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        
        # Configure axes
        self.axes.set_xlabel('True Values')
        self.axes.set_ylabel('Predicted Values')
        self.axes.set_title('Prediction vs Truth')
        self.axes.grid(True, linestyle=':')
        
        # Save if path is provided
        if self.save_path:
            self.save()


class ComparisonLine(BasePlot):
    """45-degree line plot for comparing predictions with truth."""
    
    def plot(self, y_true: np.ndarray, y_pred: np.ndarray,
            show_bounds: bool = True,
            show_statistics: bool = True):
        """Create 45-degree line plot.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            show_bounds: Whether to show confidence bounds
            show_statistics: Whether to show statistical metrics
        """
        self._validate_data(y_true, y_pred)
        
        # Create figure
        self.fig, self.axes = plt.subplots(figsize=self.figsize)
        
        # Plot comparison
        self.axes.scatter(y_true, y_pred, alpha=0.5, label='Data')
        
        # Add 45-degree line
        lims = [
            np.min([self.axes.get_xlim(), self.axes.get_ylim()]),
            np.max([self.axes.get_xlim(), self.axes.get_ylim()])
        ]
        self.axes.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Fit')
        
        if show_bounds:
            # Add confidence bounds
            std = np.std(y_pred - y_true)
            self.axes.fill_between(lims,
                                 [l - 2*std for l in lims],
                                 [l + 2*std for l in lims],
                                 alpha=0.1, color='k',
                                 label='95% Confidence')
        
        if show_statistics:
            # Calculate and display statistics
            r2 = stats.pearsonr(y_true, y_pred)[0]**2
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            stats_text = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}'
            self.axes.text(0.05, 0.95, stats_text,
                         transform=self.axes.transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Configure axes
        self.axes.set_xlabel('True Values')
        self.axes.set_ylabel('Predicted Values')
        self.axes.set_title('Prediction vs Truth')
        self.axes.grid(True, linestyle=':')
        self.axes.legend()
        
        # Save if path is provided
        if self.save_path:
            self.save()


class ComparisonAnalysis(CompositePlot):
    """Comprehensive comparison analysis with multiple plot types."""
    
    def plot(self, y_true: np.ndarray, y_pred: np.ndarray,
            show_error_dist: bool = True,
            show_statistics: bool = True):
        """Create comprehensive comparison analysis plots.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            show_error_dist: Whether to show error distribution
            show_statistics: Whether to show statistical metrics
        """
        # Create figure with GridSpec
        self.fig = plt.figure(figsize=(12, 8))
        gs = plt.GridSpec(2, 2)
        
        # Scatter plot
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                'k--', alpha=0.5)
        ax1.set_title('Scatter Plot')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        
        # Error distribution
        if show_error_dist:
            ax2 = self.fig.add_subplot(gs[0, 1])
            error = y_pred - y_true
            ax2.hist(error, bins=30, density=True, alpha=0.7)
            ax2.set_title('Error Distribution')
            ax2.set_xlabel('Error')
            ax2.set_ylabel('Density')
            
            # Add normal fit
            mu, std = np.mean(error), np.std(error)
            x = np.linspace(error.min(), error.max(), 100)
            p = stats.norm.pdf(x, mu, std)
            ax2.plot(x, p, 'k--', alpha=0.7)
        
        # Statistics
        if show_statistics:
            ax3 = self.fig.add_subplot(gs[1, :])
            stats_text = self._generate_statistics(y_true, y_pred)
            ax3.text(0.5, 0.5, stats_text,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white'))
            ax3.axis('off')
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Save if path is provided
        if self.save_path:
            self.save()
    
    def _generate_statistics(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate statistical summary."""
        error = y_pred - y_true
        stats_items = [
            ('R²', stats.pearsonr(y_true, y_pred)[0]**2),
            ('RMSE', np.sqrt(np.mean(error**2))),
            ('MAE', np.mean(np.abs(error))),
            ('Mean Error', np.mean(error)),
            ('Std Error', np.std(error))
        ]
        
        return '\n'.join(f'{name}: {value:.3f}' for name, value in stats_items) 