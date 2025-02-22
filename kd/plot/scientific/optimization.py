"""Optimization process visualization functionality.

This module provides classes for visualizing various aspects of the optimization process,
including loss history, weights evolution, population diversity, and comprehensive analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from typing import Dict, List, Optional, Union
from ..core.base import BasePlot, CompositePlot


class LossPlot(BasePlot):
    """Loss history visualization."""
    
    def plot(self, epochs: np.ndarray, loss_history: np.ndarray,
            show_trend: bool = True, show_best: bool = True):
        """Create loss history plot.
        
        Args:
            epochs: Array of epoch numbers
            loss_history: Array of loss values
            show_trend: Whether to show trend line
            show_best: Whether to show best value markers
        """
        self.fig, self.axes = plt.subplots(figsize=self.figsize)
        
        # Plot loss history
        self.axes.plot(epochs, loss_history, 'b-', alpha=0.6, label='Loss')
        
        if show_trend:
            # Add trend line using polynomial fit
            z = np.polyfit(epochs, loss_history, 3)
            p = np.poly1d(z)
            self.axes.plot(epochs, p(epochs), 'r--', alpha=0.8, label='Trend')
        
        if show_best:
            # Mark best values
            best_idx = np.argmin(loss_history)
            self.axes.plot(epochs[best_idx], loss_history[best_idx], 'r*',
                         markersize=10, label=f'Best: {loss_history[best_idx]:.3f}')
        
        # Configure axes
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Loss')
        self.axes.set_title('Loss History')
        self.axes.grid(True, linestyle=':')
        self.axes.legend()
        
        # Save if path is provided
        if self.save_path:
            self.save()


class WeightsPlot(BasePlot):
    """Weights evolution visualization."""
    
    def plot(self, epochs: np.ndarray, weights_history: np.ndarray,
            show_confidence: bool = True):
        """Create weights evolution plot.
        
        Args:
            epochs: Array of epoch numbers
            weights_history: Array of weights (n_terms x n_epochs)
            show_confidence: Whether to show confidence intervals
        """
        self.fig, self.axes = plt.subplots(figsize=self.figsize)
        
        n_terms = weights_history.shape[0]
        colors = plt.cm.tab10(np.linspace(0, 1, n_terms))
        
        for i in range(n_terms):
            weights = weights_history[i]
            self.axes.plot(epochs, weights, color=colors[i],
                         label=f'Term {i+1}', alpha=0.7)
            
            if show_confidence:
                # Add confidence intervals
                window = min(20, len(epochs)//5)  # Rolling window size
                std = np.array([np.std(weights[max(0, j-window):j+1])
                              for j in range(len(epochs))])
                self.axes.fill_between(epochs,
                                     weights - 2*std,
                                     weights + 2*std,
                                     color=colors[i], alpha=0.1)
        
        # Configure axes
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Weight Value')
        self.axes.set_title('Weights Evolution')
        self.axes.grid(True, linestyle=':')
        self.axes.legend()
        
        # Save if path is provided
        if self.save_path:
            self.save()


class DiversityPlot(BasePlot):
    """Population diversity visualization."""
    
    def plot(self, epochs: np.ndarray, diversity_history: np.ndarray,
            show_statistics: bool = True):
        """Create diversity history plot.
        
        Args:
            epochs: Array of epoch numbers
            diversity_history: Array of diversity values
            show_statistics: Whether to show diversity statistics
        """
        self.fig, self.axes = plt.subplots(figsize=self.figsize)
        
        # Plot diversity history
        self.axes.plot(epochs, diversity_history, 'g-', alpha=0.7,
                      label='Diversity')
        
        if show_statistics:
            # Calculate and display statistics
            mean_div = np.mean(diversity_history)
            std_div = np.std(diversity_history)
            
            # Add mean line
            self.axes.axhline(y=mean_div, color='r', linestyle='--',
                            alpha=0.5, label=f'Mean: {mean_div:.3f}')
            
            # Add standard deviation band
            self.axes.fill_between(epochs,
                                 mean_div - std_div,
                                 mean_div + std_div,
                                 color='r', alpha=0.1,
                                 label=f'Std: {std_div:.3f}')
        
        # Configure axes
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Diversity')
        self.axes.set_title('Population Diversity')
        self.axes.grid(True, linestyle=':')
        self.axes.legend()
        
        # Save if path is provided
        if self.save_path:
            self.save()


class OptimizationAnalysis(CompositePlot):
    """Comprehensive optimization process analysis."""
    
    def plot(self, epochs: np.ndarray,
            loss_history: Optional[np.ndarray] = None,
            weights_history: Optional[np.ndarray] = None,
            diversity_history: Optional[np.ndarray] = None,
            best_individual_history: Optional[Dict] = None):
        """Create comprehensive optimization analysis plots.
        
        Args:
            epochs: Array of epoch numbers
            loss_history: Array of loss values
            weights_history: Array of weights (n_terms x n_epochs)
            diversity_history: Array of diversity values
            best_individual_history: Dictionary containing best individual info
        """
        # Create figure with GridSpec
        self.fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # Loss history
        if loss_history is not None:
            ax1 = self.fig.add_subplot(gs[0, 0])
            self._plot_loss(epochs, loss_history, ax1)
        
        # Weights evolution
        if weights_history is not None:
            ax2 = self.fig.add_subplot(gs[0, 1])
            self._plot_weights(epochs, weights_history, ax2)
        
        # Diversity history
        if diversity_history is not None:
            ax3 = self.fig.add_subplot(gs[1, 0])
            self._plot_diversity(epochs, diversity_history, ax3)
        
        # Best individual evolution
        if best_individual_history is not None:
            ax4 = self.fig.add_subplot(gs[1, 1])
            self._plot_best_individual(epochs, best_individual_history, ax4)
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Save if path is provided
        if self.save_path:
            self.save()
    
    def _plot_loss(self, epochs: np.ndarray, loss_history: np.ndarray,
                  ax: plt.Axes):
        """Plot loss history subplot."""
        ax.plot(epochs, loss_history, 'b-', alpha=0.6)
        ax.set_title('Loss History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, linestyle=':')
    
    def _plot_weights(self, epochs: np.ndarray, weights_history: np.ndarray,
                     ax: plt.Axes):
        """Plot weights evolution subplot."""
        n_terms = weights_history.shape[0]
        colors = plt.cm.tab10(np.linspace(0, 1, n_terms))
        
        for i in range(n_terms):
            ax.plot(epochs, weights_history[i], color=colors[i],
                   label=f'Term {i+1}', alpha=0.7)
        
        ax.set_title('Weights Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight Value')
        ax.grid(True, linestyle=':')
        ax.legend()
    
    def _plot_diversity(self, epochs: np.ndarray, diversity_history: np.ndarray,
                       ax: plt.Axes):
        """Plot diversity history subplot."""
        ax.plot(epochs, diversity_history, 'g-', alpha=0.7)
        ax.set_title('Population Diversity')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Diversity')
        ax.grid(True, linestyle=':')
    
    def _plot_best_individual(self, epochs: np.ndarray,
                            best_history: Dict,
                            ax: plt.Axes):
        """Plot best individual evolution subplot."""
        # Plot fitness
        ax.plot(epochs, best_history['fitness'], 'b-',
                label='Fitness', alpha=0.7)
        
        # Plot complexity on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(epochs, best_history['complexity'], 'r--',
                label='Complexity', alpha=0.7)
        
        # Configure axes
        ax.set_title('Best Individual Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Fitness', color='b')
        ax2.set_ylabel('Complexity', color='r')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax.grid(True, linestyle=':') 