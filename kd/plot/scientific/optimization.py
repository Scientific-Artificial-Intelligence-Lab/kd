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
            val_loss_history: Optional[np.ndarray] = None,
            show_trend: bool = True, show_best: bool = True):
        """Create loss history plot.
        
        Args:
            epochs: Array of epoch numbers
            loss_history: Array of loss values
            val_loss_history: Optional array of validation loss values
            show_trend: Whether to show trend line
            show_best: Whether to show best value markers
        """
        self.fig, self.axes = plt.subplots(figsize=self.figsize)
        
        # Plot training loss
        self.axes.plot(epochs, loss_history, 'b-', alpha=0.6, label='Training Loss')
        
        # Plot validation loss if provided
        if val_loss_history is not None:
            self.axes.plot(epochs, val_loss_history, 'r-', alpha=0.6, label='Validation Loss')
        
        if show_trend and len(epochs) > 3:
            try:
                # Add trend line using polynomial fit for training loss
                degree = min(3, len(epochs) - 1)
                z = np.polyfit(epochs, loss_history, degree)
                p = np.poly1d(z)
                self.axes.plot(epochs, p(epochs), 'b--', alpha=0.4, label='Training Trend')
                
                # Add trend line for validation loss if provided
                if val_loss_history is not None:
                    z_val = np.polyfit(epochs, val_loss_history, degree)
                    p_val = np.poly1d(z_val)
                    self.axes.plot(epochs, p_val(epochs), 'r--', alpha=0.4, label='Validation Trend')
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Warning: Could not compute trend line: {e}")
        
        if show_best:
            # Mark best values for training loss
            best_idx = np.argmin(loss_history)
            self.axes.plot(epochs[best_idx], loss_history[best_idx], 'b*',
                         markersize=10, label=f'Best Training: {loss_history[best_idx]:.3f}')
            
            # Mark best values for validation loss if provided
            if val_loss_history is not None:
                best_val_idx = np.argmin(val_loss_history)
                self.axes.plot(epochs[best_val_idx], val_loss_history[best_val_idx], 'r*',
                             markersize=10, label=f'Best Validation: {val_loss_history[best_val_idx]:.3f}')
        
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
            weights_history: Array of weights (n_epochs x n_terms)
            show_confidence: Whether to show confidence intervals
        """
        self.fig, self.axes = plt.subplots(figsize=self.figsize)
        
        n_terms = weights_history.shape[1]
        colors = plt.cm.tab10(np.linspace(0, 1, n_terms))
        
        for i in range(n_terms):
            weights = weights_history[:, i]
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
            val_loss_history: Optional[np.ndarray] = None,
            weights_history: Optional[np.ndarray] = None,
            diversity_history: Optional[np.ndarray] = None,
            best_individual_history: Optional[Dict] = None):
        """Create comprehensive optimization analysis plots.
        
        Args:
            epochs: Array of epoch numbers
            loss_history: Array of training loss values
            val_loss_history: Array of validation loss values
            weights_history: Array of weights (n_terms x n_epochs)
            diversity_history: Array of diversity values
            best_individual_history: Dictionary containing best individual info
        """
        # Create figure with GridSpec
        self.fig = plt.figure(figsize=(15, 15))  # Increased height for extra subplot
        gs = gridspec.GridSpec(3, 2)  # Changed to 3x2 grid
        
        # Training loss history
        if loss_history is not None:
            ax1 = self.fig.add_subplot(gs[0, 0])
            self._plot_training_loss(epochs, loss_history, ax1)
        
        # Validation loss history
        if val_loss_history is not None:
            ax2 = self.fig.add_subplot(gs[0, 1])
            self._plot_validation_loss(epochs, val_loss_history, ax2)
        
        # Weights evolution
        if weights_history is not None:
            ax3 = self.fig.add_subplot(gs[1, :])  # Span both columns
            self._plot_weights(epochs, weights_history, ax3)
        
        # Diversity history
        if diversity_history is not None:
            ax4 = self.fig.add_subplot(gs[2, 0])
            self._plot_diversity(epochs, diversity_history, ax4)
        
        # Best individual evolution
        if best_individual_history is not None:
            ax5 = self.fig.add_subplot(gs[2, 1])
            self._plot_best_individual(epochs, best_individual_history, ax5)
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Save if path is provided
        if self.save_path:
            self.save()
    
    def _plot_training_loss(self, epochs: np.ndarray, loss_history: np.ndarray,
                         ax: plt.Axes):
        """Plot training loss history subplot."""
        # Plot training loss
        ax.plot(epochs, loss_history, 'b-', alpha=0.6, label='Training Loss')
        
        if len(epochs) > 3:
            try:
                # Add trend line
                degree = min(3, len(epochs) - 1)
                z = np.polyfit(epochs, loss_history, degree)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), 'b--', alpha=0.4, label='Trend')
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Warning: Could not compute trend line: {e}")
        
        # Mark best value
        best_idx = np.argmin(loss_history)
        ax.plot(epochs[best_idx], loss_history[best_idx], 'b*',
                markersize=10, label=f'Best: {loss_history[best_idx]:.3e}')
            
        ax.set_title('Training Loss History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.grid(True, linestyle=':')
        ax.legend()
        
        # Use scientific notation for y-axis if values are very small
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    def _plot_validation_loss(self, epochs: np.ndarray, val_loss_history: np.ndarray,
                          ax: plt.Axes):
        """Plot validation loss history subplot."""
        # Plot validation loss
        ax.plot(epochs, val_loss_history, 'r-', alpha=0.6, label='Validation Loss')
        
        if len(epochs) > 3:
            try:
                # Add trend line
                degree = min(3, len(epochs) - 1)
                z = np.polyfit(epochs, val_loss_history, degree)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), 'r--', alpha=0.4, label='Trend')
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Warning: Could not compute trend line: {e}")
        
        # Mark best value
        best_idx = np.argmin(val_loss_history)
        ax.plot(epochs[best_idx], val_loss_history[best_idx], 'r*',
                markersize=10, label=f'Best: {val_loss_history[best_idx]:.3f}')
            
        ax.set_title('Validation Loss History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.grid(True, linestyle=':')
        ax.legend()
    
    def _plot_weights(self, epochs: np.ndarray, weights_history: np.ndarray,
                     ax: plt.Axes):
        """Plot weights evolution.
        
        Args:
            epochs: Array of epoch numbers
            weights_history: Array of weights (n_epochs x n_terms)
            ax: Matplotlib axes object
        """
        n_terms = weights_history.shape[1]  # Changed from shape[0]
        colors = plt.cm.tab10(np.linspace(0, 1, n_terms))
        
        for i in range(n_terms):
            ax.plot(epochs, weights_history[:, i], color=colors[i],  # Changed from [i]
                   label=f'Term {i+1}', alpha=0.7)
            
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight Value')
        ax.set_title('Weights Evolution')
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