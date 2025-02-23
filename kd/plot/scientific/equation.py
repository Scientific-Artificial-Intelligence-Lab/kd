"""Equation terms plotting functionality.

This module provides classes for visualizing equation terms and their relationships,
including heatmaps of individual terms and comprehensive term analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional
import seaborn as sns
from scipy import stats

from ..core.base import BasePlot, CompositePlot


class TermsHeatmap(BasePlot):
    """Heatmap visualization of equation terms."""
    
    def _validate_data(self, terms: Dict[str, np.ndarray]) -> bool:
        """Validate input data shapes match."""
        shapes = [term.shape for term in terms.values()]
        if len(set(shapes)) > 1:
            raise ValueError(f"Terms have inconsistent shapes: {shapes}")
        return True
    
    def plot(self, terms: Dict[str, np.ndarray],
            x: np.ndarray, t: np.ndarray,
            show_colorbar: bool = True,
            normalize: bool = True):
        """Create heatmap of equation terms.
        
        Args:
            terms: Dictionary of term names and their values
            x: Spatial coordinates
            t: Time coordinates
            show_colorbar: Whether to show colorbar
            normalize: Whether to normalize term values
        """
        self._validate_data(terms)
        n_terms = len(terms)
        
        # Create figure with subplots
        n_cols = min(3, n_terms)  # Max 3 columns
        n_rows = (n_terms + n_cols - 1) // n_cols
        self.fig, axes = plt.subplots(n_rows, n_cols,
                                    figsize=(5*n_cols, 4*n_rows))
        if n_terms == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each term
        for i, (name, values) in enumerate(terms.items()):
            if normalize:
                values = (values - np.mean(values)) / np.std(values)
            
            im = axes[i].imshow(values,
                              extent=[x.min(), x.max(),
                                    t.min(), t.max()],
                              aspect='auto',
                              origin='lower',
                              cmap='RdBu')
            
            if show_colorbar:
                plt.colorbar(im, ax=axes[i])
            
            axes[i].set_title(name)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('t')
        
        # Hide unused subplots
        for i in range(n_terms, len(axes)):
            axes[i].axis('off')
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Save if path is provided
        if self.save_path:
            self.save()


class TermsAnalysis(CompositePlot):
    """Comprehensive analysis of equation terms."""
    
    def plot(self, terms: Dict[str, np.ndarray],
            x: np.ndarray, t: np.ndarray,
            show_correlations: bool = True,
            show_statistics: bool = True):
        """Create comprehensive term analysis plots.
        
        Args:
            terms: Dictionary of term names and their values
            x: Spatial coordinates
            t: Time coordinates
            show_correlations: Whether to show correlation matrix
            show_statistics: Whether to show term statistics
        """
        self._validate_data(terms)
        
        # Create figure with GridSpec
        self.fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # Terms heatmap
        ax1 = self.fig.add_subplot(gs[0, 0])
        self._plot_combined_heatmap(terms, x, t, ax1)
        
        # Correlation matrix
        if show_correlations:
            ax2 = self.fig.add_subplot(gs[0, 1])
            self._plot_correlation_matrix(terms, ax2)
        
        # Term magnitudes
        ax3 = self.fig.add_subplot(gs[1, 0])
        self._plot_term_magnitudes(terms, ax3)
        
        # Statistics
        if show_statistics:
            ax4 = self.fig.add_subplot(gs[1, 1])
            self._plot_statistics(terms, ax4)
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Save if path is provided
        if self.save_path:
            self.save()
    
    def _plot_combined_heatmap(self, terms: Dict[str, np.ndarray],
                             x: np.ndarray, t: np.ndarray,
                             ax: plt.Axes):
        """Plot combined heatmap of all terms."""
        # Combine terms into single array
        combined = np.zeros_like(list(terms.values())[0])
        for values in terms.values():
            combined += (values - np.mean(values)) / np.std(values)
        
        im = ax.imshow(combined,
                      extent=[x.min(), x.max(),
                             t.min(), t.max()],
                      aspect='auto',
                      origin='lower',
                      cmap='RdBu')
        plt.colorbar(im, ax=ax)
        ax.set_title('Combined Terms')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
    
    def _plot_correlation_matrix(self, terms: Dict[str, np.ndarray],
                               ax: plt.Axes):
        """Plot correlation matrix between terms."""
        # Calculate correlations
        term_arrays = [values.flatten() for values in terms.values()]
        corr_matrix = np.corrcoef(term_arrays)
        
        # Plot correlation matrix
        im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        
        # Add labels
        ax.set_xticks(range(len(terms)))
        ax.set_yticks(range(len(terms)))
        ax.set_xticklabels(terms.keys(), rotation=45)
        ax.set_yticklabels(terms.keys())
        ax.set_title('Term Correlations')
    
    def _plot_term_magnitudes(self, terms: Dict[str, np.ndarray],
                            ax: plt.Axes):
        """Plot magnitude distribution of terms."""
        positions = range(len(terms))
        ax.boxplot([values.flatten() for values in terms.values()],
                  tick_labels=terms.keys())
        ax.set_title('Term Magnitudes')
        ax.set_ylabel('Value')
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def _plot_statistics(self, terms: Dict[str, np.ndarray],
                        ax: plt.Axes):
        """Plot statistical summary of terms."""
        stats_text = []
        for name, values in terms.items():
            flat_values = values.flatten()
            stats_items = [
                ('Mean', np.mean(flat_values)),
                ('Std', np.std(flat_values)),
                ('Max', np.max(flat_values)),
                ('Min', np.min(flat_values))
            ]
            # Format each term's statistics in a more compact way
            term_stats = [f"{name:<10}  " + 
                        " | ".join(f"{stat}: {value:6.3f}" 
                                 for stat, value in stats_items)]
            stats_text.append(term_stats[0])
        
        # Add some padding between terms
        final_text = "\n\n".join(stats_text)
        
        # Create text box with adjusted position and padding
        ax.text(0.5, 0.5, final_text,
               horizontalalignment='center',
               verticalalignment='center',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round',
                        facecolor='white',
                        alpha=0.9,
                        pad=0.5),  # Add padding
               fontfamily='monospace',  # Use monospace font for alignment
               linespacing=2)  # Increase line spacing
        
        ax.axis('off')
        ax.set_title('Term Statistics', pad=20)  # Add padding to title 