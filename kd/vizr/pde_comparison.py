"""PDE solution visualization utilities.

This module provides functions for visualizing PDE solutions, including:
- Solution heatmaps
- Training data point overlays
- Time slice comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_pde_comparison(x, t, u_exact, u_pred, X_train, 
                       slice_times=[0.25, 0.5, 0.75],
                       save_path=None,
                       figsize=(12, 8)):
    """Plot PDE solution comparison with heatmap and time slices.
    
    Args:
        x: Spatial coordinates array
        t: Time coordinates array
        u_exact: Exact solution array (nt x nx)
        u_pred: Predicted solution array (nt x nx)
        X_train: Training data points array (n_points x 2)
        slice_times: List of time points for slice plots (default: [0.25, 0.5, 0.75])
        save_path: Path to save the figure (default: None)
        figsize: Figure size in inches (default: (12, 8))
        
    Returns:
        fig: The matplotlib figure object
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create GridSpec for top and bottom plots
    gs_top = gridspec.GridSpec(1, 1)
    gs_top.update(top=0.95, bottom=0.5, left=0.1, right=0.85)
    
    gs_bottom = gridspec.GridSpec(1, 3)
    gs_bottom.update(top=0.35, bottom=0.15, left=0.1, right=0.85, wspace=0.3)
    
    # Create axes
    ax_top = plt.subplot(gs_top[0])
    ax_slices = [plt.subplot(gs_bottom[0, i]) for i in range(3)]
    
    # Plot heatmap
    im = ax_top.imshow(u_pred, 
                      extent=[t.min(), t.max(), x.min(), x.max()],
                      origin='lower',
                      aspect='auto',
                      cmap='RdBu')
    
    # Add colorbar
    divider = make_axes_locatable(ax_top)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Plot training points
    ax_top.scatter(X_train[:,1], X_train[:,0], 
                  c='k', marker='x', s=20,
                  label=f'Data ({len(X_train)} points)')
    
    # Add time slice indicators
    for t_slice in slice_times:
        t_idx = int(t_slice * len(t))
        ax_top.axvline(x=t[t_idx], color='w', linestyle='-', linewidth=1)
    
    # Configure top plot
    ax_top.set_xlabel('t', fontsize=10)
    ax_top.set_ylabel('x', fontsize=10)
    ax_top.set_title('u(t,x)', fontsize=12, pad=10)
    ax_top.tick_params(labelsize=9)
    ax_top.legend(loc='lower left', fontsize=9, frameon=True,
                 fancybox=True, framealpha=0.8)
    
    # Plot time slices
    for i, t_slice in enumerate(slice_times):
        t_idx = int(t_slice * len(t))
        
        ax = ax_slices[i]
        
        # Plot exact and predicted solutions
        ax.plot(x, u_exact[t_idx], 'b-', label='Exact', linewidth=2)
        ax.plot(x, u_pred[t_idx], 'r--', label='Prediction', linewidth=2)
        
        # Configure slice plot
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('u(x,t)', fontsize=10)
        ax.set_title(f't = {t_slice:.2f}', fontsize=10, pad=5)
        ax.tick_params(labelsize=9)
        ax.grid(True, linestyle=':', alpha=0.3)
        
        # Only show legend in middle plot
        if i == 1:
            ax.legend(loc='upper center', 
                     bbox_to_anchor=(0.5, -0.35), 
                     ncol=2, 
                     frameon=False,
                     fontsize=10)
        
        # Set axis limits
        ax.set_xlim(x.min(), x.max())
        y_min = min(u_exact.min(), u_pred.min())
        y_max = max(u_exact.max(), u_pred.max())
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 