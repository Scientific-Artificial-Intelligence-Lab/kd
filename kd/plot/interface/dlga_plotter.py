"""DLGA plotting interface.

This module provides a high-level interface for DLGA model visualization,
including residual analysis, comparison plots, and evolution visualization.
"""

from typing import Optional, Union, Dict, List
import numpy as np
import os

from ..scientific.evolution import EvolutionAnimation, EvolutionSnapshot
from ..scientific.residual import (
    ResidualHeatmap,
    ResidualTimeslice,
    ResidualHistogram,
    ResidualAnalysis
)

class DLGAPlotter:
    """High-level interface for DLGA visualization.
    
    This class provides a unified interface for creating various types of
    visualizations for DLGA model analysis and results.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """Initialize plotter.
        
        Args:
            save_dir: Directory to save plots in. If None, uses default.
        """
        self.save_dir = save_dir
        
    def plot_evolution(self, evolution_data, mode="animation", **kwargs):
        """Plot evolution process
        
        Args:
            evolution_data: List/Iterator of generation data
            mode: Visualization mode, one of ['animation', 'snapshot']
            **kwargs: Additional arguments passed to specific plotter
                For animation mode:
                    - interval: Animation interval in ms (default: 100)
                    - figsize: Figure size (default: (10,8))
                    - dpi: DPI for display (default: 100)
                
                For snapshot mode:
                    - output_format: Output format ['mp4','gif','frames'] (default: 'mp4')
                    - output_path: Output file/directory path (default: None)
                    - fps: Frames per second for video (default: 30)
                    - cleanup_temp: Whether to cleanup temp files (default: True)
                    - figsize: Figure size (default: (10,8))
                    - dpi: DPI for saved images (default: 300)
                    - desired_duration: Desired duration in seconds (default: 15)
        
        Returns:
            Animation object if mode is 'animation'
            Output path if mode is 'snapshot'
            
        Raises:
            ValueError: If mode is invalid or required parameters are missing
            TypeError: If evolution_data is None or not iterable
        """
        # Validate input data
        if evolution_data is None:
            raise ValueError("evolution_data cannot be None")
            
        try:
            if len(list(evolution_data)) == 0:
                raise ValueError("evolution_data cannot be empty")
        except TypeError:
            raise TypeError("evolution_data must be iterable")
            
        # Validate mode
        if mode not in ["animation", "snapshot"]:
            raise ValueError(f"Unknown mode: {mode}")
            
        # Get common parameters
        figsize = kwargs.get("figsize", (10,8))
        
        if mode == "animation":
            dpi = kwargs.get("dpi", 100)
            interval = kwargs.get("interval", 100)
            
            plotter = EvolutionAnimation(
                figsize=figsize,
                dpi=dpi,
                interval=interval
            )
            plotter.animate(evolution_data)
            return plotter.anim
            
        else:  # mode == "snapshot"
            # Validate output format
            output_format = kwargs.get("output_format", "mp4")
            if output_format not in ["mp4", "gif", "frames"]:
                raise ValueError(f"Invalid output format: {output_format}")
                
            # Validate output path
            output_path = kwargs.get("output_path", None)
            if output_path is None:
                raise ValueError("output_path is required for snapshot mode")
                
            # Get other parameters with validation
            fps = kwargs.get("fps", 30)
            cleanup_temp = kwargs.get("cleanup_temp", True)
            dpi = kwargs.get("dpi", 300)  # Higher DPI for saved images
            desired_duration = kwargs.get("desired_duration", 15)
            
            # Validate numeric parameters
            if fps <= 0:
                raise ValueError("fps must be positive")
            if dpi <= 0:
                raise ValueError("dpi must be positive")
            if desired_duration <= 0:
                raise ValueError("desired_duration must be positive")
            
            plotter = EvolutionSnapshot(
                figsize=figsize,
                dpi=dpi,
                output_format=output_format,
                fps=fps,
                cleanup_temp=cleanup_temp,
                desired_duration=desired_duration
            )
            plotter.save_evolution(evolution_data, output_path)
            return output_path 

    def plot_residual(self,
                     u_pred: np.ndarray,
                     u_exact: np.ndarray,
                     x: np.ndarray,
                     t: np.ndarray,
                     plot_type: str = 'all',
                     save_path: Optional[str] = None,
                     **kwargs) -> None:
        """Create residual analysis plots.
        
        Args:
            u_pred: Predicted solution array (nt x nx)
            u_exact: Exact solution array (nt x nx)
            x: Spatial coordinates
            t: Time coordinates
            plot_type: Type of plot to create ('heatmap', 'timeslice', 
                      'histogram', or 'all')
            save_path: Path to save the plot
            **kwargs: Additional arguments passed to specific plot types
            
        Raises:
            ValueError: If plot_type is invalid or data shapes don't match
        """
        # Validate plot type
        valid_types = ['heatmap', 'timeslice', 'histogram', 'all']
        if plot_type not in valid_types:
            raise ValueError(f"Invalid plot_type. Must be one of {valid_types}")
            
        # Validate data shapes
        if u_pred.shape != u_exact.shape:
            raise ValueError(f"Shape mismatch: u_pred {u_pred.shape} != u_exact {u_exact.shape}")
            
        if len(u_pred.shape) != 2:
            raise ValueError(f"Expected 2D arrays, got shape {u_pred.shape}")
            
        # Create appropriate plot based on type
        if plot_type == 'heatmap':
            plotter = ResidualHeatmap(save_path=save_path)
            plotter.plot(u_pred, u_exact, x, t, **kwargs)
            
        elif plot_type == 'timeslice':
            plotter = ResidualTimeslice(save_path=save_path)
            plotter.plot(u_pred, u_exact, x, t, **kwargs)
            
        elif plot_type == 'histogram':
            plotter = ResidualHistogram(save_path=save_path)
            plotter.plot(u_pred, u_exact, **kwargs)
            
        else:  # 'all'
            plotter = ResidualAnalysis(save_path=save_path)
            plotter.plot(u_pred, u_exact, x, t, **kwargs)
            
        # Save if path is provided
        if save_path:
            plotter.save() 