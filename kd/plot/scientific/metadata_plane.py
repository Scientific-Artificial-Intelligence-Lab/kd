"""
Metadata Value Plane Visualization Module

This module provides functionality for visualizing metadata distributions in 2D parameter spaces.
Typical use cases include:
1. Visualizing model performance across different hyperparameter combinations
2. Displaying parameter search trajectories during optimization
3. Analyzing parameter sensitivity and interactions

Example:
```python
# Create a grid search visualization
x = np.linspace(0.001, 0.1, 20)  # learning rate
y = np.linspace(32, 256, 20)     # batch size
performance = np.array([...])     # corresponding model performance

plotter = MetadataValuePlane()
fig = plotter.plot(
    x_values=x,
    y_values=y, 
    z_values=performance,
    x_label='Learning Rate',
    y_label='Batch Size',
    z_label='Validation Accuracy',
    plot_type='contour'
)

# Add optimization trajectory
trajectory = np.array([...])  # parameter changes during optimization
plotter.add_trajectory(trajectory, label='Optimization Path')
```
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Union, Optional
import warnings

class MetadataValuePlane:
    """Visualize model metadata distributions in 2D parameter spaces.
    
    This class provides multiple ways to visualize metadata distributions in 2D parameter spaces,
    including contour plots, heatmaps, and 3D surface plots. It also supports adding optimization
    trajectories and special points.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """Initialize the plotter.
        
        Args:
            figsize: Figure size
        """
        self.fig = None
        self.ax = None
        self.figsize = figsize
        
    def _validate_inputs(self, x_values: np.ndarray, y_values: np.ndarray, 
                        z_values: np.ndarray) -> None:
        """Validate input data.
        
        Args:
            x_values: x-axis data
            y_values: y-axis data
            z_values: metadata values
            
        Raises:
            TypeError: If inputs are not numpy arrays
        """
        if not isinstance(x_values, np.ndarray) or not isinstance(y_values, np.ndarray):
            raise TypeError("x_values and y_values must be numpy arrays")
            
        if np.any(np.isnan(z_values)):
            warnings.warn("z_values contains NaN values, which may affect visualization")
    
    def plot(self, 
             x_values: np.ndarray,
             y_values: np.ndarray,
             z_values: np.ndarray,
             x_label: str = 'Parameter 1',
             y_label: str = 'Parameter 2',
             z_label: str = 'Metadata Value',
             plot_type: str = 'contour',
             colormap: str = 'viridis',
             show_colorbar: bool = True,
             title: Optional[str] = None,
             levels: int = 20,
             alpha: float = 1.0) -> plt.Figure:
        """Plot metadata value plane.
        
        Args:
            x_values: x-axis data
            y_values: y-axis data
            z_values: metadata values
            x_label: x-axis label
            y_label: y-axis label
            z_label: z-axis label
            plot_type: Plot type ('contour', 'surface', 'heatmap')
            colormap: Color map
            show_colorbar: Whether to show colorbar
            title: Plot title
            levels: Number of contour levels
            alpha: Transparency
            
        Returns:
            matplotlib Figure object
        """
        self._validate_inputs(x_values, y_values, z_values)
        self.fig = plt.figure(figsize=self.figsize)
        
        # Handle data shapes
        if x_values.ndim == 1 and y_values.ndim == 1:
            X, Y = np.meshgrid(x_values, y_values)
            if z_values.shape != X.shape:
                Z = z_values.reshape(len(y_values), len(x_values))
            else:
                Z = z_values
        else:
            X, Y = x_values, y_values
            Z = z_values
            
        # Choose plot type
        if plot_type == 'surface':
            self.ax = self.fig.add_subplot(111, projection='3d')
            surf = self.ax.plot_surface(X, Y, Z, cmap=colormap,
                                      linewidth=0, antialiased=True,
                                      alpha=alpha)
            self.ax.set_zlabel(z_label)
            
        elif plot_type == 'contour':
            self.ax = self.fig.add_subplot(111)
            contour = self.ax.contourf(X, Y, Z, cmap=colormap,
                                     levels=levels, alpha=alpha)
            if show_colorbar:
                plt.colorbar(contour, label=z_label)
                
        elif plot_type == 'heatmap':
            self.ax = self.fig.add_subplot(111)
            heatmap = self.ax.pcolormesh(X, Y, Z, cmap=colormap,
                                       shading='auto', alpha=alpha)
            if show_colorbar:
                plt.colorbar(heatmap, label=z_label)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        # Set labels and title
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        if title:
            self.ax.set_title(title)
            
        plt.tight_layout()
        return self.fig
        
    def add_trajectory(self, 
                      trajectory: np.ndarray, 
                      color: str = 'red',
                      marker: str = 'o-',
                      label: Optional[str] = None,
                      linewidth: float = 2.0,
                      markersize: float = 6.0) -> plt.Figure:
        """Add trajectory line to existing plot.
        
        Args:
            trajectory: Trajectory points array (n_points, 2) or (n_points, 3)
            color: Line color
            marker: Marker style
            label: Line label
            linewidth: Line width
            markersize: Marker size
            
        Returns:
            matplotlib Figure object
            
        Raises:
            ValueError: If plot not created or trajectory dimensions mismatch
        """
        if self.ax is None:
            raise ValueError("Please call plot method first to create base plot")
            
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)
            
        if trajectory.ndim != 2:
            raise ValueError("Trajectory data must be a 2D array")
            
        if trajectory.shape[1] == 2 and hasattr(self.ax, 'plot'):
            self.ax.plot(trajectory[:, 0], trajectory[:, 1],
                        marker, color=color, label=label,
                        linewidth=linewidth, markersize=markersize)
        elif trajectory.shape[1] == 3 and hasattr(self.ax, 'plot3D'):
            self.ax.plot3D(trajectory[:, 0], trajectory[:, 1],
                          trajectory[:, 2], marker, color=color,
                          label=label, linewidth=linewidth,
                          markersize=markersize)
        else:
            raise ValueError("Trajectory data dimensions do not match plot type")
        
        if label:
            self.ax.legend()
            
        return self.fig 