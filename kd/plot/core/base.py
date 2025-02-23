"""Base classes for plotting functionality.

This module provides the foundational classes for the plotting system,
including base plot class, renderer system, and style management.
"""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, Union, Dict, Any

class BasePlot(ABC):
    """Base class for all plotting classes.
    
    This class provides the basic infrastructure for creating plots,
    including style management, rendering control, and save functionality.
    
    Attributes:
        save_path (str): Path to save the plot.
        save_dir (str): Directory to save plots in.
        show_realtime (bool): Whether to show plot in real-time.
        style (str): Plot style to use.
        fig: Current figure object.
        axes: Current axes object(s).
    """
    
    # Default save directory relative to current working directory
    default_save_dir = os.path.join(os.getcwd(), ".plot_output")
    
    def __init__(self, 
                 save_path: Optional[str] = None,
                 save_dir: Optional[str] = None,
                 show_realtime: bool = False,
                 style: str = 'default',
                 figsize: tuple = (8, 6)):
        """Initialize plot.
        
        Args:
            save_path: Path to save the plot. If None, plot is not saved.
            save_dir: Directory to save plots in. If None, uses default directory.
            show_realtime: Whether to show plot updates in real-time.
            style: Style to apply to plot.
            figsize: Figure size in inches.
        """
        # Set up save directory
        self.save_dir = save_dir if save_dir else self.default_save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Set up save path
        self.save_path = save_path
        if save_path and not os.path.isabs(save_path):
            self.save_path = os.path.join(self.save_dir, save_path)
            
        self.show_realtime = show_realtime
        self.style = style
        self.figsize = figsize
        
        # Initialize plot objects
        self.fig = None
        self.axes = None
        
        # Apply style
        self._set_style()
        
    def _set_style(self):
        """Apply the selected style settings."""
        # Basic style settings
        plt.rcParams.update({
            "ytick.minor.visible": False,
            "xtick.minor.visible": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": ['Arial'],
            "mathtext.fontset": 'stix'
        })
        
    def _validate_data(self, *args, **kwargs) -> bool:
        """Validate input data.
        
        Returns:
            bool: True if data is valid.
            
        Raises:
            ValueError: If data validation fails.
        """
        return True
    
    def _preprocess_data(self, *args, **kwargs) -> Dict[str, Any]:
        """Preprocess input data.
        
        Returns:
            dict: Preprocessed data.
        """
        return {}
    
    @abstractmethod
    def plot(self, *args, **kwargs):
        """Create the plot.
        
        This method must be implemented by subclasses.
        """
        pass
    
    def save(self, dpi: int = 300):
        """Save the current plot.
        
        Args:
            dpi: Resolution for saved image.
        """
        if self.save_path and self.fig:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(self.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            # Save figure
            self.fig.savefig(self.save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved plot to: {self.save_path}")
            
    def show(self):
        """Display the plot."""
        if self.show_realtime:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.show()
            
    def close(self):
        """Close the current plot."""
        if self.fig:
            plt.close(self.fig)
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class CompositePlot(BasePlot):
    """Base class for composite plots.
    
    This class provides functionality for creating plots with multiple
    subplots or components.
    
    Attributes:
        subplots (dict): Dictionary of subplot objects.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subplots = {}
        
    def add_subplot(self, name: str, plot_type: type, *args, **kwargs):
        """Add a subplot.
        
        Args:
            name: Identifier for the subplot.
            plot_type: Type of plot to create.
            *args: Arguments for the subplot.
            **kwargs: Keyword arguments for the subplot.
        """
        self.subplots[name] = plot_type(*args, **kwargs)
        
    def plot(self, *args, **kwargs):
        """Create all subplots."""
        for name, subplot in self.subplots.items():
            subplot.plot(*args, **kwargs) 