"""Direct plot interface for scientific visualization.

提供对 scientific visualization components 的直接访问。
这是一个轻量级的接口，用户可以直接使用各个可视化组件。
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np

from ..scientific.residual import (
    ResidualHeatmap,
    ResidualTimeslice,
    ResidualHistogram,
    ResidualAnalysis
)
from ..scientific.comparison import (
    ComparisonScatter,
    ComparisonLine,
    ComparisonAnalysis
)
from ..scientific.equation import (
    TermsHeatmap,
    TermsAnalysis
)
from ..scientific.optimization import (
    LossPlot,
    WeightsPlot,
    DiversityPlot,
    OptimizationAnalysis
)
from ..scientific.evolution import (
    EvolutionAnimation,
    EvolutionSnapshot
)

def plot_residual(u_pred: np.ndarray,
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
        plot_type: Type of plot ('heatmap', 'timeslice', 'histogram', 'all')
        save_path: Path to save the plot
        **kwargs: Additional arguments for specific plot types
    """
    plot_types = {
        'heatmap': ResidualHeatmap,
        'timeslice': ResidualTimeslice,
        'histogram': ResidualHistogram,
        'all': ResidualAnalysis
    }
    
    if plot_type not in plot_types:
        raise ValueError(f"Invalid plot_type. Must be one of {list(plot_types.keys())}")
        
    plotter = plot_types[plot_type](save_path=save_path)
    if plot_type == 'histogram':
        plotter.plot(u_pred, u_exact, **kwargs)
    else:
        plotter.plot(u_pred, u_exact, x, t, **kwargs)
        
def plot_comparison(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   plot_type: str = 'all',
                   save_path: Optional[str] = None,
                   **kwargs) -> None:
    """Create comparison plots.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        plot_type: Type of plot ('scatter', 'line', 'all')
        save_path: Path to save the plot
        **kwargs: Additional arguments for specific plot types
    """
    plot_types = {
        'scatter': ComparisonScatter,
        'line': ComparisonLine,
        'all': ComparisonAnalysis
    }
    
    if plot_type not in plot_types:
        raise ValueError(f"Invalid plot_type. Must be one of {list(plot_types.keys())}")
        
    plotter = plot_types[plot_type](save_path=save_path)
    plotter.plot(y_true, y_pred, **kwargs)
    
def plot_equation_terms(terms: Dict[str, np.ndarray],
                       x: np.ndarray,
                       t: np.ndarray,
                       plot_type: str = 'all',
                       save_path: Optional[str] = None,
                       **kwargs) -> None:
    """Create equation terms visualization.
    
    Args:
        terms: Dictionary of term names and their values
        x: Spatial coordinates
        t: Time coordinates
        plot_type: Type of plot ('heatmap', 'all')
        save_path: Path to save the plot
        **kwargs: Additional arguments for specific plot types
    """
    plot_types = {
        'heatmap': TermsHeatmap,
        'all': TermsAnalysis
    }
    
    if plot_type not in plot_types:
        raise ValueError(f"Invalid plot_type. Must be one of {list(plot_types.keys())}")
        
    plotter = plot_types[plot_type](save_path=save_path)
    plotter.plot(terms, x, t, **kwargs)
    
def plot_optimization(epochs: np.ndarray,
                     loss_history: Optional[np.ndarray] = None,
                     val_loss_history: Optional[np.ndarray] = None,
                     weights_history: Optional[np.ndarray] = None,
                     diversity_history: Optional[np.ndarray] = None,
                     plot_type: str = 'all',
                     save_path: Optional[str] = None,
                     **kwargs) -> None:
    """Create optimization process visualization.
    
    Args:
        epochs: Array of epoch numbers
        loss_history: Array of training loss values
        val_loss_history: Array of validation loss values
        weights_history: Array of weights
        diversity_history: Array of diversity values
        plot_type: Type of plot ('loss', 'weights', 'diversity', 'all')
        save_path: Path to save the plot
        **kwargs: Additional arguments for specific plot types
    """
    plot_types = {
        'loss': LossPlot,
        'weights': WeightsPlot,
        'diversity': DiversityPlot,
        'all': OptimizationAnalysis
    }
    
    if plot_type not in plot_types:
        raise ValueError(f"Invalid plot_type. Must be one of {list(plot_types.keys())}")
        
    plotter = plot_types[plot_type](save_path=save_path)
    
    if plot_type == 'loss':
        plotter.plot(epochs, loss_history, val_loss_history=val_loss_history, **kwargs)
    elif plot_type == 'weights':
        plotter.plot(epochs, weights_history, **kwargs)
    elif plot_type == 'diversity':
        plotter.plot(epochs, diversity_history, **kwargs)
    else:
        plotter.plot(
            epochs=epochs,
            loss_history=loss_history,
            val_loss_history=val_loss_history,
            weights_history=weights_history,
            diversity_history=diversity_history,
            **kwargs
        )
        
def plot_evolution(evolution_data: List[Any],
                  mode: str = 'animation',
                  save_path: Optional[str] = None,
                  **kwargs) -> Any:
    """Create evolution visualization.
    
    Args:
        evolution_data: List of generation data
        mode: Visualization mode ('animation', 'snapshot')
        save_path: Path to save the output
        **kwargs: Additional arguments for specific modes
            For animation:
                - interval: Animation interval in ms
                - figsize: Figure size
                - dpi: DPI for display
            For snapshot:
                - output_format: Output format ('mp4','gif','frames')
                - fps: Frames per second
                - cleanup_temp: Whether to cleanup temp files
                - figsize: Figure size
                - dpi: DPI for saved images
                - desired_duration: Desired duration in seconds
                
    Returns:
        Animation object if mode is 'animation'
        Output path if mode is 'snapshot'
    """
    if mode == 'animation':
        plotter = EvolutionAnimation(**kwargs)
        plotter.animate(evolution_data)
        return plotter.anim
    elif mode == 'snapshot':
        if save_path is None:
            raise ValueError("save_path is required for snapshot mode")
            
        plotter = EvolutionSnapshot(**kwargs)
        return plotter.save_evolution(evolution_data, save_path)
    else:
        raise ValueError("mode must be 'animation' or 'snapshot'") 