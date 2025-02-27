"""High-level interface for DeepRL visualization.

提供对 DeepRL 可视化的高级接口,支持:
1. Training Process Visualization
   - Loss curves
   - Reward curves
   - State-action visualization
2. Model Analysis
   - Tree structure
   - Value distribution
3. Performance Analysis
   - Residual plots
   - Comparison plots
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import matplotlib.pyplot as plt

from ..scientific.optimization import (
    LossPlot,
    OptimizationAnalysis
)
from ..scientific.comparison import (
    ComparisonScatter,
    ComparisonLine
)
from ..scientific.residual import (
    ResidualAnalysis,
    ResidualHeatmap
)

class DeepRLPlotter:
    """High-level interface for DeepRL visualization.
    
    提供简单易用的接口来可视化 DeepRL 模型的:
    - 训练过程
    - 模型结构
    - 性能分析
    """
    
    def __init__(self, monitor=None):
        """Initialize plotter.
        
        Args:
            monitor: Optional DeepRLMonitor instance
        """
        self.monitor = monitor
        
    def plot_training(self, plot_type: str = 'loss', save_path: Optional[str] = None, **kwargs):
        """Plot training process visualization.
        
        Args:
            plot_type: Type of plot ('loss', 'reward', 'all')
            save_path: Path to save the plot
            **kwargs: Additional arguments for specific plot types
                For loss plot:
                    - show_trend: Whether to show trend line
                    - show_best: Whether to show best value markers
                For reward plot:
                    - show_bounds: Whether to show min/max bounds
                    - window_size: Window size for moving average
                    
        Returns:
            matplotlib Figure object
        """
        if self.monitor is None:
            raise ValueError("Monitor is required for training visualization")
            
        if plot_type == 'loss':
            # Get training data
            data = self.monitor.get_training_data()
            if not data:
                raise ValueError("No training data available")
                
            # Create loss plot
            plotter = LossPlot()
            plotter.plot(
                epochs=np.arange(len(data['loss'])),
                loss_history=data['loss'],
                val_loss_history=data.get('val_loss'),
                show_trend=kwargs.get('show_trend', True),
                show_best=kwargs.get('show_best', True)
            )
            
            if save_path:
                plotter.fig.savefig(save_path)
                
            return plotter.fig
            
        elif plot_type == 'reward':
            # Get episode data
            data = self.monitor.get_episode_data()
            if not data:
                raise ValueError("No episode data available")
                
            # Create reward plot
            plotter = LossPlot()  # Reuse LossPlot for reward visualization
            plotter.plot(
                epochs=np.arange(len(data['rewards'])),
                loss_history=data['rewards'],
                show_trend=kwargs.get('show_trend', True),
                show_best=kwargs.get('show_best', True)
            )
            
            # Customize for reward plot
            plotter.axes.set_ylabel('Reward')
            plotter.axes.set_title('Episode Rewards')
            
            if save_path:
                plotter.fig.savefig(save_path)
                
            return plotter.fig
            
        elif plot_type == 'all':
            # Get all data
            train_data = self.monitor.get_training_data()
            episode_data = self.monitor.get_episode_data()
            
            if not train_data or not episode_data:
                raise ValueError("No training/episode data available")
                
            # Create comprehensive analysis
            plotter = OptimizationAnalysis()
            plotter.plot(
                epochs=np.arange(len(train_data['loss'])),
                loss_history=train_data['loss'],
                val_loss_history=train_data.get('val_loss'),
                weights_history=None,  # Not applicable for DeepRL
                diversity_history=None,  # Not applicable for DeepRL
                best_individual_history={
                    'fitness': episode_data['rewards'],
                    'complexity': np.ones_like(episode_data['rewards'])  # Placeholder
                }
            )
            
            if save_path:
                plotter.fig.savefig(save_path)
                
            return plotter.fig
            
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
            
    def plot_model(self, model, plot_type: str = 'tree', save_path: Optional[str] = None, **kwargs):
        """Plot model analysis visualization.
        
        Args:
            model: DeepRL model instance
            plot_type: Type of plot ('tree', 'value', 'all')
            save_path: Path to save the plot
            **kwargs: Additional arguments for specific plot types
        
        Returns:
            matplotlib Figure object or other visualization output
        """
        if plot_type == 'tree':
            # Use model's built-in tree plot
            fig = model.plot(fig_type='tree').view()
            
            if save_path:
                fig.savefig(save_path)
                
            return fig
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
            
    def plot_performance(self, u_pred: np.ndarray, u_exact: np.ndarray,
                        plot_type: str = 'residual', save_path: Optional[str] = None, **kwargs):
        """Plot performance analysis visualization.
        
        Args:
            u_pred: Predicted solution
            u_exact: Exact solution
            plot_type: Type of plot ('residual', 'comparison', 'all')
            save_path: Path to save the plot
            **kwargs: Additional arguments for specific plot types
                For residual plot:
                    - x: Spatial coordinates
                    - t: Time coordinates
                For comparison plot:
                    - show_bounds: Whether to show error bounds
                    - show_statistics: Whether to show statistics
                    
        Returns:
            matplotlib Figure object
        """
        if plot_type == 'residual':
            if 'x' not in kwargs or 't' not in kwargs:
                raise ValueError("x and t coordinates are required for residual plot")
                
            plotter = ResidualAnalysis()
            plotter.plot(u_pred, u_exact, kwargs['x'], kwargs['t'])
            
            if save_path:
                plotter.fig.savefig(save_path)
                
            return plotter.fig
            
        elif plot_type == 'comparison':
            plotter = ComparisonLine()
            plotter.plot(
                u_exact.flatten(),
                u_pred.flatten(),
                show_bounds=kwargs.get('show_bounds', True),
                show_statistics=kwargs.get('show_statistics', True)
            )
            
            if save_path:
                plotter.fig.savefig(save_path)
                
            return plotter.fig
            
        elif plot_type == 'all':
            # Create both residual and comparison plots
            fig_residual = self.plot_performance(
                u_pred, u_exact,
                plot_type='residual',
                save_path=save_path.replace('.png', '_residual.png') if save_path else None,
                **kwargs
            )
            fig_comparison = self.plot_performance(
                u_pred, u_exact,
                plot_type='comparison',
                save_path=save_path.replace('.png', '_comparison.png') if save_path else None,
                **kwargs
            )
            return fig_residual, fig_comparison
            
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}") 