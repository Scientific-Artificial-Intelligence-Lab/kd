"""Equation terms plotting functionality.

This module provides classes for visualizing equation terms and their relationships,
including heatmaps of individual terms and comprehensive term analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional, Callable, Union, List
import seaborn as sns
from scipy import stats
import torch

from ..core.base import BasePlot, CompositePlot


class TermsHeatmap(BasePlot):
    """Heatmap visualization of equation terms."""
    
    def _validate_data(self, terms: Dict[str, np.ndarray]) -> bool:
        """Validate input data shapes match."""
        if not terms:
            raise ValueError("No terms provided")
            
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
        if not terms:
            print("Warning: No terms provided, skipping plot")
            return
            
        # Create figure with GridSpec
        self.fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # Terms heatmap
        ax1 = self.fig.add_subplot(gs[0, 0])
        self._plot_combined_heatmap(terms, x, t, ax1)
        
        # Correlation matrix
        if show_correlations and len(terms) > 1:
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
        if len(terms) < 2:
            ax.text(0.5, 0.5, 'Insufficient terms\nfor correlation',
                   ha='center', va='center')
            ax.axis('off')
            return
            
        try:
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
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Warning: Could not compute correlation matrix: {e}")
            ax.text(0.5, 0.5, 'Could not compute\ncorrelations',
                   ha='center', va='center')
            ax.axis('off')
    
    def _plot_term_magnitudes(self, terms: Dict[str, np.ndarray],
                            ax: plt.Axes):
        """Plot magnitude distribution of terms."""
        if not terms:
            ax.text(0.5, 0.5, 'No terms to plot',
                   ha='center', va='center')
            ax.axis('off')
            return
            
        positions = range(len(terms))
        ax.boxplot([values.flatten() for values in terms.values()],
                  tick_labels=terms.keys())
        ax.set_title('Term Magnitudes')
        ax.set_ylabel('Value')
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def _plot_statistics(self, terms: Dict[str, np.ndarray],
                        ax: plt.Axes):
        """Plot statistical summary of terms."""
        if not terms:
            ax.text(0.5, 0.5, 'No terms to analyze',
                   ha='center', va='center')
            ax.axis('off')
            return
            
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

def calculate_metadata(model, X: np.ndarray) -> Dict[str, np.ndarray]:
    """计算模型元数据（导数）。
    
    参数:
        model: 具有Net属性的模型实例 (如DLGA模型)
        X: 输入点数组 (n_samples, input_dim)
        
    返回:
        Dict: 包含u及其导数的字典
    """
    # 转换为张量
    device = model.device if hasattr(model, 'device') else 'cpu'
    X_meta = torch.from_numpy(X.astype(np.float32)).to(device)
    X_meta.requires_grad_(True)
    
    # 计算u (函数值)
    u_meta = model.Net(X_meta)
    
    # 计算一阶导数
    u_grad = torch.autograd.grad(outputs=u_meta.sum(), inputs=X_meta, create_graph=True)[0]
    
    # 创建基本元数据字典
    metadata = {
        'u': u_meta.detach().cpu().numpy(),
    }
    
    # 添加各个维度的导数
    input_dim = X_meta.shape[1]
    dim_names = ['x', 't', 'z'] if input_dim <= 3 else [f'x{i}' for i in range(input_dim)]
    
    for i in range(input_dim):
        # 一阶导数
        dim_grad = u_grad[:, i].reshape(-1, 1)
        metadata[f'u_{dim_names[i]}'] = dim_grad.detach().cpu().numpy()
        
        # 二阶导数
        dim_grad2 = torch.autograd.grad(outputs=dim_grad.sum(), inputs=X_meta, create_graph=True)[0][:, i].reshape(-1, 1)
        metadata[f'u_{dim_names[i]}{dim_names[i]}'] = dim_grad2.detach().cpu().numpy()
        
        # 仅对空间维度计算三阶导数
        if i == 0:  # 假设第一维是主要空间维度
            dim_grad3 = torch.autograd.grad(outputs=dim_grad2.sum(), inputs=X_meta, create_graph=True)[0][:, i].reshape(-1, 1)
            metadata[f'u_{dim_names[i]}{dim_names[i]}{dim_names[i]}'] = dim_grad3.detach().cpu().numpy()
    
    # 混合导数 (针对2D+情况)
    if input_dim >= 2:
        for i in range(input_dim):
            for j in range(i+1, input_dim):
                # 计算混合二阶导数 (如u_xt)
                grad_i = u_grad[:, i].reshape(-1, 1)
                grad_ij = torch.autograd.grad(outputs=grad_i.sum(), inputs=X_meta, create_graph=True)[0][:, j].reshape(-1, 1)
                metadata[f'u_{dim_names[i]}{dim_names[j]}'] = grad_ij.detach().cpu().numpy()
    
    return metadata

def calculate_equation_residual(metadata: Dict[str, np.ndarray], 
                              equation_func: Callable) -> np.ndarray:
    """计算方程残差。
    
    参数:
        metadata: 包含u及其导数的字典
        equation_func: 方程定义函数，接收metadata返回残差
        
    返回:
        方程残差数组 (n_samples, 1)
    """
    if equation_func is None:
        raise ValueError("必须提供equation_func参数，用于定义方程")
    
    residual = equation_func(metadata)
    
    # 确保输出形状为 (n_samples, 1)
    if len(residual.shape) == 1:
        residual = residual.reshape(-1, 1)
    return residual 