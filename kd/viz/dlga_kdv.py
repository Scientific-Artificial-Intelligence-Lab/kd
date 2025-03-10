"""Specialized visualization module for KdV equation analysis in DLGA framework.

This module provides visualization tools specifically designed for analyzing 
the KdV (Korteweg-de Vries) equation results from DLGA model.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .dlga_viz import PLOT_STYLE
from scipy.interpolate import griddata

def plot_metadata_plane(metadata, x, t, output_dir: str = None):
    """Visualize KdV equation characteristics on the x-t plane.
    
    Args:
        metadata: Dictionary containing equation terms and derivatives
        x: Spatial coordinates
        t: Time coordinates 
        output_dir: Output directory path
    """
    
    with plt.style.context(PLOT_STYLE):
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
        
        # 创建目标网格点
        T, X = np.meshgrid(t, x)
        nx, nt = len(x), len(t)
        grid_points = (T, X)
        
        # 准备源数据点(1000个训练点)
        n_samples = len(metadata['u_t'])
        # 使用sqrt(n_samples)作为每维度的采样点数
        n_per_dim = int(np.sqrt(n_samples))
        
        # 创建均匀网格的训练点
        x_train = np.linspace(x.min(), x.max(), n_per_dim)
        t_train = np.linspace(t.min(), t.max(), n_per_dim)
        T_train, X_train = np.meshgrid(t_train, x_train)
        
        # 重塑训练点为(n_samples, 2)形状
        points = np.vstack([T_train.ravel(), X_train.ravel()]).T
        
        # 提取元数据中的各项并重塑
        u_t = metadata['u_t'].reshape(-1)[:points.shape[0]]  # 确保维度匹配
        u = metadata['u'].reshape(-1)[:points.shape[0]]
        u_x = metadata['u_x'].reshape(-1)[:points.shape[0]]
        u_xxx = metadata['u_xxx'].reshape(-1)[:points.shape[0]]
        
        # 使用griddata进行插值
        print(f"\nInterpolation info:")
        print(f"Source points shape: {points.shape}")
        print(f"Target grid shape: T={T.shape}, X={X.shape}")
        print(f"Values shape: u_t={u_t.shape}, u={u.shape}")
        
        # 进行插值(使用linear方法避免cubic方法可能的问题)
        u_t_grid = griddata(points, u_t, grid_points, method='linear')
        u_grid = griddata(points, u, grid_points, method='linear')
        u_x_grid = griddata(points, u_x, grid_points, method='linear')
        u_xxx_grid = griddata(points, u_xxx, grid_points, method='linear')
        
        # 计算残差
        residual = u_t_grid + 6 * u_grid * u_x_grid + u_xxx_grid
        
        # 左图: 残差分布热图
        im1 = axes[0].pcolormesh(T, X, residual, 
                                cmap='coolwarm', 
                                shading='auto')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Space')
        axes[0].set_title('KdV Equation Residual Distribution')
        fig.colorbar(im1, ax=axes[0], label='Residual')
        
        # 右图: 各项的绝对值分布
        terms = {
            'u_t': np.abs(u_t_grid),
            '6uu_x': np.abs(6 * u_grid * u_x_grid),
            'u_xxx': np.abs(u_xxx_grid)
        }
        
        # 计算每个项沿时间维度的平均值
        for term_name, values in terms.items():
            # 使用nanmean以防插值产生NaN
            mean_values = np.nanmean(values, axis=1)
            axes[1].plot(x, mean_values, label=term_name, alpha=0.7)
            
        axes[1].set_xlabel('Space')
        axes[1].set_ylabel('Average Magnitude')
        axes[1].set_title('Term Magnitudes Along Space')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        if output_dir:
            viz_dir = Path(output_dir)
            viz_dir.mkdir(exist_ok=True)
            plt.savefig(viz_dir / "metadata_plane.png", dpi=300)
        else:
            plt.show()
        plt.close()

        # 打印统计信息
        print("\nMetadata Plane Analysis:")
        print(f"Residual statistics:")
        print(f"  Mean: {np.nanmean(residual):.4e}")
        print(f"  Std:  {np.nanstd(residual):.4e}")
        print(f"  Max:  {np.nanmax(np.abs(residual)):.4e}")
        print("\nTerm magnitude statistics:")
        for term_name, values in terms.items():
            mean_mag = np.nanmean(np.abs(values))
            print(f"  {term_name}: {mean_mag:.4e}")
