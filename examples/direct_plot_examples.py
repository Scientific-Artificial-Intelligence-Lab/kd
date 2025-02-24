"""Examples of using direct plot interface.

这个文件展示了如何使用 Direct Plot interface 进行各种可视化。
包括残差分析、对比分析、方程分析等。
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from kd.plot.interface.direct_plot import (
    plot_residual,
    plot_comparison,
    plot_equation_terms,
    plot_optimization,
    plot_evolution
)

# Create output directory
output_dir = Path(os.getcwd()) / ".plot_output" / "plot_examples"
output_dir.mkdir(parents=True, exist_ok=True)

def example_residual_analysis():
    """Example of residual analysis plots."""
    print("\n=== Residual Analysis Example ===")
    
    # Generate sample data
    nx, nt = 100, 100
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    
    # Create exact solution
    X, T = np.meshgrid(x, t)
    u_exact = np.sin(np.pi * X) * np.cos(np.pi * T)
    
    # Create predicted solution with some noise
    u_pred = u_exact + 0.1 * np.random.randn(nt, nx)
    
    # Create different types of residual plots
    plot_types = ['heatmap', 'timeslice', 'histogram', 'all']
    for plot_type in plot_types:
        save_path = output_dir / f"residual_{plot_type}.png"
        print(f"Creating {plot_type} plot...")
        plot_residual(
            u_pred=u_pred,
            u_exact=u_exact,
            x=x,
            t=t,
            plot_type=plot_type,
            save_path=str(save_path)
        )
        print(f"Saved to {save_path}")

def example_comparison_analysis():
    """Example of comparison plots."""
    print("\n=== Comparison Analysis Example ===")
    
    # Generate sample data
    n_samples = 1000
    y_true = np.random.normal(0, 1, n_samples)
    y_pred = y_true + 0.2 * np.random.randn(n_samples)
    
    # Create different types of comparison plots
    plot_types = ['scatter', 'line', 'all']
    for plot_type in plot_types:
        save_path = output_dir / f"comparison_{plot_type}.png"
        print(f"Creating {plot_type} plot...")
        plot_comparison(
            y_true=y_true,
            y_pred=y_pred,
            plot_type=plot_type,
            save_path=str(save_path)
        )
        print(f"Saved to {save_path}")

def example_equation_analysis():
    """Example of equation term analysis."""
    print("\n=== Equation Analysis Example ===")
    
    # Generate sample equation terms
    nx, nt = 50, 50
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    
    terms = {
        'u_t': np.sin(np.pi * X) * np.cos(np.pi * T),
        'u_x': np.cos(np.pi * X) * np.sin(np.pi * T),
        'u_xx': -np.sin(np.pi * X) * np.sin(np.pi * T)
    }
    
    # Create different types of equation plots
    plot_types = ['heatmap', 'all']
    for plot_type in plot_types:
        save_path = output_dir / f"equation_{plot_type}.png"
        print(f"Creating {plot_type} plot...")
        plot_equation_terms(
            terms=terms,
            x=x,
            t=t,
            plot_type=plot_type,
            save_path=str(save_path)
        )
        print(f"Saved to {save_path}")

def example_optimization_analysis():
    """Example of optimization process visualization."""
    print("\n=== Optimization Analysis Example ===")
    
    # Generate sample optimization data
    n_epochs = 200
    epochs = np.arange(n_epochs)
    
    # Loss history with decreasing trend
    loss = 1.0 / (1 + 0.05 * epochs) + 0.1 * np.random.randn(n_epochs)
    
    # Weights history (n_epochs x n_weights)
    n_weights = 5
    weights = np.zeros((n_epochs, n_weights))
    for i in range(n_weights):
        weights[:, i] = np.sin(2*np.pi*i/n_weights * epochs/n_epochs) + 0.1 * np.random.randn(n_epochs)
    
    # Diversity history with decreasing trend
    diversity = np.exp(-0.01 * epochs) + 0.1 * np.random.randn(n_epochs)
    
    # Create different types of optimization plots
    plot_types = ['loss', 'weights', 'diversity', 'all']
    for plot_type in plot_types:
        save_path = output_dir / f"optimization_{plot_type}.png"
        print(f"Creating {plot_type} plot...")
        plot_optimization(
            epochs=epochs,
            loss_history=loss,
            weights_history=weights,  # Now has correct shape (n_epochs, n_weights)
            diversity_history=diversity,
            plot_type=plot_type,
            save_path=str(save_path)
        )
        print(f"Saved to {save_path}")

def example_evolution_visualization():
    """Example of evolution process visualization."""
    print("\n=== Evolution Visualization Example ===")
    
    # Generate mock evolution data
    from kd.tests.test_evolution_basic import generate_mock_data
    evolution_data = generate_mock_data(n_generations=20, pop_size=50)
    
    # Create animation
    print("Creating evolution animation...")
    anim = plot_evolution(
        evolution_data,
        mode='animation',
        interval=100
    )
    
    # Create snapshots in different formats
    for fmt in ['mp4', 'gif', 'frames']:
        if fmt == 'frames':
            save_path = output_dir / "evolution_frames"
        else:
            save_path = output_dir / f"evolution.{fmt}"
            
        print(f"Creating evolution {fmt}...")
        output = plot_evolution(
            evolution_data,
            mode='snapshot',
            output_format=fmt,
            save_path=str(save_path)
        )
        print(f"Saved to {output}")

def main():
    """Run all examples."""
    print("Running Direct Plot interface examples...")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Run examples
    example_residual_analysis()
    example_comparison_analysis()
    example_equation_analysis()
    example_optimization_analysis()
    example_evolution_visualization()
    
    print("\nAll examples completed. Check output directory for results.")

if __name__ == "__main__":
    main() 