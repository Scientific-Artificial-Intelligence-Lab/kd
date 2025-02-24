"""Example of running DLGA on real PDE data (KdV equation).

The KdV equation is a nonlinear PDE:
    u_t + 6uu_x + u_xxx = 0

This script demonstrates:
1. Loading real PDE data
2. Training DLGA to discover the equation
3. Visualizing the results using both vizr and new plot system
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from pathlib import Path
from typing import Dict, Tuple

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.model.dlga import DLGA
from kd.vizr.pde_comparison import plot_pde_comparison
from kd.plot.interface.monitor_utils import enable_monitoring
from kd.plot.interface.direct_plot import (
    plot_residual,
    plot_comparison,
    plot_equation_terms,
    plot_optimization,
    plot_evolution
)
from kd.plot.scientific.evolution import EvolutionPlot

def calculate_metadata(model: DLGA, X: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate metadata (derivatives) from model.
    
    Args:
        model: DLGA model instance
        X: Input points array (n_samples, 2)
        
    Returns:
        Dictionary containing u and its derivatives
    """
    # Convert to tensor
    X_meta = torch.from_numpy(X.astype(np.float32)).to(model.device)
    X_meta.requires_grad_(True)
    
    # Calculate u and first derivatives
    u_meta = model.Net(X_meta)
    u_grad = torch.autograd.grad(outputs=u_meta.sum(), inputs=X_meta, create_graph=True)[0]
    ux = u_grad[:, 0].reshape(-1, 1)
    ut = u_grad[:, 1].reshape(-1, 1)
    
    # Calculate higher derivatives
    uxx = torch.autograd.grad(outputs=ux.sum(), inputs=X_meta, create_graph=True)[0][:, 0].reshape(-1, 1)
    uxxx = torch.autograd.grad(outputs=uxx.sum(), inputs=X_meta, create_graph=True)[0][:, 0].reshape(-1, 1)
    
    # Convert to numpy
    return {
        'u': u_meta.detach().cpu().numpy(),
        'u_x': ux.detach().cpu().numpy(),
        'u_xx': uxx.detach().cpu().numpy(),
        'u_xxx': uxxx.detach().cpu().numpy(),
        'u_t': ut.detach().cpu().numpy()
    }

def calculate_equation_residual(metadata: Dict[str, np.ndarray]) -> np.ndarray:
    """Calculate KdV equation residual.
    
    Args:
        metadata: Dictionary containing u and its derivatives
        
    Returns:
        Equation residual array (n_samples, 1)
    """
    # KdV equation: u_t + 6uu_x + u_xxx = 0
    residual = (
        metadata['u_t'] + 
        6 * metadata['u'] * metadata['u_x'] + 
        metadata['u_xxx']
    )
    # Ensure output shape is (n_samples, 1)
    if len(residual.shape) == 1:
        residual = residual.reshape(-1, 1)
    return residual

def main():
    #####################################################################
    # Load and prepare KdV equation data
    #####################################################################
    print("\n1. Loading data...")
    
    # Load data from .mat file
    data_path = os.path.join(kd_main_dir, "kd/dataset/data/KdV_equation.mat")
    data = scipy.io.loadmat(data_path)

    # Extract data arrays
    t = data['tt'].flatten()  # Time points (201)
    x = data['x'].flatten()   # Spatial points (512)
    u = data['uu']           # Solution values (512 x 201)

    # Create training dataset by sampling points
    print("2. Preparing training data...")
    X_train = []
    y_train = []

    n_samples = 1000
    t_idx = np.random.randint(0, t.shape[0], n_samples)  # Sample from 0 to 200
    x_idx = np.random.randint(0, x.shape[0], n_samples)  # Sample from 0 to 511

    for i, j in zip(t_idx, x_idx):
        X_train.append([x[j], t[i]])
        y_train.append(u[j,i])  # Note: u is (x, t) indexed

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    #####################################################################
    # Initialize and train DLGA model with monitoring
    #####################################################################
    print("\n3. Setting up model and training...")
    
    # Create output directory
    plot_dir = Path(os.getcwd()) / ".plot_output"
    if not plot_dir.exists():
        plot_dir.mkdir()
    output_dir = plot_dir / "dlga_training"
    if not output_dir.exists():
        output_dir.mkdir()

    print(f"Output directory: {output_dir}")

    # Initialize model
    model = DLGA(epi=0.2, input_dim=2)  # 2D input: (x,t)
    model.vizr.realtime = True

    # Enable monitoring
    monitor = enable_monitoring(model)

    # Train the model
    print("\nTraining DLGA model...")
    model.fit(X_train, y_train)

    #####################################################################
    # Generate predictions
    #####################################################################
    print("\n4. Generating predictions...")
    
    # Create prediction points grid
    X_pred = []
    for i in range(len(x)):
        for j in range(len(t)):
            X_pred.append([x[i], t[j]])
    X_pred = np.array(X_pred)

    # Generate predictions
    X_pred_tensor = torch.from_numpy(X_pred.astype(np.float32)).to(model.device)
    with torch.no_grad():
        y_pred = model.Net(X_pred_tensor).cpu().numpy()
    u_pred = y_pred.reshape(len(x), len(t))

    #####################################################################
    # Create visualizations
    #####################################################################
    print("\n5. Creating visualizations...")

    # Calculate metadata once
    print("  - Calculating metadata...")
    metadata = calculate_metadata(model, X_train)
    equation_residual = calculate_equation_residual(metadata)

    # 1. Residual Analysis
    print("  - Creating residual analysis plots...")

    # 1.1 Solution Residual
    print("    * Solution residual")
    plot_residual(
        u_pred=u_pred.T,  # Transpose to match (nt x nx) shape
        u_exact=u.T,
        x=x,
        t=t,
        plot_type='all',
        save_path=str(output_dir / "solution_residual.png")
    )

    # 1.2 Equation Residual
    print("    * Equation residual")
    # Create scatter plot for equation residual
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residual vs x
    scatter = axes[0].scatter(X_train[:, 0], equation_residual, 
                            c=X_train[:, 1], cmap='viridis')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Equation Residual')
    axes[0].set_title('Equation Residual vs x')
    plt.colorbar(scatter, ax=axes[0], label='t')
    
    # Residual histogram
    axes[1].hist(equation_residual, bins=50, density=True)
    axes[1].set_xlabel('Equation Residual')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Equation Residual Distribution')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / "equation_residual.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Comparison Analysis
    print("  - Creating comparison plots...")
    plot_comparison(
        y_true=u.flatten(),
        y_pred=u_pred.flatten(),
        plot_type='all',
        save_path=str(output_dir / "comparison_analysis.png")
    )

    # 3. Equation Terms Analysis
    print("  - Creating equation terms plots...")
    plot_equation_terms(
        terms=metadata,
        x=X_train[:, 0],  # Spatial coordinates
        t=X_train[:, 1],  # Time coordinates
        plot_type='all',
        save_path=str(output_dir / "equation_terms.png")
    )

    # 4. Optimization Analysis
    print("  - Creating optimization plots...")
    training_data = monitor.get_training_data()
    evolution_data = monitor.get_evolution_data()

    if training_data and evolution_data:
        plot_optimization(
            epochs=training_data['epochs'],
            loss_history=training_data.get('loss'),
            val_loss_history=training_data.get('val_loss'),
            weights_history=evolution_data.get('weights'),
            diversity_history=evolution_data.get('diversity'),
            plot_type='all',
            save_path=str(output_dir / "optimization_analysis.png")
        )

    # 5. Evolution Visualization
    print("\nEvolution data:", type(evolution_data))
    print("Keys:", evolution_data.keys())
    print("Sample data:", {k: v[:5] for k, v in evolution_data.items()})

    # Create evolution visualization
    evolution_vis = EvolutionPlot(figsize=(10, 6))
    evolution_vis.save_path = os.path.join(output_dir, "evolution_analysis.png")
    evolution_vis.plot(evolution_data)
    evolution_vis.save()

    # 6. PDE comparison plot
    print("  - Creating PDE comparison plot...")
    fig = plot_pde_comparison(
        x=x,
        t=t,
        u_exact=u.T,
        u_pred=u_pred.T,
        X_train=X_train,
        slice_times=[0.25, 0.5, 0.75]
    )
    plt.savefig(str(output_dir / "pde_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll visualizations have been saved to: {output_dir}")

if __name__ == "__main__":
    main()

