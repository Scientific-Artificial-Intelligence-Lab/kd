"""Example of running DLGA on real PDE data (KdV equation).

The KdV equation is a nonlinear PDE:
    u_t + 6uu_x + u_xxx = 0

This tutorial demonstrates:
1. Loading real PDE data
2. Training DLGA to discover the equation
3. Visualizing the results using various methods
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
from kd.plot.scientific import calculate_metadata, calculate_equation_residual
from kd.plot.scientific.equations import kdv_equation
from kd.plot.scientific.residual import ResidualAnalysis

#####################################################################
# Step 1: Load and prepare KdV equation data
#####################################################################
print("\n1. Loading data...")

# Load data from .mat file
data_path = os.path.join(kd_main_dir, "kd/dataset/data/KdV_equation.mat")
data = scipy.io.loadmat(data_path)

# Extract data arrays
t = data['tt'].flatten()  # Time points (201)
x = data['x'].flatten()   # Spatial points (512)
u = data['uu']           # Solution values (512 x 201)

print(f"Data shapes: x={x.shape}, t={t.shape}, u={u.shape}")
print(f"x range: [{x.min():.2f}, {x.max():.2f}]")
print(f"t range: [{t.min():.2f}, {t.max():.2f}]")

# Create training dataset by sampling points
print("\n2. Preparing training data...")
X_train = []
y_train = []

# Randomly sample 1000 points
n_samples = 1000
t_idx = np.random.randint(0, t.shape[0], n_samples)  # Sample from 0 to 200
x_idx = np.random.randint(0, x.shape[0], n_samples)  # Sample from 0 to 511

for i, j in zip(t_idx, x_idx):
    X_train.append([x[j], t[i]])
    y_train.append(u[j,i])  # Note: u is (x, t) indexed

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Training data: X_train={X_train.shape}, y_train={y_train.shape}")

#####################################################################
# Step 2: Initialize and train DLGA model
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
print("Model training completed!")

#####################################################################
# Step 3: Generate predictions
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

print(f"Prediction completed: u_pred shape={u_pred.shape}")

#####################################################################
# Step 4: Create visualizations
#####################################################################
print("\n5. Creating visualizations...")

# Calculate metadata
print("  - Computing metadata...")
metadata = calculate_metadata(model, X_train)

# Calculate KdV equation residual (u_t + 6*u*u_x + u_xxx = 0)
kdv_residual = calculate_equation_residual(metadata, kdv_equation)

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

# Use ResidualAnalysis to visualize equation residuals
# Since ResidualAnalysis expects u_pred and u_exact, we use a trick:
# - Create a dummy u_exact of zeros
# - Use kdv_residual as (u_pred - u_exact)
dummy_u = np.zeros_like(kdv_residual)
residual_analyzer = ResidualAnalysis(figsize=(12, 5), save_path=str(output_dir / "equation_residual.png"))

# Create figure with simpler manual plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot (spatial coordinates colored by time)
scatter = axes[0].scatter(X_train[:, 0], kdv_residual, c=X_train[:, 1], cmap='viridis')
axes[0].set_xlabel('x')
axes[0].set_ylabel('Equation Residual')
axes[0].set_title('Equation Residual vs x')
plt.colorbar(scatter, ax=axes[0], label='t')

# Histogram
axes[1].hist(kdv_residual, bins=50, density=True)
axes[1].set_xlabel('Equation Residual')
axes[1].set_ylabel('Density')
axes[1].set_title('Equation Residual Distribution')

plt.tight_layout()
plt.savefig(str(output_dir / "equation_residual.png"), dpi=300, bbox_inches='tight')
plt.close(fig)

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
# Prepare KdV equation terms
terms = {
    'u_t': metadata['u_t'],
    'u*u_x': 6 * metadata['u'] * metadata['u_x'],
    'u_xxx': metadata['u_xxx'],
    'Residual': kdv_residual
}

plot_equation_terms(
    terms=terms,
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
print("  - Creating evolution visualization...")
if evolution_data:
    evolution_vis = EvolutionPlot(figsize=(10, 6))
    evolution_vis.save_path = os.path.join(output_dir, "evolution_analysis.png")
    evolution_vis.plot(evolution_data)
    evolution_vis.save()
else:
    print("    * Warning: No evolution data available for visualization")

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

# 7. metadata plane
print("\n7. Creating metadata plane visualizations...")

# 7.1 Equation residual in x-t plane
print("  - Creating equation residual in x-t plane...")
from kd.plot.scientific.metadata_plane import MetadataValuePlane
from scipy.interpolate import griddata

# Create residual distribution plot in the x-t plane
# For better visualization, interpolate irregular training points to a regular grid
x_unique = np.linspace(x.min(), x.max(), 100)
t_unique = np.linspace(t.min(), t.max(), 100)
X_mesh, T_mesh = np.meshgrid(x_unique, t_unique)

# Use training points to interpolate residuals onto the grid
grid_points = (X_mesh.flatten(), T_mesh.flatten())
interpolated_residuals = griddata(
    (X_train[:, 0], X_train[:, 1]), 
    kdv_residual.flatten(), 
    grid_points, 
    method='linear', 
    fill_value=np.nan
)
residual_grid = interpolated_residuals.reshape(X_mesh.shape)

# Create MetadataValuePlane visualization
metadata_plotter = MetadataValuePlane(figsize=(10, 8))
fig = metadata_plotter.plot(
    x_values=x_unique,
    y_values=t_unique, 
    z_values=residual_grid,
    x_label='Spatial coordinate (x)',
    y_label='Time (t)',
    z_label='Equation Residual',
    plot_type='contour',
    colormap='viridis',
    title='KdV Equation Residual Distribution'
)

# Add training points as scatter plot
sample_indices = np.random.choice(len(X_train), min(300, len(X_train)), replace=False)
metadata_plotter.ax.scatter(
    X_train[sample_indices, 0], 
    X_train[sample_indices, 1], 
    c='red', 
    s=10, 
    alpha=0.5, 
    label='Training points'
)
metadata_plotter.ax.legend()

plt.savefig(str(output_dir / "metadata_plane_residual.png"), dpi=300, bbox_inches='tight')
plt.close(fig)

# 7.2 Derivative relationships in PDE discovery
print("  - Creating derivative relationship visualization...")

# Extract relevant KdV equation terms
u_values = metadata['u'].flatten()
u_x_values = metadata['u_x'].flatten()
u_xxx_values = metadata['u_xxx'].flatten()
u_t_values = metadata['u_t'].flatten()

# Create combined term for KdV equation: 6*u*u_x
combined_term = 6 * u_values * u_x_values

# Sample points to avoid overcrowding in plots
if len(u_values) > 300:
    sample_indices = np.random.choice(len(u_values), 300, replace=False)
    u_values = u_values[sample_indices]
    u_x_values = u_x_values[sample_indices]
    u_xxx_values = u_xxx_values[sample_indices]
    u_t_values = u_t_values[sample_indices]
    combined_term = combined_term[sample_indices]
    sample_coords = X_train[sample_indices]
else:
    sample_coords = X_train

# Create standard matplotlib figure with subplots instead of using CompositePlot
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# First subplot: u_t vs 6*u*u_x
scatter1 = axes[0].scatter(
    combined_term, 
    u_t_values, 
    c=sample_coords[:, 0],  # Color by spatial coordinates
    alpha=0.7,
    cmap='viridis'
)
axes[0].set_xlabel('6*u*u_x')
axes[0].set_ylabel('u_t')
axes[0].set_title('u_t vs 6*u*u_x Term Relationship')
axes[0].grid(True, linestyle=':')
plt.colorbar(scatter1, ax=axes[0], label='x coordinate')

# Second subplot: u_t vs u_xxx
scatter2 = axes[1].scatter(
    u_xxx_values, 
    u_t_values, 
    c=sample_coords[:, 1],  # Color by time coordinates
    alpha=0.7,
    cmap='plasma'
)
axes[1].set_xlabel('u_xxx')
axes[1].set_ylabel('u_t')
axes[1].set_title('u_t vs u_xxx Term Relationship')
axes[1].grid(True, linestyle=':')
plt.colorbar(scatter2, ax=axes[1], label='t coordinate')

# Add KdV equation reference lines: u_t + 6*u*u_x + u_xxx = 0
# => u_t = -6*u*u_x - u_xxx
x_range1 = np.linspace(min(combined_term), max(combined_term), 100)
axes[0].plot(x_range1, -x_range1, 'r--', label='KdV equation relationship')
axes[0].legend()

x_range2 = np.linspace(min(u_xxx_values), max(u_xxx_values), 100)
axes[1].plot(x_range2, -x_range2, 'r--', label='KdV equation relationship')
axes[1].legend()

plt.tight_layout()
plt.savefig(str(output_dir / "metadata_plane_derivatives.png"), dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\nAll visualizations have been saved to: {output_dir}")