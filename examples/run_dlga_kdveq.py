"""Example of running DLGA on real PDE data (KdV equation).

The KdV equation is a nonlinear PDE:
    u_t + 6uu_x + u_xxx = 0

This script demonstrates:
1. Loading real PDE data
2. Training DLGA to discover the equation
3. Visualizing the results
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.dataset import load_kdv_equation
from kd.model.dlga import DLGA
from kd.vizr.pde_comparison import plot_pde_comparison

#####################################################################
# Load and prepare KdV equation data
#####################################################################

# Load KdV equation data
kdv_data = load_kdv_equation()

# Extract data
X_train, y_train = kdv_data.sample(n_samples=1000)

#####################################################################
# Initialize and train DLGA model
#####################################################################

# Initialize model
model = DLGA(epi=0.2, input_dim=2)  # 2D input: (x,t)
model.vizr.realtime = True

# Train the model
print("\nTraining DLGA model...")
model.fit(X_train, y_train)

#####################################################################
# Generate predictions and create visualizations
#####################################################################

print("\nGenerating predictions for visualization...")
# Create prediction points grid

# x = kdv_data.x
# t = kdv_data.t
# u = kdv_data.u
# X_pred = []
# for i in range(len(x)):
#     for j in range(len(t)):
#         X_pred.append([x[i], t[j]])
# X_pred = np.array(X_pred)

X_pred = kdv_data.mesh()

data_shape = kdv_data.get_size()

# Generate predictions
X_pred_tensor = torch.from_numpy(X_pred.astype(np.float32)).to(model.device)
with torch.no_grad():
    y_pred = model.Net(X_pred_tensor).cpu().numpy()
u_pred = y_pred.reshape(*data_shape)

# Create and save visualization
print("\nCreating visualization...")
try:
    # Setup result directory
    result_dir = os.path.join(os.getcwd(), 'result_save')
    print(f"Using result directory at: {result_dir}")
    
    # Create comparison plot
    print("Creating comparison plot...")
    fig = plot_pde_comparison(
        x=kdv_data.x,
        t=kdv_data.t,
        u_exact=kdv_data.u.T,  # Transpose to match expected shape
        u_pred=u_pred.T,  # Transpose to match expected shape
        X_train=X_train,
        slice_times=[0.25, 0.5, 0.75]
    )
    
    # Save plot
    save_path = os.path.join(result_dir, 'kdv_comparison.png')
    print(f"Saving figure to: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Successfully saved visualization as {save_path}")
except Exception as e:
    print(f"Error during visualization: {str(e)}")
    import traceback
    traceback.print_exc()
