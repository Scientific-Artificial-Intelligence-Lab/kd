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

current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.model.dlga import DLGA
from kd.vizr.pde_comparison import plot_pde_comparison

# Load KdV equation data
data_path = os.path.join(kd_main_dir, "kd/dataset/data/KdV_equation.mat")
data = scipy.io.loadmat(data_path)

# Extract data with correct key names
t = data['tt'].flatten()  # Time points (201)
x = data['x'].flatten()  # Spatial points (512)
u = data['uu']           # Solution values (512 x 201)

# Create training data
X_train = []
y_train = []

# Sample some points for training
n_samples = 1000
t_idx = np.random.randint(0, t.shape[0], n_samples)  # 0 to 200
x_idx = np.random.randint(0, x.shape[0], n_samples)  # 0 to 511

for i, j in zip(t_idx, x_idx):
    X_train.append([x[j], t[i]])
    y_train.append(u[j,i])  # Note: u is (x, t) indexed

X_train = np.array(X_train)
y_train = np.array(y_train)

# Initialize and train DLGA
model = DLGA(epi=0.2, input_dim=2)  # 2D input: (x,t)

# Monkey patch the train_NN method to use fewer epochs
original_train_NN = model.train_NN
def train_NN_with_fewer_epochs(self, X, y):
    # Save original range
    original_range = range(50000)
    # Replace with shorter range
    range_replacement = range(5000)  # Reduce from 50000
    # Patch the range
    import builtins
    original_range_func = builtins.range
    builtins.range = lambda *args: range_replacement if args == (50000,) else original_range_func(*args)
    # Call original method
    result = original_train_NN(X, y)
    # Restore original range
    builtins.range = original_range_func
    return result

model.train_NN = train_NN_with_fewer_epochs.__get__(model, DLGA)
model.n_generations = 20  # Reduce GA generations from 100

model.vizr.realtime = False

# Train the model
model.fit(X_train, y_train)

# Generate predictions for visualization
print("\nGenerating predictions for visualization...")
X_pred = []
for i in range(len(x)):
    for j in range(len(t)):
        X_pred.append([x[i], t[j]])
X_pred = np.array(X_pred)

# Get predictions using the neural network directly
X_pred_tensor = torch.from_numpy(X_pred.astype(np.float32)).to(model.device)
with torch.no_grad():
    y_pred = model.Net(X_pred_tensor).cpu().numpy()
u_pred = y_pred.reshape(len(x), len(t))

print("\nCreating visualization...")
try:
    # Use result_save directory for saving the plot
    result_dir = os.path.join(os.getcwd(), 'result_save')
    print(f"Using result directory at: {result_dir}")
    
    # Create the comparison plot
    print("Creating comparison plot...")
    fig = plot_pde_comparison(
        x=x,
        t=t,
        u_exact=u.T,  # Transpose to match expected shape
        u_pred=u_pred.T,  # Transpose to match expected shape
        X_train=X_train,
        slice_times=[0.25, 0.5, 0.75]
    )
    
    # Save the figure
    save_path = os.path.join(result_dir, 'kdv_comparison.png')
    print(f"Saving figure to: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Successfully saved visualization as {save_path}")
except Exception as e:
    print(f"Error during visualization: {str(e)}")
    import traceback
    traceback.print_exc()
