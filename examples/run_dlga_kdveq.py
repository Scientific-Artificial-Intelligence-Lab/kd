"""Example of running DLGA on real PDE data (KdV equation).
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.dataset import load_kdv_equation
from kd.model.dlga import DLGA
from kd.viz.dlga_viz import *
from kd.viz.dlga_kdv import *

#####################################################################
# 1. Load and prepare data
#####################################################################

# Load KdV equation data
kdv_data = load_kdv_equation()
x, t, u = kdv_data.x, kdv_data.t, kdv_data.usol

# Extract data
X_train, y_train = kdv_data.sample(n_samples=1000)

#####################################################################
# 2. Model training
#####################################################################

# Initialize model
model = DLGA(epi=kdv_data.epi, input_dim=2)  # 2D input: (x,t)

# Train the model
print("\nTraining DLGA model...")
model.fit(X_train, y_train)

#####################################################################
# 3. Generate predictions
#####################################################################
print("\nGenerating predictions...")

# Create full grid for visualization
X_full = kdv_data.mesh()

# Convert to tensor and predict
X_tensor = torch.from_numpy(X_full.astype(np.float32)).to(model.device)
with torch.no_grad():
    u_pred = model.Net(X_tensor).cpu().numpy().reshape(u.shape)

#####################################################################
# 4. Visualizations
#####################################################################
print("\nCreating visualizations...")

# 配置全局绘图样式
configure_plotting(cmap='viridis')

# 创建输出目录
VIZ_DIR = Path(".plot_output")

#-------------------------------------------------------------------
# 1. Training Process Visualization
#-------------------------------------------------------------------
# 1.1 训练损失曲线
plot_training_loss(model, output_dir=VIZ_DIR)

# 1.2 验证损失曲线
plot_validation_loss(model, output_dir=VIZ_DIR)

# 1.3 优化分析（权重和多样性历史）
plot_optimization_analysis(model, output_dir=VIZ_DIR)

# 1.4 进化可视化
plot_evolution(model, output_dir=VIZ_DIR)

#-------------------------------------------------------------------
# 2. Solution Analysis
#-------------------------------------------------------------------
# 2.1 PDE解对比图
plot_pde_comparison(
    x=x,
    t=t,
    u_true=u,
    u_pred=u_pred,
    output_dir=VIZ_DIR
)

# 2.2 残差分析
plot_residual_analysis(
    model=model,
    X_train=X_train,
    y_train=y_train,
    u_true=u,
    u_pred=u_pred,
    output_dir=VIZ_DIR
)

# 2.3 时间切片对比图
plot_time_slices(
    x=x,
    t=t,
    u_true=u,
    u_pred=u_pred,
    slice_times=[0.25, 0.5, 0.75],  # 示例时间切片
    output_dir=VIZ_DIR
)

#-------------------------------------------------------------------
# 3. Equation Discovery Analysis
#-------------------------------------------------------------------
# 3.1 方程项关系可视化
plot_equation_terms(
    model.metadata,
    terms={
        'x_term': {'vars': ['u', 'u_x'], 'label': '6uu_x'},
        'y_term': {'vars': ['u_xxx'], 'label': '-u_xxx'}
    },
    equation_name="KdV Equation",
    output_dir=VIZ_DIR
)

# 3.2 元数据平面可视化（x-t平面上的方程残差分布）
plot_metadata_plane(
    metadata=model.metadata,
    x=x,
    t=t,
    output_dir=VIZ_DIR
)

# 3.3 导数关系可视化
plot_derivative_relationships(
    metadata=model.metadata,
    output_dir=VIZ_DIR
)
