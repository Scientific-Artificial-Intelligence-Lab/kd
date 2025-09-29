"""Example showcasing the user-friendly KD visualization helpers with KD_DLGA.

This script mirrors ``examples/kd_dlga_example.py`` but routes visualization
through the high-level helper functions (``kd.viz.plot_*`` and
``kd.viz.render_equation``), covering all intents exposed by
``DLGAVizAdapter``.
"""

import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.dataset import load_pde
from kd.model.kd_dlga import KD_DLGA
from kd.viz import (
    configure,
    list_capabilities,
    plot_derivative_relationships,
    plot_field_comparison,
    plot_optimization,
    plot_parity,
    plot_residuals,
    plot_search_evolution,
    plot_time_slices,
    plot_training_curve,
    plot_validation_curve,
    render_equation,
)


# --- Data loading ---------------------------------------------------------
kdv_data = load_pde('kdv')
X_train, y_train = kdv_data.sample(n_samples=1000)


# --- Model setup ----------------------------------------------------------
model = KD_DLGA(
    operators=['u', 'u_x', 'u_xx', 'u_xxx'],
    epi=0.1,
    input_dim=2,
    verbose=False,
    max_iter=9000,
)


# --- Training -------------------------------------------------------------
model.fit(X_train, y_train)


# Prepare reusable predictions for diagnostics
y_pred_train = model.predict(X_train).reshape(-1)
X_full = kdv_data.mesh()
u_pred_field = model.predict(X_full).reshape(kdv_data.get_size())


# --- Unified viz façade configuration -------------------------------------
output_root = Path("artifacts") / "dlga_viz"
configure(save_dir=output_root)

caps = list_capabilities(model)
print("\nDLGA adapter capabilities:", ", ".join(sorted(caps)))


# --- Visualizations via helper functions ----------------------------------
# 训练损失曲线
plot_training_curve(model)

# 验证损失曲线
plot_validation_curve(model)

# 搜索/进化过程可视化
plot_search_evolution(model)

# 优化指标（复杂度/种群等）分析
plot_optimization(model)

# 方程渲染（LaTeX → PNG）
render_equation(model, font_size=14)

# 残差诊断（提供 actual/predicted 与可选坐标）
plot_residuals(
    model,
    actual=y_train.reshape(-1),
    predicted=y_pred_train,
    coordinates=X_train,
    bins=40,
)

# PDE 场对比（真实场 vs 预测场）
plot_field_comparison(
    model,
    x_coords=kdv_data.x,
    t_coords=kdv_data.t,
    true_field=kdv_data.usol,
    predicted_field=u_pred_field,
)

# 时间切片对比
plot_time_slices(
    model,
    x_coords=kdv_data.x,
    t_coords=kdv_data.t,
    true_field=kdv_data.usol,
    predicted_field=u_pred_field,
    slice_times=[0.25, 0.5, 0.75],
)

# 导数项关系
plot_derivative_relationships(model, top_n_terms=4)

# 方程奇偶图
plot_parity(model, title="Parity Plot of Discovered PDE")


print("\nDone. Check the artifacts directory for generated figures.")
