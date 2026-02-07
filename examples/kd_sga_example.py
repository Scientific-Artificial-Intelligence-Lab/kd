"""KD_SGA end-to-end example (script style).

This script shows how to:

- Load a PDE dataset via ``kd.dataset.load_pde``;
- Discover a PDE using ``KD_SGA.fit_dataset``;
- Visualize the discovered equation using the unified ``kd.viz`` façade
  (plus the legacy ``model.plot_results`` helper for compatibility).
"""

from _bootstrap import ensure_project_root_on_syspath

# 1. Project root & imports / 工程根目录与依赖导入
PROJECT_ROOT = ensure_project_root_on_syspath()

from kd.dataset import load_pde
from kd.model.kd_sga import KD_SGA
from kd.viz import (
    configure,
    render_equation,
    plot_field_comparison,
    plot_time_slices,
    plot_parity,
    plot_residuals,
)

# 2. Basic configuration / 基本配置

# Dataset name / 数据集名称
DATASET_NAME = "burgers"  # 可选: 'chafee-infante', 'burgers', 'kdv'

# Directory for visualization artifacts / 可视化输出目录
SAVE_DIR = PROJECT_ROOT / "artifacts" / "sga_viz"

# Configure viz façade early so that all following plots share the same output dir.
# 提前配置可视化前端，后续所有图像都会写入同一目录。
configure(save_dir=SAVE_DIR)


# 3. Load dataset / 加载 PDE 数据集
dataset = load_pde(DATASET_NAME)


# 4. Create and train KD_SGA / 创建并训练 KD_SGA 模型
model = KD_SGA(
    sga_run=100,
    num=20,
    depth=4,
    width=5,
    p_var=0.5,
    p_mute=0.3,
    p_cro=0.5,
    seed=0,
)

print("[KD_SGA Example] Starting PDE discovery with fit_dataset(PDEDataset)...")
model.fit_dataset(dataset)


# 5. Print discovered equation / 打印发现的方程
print("\n[Result] Discovered equation (raw string):")
print(f"  {model.best_pde_}")

latex_full = model.equation_latex()
latex_structure = model.equation_latex(include_coefficients=False)

print("[Result] LaTeX (with coefficients):")
print(f"  {latex_full}")
print("[Result] LaTeX (structure only):")
print(f"  {latex_structure}")


# 6. Equation figure via kd.viz API / 使用 kd.viz API 渲染方程图像
# Render the discovered PDE as a LaTeX equation image.
render_equation(model)


# 7. Additional visualizations via named helpers /
#    使用辅助函数进行额外可视化
#
# For KD_SGA the adapter reads everything from model.context_, so the
# generic arguments (x_coords, actual, predicted, etc.) are not used.
# We still pass None to match the shared API shape.

# Compare metadata-generated field vs original field on a heatmap.
# 比较元数据场与原始场（热力图形式）。
plot_field_comparison(model, x_coords=None, t_coords=None, true_field=None, predicted_field=None)

# Plot several time slices of u(x, t) to compare shapes over time.
# 在若干时间点比较 u(x, t) 的截面形状。
plot_time_slices(
    model,
    x_coords=None,
    t_coords=None,
    true_field=None,
    predicted_field=None,
    slice_times=[0.0, 0.5, 1.0],
)

# Parity plot: predicted RHS vs true u_t.
# 奇偶图：预测 RHS 与真实 u_t 的散点比较。
plot_parity(model, title="KD_SGA Parity Plot")

# Residual diagnostics: histogram and heatmap of u_t - RHS.
# 残差诊断：u_t - RHS 的直方图与热力图。
plot_residuals(model, actual=None, predicted=None, bins=40)


# 8. Legacy SGA visualizer / 旧版 SGA 可视化
print("\n[KD_SGA Example] Calling legacy model.plot_results() (writes assets into CWD)...")
model.plot_results()
