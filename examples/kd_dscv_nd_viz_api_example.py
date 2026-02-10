"""Unified visualization facade usage for KD_DSCV with N-D data.

This script demonstrates how to:

- Construct a 3D dataset using the N-D PDEDataset API
- Run PDE discovery with KD_DSCV (Regular/FD mode) on multi-spatial data
- Visualize results exclusively through the unified ``kd.viz`` facade
  (VizRequest / configure / list_capabilities / render)

Seven visualization intents are exercised: search_evolution, density, tree,
equation, residual, field_comparison, parity.
"""

import numpy as np
import warnings

from _bootstrap import ensure_project_root_on_syspath

PROJECT_ROOT = ensure_project_root_on_syspath()

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings("ignore", category=FutureWarning, module='numpy.*')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.*')

from kd.dataset import PDEDataset
from kd.model import KD_DSCV
from kd.viz import VizRequest, configure, list_capabilities, render


def _log(intent: str, result) -> None:
    if result.warnings:
        print(f"[viz_api:{intent}] warning -> {'; '.join(result.warnings)}")
    else:
        saved = ', '.join(str(p) for p in result.paths) or '<no files>'
        print(f"[viz_api:{intent}] saved -> {saved}")


# --- Synthetic 3D data (x, y, t) / 合成 3D 数据 --------------------------------

nx, ny, nt = 20, 20, 30
x = np.linspace(0, 2 * np.pi, nx)
y = np.linspace(0, 2 * np.pi, ny)
t = np.linspace(0, 1, nt)

X, Y, T = np.meshgrid(x, y, t, indexing="ij")

# u(x, y, t) = sin(x) * cos(y) * exp(-t)  ->  u_t = -u
u = np.sin(X) * np.cos(Y) * np.exp(-T)

print(f"[DSCV N-D Viz API] Generated 3D data with shape: {u.shape}")
print(f"  x: {nx} points, y: {ny} points, t: {nt} points")


# --- PDEDataset construction / 构造 PDEDataset ---------------------------------

dataset = PDEDataset(
    equation_name="2d_decay",
    fields_data={"u": u},
    coords_1d={"x": x, "y": y, "t": t},
    axis_order=["x", "y", "t"],
    target_field="u",
    lhs_axis="t",
)

print(f"[DSCV N-D Viz API] Created PDEDataset:")
print(f"  equation_name: {dataset.equation_name}")
print(f"  axis_order: {dataset.axis_order}")


# --- Model creation & training / 创建模型并训练 ---------------------------------

model = KD_DSCV(
    binary_operators=["add", "mul", "Diff"],
    unary_operators=['n2'],
    n_samples_per_batch=500,
)

np.random.seed(42)

model.import_dataset(dataset)
step_output = model.train(n_epochs=11, verbose=False)

print(f"\n[DSCV N-D Viz API] Best expression: {step_output['expression']}")
print(f"[DSCV N-D Viz API] Best reward: {step_output['r']}")


# --- Unified facade configuration / 统一可视化前端配置 --------------------------

output_root = PROJECT_ROOT / "artifacts" / "dscv_nd_viz_api"
configure(save_dir=output_root)

caps = list_capabilities(model)
print("\nDSCV adapter capabilities:", ", ".join(sorted(caps)))


# --- Visualizations via facade / 通过统一前端生成可视化 --------------------------

_log('search_evolution', render(VizRequest('search_evolution', model, options={'output_dir': output_root})))

# KDE density (requires seaborn); gracefully degrades to warning if missing.
_log('density', render(VizRequest('density', model, options={'output_dir': output_root, 'epoches': [0, 1, 2]})))

# Expression tree (requires graphviz); returns warning until dependency present.
_log('tree', render(VizRequest('tree', model, options={'output_dir': output_root})))

# Equation rendering
_log('equation', render(VizRequest('equation', model, options={'output_dir': output_root})))

# Residual diagnostics
_log('residual', render(VizRequest('residual', model, options={'output_dir': output_root})))

# Field comparison & parity diagnostics
_log('field_comparison', render(VizRequest('field_comparison', model, options={'output_dir': output_root})))
_log('parity', render(VizRequest('parity', model, options={'output_dir': output_root})))

print("\nDone. Check the artifacts directory and console warnings for details.")
