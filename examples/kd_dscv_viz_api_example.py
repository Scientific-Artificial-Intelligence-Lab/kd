"""Unified visualization façade usage for KD_DSCV.

This mirrors ``examples/kd_dscv_example.py`` but routes plotting through
``kd.viz.render`` so the new adapter can be exercised incrementally.
"""

import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

import numpy as np
import warnings

from kd.dataset import load_pde
from kd.model import KD_DSCV
from kd.viz import VizRequest, configure, list_capabilities, render

warnings.filterwarnings("ignore", category=FutureWarning, module='numpy.*')


def _log(intent: str, result) -> None:
    if result.warnings:
        print(f"[viz_api:{intent}] warning -> {'; '.join(result.warnings)}")
    else:
        saved = ', '.join(str(p) for p in result.paths) or '<no files>'
        print(f"[viz_api:{intent}] saved -> {saved}")


# --- Data loading & model setup -------------------------------------------------
pde_dataset = load_pde('burgers')
model = KD_DSCV(
    binary_operators=["add", "mul", "diff"],
    unary_operators=['n2'],
    n_samples_per_batch=500,
)

np.random.seed(42)

# 推荐入口：fit_dataset，可以一步完成导入 + 训练；
# 这里保持与旧示例一致，显式展示 import_dataset + train 的组合。
model.import_dataset(pde_dataset)
step_output = model.train(n_epochs=11, verbose=False)
print(f"Current best expression is {step_output['expression']} and its reward is {step_output['r']}")


# --- Unified façade configuration ----------------------------------------------
output_root = Path('artifacts') / 'dscv_viz'
configure(save_dir=output_root)

caps = list_capabilities(model)
print("\nDSCV adapter capabilities:", ", ".join(sorted(caps)))


# --- Visualizations via façade --------------------------------------------------
_log('search_evolution', render(VizRequest('search_evolution', model, options={'output_dir': output_root})))

# KDE density (requires seaborn); gracefully degrades to warning if missing.
_log('density', render(VizRequest('density', model, options={'output_dir': output_root, 'epoches': [0, 1, 2]})))

# Expression tree (requires graphviz); currently returns warning until dependency present.
_log('tree', render(VizRequest('tree', model, options={'output_dir': output_root})))

# Equation rendering (may warn until program metadata is complete)
_log('equation', render(VizRequest('equation', model, options={'output_dir': output_root})))

# Residual diagnostics (requires PDE field helpers to succeed)
_log('residual', render(VizRequest('residual', model, options={'output_dir': output_root})))

# Field comparison & parity diagnostics (may warn until full data plumbed)
_log('field_comparison', render(VizRequest('field_comparison', model, options={'output_dir': output_root})))
_log('parity', render(VizRequest('parity', model, options={'output_dir': output_root})))

print("\nDone. Check the artifacts directory and console warnings for details.")
