"""Unified visualization façade usage for KD_DSCV_SPR."""

from pathlib import Path
import numpy as np
import warnings

from _bootstrap import ensure_project_root_on_syspath

PROJECT_ROOT = ensure_project_root_on_syspath()

from kd.dataset import load_pde
from kd.model import KD_DSCV_SPR
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
model = KD_DSCV_SPR(
    n_samples_per_batch=100,
    binary_operators=["add_t", "mul_t", "div_t", "diff_t", "diff2_t"],
    unary_operators=['n2_t'],
)

np.random.seed(42)
model.import_dataset(
    pde_dataset,
    sample_ratio=0.1,
    colloc_num=512,
    random_state=42,
)

step_output = model.train(n_epochs=100, verbose=False)
print(f"Current best expression is {step_output['expression']} and its reward is {step_output['r']}")


# --- Unified façade configuration ----------------------------------------------
output_root = Path('artifacts') / 'dscvspr_viz'
configure(save_dir=output_root)

caps = list_capabilities(model)
print("\nDSCV_SPR adapter capabilities:", ", ".join(sorted(caps)))


# --- Visualizations via façade --------------------------------------------------
_log('search_evolution', render(VizRequest('search_evolution', model, options={'output_dir': output_root})))
_log('density', render(VizRequest('density', model, options={'output_dir': output_root, 'epoches': [0, 1, 2]})))
_log('tree', render(VizRequest('tree', model, options={'output_dir': output_root})))

# Equation rendering (shares façade with KD_DSCV); residual/SPR-specific
# intents remain TODO until the adapter supports them fully.
_log('equation', render(VizRequest('equation', model, options={'output_dir': output_root})))

# SPR diagnostics (currently rely on PINN helper scaffolding; may emit warnings)
_log('spr_residual', render(VizRequest('spr_residual', model, options={'output_dir': output_root})))
_log('spr_field_comparison', render(VizRequest('spr_field_comparison', model, options={'output_dir': output_root})))

# TODO: Once SPR-specific intents (`spr_residual`, `spr_field_comparison`) are implemented,
#       invoke them here to complete the façade migration.

print("\nDone. Check the artifacts directory and console warnings for details.")
