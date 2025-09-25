"""Example showcasing the user-friendly KD visualization helpers with KD_DLGA.

This script mirrors ``examples/kd_dlga_example.py`` but routes visualization
through the high-level helper functions (``kd.viz.plot_*`` and
``kd.viz.render_equation``). Only the intents currently wired into
``DLGAVizAdapter`` are invoked; future integrations can reuse the TODO
placeholders at the bottom.
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
    plot_field_comparison,
    plot_optimization,
    plot_residuals,
    plot_search_evolution,
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
print("\nTraining KD_DLGA model...")
model.fit(X_train, y_train)


# Prepare reusable predictions for diagnostics
print("Generating predictions for diagnostics...")
y_pred_train = model.predict(X_train).reshape(-1)
X_full = kdv_data.mesh()
u_pred_field = model.predict(X_full).reshape(kdv_data.get_size())


# --- Unified viz façade configuration -------------------------------------
output_root = Path("artifacts") / "dlga_viz"
configure(save_dir=output_root)

caps = list_capabilities(model)
print("\nDLGA adapter capabilities:", ", ".join(sorted(caps)))


# --- Visualizations via helper functions ----------------------------------
def _maybe_print_result(name, result):
    if result.warnings:
        print(f"[{name}] warnings: {'; '.join(result.warnings)}")
    else:
        paths = ', '.join(str(path) for path in result.paths)
        print(f"[{name}] saved to: {paths}")


_maybe_print_result('training_curve', plot_training_curve(model))
_maybe_print_result('validation_curve', plot_validation_curve(model))
_maybe_print_result('search_evolution', plot_search_evolution(model))
_maybe_print_result('optimization', plot_optimization(model))
_maybe_print_result('equation', render_equation(model, font_size=14))
_maybe_print_result(
    'residuals',
    plot_residuals(
        model,
        actual=y_train.reshape(-1),
        predicted=y_pred_train,
        coordinates=X_train,
        bins=40,
    ),
)
_maybe_print_result(
    'field_comparison',
    plot_field_comparison(
        model,
        x_coords=kdv_data.x,
        t_coords=kdv_data.t,
        true_field=kdv_data.usol,
        predicted_field=u_pred_field,
    ),
)


# --- Additional analyses (not yet unified) --------------------------------
# TODO: Integrate derivative relationships / parity plots into façade when ready.

# Legacy helper calls retained as guidance for future integration.
# ---------------------------------------------------------------------------
# from kd.viz.dlga_viz import (
#     plot_residual_analysis,
#     plot_optimization_analysis,
#     plot_pde_comparison,
#     plot_time_slices,
#     plot_derivative_relationships,
#     plot_pde_parity,
# )
#
# X_full = kdv_data.mesh()
# u_pred = model.predict(X_full).reshape(kdv_data.get_size())
#
# plot_residual_analysis(model, X_train, y_train, kdv_data.usol, u_pred)
# plot_optimization_analysis(model)
# plot_pde_comparison(kdv_data.x, kdv_data.t, kdv_data.usol, u_pred)
# plot_time_slices(kdv_data.x, kdv_data.t, kdv_data.usol, u_pred, slice_times=[0.25, 0.5, 0.75])
# plot_derivative_relationships(model)
# plot_pde_parity(model, title="Final Validation of Discovered Equation")

print("\nDone. Check the artifacts directory for generated figures.")
