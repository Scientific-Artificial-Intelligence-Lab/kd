"""Generate high-quality Gallery images for README.

Use reference-implementation-aligned parameters for best results.
Output: artifacts/gallery/{sga,dscv}/
"""

from kd.dataset import load_pde
from kd.model.kd_sga import KD_SGA
from kd.model.kd_dscv import KD_DSCV
from kd.viz import (
    configure,
    render_equation,
    plot_field_comparison,
    plot_parity,
)

dataset_sga = load_pde("burgers")
dataset_dscv = load_pde("burgers")

# ===== SGA on Burgers =====

print("=" * 60)
print("[Gallery] SGA on Burgers")
print("=" * 60)

configure(save_dir="artifacts/gallery/sga")

sga = KD_SGA(
    sga_run=20, num=20, depth=4, width=5,
    p_var=0.5, p_mute=0.3, p_cro=0.5, seed=0,
)
sga.fit_dataset(dataset_sga)

print(f"[SGA] Best PDE: {sga.best_pde_}")
print(f"[SGA] LaTeX:    {sga.equation_latex()}")

render_equation(sga)
plot_field_comparison(
    sga, x_coords=None, t_coords=None,
    true_field=None, predicted_field=None,
)
plot_parity(sga, title="Parity Plot")

# ===== DSCV on Burgers =====

print()
print("=" * 60)
print("[Gallery] DSCV on Burgers")
print("=" * 60)

configure(save_dir="artifacts/gallery/dscv")

dscv = KD_DSCV(
    binary_operators=["add", "mul", "div", "diff", "diff2", "diff3"],
    unary_operators=["n2", "n3"],
    n_samples_per_batch=500,
    n_iterations=100,
    seed=0,
)
dscv.fit_dataset(dataset_dscv, n_epochs=100)

render_equation(dscv)
plot_field_comparison(
    dscv, x_coords=dataset_dscv.x, t_coords=dataset_dscv.t,
    true_field=None, predicted_field=None,
)
plot_parity(dscv, title="Parity Plot")
