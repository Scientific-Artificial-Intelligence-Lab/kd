"""Generate high-quality Gallery images for README.

Use reference-implementation-aligned parameters for best results.
Output: artifacts/gallery/{sga,discover}/
"""

from kd.dataset import load_pde
from kd.model.kd_sga import KD_SGA
from kd.model.kd_discover import KD_Discover
from kd.viz import (
    configure,
    render_equation,
    plot_field_comparison,
    plot_parity,
)

dataset_sga = load_pde("burgers")
dataset_discover = load_pde("burgers")

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

# ===== Discover on Burgers =====

print()
print("=" * 60)
print("[Gallery] Discover on Burgers")
print("=" * 60)

configure(save_dir="artifacts/gallery/discover")

discover = KD_Discover(
    binary_operators=["add", "mul", "div", "diff", "diff2", "diff3"],
    unary_operators=["n2", "n3"],
    n_samples_per_batch=500,
    n_iterations=100,
    seed=0,
)
discover.fit_dataset(dataset_discover, n_epochs=100)

render_equation(discover)
plot_field_comparison(
    discover, x_coords=dataset_discover.x, t_coords=dataset_discover.t,
    true_field=None, predicted_field=None,
)
plot_parity(discover, title="Parity Plot")
