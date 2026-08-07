"""Example 08 - DLGA: gene-expression GA + neural-network surrogate for Burgers.

DLGA (Xu et al. 2020) discovers PDEs in two stages: it first trains a neural
network surrogate (NN_1) to denoise the field and supply smooth derivatives,
then evolves gene-encoded operator combinations with a genetic algorithm,
scoring each candidate by a GA fitness (NMSE + epsilon*length; lower is
better — NOT an AIC). The kd facade plugs it into the same one-line API as
SGA and DISCOVER.

Burgers: u_t + u * u_x - 0.1 * u_xx = 0 (i.e., u_t = -u * u_x + 0.1 * u_xx)

NOTE: unlike SGA's finite-difference path, DLGA trains an NN surrogate first,
so it is the slowest of the three quickstarts (expect a couple of minutes on
CPU). ``surrogate_max_epochs`` below is deliberately small for a fast demo —
Xu's reference trains the surrogate to ~1e-7 for accurate recovery; raise it
and ``generations`` for a serious attempt.

Run: python examples/08_dlga.py
"""

from pathlib import Path

import kd
from kd import DLGAConfig

# 1. Generate synthetic data with a known ground truth.
dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")

# 2. Configure the model. ``DLGAConfig.burgers_preset()`` locks the per-PDE
# epsilon validated by the recovery suite (Burgers uses the 1e-3 default);
# ``**overrides`` customize on top — here a small surrogate budget +
# population to keep the demo fast (raise them for real recovery). DLGA also
# ships kdv_preset() / wave_preset() / kg_preset() / chafee_preset() for
# those PDEs.
model = kd.Model(
    algorithm="dlga",
    generations=20,
    config=DLGAConfig.burgers_preset(
        pop_size=80,
        seed=0,
        surrogate_max_epochs=2000, # paper uses 50000; small = fast demo
    ),
)

# 3. Fit. Trains the NN_1 surrogate, then runs the GA. Progress prints to stdout.
model.fit(dataset)

# 4. Inspect the result.
print()
print(f"Discovered: {model.best_expr_}")
print(f"Best fitness: {model.best_score_:.4f}")

# 5. Visualize. ``render_all`` includes DLGA's 4 VizExtension diagnostics
# (fitness_spread / population_diversity / complexity_evolution /
# surrogate_training — the NN_1 train/val loss curve) automatically,
# alongside the universal report figures.
out_dir = Path(__file__).parent / "out" / "08_dlga"
report = kd.VizEngine(output_dir=out_dir).render_all(
    model.result_, algorithm=model.algorithm_, dataset=dataset
)
print(f"Report: {report.report}")
