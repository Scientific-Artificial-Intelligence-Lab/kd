"""Example 07 - DISCOVER: LSTM-controller symbolic regression for Burgers.

DISCOVER (Du et al. 2024) is a reinforcement-learning-based symbolic
regression algorithm: an LSTM controller proposes candidate equations,
and a risk-seeking policy gradient (RSPG) trains the controller toward
high-reward expressions. The kd facade plugs it into the same one-line
API as SGA and DLGA.

Burgers: u_t + u * u_x - 0.1 * u_xx = 0 (i.e., u_t = -u * u_x + 0.1 * u_xx)

DISCOVER converges more slowly than SGA on short runs; for a serious
attempt at recovering Burgers, bump ``generations`` to 200+ (the value
below is enough to see the search work but not enough to converge).

Run: python examples/07_discover.py
"""

from pathlib import Path

import kd

# 1. Generate synthetic data with a known ground truth.
dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")

# 2. Configure the model. ``algorithm="discover"`` selects the LSTM-controller
# plugin; the facade exposes ``seed`` and ``generations`` directly. For
# other hyperparameters (operators, batch_size, entropy weight, etc.)
# pass a full ``kd.DiscoverConfig`` via ``config=``.
model = kd.Model(
    algorithm="discover",
    generations=20,
    seed=0,
)

# 3. Fit. Progress prints to stdout because verbose=True (default).
model.fit(dataset)

# 4. Inspect the result.
print()
print(f"Discovered: {model.best_expr_}")
print(f"Best reward: {model.best_score_:.4f}")

# 5. Visualize (universal + DISCOVER reward/entropy_loss/baseline plots)

out_dir = Path(__file__).parent / "out" / "07_discover"
report = kd.VizEngine(output_dir=out_dir).render_all(
    model.result_, algorithm=model.algorithm_, dataset=dataset
)
print(f"Report: {report.report}")
