"""Example 01 - Quickstart: discover the Burgers equation in ~30 lines.

This generates synthetic Burgers data so you can see kd work end-to-end
in 30 seconds. For your own scientific data, see ``02_your_data.py``.

Burgers: u_t + u * u_x - 0.1 * u_xx = 0 (i.e., u_t = -u * u_x + 0.1 * u_xx)

Run: python examples/01_quickstart.py
"""

import kd

# 1. Generate synthetic data with a known ground truth.
dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")

# 2. Configure the model. Defaults are reasonable for a quick demo.
model = kd.Model(
    algorithm="sga",
    generations=30,
    population=15,
    seed=0,
)

# 3. Fit. Progress prints to stdout because verbose=True (default).
model.fit(dataset)

# 4. Inspect the result.
print()
print(f"Discovered: {model.best_expr_}")
print(f"Best AIC: {model.best_score_:.4f}")
