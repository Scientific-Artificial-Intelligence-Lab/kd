"""Example 04 - 2D diffusion equation.

Discover u_t = alpha * (u_xx + u_yy) from synthetic 2D data.

The generator picks the number of spatial dimensions from len(waves):
two-element tuples here mean a 2D problem.

Run: python examples/internals/04_2d_diffusion.py
"""

import kd

# 1. 2D diffusion data: alpha=0.1, two spatial axes (x, y) + time.
dataset = kd.generate_diffusion_data(
    alpha=0.1,
    waves=(1.0, 1.0),
    grid_sizes=(32, 32),
    nt=16,
    seed=0,
)
print(f"Ground truth: {dataset.ground_truth}")
print(f"Shape: {dataset.get_shape()}")

# 2. Fit. 2D problems have a richer term library (u_xx, u_yy, u_xy, ...),
# so we bump the SGA budget above the defaults: more generations, a
# larger population, and slightly deeper / wider expressions.
model = kd.Model(
    algorithm="sga",
    generations=50,
    population=25,
    depth=5,
    width=6,
    seed=0,
    verbose=False,
)
model.fit(dataset)

print()
print(f"Discovered: {model.best_expr_}")
print(f"Best AIC: {model.best_score_:.4f}")
