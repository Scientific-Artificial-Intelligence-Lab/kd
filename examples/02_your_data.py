"""Example 02 - Bring your own data.

You have scientific data (numpy/torch arrays from a simulation or
experiment). How do you wrap it into a ``PDEDataset`` so kd can
discover an equation from it?

This example walks through the heat equation: u_t = 0.1 * u_xx
on a periodic 1D domain. The same recipe works for any field on a
regular grid - change the arrays, keep the structure.

For other file formats (.npy, .npz, .csv, .mat, .h5), load them
with the matching numpy/scipy reader and follow the same recipe.

Run: python examples/02_your_data.py
"""

import math

import torch

import kd

# 1. Build / load your data into 1D coordinate arrays + an nD field array.
# Heat-equation solution with two Fourier modes:
# u(x,t) = sin(x)*exp(-alpha*t) + sin(2x)*exp(-4*alpha*t)
# Two modes break the sin(x)-only degeneracy (where u_t = -alpha*u and
# u_t = alpha*u_xx are equivalent on the data) so SGA must pick the real
# diffusion form u_t = alpha * u_xx.
nx, nt = 64, 32
x = torch.linspace(0.0, 2 * math.pi, nx + 1, dtype=torch.float64)[:-1] # periodic
t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
xx, tt = torch.meshgrid(x, t, indexing="ij") # shape (nx, nt)
alpha = 0.1
u = torch.exp(-alpha * tt) * torch.sin(xx) + torch.exp(-4 * alpha * tt) * torch.sin(
    2 * xx
)

# 2. Wrap arrays into a PDEDataset in ONE call.
# - coords: insertion order defines axis_order
# - fields: tensor shape must match (len(x), len(t)) here
# - lhs: "{field}_{axis}" - the LHS of the equation kd will discover
# - periodic: axes with periodic boundary (optional but improves fits)
dataset = kd.PDEDataset.from_arrays(
    coords={"x": x, "t": t},
    fields={"u": u},
    lhs="u_t",
    periodic={"x"},
    name="my_heat_equation",
    ground_truth=f"u_t = {alpha} * u_xx",
)

# 3. Sanity-check before you fit. Catches grid-spacing problems, NaN/Inf,
# too-small grids, dtype mismatches.
kd.preview(dataset)

# 4. Fit. SGA searches for an equation that explains u_t in terms of u, u_x, u_xx, ...
model = kd.Model(
    algorithm="sga",
    generations=20,
    population=12,
    seed=0,
    verbose=False,
)
model.fit(dataset)

print()
print(f"Discovered: {model.best_expr_}")
print(f"Best AIC: {model.best_score_:.4f}")
print(f"Ground truth: {dataset.ground_truth}")
