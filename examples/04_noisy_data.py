"""Example 04 - Noisy real-world data: finite-difference vs autograd.

When data is noisy (e.g. lab measurements), naive finite-difference
derivatives blow up the noise: u_t and u_xx end up dominated by it.
Autograd-based derivatives use a neural-network surrogate that smooths
the field first - typically much more robust.

This example takes clean Burgers data, adds 5% Gaussian noise, then
compares ``derivatives='finite_diff'`` vs ``derivatives='autograd'``.

Rule of thumb:
  - noise < 1% -> finite_diff is fine
  - noise 1-5% -> either, autograd if FD struggles
  - noise > 5% -> use autograd

Run: python examples/04_noisy_data.py
"""

import torch

import kd

# 1. Generate clean Burgers data and add 5% noise.
torch.manual_seed(0)
clean = kd.generate_burgers_data(nx=128, nt=64, nu=0.1, seed=0)

# Re-wrap with noisy field (kd.preview will warn if anything goes wrong).
clean_u = clean.fields["u"].values
noise_scale = 0.05 * clean_u.abs().max()
noisy_u = clean_u + noise_scale * torch.randn_like(clean_u)

dataset = kd.PDEDataset.from_arrays(
    coords={"x": clean.axes["x"].values, "t": clean.axes["t"].values},
    fields={"u": noisy_u},
    lhs="u_t",
    periodic={"x"},
    name="burgers_5pct_noise",
    ground_truth=clean.ground_truth,
)
kd.preview(dataset)

# 2. Fit twice - once with FD, once with autograd.
print("\n--- Finite-difference derivatives ---")
fd = kd.Model(
    algorithm="sga",
    generations=40,
    population=15,
    derivatives="finite_diff",
    seed=0,
    verbose=False,
).fit(dataset)
print(f"Discovered: {fd.best_expr_}")
print(f"Best AIC: {fd.best_score_:.4f}")

print("\n--- Autograd (NN surrogate) derivatives ---")
# ``autograd_train_epochs`` is an honest fixed-step budget: the surrogate trains
# exactly this many epochs. 500 is enough for this small demo; noisier or
# higher-resolution data wants more. To stop automatically once the surrogate
# plateaus, add ``autograd_train_patience=...`` + ``autograd_train_val_ratio=...``
# (early stopping needs a validation split).
ag = kd.Model(
    algorithm="sga",
    generations=40,
    population=15,
    derivatives="autograd",
    autograd_train_epochs=500,
    seed=0,
    verbose=False,
).fit(dataset)
print(f"Discovered: {ag.best_expr_}")
print(f"Best AIC: {ag.best_score_:.4f}")

print(f"\nGround truth: {dataset.ground_truth}")
