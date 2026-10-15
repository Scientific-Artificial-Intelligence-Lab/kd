"""Example 04 - Noisy real-world data: finite-difference vs autograd.

When data is noisy (e.g. lab measurements), naive finite-difference
derivatives blow up the noise: on this 5% noisy field u_x and u_xx are
dominated by it, and the search settles on a term with no derivative in it
at all. Autograd-based derivatives fit a neural-network surrogate to the
samples and differentiate that instead, which is why the true law is
recoverable here.

This example takes clean Burgers data, adds 5% Gaussian noise, then runs
the same SGA search twice: ``derivatives='finite_diff'`` vs
``derivatives='autograd'``.

What it prints at the budget below (seed 0, measured 2026-09-18):

  finite_diff u_t = -1.102*u (-1.103*u under single-threaded BLAS)
  autograd u_t = -0.9355*u*u_x + 0.09737*u_x_x

against the ground truth ``u_t = -u*u_x + 0.1*u_xx``.

Two measurements behind that budget, both on this dataset:

  - The surrogate has to be trained far enough. At
    ``autograd_train_epochs=500`` the autograd run returns
    ``-1.717*x/(-u**2 + x/u)``, built from ``u`` and the coordinate ``x`` with
    no derivative in it, so both modes are wrong; 3000 is where the true law
    wins.
  - Recovery is still seed-dependent: seeds 1 and 2 return ``-1.118*u`` and
    ``-1.1*u``, and neither 5000 epochs nor a 60x30 search recovers the true
    law on them (at 60x30 they return different wrong expressions, one of
    them a five-term bloat). A single run on noisy data does not settle
    whether a law is recoverable.

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
print(f"Discovered: {fd.result_.equation}")
print(f"Best AIC: {fd.best_score_:.4f}")

print("\n--- Autograd (NN surrogate) derivatives ---")
# ``autograd_train_epochs`` is an honest fixed-step budget: the surrogate trains
# exactly this many epochs. 3000 is what this 5% noisy field needs (the module
# docstring records the 500-epoch result); noisier or higher-resolution data
# wants more. To stop automatically once the surrogate plateaus, add
# ``autograd_train_patience=...`` + ``autograd_train_val_ratio=...``
# (early stopping needs a validation split).
ag = kd.Model(
    algorithm="sga",
    generations=40,
    population=15,
    derivatives="autograd",
    autograd_train_epochs=3000,
    seed=0,
    verbose=False,
).fit(dataset)
print(f"Discovered: {ag.result_.equation}")
print(f"Best AIC: {ag.best_score_:.4f}")

print(f"\nGround truth: {dataset.ground_truth}")
