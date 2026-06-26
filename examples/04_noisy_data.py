
import torch

import kd


torch.manual_seed(0)
clean = kd.generate_burgers_data(nx=128, nt=64, nu=0.1, seed=0)


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
