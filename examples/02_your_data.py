
import math

import torch

import kd







nx, nt = 64, 32
x = torch.linspace(0.0, 2 * math.pi, nx + 1, dtype=torch.float64)[:-1]
t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
xx, tt = torch.meshgrid(x, t, indexing="ij")
alpha = 0.1
u = torch.exp(-alpha * tt) * torch.sin(xx) + torch.exp(-4 * alpha * tt) * torch.sin(
    2 * xx
)






dataset = kd.PDEDataset.from_arrays(
    coords={"x": x, "t": t},
    fields={"u": u},
    lhs="u_t",
    periodic={"x"},
    name="my_heat_equation",
    ground_truth=f"u_t = {alpha} * u_xx",
)



kd.preview(dataset)


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
