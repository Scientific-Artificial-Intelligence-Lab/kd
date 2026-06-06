
import kd


dataset = kd.generate_diffusion_data(
    alpha=0.1,
    waves=(1.0, 1.0),
    grid_sizes=(32, 32),
    nt=16,
    seed=0,
)
print(f"Ground truth: {dataset.ground_truth}")
print(f"Shape: {dataset.get_shape()}")




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
