
import kd


dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")


model = kd.Model(
    algorithm="sga",
    generations=30,
    population=15,
    seed=0,
)


model.fit(dataset)


print()
print(f"Discovered: {model.best_expr_}")
print(f"Best AIC: {model.best_score_:.4f}")
