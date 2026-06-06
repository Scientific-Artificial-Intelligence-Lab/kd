
from pathlib import Path

import kd


dataset = kd.load_pde_divide()
print(f"Loaded: {dataset.name}")
print(f"Ground truth: {dataset.ground_truth}")

model = kd.Model(
    algorithm="sga",
    generations=30,
    population=15,
    seed=0,
    verbose=False,
).fit(dataset)
print(f"Discovered: {model.best_expr_}")
print(f"Best AIC: {model.best_score_:.4f}")


out_path = Path(__file__).parent / "out" / "pde_divide.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
model.result_.save(out_path)
print(f"Saved: {out_path}")


restored = kd.ExperimentResult.load(out_path)
print(f"Loaded back: best={restored.best_expression!r} AIC={restored.best_score:.4f}")

assert restored.best_expression == model.best_expr_, "Round-trip failed!"
print("Round-trip OK.")
