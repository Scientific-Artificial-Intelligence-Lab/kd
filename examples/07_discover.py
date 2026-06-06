
from pathlib import Path

import kd


dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")





model = kd.Model(
    algorithm="discover",
    generations=20,
    seed=0,
)


model.fit(dataset)


print()
print(f"Discovered: {model.best_expr_}")
print(f"Best reward: {model.best_score_:.4f}")



out_dir = Path(__file__).parent / "out" / "07_discover"
report = kd.VizEngine(output_dir=out_dir).render_all(
    model.result_, algorithm=model.algorithm_, dataset=dataset
)
print(f"Report: {report.report}")
