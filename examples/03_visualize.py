
from pathlib import Path

import kd


dataset = kd.load_chafee_infante()
print(f"Loaded: {dataset.name}, ground truth: {dataset.ground_truth}")
kd.preview(dataset)



model = kd.Model(
    algorithm="sga",
    generations=60,
    population=20,
    seed=0,
    verbose=False,
)
model.fit(dataset)

print()
print(f"Discovered: {model.best_expr_}")
print(f"Best AIC: {model.best_score_:.4f}")




out_dir = Path(__file__).parent / "out" / "chafee_infante"
viz = kd.VizEngine(output_dir=out_dir)
report = viz.render_all(
    model.result_,
    algorithm=model.algorithm_,
    dataset=dataset,
)

print(f"Report: {report.report}")
print(f"Figures: {len(report.figures)} files")
