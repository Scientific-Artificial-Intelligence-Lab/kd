
from pathlib import Path

import kd
from kd.search.dlga import DLGAConfig


dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
print(f"Ground truth: {dataset.ground_truth}")






model = kd.Model(
    algorithm="dlga",
    generations=20,
    config=DLGAConfig.burgers_preset(
        pop_size=80,
        seed=0,
        surrogate_max_epochs=2000,
    ),
)


model.fit(dataset)


print()
print(f"Discovered: {model.best_expr_}")
print(f"Best fitness: {model.best_score_:.4f}")





out_dir = Path(__file__).parent / "out" / "08_dlga"
report = kd.VizEngine(output_dir=out_dir).render_all(
    model.result_, algorithm=model.algorithm_, dataset=dataset
)
print(f"Report: {report.report}")
