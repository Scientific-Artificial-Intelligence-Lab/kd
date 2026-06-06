
from pathlib import Path

import kd

OUT_BASE = Path(__file__).parent / "out" / "realworld"


DATASETS = [
    kd.load_chafee_infante,
    kd.load_kdv,
    kd.load_pde_divide,
    kd.load_pde_compound,
]




GENERATIONS = 80
POPULATION = 20
DEPTH = 4
WIDTH = 5
AUTOGRAD_TRAIN_EPOCHS = 15_000

results: list[
    tuple[str, str, str, str, float]
] = []

for loader in DATASETS:
    dataset = loader()
    print()
    print(f"=== {dataset.name} ===")
    print(f"Ground truth: {dataset.ground_truth}")
    kd.preview(dataset)

    for mode_label, derivatives in [("fd", "finite_diff"), ("nn", "autograd")]:
        print(f"\n--- {dataset.name} / {mode_label} ---")
        model = kd.Model(
            algorithm="sga",
            generations=GENERATIONS,
            population=POPULATION,
            depth=DEPTH,
            width=WIDTH,
            derivatives=derivatives,
            autograd_train_epochs=AUTOGRAD_TRAIN_EPOCHS,
            seed=0,
            verbose=False,
        )
        model.fit(dataset)

        print(f"Discovered: {model.best_expr_}")
        print(f"Best AIC: {model.best_score_:.4f}")

        out_dir = OUT_BASE / dataset.name.replace("-", "_") / mode_label
        viz = kd.VizEngine(output_dir=out_dir)
        report = viz.render_all(
            model.result_,
            algorithm=model.algorithm_,
            dataset=dataset,
        )
        print(f"Report: {report.report}")
        print(f"Figures: {len(report.figures)} files")

        results.append(
            (
                dataset.name,
                mode_label,
                dataset.ground_truth or "",
                model.best_expr_,
                model.best_score_,
            )
        )


print()
print("=" * 80)
print("SUMMARY (lower AIC = better fit)")
print("=" * 80)
current = ""
for name, mode, truth, found, aic in results:
    if name != current:
        print()
        print(name)
        print(f" Ground truth: {truth}")
        current = name
    print(f" [{mode:>2}] AIC={aic:>9.4f} | {found}")
print()
print(f"All reports saved under: {OUT_BASE}")
