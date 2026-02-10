"""SGA multi-dataset benchmark: 3 datasets × 2 generation counts.

Runs KD_SGA on chafee-infante, burgers, kdv with sga_run=30 and sga_run=100,
prints AIC and discovered equation for each combination.

Usage:
    python tests/manual_scripts/sga_multi_dataset_test.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kd.dataset import load_pde
from kd.model.kd_sga import KD_SGA


DATASETS = ["chafee-infante", "burgers", "kdv"]
GENERATIONS = [30, 100]
SEED = 0


def run_one(dataset_name: str, sga_run: int) -> dict:
    """Run SGA on one dataset with given generation count."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}  |  Generations: {sga_run}")
    print(f"{'='*60}")

    dataset = load_pde(dataset_name)
    model = KD_SGA(
        sga_run=sga_run,
        num=20,
        depth=4,
        width=5,
        p_var=0.5,
        p_mute=0.3,
        p_cro=0.5,
        seed=SEED,
    )
    model.fit_dataset(dataset)

    aic = model.best_score_
    eq_raw = model.best_pde_
    try:
        eq_latex = model.equation_latex(include_coefficients=False)
    except Exception:
        eq_latex = "(render failed)"

    print(f"  AIC:      {aic:.4f}")
    print(f"  Equation: {eq_raw}")
    print(f"  LaTeX:    {eq_latex}")

    return {
        "dataset": dataset_name,
        "sga_run": sga_run,
        "aic": aic,
        "equation": eq_raw,
        "latex": eq_latex,
    }


def main():
    results = []
    for ds in DATASETS:
        for gen in GENERATIONS:
            result = run_one(ds, gen)
            results.append(result)

    # Summary table
    print(f"\n\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'Gens':>5} {'AIC':>12}  {'Equation'}")
    print(f"{'-'*20} {'-'*5} {'-'*12}  {'-'*40}")
    for r in results:
        print(f"{r['dataset']:<20} {r['sga_run']:>5} {r['aic']:>12.4f}  {r['equation']}")


if __name__ == "__main__":
    main()
