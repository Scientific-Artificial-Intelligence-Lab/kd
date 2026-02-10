"""DSCV multi-dataset benchmark: 7 datasets x 2 configurations.

Runs KD_DSCV on all DSCV-supported datasets with quick and full configs,
prints discovered equation, reward, and ground truth for each combination.

Usage:
    python tests/manual_scripts/dscv_multi_dataset_test.py
"""

import logging
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kd.dataset import load_pde
from kd.dataset._registry import get_dataset_sym_true
from kd.model.kd_dscv import KD_DSCV

logger = logging.getLogger(__name__)

DATASETS = [
    "chafee-infante",
    "burgers",
    "kdv",
    "PDE_divide",
    "PDE_compound",
    "fisher",
    "fisher_linear",
]

CONFIGS: dict[str, dict[str, int]] = {
    "quick": {"n_iterations": 5, "n_samples_per_batch": 500, "n_epochs": 5},
    "full": {"n_iterations": 20, "n_samples_per_batch": 2000, "n_epochs": 20},
}

SEED = 0


def run_one(dataset_name: str, config_name: str, config: dict[str, int]) -> dict:
    """Run DSCV on one dataset with the given configuration.

    Returns a dict with dataset, config, reward, expression, sym_true, and
    elapsed time.  If training raises, captures the error message instead.
    """
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}  |  Config: {config_name}")
    print(f"{'=' * 60}")

    sym_true = get_dataset_sym_true(dataset_name)
    start = time.time()

    try:
        dataset = load_pde(dataset_name)
        model = KD_DSCV(
            binary_operators=["add", "mul", "diff"],
            unary_operators=["n2"],
            n_iterations=config["n_iterations"],
            n_samples_per_batch=config["n_samples_per_batch"],
            seed=SEED,
        )
        model.import_dataset(dataset)
        result = model.train(n_epochs=config["n_epochs"], verbose=False)

        expression = result.get("expression", "N/A")
        reward = result.get("r", float("nan"))
        elapsed = time.time() - start

        print(f"  Reward:     {reward:.6f}")
        print(f"  Expression: {expression}")
        print(f"  Sym true:   {sym_true}")
        print(f"  Time:       {elapsed:.1f}s")

        return {
            "dataset": dataset_name,
            "config": config_name,
            "reward": reward,
            "expression": str(expression),
            "sym_true": sym_true or "N/A",
            "elapsed": elapsed,
            "error": None,
        }

    except Exception:
        elapsed = time.time() - start
        error_msg = traceback.format_exc()
        logger.error("Failed %s/%s:\n%s", dataset_name, config_name, error_msg)
        print(f"  ERROR: {error_msg.splitlines()[-1]}")
        print(f"  Time:  {elapsed:.1f}s")

        return {
            "dataset": dataset_name,
            "config": config_name,
            "reward": float("nan"),
            "expression": "FAILED",
            "sym_true": sym_true or "N/A",
            "elapsed": elapsed,
            "error": error_msg.splitlines()[-1],
        }


def print_summary(results: list[dict]) -> None:
    """Print a summary table of all benchmark results."""
    print(f"\n\n{'=' * 100}")
    print("  DSCV BENCHMARK SUMMARY")
    print(f"{'=' * 100}")
    header = (
        f"{'Dataset':<20} {'Config':<8} {'Reward':>10} {'Time':>7}  "
        f"{'Expression'}"
    )
    print(header)
    print(f"{'-' * 20} {'-' * 8} {'-' * 10} {'-' * 7}  {'-' * 50}")

    for r in results:
        reward_str = f"{r['reward']:.4f}" if r["error"] is None else "ERROR"
        print(
            f"{r['dataset']:<20} {r['config']:<8} {reward_str:>10} "
            f"{r['elapsed']:>6.1f}s  {r['expression']}"
        )

    # Ground truth reference
    print(f"\n{'=' * 100}")
    print("  GROUND TRUTH REFERENCE")
    print(f"{'=' * 100}")
    seen: set[str] = set()
    for r in results:
        if r["dataset"] not in seen:
            seen.add(r["dataset"])
            print(f"  {r['dataset']:<20} {r['sym_true']}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results: list[dict] = []
    for ds in DATASETS:
        for config_name, config in CONFIGS.items():
            result = run_one(ds, config_name, config)
            results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
