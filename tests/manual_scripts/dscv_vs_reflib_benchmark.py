"""DSCV vs ref_lib DISCOVER: 5-dataset side-by-side benchmark.

Runs kd DSCV and ref_lib DISCOVER on the 5 shared datasets with aligned
parameters, then prints a comparison table of reward, time, expression, and
number of terms.

Both sides use the ref_lib default config (MODE1):
  - n_samples=50000, batch_size=500  → 100 epochs
  - function_set: [add, mul, div, diff, diff2, diff3, n2, n3]
  - seed=0

Environment: Use the ref-discover conda env (has both kd and TF 2.x):
    conda activate ref-discover

Usage:
    # Run BOTH sides (default):
    python tests/manual_scripts/dscv_vs_reflib_benchmark.py

    # Run only one side:
    python tests/manual_scripts/dscv_vs_reflib_benchmark.py --side kd
    python tests/manual_scripts/dscv_vs_reflib_benchmark.py --side reflib

    # Quick mode (fewer iterations for sanity check):
    python tests/manual_scripts/dscv_vs_reflib_benchmark.py --quick

    # Subset of datasets:
    python tests/manual_scripts/dscv_vs_reflib_benchmark.py --datasets kdv burgers
"""

import argparse
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFLIB_DSO_DIR = PROJECT_ROOT / "ref_lib" / "DISCOVER" / "dso"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Dataset mapping: kd name → ref_lib config suffix
# ---------------------------------------------------------------------------
DATASETS = [
    # (kd_name, reflib_config_suffix, reflib_dataset_name, ground_truth)
    ("burgers", "Burgers", "Burgers",
     "ut = -u*ux + v*uxx"),
    ("kdv", "KdV", "Kdv",
     "ut = -u*ux + v*uxxx"),
    ("chafee-infante", "Chafee", "chafee-infante",
     "ut = u + u^3 + uxx"),
    ("PDE_divide", "Divide", "PDE_divide",
     "ut = -ux/x + v*uxx"),
    ("PDE_compound", "Compound", "PDE_compound",
     "ut = u*uxx + ux^2"),
]

SEED = 0

# Aligned parameters (matching ref_lib MODE1 default)
FULL_CONFIG = {
    "n_samples": 50000,
    "batch_size": 500,
}

QUICK_CONFIG = {
    "n_samples": 5000,
    "batch_size": 500,
}

# MODE1 config overrides for kd side.
# ref_lib MODE1 configs override config_common.json with these values.
# kd's base config_pde.json differs from ref_lib MODE1 in several parameters;
# this dict aligns them so the only remaining difference is RNG (TF vs PyTorch).
MODE1_KD_OVERRIDES = {
    "controller": {
        "max_length": 256,
        "attention": True,
        "pqt_use_pg": True,
    },
    "prior": {
        "length": {"max_": 256},
    },
    "training": {
        "early_stopping": False,
        "hof": 100,
    },
}


# ---------------------------------------------------------------------------
# kd side
# ---------------------------------------------------------------------------
def run_kd(dataset_name: str, config: dict) -> dict:
    """Run kd DSCV on one dataset.

    Returns dict with keys: reward, expression, elapsed, n_terms, error.
    """
    from kd.dataset import load_pde
    from kd.model.kd_dscv import KD_DSCV

    n_epochs = config["n_samples"] // config["batch_size"]
    start = time.time()
    try:
        dataset = load_pde(dataset_name)
        model = KD_DSCV(
            binary_operators=["add", "mul", "div", "diff", "diff2", "diff3"],
            unary_operators=["n2", "n3"],
            n_iterations=n_epochs,
            n_samples_per_batch=config["batch_size"],
            seed=SEED,
            config_out=MODE1_KD_OVERRIDES,
        )
        model.import_dataset(dataset)
        result = model.train(n_epochs=n_epochs, verbose=True)

        expression = str(result.get("expression", "N/A"))
        reward = result.get("r", float("nan"))
        # Count terms by number of '+' in the expression + 1
        n_terms = expression.count("+") + 1 if expression != "N/A" else 0
        elapsed = time.time() - start

        return {
            "reward": reward,
            "expression": expression,
            "elapsed": elapsed,
            "n_terms": n_terms,
            "error": None,
        }
    except Exception:
        elapsed = time.time() - start
        tb = traceback.format_exc()
        logger.error("kd %s failed:\n%s", dataset_name, tb)
        return {
            "reward": float("nan"),
            "expression": "FAILED",
            "elapsed": elapsed,
            "n_terms": 0,
            "error": tb.splitlines()[-1],
        }


# ---------------------------------------------------------------------------
# ref_lib side
# ---------------------------------------------------------------------------
def run_reflib(dataset_name: str, config_suffix: str, config: dict) -> dict:
    """Run ref_lib DISCOVER on one dataset.

    Returns dict with keys: reward, expression, elapsed, n_terms, error.
    """
    # Must import tf_compat_shim before dso
    dso_dir = str(REFLIB_DSO_DIR)
    if dso_dir not in sys.path:
        sys.path.insert(0, dso_dir)

    # Import shim (idempotent after first import)
    import tf_compat_shim  # noqa: F401
    import tensorflow as tf

    config_path = str(
        REFLIB_DSO_DIR / "dso" / "config" / "MODE1"
        / f"config_pde_{config_suffix}.json"
    )
    if not os.path.exists(config_path):
        return {
            "reward": float("nan"),
            "expression": f"CONFIG NOT FOUND: {config_path}",
            "elapsed": 0.0,
            "n_terms": 0,
            "error": "config not found",
        }

    start = time.time()
    # Save and restore cwd since ref_lib uses relative paths
    orig_cwd = os.getcwd()
    try:
        os.chdir(str(REFLIB_DSO_DIR))

        # Override n_samples in config via pde_config
        pde_config = {
            "training": {
                "n_samples": config["n_samples"],
                "batch_size": config["batch_size"],
            },
            "experiment": {"seed": SEED},
        }

        from dso import DeepSymbolicOptimizer_PDE

        # Reset TF graph for each run
        tf.compat.v1.reset_default_graph()

        model = DeepSymbolicOptimizer_PDE(config_path, pde_config=pde_config)
        result = model.train()

        expression = str(result.get("expression", "N/A"))
        reward = float(result.get("r", float("nan")))
        n_terms = expression.count("+") + 1 if expression != "N/A" else 0
        elapsed = time.time() - start

        return {
            "reward": reward,
            "expression": expression,
            "elapsed": elapsed,
            "n_terms": n_terms,
            "error": None,
        }
    except Exception:
        elapsed = time.time() - start
        tb = traceback.format_exc()
        logger.error("ref_lib %s failed:\n%s", config_suffix, tb)
        return {
            "reward": float("nan"),
            "expression": "FAILED",
            "elapsed": elapsed,
            "n_terms": 0,
            "error": tb.splitlines()[-1],
        }
    finally:
        os.chdir(orig_cwd)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison(results: list[dict]) -> None:
    """Print side-by-side comparison table."""
    sep = "=" * 120
    print(f"\n{sep}")
    print("  DSCV vs ref_lib DISCOVER — BENCHMARK COMPARISON")
    print(sep)

    header = (
        f"{'Dataset':<18} {'Side':<8} {'Reward':>8} {'Time':>8} "
        f"{'#Terms':>6}  {'Expression'}"
    )
    print(header)
    print(
        f"{'-' * 18} {'-' * 8} {'-' * 8} {'-' * 8} "
        f"{'-' * 6}  {'-' * 60}"
    )

    for r in results:
        reward_str = f"{r['reward']:.4f}" if r["error"] is None else "ERROR"
        time_str = f"{r['elapsed']:.1f}s"
        expr = r["expression"]
        if len(expr) > 80:
            expr = expr[:77] + "..."
        print(
            f"{r['dataset']:<18} {r['side']:<8} {reward_str:>8} "
            f"{time_str:>8} {r['n_terms']:>6}  {expr}"
        )

    # Summary: dataset-level winner
    print(f"\n{sep}")
    print("  DATASET-LEVEL SUMMARY")
    print(sep)
    print(f"{'Dataset':<18} {'kd R':>8} {'ref R':>8} {'kd Time':>9} "
          f"{'ref Time':>9}  {'R Winner':<10} {'Speed Winner'}")
    print(f"{'-' * 18} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 9}  "
          f"{'-' * 10} {'-' * 12}")

    datasets_seen: list[str] = []
    for r in results:
        if r["dataset"] not in datasets_seen:
            datasets_seen.append(r["dataset"])

    for ds in datasets_seen:
        ds_results = [r for r in results if r["dataset"] == ds]
        kd_r = next((r for r in ds_results if r["side"] == "kd"), None)
        ref_r = next((r for r in ds_results if r["side"] == "ref_lib"), None)

        kd_reward = kd_r["reward"] if kd_r and kd_r["error"] is None else float("nan")
        ref_reward = ref_r["reward"] if ref_r and ref_r["error"] is None else float("nan")
        kd_time = kd_r["elapsed"] if kd_r else float("nan")
        ref_time = ref_r["elapsed"] if ref_r else float("nan")

        if math.isnan(kd_reward) and math.isnan(ref_reward):
            r_winner = "BOTH ERR"
        elif math.isnan(kd_reward):
            r_winner = "ref_lib"
        elif math.isnan(ref_reward):
            r_winner = "kd"
        elif abs(kd_reward - ref_reward) < 0.005:
            r_winner = "TIE"
        elif kd_reward > ref_reward:
            r_winner = "kd"
        else:
            r_winner = "ref_lib"

        if math.isnan(kd_time) or math.isnan(ref_time):
            t_winner = "N/A"
        elif kd_time < ref_time:
            t_winner = f"kd ({ref_time / kd_time:.1f}x)"
        else:
            t_winner = f"ref_lib ({kd_time / ref_time:.1f}x)"

        kd_r_str = f"{kd_reward:.4f}" if not math.isnan(kd_reward) else "ERROR"
        ref_r_str = f"{ref_reward:.4f}" if not math.isnan(ref_reward) else "ERROR"
        kd_t_str = f"{kd_time:.1f}s" if not math.isnan(kd_time) else "N/A"
        ref_t_str = f"{ref_time:.1f}s" if not math.isnan(ref_time) else "N/A"

        print(
            f"{ds:<18} {kd_r_str:>8} {ref_r_str:>8} {kd_t_str:>9} "
            f"{ref_t_str:>9}  {r_winner:<10} {t_winner}"
        )

    # Ground truth
    print(f"\n{sep}")
    print("  GROUND TRUTH REFERENCE")
    print(sep)
    for kd_name, _, _, gt in DATASETS:
        print(f"  {kd_name:<18} {gt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DSCV vs ref_lib DISCOVER benchmark"
    )
    parser.add_argument(
        "--side",
        choices=["kd", "reflib", "both"],
        default="both",
        help="Which side to run (default: both)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer iterations for sanity check",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Subset of datasets to run (kd names)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = QUICK_CONFIG if args.quick else FULL_CONFIG
    mode_label = "QUICK" if args.quick else "FULL"
    n_epochs = config["n_samples"] // config["batch_size"]

    print(f"Config: {mode_label} (n_samples={config['n_samples']}, "
          f"batch_size={config['batch_size']}, n_epochs={n_epochs}, seed={SEED})")
    print(f"Side: {args.side}")

    # Filter datasets if requested
    datasets = DATASETS
    if args.datasets:
        datasets = [d for d in DATASETS if d[0] in args.datasets]
        if not datasets:
            print(f"ERROR: No matching datasets for {args.datasets}")
            sys.exit(1)

    results: list[dict] = []
    run_kd_side = args.side in ("kd", "both")
    run_reflib_side = args.side in ("reflib", "both")

    for kd_name, config_suffix, reflib_ds_name, ground_truth in datasets:
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {kd_name}")
        print(f"{'=' * 60}")

        if run_kd_side:
            print(f"\n--- kd DSCV ---")
            r = run_kd(kd_name, config)
            r["dataset"] = kd_name
            r["side"] = "kd"
            results.append(r)
            print(f"  Reward: {r['reward']:.6f}" if r["error"] is None
                  else f"  ERROR: {r['error']}")
            print(f"  Time:   {r['elapsed']:.1f}s")

        if run_reflib_side:
            print(f"\n--- ref_lib DISCOVER ---")
            r = run_reflib(kd_name, config_suffix, config)
            r["dataset"] = kd_name
            r["side"] = "ref_lib"
            results.append(r)
            print(f"  Reward: {r['reward']:.6f}" if r["error"] is None
                  else f"  ERROR: {r['error']}")
            print(f"  Time:   {r['elapsed']:.1f}s")

    print_comparison(results)


if __name__ == "__main__":
    main()
