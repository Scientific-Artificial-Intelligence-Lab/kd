"""
Smoke-test every enabled dataset+model combination in the app.
Runs each combo with minimal params and reports pass/fail.

Usage:
    python tests/test_all_combos.py
"""
import sys, os, traceback, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kd.dataset._registry import PDE_REGISTRY
from kd.dataset import load_pde
from kd.viz.core import configure

MODEL_KEYS = {
    "KD_SGA": "sga",
    "KD_DLGA": "dlga",
    "KD_Discover": "discover",
    "KD_Discover_SPR": "discover_spr",
}

# Minimal params for each model (fast execution)
def run_combo(dataset_name, model_name):
    """Return (success: bool, equation_or_error: str, elapsed: float)."""
    configure(save_dir=None)
    import numpy as np

    t0 = time.time()
    try:
        dataset = load_pde(dataset_name)

        if model_name == "KD_SGA":
            from kd.model.kd_sga import KD_SGA
            model = KD_SGA(sga_run=1, num=5, depth=2, seed=0)
            model.fit_dataset(dataset)
            eq = model.equation_latex()
            return True, eq, time.time() - t0

        elif model_name == "KD_DLGA":
            from kd.model.kd_dlga import KD_DLGA
            model = KD_DLGA(
                operators=["u", "u_x", "u_xx"],
                epi=0.1, input_dim=2, verbose=False, max_iter=50,
            )
            model.fit_dataset(dataset, sample=100)
            eq = getattr(model, "eq_latex", "(no eq)")
            return True, eq, time.time() - t0

        elif model_name == "KD_Discover":
            from kd.model.kd_discover import KD_Discover
            np.random.seed(0)
            model = KD_Discover(
                binary_operators=["add", "mul"],
                unary_operators=["n2"],
                n_samples_per_batch=50,
                seed=0,
            )
            result = model.fit_dataset(dataset, n_epochs=2)
            eq = result.get("expression", "(no expr)")
            return True, eq, time.time() - t0

        elif model_name == "KD_Discover_SPR":
            from kd.model.kd_discover import KD_Discover_SPR
            np.random.seed(0)
            model = KD_Discover_SPR(
                binary_operators=["add_t", "mul_t", "diff_t"],
                unary_operators=["n2_t"],
                n_samples_per_batch=50,
                seed=0,
            )
            result = model.fit_dataset(dataset, n_epochs=2, sample_ratio=0.5)
            eq = result.get("expression", "(no expr)")
            return True, eq, time.time() - t0

        else:
            return False, f"Unknown model: {model_name}", time.time() - t0

    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}\n{traceback.format_exc()}", time.time() - t0


def main():
    # Enumerate all enabled combos
    combos = []
    for ds_name, info in PDE_REGISTRY.items():
        if info.get("status") == "pending":
            continue
        models_map = info.get("models", {})
        for display_name, reg_key in MODEL_KEYS.items():
            if models_map.get(reg_key, False):
                combos.append((ds_name, display_name))

    print(f"=== Testing {len(combos)} dataset+model combinations ===\n")

    results = []
    for ds, mdl in combos:
        tag = f"{ds} + {mdl}"
        print(f"[RUN ] {tag} ... ", end="", flush=True)
        ok, msg, elapsed = run_combo(ds, mdl)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] ({elapsed:.1f}s)")
        if not ok:
            # Print first 3 lines of error
            err_lines = msg.strip().split("\n")
            for line in err_lines[:3]:
                print(f"       {line}")
            if len(err_lines) > 3:
                print(f"       ... ({len(err_lines)-3} more lines)")
        results.append((tag, ok, msg, elapsed))

    # Summary
    print(f"\n{'='*60}")
    passed = sum(1 for _, ok, _, _ in results if ok)
    failed = sum(1 for _, ok, _, _ in results if not ok)
    print(f"PASSED: {passed}  FAILED: {failed}  TOTAL: {len(results)}")

    if failed:
        print(f"\nFailed combinations:")
        for tag, ok, msg, _ in results:
            if not ok:
                first_line = msg.strip().split("\n")[0]
                print(f"  - {tag}: {first_line}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
