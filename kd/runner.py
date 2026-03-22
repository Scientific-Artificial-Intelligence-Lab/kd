"""Standalone runner for GPU-intensive KD models (DLGA, DSCV_SPR).

Entry point for Bohrium offline GPU tasks.  Also runnable locally for testing.

Usage (local):
    python -m kd.runner --config config.json --output_dir output/

Config JSON format::

    {
        "model": "dlga",              // "dlga" or "dscv_spr"
        "dataset_name": "burgers",    // built-in dataset name
        "dataset_file": null,         // OR path to uploaded .mat/.npy
        "output_dir": "output",
        "params": { ... }             // model-specific parameters
    }
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np


# ── Dataset loading ───────────────────────────────────────────

def _parse_ops(ops):
    """Parse comma-separated operator string or list into a list."""
    if isinstance(ops, str):
        return [op.strip() for op in ops.split(",") if op.strip()]
    return list(ops)


def load_dataset(config):
    """Load dataset from built-in registry or uploaded file."""
    dataset_name = config.get("dataset_name")
    dataset_file = config.get("dataset_file")

    if dataset_file and os.path.exists(dataset_file):
        from kd.dataset._base import PDEDataset, load_mat_file

        file_config = config.get("file_config", {})
        ext = os.path.splitext(dataset_file)[1].lower()

        if ext == ".mat":
            raw = load_mat_file(dataset_file)
            x = np.asarray(raw[file_config.get("x_key", "x")], dtype=float).flatten()
            t = np.asarray(raw[file_config.get("t_key", "t")], dtype=float).flatten()
            u = np.asarray(raw[file_config.get("u_key", "usol")])
            if u.shape == (len(t), len(x)):
                u = u.T
            return PDEDataset(equation_name="uploaded", x=x, t=t, usol=u)

        elif ext == ".npy":
            data = np.load(dataset_file)
            shape = file_config.get("shape")
            if not shape:
                raise ValueError(".npy files require 'shape' in file_config, e.g. [100, 201]")
            nx, nt = shape
            domain = file_config.get("domain")
            x_range = domain["x"] if domain and "x" in domain else (0, 1)
            t_range = domain["t"] if domain and "t" in domain else (0, 1)
            x = np.linspace(*x_range, nx)
            t = np.linspace(*t_range, nt)
            return PDEDataset(equation_name="uploaded", x=x, t=t, usol=data.reshape(nx, nt))

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    elif dataset_name:
        from kd.dataset import load_pde
        return load_pde(dataset_name)

    else:
        raise ValueError("Either 'dataset_name' or 'dataset_file' must be provided.")


# ── Model runners ─────────────────────────────────────────────

def run_dlga(dataset, params):
    """Run KD_DLGA and return ``(equation_str, model)``."""
    from kd.model.kd_dlga import KD_DLGA

    model = KD_DLGA(
        operators=_parse_ops(params.get("operators", "u,u_x,u_xx,u_xxx")),
        epi=float(params.get("epi", 0.1)),
        input_dim=int(params.get("input_dim", 2)),
        verbose=True,
        max_iter=int(params.get("max_iter", 3000)),
    )
    sample = params.get("sample")
    model.fit_dataset(dataset, sample=int(sample) if sample else None)
    equation = getattr(model, "eq_latex", None) or "(equation not available)"
    return equation, model


def run_dscv_spr(dataset, params):
    """Run KD_Discover_SPR and return ``(equation_str, model)``."""
    from kd.model.kd_discover import KD_Discover_SPR

    model = KD_Discover_SPR(
        binary_operators=_parse_ops(
            params.get("binary_operators", "add_t,mul_t,div_t,diff_t,diff2_t")
        ),
        unary_operators=_parse_ops(params.get("unary_operators", "n2_t")),
        n_samples_per_batch=int(params.get("n_samples_per_batch", 200)),
        seed=int(params.get("seed", 0)),
    )
    np.random.seed(int(params.get("seed", 0)))
    result = model.fit_dataset(
        dataset,
        n_epochs=int(params.get("n_epochs", 5)),
        sample_ratio=float(params.get("sample_ratio", 0.1)),
    )
    equation = result.get("expression", "(no expression)") if result else "(no result)"
    return equation, model


RUNNERS = {
    "dlga": run_dlga,
    "dscv_spr": run_dscv_spr,
}


# ── Output ────────────────────────────────────────────────────

def save_output(output_dir, model_name, dataset_info, equation, elapsed,
                params, error=None):
    """Write ``result.json`` to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "model": model_name,
        "dataset": dataset_info,
        "equation": equation,
        "elapsed_seconds": round(elapsed, 2),
        "status": "error" if error else "success",
        "params": params,
    }
    if error:
        result["error"] = str(error)
    path = os.path.join(output_dir, "result.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")


def save_viz(output_dir, model):
    """Generate and save key visualization images alongside result.json."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from kd.viz.core import VizRequest, render, configure, list_capabilities

        configure(save_dir=output_dir)
        caps = set(list_capabilities(model))

        for viz_type in ("equation", "search_evolution", "training_curve"):
            if viz_type not in caps and viz_type != "equation":
                continue
            try:
                vr = render(VizRequest(viz_type, model))
                if vr.figure is not None:
                    vr.figure.savefig(
                        os.path.join(output_dir, f"{viz_type}.png"),
                        dpi=150, bbox_inches="tight",
                    )
                    plt.close(vr.figure)
            except Exception:
                pass

        configure(save_dir=None)
    except Exception as e:
        print(f"[runner] Viz generation skipped: {e}")


# ── CLI entry point ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KD GPU Model Runner")
    parser.add_argument("--config", required=True, help="JSON config file path")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    model_name = config.get("model")
    if not model_name:
        print("Error: 'model' field is required in config JSON.")
        sys.exit(1)
    if model_name not in RUNNERS:
        print(f"Error: unknown model '{model_name}'. Available: {list(RUNNERS.keys())}")
        sys.exit(1)

    output_dir = args.output_dir or config.get("output_dir", "output")
    params = config.get("params", {})
    dataset_info = config.get("dataset_name") or config.get("dataset_file", "unknown")

    print(f"=== KD GPU Runner ===")
    print(f"  Model   : {model_name}")
    print(f"  Dataset : {dataset_info}")
    print(f"  Output  : {output_dir}")

    t0 = time.time()
    try:
        dataset = load_dataset(config)
        equation, model = RUNNERS[model_name](dataset, params)
        elapsed = time.time() - t0

        save_output(output_dir, model_name, dataset_info, equation, elapsed, params)
        save_viz(output_dir, model)

        print(f"\nDone in {elapsed:.1f}s")
        print(f"Equation: {equation}")
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        print(f"Error: {e}\n{tb}")
        save_output(output_dir, model_name, dataset_info, "(error)", elapsed, params, error=tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
