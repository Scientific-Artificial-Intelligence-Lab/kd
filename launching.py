"""KD PDE Discovery — Bohrium Launching Framework Entry Point.

This is the EntryPoint for the Bohrium standard offline task app.
It wraps kd.runner logic using the dp.launching framework.

EntryPoint path in image: /root/kd/launching.py
"""

import sys
import json
import os
import time
import traceback
from pathlib import Path

from dp.launching.typing import BaseModel, Field, Int, Float, String, Enum, OutputDirectory
from dp.launching.cli import to_runner, default_minimal_exception_handler

import numpy as np


# ── Input schema ──────────────────────────────────────────────

class ModelChoice(String, Enum):
    dlga = "dlga"
    dscv_spr = "dscv_spr"
    eqgpt = "eqgpt"


class Options(BaseModel):
    model: ModelChoice = Field(
        default="dlga",
        description="Model to run: dlga or dscv_spr",
    )
    dataset_name: String = Field(
        default="burgers",
        description="Built-in dataset name (e.g. burgers, kdv, ks)",
    )

    # DLGA parameters
    operators: String = Field(
        default="u,u_x,u_xx,u_xxx",
        description="[DLGA] Comma-separated operator list",
    )
    epi: Float = Field(default=0.1, description="[DLGA] Sparsity penalty")
    max_iter: Int = Field(default=3000, description="[DLGA] NN max iterations")
    sample: Int = Field(default=500, description="[DLGA] Sample points")

    # DSCV_SPR parameters
    binary_operators: String = Field(
        default="add_t,mul_t,div_t,diff_t,diff2_t",
        description="[DSCV_SPR] Binary operators",
    )
    unary_operators: String = Field(
        default="n2_t",
        description="[DSCV_SPR] Unary operators",
    )
    n_samples_per_batch: Int = Field(
        default=200,
        description="[DSCV_SPR] Samples per batch",
    )
    seed: Int = Field(default=0, description="Random seed")
    n_epochs: Int = Field(default=5, description="[DSCV_SPR] Search epochs")
    sample_ratio: Float = Field(
        default=0.1,
        description="[DSCV_SPR] Observation ratio (0.01-1.0)",
    )

    # EqGPT parameters
    optimize_epochs: Int = Field(
        default=5, description="[EqGPT] RL search epochs",
    )
    samples_per_epoch: Int = Field(
        default=400, description="[EqGPT] Candidates per epoch",
    )
    case_filter: String = Field(
        default="N", description="[EqGPT] 'N' for 12 N-type cases, 'all' for 23 cases",
    )

    output_dir: OutputDirectory = Field(default="./output")


# ── Entry function ────────────────────────────────────────────

def entry_function(opts: Options) -> int:
    output_dir = Path(opts.output_dir.get_path())
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = opts.model
    dataset_name = opts.dataset_name

    print(f"=== KD GPU Runner (Launching) ===")
    print(f"  Model   : {model_name}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Output  : {output_dir}")

    t0 = time.time()

    try:
        # Build runner config
        if model_name == "dlga":
            params = {
                "operators": opts.operators,
                "epi": float(opts.epi),
                "max_iter": int(opts.max_iter),
                "sample": int(opts.sample),
            }
        elif model_name == "eqgpt":
            params = {
                "optimize_epochs": int(opts.optimize_epochs),
                "samples_per_epoch": int(opts.samples_per_epoch),
                "case_filter": opts.case_filter,
                "seed": int(opts.seed),
            }
        else:  # dscv_spr
            params = {
                "binary_operators": opts.binary_operators,
                "unary_operators": opts.unary_operators,
                "n_samples_per_batch": int(opts.n_samples_per_batch),
                "seed": int(opts.seed),
                "n_epochs": int(opts.n_epochs),
                "sample_ratio": float(opts.sample_ratio),
            }
            np.random.seed(int(opts.seed))

        # Import and run
        from kd.runner import RUNNERS, save_output, save_viz

        # EqGPT loads its own data; others need a dataset
        if model_name == "eqgpt":
            dataset = None
        else:
            from kd.runner import load_dataset
            config = {"dataset_name": dataset_name}
            dataset = load_dataset(config)
        equation, model = RUNNERS[model_name](dataset, params)
        elapsed = time.time() - t0

        save_output(str(output_dir), model_name, dataset_name,
                    equation, elapsed, params, model=model)
        save_viz(str(output_dir), model)

        print(f"\nDone in {elapsed:.1f}s")
        print(f"Equation: {equation}")
        return 0

    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        print(f"Error: {e}\n{tb}")
        save_output(str(output_dir), model_name, dataset_name,
                    "(error)", elapsed, {}, error=tb)
        return 1


# ── Launching boilerplate ─────────────────────────────────────

def to_parser():
    return to_runner(
        Options,
        entry_function,
        version="1.0.0",
        exception_handler=default_minimal_exception_handler,
    )


if __name__ == "__main__":
    to_parser()(sys.argv[1:])
