"""Manual DSCV matrix check over core PDE datasets.

Usage (from repo root, in kd-env):

    python tests/manual_scripts/dscv_dataset_check.py

会依次对若干 registry 中标记为适合 DSCV 的数据集运行 KD_DSCV.fit_dataset，
并打印发现的表达式与 reward；同时通过 kd.viz 生成简单的 equation/residual/field_comparison
图像，写入 artifacts/dscv_matrix/<dataset_name>/。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent
sys.path.insert(0, str(repo_root))

from kd.dataset import load_pde
from kd.model import KD_DSCV
from kd.viz import VizRequest, configure, render, render_equation


DATASETS = [
    "chafee-infante",
    "burgers",
    "kdv",
    "PDE_divide",
    "PDE_compound",
    "fisher",
    "fisher_linear",
]


def main() -> None:
    output_root = Path("artifacts") / "dscv_matrix"
    configure(save_dir=output_root)
    np.random.seed(42)

    for name in DATASETS:
        print(f"\n=== Dataset: {name} ===")
        try:
            dataset = load_pde(name)
        except Exception as exc:
            print(f"[load] failed: {exc}")
            continue

        model = KD_DSCV(
            binary_operators=["add", "mul", "diff"],
            unary_operators=["n2"],
            n_iterations=5,
            n_samples_per_batch=500,
            seed=0,
        )

        try:
            result = model.fit_dataset(dataset, n_epochs=5, verbose=False)
        except Exception as exc:
            print(f"[fit_dataset] failed: {exc}")
            continue

        expr = result.get("expression")
        reward = result.get("r")
        print(f"[train] expression: {expr}, reward: {reward}")

        # Basic viz sanity checks (best-effort; failures只打印 warning)
        try:
            eq_res = render_equation(model, show_info=False, output_dir=output_root / name)
            print(f"[viz:equation] paths={len(eq_res.paths)}, warnings={eq_res.warnings}")
        except Exception as exc:
            print(f"[viz:equation] failed: {exc}")

        try:
            res_res = render(
                VizRequest(
                    kind="residual",
                    target=model,
                    options={"output_dir": output_root / name},
                )
            )
            print(f"[viz:residual] paths={len(res_res.paths)}, warnings={res_res.warnings}")
        except Exception as exc:
            print(f"[viz:residual] failed: {exc}")

        try:
            fc_res = render(
                VizRequest(
                    kind="field_comparison",
                    target=model,
                    options={"output_dir": output_root / name},
                )
            )
            print(f"[viz:field_comparison] paths={len(fc_res.paths)}, warnings={fc_res.warnings}")
        except Exception as exc:
            print(f"[viz:field_comparison] failed: {exc}")


if __name__ == "__main__":
    main()
