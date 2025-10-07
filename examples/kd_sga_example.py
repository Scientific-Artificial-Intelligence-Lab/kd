"""Minimal KD_SGA example using the unified dataset + viz façade.

This mirrors the verification defaults documented in
`notes/KD vs SGA Verification Report .md` so that teammates can
reproduce the comparison quickly.
"""

import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kd.dataset import load_pde
from kd.model.kd_sga import KD_SGA
from kd.viz import VizRequest, configure, render, render_equation

# ---------------------------------------------------------------------------
# Configuration (kept simple on purpose)
# ---------------------------------------------------------------------------
DATASET_NAME = "burgers"  # Available: 'chafee-infante', 'burgers', 'kdv'
SAVE_DIR = PROJECT_ROOT / "artifacts" / "sga_viz"
SGA_PARAMS = dict(
    num=20,
    depth=4,
    width=5,
    p_var=0.5,
    p_mute=0.3,
    p_cro=0.5,
    sga_run=100,
    seed=0,
)


def main() -> None:
    print(f"Loading dataset '{DATASET_NAME}' with kd.dataset.load_pde ...")
    dataset = load_pde(DATASET_NAME)

    print("Initialising KD_SGA with verification-aligned hyperparameters:")
    for key, value in SGA_PARAMS.items():
        print(f"  - {key} = {value}")
    model = KD_SGA(**SGA_PARAMS)

    model.fit_dataset(dataset)

    print("\nDiscovered equation (raw string):")
    print(f"  {model.best_pde_}")
    latex_full = model.equation_latex()
    latex_structure = model.equation_latex(include_coefficients=False)
    print("LaTeX (with coefficients):")
    print(f"  {latex_full}")
    print("LaTeX (structure only):")
    print(f"  {latex_structure}")

    configure(save_dir=str(SAVE_DIR))

    eq_result = render_equation(model)
    if eq_result.paths:
        print("\nEquation figures saved:")
        for path in eq_result.paths:
            print(f"  - {path}")

    intents = [
        ("field_comparison", {}),
        ("time_slices", {"slice_times": [0.0, 0.5, 1.0]}),
        ("residual", {"bins": 40}),
        ("parity", {}),
    ]
    for intent, options in intents:
        result = render(VizRequest(kind=intent, target=model, options=options))
        if result.paths:
            print(f"\n{intent.replace('_', ' ').title()} figures saved:")
            for path in result.paths:
                print(f"  - {path}")
        if result.metadata:
            print(f"{intent.title()} metadata: {result.metadata}")

    print("\nCalling legacy visualizer (writes assets into the working directory)...")
    model.plot_results()


if __name__ == "__main__":
    main()
