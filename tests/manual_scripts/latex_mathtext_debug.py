"""
Small experiment to debug DSCV LaTeX rendering issues.

Usage (from repo root, in kd-env):

    python tests/manual_scripts/latex_mathtext_debug.py

It will generate a few PNGs under ``artifacts/latex_debug/`` using both
the shared ``render_latex_to_image`` helper and plain Matplotlib calls,
for three representative cases:

- a simple Chafee-style polynomial (baseline, expected to render fine)
- a Burgers-style expression with spatial derivatives
- a KdV-style expression involving squared derivatives (with \\left/\\right)
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from kd.viz.equation_renderer import render_latex_to_image


OUTPUT_ROOT = Path("artifacts") / "latex_debug"


CASES = {
    "chafee_like": r"$u_t = 0.1787\,u_1^4$",
    "burgers_like": r"$u_t = - 0.4499 \frac{d}{d x_{1}} u_{1}^{2} + 0.1279 u_{1} - 0.0017 x_{1}^{2} \frac{d}{d x_{1}} u_{1} - 0.3534 u_{1}^{2}$",
    "kdv_like": r"$u_t = -0.0026 \left(\frac{d}{d x_{1}}\right)^{2} - 0.6277 x_{1} \frac{d}{d x_{1}} u_{1} + 0.1075 \frac{d}{d x_{1}} u_{1}$",
}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Matplotlib version:", matplotlib.__version__)
    print("rcParams['text.usetex'] =", matplotlib.rcParams.get("text.usetex"))

    for name, latex_str in CASES.items():
        print(f"\n=== Case: {name} ===")
        print("latex_str:", latex_str)

        # 1) Via shared renderer (same path as DSCV viz)
        out_renderer = OUTPUT_ROOT / f"{name}_renderer.png"
        print("  [renderer] saving to", out_renderer)
        render_latex_to_image(
            latex_str,
            output_path=str(out_renderer),
            show=False,
            font_size=16,
        )

        # 2) Direct Matplotlib mathtext, wrap=False（baseline，之前已验证为正常）
        out_direct = OUTPUT_ROOT / f"{name}_direct.png"
        print("  [direct]           saving to", out_direct)
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.text(0.5, 0.5, latex_str, ha="center", va="center", fontsize=16)
        ax.axis("off")
        fig.savefig(str(out_direct), dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

        # 3) Direct Matplotlib mathtext, wrap=True（模拟 renderer 的 wrap 参数）
        out_direct_wrap = OUTPUT_ROOT / f"{name}_direct_wrap.png"
        print("  [direct+wrap=True] saving to", out_direct_wrap)
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.text(0.5, 0.5, latex_str, ha="center", va="center", fontsize=16, wrap=True)
        ax.axis("off")
        fig.savefig(str(out_direct_wrap), dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    print("\nDone. Inspect artifacts/latex_debug/*.png for differences.")


if __name__ == "__main__":
    main()
