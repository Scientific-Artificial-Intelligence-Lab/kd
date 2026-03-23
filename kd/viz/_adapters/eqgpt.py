"""EqGPT visualization adapter: equation rendering + reward ranking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..core import VizResult
from ..equation_renderer import render_latex_to_image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants: EqGPT token -> LaTeX mapping
# ---------------------------------------------------------------------------
# Built from the 57-token vocabulary in dict_datas_0725.json.
# Operators (+, *, /) are handled during equation splitting, not here.

_TOKEN_TO_LATEX: Dict[str, str] = {
    # Special / control tokens
    "<pad>": "",
    "S": "",
    "E": "",
    # Operators (handled during split, but kept for single-token queries)
    "+": "+",
    "*": r"\cdot",
    "/": "/",
    # Plain variable
    "u": "u",
    # Simple derivatives
    "ut": "u_t",
    "ux": "u_x",
    "uy": "u_y",
    "uz": "u_z",
    "uxx": "u_{xx}",
    "uyy": "u_{yy}",
    "uzz": "u_{zz}",
    "utt": "u_{tt}",
    "uxt": "u_{xt}",
    "uxy": "u_{xy}",
    "uxxt": "u_{xxt}",
    "uxxx": "u_{xxx}",
    "uyyy": "u_{yyy}",
    "uxxtt": "u_{xxtt}",
    "uxxxx": "u_{xxxx}",
    "uxxxxx": "u_{xxxxx}",
    "uyyt": "u_{yyt}",
    # Powers of u and derivatives
    "u^2": "u^{2}",
    "u^3": "u^{3}",
    "ut^2": "u_t^{2}",
    "ut^3": "u_t^{3}",
    "ux^2": "u_x^{2}",
    "uy^2": "u_y^{2}",
    # Composite derivative terms
    "(uux)x": "(u u_x)_x",
    "(uux)t": "(u u_x)_t",
    "(uux)xx": "(u u_x)_{xx}",
    "(u^4)xx": r"(u^{4})_{xx}",
    "(u^3)xx": r"(u^{3})_{xx}",
    "(1/u)xx": "(1/u)_{xx}",
    "(u^-2*ux)x": r"(u^{-2} u_x)_x",
    "(u(u^2)xx)xx": r"(u (u^{2})_{xx})_{xx}",
    # Function tokens
    "Laplace(u)": r"\nabla^2 u",
    "BiLaplace(u)": r"\nabla^4 u",
    "Laplace(utt)": r"\nabla^2 u_{tt}",
    "sin(u)": r"\sin(u)",
    "sinh(u)": r"\sinh(u)",
    "sqrt(u)": r"\sqrt{u}",
    "exp(x)": r"\exp(x)",
    "exp(-y)": r"\exp(-y)",
    "sqrt(x)": r"\sqrt{x}",
    # Coordinate tokens
    "x": "x",
    "y": "y",
    "t": "t",
    "x^2": "x^{2}",
    "y^2": "y^{2}",
    "x^4": "x^{4}",
    "(x+y)": "(x+y)",
    # Special combined tokens
    "sinx": r"\sin(x)",
    "sint": r"\sin(t)",
    "(uxx+ux/x)^2": "(u_{xx}+u_x/x)^{2}",
}

# Maximum number of equations shown in reward ranking chart
_MAX_RANKING_BARS = 10


def _eqgpt_to_latex(eq_str: str) -> str:
    """Convert EqGPT token string to LaTeX.

    Parameters
    ----------
    eq_str : str
        Equation string using EqGPT token vocabulary,
        e.g. ``"ut+u*ux+uxx"``.

    Returns
    -------
    str
        LaTeX representation, e.g. ``"u_t + u \\cdot u_x + u_{xx}"``.
    """
    if not eq_str:
        return ""

    # Split the equation on operators (+, *, /) while keeping delimiters.
    # Strategy: tokenize by splitting around +, *, / operators that are
    # NOT inside parentheses.
    parts = _split_equation(eq_str)

    converted_parts: List[str] = []
    for part in parts:
        stripped = part.strip()
        if stripped in _TOKEN_TO_LATEX:
            converted = _TOKEN_TO_LATEX[stripped]
            if converted:  # skip empty (S, E, <pad>)
                converted_parts.append(converted)
        elif stripped:
            # Unknown token: passthrough
            converted_parts.append(stripped)

    # Join: spaces around \cdot, no spaces around + and /
    return _join_latex_parts(converted_parts)


def _split_equation(eq_str: str) -> List[str]:
    """Split an EqGPT equation string into tokens.

    Splits on +, *, / operators that appear at the top level
    (not inside parentheses). Each operator becomes its own token.
    """
    tokens: List[str] = []
    current: List[str] = []
    depth = 0

    for ch in eq_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch in ('+', '*', '/') and depth == 0:
            # Flush current accumulated token
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            tokens.append(ch)
            current = []
        else:
            current.append(ch)

    # Flush remaining
    token = "".join(current).strip()
    if token:
        tokens.append(token)

    return tokens


def _join_latex_parts(parts: List[str]) -> str:
    """Join converted LaTeX parts with correct spacing.

    * ``\\cdot`` gets spaces on both sides when between operands.
    * ``+`` and ``/`` are concatenated without extra spaces.
    """
    if not parts:
        return ""
    pieces: List[str] = []
    for i, p in enumerate(parts):
        if p == r"\cdot" and 0 < i < len(parts) - 1:
            pieces.append(f" {p} ")
        else:
            pieces.append(p)
    return "".join(pieces)


class EqGPTVizAdapter:
    """Visualization adapter for EqGPT models.

    Supported intents:

    * ``'equation'``        -- render best equation as LaTeX image;
    * ``'reward_ranking'``  -- top-N reward bar chart.
    """

    capabilities: Iterable[str] = {"equation", "reward_ranking"}

    def __init__(self, *, subdir: str = "eqgpt") -> None:
        self._subdir = subdir

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def render(self, request: Any, ctx: Any) -> VizResult:
        model = request.target
        kind = request.kind
        handler = {
            "equation": self._equation,
            "reward_ranking": self._reward_ranking,
        }.get(kind)
        if handler is None:
            return VizResult(
                intent=kind,
                warnings=[
                    f"EqGPT adapter does not support intent '{kind}'."
                ],
                metadata={"capabilities": sorted(self.capabilities)},
            )
        return handler(model, ctx)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_output(
        self, ctx: Any, filename: str
    ) -> Tuple[Path, Path]:
        """Resolve output directory and file path."""
        base = ctx.options.get("output_dir")
        if base is not None:
            base_path = Path(base)
            if self._subdir:
                base_path = base_path / self._subdir
            base_path.mkdir(parents=True, exist_ok=True)
            return base_path, base_path / filename

        relative = (
            f"{self._subdir}/{filename}" if self._subdir else filename
        )
        path = ctx.save_path(relative)
        return path.parent, path

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------
    @staticmethod
    def _latex_figure(latex: str, font_size: int = 18) -> Any:
        """Create an in-memory matplotlib figure with rendered LaTeX."""
        math = f"${latex}$" if latex and not latex.startswith("$") else latex
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, math or "", fontsize=font_size,
                ha="center", va="center")
        ax.axis("off")
        return fig

    def _equation(self, model: Any, ctx: Any) -> VizResult:
        """Render the best discovered equation as a LaTeX image."""
        if not hasattr(model, "result_"):
            return VizResult(
                intent="equation",
                warnings=["Model has no result_ attribute."],
            )

        result = model.result_
        best_eq = result.get("best_equation", "")
        latex = _eqgpt_to_latex(best_eq)

        _, path = self._resolve_output(ctx, "equation.png")
        font_size = ctx.options.get("font_size", 18)
        try:
            render_latex_to_image(
                latex,
                output_path=str(path),
                font_size=font_size,
                show=False,
            )
        except Exception as exc:
            logger.warning("Failed to render EqGPT equation: %s", exc)
            return VizResult(
                intent="equation",
                warnings=[
                    f"Failed to render EqGPT equation: {exc}"
                ],
            )

        # render_latex_to_image may silently swallow errors; verify output
        if not path.exists():
            return VizResult(
                intent="equation",
                warnings=["Equation render produced no output file."],
                metadata={"latex": latex},
            )

        # Build in-memory figure for callers that check VizResult.figure
        # (e.g. app.py _render_equation_fig)
        fig = self._latex_figure(latex, font_size)

        return VizResult(
            intent="equation",
            figure=fig,
            paths=[path],
            metadata={"latex": latex},
        )

    def _reward_ranking(self, model: Any, ctx: Any) -> VizResult:
        """Create a horizontal bar chart of top-N equation rewards."""
        if not hasattr(model, "result_"):
            return VizResult(
                intent="reward_ranking",
                warnings=["Model has no result_ attribute."],
            )

        result = model.result_
        equations: List[str] = result.get("equations", [])
        rewards: List[float] = result.get("rewards", [])

        # Warn on empty results
        if not equations or not rewards:
            return VizResult(
                intent="reward_ranking",
                warnings=["No equations/rewards available."],
            )

        warnings: List[str] = []

        # Warn on length mismatch; truncate to shorter
        if len(equations) != len(rewards):
            msg = (
                f"equations ({len(equations)}) and rewards "
                f"({len(rewards)}) length mismatch"
            )
            logger.warning(msg)
            warnings.append(msg)
            min_len = min(len(equations), len(rewards))
            equations = equations[:min_len]
            rewards = rewards[:min_len]

        # Filter NaN/Inf rewards
        rewards_arr = np.array(rewards, dtype=float)
        valid_mask = np.isfinite(rewards_arr)
        n_invalid = int(np.sum(~valid_mask))
        if n_invalid > 0:
            msg = f"Filtered {n_invalid} non-finite rewards (NaN/Inf)"
            logger.warning(msg)
            warnings.append(msg)

        # Check if ALL are non-finite
        if not np.any(valid_mask):
            warnings.append("All rewards are NaN/Inf; cannot plot.")
            return VizResult(
                intent="reward_ranking",
                warnings=warnings,
            )

        # Apply filter
        valid_eqs = [
            eq for eq, v in zip(equations, valid_mask) if v
        ]
        valid_rewards = rewards_arr[valid_mask]

        # Cap at top N
        n_bars = min(len(valid_eqs), _MAX_RANKING_BARS)
        top_eqs = valid_eqs[:n_bars]
        top_rewards = valid_rewards[:n_bars]

        return self._plot_ranking(
            top_eqs, top_rewards, n_bars, ctx, warnings
        )

    def _plot_ranking(
        self,
        equations: List[str],
        rewards: np.ndarray,
        n_bars: int,
        ctx: Any,
        warnings: Optional[List[str]] = None,
    ) -> VizResult:
        """Render the reward ranking bar chart and save."""
        fig, ax = plt.subplots(
            figsize=ctx.options.get("figsize", (8, 5))
        )
        try:
            ranks = np.arange(1, n_bars + 1)
            ax.barh(ranks, rewards, align="center")

            ax.set_yticks(ranks)
            ax.set_yticklabels([f"#{i}" for i in ranks])
            ax.set_xlabel("Reward")
            ax.set_title(
                ctx.options.get("title", "EqGPT Reward Ranking")
            )
            ax.invert_yaxis()
            ax.grid(True, linestyle="--", alpha=0.3)

            _, path = self._resolve_output(ctx, "reward_ranking.png")
            fig.savefig(
                str(path),
                dpi=ctx.options.get("dpi", 300),
                bbox_inches="tight",
            )
        finally:
            plt.close(fig)

        return VizResult(
            intent="reward_ranking",
            paths=[path],
            warnings=warnings or [],
            metadata={"n_bars": n_bars},
        )


__all__ = ["EqGPTVizAdapter", "_eqgpt_to_latex"]
