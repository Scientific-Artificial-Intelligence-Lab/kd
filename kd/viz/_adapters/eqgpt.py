"""EqGPT visualization adapter: equation rendering + reward ranking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .._contracts import ParityPlotData, RewardEvolutionData
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
    * ``'reward_evolution'`` -- per-epoch top-k reward lines.
    * ``'parity'``          -- LHS vs RHS scatter plot.
    """

    capabilities: Iterable[str] = {
        "equation",
        "reward_ranking",
        "reward_evolution",
        "parity",
    }

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
            "reward_evolution": self._reward_evolution,
            "parity": self._parity,
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
        if latex:
            latex = latex + " = 0"

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

    def _reward_evolution(self, model: Any, ctx: Any) -> VizResult:
        """Create a line chart of per-epoch EqGPT top-k rewards."""
        if not hasattr(model, "result_"):
            return VizResult(
                intent="reward_evolution",
                warnings=["Model has no result_ attribute."],
            )

        data, warnings = self._build_reward_evolution(
            model.result_.get("reward_history")
        )
        if data is None:
            return VizResult(
                intent="reward_evolution",
                warnings=warnings or ["No reward history available."],
            )

        fig, ax = plt.subplots(
            figsize=ctx.options.get("figsize", (8, 5))
        )
        try:
            self._plot_reward_series(
                ax, data.steps, data.best_reward,
                label="Top-1", color="black", linestyle="-",
            )
            if data.max_reward is not None:
                self._plot_reward_series(
                    ax, data.steps, data.max_reward,
                    label="Top-5", color="#F47E62", linestyle="--",
                )
            if data.mean_reward is not None:
                self._plot_reward_series(
                    ax, data.steps, data.mean_reward,
                    label="Top-10", color="#4F8FBA", linestyle="-.",
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Reward")
            ax.set_title(
                ctx.options.get("title", "EqGPT Reward Evolution")
            )
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best", frameon=False)

            _, path = self._resolve_output(ctx, "reward_evolution.png")
            fig.savefig(
                str(path),
                dpi=ctx.options.get("dpi", 300),
                bbox_inches="tight",
            )
        finally:
            plt.close(fig)

        return VizResult(
            intent="reward_evolution",
            paths=[path],
            warnings=warnings,
            metadata={"reward_evolution": data},
        )

    def _parity(self, model: Any, ctx: Any) -> VizResult:
        """Create a LHS vs RHS parity scatter plot."""
        if not hasattr(model, "result_"):
            return VizResult(
                intent="parity",
                warnings=["Model has no result_ attribute."],
            )

        parity_raw = model.result_.get("parity_data")
        if parity_raw is None:
            return VizResult(
                intent="parity",
                warnings=["No parity_data available in result_."],
            )

        lhs, rhs, warnings = self._validate_parity_arrays(parity_raw)
        if lhs is None or rhs is None:
            return VizResult(
                intent="parity", warnings=warnings,
            )

        parity_data = ParityPlotData.from_actual_predicted(lhs, rhs)
        summary = self._compute_residual_stats(lhs, rhs)

        fig, ax = plt.subplots(
            figsize=ctx.options.get("figsize", (7, 7))
        )
        try:
            ax.scatter(rhs, lhs, alpha=0.35, s=20, label="RHS vs LHS")
            lo = float(min(lhs.min(), rhs.min()))
            hi = float(max(lhs.max(), rhs.max()))
            ref = np.linspace(lo, hi, 100)
            ax.plot(ref, ref, "r--", linewidth=1.5, label="y = x")
            ax.set_xlabel("RHS (predicted)")
            ax.set_ylabel("LHS (actual)")
            ax.set_title(ctx.options.get("title", "EqGPT Parity Plot"))
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_aspect("equal", "box")

            _, path = self._resolve_output(ctx, "parity_plot.png")
            fig.savefig(
                str(path),
                dpi=ctx.options.get("dpi", 300),
                bbox_inches="tight",
            )
        finally:
            plt.close(fig)

        return VizResult(
            intent="parity",
            paths=[path],
            warnings=warnings,
            metadata={"parity": parity_data, "summary": summary},
        )

    @staticmethod
    def _validate_parity_arrays(
        parity_raw: Any,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Extract and validate lhs/rhs arrays from parity_data dict.

        Returns (lhs, rhs, warnings). lhs/rhs are None on failure.
        """
        warnings: List[str] = []

        if not isinstance(parity_raw, dict):
            return None, None, ["parity_data is not a dict."]

        lhs_raw = parity_raw.get("lhs")
        rhs_raw = parity_raw.get("rhs")
        if lhs_raw is None or rhs_raw is None:
            return None, None, ["parity_data missing 'lhs' or 'rhs' key."]

        lhs = np.asarray(lhs_raw, dtype=float).reshape(-1)
        rhs = np.asarray(rhs_raw, dtype=float).reshape(-1)

        # Shape mismatch
        if lhs.shape[0] != rhs.shape[0]:
            return None, None, [
                f"lhs length ({lhs.shape[0]}) and rhs length "
                f"({rhs.shape[0]}) mismatch."
            ]

        # Empty arrays
        if lhs.size == 0:
            return None, None, ["parity_data arrays are empty."]

        # Filter NaN/Inf
        finite_mask = np.isfinite(lhs) & np.isfinite(rhs)
        n_invalid = int(np.sum(~finite_mask))
        if n_invalid > 0:
            warnings.append(
                f"Filtered {n_invalid} non-finite points from parity data."
            )
        if not np.any(finite_mask):
            return None, None, [
                "All parity data points are NaN/Inf; no finite data to plot."
            ]

        return lhs[finite_mask], rhs[finite_mask], warnings

    @staticmethod
    def _compute_residual_stats(
        lhs: np.ndarray, rhs: np.ndarray,
    ) -> Dict[str, float]:
        """Compute residual statistics for parity data."""
        residuals = lhs - rhs
        return {
            "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
            "mean_residual": float(np.mean(residuals)),
            "max_abs_residual": float(np.max(np.abs(residuals))),
        }

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

            # Zoom x-axis to show differences when rewards are close
            r_min, r_max = float(rewards.min()), float(rewards.max())
            margin = max((r_max - r_min) * 0.3, 0.005)
            ax.set_xlim(r_min - margin, r_max + margin)

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

    @staticmethod
    def _plot_reward_series(
        ax: Any,
        steps: np.ndarray,
        values: np.ndarray,
        *,
        label: str,
        color: str,
        linestyle: str,
    ) -> None:
        """Plot one reward series while skipping non-finite points."""
        values_arr = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(values_arr)
        if not np.any(finite_mask):
            return
        ax.plot(
            steps[finite_mask],
            values_arr[finite_mask],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )

    def _build_reward_evolution(
        self,
        reward_history: Any,
    ) -> Tuple[Optional[RewardEvolutionData], List[str]]:
        """Convert raw per-epoch top-k rewards into plotting data."""
        if reward_history is None:
            return None, ["No reward_history available."]

        try:
            history = list(reward_history)
        except TypeError:
            return None, ["reward_history is not iterable."]
        if not history:
            return None, ["Reward history is empty."]

        warnings: List[str] = []
        top1: List[float] = []
        top5: List[float] = []
        top10: List[float] = []

        filtered_nonfinite = False
        for epoch_idx, epoch_rewards in enumerate(history, start=1):
            try:
                rewards_arr = np.asarray(epoch_rewards, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                warnings.append(
                    f"Epoch {epoch_idx} reward data is invalid; leaving a gap."
                )
                top1.append(np.nan)
                top5.append(np.nan)
                top10.append(np.nan)
                continue
            finite_rewards = rewards_arr[np.isfinite(rewards_arr)]

            if finite_rewards.size != rewards_arr.size:
                filtered_nonfinite = True
            if finite_rewards.size == 0:
                warnings.append(
                    f"Epoch {epoch_idx} has no finite rewards; leaving a gap."
                )
                top1.append(np.nan)
                top5.append(np.nan)
                top10.append(np.nan)
                continue

            top1.append(float(finite_rewards[0]))
            top5.append(self._reward_at_rank(finite_rewards, 5))
            top10.append(self._reward_at_rank(finite_rewards, 10))

        if filtered_nonfinite:
            warnings.append(
                "Filtered non-finite values from reward history."
            )

        best_arr = np.asarray(top1, dtype=float)
        top5_arr = np.asarray(top5, dtype=float)
        top10_arr = np.asarray(top10, dtype=float)
        if not np.isfinite(best_arr).any():
            return None, warnings + ["Reward history has no finite values."]

        data = RewardEvolutionData(
            steps=np.arange(1, len(history) + 1),
            best_reward=best_arr,
            max_reward=top5_arr,
            mean_reward=top10_arr,
            metadata={
                "series_labels": {
                    "best_reward": "Top-1",
                    "max_reward": "Top-5",
                    "mean_reward": "Top-10",
                }
            },
        )
        return data, warnings

    @staticmethod
    def _reward_at_rank(
        rewards: np.ndarray,
        rank: int,
    ) -> float:
        """Return the reward at the given 1-based rank or NaN if missing."""
        if rewards.size < rank:
            return np.nan
        return float(rewards[rank - 1])


__all__ = ["EqGPTVizAdapter", "_eqgpt_to_latex"]
