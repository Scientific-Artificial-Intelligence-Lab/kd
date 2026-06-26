
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import sympy

from kd.core.expr.sympy_bridge import to_sympy
from kd.viz.tree_layout import RenderNode, draw_tree, forest_to_render

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_DERIV_PREFIXES = ("diff", "lap")


def _format_number(expr: sympy.Basic) -> str:
    try:
        value = float(expr)
    except (TypeError, ValueError):
        return str(expr)
    if not math.isfinite(value):
        return str(expr)
    if value == int(value):
        return str(int(value))
    return f"{value:.3g}"


def _function_label(expr: sympy.Basic) -> str:
    return type(expr).__name__


def sympy_to_render(expr: sympy.Basic) -> RenderNode:
    if expr.is_number:
        return RenderNode(_format_number(expr), kind="const")
    if isinstance(expr, sympy.Symbol):
        return RenderNode(str(expr), kind="var")
    if isinstance(expr, sympy.Mul):
        non_numeric = [arg for arg in expr.args if not arg.is_number]
        if not non_numeric:
            return RenderNode(_format_number(expr), kind="const")
        children = [sympy_to_render(arg) for arg in non_numeric]
        if len(children) == 1:
            return children[0]
        return RenderNode("*", tuple(children), kind="op")
    if isinstance(expr, sympy.Add):
        children = []
        for arg in expr.args:
            if arg.could_extract_minus_sign():



                children.append(RenderNode("-", (sympy_to_render(-arg),), kind="op"))
            else:
                children.append(sympy_to_render(arg))
        return RenderNode("+", tuple(children), kind="op")
    if isinstance(expr, sympy.Pow):
        return RenderNode(
            "^", tuple(sympy_to_render(arg) for arg in expr.args), kind="op"
        )
    label = _function_label(expr)
    kind = "deriv" if label.startswith(_DERIV_PREFIXES) else "op"
    return RenderNode(
        label, tuple(sympy_to_render(arg) for arg in expr.args), kind=kind
    )


def _build_render(result: ExperimentResult) -> tuple[RenderNode | None, list[str]]:
    warnings: list[str] = []
    final_eval = result.final_eval
    terms = final_eval.terms





    if terms is not None:
        if final_eval.selected_indices is not None:
            indices = [i for i in final_eval.selected_indices if 0 <= i < len(terms)]


            dropped = len(final_eval.selected_indices) - len(indices)
            if dropped:
                warnings.append(
                    f"{dropped} selected term index(es) out of range for "
                    f"{len(terms)} term(s); equation tree shows the in-range subset."
                )
        else:
            indices = list(range(len(terms)))
        rendered = [sympy_to_render(to_sympy(terms[i], strict=False)) for i in indices]
        if rendered:
            return forest_to_render(rendered, op="+"), warnings
        return None, warnings
    expr_str = result.best_expression
    if not expr_str:
        return None, warnings
    return sympy_to_render(to_sympy(expr_str, strict=False)), warnings


def _draw_placeholder(ax: Axes, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")


def plot_equation_tree(result: ExperimentResult, ax: Axes) -> list[str]:
    root, warnings = _build_render(result)
    if root is None:
        warnings.append("Empty expression; skipping equation tree")
        _draw_placeholder(ax, "No expression")
        ax.set_title("Expression Tree")
        return warnings

    warnings.extend(draw_tree(root, ax))
    ax.set_title("Expression Tree")
    return warnings
