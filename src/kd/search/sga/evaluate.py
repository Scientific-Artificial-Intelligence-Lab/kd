
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from kd.core.safety import safe_div
from kd.search.sga.pde import PDE
from kd.search.sga.tree import Node, Tree

logger = logging.getLogger(__name__)





@dataclass
class DiffContext:

    field_shape: tuple[int, ...]
    axis_map: dict[str, int] = field(default_factory=dict)
    delta: dict[str, float] = field(default_factory=dict)
    lhs_axis: str | None = None


def _finite_diff_torch(
    f: Tensor,
    dx: float,
    axis: int,
    order: int,
) -> Tensor:
    if order not in (1, 2):
        raise ValueError(
            f"Only first and second derivatives are supported; got {order}."
        )
    if not math.isfinite(dx) or dx == 0.0:
        raise ValueError(f"Finite difference requires a finite, non-zero dx; got {dx}.")

    f_moved = f.to(dtype=_fd_dtype(f)).movedim(axis, 0)
    n_points = f_moved.shape[0]
    min_points = 3 if order == 1 else 4
    if n_points < min_points:
        raise ValueError(
            f"Order {order} finite difference requires at least {min_points} points; "
            f"got {n_points}."
        )

    result = torch.zeros_like(f_moved)
    if order == 1:
        if n_points >= 5:
            result[2:-2] = (
                -f_moved[4:] + 8.0 * f_moved[3:-1] - 8.0 * f_moved[1:-3] + f_moved[:-4]
            ) / (12.0 * dx)
            result[0] = (
                -25.0 * f_moved[0]
                + 48.0 * f_moved[1]
                - 36.0 * f_moved[2]
                + 16.0 * f_moved[3]
                - 3.0 * f_moved[4]
            ) / (12.0 * dx)
            result[1] = (
                -3.0 * f_moved[0]
                - 10.0 * f_moved[1]
                + 18.0 * f_moved[2]
                - 6.0 * f_moved[3]
                + f_moved[4]
            ) / (12.0 * dx)
            result[-2] = (
                -f_moved[-5]
                + 6.0 * f_moved[-4]
                - 18.0 * f_moved[-3]
                + 10.0 * f_moved[-2]
                + 3.0 * f_moved[-1]
            ) / (12.0 * dx)
            result[-1] = (
                3.0 * f_moved[-5]
                - 16.0 * f_moved[-4]
                + 36.0 * f_moved[-3]
                - 48.0 * f_moved[-2]
                + 25.0 * f_moved[-1]
            ) / (12.0 * dx)
        else:
            result[1:-1] = (f_moved[2:] - f_moved[:-2]) / (2.0 * dx)
            result[0] = (-1.5 * f_moved[0] + 2.0 * f_moved[1] - 0.5 * f_moved[2]) / dx
            result[-1] = (
                1.5 * f_moved[-1] - 2.0 * f_moved[-2] + 0.5 * f_moved[-3]
            ) / dx
    else:
        dx2 = dx**2
        if n_points >= 5:
            result[2:-2] = (
                -f_moved[4:]
                + 16.0 * f_moved[3:-1]
                - 30.0 * f_moved[2:-2]
                + 16.0 * f_moved[1:-3]
                - f_moved[:-4]
            ) / (12.0 * dx2)
            result[0] = (
                35.0 * f_moved[0]
                - 104.0 * f_moved[1]
                + 114.0 * f_moved[2]
                - 56.0 * f_moved[3]
                + 11.0 * f_moved[4]
            ) / (12.0 * dx2)
            if n_points >= 6:
                result[1] = (
                    10.0 * f_moved[0]
                    - 15.0 * f_moved[1]
                    - 4.0 * f_moved[2]
                    + 14.0 * f_moved[3]
                    - 6.0 * f_moved[4]
                    + f_moved[5]
                ) / (12.0 * dx2)
                result[-2] = (
                    f_moved[-6]
                    - 6.0 * f_moved[-5]
                    + 14.0 * f_moved[-4]
                    - 4.0 * f_moved[-3]
                    - 15.0 * f_moved[-2]
                    + 10.0 * f_moved[-1]
                ) / (12.0 * dx2)
            else:
                result[1] = (
                    11.0 * f_moved[0]
                    - 20.0 * f_moved[1]
                    + 6.0 * f_moved[2]
                    + 4.0 * f_moved[3]
                    - f_moved[4]
                ) / (12.0 * dx2)
                result[-2] = (
                    -f_moved[-5]
                    + 4.0 * f_moved[-4]
                    + 6.0 * f_moved[-3]
                    - 20.0 * f_moved[-2]
                    + 11.0 * f_moved[-1]
                ) / (12.0 * dx2)
            result[-1] = (
                11.0 * f_moved[-5]
                - 56.0 * f_moved[-4]
                + 114.0 * f_moved[-3]
                - 104.0 * f_moved[-2]
                + 35.0 * f_moved[-1]
            ) / (12.0 * dx2)
        else:
            result[1:-1] = (f_moved[2:] - 2.0 * f_moved[1:-1] + f_moved[:-2]) / dx2
            result[0] = (
                2.0 * f_moved[0] - 5.0 * f_moved[1] + 4.0 * f_moved[2] - f_moved[3]
            ) / dx2
            result[-1] = (
                2.0 * f_moved[-1] - 5.0 * f_moved[-2] + 4.0 * f_moved[-3] - f_moved[-4]
            ) / dx2

    return result.movedim(0, axis)




ZERO_COLUMN_EPS = 1e-10
"""Threshold for filtering near-zero columns in execute_pde."""

_DERIVATIVE_OPS = {"d", "d^2"}
"""Binary derivative operators handled outside the arithmetic dispatch table."""






DISPATCH: dict[str, Callable[..., Tensor]] = {
    "+": torch.add,
    "-": torch.sub,
    "*": torch.mul,
    "/": safe_div,
    "^2": lambda x: x**2,
    "^3": lambda x: x**3,
}





def _fd_dtype(u: Tensor) -> torch.dtype:
    if torch.is_floating_point(u):
        return u.dtype
    return torch.get_default_dtype()


def _execute_derivative(
    node: Node,
    data_dict: dict[str, Tensor],
    diff_ctx: DiffContext | None,
) -> Tensor:
    if diff_ctx is None:
        raise ValueError("diff_ctx is required for derivative execution.")
    if len(node.children) != 2 or not node.children[1].is_leaf:
        raise ValueError(
            "Derivative nodes require a leaf axis-name as the right child."
        )
    axis_name = node.children[1].name
    if axis_name not in diff_ctx.axis_map:
        raise KeyError(f"Missing axis_map entry for axis '{axis_name}'.")
    if axis_name not in diff_ctx.delta:
        raise KeyError(f"Missing delta for axis '{axis_name}'.")

    values = _execute_node(node.children[0], data_dict, diff_ctx)
    expected_size = math.prod(diff_ctx.field_shape)
    if values.numel() != expected_size:
        raise ValueError(
            "Derivative input size does not match DiffContext.field_shape: "
            f"{values.numel()} != {expected_size}."
        )

    field = values.reshape(diff_ctx.field_shape)
    deriv = _finite_diff_torch(
        field,
        diff_ctx.delta[axis_name],
        diff_ctx.axis_map[axis_name],
        order=1 if node.name == "d" else 2,
    )
    return deriv.reshape(-1)


def _execute_node(
    node: Node,
    data_dict: dict[str, Tensor],
    diff_ctx: DiffContext | None,
) -> Tensor:
    if node.is_leaf:
        if node.name not in data_dict:
            raise KeyError(
                f"Variable '{node.name}' not found in data_dict. "
                f"Available: {list(data_dict.keys())}"
            )
        return data_dict[node.name].reshape(-1)

    if node.name in _DERIVATIVE_OPS:
        return _execute_derivative(node, data_dict, diff_ctx)

    op = DISPATCH.get(node.name)
    if op is None:
        raise ValueError(
            f"Unknown operator '{node.name}'. Known operators: {list(DISPATCH.keys())}"
        )

    child_results = [
        _execute_node(child, data_dict, diff_ctx) for child in node.children
    ]

    if node.arity == 1:
        return op(child_results[0])
    if node.arity == 2:
        return op(child_results[0], child_results[1])
    raise ValueError(f"Unsupported arity {node.arity} for operator '{node.name}'")


def execute_tree(
    tree: Tree,
    data_dict: dict[str, Tensor],
    diff_ctx: DiffContext | None = None,
) -> Tensor:
    with torch.no_grad():
        return _execute_node(tree.root, data_dict, diff_ctx)





def _is_column_valid(col: Tensor) -> bool:
    if not torch.isfinite(col).all():
        return False
    return not torch.norm(col).item() < ZERO_COLUMN_EPS


def _contains_lhs_derivative(node: Node, lhs_axis: str | None) -> bool:
    if lhs_axis is None:
        return False
    if (
        node.name in _DERIVATIVE_OPS
        and len(node.children) == 2
        and node.children[1].is_leaf
        and node.children[1].name == lhs_axis
    ):
        return True
    return any(_contains_lhs_derivative(child, lhs_axis) for child in node.children)


def prune_invalid_terms(
    pde: PDE,
    data_dict: dict[str, Tensor],
    diff_ctx: DiffContext | None = None,
) -> tuple[PDE, Tensor, list[int]]:
    valid_terms, valid_indices = execute_pde(pde, data_dict, diff_ctx=diff_ctx)
    surviving_terms = [pde.terms[i] for i in valid_indices]
    pruned_pde = PDE(terms=[t.copy() for t in surviving_terms])
    return pruned_pde, valid_terms, valid_indices


def execute_pde(
    pde: PDE,
    data_dict: dict[str, Tensor],
    diff_ctx: DiffContext | None = None,
) -> tuple[Tensor, list[int]]:

    if data_dict:
        sample_tensor = next(iter(data_dict.values()))
        n_samples = sample_tensor.numel()
        device = sample_tensor.device
    else:
        n_samples = 0
        device = torch.device("cpu")

    if len(pde.terms) == 0:
        return torch.zeros(n_samples, 0, device=device), []

    valid_columns: list[Tensor] = []
    valid_indices: list[int] = []
    lhs_axis = diff_ctx.lhs_axis if diff_ctx is not None else None

    with torch.no_grad():
        for idx, term in enumerate(pde.terms):
            if _contains_lhs_derivative(term.root, lhs_axis):
                logger.debug(
                    "Discarding term %d due to RHS derivative along lhs_axis",
                    idx,
                )
                continue
            try:
                col = _execute_node(term.root, data_dict, diff_ctx)
            except (KeyError, ValueError, RuntimeError):





                logger.debug("Term %d execution failed, skipping", idx)
                continue

            if _is_column_valid(col):
                valid_columns.append(col)
                valid_indices.append(idx)

    if not valid_columns:
        return torch.zeros(n_samples, 0, device=device), []


    result = torch.stack(valid_columns, dim=1)
    return result, valid_indices





def build_theta(
    valid_terms: Tensor,
    default_terms: Tensor | None = None,
) -> Tensor:
    if default_terms is None:
        return valid_terms

    if default_terms.shape[0] != valid_terms.shape[0]:
        raise ValueError(
            f"Row count mismatch: default_terms has {default_terms.shape[0]} "
            f"rows but valid_terms has {valid_terms.shape[0]} rows"
        )

    return torch.cat([default_terms, valid_terms], dim=1)
