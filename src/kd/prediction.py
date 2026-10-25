
from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import cast

import torch
from torch import Tensor

from kd.core.equation._sketch_terms import lower_term
from kd.core.expr.registry import FunctionRegistry

Derivative = Callable[[Tensor, str, int], Tensor]


def _evaluate(
    node: ast.AST,
    symbols: Mapping[str, Tensor],
    template: Tensor,
    registry: FunctionRegistry,
    differentiate: Derivative | None,
) -> Tensor:
    if isinstance(node, ast.Name):
        if node.id == "one":
            return torch.ones_like(template)
        if node.id not in symbols:
            raise ValueError(f"unknown symbol {node.id!r}")
        return symbols[node.id]
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        literal = float(cast("int | float", node.value))
        if not math.isfinite(literal):
            raise ValueError("numeric literals must be finite")
        return torch.full_like(template, literal)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate(node.operand, symbols, template, registry, differentiate)
        return -value if isinstance(node.op, ast.USub) else value
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.keywords
    ):
        raise ValueError("unsupported expression syntax")
    name = node.func.id
    derivative = re.fullmatch(r"diff([1-9][0-9]*)?_([a-zA-Z][a-zA-Z0-9]*)", name)
    if derivative:
        if differentiate is None or len(node.args) != 1:
            raise ValueError(f"{name} requires a derivative callback and one argument")
        value = _evaluate(node.args[0], symbols, template, registry, differentiate)
        return differentiate(value, derivative[2], int(derivative[1] or 1))
    if not registry.has(name) or name == "lap":
        raise ValueError(f"unsupported operator {name!r}")
    if len(node.args) != registry.get_arity(name):
        raise ValueError(f"wrong argument count for {name!r}")
    arguments = [
        _evaluate(arg, symbols, template, registry, differentiate) for arg in node.args
    ]
    return cast("Tensor", registry.get_func(name)(*arguments))


def predict_terms(
    terms: Sequence[str],
    coefficients: Sequence[float],
    symbols: Mapping[str, Tensor],
    *,
    differentiate: Derivative | None = None,
) -> Tensor:
    if len(terms) != len(coefficients):
        raise ValueError("terms and coefficients must have matching lengths")
    if not symbols:
        raise ValueError("symbols must provide a sample shape")
    template = next(iter(symbols.values()))
    for value in symbols.values():
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("symbols must be finite floating tensors")
        if (value.shape, value.dtype, value.device) != (
            template.shape,
            template.dtype,
            template.device,
        ):
            raise ValueError("symbol shape, dtype and device must match")
    if any(not math.isfinite(c) for c in coefficients):
        raise ValueError("coefficients must be finite")
    registry = FunctionRegistry.create_default()
    result = torch.zeros_like(template)
    for term, coefficient in zip(terms, coefficients, strict=True):
        try:
            node = ast.parse(lower_term(term), mode="eval").body
        except SyntaxError as exc:
            raise ValueError("invalid expression syntax") from exc
        result = result + coefficient * _evaluate(
            node, symbols, template, registry, differentiate
        )
    return result


__all__ = ["predict_terms"]
