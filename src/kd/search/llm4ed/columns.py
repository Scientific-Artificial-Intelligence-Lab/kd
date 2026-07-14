
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import torch

from kd.core.expr.executor import PythonExecutor
from kd.core.expr.registry import FunctionRegistry

if TYPE_CHECKING:
    from kd.core.executor.context import ExecutionContext
    from kd.search.llm4ed.parse import ParsedEquation

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]


def _raw_recip(value: torch.Tensor) -> torch.Tensor:
    return 1.0 / value


def _raw_square(value: torch.Tensor) -> torch.Tensor:
    return value**2


def _raw_cube(value: torch.Tensor) -> torch.Tensor:
    return value**3


def _neg(value: torch.Tensor) -> torch.Tensor:
    return -value


def build_llm4ed_registry() -> FunctionRegistry:
    reg = FunctionRegistry()
    reg.register("add", torch.add, arity=2, commutative=True)
    reg.register("mul", torch.mul, arity=2, commutative=True)
    reg.register("sub", torch.sub, arity=2, commutative=False)
    reg.register("div", torch.div, arity=2, commutative=False)
    reg.register("neg", _neg, arity=1)
    reg.register("n2", _raw_square, arity=1)
    reg.register("n3", _raw_cube, arity=1)
    reg.register("recip", _raw_recip, arity=1)
    return reg




_REGISTRY = build_llm4ed_registry()
_EXECUTOR = PythonExecutor(_REGISTRY)


class _ColumnDataset:

    def __init__(self, names: tuple[str, ...]) -> None:



        self.fields: dict[str, None] = dict.fromkeys(names)
        self.axes: dict[str, None] | None = None


class _ColumnContext:

    def __init__(self, columns: Mapping[str, torch.Tensor]) -> None:
        self._columns = columns
        self.dataset = _ColumnDataset(tuple(columns.keys()))
        self.device = torch.device("cpu")

    def get_variable(self, name: str) -> torch.Tensor:
        try:
            return self._columns[name]
        except KeyError as exc:
            raise KeyError(f"Unknown operand: {name}") from exc

    def get_derivative(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:


        raise KeyError("llm4ed columns provide no on-the-fly derivatives")

    def get_constant(self, name: str) -> float:
        raise KeyError(f"Unknown constant: {name}")


@dataclass(frozen=True)
class ColumnResult:

    valid: bool
    columns: tuple[FloatArray, ...]
    term_strs: tuple[str, ...]
    error: str | None


def evaluate_ir_column(
    ir: str, features: Mapping[str, FloatArray]
) -> FloatArray:
    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(col, dtype=np.float64))
        for name, col in features.items()
    }
    context = _ColumnContext(tensors)
    result = _EXECUTOR.execute(ir, cast("ExecutionContext", context))
    column = result.value.detach().cpu().numpy().reshape(-1)
    return np.ascontiguousarray(column, dtype=np.float64)


def build_columns(
    parsed: ParsedEquation, features: Mapping[str, FloatArray]
) -> ColumnResult:
    columns: list[FloatArray] = []
    term_strs: list[str] = []
    nonfinite_terms: list[str] = []
    for term in parsed.terms:
        column = evaluate_ir_column(term.ir, features)
        columns.append(column)
        term_strs.append(term.term_str)
        if not np.isfinite(column).all():
            nonfinite_terms.append(term.term_str)

    if nonfinite_terms:
        message = (
            f"non-finite column(s) from raw evaluation: {nonfinite_terms} "
            "(EDL-faithful drop, D8)"
        )
        logger.debug("dropping candidate %r: %s", parsed.equation, message)
        return ColumnResult(
            valid=False,
            columns=tuple(columns),
            term_strs=tuple(term_strs),
            error=message,
        )

    return ColumnResult(
        valid=True,
        columns=tuple(columns),
        term_strs=tuple(term_strs),
        error=None,
    )
