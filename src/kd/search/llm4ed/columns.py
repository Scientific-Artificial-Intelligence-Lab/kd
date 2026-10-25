
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import numpy.typing as npt
import torch

from kd.core.expr.executor import PythonExecutor
from kd.core.expr.registry import PLAIN_FUNCTION_NAMES, FunctionRegistry

if TYPE_CHECKING:
    from kd.core.executor.context import ExecutionContext
    from kd.search.llm4ed.parse import ParsedEquation

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]




FUNCTION_NAMES: Final[frozenset[str]] = PLAIN_FUNCTION_NAMES


def build_llm4ed_registry() -> FunctionRegistry:
    return FunctionRegistry.create_plain()




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

    def unity_column(self) -> torch.Tensor:
        return torch.ones_like(next(iter(self._columns.values())))

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


def evaluate_ir_column(ir: str, features: Mapping[str, FloatArray]) -> FloatArray:
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
