
from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import numpy.typing as npt
import torch
from scipy.optimize import minimize
from torch import Tensor

from kd.core.expr.columns import ColumnContext as _ColumnContext
from kd.core.expr.executor import PythonExecutor
from kd.core.expr.registry import FunctionRegistry
from kd.core.metrics import nmse

if TYPE_CHECKING:
    from kd.core.executor.context import ExecutionContext

FloatArray = npt.NDArray[np.float64]

DEFAULT_STARTS: Final[tuple[float, ...]] = (1.0, -1.0, 0.1)


_GTOL: Final[float] = 1e-10


_PLACEHOLDER: Final = re.compile(r"\bc[1-9][0-9]*\b")




_SCALE_ARGUMENTS: Final[Mapping[str, frozenset[int]]] = {
    "mul": frozenset({0, 1}),
    "neg": frozenset({0}),
    "div": frozenset({0}),
}





_IMPROVEMENT: Final[float] = 1.0 - 1e-9
_FLOOR: Final[float] = 1e-20


_Literal = tuple[ast.expr, float]


def placeholder_names(term_irs: Sequence[str]) -> list[str]:
    found = {name for term_ir in term_irs for name in _PLACEHOLDER.findall(term_ir)}
    return sorted(found, key=lambda name: int(name[1:]))


def substitute_placeholders(term_ir: str, values: Mapping[str, float]) -> str:
    return _PLACEHOLDER.sub(lambda match: repr(float(values[match.group()])), term_ir)


class _Projection:

    def __init__(
        self,
        term_irs: Sequence[str],
        names: Sequence[str],
        columns: Mapping[str, Tensor],
        target: Tensor,
        registry: FunctionRegistry,
    ) -> None:
        self._term_irs = tuple(term_irs)
        self._names = tuple(names)
        self._columns = {
            name: column.to(torch.float64) for name, column in columns.items()
        }
        target64 = target.to(torch.float64)
        self._unity = torch.ones_like(target64)
        self._target = target64.detach().cpu().numpy()
        norm = float(self._target @ self._target)
        self._scale = norm if norm > 0.0 else 1.0
        self._executor = PythonExecutor(registry)

    def __call__(self, values: FloatArray) -> float:
        if not np.isfinite(values).all():
            return math.inf
        design = self._design(values)
        if not np.isfinite(design).all():
            return math.inf
        residual = self._solve(design)[1]
        objective = float(residual @ residual) / self._scale
        return objective if math.isfinite(objective) else math.inf

    def outer_fit(self, values: FloatArray) -> tuple[FloatArray, float]:
        coefficients, residual = self._solve(self._design(values))
        mse = float(np.mean(residual**2))
        return coefficients, nmse(mse, float(np.var(self._target)))

    def _solve(self, design: FloatArray) -> tuple[FloatArray, FloatArray]:
        coefficients = np.linalg.lstsq(design, self._target, rcond=None)[0]
        return coefficients, self._target - design @ coefficients

    def _design(self, values: FloatArray) -> FloatArray:
        bound = {
            name: torch.tensor(
                float(value), dtype=torch.float64, device=self._unity.device
            )
            for name, value in zip(self._names, values, strict=True)
        }
        context = cast(
            "ExecutionContext", _ColumnContext({**self._columns, **bound}, self._unity)
        )
        columns = [self._executor.execute(ir, context).value for ir in self._term_irs]
        for ir, column in zip(self._term_irs, columns, strict=True):
            if column.dim() == 0:
                raise ValueError(
                    f"Term {ir!r} evaluates to a scalar, not a column: every term "
                    "must reference a column or `one`"
                )
        return torch.stack(columns, dim=1).detach().cpu().numpy()


def _fit_from(projection: _Projection, start: float, size: int) -> FloatArray:
    return _descend(projection, np.full(size, start, dtype=np.float64))


def _descend(projection: _Projection, initial: FloatArray) -> FloatArray:

    with np.errstate(all="ignore"):
        result = minimize(projection, initial, method="BFGS", options={"gtol": _GTOL})
    values = np.asarray(result.x, dtype=np.float64)
    return np.where(np.isfinite(values), values, initial)


def fit_placeholders(
    term_irs: Sequence[str],
    columns: Mapping[str, Tensor],
    target: Tensor,
    *,
    registry: FunctionRegistry,
    starts: Sequence[float] = DEFAULT_STARTS,
) -> tuple[list[str], FloatArray]:
    names = placeholder_names(term_irs)
    if not names:
        return list(term_irs), np.empty(0, dtype=np.float64)
    projection = _Projection(term_irs, names, columns, target, registry)
    candidates = [_fit_from(projection, start, len(names)) for start in starts]

    best = min(candidates, key=projection)
    values = dict(zip(names, best.tolist(), strict=True))
    return [substitute_placeholders(ir, values) for ir in term_irs], best


@dataclass(frozen=True)
class ConstantRefit:

    terms: tuple[str, ...]
    constants: dict[str, float]
    initial: dict[str, float]
    coefficients: tuple[float, ...]
    nmse: float


def literals_to_placeholders(
    term_irs: Sequence[str],
) -> tuple[list[str], dict[str, float]]:
    placeholder_irs: list[str] = []
    initial: dict[str, float] = {}
    for term_ir in term_irs:

        encoded = term_ir.encode()
        line_starts = _line_starts(encoded)
        pieces: list[bytes] = []
        end = 0
        for node, value in _refittable_literals(_parse(term_ir), outer=True):
            start = line_starts[node.lineno - 1] + node.col_offset
            end_line = cast("int", node.end_lineno)
            stop = line_starts[end_line - 1] + cast("int", node.end_col_offset)
            name = f"c{len(initial) + 1}"
            initial[name] = value
            pieces += [encoded[end:start], name.encode()]
            end = stop
        pieces.append(encoded[end:])
        placeholder_irs.append(b"".join(pieces).decode())
    if initial:
        _refuse_placeholder_names(term_irs)
    return placeholder_irs, initial


def _parse(term_ir: str) -> ast.expr:
    try:
        return ast.parse(term_ir, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid term IR {term_ir!r}: {exc.msg}") from exc


def _line_starts(encoded: bytes) -> list[int]:
    starts = [0]
    for line in encoded.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def _refuse_placeholder_names(term_irs: Sequence[str]) -> None:
    for term_ir in term_irs:
        found = _PLACEHOLDER.search(term_ir)
        if found is not None:
            raise ValueError(
                f"Term {term_ir!r} names {found.group()!r}, which clashes with the "
                "placeholder names c<k> the literals become; rename that column"
            )


def _refittable_literals(node: ast.expr, *, outer: bool) -> list[_Literal]:
    value = _literal_value(node)
    if value is not None:
        return [] if outer else [(node, value)]
    if isinstance(node, ast.Name):
        return []
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.keywords
    ):
        raise ValueError(f"Unsupported term IR syntax: {ast.unparse(node)}")
    scale = _SCALE_ARGUMENTS.get(node.func.id, frozenset())
    return [
        literal
        for index, argument in enumerate(node.args)
        if not (node.func.id == "mul" and _literal_value(argument) == -1.0)
        for literal in _refittable_literals(argument, outer=outer and index in scale)
    ]


def _literal_value(node: ast.expr) -> float | None:
    sign = 1.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node, sign = node.operand, -1.0
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return sign * float(node.value)
    return None


def refit_literals(
    term_irs: Sequence[str],
    columns: Mapping[str, Tensor],
    target: Tensor,
    *,
    registry: FunctionRegistry,
    starts: Sequence[float] = DEFAULT_STARTS,
) -> ConstantRefit:
    _check_target(target)
    placeholder_irs, initial = literals_to_placeholders(term_irs)
    names = list(initial)
    projection = _Projection(placeholder_irs, names, columns, target, registry)
    literals = np.array(list(initial.values()), dtype=np.float64)
    candidates = [literals]
    if names:
        candidates.append(_descend(projection, literals))
        candidates += [_fit_from(projection, start, len(names)) for start in starts]
    best, objective = _select(projection, candidates)
    if not math.isfinite(objective):
        raise _no_finite_fit(initial)
    coefficients, fit_nmse = projection.outer_fit(best)
    constants = dict(zip(names, best.tolist(), strict=True))
    terms = (
        tuple(term_irs)
        if best is literals
        else tuple(substitute_placeholders(ir, constants) for ir in placeholder_irs)
    )
    return ConstantRefit(
        terms=terms,
        constants=constants,
        initial=initial,
        coefficients=tuple(coefficients.tolist()),
        nmse=fit_nmse,
    )


def _check_target(target: Tensor) -> None:
    if target.numel() == 0:
        raise ValueError("Cannot refit on zero rows: the target is empty")
    bad = int((~torch.isfinite(target)).sum())
    if bad:
        raise ValueError(f"The target has {bad} non-finite value(s); no fit is finite")


def _select(
    projection: _Projection, candidates: Sequence[FloatArray]
) -> tuple[FloatArray, float]:
    best, best_objective = candidates[0], projection(candidates[0])
    for candidate in candidates[1:]:
        objective = projection(candidate)
        if objective < best_objective * _IMPROVEMENT - _FLOOR:
            best, best_objective = candidate, objective
    return best, best_objective


def _no_finite_fit(initial: Mapping[str, float]) -> ValueError:
    bad = {name: value for name, value in initial.items() if not math.isfinite(value)}
    if bad:
        return ValueError(
            f"No finite fit: the literals {bad} are not finite and no start "
            "gives a finite fit"
        )
    return ValueError(
        "No finite fit: every candidate gives a non-finite column (a division by "
        "zero, a domain error or an overflow on some row)"
    )
