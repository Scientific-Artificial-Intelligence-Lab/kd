
from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.search.discover.core.tree import trim_to_natural
from kd.search.discover.evaluation.dedup import Deduplicator
from kd.search.discover.tokens.library import Library, Token

FloatArray = NDArray[np.float64]
TokenLike = Token | str
_CONST_NAME = "const"
_BINARY_OPERATORS = frozenset({"add", "sub", "mul", "div"})


class _TermEvaluator(Protocol):
    @property
    def lhs_target(self) -> Tensor: ...

    def build_theta_matrix(
        self, terms: list[str], *, skip_invalid: bool = False
    ) -> tuple[Tensor, list[str]]: ...

    def evaluate_terms(
        self, terms: list[str], *, skip_invalid: bool = False
    ) -> EvaluationResult: ...

    def invalidate_term_cache(self) -> None: ...


def count_tokens(expression: str) -> int:
    try:
        parsed = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid IR syntax: {exc}") from exc
    return _count_ast_node(parsed.body)


def _count_ast_node(root: ast.expr) -> int:
    count = 0
    stack = [root]
    while stack:
        node = stack.pop()
        count += 1
        if isinstance(node, ast.Name):
            continue
        if isinstance(node, ast.Constant) and _is_number(node.value):
            continue
        if _is_negative_number(node):
            continue
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError(f"Unsupported IR syntax: {type(node).__name__}")
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed in IR")
        stack.extend(node.args)
    return count


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_negative_number(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and _is_number(node.operand.value)
    )


def fit_constants(
    tokens: Sequence[TokenLike],
    columns: Mapping[str, np.ndarray],
    target: np.ndarray,
) -> FloatArray:
    names = [_token_name(token) for token in tokens]
    n_constants = names.count(_CONST_NAME)
    if n_constants == 0:
        return np.empty(0, dtype=np.float64)
    feature_columns = {
        name: np.asarray(column, dtype=np.float64).reshape(-1)
        for name, column in columns.items()
    }
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)

    def objective(constants: FloatArray) -> float:
        with np.errstate(all="ignore"):
            candidate = _execute(names, feature_columns, constants)
            if (
                candidate.shape != target_values.shape
                or not np.isfinite(candidate).all()
            ):
                return math.inf
            denominator = float(np.dot(candidate, candidate))
            if not math.isfinite(denominator) or denominator == 0.0:
                return math.inf
            scale = float(np.dot(candidate, target_values)) / denominator
            residual = target_values - scale * candidate
            loss = float(np.mean(np.square(residual)))
        return loss if math.isfinite(loss) else math.inf

    with np.errstate(all="ignore"):
        result = minimize(
            objective,
            np.ones(n_constants, dtype=np.float64),
            method="BFGS",
        )
    constants = np.asarray(result.x, dtype=np.float64)




    constants[~np.isfinite(constants)] = 1.0
    return constants


def _execute(
    names: Sequence[str],
    columns: Mapping[str, FloatArray],
    constants: FloatArray,
) -> FloatArray:
    const_positions = _constant_positions(names)
    stack: list[FloatArray] = []
    for position in range(len(names) - 1, -1, -1):
        name = names[position]
        if name == _CONST_NAME:
            stack.append(np.asarray(constants[const_positions[position]]))
        elif name in columns:
            stack.append(columns[name])
        elif name in _BINARY_OPERATORS:
            if len(stack) < 2:
                raise ValueError("Incomplete tabular token traversal")
            left, right = stack.pop(), stack.pop()
            stack.append(_apply_binary(name, left, right))
        else:
            raise ValueError(f"Unsupported tabular token: {name!r}")
    if len(stack) != 1:
        raise ValueError("Tabular token traversal contains trailing tokens")
    return np.asarray(stack[0], dtype=np.float64)


def _constant_positions(names: Sequence[str]) -> dict[int, int]:
    positions: dict[int, int] = {}
    next_index = 0
    for position, name in enumerate(names):
        if name == _CONST_NAME:
            positions[position] = next_index
            next_index += 1
    return positions


def _apply_binary(name: str, left: FloatArray, right: FloatArray) -> FloatArray:
    if name == "add":
        return np.add(left, right)
    if name == "sub":
        return np.subtract(left, right)
    if name == "mul":
        return np.multiply(left, right)
    return np.divide(left, right)


def _token_name(token: TokenLike) -> str:
    return token.name if isinstance(token, Token) else token


def _render_tokens(tokens: Sequence[TokenLike], constants: FloatArray) -> str:
    names = [_token_name(token) for token in tokens]
    const_positions = _constant_positions(names)
    stack: list[str] = []
    for position in range(len(names) - 1, -1, -1):
        name = names[position]
        if name == _CONST_NAME:
            stack.append(repr(float(constants[const_positions[position]])))
        elif name in _BINARY_OPERATORS:
            if len(stack) < 2:
                raise ValueError("Incomplete tabular token traversal")
            left, right = stack.pop(), stack.pop()
            stack.append(f"{name}({left},{right})")
        else:
            stack.append(name)
    if len(stack) != 1:
        raise ValueError("Tabular token traversal contains trailing tokens")
    return stack[0]


class TabularCandidateScorer:

    def __init__(self, evaluator: _TermEvaluator, *, lhs_name: str) -> None:
        self._evaluator = evaluator
        self._lhs_name = lhs_name

    @property
    def lhs_target(self) -> Tensor:
        return self._evaluator.lhs_target

    def evaluate_expression(self, expression: str) -> EvaluationResult:
        result = self._evaluator.evaluate_terms([expression], skip_invalid=False)
        if not result.is_valid:





            return replace(result, lhs_name=self._lhs_name)
        return replace(
            result,
            complexity=count_tokens(expression),
            lhs_name=self._lhs_name,
        )

    def invalidate_term_cache(self) -> None:
        self._evaluator.invalidate_term_cache()


@dataclass(frozen=True, slots=True)
class _FrontEntry:
    expression: str
    complexity: int
    loss: float
    scale: float


class ParetoTracker:

    def __init__(self) -> None:
        self._entries: list[_FrontEntry] = []

    def offer(
        self, expression: str, complexity: int, nmse: float, scale: float
    ) -> None:
        if not math.isfinite(nmse) or not math.isfinite(scale):
            return
        candidate = _FrontEntry(expression, complexity, nmse, scale)
        if any(_dominates(entry, candidate) for entry in self._entries):
            return
        self._entries = [
            entry for entry in self._entries if not _dominates(candidate, entry)
        ]
        self._entries.append(candidate)

    def entries(self) -> list[_FrontEntry]:
        return sorted(self._entries, key=lambda entry: entry.complexity)

    def state(self) -> list[dict[str, object]]:
        return [
            {
                "expression": entry.expression,
                "complexity": entry.complexity,
                "loss": entry.loss,
                "scale": entry.scale,
            }
            for entry in self.entries()
        ]

    @classmethod
    def from_state(cls, state: Sequence[Mapping[str, object]]) -> ParetoTracker:
        tracker = cls()
        for row in state:
            tracker.offer(
                expression=cast(str, row["expression"]),
                complexity=cast(int, row["complexity"]),
                nmse=cast(float, row["loss"]),
                scale=cast(float, row["scale"]),
            )
        return tracker


def _dominates(left: _FrontEntry, right: _FrontEntry) -> bool:
    return left.complexity <= right.complexity and left.loss <= right.loss


class TabularDeduplicator(Deduplicator):

    def __init__(
        self,
        library: Library,
        columns: Mapping[str, np.ndarray],
        target: np.ndarray,
    ) -> None:
        super().__init__(library)
        self._library = library
        self._columns = {
            name: np.asarray(value).copy() for name, value in columns.items()
        }
        self._target = np.asarray(target).copy()

    def deduplicate(self, actions: np.ndarray) -> tuple[list[str], np.ndarray]:
        rows = np.asarray(actions, dtype=np.int32)
        if rows.ndim != 2:
            raise ValueError(f"actions must be 2D (B, L), got ndim={rows.ndim}")
        unique: list[str] = []
        positions: dict[tuple[int, ...], int] = {}
        scatter = np.empty(rows.shape[0], dtype=np.int64)
        for row_index, row in enumerate(rows):
            trimmed = trim_to_natural(row, self._library)
            traversal = tuple(int(index) for index in trimmed)
            if traversal in positions:
                scatter[row_index] = positions[traversal]
                continue
            tokens = [self._library[index] for index in traversal]
            constants = fit_constants(tokens, self._columns, self._target)
            expression = _render_tokens(tokens, constants)
            positions[traversal] = len(unique)
            unique.append(expression)
            scatter[row_index] = positions[traversal]
        return unique, scatter


__all__ = [
    "ParetoTracker",
    "TabularCandidateScorer",
    "count_tokens",
    "fit_constants",
]
