
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from kd.core.expr.executor import _DIFF_PATTERN, _SPECIAL_OPERATORS, _UNITY_TOKEN
from kd.core.expr.naming import parse_compound_derivative




_DIFF_HEAD = _DIFF_PATTERN


@dataclass(frozen=True)
class TermVocabulary:

    fields: frozenset[str]
    coordinates: frozenset[str]

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("TermVocabulary fields must be non-empty")
        if not self.coordinates:
            raise ValueError("TermVocabulary coordinates must be non-empty")
        names = self.fields | self.coordinates
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("TermVocabulary names must not contain an empty name")
        if _UNITY_TOKEN in names:
            raise ValueError(
                f"TermVocabulary name {_UNITY_TOKEN!r} is reserved for unity"
            )
        if any("_" in name for name in names):
            raise ValueError("TermVocabulary names must not contain an underscore")
        overlap = self.fields & self.coordinates
        if overlap:
            raise ValueError(
                f"TermVocabulary fields/coordinates overlap: {sorted(overlap)!r}"
            )


DerivativeMultiindex = tuple[str, tuple[tuple[str, int], ...]]


@dataclass(frozen=True)
class TermFeatures:

    base_fields: frozenset[str]
    coordinate_dependencies: frozenset[str]
    derivative_multiindices: frozenset[DerivativeMultiindex]
    max_total_derivative_order: int
    operators: frozenset[str]


class _FeatureCollector:
    def __init__(self, vocabulary: TermVocabulary) -> None:
        self.vocabulary = vocabulary
        self.base_fields: set[str] = set()
        self.coordinates: set[str] = set()
        self.derivatives: set[DerivativeMultiindex] = set()
        self.operators: set[str] = set()

    def visit(self, node: ast.expr, path: dict[str, int]) -> None:
        if isinstance(node, ast.Name):
            self._visit_name(node.id, path)
            return
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError(f"Unsupported term syntax: {type(node).__name__}")
        head = node.func.id
        match = _DIFF_HEAD.fullmatch(head)
        if match is not None:
            self._visit_diff(node, head, match, path)
            return
        if head in _SPECIAL_OPERATORS:
            raise ValueError(
                f"Derivative-bearing operator {head!r} cannot be decomposed "
                "into per-axis order features"
            )
        self.operators.add(head)
        for argument in node.args:
            self.visit(argument, path)

    def _visit_diff(
        self,
        node: ast.Call,
        head: str,
        match: re.Match[str],
        path: dict[str, int],
    ) -> None:
        if len(node.args) != 1:
            raise ValueError(
                f"Open-form derivative {head} requires exactly one argument"
            )
        axis = match.group(2)
        if axis not in self.vocabulary.coordinates:
            raise ValueError(f"Derivative axis {axis!r} is outside the vocabulary")
        order = int(match.group(1) or "1")
        if order < 1:

            raise ValueError(f"Diff order must be >= 1, got {order} in {head!r}")
        nested_path = dict(path)
        nested_path[axis] = nested_path.get(axis, 0) + order
        self.visit(node.args[0], nested_path)

    def _visit_name(self, name: str, path: dict[str, int]) -> None:
        if name == _UNITY_TOKEN:
            if path:
                raise ValueError(
                    f"Reserved unity token {_UNITY_TOKEN!r} cannot appear "
                    "under a derivative path"
                )
            return
        if name in self.vocabulary.coordinates:
            self.coordinates.add(name)
            return
        if name in self.vocabulary.fields:
            self._record_field(name, (), path)
            return
        parsed = parse_compound_derivative(
            name,
            known_fields=set(self.vocabulary.fields),
            known_axes=set(self.vocabulary.coordinates),
        )
        if parsed is None:
            raise ValueError(f"Unresolvable term symbol {name!r}")
        field, segments = parsed
        self._record_field(field, tuple(segments), path)

    def _record_field(
        self,
        field: str,
        segments: tuple[tuple[str, int], ...],
        path: dict[str, int],
    ) -> None:
        self.base_fields.add(field)
        orders = dict(path)
        for axis, order in segments:
            orders[axis] = orders.get(axis, 0) + order
        multiindex = tuple(
            sorted((axis, order) for axis, order in orders.items() if order != 0)
        )
        if multiindex:
            self.derivatives.add((field, multiindex))

    def features(self) -> TermFeatures:
        max_order = max(
            (
                sum(order for _axis, order in multiindex)
                for _field, multiindex in self.derivatives
            ),
            default=0,
        )
        return TermFeatures(
            base_fields=frozenset(self.base_fields),
            coordinate_dependencies=frozenset(self.coordinates),
            derivative_multiindices=frozenset(self.derivatives),
            max_total_derivative_order=max_order,
            operators=frozenset(self.operators),
        )


def analyze_term(term_ir: str, vocabulary: TermVocabulary) -> TermFeatures:
    from kd.core.equation.signature import law_term_key

    normalized = law_term_key(term_ir)
    parsed = ast.parse(normalized, mode="eval")
    collector = _FeatureCollector(vocabulary)
    collector.visit(parsed.body, {})
    return collector.features()


__all__ = ["TermFeatures", "TermVocabulary", "analyze_term"]
