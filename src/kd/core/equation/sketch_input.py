
from __future__ import annotations

import functools
import json
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from kd.core.equation._sketch_serialize import (
    as_mapping,
    as_sequence,
    lhs_from_dict,
    lhs_to_dict,
    optional_int,
    optional_names,
    policy_from_dict,
    policy_to_dict,
    required_float,
    required_int,
    required_str,
    strict,
    vocabulary_from_dict,
    vocabulary_to_dict,
)
from kd.core.equation._sketch_terms import shorthand_term
from kd.core.equation.sketch import (
    AnchoredTerm,
    PinnedTerm,
    SketchMatchPolicy,
    TermConstraint,
    TermHole,
    sketch_from_dict,
    sketch_to_dict,
)
from kd.core.equation.sketch import Sketch as CompleteSketch
from kd.core.equation.sketch_builder import (
    dataset_sketch,
    dataset_vocabulary,
    term_refusal,
)
from kd.core.equation.types import LhsSpec
from kd.core.expr.executor import _SPECIAL_OPERATORS
from kd.core.expr.registry import FunctionRegistry
from kd.core.expr.term_features import TermVocabulary
from kd.core.platform.builder import resolve_lhs_defaults

if TYPE_CHECKING:
    from kd.data.schema import PDEDataset

_TYPED_CLAUSES = (PinnedTerm, AnchoredTerm, TermHole)

SKETCH_SPEC_SCHEMA_TAG = "kd-sketch-spec-v1"

_SPEC_KEYS = frozenset(
    {
        "schema",
        "pinned",
        "anchored",
        "holes",
        "lhs_spec",
        "vocabulary",
        "match_policy",
    }
)
_SPEC_PIN_KEYS = frozenset({"term", "value"})
_SPEC_HOLE_KEYS = frozenset(
    {"max_terms", "min_terms", "max_deriv_order", "operators", "fields"}
)


@functools.cache
def _law_operator_names() -> frozenset[str]:
    names = frozenset(FunctionRegistry.create_default().list_names())
    return names - {"div"} - _SPECIAL_OPERATORS


def _require_count(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def _names(name: str, values: Collection[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise TypeError(
            f"{name} takes a collection of names such as [{values!r}], not one string"
        )
    return frozenset(values)


@dataclass(frozen=True, init=False)
class Hole:

    max_terms: int
    min_terms: int
    max_deriv_order: int | None
    operators: frozenset[str] | None
    fields: frozenset[str] | None

    def __init__(
        self,
        max_terms: int,
        min_terms: int = 1,
        max_deriv_order: int | None = None,
        operators: Collection[str] | None = None,
        fields: Collection[str] | None = None,
    ) -> None:
        _require_count("max_terms", max_terms, 1)
        _require_count("min_terms", min_terms, 0)
        if min_terms > max_terms:
            raise ValueError(
                f"min_terms must be at most max_terms, got {min_terms} > {max_terms}"
            )
        if max_deriv_order is not None:
            _require_count("max_deriv_order", max_deriv_order, 0)
        operator_names = _names("operators", operators)
        field_names = _names("fields", fields)
        if field_names is not None and not field_names:
            raise ValueError("fields must name at least one field; None admits all")
        if operator_names is not None:
            unknown = sorted(operator_names - _law_operator_names())
            if unknown:
                raise ValueError(
                    f"operators {unknown} are not law feature names; use "
                    f"{sorted(_law_operator_names())} (division is 'mul' "
                    "with 'recip')"
                )
        object.__setattr__(self, "max_terms", max_terms)
        object.__setattr__(self, "min_terms", min_terms)
        object.__setattr__(self, "max_deriv_order", max_deriv_order)
        object.__setattr__(self, "operators", operator_names)
        object.__setattr__(self, "fields", field_names)


def _term_hole(hole: Hole, index: int) -> TermHole:
    return TermHole(
        id=f"hole_{index}",
        min_count=hole.min_terms,
        max_count=hole.max_terms,
        constraint=TermConstraint(
            max_deriv_order=hole.max_deriv_order,
            operators=hole.operators,
            fields=hole.fields,
            axes=None,
        ),
    )


def _bound_term(
    position: str, text: str, vocabulary: TermVocabulary, registry: FunctionRegistry
) -> str:
    try:
        term = shorthand_term(text)
    except ValueError as exc:
        raise ValueError(f"{position}: invalid term {text!r}: {exc}") from exc
    reason = term_refusal(term, vocabulary, registry)
    if reason is not None:
        raise ValueError(
            f"{position}: term {text!r} cannot be evaluated on fields "
            f"{sorted(vocabulary.fields)} and axes "
            f"{sorted(vocabulary.coordinates)}: {reason}"
        )
    return term


@dataclass(frozen=True)
class SketchSpec:

    pinned: tuple[tuple[str, float], ...] = ()
    anchored: tuple[str, ...] = ()
    holes: tuple[Hole, ...] = ()
    lhs_spec: LhsSpec | None = None
    vocabulary: TermVocabulary | None = None
    match_policy: SketchMatchPolicy | None = None

    def __post_init__(self) -> None:
        if not self.pinned and not self.anchored and not self.holes:
            raise ValueError(
                "a Sketch must not be fully empty: give pinned, anchored or holes"
            )

    def resolve(self, dataset: PDEDataset) -> CompleteSketch:
        vocabulary = dataset_vocabulary(dataset, self.vocabulary)
        registry = FunctionRegistry.create_default()
        pins = [
            PinnedTerm(
                _bound_term(f"pinned[{text!r}]", text, vocabulary, registry), value
            )
            for text, value in self.pinned
        ]
        anchors = tuple(
            AnchoredTerm(_bound_term(f"anchored[{index}]", text, vocabulary, registry))
            for index, text in enumerate(self.anchored)
        )
        holes = tuple(_term_hole(hole, index) for index, hole in enumerate(self.holes))
        return dataset_sketch(
            dataset,
            vocabulary=vocabulary,
            pinned=tuple(sorted(pins, key=lambda pin: pin.term_ir)),
            anchored=anchors,
            holes=holes,
            lhs_spec=self.lhs_spec,
            match_policy=self.match_policy,
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        save_sketch(self, path)


def _holds_typed(value: object) -> bool:
    return (
        isinstance(value, Collection)
        and not isinstance(value, Mapping | str)
        and any(isinstance(item, _TYPED_CLAUSES) for item in value)
    )


def _holds_shorthand(name: str, value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(value)
    if not isinstance(value, Collection) or isinstance(value, str):
        return value is not None
    item_type = Hole if name == "holes" else str
    return any(isinstance(item, item_type) for item in value)


@overload
def Sketch(
    *,
    lhs_spec: LhsSpec,
    vocabulary: TermVocabulary,
    pinned: tuple[PinnedTerm, ...],
    anchored: tuple[AnchoredTerm, ...],
    holes: tuple[TermHole, ...],
    match_policy: SketchMatchPolicy,
) -> CompleteSketch: ...


@overload
def Sketch(
    *,
    pinned: Mapping[str, float] | None = None,
    anchored: Sequence[str] | None = None,
    holes: Sequence[Hole] | None = None,
    lhs_spec: LhsSpec | None = None,
    vocabulary: TermVocabulary | None = None,
    match_policy: SketchMatchPolicy | None = None,
) -> SketchSpec: ...


def Sketch(
    *,
    pinned: Any = None,
    anchored: Any = None,
    holes: Any = None,
    lhs_spec: Any = None,
    vocabulary: Any = None,
    match_policy: Any = None,
) -> CompleteSketch | SketchSpec:
    clauses = {"pinned": pinned, "anchored": anchored, "holes": holes}
    typed = [name for name, value in clauses.items() if _holds_typed(value)]
    shorthand = [
        name for name, value in clauses.items() if _holds_shorthand(name, value)
    ]
    if typed and shorthand:
        raise TypeError(
            f"kd.Sketch got typed clauses in {typed} beside shorthand clauses in "
            f"{shorthand}; write every clause in one form"
        )
    if typed:
        return _typed_sketch(
            lhs_spec, vocabulary, pinned, anchored, holes, match_policy
        )
    if isinstance(anchored, str):
        raise TypeError(
            f"anchored takes a list of term strings such as [{anchored!r}], "
            "not one string"
        )
    return SketchSpec(
        pinned=() if pinned is None else tuple(dict(pinned).items()),
        anchored=() if anchored is None else tuple(anchored),
        holes=() if holes is None else tuple(holes),
        lhs_spec=lhs_spec,
        vocabulary=vocabulary,
        match_policy=match_policy,
    )


def _typed_sketch(
    lhs_spec: Any,
    vocabulary: Any,
    pinned: Any,
    anchored: Any,
    holes: Any,
    match_policy: Any,
) -> CompleteSketch:
    fields = {
        "lhs_spec": lhs_spec,
        "vocabulary": vocabulary,
        "pinned": pinned,
        "anchored": anchored,
        "holes": holes,
        "match_policy": match_policy,
    }
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        raise TypeError(f"the typed kd.Sketch takes all six fields; missing {missing}")
    return CompleteSketch(**fields)


def save_sketch(
    sketch: CompleteSketch | SketchSpec, path: str | os.PathLike[str]
) -> None:
    if isinstance(sketch, SketchSpec):
        payload = _spec_to_dict(sketch)
    else:
        payload = sketch_to_dict(sketch)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")


@overload
def load_sketch(
    source: str | os.PathLike[str] | Mapping[str, object],
    *,
    dataset: None = None,
) -> CompleteSketch | SketchSpec: ...


@overload
def load_sketch(
    source: str | os.PathLike[str] | Mapping[str, object],
    *,
    dataset: PDEDataset,
) -> CompleteSketch: ...


def load_sketch(
    source: str | os.PathLike[str] | Mapping[str, object],
    *,
    dataset: PDEDataset | None = None,
) -> CompleteSketch | SketchSpec:
    payload = as_mapping(_read_payload(source), "sketch file")
    if payload.get("schema") == SKETCH_SPEC_SCHEMA_TAG:
        spec = _spec_from_dict(payload)
        return spec if dataset is None else spec.resolve(_evolution(dataset))
    sketch = sketch_from_dict(payload)
    return sketch if dataset is None else _bind_complete(sketch, _evolution(dataset))


def _read_payload(source: str | os.PathLike[str] | Mapping[str, object]) -> object:
    if isinstance(source, Mapping):
        return source
    text = Path(source).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"sketch file {str(source)!r} is not JSON: {exc}") from exc


def _evolution(dataset: PDEDataset) -> PDEDataset:
    if dataset.lhs_order == 0:
        raise ValueError("a Sketch binds to an evolution dataset; lhs_order is 0")
    return resolve_lhs_defaults(dataset)


def _bind_complete(sketch: CompleteSketch, dataset: PDEDataset) -> CompleteSketch:
    return dataset_sketch(
        dataset,
        vocabulary=dataset_vocabulary(dataset, sketch.vocabulary),
        pinned=sketch.pinned,
        anchored=sketch.anchored,
        holes=sketch.holes,
        lhs_spec=sketch.lhs_spec,
        match_policy=sketch.match_policy,
    )


def _spec_to_dict(spec: SketchSpec) -> dict[str, object]:
    return {
        "schema": SKETCH_SPEC_SCHEMA_TAG,
        "pinned": [
            {"term": term, "value": float(value)} for term, value in spec.pinned
        ],
        "anchored": list(spec.anchored),
        "holes": [_hole_to_dict(hole) for hole in spec.holes],
        "lhs_spec": None if spec.lhs_spec is None else lhs_to_dict(spec.lhs_spec),
        "vocabulary": (
            None if spec.vocabulary is None else vocabulary_to_dict(spec.vocabulary)
        ),
        "match_policy": (
            None if spec.match_policy is None else policy_to_dict(spec.match_policy)
        ),
    }


def _hole_to_dict(hole: Hole) -> dict[str, object]:
    return {
        "max_terms": hole.max_terms,
        "min_terms": hole.min_terms,
        "max_deriv_order": hole.max_deriv_order,
        "operators": None if hole.operators is None else sorted(hole.operators),
        "fields": None if hole.fields is None else sorted(hole.fields),
    }


def _spec_from_dict(payload: Mapping[str, object]) -> SketchSpec:
    strict(payload, "sketch spec", _SPEC_KEYS)
    pinned = as_sequence(payload["pinned"], "pinned")
    anchored = as_sequence(payload["anchored"], "anchored")
    holes = as_sequence(payload["holes"], "holes")
    lhs, vocabulary, policy = (
        payload["lhs_spec"],
        payload["vocabulary"],
        payload["match_policy"],
    )
    return SketchSpec(
        pinned=tuple(_spec_pin(value, i) for i, value in enumerate(pinned)),
        anchored=tuple(_spec_anchor(value, i) for i, value in enumerate(anchored)),
        holes=tuple(_spec_hole(value, i) for i, value in enumerate(holes)),
        lhs_spec=None if lhs is None else lhs_from_dict(lhs),
        vocabulary=None if vocabulary is None else vocabulary_from_dict(vocabulary),
        match_policy=None if policy is None else policy_from_dict(policy),
    )


def _spec_pin(value: object, index: int) -> tuple[str, float]:
    name = f"sketch spec pinned[{index}]"
    data = as_mapping(value, name)
    strict(data, name, _SPEC_PIN_KEYS)
    term, coefficient = required_str(data, "term"), required_float(data, "value")
    try:
        PinnedTerm(shorthand_term(term), coefficient)
    except ValueError as exc:
        raise ValueError(f"{name} ({term!r}): {exc}") from exc
    return term, coefficient


def _spec_anchor(value: object, index: int) -> str:
    name = f"sketch spec anchored[{index}]"
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a term string")
    try:
        shorthand_term(value)
    except ValueError as exc:
        raise ValueError(f"{name}: invalid term {value!r}: {exc}") from exc
    return value


def _spec_hole(value: object, index: int) -> Hole:
    name = f"sketch spec holes[{index}]"
    data = as_mapping(value, name)
    strict(data, name, _SPEC_HOLE_KEYS)
    try:
        return Hole(
            max_terms=required_int(data, "max_terms"),
            min_terms=required_int(data, "min_terms"),
            max_deriv_order=optional_int(data, "max_deriv_order"),
            operators=optional_names(data["operators"], f"{name}.operators"),
            fields=optional_names(data["fields"], f"{name}.fields"),
        )
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc


__all__ = [
    "SKETCH_SPEC_SCHEMA_TAG",
    "Hole",
    "Sketch",
    "SketchSpec",
    "load_sketch",
    "save_sketch",
]
