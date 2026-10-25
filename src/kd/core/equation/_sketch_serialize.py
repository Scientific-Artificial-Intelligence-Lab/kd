
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from kd.core.equation.sketch import (
    SKETCH_SCHEMA_TAG,
    SKETCH_V1_SCHEMA_TAG,
    AnchoredTerm,
    PinnedTerm,
    Sketch,
    SketchMatchPolicy,
    TermConstraint,
    TermHole,
)
from kd.core.equation.types import LhsSpec
from kd.core.expr.term_features import TermVocabulary
from kd.core.strict_keys import strict_keys

_TOP_KEYS = frozenset(
    {
        "schema",
        "lhs_spec",
        "vocabulary",
        "pinned",
        "anchored",
        "holes",
        "match_policy",
    }
)
_READABLE_SCHEMAS = (SKETCH_V1_SCHEMA_TAG, SKETCH_SCHEMA_TAG)
_LHS_KEYS = frozenset({"field", "axis", "order"})
_VOCABULARY_KEYS = frozenset({"fields", "coordinates"})
_PINNED_KEYS = frozenset({"term_ir", "value"})
_ANCHORED_KEYS = frozenset({"term_ir"})
_HOLE_KEYS = frozenset({"id", "min_count", "max_count", "constraint"})
_CONSTRAINT_KEYS = frozenset({"max_deriv_order", "operators", "fields", "axes"})
_POLICY_KEYS = frozenset(
    {
        "coeff_atol",
        "coeff_rtol",
        "support_threshold",
        "term_identity",
        "hole_assignment",
        "derivative_order",
    }
)


def strict(
    data: Mapping[str, object], object_name: str, required: frozenset[str]
) -> None:
    strict_keys(
        data,
        object_name=object_name,
        required=required,
        error_cls=ValueError,
    )


def as_mapping(value: object, object_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{object_name} must be an object")
    return cast(Mapping[str, object], value)


def as_sequence(value: object, object_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{object_name} must be a sequence")
    return cast(Sequence[object], value)


def required_str(data: Mapping[str, object], key: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def required_int(data: Mapping[str, object], key: str) -> int:
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def required_float(data: Mapping[str, object], key: str) -> float:
    value = data[key]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def optional_int(data: Mapping[str, object], key: str) -> int | None:
    value = data[key]
    if value is None:
        return None
    return required_int(data, key)


def _names(value: object, object_name: str) -> frozenset[str]:
    sequence = as_sequence(value, object_name)
    names: list[str] = []
    for item in sequence:
        if not isinstance(item, str):
            raise ValueError(f"{object_name} entries must be strings")
        names.append(item)
    return frozenset(names)


def optional_names(value: object, object_name: str) -> frozenset[str] | None:
    if value is None:
        return None
    return _names(value, object_name)


def lhs_from_dict(value: object) -> LhsSpec:
    data = as_mapping(value, "lhs_spec")
    strict(data, "sketch lhs_spec", _LHS_KEYS)
    return LhsSpec(
        field=required_str(data, "field"),
        axis=required_str(data, "axis"),
        order=required_int(data, "order"),
    )


def vocabulary_from_dict(value: object) -> TermVocabulary:
    data = as_mapping(value, "vocabulary")
    strict(data, "sketch vocabulary", _VOCABULARY_KEYS)
    return TermVocabulary(
        fields=_names(data["fields"], "vocabulary.fields"),
        coordinates=_names(data["coordinates"], "vocabulary.coordinates"),
    )


def _pinned_from_dict(value: object, index: int) -> PinnedTerm:
    name = f"sketch pinned[{index}]"
    data = as_mapping(value, name)
    strict(data, name, _PINNED_KEYS)
    return PinnedTerm(
        term_ir=required_str(data, "term_ir"),
        value=required_float(data, "value"),
    )


def _anchored_from_dict(value: object, index: int) -> AnchoredTerm:
    name = f"sketch anchored[{index}]"
    data = as_mapping(value, name)
    strict(data, name, _ANCHORED_KEYS)
    return AnchoredTerm(term_ir=required_str(data, "term_ir"))


def _constraint_from_dict(value: object, index: int) -> TermConstraint:
    name = f"sketch holes[{index}].constraint"
    data = as_mapping(value, name)
    strict(data, name, _CONSTRAINT_KEYS)
    return TermConstraint(
        max_deriv_order=optional_int(data, "max_deriv_order"),
        operators=optional_names(data["operators"], f"{name}.operators"),
        fields=optional_names(data["fields"], f"{name}.fields"),
        axes=optional_names(data["axes"], f"{name}.axes"),
    )


def _hole_from_dict(value: object, index: int) -> TermHole:
    name = f"sketch holes[{index}]"
    data = as_mapping(value, name)
    strict(data, name, _HOLE_KEYS)
    return TermHole(
        id=required_str(data, "id"),
        min_count=required_int(data, "min_count"),
        max_count=required_int(data, "max_count"),
        constraint=_constraint_from_dict(data["constraint"], index),
    )


def policy_from_dict(value: object) -> SketchMatchPolicy:
    data = as_mapping(value, "match_policy")
    strict(data, "sketch match_policy", _POLICY_KEYS)
    return SketchMatchPolicy(
        coeff_atol=required_float(data, "coeff_atol"),
        coeff_rtol=required_float(data, "coeff_rtol"),
        support_threshold=required_float(data, "support_threshold"),
        term_identity=required_str(data, "term_identity"),
        hole_assignment=required_str(data, "hole_assignment"),
        derivative_order=required_str(data, "derivative_order"),
    )


def sketch_from_dict_impl(payload: Mapping[str, object]) -> Sketch:
    strict(payload, "sketch", _TOP_KEYS)
    if payload["schema"] not in _READABLE_SCHEMAS:
        raise ValueError(f"Unsupported sketch schema: {payload['schema']!r}")
    pinned = as_sequence(payload["pinned"], "pinned")
    anchored = as_sequence(payload["anchored"], "anchored")
    holes = as_sequence(payload["holes"], "holes")
    return Sketch(
        lhs_spec=lhs_from_dict(payload["lhs_spec"]),
        vocabulary=vocabulary_from_dict(payload["vocabulary"]),
        pinned=tuple(_pinned_from_dict(value, i) for i, value in enumerate(pinned)),
        anchored=tuple(
            _anchored_from_dict(value, i) for i, value in enumerate(anchored)
        ),
        holes=tuple(_hole_from_dict(value, i) for i, value in enumerate(holes)),
        match_policy=policy_from_dict(payload["match_policy"]),
    )


def _constraint_to_dict(constraint: TermConstraint) -> dict[str, object]:
    return {
        "max_deriv_order": constraint.max_deriv_order,
        "operators": None
        if constraint.operators is None
        else sorted(constraint.operators),
        "fields": None if constraint.fields is None else sorted(constraint.fields),
        "axes": None if constraint.axes is None else sorted(constraint.axes),
    }


def _hole_to_dict(hole: TermHole) -> dict[str, object]:
    return {
        "id": hole.id,
        "min_count": hole.min_count,
        "max_count": hole.max_count,
        "constraint": _constraint_to_dict(hole.constraint),
    }


def lhs_to_dict(lhs: LhsSpec) -> dict[str, object]:
    return {"field": lhs.field, "axis": lhs.axis, "order": lhs.order}


def vocabulary_to_dict(vocabulary: TermVocabulary) -> dict[str, object]:
    return {
        "fields": sorted(vocabulary.fields),
        "coordinates": sorted(vocabulary.coordinates),
    }


def policy_to_dict(policy: SketchMatchPolicy) -> dict[str, object]:
    return {
        "coeff_atol": policy.coeff_atol,
        "coeff_rtol": policy.coeff_rtol,
        "support_threshold": policy.support_threshold,
        "term_identity": policy.term_identity,
        "hole_assignment": policy.hole_assignment,
        "derivative_order": policy.derivative_order,
    }


def sketch_to_dict_impl(sketch: Sketch) -> dict[str, object]:
    return {
        "schema": SKETCH_SCHEMA_TAG,
        "lhs_spec": lhs_to_dict(sketch.lhs_spec),
        "vocabulary": vocabulary_to_dict(sketch.vocabulary),
        "pinned": [
            {"term_ir": pin.term_ir, "value": pin.value} for pin in sketch.pinned
        ],
        "anchored": [{"term_ir": anchor.term_ir} for anchor in sketch.anchored],
        "holes": [_hole_to_dict(hole) for hole in sketch.holes],
        "match_policy": policy_to_dict(sketch.match_policy),
    }


__all__: list[str] = []
