
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from kd.core.equation.sketch import (
    SKETCH_SCHEMA_TAG,
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


def _strict(
    data: Mapping[str, object], object_name: str, required: frozenset[str]
) -> None:
    strict_keys(
        data,
        object_name=object_name,
        required=required,
        error_cls=ValueError,
    )


def _as_mapping(value: object, object_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{object_name} must be an object")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, object_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{object_name} must be a sequence")
    return cast(Sequence[object], value)


def _required_str(data: Mapping[str, object], key: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _required_int(data: Mapping[str, object], key: str) -> int:
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_float(data: Mapping[str, object], key: str) -> float:
    value = data[key]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _optional_int(data: Mapping[str, object], key: str) -> int | None:
    value = data[key]
    if value is None:
        return None
    return _required_int(data, key)


def _names(value: object, object_name: str) -> frozenset[str]:
    sequence = _as_sequence(value, object_name)
    names: list[str] = []
    for item in sequence:
        if not isinstance(item, str):
            raise ValueError(f"{object_name} entries must be strings")
        names.append(item)
    return frozenset(names)


def _optional_names(value: object, object_name: str) -> frozenset[str] | None:
    if value is None:
        return None
    return _names(value, object_name)


def _lhs_from_dict(value: object) -> LhsSpec:
    data = _as_mapping(value, "lhs_spec")
    _strict(data, "sketch lhs_spec", _LHS_KEYS)
    return LhsSpec(
        field=_required_str(data, "field"),
        axis=_required_str(data, "axis"),
        order=_required_int(data, "order"),
    )


def _vocabulary_from_dict(value: object) -> TermVocabulary:
    data = _as_mapping(value, "vocabulary")
    _strict(data, "sketch vocabulary", _VOCABULARY_KEYS)
    return TermVocabulary(
        fields=_names(data["fields"], "vocabulary.fields"),
        coordinates=_names(data["coordinates"], "vocabulary.coordinates"),
    )


def _pinned_from_dict(value: object, index: int) -> PinnedTerm:
    name = f"sketch pinned[{index}]"
    data = _as_mapping(value, name)
    _strict(data, name, _PINNED_KEYS)
    return PinnedTerm(
        term_ir=_required_str(data, "term_ir"),
        value=_required_float(data, "value"),
    )


def _anchored_from_dict(value: object, index: int) -> AnchoredTerm:
    name = f"sketch anchored[{index}]"
    data = _as_mapping(value, name)
    _strict(data, name, _ANCHORED_KEYS)
    return AnchoredTerm(term_ir=_required_str(data, "term_ir"))


def _constraint_from_dict(value: object, index: int) -> TermConstraint:
    name = f"sketch holes[{index}].constraint"
    data = _as_mapping(value, name)
    _strict(data, name, _CONSTRAINT_KEYS)
    return TermConstraint(
        max_deriv_order=_optional_int(data, "max_deriv_order"),
        operators=_optional_names(data["operators"], f"{name}.operators"),
        fields=_optional_names(data["fields"], f"{name}.fields"),
        axes=_optional_names(data["axes"], f"{name}.axes"),
    )


def _hole_from_dict(value: object, index: int) -> TermHole:
    name = f"sketch holes[{index}]"
    data = _as_mapping(value, name)
    _strict(data, name, _HOLE_KEYS)
    return TermHole(
        id=_required_str(data, "id"),
        min_count=_required_int(data, "min_count"),
        max_count=_required_int(data, "max_count"),
        constraint=_constraint_from_dict(data["constraint"], index),
    )


def _policy_from_dict(value: object) -> SketchMatchPolicy:
    data = _as_mapping(value, "match_policy")
    _strict(data, "sketch match_policy", _POLICY_KEYS)
    return SketchMatchPolicy(
        coeff_atol=_required_float(data, "coeff_atol"),
        coeff_rtol=_required_float(data, "coeff_rtol"),
        support_threshold=_required_float(data, "support_threshold"),
        term_identity=_required_str(data, "term_identity"),
        hole_assignment=_required_str(data, "hole_assignment"),
        derivative_order=_required_str(data, "derivative_order"),
    )


def sketch_from_dict_impl(payload: Mapping[str, object]) -> Sketch:
    _strict(payload, "sketch", _TOP_KEYS)
    if payload["schema"] != SKETCH_SCHEMA_TAG:
        raise ValueError(f"Unsupported sketch schema: {payload['schema']!r}")
    pinned = _as_sequence(payload["pinned"], "pinned")
    anchored = _as_sequence(payload["anchored"], "anchored")
    holes = _as_sequence(payload["holes"], "holes")
    return Sketch(
        lhs_spec=_lhs_from_dict(payload["lhs_spec"]),
        vocabulary=_vocabulary_from_dict(payload["vocabulary"]),
        pinned=tuple(_pinned_from_dict(value, i) for i, value in enumerate(pinned)),
        anchored=tuple(
            _anchored_from_dict(value, i) for i, value in enumerate(anchored)
        ),
        holes=tuple(_hole_from_dict(value, i) for i, value in enumerate(holes)),
        match_policy=_policy_from_dict(payload["match_policy"]),
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


def sketch_to_dict_impl(sketch: Sketch) -> dict[str, object]:
    policy = sketch.match_policy
    return {
        "schema": SKETCH_SCHEMA_TAG,
        "lhs_spec": {
            "field": sketch.lhs_spec.field,
            "axis": sketch.lhs_spec.axis,
            "order": sketch.lhs_spec.order,
        },
        "vocabulary": {
            "fields": sorted(sketch.vocabulary.fields),
            "coordinates": sorted(sketch.vocabulary.coordinates),
        },
        "pinned": [
            {"term_ir": pin.term_ir, "value": pin.value} for pin in sketch.pinned
        ],
        "anchored": [{"term_ir": anchor.term_ir} for anchor in sketch.anchored],
        "holes": [_hole_to_dict(hole) for hole in sketch.holes],
        "match_policy": {
            "coeff_atol": policy.coeff_atol,
            "coeff_rtol": policy.coeff_rtol,
            "support_threshold": policy.support_threshold,
            "term_identity": policy.term_identity,
            "hole_assignment": policy.hole_assignment,
            "derivative_order": policy.derivative_order,
        },
    }


__all__: list[str] = []
