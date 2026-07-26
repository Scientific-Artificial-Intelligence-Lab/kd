
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias, assert_never, cast

from kd.core.equation.construct import make_evolution, make_homogeneous
from kd.core.equation.types import (
    Coefficient,
    Equation,
    EquationAttrs,
    Evolution,
    Form,
    Homogeneous,
    LhsSpec,
    Scalar,
    Term,
    fold_terms,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def to_dict(eq: Equation) -> JsonObject:
    match eq:
        case Evolution():
            return {
                "form": Form.EVOLUTION.name,
                "lhs_spec": _lhs_spec_to_dict(eq.lhs_spec),
                "terms": _terms_to_list(eq.terms),
                "attrs": _attrs_to_dict(eq.attrs),
                "active_indices": _active_indices_to_list(eq.active_indices),
            }
        case Homogeneous():
            return {
                "form": Form.HOMOGENEOUS.name,
                "lhs_spec": None,
                "terms": _terms_to_list(eq.terms),
                "attrs": None,
                "active_indices": _active_indices_to_list(eq.active_indices),
            }
    assert_never(eq)


def from_dict(payload: Mapping[str, object]) -> Equation:
    form = _form_from_dict(payload)
    terms = _terms_from_list(_required(payload, "terms"))
    _attrs_from_dict(payload.get("attrs"))


    active_indices = _active_indices_from_dict(payload.get("active_indices"))
    if form is Form.EVOLUTION:
        return make_evolution(
            _lhs_spec_from_dict(payload.get("lhs_spec")),
            terms,
            active_indices=active_indices,
        )
    if form is Form.HOMOGENEOUS:




        if payload.get("lhs_spec") is not None:
            raise ValueError("HOMOGENEOUS equations must not carry a lhs_spec")
        return make_homogeneous(terms, active_indices=active_indices)
    raise NotImplementedError(f"{form.name} equations are reserved")


def _lhs_spec_to_dict(lhs_spec: LhsSpec | None) -> JsonObject | None:
    if lhs_spec is None:
        return None
    return {
        "field": lhs_spec.field,
        "axis": lhs_spec.axis,
        "order": lhs_spec.order,
    }


def _attrs_to_dict(_attrs: EquationAttrs) -> JsonObject | None:
    return None


def _active_indices_to_list(
    active_indices: tuple[int, ...] | None,
) -> list[JsonValue] | None:
    if active_indices is None:
        return None
    payload: list[JsonValue] = []
    for index in active_indices:
        payload.append(index)
    return payload


def _terms_to_list(terms: Sequence[Term]) -> list[JsonValue]:
    initial: list[JsonValue] = []
    return fold_terms(terms, initial, _append_term_payload)


def _append_term_payload(terms_payload: list[JsonValue], term: Term) -> list[JsonValue]:
    term_ir, coefficient = term
    term_payload: list[JsonValue] = [term_ir, _coefficient_to_dict(coefficient)]
    terms_payload.append(term_payload)
    return terms_payload


def _coefficient_to_dict(coefficient: Coefficient) -> JsonObject:
    if isinstance(coefficient, Scalar):
        return {"kind": "Scalar", "value": coefficient.value}
    raise NotImplementedError(f"{type(coefficient).__name__} serialization is reserved")


def _form_from_dict(payload: Mapping[str, object]) -> Form:
    form_name = _required_str(payload, "form")
    try:
        return Form[form_name]
    except KeyError as exc:
        raise ValueError(f"unknown equation form: {form_name}") from exc


def _lhs_spec_from_dict(value: object) -> LhsSpec | None:
    if value is None:
        return None
    payload = _as_mapping(value, field="lhs_spec")
    field = _required_str(payload, "field")
    axis = _required_str(payload, "axis")
    if not field:
        raise ValueError("lhs_spec.field must be non-empty")
    if not axis:
        raise ValueError("lhs_spec.axis must be non-empty")
    return LhsSpec(
        field=field,
        axis=axis,
        order=_required_int(payload, "order"),
    )


def _attrs_from_dict(value: object) -> EquationAttrs:
    if value is not None:
        raise ValueError("attrs must be null in step 1")
    return EquationAttrs()


def _active_indices_from_dict(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    payload = _as_sequence(value, field="active_indices")
    indices: list[int] = []
    for index, item in enumerate(payload):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"active_indices[{index}] must be an integer")
        indices.append(item)
    return tuple(indices)


def _terms_from_list(value: object) -> tuple[Term, ...]:
    payload = _as_sequence(value, field="terms")
    terms: list[Term] = []
    for index, item in enumerate(payload):
        pair = _as_sequence(item, field=f"terms[{index}]")
        if len(pair) != 2:
            raise ValueError(f"terms[{index}] must contain term IR and coefficient")
        term_ir = pair[0]
        if not isinstance(term_ir, str):
            raise ValueError(f"terms[{index}][0] must be a string")
        terms.append((term_ir, _coefficient_from_dict(pair[1])))
    return tuple(terms)


def _coefficient_from_dict(value: object) -> Coefficient:
    payload = _as_mapping(value, field="coefficient")
    kind = _required_str(payload, "kind")
    if kind != "Scalar":
        raise NotImplementedError(f"{kind} coefficients are reserved")
    raw_value = _required(payload, "value")
    if not isinstance(raw_value, int | float) or isinstance(raw_value, bool):
        raise ValueError("Scalar.value must be numeric")
    return Scalar(float(raw_value))


def _required(payload: Mapping[str, object], key: str) -> object:
    try:
        return payload[key]
    except KeyError as exc:
        raise ValueError(f"missing required equation field: {key}") from exc


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = _required(payload, key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = _required(payload, key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{field} must be a sequence")
    return cast(Sequence[object], value)
