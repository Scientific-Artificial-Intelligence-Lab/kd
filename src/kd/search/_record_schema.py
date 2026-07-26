
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast


class RecordSchemaError(ValueError):
    pass


class SchemaVersionError(RecordSchemaError):
    pass


class EvidenceHashSchemeError(RecordSchemaError):
    pass


class StrictDecodeError(RecordSchemaError):
    pass


class RunCostValueError(RecordSchemaError):
    pass


class EvidenceHashError(RecordSchemaError):
    pass


class RecordHashError(RecordSchemaError):
    pass


class RecordHashSchemeError(RecordSchemaError):
    pass


class RunSpecHashSchemeError(RecordSchemaError):
    pass


class InvalidReasonError(RecordSchemaError):
    pass


class HeadlineCoefficientSourceError(RecordSchemaError):
    pass


class RunSpecHashError(RecordSchemaError):
    pass


def strict_keys(
    data: Mapping[str, object],
    *,
    object_name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    actual = frozenset(data)
    unknown = actual - required - optional
    if unknown:
        keys = ", ".join(repr(key) for key in sorted(unknown))
        raise StrictDecodeError(f"Unknown {object_name} field(s): {keys}")
    missing = required - actual
    if missing:
        keys = ", ".join(repr(key) for key in sorted(missing))
        raise StrictDecodeError(f"Missing required {object_name} field(s): {keys}")


def as_dict(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StrictDecodeError(f"{field} must be an object")
    return cast(dict[str, Any], value)


_EQUATION_FIELDS = frozenset({"form", "lhs_spec", "terms", "attrs", "active_indices"})
_EQUATION_LHS_FIELDS = frozenset({"field", "axis", "order"})
_EQUATION_COEFFICIENT_FIELDS = frozenset({"kind", "value"})


def _validate_equation_coefficient(value: object, *, field: str) -> None:
    if isinstance(value, Mapping):
        strict_keys(
            cast(Mapping[str, object], value),
            object_name=field,
            required=_EQUATION_COEFFICIENT_FIELDS,
        )


def _validate_equation_terms(value: object) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return
    for index, term in enumerate(value):
        if isinstance(term, Sequence) and not isinstance(
            term, str | bytes | bytearray
        ) and len(term) >= 2:
            _validate_equation_coefficient(
                term[1], field=f"evidence.catalog_fit.terms[{index}].coefficient"
            )


def validate_equation_payload(data: dict[str, Any]) -> None:
    strict_keys(data, object_name="evidence.catalog_fit", required=_EQUATION_FIELDS)
    lhs_spec = data["lhs_spec"]
    if isinstance(lhs_spec, Mapping):
        strict_keys(
            cast(Mapping[str, object], lhs_spec),
            object_name="evidence.catalog_fit.lhs_spec",
            required=_EQUATION_LHS_FIELDS,
        )
    attrs = data["attrs"]
    if isinstance(attrs, Mapping):
        strict_keys(
            cast(Mapping[str, object], attrs),
            object_name="evidence.catalog_fit.attrs",
            required=frozenset(),
        )
    _validate_equation_terms(data["terms"])


__all__ = [
    "EvidenceHashError",
    "EvidenceHashSchemeError",
    "HeadlineCoefficientSourceError",
    "InvalidReasonError",
    "RecordHashError",
    "RecordHashSchemeError",
    "RecordSchemaError",
    "RunCostValueError",
    "RunSpecHashError",
    "RunSpecHashSchemeError",
    "SchemaVersionError",
    "StrictDecodeError",
    "as_dict",
    "strict_keys",
    "validate_equation_payload",
]
