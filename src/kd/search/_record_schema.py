
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from kd.core.strict_keys import strict_keys as _strict_keys_core


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
    _strict_keys_core(
        data,
        object_name=object_name,
        required=required,
        optional=optional,
        error_cls=StrictDecodeError,
    )


def as_dict(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StrictDecodeError(f"{field} must be an object")
    return cast(dict[str, Any], value)








LINEAGE_FIELDS: tuple[str, ...] = (
    "resume_from",
    "source_run_id",
    "source_config_hash",
    "source_final_status",
    "source_iteration",
)
_LINEAGE_FIELD_SET = frozenset(LINEAGE_FIELDS)


def validate_lineage(
    lineage: object,
    *,
    error_cls: type[Exception] = StrictDecodeError,
) -> dict[str, Any] | None:
    if lineage is None:
        return None
    if not isinstance(lineage, Mapping):
        raise error_cls(f"lineage must be an object or null; got {lineage!r}")
    keys = frozenset(lineage)
    if keys != _LINEAGE_FIELD_SET:
        raise error_cls(
            f"lineage keys must be exactly {sorted(_LINEAGE_FIELD_SET)!r}; "
            f"got {sorted(keys)!r}"
        )
    resume_from = lineage["resume_from"]
    if not isinstance(resume_from, str) or not resume_from:
        raise error_cls(
            f"lineage.resume_from must be a non-empty str; got {resume_from!r}"
        )
    for field in ("source_run_id", "source_config_hash", "source_final_status"):
        value = lineage[field]
        if value is not None and (not isinstance(value, str) or not value):
            raise error_cls(
                f"lineage.{field} must be a non-empty str or None; got {value!r}"
            )
    iteration = lineage["source_iteration"]
    if iteration is not None and (
        type(iteration) is not int or iteration < 0
    ):
        raise error_cls(
            "lineage.source_iteration must be a non-negative int or None "
            f"(bool rejected); got {iteration!r}"
        )
    return {field: lineage[field] for field in LINEAGE_FIELDS}


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
