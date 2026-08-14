
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from kd.core.jsonsafe import JSON_INDENT_SPACES
from kd.search._record_schema import (
    EvidenceHashError,
    EvidenceHashSchemeError,
    HeadlineCoefficientSourceError,
    InvalidReasonError,
    RecordHashError,
    RecordHashSchemeError,
    RunCostValueError,
    RunSpecHashError,
    RunSpecHashSchemeError,
    SchemaVersionError,
)
from kd.search._record_schema import (
    RecordSchemaError as RecordSchemaError,
)
from kd.search._record_schema import (
    StrictDecodeError as StrictDecodeError,
)
from kd.search._record_schema import (
    as_dict as _as_dict,
)
from kd.search._record_schema import (
    strict_keys as _strict_keys,
)
from kd.search._record_schema import (
    validate_equation_payload as _validate_equation_payload,
)

if TYPE_CHECKING:
    from kd.search.run_spec import RunSpec

RUN_RECORD_SCHEMA_VERSION: Final[int] = 2
EVIDENCE_HASH_SCHEME: Final[str] = "kd-evidence-v1"
RECORD_HASH_SCHEME: Final[str] = "kd-record-v1"
INVALID_REASON_VOCAB: Final[frozenset[str]] = frozenset(
    {
        "no_candidate",
        "evaluation_error",
        "non_finite",
        "structural_reject",
        "unclassified",
    }
)
HEADLINE_COEFFICIENT_SOURCE_VOCAB: Final[frozenset[str]] = frozenset(
    {"native", "platform_refit", "undeclared"}
)

_RUN_COST_REQUIRED_FIELDS = frozenset(
    {
        "wallclock_seconds", "search_seconds",
        "boundary_results", "boundary_invalid_results",
    }
)
_RUN_COST_OPTIONAL_FIELDS = frozenset(
    {
        "preprocessing_seconds", "surrogate_train_seconds", "cpu_seconds",
        "tokens_in", "tokens_out", "tokens_cached", "api_cost_usd",
    }
)
_RESIDUAL_STATS_FIELDS = frozenset({"mean", "std", "max_abs", "n"})
_EVIDENCE_REQUIRED_FIELDS = frozenset(
    {
        "instrument", "dataset_name", "dataset_cache_fingerprint", "seed",
        "is_valid", "expression", "score_kind", "score_direction",
        "headline_coefficient_source",
    }
)
_EVIDENCE_OPTIONAL_FIELDS = frozenset(
    {
        "catalog_fit", "support", "coefficients", "complexity", "mse", "nmse",
        "r2", "score", "residual_stats", "invalid_reason", "error_detail",
    }
)
_RUN_RECORD_FIELDS = frozenset(
    {
        "schema_version", "evidence_hash_scheme", "created_at", "cost",
        "evidence", "evidence_hash", "run_spec", "run_spec_hash",
        "run_spec_hash_scheme", "record_hash", "record_hash_scheme",
    }
)












_EVIDENCE_V1_SKELETON: Final[frozenset[str]] = frozenset(
    {
        "instrument", "dataset_name", "dataset_cache_fingerprint", "seed",
        "is_valid", "expression", "score_kind", "score_direction",
        "headline_coefficient_source",
    }
)
_EVIDENCE_V1_OPTIONAL: Final[frozenset[str]] = frozenset(
    {
        "catalog_fit", "support", "coefficients", "complexity", "mse", "nmse",
        "r2", "score", "residual_stats", "invalid_reason",
    }
)
_EVIDENCE_V1_EXCLUDED: Final[frozenset[str]] = frozenset({"error_detail"})


_HASH_EXCLUDED_FIELDS: Final[frozenset[str]] = _EVIDENCE_V1_EXCLUDED

def validate_invalid_reason(reason: object) -> str:
    if not isinstance(reason, str) or reason not in INVALID_REASON_VOCAB:
        raise InvalidReasonError(
            f"Unsupported invalid_reason: got {reason!r}; "
            f"supported values: {sorted(INVALID_REASON_VOCAB)!r}"
        )
    return reason


def _validate_headline_coefficient_source(source: object) -> str:
    if not isinstance(source, str) or source not in HEADLINE_COEFFICIENT_SOURCE_VOCAB:
        raise HeadlineCoefficientSourceError(
            f"Unsupported headline_coefficient_source: got {source!r}; "
            f"supported values: {sorted(HEADLINE_COEFFICIENT_SOURCE_VOCAB)!r}"
        )
    return source


@dataclass(frozen=True)
class RunCost:

    wallclock_seconds: float
    search_seconds: float
    boundary_results: int
    boundary_invalid_results: int
    preprocessing_seconds: float | None = None
    surrogate_train_seconds: float | None = None
    cpu_seconds: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_cached: int | None = None
    api_cost_usd: float | None = None

    def __post_init__(self) -> None:
        for field in ("wallclock_seconds", "search_seconds"):
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0:
                raise RunCostValueError(
                    f"RunCost.{field} must be finite and non-negative; "
                    f"got {value!r}"
                )
        for field in (
            "preprocessing_seconds",
            "surrogate_train_seconds",
            "cpu_seconds",
            "api_cost_usd",
        ):
            value = getattr(self, field)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise RunCostValueError(
                    f"RunCost.{field} must be finite and non-negative when set; "
                    f"got {value!r}"
                )
        for field in ("tokens_in", "tokens_out", "tokens_cached"):
            value = getattr(self, field)
            if value is not None and value < 0:
                raise RunCostValueError(
                    f"RunCost.{field} must be non-negative when set; got {value!r}"
                )
        for field in ("boundary_results", "boundary_invalid_results"):
            value = getattr(self, field)
            if value < 0:
                raise RunCostValueError(
                    f"RunCost.{field} must be non-negative; got {value!r}"
                )


        if self.boundary_invalid_results > self.boundary_results:
            raise RunCostValueError(
                "RunCost.boundary_invalid_results must not exceed "
                f"boundary_results; got {self.boundary_invalid_results!r} > "
                f"{self.boundary_results!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunCost:
        _strict_keys(
            data,
            object_name="cost",
            required=_RUN_COST_REQUIRED_FIELDS,
            optional=_RUN_COST_OPTIONAL_FIELDS,
        )
        return cls(
            wallclock_seconds=data["wallclock_seconds"],
            search_seconds=data["search_seconds"],
            boundary_results=data["boundary_results"],
            boundary_invalid_results=data["boundary_invalid_results"],
            preprocessing_seconds=data.get("preprocessing_seconds"),
            surrogate_train_seconds=data.get("surrogate_train_seconds"),
            cpu_seconds=data.get("cpu_seconds"),
            tokens_in=data.get("tokens_in"),
            tokens_out=data.get("tokens_out"),
            tokens_cached=data.get("tokens_cached"),
            api_cost_usd=data.get("api_cost_usd"),
        )


@dataclass(frozen=True)
class ResidualStats:

    mean: float | None
    std: float | None
    max_abs: float | None
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResidualStats:
        _strict_keys(
            data,
            object_name="residual_stats",
            required=_RESIDUAL_STATS_FIELDS,
        )
        return cls(
            mean=data["mean"],
            std=data["std"],
            max_abs=data["max_abs"],
            n=data["n"],
        )


@dataclass(frozen=True)
class EvidenceRecord:

    instrument: str
    dataset_name: str
    dataset_cache_fingerprint: str
    seed: int | None
    is_valid: bool
    expression: str
    score_kind: str
    score_direction: str
    headline_coefficient_source: str
    catalog_fit: dict[str, Any] | None = None
    support: list[str] | None = None
    coefficients: list[float | None] | None = None
    complexity: int | None = None
    mse: float | None = None
    nmse: float | None = None
    r2: float | None = None
    score: float | None = None
    residual_stats: ResidualStats | None = None
    invalid_reason: str | None = None
    error_detail: str | None = None

    def __post_init__(self) -> None:
        _validate_headline_coefficient_source(self.headline_coefficient_source)
        if self.invalid_reason is not None:
            validate_invalid_reason(self.invalid_reason)





        if self.is_valid:
            for field in ("invalid_reason", "error_detail"):
                value = getattr(self, field)
                if value is not None:
                    raise StrictDecodeError(
                        f"A valid EvidenceRecord must not set {field}; "
                        f"got {value!r}"
                    )
        if self.catalog_fit is not None:
            object.__setattr__(self, "catalog_fit", deepcopy(self.catalog_fit))
        for field in ("support", "coefficients"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, list(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceRecord:
        _strict_keys(
            data,
            object_name="evidence",
            required=_EVIDENCE_REQUIRED_FIELDS,
            optional=_EVIDENCE_OPTIONAL_FIELDS,
        )
        catalog_fit_data = data.get("catalog_fit")
        if catalog_fit_data is not None:
            _validate_equation_payload(
                _as_dict(catalog_fit_data, field="evidence.catalog_fit")
            )
        residual_data = data.get("residual_stats")
        residual_stats = None
        if residual_data is not None:
            residual_stats = ResidualStats.from_dict(
                _as_dict(residual_data, field="evidence.residual_stats")
            )
        return cls._from_validated_dict(data, residual_stats=residual_stats)

    @classmethod
    def _from_validated_dict(
        cls,
        data: dict[str, Any],
        *,
        residual_stats: ResidualStats | None,
    ) -> EvidenceRecord:
        return cls(
            instrument=data["instrument"],
            dataset_name=data["dataset_name"],
            dataset_cache_fingerprint=data["dataset_cache_fingerprint"],
            seed=data["seed"],
            is_valid=data["is_valid"],
            expression=data["expression"],
            score_kind=data["score_kind"],
            score_direction=data["score_direction"],
            headline_coefficient_source=data["headline_coefficient_source"],
            catalog_fit=data.get("catalog_fit"),
            support=data.get("support"),
            coefficients=data.get("coefficients"),
            complexity=data.get("complexity"),
            mse=data.get("mse"),
            nmse=data.get("nmse"),
            r2=data.get("r2"),
            score=data.get("score"),
            residual_stats=residual_stats,
            invalid_reason=data.get("invalid_reason"),
            error_detail=data.get("error_detail"),
        )

    def content_hash(self, evidence_hash_scheme: str = EVIDENCE_HASH_SCHEME) -> str:
        return _evidence_hasher(evidence_hash_scheme)(self)


def _evidence_v1_payload(evidence: EvidenceRecord) -> dict[str, Any]:






    full_payload = evidence.to_dict()
    payload = {field: full_payload[field] for field in _EVIDENCE_V1_SKELETON}
    for field in _EVIDENCE_V1_OPTIONAL:
        value = full_payload[field]
        if value is not None:
            payload[field] = value
    return payload


def _evidence_v1_hash(evidence: EvidenceRecord) -> str:
    canonical = json.dumps(
        _evidence_v1_payload(evidence),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    hash_input = f"{EVIDENCE_HASH_SCHEME}:{canonical}"
    hashed_bytes = hash_input.encode("utf-8")
    return hashlib.sha256(hashed_bytes).hexdigest()


_EVIDENCE_HASHERS: dict[str, Callable[[EvidenceRecord], str]] = {
    EVIDENCE_HASH_SCHEME: _evidence_v1_hash,
}


def _evidence_hasher(scheme: object) -> Callable[[EvidenceRecord], str]:
    if not isinstance(scheme, str) or scheme not in _EVIDENCE_HASHERS:
        supported = sorted(_EVIDENCE_HASHERS)
        raise EvidenceHashSchemeError(
            f"Unsupported evidence_hash_scheme: got {scheme!r}; "
            f"supported schemes: {supported!r}"
        )
    return _EVIDENCE_HASHERS[scheme]


@dataclass(frozen=True)
class RunRecord:

    schema_version: int
    evidence_hash_scheme: str
    created_at: str
    cost: RunCost
    evidence: EvidenceRecord
    evidence_hash: str
    run_spec: RunSpec
    run_spec_hash: str
    run_spec_hash_scheme: str
    record_hash: str
    record_hash_scheme: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        version = _schema_version(data.get("schema_version", _MISSING))
        return _RUN_RECORD_DECODERS[version](data)

    def save(self, path: Path | str) -> None:
        _schema_version(self.schema_version)
        _evidence_hasher(self.evidence_hash_scheme)
        _record_hasher(self.record_hash_scheme)
        _validate_run_spec_hash_scheme(self.run_spec_hash_scheme)
        _require_valid_evidence_hash(self)
        _require_valid_run_spec_hash(self)
        _require_valid_record_hash(self)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)


        tmp_path = output_path.with_name(f"{output_path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=JSON_INDENT_SPACES,
                allow_nan=False,
            )
        os.replace(tmp_path, output_path)

    @classmethod
    def load(cls, path: Path | str) -> RunRecord:
        input_path = Path(path)
        with input_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise StrictDecodeError("run_record must be an object")
        return cls.from_dict(cast(dict[str, Any], data))

    def verify_evidence_hash(self) -> bool:
        expected = self.evidence.content_hash(self.evidence_hash_scheme)
        return expected == self.evidence_hash

    def verify_run_spec_hash(self) -> bool:
        return self.run_spec.run_spec_hash == self.run_spec_hash

    def record_content_hash(
        self, record_hash_scheme: str = RECORD_HASH_SCHEME
    ) -> str:
        return _record_hasher(record_hash_scheme)(self)

    def verify_record_hash(self) -> bool:
        return self.record_content_hash(self.record_hash_scheme) == self.record_hash


_MISSING = object()


def _schema_version(value: object) -> int:
    if type(value) is not int or value not in _RUN_RECORD_DECODERS:
        supported = sorted(_RUN_RECORD_DECODERS)
        raise SchemaVersionError(
            f"Unsupported schema_version: got {value!r}; "
            f"supported versions: {supported!r}"
        )
    return value


def _decode_run_record_v2(data: dict[str, Any]) -> RunRecord:
    from kd.search.run_spec import RunSpec

    scheme = data.get("evidence_hash_scheme", _MISSING)
    _evidence_hasher(scheme)
    record_scheme = data.get("record_hash_scheme", _MISSING)
    _record_hasher(record_scheme)
    run_spec_scheme = data.get("run_spec_hash_scheme", _MISSING)
    _validate_run_spec_hash_scheme(run_spec_scheme)
    _strict_keys(data, object_name="run_record", required=_RUN_RECORD_FIELDS)
    record = RunRecord(
        schema_version=data["schema_version"],
        evidence_hash_scheme=cast(str, scheme),
        created_at=data["created_at"],
        cost=RunCost.from_dict(_as_dict(data["cost"], field="cost")),
        evidence=EvidenceRecord.from_dict(
            _as_dict(data["evidence"], field="evidence")
        ),
        evidence_hash=data["evidence_hash"],
        run_spec=RunSpec.from_dict(_as_dict(data["run_spec"], field="run_spec")),
        run_spec_hash=data["run_spec_hash"],
        run_spec_hash_scheme=cast(str, run_spec_scheme),
        record_hash=data["record_hash"],
        record_hash_scheme=cast(str, record_scheme),
    )
    _require_valid_evidence_hash(record)
    _require_valid_run_spec_hash(record)
    _require_valid_record_hash(record)
    return record


def _require_valid_evidence_hash(record: RunRecord) -> None:
    if not record.verify_evidence_hash():
        expected = record.evidence.content_hash(record.evidence_hash_scheme)
        raise EvidenceHashError(
            "Evidence hash mismatch: "
            f"got {record.evidence_hash!r}; expected {expected!r}"
        )


def _require_valid_run_spec_hash(record: RunRecord) -> None:
    if not record.verify_run_spec_hash():
        expected = record.run_spec.run_spec_hash
        raise RunSpecHashError(
            "run_spec hash mismatch: "
            f"got {record.run_spec_hash!r}; expected {expected!r}"
        )


def _require_valid_record_hash(record: RunRecord) -> None:
    if not record.verify_record_hash():
        expected = record.record_content_hash(record.record_hash_scheme)
        raise RecordHashError(
            "record hash mismatch: "
            f"got {record.record_hash!r}; expected {expected!r}"
        )


def _record_v1_hash(record: RunRecord) -> str:






    payload = record.to_dict()
    payload.pop("record_hash")
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    hash_input = f"{RECORD_HASH_SCHEME}:{canonical}"
    hashed_bytes = hash_input.encode("utf-8")
    return hashlib.sha256(hashed_bytes).hexdigest()


_RECORD_HASHERS: dict[str, Callable[[RunRecord], str]] = {
    RECORD_HASH_SCHEME: _record_v1_hash,
}


def _record_hasher(scheme: object) -> Callable[[RunRecord], str]:
    if not isinstance(scheme, str) or scheme not in _RECORD_HASHERS:
        supported = sorted(_RECORD_HASHERS)
        raise RecordHashSchemeError(
            f"Unsupported record_hash_scheme: got {scheme!r}; "
            f"supported schemes: {supported!r}"
        )
    return _RECORD_HASHERS[scheme]


def _validate_run_spec_hash_scheme(scheme: object) -> str:
    from kd.search.run_spec import RUN_SPEC_HASH_SCHEME

    supported = {RUN_SPEC_HASH_SCHEME}
    if not isinstance(scheme, str) or scheme not in supported:
        raise RunSpecHashSchemeError(
            f"Unsupported run_spec_hash_scheme: got {scheme!r}; "
            f"supported schemes: {sorted(supported)!r}"
        )
    return scheme


def seal_record_hash(record: RunRecord) -> RunRecord:
    record_hash = _record_hasher(record.record_hash_scheme)(record)
    return replace(record, record_hash=record_hash)


_RUN_RECORD_DECODERS: dict[int, Callable[[dict[str, Any]], RunRecord]] = {
    RUN_RECORD_SCHEMA_VERSION: _decode_run_record_v2
}
