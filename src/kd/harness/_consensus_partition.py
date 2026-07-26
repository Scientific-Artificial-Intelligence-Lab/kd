
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kd.core.equation.serialize import from_dict
from kd.core.equation.signature import LawSignature, law_signature
from kd.core.equation.types import Equation
from kd.harness.consensus import DependencyStratum, UnsignableEntry
from kd.harness.plan import PlanEntry
from kd.harness.store import EvidenceStore
from kd.search.records import RunRecord
from kd.search.run_spec import RunSpec

_DECLARED_STRATA: frozenset[str] = frozenset({"finite_diff", "autograd", "none"})


@dataclass(frozen=True, kw_only=True)
class _SignableRecord:

    entry_index: int
    entry: PlanEntry
    record: RunRecord
    equation: Equation
    signature: LawSignature


def _partition_signable(
    store: EvidenceStore,
) -> tuple[tuple[_SignableRecord, ...], tuple[UnsignableEntry, ...]]:
    signables: list[_SignableRecord] = []
    unsignable: list[UnsignableEntry] = []
    for entry_index in sorted(store.records):
        record = store.records[entry_index]
        entry = store.plan.entries[entry_index]
        evidence = record.evidence
        if evidence.is_valid is False:
            unsignable.append(
                UnsignableEntry(
                    entry_index=entry_index,
                    instrument=entry.instrument,
                    dataset_ref=entry.dataset_ref,
                    seed=entry.seed,
                    reason="invalid_record",
                    invalid_reason=evidence.invalid_reason,
                    error_type=None,
                    error_message=None,
                )
            )
            continue
        if evidence.catalog_fit is None:
            unsignable.append(
                UnsignableEntry(
                    entry_index=entry_index,
                    instrument=entry.instrument,
                    dataset_ref=entry.dataset_ref,
                    seed=entry.seed,
                    reason="no_catalog_fit",
                    invalid_reason=None,
                    error_type=None,
                    error_message=None,
                )
            )
            continue
        try:
            equation = from_dict(evidence.catalog_fit)
            signature = law_signature(equation)
        except Exception as exc:
            unsignable.append(
                UnsignableEntry(
                    entry_index=entry_index,
                    instrument=entry.instrument,
                    dataset_ref=entry.dataset_ref,
                    seed=entry.seed,
                    reason="signature_error",
                    invalid_reason=None,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue
        signables.append(
            _SignableRecord(
                entry_index=entry_index,
                entry=entry,
                record=record,
                equation=equation,
                signature=signature,
            )
        )
    return tuple(signables), tuple(unsignable)


def _dependency_stratum(
    instrument: str,
    run_spec: RunSpec,
    schema_by_instrument: Mapping[str, Mapping[str, Any]],
) -> DependencyStratum:
    if instrument not in schema_by_instrument:
        return "undeclared"
    if instrument == "sga":
        use_autograd = run_spec.config.get("use_autograd")
        if use_autograd is True:
            return "autograd"
        if use_autograd is False:
            return "finite_diff"
        return "undeclared"
    modes = schema_by_instrument[instrument]["modes"]
    provider_kinds = {mode["provider_kind"] for mode in modes}
    if len(provider_kinds) == 1:
        return _as_stratum(next(iter(provider_kinds)))
    provided = run_spec.config.get("provider_kind")
    if provided in _DECLARED_STRATA:
        return _as_stratum(provided)
    return "undeclared"


def _as_stratum(value: Any) -> DependencyStratum:
    if value == "finite_diff":
        return "finite_diff"
    if value == "autograd":
        return "autograd"
    if value == "none":
        return "none"
    if value == "undeclared":
        return "undeclared"
    raise ValueError(
        f"provider_kind {value!r} is outside the stratum vocabulary "
        "{finite_diff, autograd, none, undeclared}"
    )
