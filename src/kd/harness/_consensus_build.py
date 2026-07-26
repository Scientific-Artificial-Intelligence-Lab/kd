
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kd.core.platform.builder import resolve_lhs_defaults
from kd.data.schema import DataTopology, PDEDataset, compute_dataset_fingerprint
from kd.harness._consensus_axes import (
    _adjacency,
    _coefficient_axis,
    _empirical_axis,
    _instrument_stability,
    _member_flags,
    _structure_axis,
    _support_axis,
)
from kd.harness._consensus_partition import (
    _dependency_stratum,
    _partition_signable,
    _SignableRecord,
)
from kd.harness._consensus_strata import (
    _absent_outcomes,
    _class_eligibility,
    _instruments_for_ref,
    _resolve_lhs_term,
    _stratification,
)
from kd.harness.consensus import (
    CONSENSUS_ARTIFACT_TAG,
    CORROBORATION_ORDER,
    ELIGIBILITY_RULES_VERSION,
    STRATUM_RULES_VERSION,
    AbsentReason,
    ClassAxes,
    ClassMember,
    ConsensusClass,
    ConsensusPolicy,
    ConsensusProvenance,
    ConsensusReport,
    DatasetConsensus,
    DependencyProfile,
    DependencyStratum,
    StoreLevelSummary,
    VerifierProvenance,
)
from kd.harness.consensus_verify import (
    MemberVerification,
    VerifyContextFactory,
    default_verify_context_factory,
    verify_members,
)
from kd.harness.store import EvidenceStore


def _class_sort_key(cls: ConsensusClass) -> tuple[int, int, int, str]:
    strat = cls.stratification
    return (
        CORROBORATION_ORDER.index(strat.corroboration_level),
        -strat.n_instruments,
        -strat.n_members,
        cls.structure_key,
    )


def _build_class(
    structure_key: str,
    members: tuple[ClassMember, ...],
    *,
    context_available: bool,
    dataset_instruments: Sequence[str],
    dataset_topology: DataTopology | None,
    dataset_lhs_term: str | None,
    absent_outcomes: Mapping[str, AbsentReason],
    schema_by_instrument: Mapping[str, Mapping[str, Any]],
    policy: ConsensusPolicy,
) -> ConsensusClass:
    terms = members[0].signature.terms
    present = frozenset(member.instrument for member in members)
    eligibility = _class_eligibility(
        terms,
        present,
        dataset_instruments,
        dataset_topology,
        dataset_lhs_term,
        absent_outcomes,
        schema_by_instrument,
        policy,
    )
    axes = ClassAxes(
        structure=_structure_axis(members),
        support=_support_axis(members, policy),
        coefficient=_coefficient_axis(members, policy),
        empirical=_empirical_axis(members, policy, context_available),
    )
    return ConsensusClass(
        structure_key=structure_key,
        terms=terms,
        native_forms_present=tuple(
            sorted({member.signature.native_form.value for member in members})
        ),
        members=members,
        axes=axes,
        stratification=_stratification(members, eligibility),
        flags=_member_flags(members, policy),
    )


def _build_dataset_node(
    ref: str,
    *,
    store: EvidenceStore,
    signables: Sequence[_SignableRecord],
    strata: Mapping[int, DependencyStratum],
    provided: Mapping[str, PDEDataset],
    factory: VerifyContextFactory,
    fingerprint_by_ref: Mapping[str, str],
    schema_by_instrument: Mapping[str, Mapping[str, Any]],
    policy: ConsensusPolicy,
    provider_kinds_seen: set[str],
) -> DatasetConsensus:
    plan = store.plan
    signables_for_ref = [s for s in signables if s.entry.dataset_ref == ref]
    equations = {s.entry_index: s.equation for s in signables_for_ref}

    execution = None
    if ref not in provided:
        empirical_state = "no_dataset"
        verifications = verify_members(equations, None, policy=policy.verify)
    elif not signables_for_ref:
        empirical_state = "available"
        verifications = {}
    else:
        try:
            execution = factory(provided[ref])
        except Exception as exc:
            empirical_state = "context_build_failed"
            verifications = {
                entry_index: MemberVerification(
                    status="failed",
                    stage="context_build",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    report=None,
                )
                for entry_index in equations
            }
        else:
            provider_kinds_seen.add(execution.provider_kind)
            empirical_state = "available"
            verifications = verify_members(
                equations, execution, policy=policy.verify
            )

    context_available = execution is not None
    members_by_key: dict[str, list[ClassMember]] = {}
    for signable in signables_for_ref:
        evidence = signable.record.evidence
        member = ClassMember(
            entry_index=signable.entry_index,
            instrument=signable.entry.instrument,
            seed=signable.entry.seed,
            signature=signable.signature,
            dependency=DependencyProfile(stratum=strata[signable.entry_index]),
            native_score_kind=evidence.score_kind,
            native_score_direction=evidence.score_direction,
            native_score=evidence.score,
            native_nmse=evidence.nmse,
            native_r2=evidence.r2,
            headline_coefficient_source=evidence.headline_coefficient_source,
            verification=verifications[signable.entry_index],
        )
        members_by_key.setdefault(signable.signature.structure_key, []).append(member)

    dataset = provided.get(ref)
    dataset_instruments = _instruments_for_ref(plan, ref)
    dataset_topology = dataset.topology if dataset is not None else None
    dataset_lhs_term = _resolve_lhs_term(signables_for_ref, dataset)
    absent_outcomes = _absent_outcomes(
        ref, plan, signables, store, dataset_instruments
    )

    classes = [
        _build_class(
            structure_key,
            tuple(sorted(members, key=lambda m: m.entry_index)),
            context_available=context_available,
            dataset_instruments=dataset_instruments,
            dataset_topology=dataset_topology,
            dataset_lhs_term=dataset_lhs_term,
            absent_outcomes=absent_outcomes,
            schema_by_instrument=schema_by_instrument,
            policy=policy,
        )
        for structure_key, members in members_by_key.items()
    ]
    classes.sort(key=_class_sort_key)

    return DatasetConsensus(
        dataset_ref=ref,
        dataset_fingerprint=fingerprint_by_ref.get(ref),
        empirical_state=empirical_state,
        classes=tuple(classes),
        adjacency=_adjacency(classes),
        instrument_stability=_instrument_stability(
            ref, plan, signables_for_ref, classes, store, dataset_instruments
        ),
    )


def build_consensus(
    store: EvidenceStore,
    *,
    datasets: Mapping[str, PDEDataset] | None = None,
    policy: ConsensusPolicy = ConsensusPolicy(),
    instrument_schemas: Sequence[Mapping[str, Any]] | None = None,
    context_factory: VerifyContextFactory | None = None,
) -> ConsensusReport:


    if (
        policy.stratum_rules_version != STRATUM_RULES_VERSION
        or policy.eligibility_rules_version != ELIGIBILITY_RULES_VERSION
    ):
        raise ValueError(
            "consensus policy rule-table version(s) not implemented by v1: "
            f"stratum={policy.stratum_rules_version!r}, "
            f"eligibility={policy.eligibility_rules_version!r}"
        )

    if instrument_schemas is None:
        from kd.api import instrument_schemas as _default_instrument_schemas

        raw_schemas: Sequence[Mapping[str, Any]] = _default_instrument_schemas()
    else:
        raw_schemas = instrument_schemas
    schema_by_instrument: dict[str, Mapping[str, Any]] = {}
    for schema in raw_schemas:
        if "algorithm" not in schema or "modes" not in schema:
            raise ValueError(
                "each instrument schema must carry 'algorithm' and 'modes' keys"
            )
        schema_by_instrument[schema["algorithm"]] = schema

    factory = (
        context_factory
        if context_factory is not None
        else default_verify_context_factory
    )

    plan = store.plan
    ref_order: list[str] = []
    ref_seen: set[str] = set()
    for entry in plan.entries:
        if entry.dataset_ref not in ref_seen:
            ref_seen.add(entry.dataset_ref)
            ref_order.append(entry.dataset_ref)

    raw_provided: Mapping[str, PDEDataset] = datasets or {}
    for ref in raw_provided:
        if ref not in ref_seen:
            raise ValueError(
                f"datasets ref {ref!r} is not a plan dataset_ref: {sorted(ref_seen)!r}"
            )






    provided: dict[str, PDEDataset] = {
        ref: resolve_lhs_defaults(dataset) for ref, dataset in raw_provided.items()
    }

    signables, unsignable = _partition_signable(store)
    strata: dict[int, DependencyStratum] = {
        signable.entry_index: _dependency_stratum(
            signable.entry.instrument, signable.record.run_spec, schema_by_instrument
        )
        for signable in signables
    }

    fingerprint_by_ref: dict[str, str] = {}
    for entry_index in sorted(store.records):
        ref = plan.entries[entry_index].dataset_ref
        if ref not in fingerprint_by_ref:
            fingerprint_by_ref[ref] = store.records[
                entry_index
            ].evidence.dataset_cache_fingerprint






    provenance_fingerprints: dict[str, str] = dict(fingerprint_by_ref)
    for ref, dataset in provided.items():
        supplied = compute_dataset_fingerprint(dataset)
        recorded = fingerprint_by_ref.get(ref)
        if recorded is not None and supplied != recorded:
            raise ValueError(
                f"provided dataset for ref {ref!r} has fingerprint {supplied!r} "
                f"but the store recorded {recorded!r}"
            )
        provenance_fingerprints[ref] = supplied

    provider_kinds_seen: set[str] = set()
    dataset_nodes = tuple(
        _build_dataset_node(
            ref,
            store=store,
            signables=signables,
            strata=strata,
            provided=provided,
            factory=factory,
            fingerprint_by_ref=fingerprint_by_ref,
            schema_by_instrument=schema_by_instrument,
            policy=policy,
            provider_kinds_seen=provider_kinds_seen,
        )
        for ref in ref_order
    )

    if len(provider_kinds_seen) > 1:
        raise ValueError(
            "verifier provider_kind disagreement across datasets "
            f"({sorted(provider_kinds_seen)!r}): a report promises one provider_kind"
        )




    declared_kind = getattr(factory, "provider_kind", None)
    if not isinstance(declared_kind, str):
        declared_kind = None

    verifier = None
    if provided:
        provider_kind: str | None
        if provider_kinds_seen:
            observed_kind = next(iter(provider_kinds_seen))
            if declared_kind is not None and declared_kind != observed_kind:
                raise ValueError(
                    "verifier context factory declared provider_kind "
                    f"{declared_kind!r} but its VerifyExecution reported "
                    f"{observed_kind!r}: a factory may not lie about its kind"
                )
            provider_kind = observed_kind
        else:
            provider_kind = declared_kind
        verifier = VerifierProvenance(provider_kind=provider_kind)

    env = store.env
    provenance = ConsensusProvenance(
        schema=CONSENSUS_ARTIFACT_TAG,


        store_root=str(Path(store.root).resolve()),
        plan_hash=store.plan_hash,
        plan_name=plan.name,
        kd_version=env.get("kd_version"),
        env=tuple(sorted(env.items())),
        dataset_fingerprints=tuple(
            (ref, provenance_fingerprints.get(ref)) for ref in ref_order
        ),
        stratum_rules_version=policy.stratum_rules_version,
        eligibility_rules_version=policy.eligibility_rules_version,
        verify_policy=policy.verify,
        native_nmse_flag_atol=policy.native_nmse_flag_atol,
        inactive_mass_flag_max=policy.inactive_mass_flag_max,
        datasets_provided=tuple(sorted(provided)),
        verifier=verifier,
    )

    attempts = store.attempts
    store_level = StoreLevelSummary(
        n_attempts=len(attempts),
        n_completed=sum(1 for a in attempts if a["status"] == "completed"),
        n_raised=sum(1 for a in attempts if a["status"] == "raised"),
        n_no_record=sum(1 for a in attempts if a["status"] == "no_record"),
        unsignable=tuple(sorted(unsignable, key=lambda u: u.entry_index)),
    )

    return ConsensusReport(
        provenance=provenance, datasets=dataset_nodes, store_level=store_level
    )
