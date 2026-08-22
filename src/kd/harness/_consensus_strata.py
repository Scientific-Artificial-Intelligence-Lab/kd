
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kd.core.equation.canonical import canonicalize_expression
from kd.core.equation.rendering import render_lhs_label
from kd.core.equation.types import Form, LhsSpec
from kd.data.schema import DataTopology, PDEDataset
from kd.harness._consensus_partition import (
    _DECLARED_STRATA,
    _as_stratum,
    _SignableRecord,
)
from kd.harness.consensus import (
    AbsentInstrument,
    AbsentReason,
    ClassEligibility,
    ClassMember,
    ClassStratification,
    ConsensusPolicy,
    CorroborationLevel,
    StratumCount,
)
from kd.harness.plan import ExperimentPlan
from kd.harness.store import EvidenceStore


def _corroboration_level(members: Sequence[ClassMember]) -> CorroborationLevel:
    if len(members) == 1:
        return "single_run"
    instruments = {member.instrument for member in members}
    if len(instruments) == 1:
        return "single_instrument"
    declared = {
        member.dependency.stratum
        for member in members
        if member.dependency.stratum in _DECLARED_STRATA
    }
    if len(declared) >= 2:
        return "multi_instrument_cross_boundary"
    return "multi_instrument_correlated"


def _stratification(
    members: Sequence[ClassMember], eligibility: ClassEligibility
) -> ClassStratification:
    instruments = {member.instrument for member in members}
    member_counts: dict[str, int] = {}
    instr_by_stratum: dict[str, set[str]] = {}
    for member in members:
        stratum = member.dependency.stratum
        member_counts[stratum] = member_counts.get(stratum, 0) + 1
        instr_by_stratum.setdefault(stratum, set()).add(member.instrument)
    strata_member_counts = tuple(
        StratumCount(stratum=_as_stratum(name), count=member_counts[name])
        for name in sorted(member_counts)
    )
    strata_instrument_counts = tuple(
        StratumCount(stratum=_as_stratum(name), count=len(instr_by_stratum[name]))
        for name in sorted(instr_by_stratum)
    )
    return ClassStratification(
        n_members=len(members),
        n_instruments=len(instruments),
        strata_member_counts=strata_member_counts,
        strata_instrument_counts=strata_instrument_counts,
        corroboration_level=_corroboration_level(members),
        eligibility=eligibility,
    )


def _classify_instrument(
    schema: Mapping[str, Any],
    cls_terms: tuple[str, ...],
    topology: DataTopology | None,
    lhs_term: str | None,
) -> str:
    mode_results: list[str] = []
    for mode in schema["modes"]:

        topologies = mode["topologies"]
        if topology is None:
            e1 = "unknown"
        elif topology.value in topologies:
            e1 = "pass"
        else:
            e1 = "fail"

        forms = mode["forms"]
        if "HOMOGENEOUS" in forms or (
            "REGRESSION" in forms and topology is DataTopology.TABULAR
        ):
            e2 = "pass"
        elif lhs_term is None:
            e2 = "unknown"
        elif lhs_term == "":
            e2 = "fail"
        elif lhs_term in cls_terms:
            e2 = "pass"
        else:
            e2 = "fail"

        if e1 == "fail" or e2 == "fail":
            mode_results.append("fail")
        elif e1 == "unknown" or e2 == "unknown":
            mode_results.append("unknown")
        else:
            mode_results.append("pass")

    if "pass" in mode_results:
        return "eligible"
    if "unknown" in mode_results:
        return "unknown"
    return "ineligible"


def _class_eligibility(
    cls_terms: tuple[str, ...],
    present_instruments: frozenset[str],
    dataset_instruments: Sequence[str],
    dataset_topology: DataTopology | None,
    dataset_lhs_term: str | None,
    absent_outcomes: Mapping[str, AbsentReason],
    schema_by_instrument: Mapping[str, Mapping[str, Any]],
    policy: ConsensusPolicy,
) -> ClassEligibility:
    rule_conflicts: list[str] = []
    for instrument in sorted(present_instruments):
        schema = schema_by_instrument.get(instrument)
        if schema is not None and (
            _classify_instrument(schema, cls_terms, dataset_topology, dataset_lhs_term)
            == "ineligible"
        ):
            rule_conflicts.append(instrument)

    eligible_absent: list[AbsentInstrument] = []
    ineligible: list[str] = []
    unknown: list[str] = []
    absent = [i for i in dataset_instruments if i not in present_instruments]
    for instrument in sorted(absent):
        schema = schema_by_instrument.get(instrument)
        verdict = (
            _classify_instrument(
                schema, cls_terms, dataset_topology, dataset_lhs_term
            )
            if schema is not None
            else "unknown"
        )
        if verdict == "eligible":
            eligible_absent.append(
                AbsentInstrument(
                    instrument=instrument, reason=absent_outcomes[instrument]
                )
            )
        elif verdict == "ineligible":
            ineligible.append(instrument)
        else:
            unknown.append(instrument)
    return ClassEligibility(
        rules_version=policy.eligibility_rules_version,
        eligible_present=tuple(sorted(present_instruments)),
        eligible_absent=tuple(eligible_absent),
        ineligible=tuple(ineligible),
        unknown=tuple(unknown),
        rule_conflicts=tuple(rule_conflicts),
    )


def _resolve_lhs_term(
    signables_for_ref: Sequence[_SignableRecord],
    dataset: PDEDataset | None,
) -> str | None:
    observed_lhs = [
        s.signature.native_lhs
        for s in signables_for_ref
        if s.signature.native_form in {Form.EVOLUTION, Form.REGRESSION}
    ]
    observed: str | None = None
    if observed_lhs:
        first = observed_lhs[0]
        if first is not None and all(lhs == first for lhs in observed_lhs):
            observed = (
                canonicalize_expression(first.field)
                if first.order == 0
                else canonicalize_expression(render_lhs_label(first))
            )
    data: str | None = None
    if dataset is not None:
        if (
            dataset.topology is DataTopology.TABULAR
            and dataset.lhs_order == 0
        ):
            data = canonicalize_expression(dataset.lhs_field)
        elif dataset.lhs_order == 0:
            data = ""
        elif dataset.lhs_order >= 1 and len(dataset.lhs_axis) == 1:
            data = canonicalize_expression(
                render_lhs_label(
                    LhsSpec(
                        field=dataset.lhs_field,
                        axis=dataset.lhs_axis,
                        order=dataset.lhs_order,
                    )
                )
            )
    if observed is not None and data is not None:
        return observed if observed == data else None
    if observed is not None:
        return observed
    return data


def _absent_outcomes(
    ref: str,
    plan: ExperimentPlan,
    signables: Sequence[_SignableRecord],
    store: EvidenceStore,
    dataset_instruments: Sequence[str],
) -> dict[str, AbsentReason]:
    signable_instr = {
        s.entry.instrument for s in signables if s.entry.dataset_ref == ref
    }
    completed_instr: set[str] = set()
    for entry_index in store.records:
        entry = plan.entries[entry_index]
        if entry.dataset_ref == ref:
            completed_instr.add(entry.instrument)
    outcomes: dict[str, AbsentReason] = {}
    for instrument in dataset_instruments:
        if instrument in signable_instr:
            outcomes[instrument] = "other_class"
        elif instrument in completed_instr:
            outcomes[instrument] = "unsignable"
        else:
            outcomes[instrument] = "failed"
    return outcomes


def _instruments_for_ref(plan: ExperimentPlan, ref: str) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for entry in plan.entries:
        if entry.dataset_ref == ref and entry.instrument not in seen:
            seen.add(entry.instrument)
            order.append(entry.instrument)
    return order
