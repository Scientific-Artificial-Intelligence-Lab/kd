
from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from itertools import combinations

from kd.core.equation.types import Form, LhsSpec
from kd.core.verify import empirical_agreement, law_agreement
from kd.harness._consensus_partition import _SignableRecord
from kd.harness.consensus import (
    ClassAdjacency,
    ClassMember,
    ClassShare,
    CoefficientAxis,
    CoefficientPair,
    ConsensusClass,
    ConsensusPolicy,
    EmpiricalAxis,
    EmpiricalPair,
    EmpiricalSubgroup,
    InstrumentStability,
    MemberFlag,
    StructureAxis,
    SupportAxis,
    SupportVariant,
)
from kd.harness.plan import ExperimentPlan
from kd.harness.store import EvidenceStore


def _structure_axis(members: Sequence[ClassMember]) -> StructureAxis:
    n = len(members)
    if n >= 2:
        return StructureAxis(state="evaluated", agreed=True, n_members=n)
    return StructureAxis(state="unreplicated", agreed=None, n_members=n)


def _support_axis(
    members: Sequence[ClassMember], policy: ConsensusPolicy
) -> SupportAxis:
    if len(members) < 2:
        return SupportAxis(
            state="not_applicable",
            all_agree=None,
            n_pairs=None,
            n_agree_pairs=None,
            variants=(),
        )
    n_pairs = 0
    n_agree = 0
    all_agree = True
    for a, b in combinations(members, 2):
        agreement = law_agreement(a.signature, b.signature, policy=policy.verify)
        n_pairs += 1
        if agreement.support:
            n_agree += 1
        else:
            all_agree = False
    groups: dict[tuple[Form, LhsSpec | None], list[int]] = {}
    for member in members:
        key = (member.signature.native_form, member.signature.native_lhs)
        groups.setdefault(key, []).append(member.entry_index)
    variants = [
        SupportVariant(
            native_form=form,
            native_lhs=lhs,
            entry_indices=tuple(sorted(indices)),
        )
        for (form, lhs), indices in groups.items()
    ]
    variants.sort(key=_variant_sort_key)
    return SupportAxis(
        state="evaluated",
        all_agree=all_agree,
        n_pairs=n_pairs,
        n_agree_pairs=n_agree,
        variants=tuple(variants),
    )


def _variant_sort_key(variant: SupportVariant) -> tuple[str, int, str, str, int]:
    lhs = variant.native_lhs
    return (
        variant.native_form.value,
        0 if lhs is None else 1,
        lhs.field if lhs is not None else "",
        lhs.axis if lhs is not None else "",
        lhs.order if lhs is not None else 0,
    )


def _coefficient_axis(
    members: Sequence[ClassMember], policy: ConsensusPolicy
) -> CoefficientAxis:
    if len(members) < 2:
        return CoefficientAxis(
            state="not_applicable",
            all_within_atol=None,
            max_delta=None,
            median_delta=None,
            n_pairs=None,
            n_agree_pairs=None,
            pairs=(),
        )
    pairs: list[CoefficientPair] = []
    deltas: list[float] = []
    n_agree = 0
    all_within = True
    for a, b in combinations(members, 2):
        agreement = law_agreement(a.signature, b.signature, policy=policy.verify)
        delta = agreement.max_abs_delta
        within = agreement.coefficient
        if delta is None or within is None:
            raise ValueError(
                "intra-class coefficient comparison must yield a delta and verdict"
            )
        pairs.append(
            CoefficientPair(
                entry_index_a=a.entry_index,
                entry_index_b=b.entry_index,
                max_abs_delta=delta,
                within_atol=within,
            )
        )
        deltas.append(delta)
        if within:
            n_agree += 1
        else:
            all_within = False
    return CoefficientAxis(
        state="evaluated",
        all_within_atol=all_within,
        max_delta=max(deltas),
        median_delta=statistics.median(deltas),
        n_pairs=len(pairs),
        n_agree_pairs=n_agree,
        pairs=tuple(pairs),
    )


def _empirical_axis(
    members: Sequence[ClassMember],
    policy: ConsensusPolicy,
    context_available: bool,
) -> EmpiricalAxis:
    n_unverified = sum(
        1 for member in members if member.verification.status != "verified"
    )
    verified = [
        member
        for member in members
        if member.verification.status == "verified"
        and member.verification.report is not None
    ]
    if not context_available:
        return _empirical_degenerate("not_evaluated", n_unverified=n_unverified)
    if len(members) == 1:
        return _empirical_degenerate("not_applicable", n_unverified=n_unverified)
    if len(verified) == 0:
        return _empirical_degenerate("not_evaluated", n_unverified=n_unverified)
    if len(verified) == 1:
        return _empirical_degenerate("not_applicable", n_unverified=n_unverified)
    groups: dict[str, list[ClassMember]] = {}
    for member in verified:
        report = member.verification.report
        assert report is not None
        groups.setdefault(report.normalizer_term, []).append(member)
    subgroups: list[EmpiricalSubgroup] = []
    total_intra_pairs = 0
    for term in sorted(groups):
        group = sorted(groups[term], key=lambda m: m.entry_index)
        pairs: list[EmpiricalPair] = []
        n_agree = 0
        for a, b in combinations(group, 2):
            rep_a = a.verification.report
            rep_b = b.verification.report
            assert rep_a is not None and rep_b is not None





            verdict = empirical_agreement(rep_a, rep_b, policy=policy.verify)
            if verdict is None:
                raise ValueError(
                    "intra-subgroup empirical comparison must yield a verdict "
                    "(same dataset/normalizer basis cannot be incomparable)"
                )
            agree = verdict
            pairs.append(
                EmpiricalPair(
                    entry_index_a=a.entry_index,
                    entry_index_b=b.entry_index,
                    nmse_delta=abs(rep_a.nmse - rep_b.nmse),
                    agree=agree,
                )
            )
            if agree:
                n_agree += 1
        total_intra_pairs += len(pairs)
        subgroups.append(
            EmpiricalSubgroup(
                normalizer_term=term,
                entry_indices=tuple(member.entry_index for member in group),
                n_pairs=len(pairs),
                n_agree_pairs=n_agree,
                pairs=tuple(pairs),
            )
        )
    n_verified = len(verified)
    total_pairs = n_verified * (n_verified - 1) // 2
    return EmpiricalAxis(
        state="evaluated",
        subgroups=tuple(subgroups),
        n_not_comparable_pairs=total_pairs - total_intra_pairs,
        n_unverified_members=n_unverified,
    )


def _empirical_degenerate(
    state: str, *, n_unverified: int | None = None
) -> EmpiricalAxis:
    return EmpiricalAxis(
        state=state,
        subgroups=(),
        n_not_comparable_pairs=None,
        n_unverified_members=n_unverified,
    )


def _member_flags(
    members: Sequence[ClassMember], policy: ConsensusPolicy
) -> tuple[MemberFlag, ...]:
    flags: list[MemberFlag] = []
    for member in members:
        if member.verification.status != "verified":
            continue
        report = member.verification.report
        if report is None:
            continue
        if member.native_nmse is not None:
            value = abs(member.native_nmse - report.nmse)
            if value > policy.native_nmse_flag_atol:
                flags.append(
                    MemberFlag(
                        entry_index=member.entry_index,
                        flag="native_nmse_divergence",
                        value=value,
                    )
                )
        mass = report.inactive_coefficient_mass
        if mass > policy.inactive_mass_flag_max:
            flags.append(
                MemberFlag(
                    entry_index=member.entry_index,
                    flag="high_inactive_mass",
                    value=mass,
                )
            )
    flags.sort(key=lambda flag: (flag.entry_index, flag.flag))
    return tuple(flags)


def _adjacency(classes: Sequence[ConsensusClass]) -> tuple[ClassAdjacency, ...]:
    adjacency: list[ClassAdjacency] = []
    for i in range(len(classes)):
        terms_a = set(classes[i].terms)
        for j in range(i + 1, len(classes)):
            terms_b = set(classes[j].terms)
            intersection = terms_a & terms_b
            union = terms_a | terms_b
            if terms_a < terms_b:
                relation = "subset"
            elif terms_a > terms_b:
                relation = "superset"
            elif not intersection:
                relation = "disjoint"
            else:
                relation = "overlap"
            adjacency.append(
                ClassAdjacency(
                    structure_key_a=classes[i].structure_key,
                    structure_key_b=classes[j].structure_key,
                    relation=relation,
                    jaccard=len(intersection) / len(union),
                )
            )
    return tuple(adjacency)


def _instrument_stability(
    ref: str,
    plan: ExperimentPlan,
    signables_for_ref: Sequence[_SignableRecord],
    classes: Sequence[ConsensusClass],
    store: EvidenceStore,
    dataset_instruments: Sequence[str],
) -> tuple[InstrumentStability, ...]:
    completed_by_instr: Counter[str] = Counter()
    for entry_index in store.records:
        entry = plan.entries[entry_index]
        if entry.dataset_ref == ref:
            completed_by_instr[entry.instrument] += 1
    signable_keys_by_instr: dict[str, list[str]] = {}
    for signable in signables_for_ref:
        signable_keys_by_instr.setdefault(signable.entry.instrument, []).append(
            signable.signature.structure_key
        )
    display_keys = [cls.structure_key for cls in classes]
    stability: list[InstrumentStability] = []
    for instrument in dataset_instruments:
        keys = signable_keys_by_instr.get(instrument, [])
        n_signable = len(keys)
        counts: Counter[str] = Counter(keys)
        class_counts = tuple(
            ClassShare(structure_key=key, n_members=counts[key])
            for key in display_keys
            if counts.get(key, 0) > 0
        )
        modal_key: str | None = None
        modal_share: float | None = None
        if n_signable > 0:
            best_count = -1
            for key in display_keys:
                count = counts.get(key, 0)
                if count > best_count:
                    best_count = count
                    modal_key = key
            modal_share = best_count / n_signable
        stability.append(
            InstrumentStability(
                instrument=instrument,
                n_completed=completed_by_instr.get(instrument, 0),
                n_signable=n_signable,
                class_counts=class_counts,
                modal_structure_key=modal_key,
                modal_share=modal_share,
            )
        )
    return tuple(stability)
