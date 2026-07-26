
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from kd.core.equation.signature import LawSignature
from kd.core.equation.types import Form, LhsSpec
from kd.core.verify import VerifyPolicy
from kd.harness.consensus_verify import MemberVerification





CONSENSUS_ARTIFACT_TAG: Final[str] = "kd-consensus-v1"
STRATUM_RULES_VERSION: Final[str] = "kd-stratum-rules-v1"
ELIGIBILITY_RULES_VERSION: Final[str] = "kd-eligibility-rules-v1"


DependencyStratum = Literal["finite_diff", "autograd", "none", "undeclared"]
DEPENDENCY_STRATA: Final[frozenset[str]] = frozenset(
    {"finite_diff", "autograd", "none", "undeclared"}
)



CorroborationLevel = Literal[
    "single_run",
    "single_instrument",
    "multi_instrument_correlated",
    "multi_instrument_cross_boundary",
]
CORROBORATION_ORDER: Final[tuple[str, ...]] = (
    "multi_instrument_cross_boundary",
    "multi_instrument_correlated",
    "single_instrument",
    "single_run",
)


UnsignableReason = Literal["invalid_record", "no_catalog_fit", "signature_error"]
UNSIGNABLE_REASONS: Final[frozenset[str]] = frozenset(
    {"invalid_record", "no_catalog_fit", "signature_error"}
)



AdjacencyRelation = Literal["subset", "superset", "overlap", "disjoint"]


MemberFlagKind = Literal["native_nmse_divergence", "high_inactive_mass"]



AbsentReason = Literal["other_class", "unsignable", "failed"]



CountLevel = Literal["member", "member_pair", "instrument", "attempt", "sample"]

COUNT_FIELD_LEVELS: Final[Mapping[str, CountLevel]] = {
    "n_members": "member",
    "n_unverified_members": "member",
    "n_completed": "member",
    "n_signable": "member",
    "n_instruments": "instrument",
    "n_pairs": "member_pair",
    "n_agree_pairs": "member_pair",
    "n_not_comparable_pairs": "member_pair",
    "n_attempts": "attempt",
    "n_raised": "attempt",
    "n_no_record": "attempt",
    "n_samples": "sample",
}





@dataclass(frozen=True)
class ConsensusPolicy:

    verify: VerifyPolicy = VerifyPolicy()
    stratum_rules_version: str = STRATUM_RULES_VERSION
    eligibility_rules_version: str = ELIGIBILITY_RULES_VERSION
    native_nmse_flag_atol: float = 1e-2
    inactive_mass_flag_max: float = 1e-6

    def __post_init__(self) -> None:
        for version in (self.stratum_rules_version, self.eligibility_rules_version):
            if not isinstance(version, str) or not version:
                raise ValueError(
                    "consensus policy rule-table version strings must be non-empty"
                )
        for threshold in (self.native_nmse_flag_atol, self.inactive_mass_flag_max):
            if not math.isfinite(threshold) or threshold < 0:
                raise ValueError(
                    "consensus policy thresholds must be finite and non-negative"
                )






@dataclass(frozen=True, kw_only=True)
class DependencyProfile:

    stratum: DependencyStratum


@dataclass(frozen=True, kw_only=True)
class ClassMember:

    entry_index: int
    instrument: str
    seed: int
    signature: LawSignature
    dependency: DependencyProfile
    native_score_kind: str
    native_score_direction: str
    native_score: float | None
    native_nmse: float | None
    native_r2: float | None
    headline_coefficient_source: str
    verification: MemberVerification


@dataclass(frozen=True, kw_only=True)
class StructureAxis:

    state: Literal["evaluated", "unreplicated"]
    agreed: bool | None
    n_members: int


@dataclass(frozen=True, kw_only=True)
class SupportVariant:

    native_form: Form
    native_lhs: LhsSpec | None
    entry_indices: tuple[int, ...]


@dataclass(frozen=True, kw_only=True)
class SupportAxis:

    state: Literal["evaluated", "not_applicable"]
    all_agree: bool | None
    n_pairs: int | None
    n_agree_pairs: int | None
    variants: tuple[SupportVariant, ...]


@dataclass(frozen=True, kw_only=True)
class CoefficientPair:

    entry_index_a: int
    entry_index_b: int
    max_abs_delta: float
    within_atol: bool


@dataclass(frozen=True, kw_only=True)
class CoefficientAxis:

    state: Literal["evaluated", "not_applicable"]
    all_within_atol: bool | None
    max_delta: float | None
    median_delta: float | None
    n_pairs: int | None
    n_agree_pairs: int | None
    pairs: tuple[CoefficientPair, ...]


@dataclass(frozen=True, kw_only=True)
class EmpiricalPair:

    entry_index_a: int
    entry_index_b: int
    nmse_delta: float
    agree: bool


@dataclass(frozen=True, kw_only=True)
class EmpiricalSubgroup:

    normalizer_term: str
    entry_indices: tuple[int, ...]
    n_pairs: int
    n_agree_pairs: int
    pairs: tuple[EmpiricalPair, ...]


@dataclass(frozen=True, kw_only=True)
class EmpiricalAxis:

    state: Literal["evaluated", "not_applicable", "not_evaluated"]
    subgroups: tuple[EmpiricalSubgroup, ...]
    n_not_comparable_pairs: int | None
    n_unverified_members: int | None


@dataclass(frozen=True, kw_only=True)
class ClassAxes:

    structure: StructureAxis
    support: SupportAxis
    coefficient: CoefficientAxis
    empirical: EmpiricalAxis


@dataclass(frozen=True, kw_only=True)
class StratumCount:

    stratum: DependencyStratum
    count: int


@dataclass(frozen=True, kw_only=True)
class AbsentInstrument:

    instrument: str
    reason: AbsentReason


@dataclass(frozen=True, kw_only=True)
class ClassEligibility:

    rules_version: str
    eligible_present: tuple[str, ...]
    eligible_absent: tuple[AbsentInstrument, ...]
    ineligible: tuple[str, ...]
    unknown: tuple[str, ...]
    rule_conflicts: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class ClassStratification:

    n_members: int
    n_instruments: int
    strata_member_counts: tuple[StratumCount, ...]
    strata_instrument_counts: tuple[StratumCount, ...]
    corroboration_level: CorroborationLevel
    eligibility: ClassEligibility


@dataclass(frozen=True, kw_only=True)
class MemberFlag:

    entry_index: int
    flag: MemberFlagKind
    value: float


@dataclass(frozen=True, kw_only=True)
class ConsensusClass:

    structure_key: str
    terms: tuple[str, ...]
    native_forms_present: tuple[str, ...]
    members: tuple[ClassMember, ...]
    axes: ClassAxes
    stratification: ClassStratification
    flags: tuple[MemberFlag, ...]


@dataclass(frozen=True, kw_only=True)
class ClassAdjacency:

    structure_key_a: str
    structure_key_b: str
    relation: AdjacencyRelation
    jaccard: float


@dataclass(frozen=True, kw_only=True)
class ClassShare:

    structure_key: str
    n_members: int


@dataclass(frozen=True, kw_only=True)
class InstrumentStability:

    instrument: str
    n_completed: int
    n_signable: int
    class_counts: tuple[ClassShare, ...]
    modal_structure_key: str | None
    modal_share: float | None


@dataclass(frozen=True, kw_only=True)
class DatasetConsensus:

    dataset_ref: str
    dataset_fingerprint: str | None
    empirical_state: Literal["available", "no_dataset", "context_build_failed"]
    classes: tuple[ConsensusClass, ...]
    adjacency: tuple[ClassAdjacency, ...]
    instrument_stability: tuple[InstrumentStability, ...]


@dataclass(frozen=True, kw_only=True)
class UnsignableEntry:

    entry_index: int
    instrument: str
    dataset_ref: str
    seed: int
    reason: UnsignableReason
    invalid_reason: str | None
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True, kw_only=True)
class StoreLevelSummary:

    n_attempts: int
    n_completed: int
    n_raised: int
    n_no_record: int
    unsignable: tuple[UnsignableEntry, ...]


@dataclass(frozen=True, kw_only=True)
class VerifierProvenance:

    provider_kind: str | None


@dataclass(frozen=True, kw_only=True)
class ConsensusProvenance:

    schema: str
    store_root: str
    plan_hash: str
    plan_name: str
    kd_version: str | None
    env: tuple[tuple[str, str], ...]
    dataset_fingerprints: tuple[tuple[str, str | None], ...]
    stratum_rules_version: str
    eligibility_rules_version: str
    verify_policy: VerifyPolicy
    native_nmse_flag_atol: float
    inactive_mass_flag_max: float
    datasets_provided: tuple[str, ...]
    verifier: VerifierProvenance | None


@dataclass(frozen=True, kw_only=True)
class ConsensusReport:

    provenance: ConsensusProvenance
    datasets: tuple[DatasetConsensus, ...]
    store_level: StoreLevelSummary






from kd.harness._consensus_build import build_consensus

__all__ = [
    "AbsentInstrument",
    "AbsentReason",
    "AdjacencyRelation",
    "CONSENSUS_ARTIFACT_TAG",
    "CORROBORATION_ORDER",
    "COUNT_FIELD_LEVELS",
    "ClassAdjacency",
    "ClassAxes",
    "ClassEligibility",
    "ClassMember",
    "ClassShare",
    "ClassStratification",
    "CoefficientAxis",
    "CoefficientPair",
    "ConsensusClass",
    "ConsensusPolicy",
    "ConsensusProvenance",
    "ConsensusReport",
    "CorroborationLevel",
    "CountLevel",
    "DEPENDENCY_STRATA",
    "DatasetConsensus",
    "DependencyProfile",
    "DependencyStratum",
    "ELIGIBILITY_RULES_VERSION",
    "EmpiricalAxis",
    "EmpiricalPair",
    "EmpiricalSubgroup",
    "MemberFlag",
    "MemberFlagKind",
    "STRATUM_RULES_VERSION",
    "StoreLevelSummary",
    "StratumCount",
    "StructureAxis",
    "SupportAxis",
    "SupportVariant",
    "UNSIGNABLE_REASONS",
    "UnsignableEntry",
    "UnsignableReason",
    "VerifierProvenance",
    "build_consensus",
]
