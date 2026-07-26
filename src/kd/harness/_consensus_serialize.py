
from __future__ import annotations

from typing import Any, Final

from kd.harness.consensus import (
    ClassAdjacency,
    ClassEligibility,
    ClassMember,
    ClassStratification,
    CoefficientAxis,
    ConsensusClass,
    ConsensusProvenance,
    ConsensusReport,
    DatasetConsensus,
    EmpiricalAxis,
    InstrumentStability,
    StoreLevelSummary,
    StructureAxis,
    SupportAxis,
)
from kd.harness.consensus_verify import MemberVerification




_V1_REPORT_KEYS: Final[frozenset[str]] = frozenset(
    {"provenance", "datasets", "store_level"}
)
_V1_PROVENANCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema", "store_root", "plan_hash", "plan_name", "kd_version",
        "env", "dataset_fingerprints", "stratum_rules_version",
        "eligibility_rules_version", "verify_policy", "native_nmse_flag_atol",
        "inactive_mass_flag_max", "datasets_provided", "verifier",
    }
)
_V1_VERIFIER_KEYS: Final[frozenset[str]] = frozenset({"provider_kind"})
_V1_VERIFY_POLICY_KEYS: Final[frozenset[str]] = frozenset(
    {"nmse_max", "coeff_atol", "empirical_atol", "pivot_unity_rtol"}
)
_V1_DATASET_FINGERPRINT_KEYS: Final[frozenset[str]] = frozenset(
    {"dataset_ref", "dataset_cache_fingerprint"}
)
_V1_DATASET_KEYS: Final[frozenset[str]] = frozenset(
    {
        "dataset_ref", "dataset_fingerprint", "empirical_state",
        "classes", "adjacency", "instrument_stability",
    }
)
_V1_CLASS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "structure_key", "terms", "native_forms_present", "members",
        "axes", "stratification", "flags",
    }
)
_V1_MEMBER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "entry_index", "instrument", "seed", "native_form", "native_lhs",
        "normalized_coefficients", "native", "dependency", "verification",
    }
)
_V1_MEMBER_NATIVE_KEYS: Final[frozenset[str]] = frozenset(
    {"score_kind", "score_direction", "score", "nmse", "r2",
     "headline_coefficient_source"}
)
_V1_DEPENDENCY_KEYS: Final[frozenset[str]] = frozenset({"stratum"})
_V1_VERIFICATION_KEYS: Final[frozenset[str]] = frozenset(
    {"status", "stage", "error_type", "error_message", "measurements"}
)
_V1_MEASUREMENTS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "mse", "nmse", "r2", "residual_mean", "residual_std", "residual_max_abs",
        "n_samples", "inactive_coefficient_mass", "normalizer_term",
        "normalizer_variance", "passed",
    }
)
_V1_AXES_KEYS: Final[frozenset[str]] = frozenset(
    {"structure", "support", "coefficient", "empirical"}
)
_V1_STRUCTURE_KEYS: Final[frozenset[str]] = frozenset(
    {"state", "agreed", "n_members"}
)
_V1_SUPPORT_KEYS: Final[frozenset[str]] = frozenset(
    {"state", "all_agree", "n_pairs", "n_agree_pairs", "variants"}
)
_V1_SUPPORT_VARIANT_KEYS: Final[frozenset[str]] = frozenset(
    {"native_form", "native_lhs", "entry_indices"}
)
_V1_LHS_KEYS: Final[frozenset[str]] = frozenset({"field", "axis", "order"})
_V1_COEFFICIENT_KEYS: Final[frozenset[str]] = frozenset(
    {"state", "all_within_atol", "max_delta", "median_delta",
     "n_pairs", "n_agree_pairs", "pairs"}
)
_V1_COEFF_PAIR_KEYS: Final[frozenset[str]] = frozenset(
    {"entry_index_a", "entry_index_b", "max_abs_delta", "within_atol"}
)
_V1_EMPIRICAL_KEYS: Final[frozenset[str]] = frozenset(
    {"state", "subgroups", "n_not_comparable_pairs", "n_unverified_members"}
)
_V1_SUBGROUP_KEYS: Final[frozenset[str]] = frozenset(
    {"normalizer_term", "entry_indices", "n_pairs", "n_agree_pairs", "pairs"}
)
_V1_EMP_PAIR_KEYS: Final[frozenset[str]] = frozenset(
    {"entry_index_a", "entry_index_b", "nmse_delta", "agree"}
)
_V1_STRATIFICATION_KEYS: Final[frozenset[str]] = frozenset(
    {"n_members", "n_instruments", "strata_member_counts",
     "strata_instrument_counts", "corroboration_level", "eligibility"}
)
_V1_STRATUM_MEMBER_COUNT_KEYS: Final[frozenset[str]] = frozenset(
    {"stratum", "n_members"}
)
_V1_STRATUM_INSTRUMENT_COUNT_KEYS: Final[frozenset[str]] = frozenset(
    {"stratum", "n_instruments"}
)
_V1_ELIGIBILITY_KEYS: Final[frozenset[str]] = frozenset(
    {"rules_version", "eligible_present", "eligible_absent",
     "ineligible", "unknown", "rule_conflicts"}
)
_V1_ABSENT_KEYS: Final[frozenset[str]] = frozenset({"instrument", "reason"})
_V1_FLAG_KEYS: Final[frozenset[str]] = frozenset({"entry_index", "flag", "value"})
_V1_ADJACENCY_KEYS: Final[frozenset[str]] = frozenset(
    {"structure_key_a", "structure_key_b", "relation", "jaccard"}
)
_V1_STABILITY_KEYS: Final[frozenset[str]] = frozenset(
    {"instrument", "n_completed", "n_signable", "class_counts",
     "modal_structure_key", "modal_share"}
)
_V1_CLASS_SHARE_KEYS: Final[frozenset[str]] = frozenset(
    {"structure_key", "n_members"}
)
_V1_STORE_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {"n_attempts", "n_completed", "n_raised", "n_no_record", "unsignable"}
)
_V1_UNSIGNABLE_KEYS: Final[frozenset[str]] = frozenset(
    {"entry_index", "instrument", "dataset_ref", "seed", "reason",
     "invalid_reason", "error_type", "error_message"}
)
_V1_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {"artifact", "plan_hash", "store_root", "report"}
)





def _lhs_to_dict(lhs: Any) -> dict[str, Any] | None:
    if lhs is None:
        return None
    return {"field": lhs.field, "axis": lhs.axis, "order": lhs.order}


def _measurements_to_dict(report: Any) -> dict[str, Any]:
    return {
        "mse": report.mse,
        "nmse": report.nmse,
        "r2": report.r2,
        "residual_mean": report.residual_mean,
        "residual_std": report.residual_std,
        "residual_max_abs": report.residual_max_abs,
        "n_samples": report.n_samples,
        "inactive_coefficient_mass": report.inactive_coefficient_mass,
        "normalizer_term": report.normalizer_term,
        "normalizer_variance": report.normalizer_variance,
        "passed": report.passed,
    }


def _verification_to_dict(verification: MemberVerification) -> dict[str, Any]:
    report = verification.report
    return {
        "status": verification.status,
        "stage": verification.stage,
        "error_type": verification.error_type,
        "error_message": verification.error_message,
        "measurements": _measurements_to_dict(report) if report is not None else None,
    }


def _member_to_dict(member: ClassMember) -> dict[str, Any]:
    signature = member.signature
    return {
        "entry_index": member.entry_index,
        "instrument": member.instrument,
        "seed": member.seed,
        "native_form": signature.native_form.value,
        "native_lhs": _lhs_to_dict(signature.native_lhs),
        "normalized_coefficients": list(signature.coefficients),
        "native": {
            "score_kind": member.native_score_kind,
            "score_direction": member.native_score_direction,
            "score": member.native_score,
            "nmse": member.native_nmse,
            "r2": member.native_r2,
            "headline_coefficient_source": member.headline_coefficient_source,
        },
        "dependency": {"stratum": member.dependency.stratum},
        "verification": _verification_to_dict(member.verification),
    }


def _structure_to_dict(axis: StructureAxis) -> dict[str, Any]:
    return {"state": axis.state, "agreed": axis.agreed, "n_members": axis.n_members}


def _support_to_dict(axis: SupportAxis) -> dict[str, Any]:
    return {
        "state": axis.state,
        "all_agree": axis.all_agree,
        "n_pairs": axis.n_pairs,
        "n_agree_pairs": axis.n_agree_pairs,
        "variants": [
            {
                "native_form": variant.native_form.value,
                "native_lhs": _lhs_to_dict(variant.native_lhs),
                "entry_indices": list(variant.entry_indices),
            }
            for variant in axis.variants
        ],
    }


def _coefficient_to_dict(axis: CoefficientAxis) -> dict[str, Any]:
    return {
        "state": axis.state,
        "all_within_atol": axis.all_within_atol,
        "max_delta": axis.max_delta,
        "median_delta": axis.median_delta,
        "n_pairs": axis.n_pairs,
        "n_agree_pairs": axis.n_agree_pairs,
        "pairs": [
            {
                "entry_index_a": pair.entry_index_a,
                "entry_index_b": pair.entry_index_b,
                "max_abs_delta": pair.max_abs_delta,
                "within_atol": pair.within_atol,
            }
            for pair in axis.pairs
        ],
    }


def _empirical_to_dict(axis: EmpiricalAxis) -> dict[str, Any]:
    return {
        "state": axis.state,
        "subgroups": [
            {
                "normalizer_term": subgroup.normalizer_term,
                "entry_indices": list(subgroup.entry_indices),
                "n_pairs": subgroup.n_pairs,
                "n_agree_pairs": subgroup.n_agree_pairs,
                "pairs": [
                    {
                        "entry_index_a": pair.entry_index_a,
                        "entry_index_b": pair.entry_index_b,
                        "nmse_delta": pair.nmse_delta,
                        "agree": pair.agree,
                    }
                    for pair in subgroup.pairs
                ],
            }
            for subgroup in axis.subgroups
        ],
        "n_not_comparable_pairs": axis.n_not_comparable_pairs,
        "n_unverified_members": axis.n_unverified_members,
    }


def _eligibility_to_dict(eligibility: ClassEligibility) -> dict[str, Any]:
    return {
        "rules_version": eligibility.rules_version,
        "eligible_present": list(eligibility.eligible_present),
        "eligible_absent": [
            {"instrument": absent.instrument, "reason": absent.reason}
            for absent in eligibility.eligible_absent
        ],
        "ineligible": list(eligibility.ineligible),
        "unknown": list(eligibility.unknown),
        "rule_conflicts": list(eligibility.rule_conflicts),
    }


def _stratification_to_dict(strat: ClassStratification) -> dict[str, Any]:
    return {
        "n_members": strat.n_members,
        "n_instruments": strat.n_instruments,
        "strata_member_counts": [
            {"stratum": item.stratum, "n_members": item.count}
            for item in strat.strata_member_counts
        ],
        "strata_instrument_counts": [
            {"stratum": item.stratum, "n_instruments": item.count}
            for item in strat.strata_instrument_counts
        ],
        "corroboration_level": strat.corroboration_level,
        "eligibility": _eligibility_to_dict(strat.eligibility),
    }


def _class_to_dict(cls: ConsensusClass) -> dict[str, Any]:
    return {
        "structure_key": cls.structure_key,
        "terms": list(cls.terms),
        "native_forms_present": list(cls.native_forms_present),
        "members": [_member_to_dict(member) for member in cls.members],
        "axes": {
            "structure": _structure_to_dict(cls.axes.structure),
            "support": _support_to_dict(cls.axes.support),
            "coefficient": _coefficient_to_dict(cls.axes.coefficient),
            "empirical": _empirical_to_dict(cls.axes.empirical),
        },
        "stratification": _stratification_to_dict(cls.stratification),
        "flags": [
            {"entry_index": flag.entry_index, "flag": flag.flag, "value": flag.value}
            for flag in cls.flags
        ],
    }


def _adjacency_to_dict(adjacency: ClassAdjacency) -> dict[str, Any]:
    return {
        "structure_key_a": adjacency.structure_key_a,
        "structure_key_b": adjacency.structure_key_b,
        "relation": adjacency.relation,
        "jaccard": adjacency.jaccard,
    }


def _stability_to_dict(stability: InstrumentStability) -> dict[str, Any]:
    return {
        "instrument": stability.instrument,
        "n_completed": stability.n_completed,
        "n_signable": stability.n_signable,
        "class_counts": [
            {"structure_key": share.structure_key, "n_members": share.n_members}
            for share in stability.class_counts
        ],
        "modal_structure_key": stability.modal_structure_key,
        "modal_share": stability.modal_share,
    }


def _dataset_to_dict(dataset: DatasetConsensus) -> dict[str, Any]:
    return {
        "dataset_ref": dataset.dataset_ref,
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "empirical_state": dataset.empirical_state,
        "classes": [_class_to_dict(cls) for cls in dataset.classes],
        "adjacency": [_adjacency_to_dict(a) for a in dataset.adjacency],
        "instrument_stability": [
            _stability_to_dict(s) for s in dataset.instrument_stability
        ],
    }


def _verify_policy_to_dict(policy: Any) -> dict[str, Any]:
    return {
        "nmse_max": policy.nmse_max,
        "coeff_atol": policy.coeff_atol,
        "empirical_atol": policy.empirical_atol,
        "pivot_unity_rtol": policy.pivot_unity_rtol,
    }


def _provenance_to_dict(provenance: ConsensusProvenance) -> dict[str, Any]:
    verifier = provenance.verifier
    return {
        "schema": provenance.schema,
        "store_root": provenance.store_root,
        "plan_hash": provenance.plan_hash,
        "plan_name": provenance.plan_name,
        "kd_version": provenance.kd_version,
        "env": {key: value for key, value in provenance.env},
        "dataset_fingerprints": [
            {"dataset_ref": ref, "dataset_cache_fingerprint": fingerprint}
            for ref, fingerprint in provenance.dataset_fingerprints
        ],
        "stratum_rules_version": provenance.stratum_rules_version,
        "eligibility_rules_version": provenance.eligibility_rules_version,
        "verify_policy": _verify_policy_to_dict(provenance.verify_policy),
        "native_nmse_flag_atol": provenance.native_nmse_flag_atol,
        "inactive_mass_flag_max": provenance.inactive_mass_flag_max,
        "datasets_provided": list(provenance.datasets_provided),
        "verifier": (
            {"provider_kind": verifier.provider_kind}
            if verifier is not None
            else None
        ),
    }


def _store_level_to_dict(store_level: StoreLevelSummary) -> dict[str, Any]:
    return {
        "n_attempts": store_level.n_attempts,
        "n_completed": store_level.n_completed,
        "n_raised": store_level.n_raised,
        "n_no_record": store_level.n_no_record,
        "unsignable": [
            {
                "entry_index": entry.entry_index,
                "instrument": entry.instrument,
                "dataset_ref": entry.dataset_ref,
                "seed": entry.seed,
                "reason": entry.reason,
                "invalid_reason": entry.invalid_reason,
                "error_type": entry.error_type,
                "error_message": entry.error_message,
            }
            for entry in store_level.unsignable
        ],
    }


def consensus_to_dict(report: ConsensusReport) -> dict[str, Any]:
    return {
        "provenance": _provenance_to_dict(report.provenance),
        "datasets": [_dataset_to_dict(dataset) for dataset in report.datasets],
        "store_level": _store_level_to_dict(report.store_level),
    }
