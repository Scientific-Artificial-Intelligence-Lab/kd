
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from kd.core.strict_keys import strict_keys as _strict_keys_core
from kd.harness._consensus_serialize import (
    _V1_ABSENT_KEYS,
    _V1_ADJACENCY_KEYS,
    _V1_ARTIFACT_KEYS,
    _V1_AXES_KEYS,
    _V1_CLASS_KEYS,
    _V1_CLASS_SHARE_KEYS,
    _V1_COEFF_PAIR_KEYS,
    _V1_COEFFICIENT_KEYS,
    _V1_DATASET_FINGERPRINT_KEYS,
    _V1_DATASET_KEYS,
    _V1_DEPENDENCY_KEYS,
    _V1_ELIGIBILITY_KEYS,
    _V1_EMP_PAIR_KEYS,
    _V1_EMPIRICAL_KEYS,
    _V1_FLAG_KEYS,
    _V1_LHS_KEYS,
    _V1_MEASUREMENTS_KEYS,
    _V1_MEMBER_KEYS,
    _V1_MEMBER_NATIVE_KEYS,
    _V1_PROVENANCE_KEYS,
    _V1_REPORT_KEYS,
    _V1_STABILITY_KEYS,
    _V1_STORE_LEVEL_KEYS,
    _V1_STRATIFICATION_KEYS,
    _V1_STRATUM_INSTRUMENT_COUNT_KEYS,
    _V1_STRATUM_MEMBER_COUNT_KEYS,
    _V1_STRUCTURE_KEYS,
    _V1_SUBGROUP_KEYS,
    _V1_SUPPORT_KEYS,
    _V1_SUPPORT_VARIANT_KEYS,
    _V1_UNSIGNABLE_KEYS,
    _V1_VERIFICATION_KEYS,
    _V1_VERIFIER_KEYS,
    _V1_VERIFY_POLICY_KEYS,
    consensus_to_dict,
)
from kd.harness.consensus import (
    CONSENSUS_ARTIFACT_TAG,
    ConsensusClass,
    ConsensusReport,
    DatasetConsensus,
)
from kd.search.mini_table import escape_cell as _escape_cell

__all__ = [
    "ConsensusArtifactError",
    "consensus_to_dict",
    "read_consensus_artifact",
    "render_consensus_markdown",
    "write_consensus_artifact",
]


class ConsensusArtifactError(ValueError):
    pass





def write_consensus_artifact(report: ConsensusReport, *, path: str | Path) -> Path:
    plan_hash = report.provenance.plan_hash
    if not plan_hash:
        raise ValueError("plan_hash must be a non-empty string")
    payload = {
        "artifact": CONSENSUS_ARTIFACT_TAG,
        "plan_hash": plan_hash,
        "store_root": report.provenance.store_root,
        "report": consensus_to_dict(report),
    }
    target = Path(path)


    tmp_path = target.with_name(f"{target.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(tmp_path, target)
    return target


def _check_keys(node: Any, table: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise ConsensusArtifactError(f"{name} must be a JSON object")
    _strict_keys_core(
        node,
        object_name=name,
        required=table,
        error_cls=ConsensusArtifactError,
    )
    return node


def _validate_lhs(node: Any) -> None:
    if node is not None:
        _check_keys(node, _V1_LHS_KEYS, "native_lhs")


def _validate_member(member: Any) -> None:
    _check_keys(member, _V1_MEMBER_KEYS, "member")
    _validate_lhs(member["native_lhs"])
    _check_keys(member["native"], _V1_MEMBER_NATIVE_KEYS, "member.native")
    _check_keys(member["dependency"], _V1_DEPENDENCY_KEYS, "member.dependency")
    verification = _check_keys(
        member["verification"], _V1_VERIFICATION_KEYS, "member.verification"
    )
    measurements = verification["measurements"]
    if measurements is not None:
        _check_keys(measurements, _V1_MEASUREMENTS_KEYS, "measurements")


def _validate_axes(axes: Any) -> None:
    _check_keys(axes, _V1_AXES_KEYS, "axes")
    _check_keys(axes["structure"], _V1_STRUCTURE_KEYS, "axes.structure")
    support = _check_keys(axes["support"], _V1_SUPPORT_KEYS, "axes.support")
    for variant in support["variants"]:
        _check_keys(variant, _V1_SUPPORT_VARIANT_KEYS, "support.variant")
        _validate_lhs(variant["native_lhs"])
    coefficient = _check_keys(
        axes["coefficient"], _V1_COEFFICIENT_KEYS, "axes.coefficient"
    )
    for pair in coefficient["pairs"]:
        _check_keys(pair, _V1_COEFF_PAIR_KEYS, "coefficient.pair")
    empirical = _check_keys(axes["empirical"], _V1_EMPIRICAL_KEYS, "axes.empirical")
    for subgroup in empirical["subgroups"]:
        _check_keys(subgroup, _V1_SUBGROUP_KEYS, "empirical.subgroup")
        for pair in subgroup["pairs"]:
            _check_keys(pair, _V1_EMP_PAIR_KEYS, "empirical.pair")


def _validate_stratification(strat: Any) -> None:
    _check_keys(strat, _V1_STRATIFICATION_KEYS, "stratification")
    for item in strat["strata_member_counts"]:
        _check_keys(item, _V1_STRATUM_MEMBER_COUNT_KEYS, "strata_member_counts")
    for item in strat["strata_instrument_counts"]:
        _check_keys(
            item, _V1_STRATUM_INSTRUMENT_COUNT_KEYS, "strata_instrument_counts"
        )
    eligibility = _check_keys(
        strat["eligibility"], _V1_ELIGIBILITY_KEYS, "eligibility"
    )
    for absent in eligibility["eligible_absent"]:
        _check_keys(absent, _V1_ABSENT_KEYS, "eligible_absent")


def _validate_class(cls: Any) -> None:
    _check_keys(cls, _V1_CLASS_KEYS, "class")
    for member in cls["members"]:
        _validate_member(member)
    _validate_axes(cls["axes"])
    _validate_stratification(cls["stratification"])
    for flag in cls["flags"]:
        _check_keys(flag, _V1_FLAG_KEYS, "flag")


def _validate_dataset(dataset: Any) -> None:
    _check_keys(dataset, _V1_DATASET_KEYS, "dataset")
    for cls in dataset["classes"]:
        _validate_class(cls)
    for adjacency in dataset["adjacency"]:
        _check_keys(adjacency, _V1_ADJACENCY_KEYS, "adjacency")
    for stability in dataset["instrument_stability"]:
        _check_keys(stability, _V1_STABILITY_KEYS, "instrument_stability")
        for share in stability["class_counts"]:
            _check_keys(share, _V1_CLASS_SHARE_KEYS, "class_counts")


def _validate_provenance(provenance: Any) -> None:
    _check_keys(provenance, _V1_PROVENANCE_KEYS, "provenance")
    _check_keys(provenance["verify_policy"], _V1_VERIFY_POLICY_KEYS, "verify_policy")
    for fingerprint in provenance["dataset_fingerprints"]:
        _check_keys(
            fingerprint, _V1_DATASET_FINGERPRINT_KEYS, "dataset_fingerprints"
        )
    verifier = provenance["verifier"]
    if verifier is not None:
        _check_keys(verifier, _V1_VERIFIER_KEYS, "verifier")


def _validate_store_level(store_level: Any) -> None:
    _check_keys(store_level, _V1_STORE_LEVEL_KEYS, "store_level")
    for entry in store_level["unsignable"]:
        _check_keys(entry, _V1_UNSIGNABLE_KEYS, "unsignable")


def _reject_json_constant(token: str) -> Any:
    raise ConsensusArtifactError(
        f"artifact contains a non-standard JSON numeric token {token!r} "
        "(NaN / Infinity / -Infinity are not permitted)"
    )


def read_consensus_artifact(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ConsensusArtifactError(f"artifact is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConsensusArtifactError("artifact must be a JSON object")
    _check_keys(payload, _V1_ARTIFACT_KEYS, "artifact envelope")
    if payload["artifact"] != CONSENSUS_ARTIFACT_TAG:
        raise ConsensusArtifactError(
            f"artifact tag mismatch: got {payload['artifact']!r}, "
            f"expected {CONSENSUS_ARTIFACT_TAG!r}"
        )
    report = _check_keys(payload["report"], _V1_REPORT_KEYS, "report")
    provenance = report["provenance"]
    _validate_provenance(provenance)
    for dataset in report["datasets"]:
        _validate_dataset(dataset)
    _validate_store_level(report["store_level"])


    if payload["plan_hash"] != provenance["plan_hash"]:
        raise ConsensusArtifactError(
            "envelope plan_hash does not match provenance plan_hash: "
            f"{payload['plan_hash']!r} != {provenance['plan_hash']!r}"
        )
    if payload["store_root"] != provenance["store_root"]:
        raise ConsensusArtifactError(
            "envelope store_root does not match provenance store_root: "
            f"{payload['store_root']!r} != {provenance['store_root']!r}"
        )
    if provenance["schema"] != CONSENSUS_ARTIFACT_TAG:
        raise ConsensusArtifactError(
            f"provenance schema {provenance['schema']!r} does not match "
            f"expected {CONSENSUS_ARTIFACT_TAG!r}"
        )
    return payload





def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def _short(value: str | None) -> str:
    return value[:10] if isinstance(value, str) else "n/a"


def _policy_line(report: ConsensusReport) -> str:
    prov = report.provenance
    policy = prov.verify_policy
    return (
        f"policy: nmse_max={_fmt(policy.nmse_max)}, "
        f"coeff_atol={_fmt(policy.coeff_atol)}, "
        f"empirical_atol={_fmt(policy.empirical_atol)}, "
        f"native_nmse_flag_atol={_fmt(prov.native_nmse_flag_atol)}, "
        f"inactive_mass_flag_max={_fmt(prov.inactive_mass_flag_max)}"
    )


def _verifier_line(report: ConsensusReport) -> str:
    verifier = report.provenance.verifier
    if verifier is None:
        return "verifier: not_evaluated (no datasets provided)"
    kind = verifier.provider_kind
    kind_str = _escape_cell(kind) if kind is not None else "n/a"
    return f"verifier: provider_kind={kind_str}"


def _class_lines(cls: ConsensusClass) -> list[str]:
    strat = cls.stratification
    lines = [
        f"### class {_short(cls.structure_key)} [{strat.corroboration_level}] "
        f"({strat.n_members} members / {strat.n_instruments} instruments)"
    ]
    terms = ", ".join(_escape_cell(term) for term in cls.terms)
    lines.append(f"terms: {terms}")
    axes = cls.axes
    lines.append(
        f"axes: structure={axes.structure.state}, support={axes.support.state}, "
        f"coefficient={axes.coefficient.state}, empirical={axes.empirical.state}"
    )
    elig = strat.eligibility
    present = ", ".join(_escape_cell(name) for name in elig.eligible_present)
    absent = ", ".join(_escape_cell(a.instrument) for a in elig.eligible_absent)
    ineligible = ", ".join(_escape_cell(name) for name in elig.ineligible)
    unknown = ", ".join(_escape_cell(name) for name in elig.unknown)
    lines.append(
        f"eligibility ({elig.rules_version}): present=[{present}], "
        f"absent=[{absent}], ineligible=[{ineligible}], unknown=[{unknown}]"
    )
    lines.append(
        "| entry | instrument | seed | stratum | form | score_kind | "
        "native_score | native_nmse | verified_nmse | verify_status |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for member in cls.members:
        report = member.verification.report
        verified_nmse = _fmt(report.nmse) if report is not None else "n/a"
        cells = (
            str(member.entry_index),
            _escape_cell(member.instrument),
            str(member.seed),
            member.dependency.stratum,
            member.signature.native_form.value,
            _escape_cell(member.native_score_kind),
            _fmt(member.native_score),
            _fmt(member.native_nmse),
            verified_nmse,
            member.verification.status,
        )
        lines.append("| " + " | ".join(cells) + " |")
    if cls.flags:
        flag_text = ", ".join(
            f"{flag.entry_index}:{flag.flag}={_fmt(flag.value)}" for flag in cls.flags
        )
        lines.append(f"flags: {flag_text}")
    return lines


def _unsignable_counts(report: ConsensusReport) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in report.store_level.unsignable:
        counts[entry.dataset_ref] = counts.get(entry.dataset_ref, 0) + 1
    return counts


def _dataset_lines(dataset: DatasetConsensus, n_unsignable: int) -> list[str]:
    lines = [
        f"## dataset {_escape_cell(dataset.dataset_ref)} @ "
        f"{_short(dataset.dataset_fingerprint)} "
        f"(empirical: {dataset.empirical_state}, {n_unsignable} unsignable)"
    ]
    if not dataset.classes:
        lines.append("0 classes")
    for cls in dataset.classes:
        lines.extend(_class_lines(cls))
    if len(dataset.classes) >= 2:
        lines.append("### adjacency")
        lines.append("| class_a | class_b | relation | jaccard |")
        lines.append("| --- | --- | --- | --- |")
        for adjacency in dataset.adjacency:
            lines.append(
                f"| {_short(adjacency.structure_key_a)} | "
                f"{_short(adjacency.structure_key_b)} | {adjacency.relation} | "
                f"{_fmt(adjacency.jaccard)} |"
            )
    lines.append("### instrument stability")
    lines.append("| instrument | completed | signable | modal_class | modal_share |")
    lines.append("| --- | --- | --- | --- | --- |")
    for stability in dataset.instrument_stability:
        lines.append(
            f"| {_escape_cell(stability.instrument)} | {stability.n_completed} | "
            f"{stability.n_signable} | {_short(stability.modal_structure_key)} | "
            f"{_fmt(stability.modal_share)} |"
        )
    return lines


def render_consensus_markdown(report: ConsensusReport) -> str:
    provenance = report.provenance
    kd_version = provenance.kd_version or "unknown"
    lines: list[str] = [
        f"# consensus {_escape_cell(provenance.plan_name)} "
        f"({provenance.plan_hash[:10]}, kd {_escape_cell(kd_version)})",
        f"schema {CONSENSUS_ARTIFACT_TAG} | "
        f"stratum rules {provenance.stratum_rules_version} | "
        f"eligibility rules {provenance.eligibility_rules_version}",
        _policy_line(report),
        _verifier_line(report),
        "",
    ]
    unsignable_by_ref = _unsignable_counts(report)
    for dataset in report.datasets:
        lines.extend(
            _dataset_lines(dataset, unsignable_by_ref.get(dataset.dataset_ref, 0))
        )
        lines.append("")

    store_level = report.store_level
    lines.append("## store")
    lines.append(
        f"{store_level.n_attempts} attempts / {store_level.n_completed} completed / "
        f"{store_level.n_raised} raised / {store_level.n_no_record} no_record"
    )
    if store_level.unsignable:
        lines.append("### unsignable")
        lines.append("| entry | instrument | dataset_ref | seed | reason | detail |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for entry in store_level.unsignable:
            if entry.invalid_reason is not None:
                detail = entry.invalid_reason
            elif entry.error_type is not None:
                detail = f"{entry.error_type}: {entry.error_message}"
            else:
                detail = ""
            cells = (
                str(entry.entry_index),
                _escape_cell(entry.instrument),
                _escape_cell(entry.dataset_ref),
                str(entry.seed),
                entry.reason,
                _escape_cell(detail),
            )
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines).rstrip() + "\n"
