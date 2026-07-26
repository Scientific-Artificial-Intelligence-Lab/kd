
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from kd.harness.consensus import CONSENSUS_ARTIFACT_TAG, build_consensus
from kd.harness.consensus_report import (
    _V1_ARTIFACT_KEYS,
    _V1_AXES_KEYS,
    _V1_CLASS_KEYS,
    _V1_COEFFICIENT_KEYS,
    _V1_DATASET_KEYS,
    _V1_DEPENDENCY_KEYS,
    _V1_ELIGIBILITY_KEYS,
    _V1_EMPIRICAL_KEYS,
    _V1_LHS_KEYS,
    _V1_MEASUREMENTS_KEYS,
    _V1_MEMBER_KEYS,
    _V1_MEMBER_NATIVE_KEYS,
    _V1_PROVENANCE_KEYS,
    _V1_REPORT_KEYS,
    _V1_STORE_LEVEL_KEYS,
    _V1_STRATIFICATION_KEYS,
    _V1_STRUCTURE_KEYS,
    _V1_SUPPORT_KEYS,
    _V1_SUPPORT_VARIANT_KEYS,
    _V1_UNSIGNABLE_KEYS,
    _V1_VERIFICATION_KEYS,
    _V1_VERIFIER_KEYS,
    _V1_VERIFY_POLICY_KEYS,
    ConsensusArtifactError,
    consensus_to_dict,
    read_consensus_artifact,
    render_consensus_markdown,
    write_consensus_artifact,
)

from kd.core.equation import (
    Form,
    LhsSpec,
    Scalar,
    from_dict,
    make_evolution,
    to_dict,
)
from kd.core.equation.signature import law_signature
from kd.core.verify import VerificationReport, VerifyPolicy
from kd.data.schema import PDEDataset, compute_dataset_fingerprint
from kd.harness.plan import PlanEntry
from kd.search.records import RunRecord

from ._helpers import build_sealed_store, make_entry, make_record

_UT = LhsSpec(field="u", axis="t", order=1)


def _evo(terms: list[tuple[str, float]]) -> dict[str, Any]:
    return to_dict(make_evolution(_UT, [(ir, Scalar(v)) for ir, v in terms]))


CF_HEAT = _evo([("u_xx", 3.0)])
CF_HEAT_B = _evo([("u_xx", 3.02)])
CF_ADVEC = _evo([("u_x", -1.0)])

_SENTINEL = object()


def _tiny_dataset(name: str = "ds_evo") -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 8)
    u = torch.sin(x).reshape(-1, 1) * torch.cos(t).reshape(1, -1)
    return PDEDataset.from_arrays(
        name=name, coords={"x": x, "t": t}, fields={"u": u}, lhs="u_t"
    )


def _fake_report(
    nmse: float = 0.01, normalizer_term: str = "u_t"
) -> VerificationReport:
    sig = law_signature(from_dict(CF_HEAT))
    return VerificationReport(
        signature=sig,
        form=Form.EVOLUTION,
        dataset_name="ds_evo",
        dataset_fingerprint="fp",
        mse=0.001,
        nmse=nmse,
        r2=1.0 - nmse,
        residual_mean=0.0,
        residual_std=0.1,
        residual_max_abs=0.2,
        n_samples=42,
        inactive_coefficient_mass=0.0,
        normalizer_term=normalizer_term,
        normalizer_variance=1.0,
        policy=VerifyPolicy(),
        passed=None,
    )


def _rich_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(
        "kd.harness.consensus_verify.verify_equation",
        lambda eq, **kwargs: _fake_report(),
    )
    from kd.harness.consensus_verify import VerifyExecution

    dataset = _tiny_dataset()
    fp = compute_dataset_fingerprint(dataset)
    ref = dataset.name

    entries = [
        PlanEntry(instrument="sga", dataset_ref=ref, seed=0, model_kwargs={}),
        PlanEntry(instrument="discover", dataset_ref=ref, seed=1, model_kwargs={}),
        PlanEntry(instrument="pysindy", dataset_ref=ref, seed=2, model_kwargs={}),
        PlanEntry(instrument="eqgpt", dataset_ref=ref, seed=3, model_kwargs={}),
    ]

    def rec(instr: str, seed: int, **kw: Any) -> RunRecord:
        return make_record(
            instr, seed=seed, dataset_name=ref, dataset_cache_fingerprint=fp, **kw
        )

    records = {

        0: rec("sga", 0, catalog_fit=CF_HEAT, nmse=0.9),
        1: rec("discover", 1, catalog_fit=CF_HEAT_B, nmse=0.01),
        2: rec("pysindy", 2, catalog_fit=CF_ADVEC),
        3: rec("eqgpt", 3, catalog_fit=None),
    }
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)
    return build_consensus(
        store,
        datasets={ref: dataset},
        context_factory=lambda ds: VerifyExecution(
            executor=_SENTINEL,
            context=_SENTINEL,
            provider_kind="finite_diff",
        ),
    )





def test_frozen_key_tables_are_the_literal_scheme() -> None:
    assert CONSENSUS_ARTIFACT_TAG == "kd-consensus-v1"
    assert frozenset({"provenance", "datasets", "store_level"}) == _V1_REPORT_KEYS
    assert frozenset({"field", "axis", "order"}) == _V1_LHS_KEYS
    assert (
        frozenset(
            {
                "mse",
                "nmse",
                "r2",
                "residual_mean",
                "residual_std",
                "residual_max_abs",
                "n_samples",
                "inactive_coefficient_mass",
                "normalizer_term",
                "normalizer_variance",
                "passed",
            }
        )
        == _V1_MEASUREMENTS_KEYS
    )
    assert (
        frozenset({"artifact", "plan_hash", "store_root", "report"})
        == _V1_ARTIFACT_KEYS
    )

    assert (
        frozenset({"nmse_max", "coeff_atol", "empirical_atol", "pivot_unity_rtol"})
        == _V1_VERIFY_POLICY_KEYS
    )


def test_serialized_tree_key_sets_match_frozen_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _rich_report(tmp_path, monkeypatch)
    payload = consensus_to_dict(report)

    assert set(payload) == set(_V1_REPORT_KEYS)
    assert set(payload["provenance"]) == set(_V1_PROVENANCE_KEYS)
    assert set(payload["provenance"]["verify_policy"]) == set(_V1_VERIFY_POLICY_KEYS)
    assert set(payload["provenance"]["verifier"]) == set(_V1_VERIFIER_KEYS)
    assert set(payload["store_level"]) == set(_V1_STORE_LEVEL_KEYS)
    for entry in payload["store_level"]["unsignable"]:
        assert set(entry) == set(_V1_UNSIGNABLE_KEYS)

    saw_lhs_object = False
    saw_measurements = False
    for dataset in payload["datasets"]:
        assert set(dataset) == set(_V1_DATASET_KEYS)
        for cls in dataset["classes"]:
            assert set(cls) == set(_V1_CLASS_KEYS)
            assert set(cls["axes"]) == set(_V1_AXES_KEYS)
            assert set(cls["axes"]["structure"]) == set(_V1_STRUCTURE_KEYS)
            assert set(cls["axes"]["support"]) == set(_V1_SUPPORT_KEYS)
            assert set(cls["axes"]["coefficient"]) == set(_V1_COEFFICIENT_KEYS)
            assert set(cls["axes"]["empirical"]) == set(_V1_EMPIRICAL_KEYS)
            assert set(cls["stratification"]) == set(_V1_STRATIFICATION_KEYS)
            assert set(cls["stratification"]["eligibility"]) == set(
                _V1_ELIGIBILITY_KEYS
            )
            for variant in cls["axes"]["support"]["variants"]:
                assert set(variant) == set(_V1_SUPPORT_VARIANT_KEYS)
                if variant["native_lhs"] is not None:
                    assert set(variant["native_lhs"]) == set(_V1_LHS_KEYS)
                    saw_lhs_object = True
            for member in cls["members"]:
                assert set(member) == set(_V1_MEMBER_KEYS)
                assert set(member["native"]) == set(_V1_MEMBER_NATIVE_KEYS)
                assert set(member["dependency"]) == set(_V1_DEPENDENCY_KEYS)
                assert set(member["verification"]) == set(_V1_VERIFICATION_KEYS)
                if member["native_lhs"] is not None:
                    assert set(member["native_lhs"]) == set(_V1_LHS_KEYS)
                    saw_lhs_object = True
                measurements = member["verification"]["measurements"]
                if measurements is not None:
                    assert set(measurements) == set(_V1_MEASUREMENTS_KEYS)
                    saw_measurements = True

    assert saw_lhs_object, "fixture must exercise a native_lhs object (_V1_LHS_KEYS)"
    assert saw_measurements, "fixture must exercise a verified measurements block"





def test_artifact_write_read_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _rich_report(tmp_path, monkeypatch)
    path = tmp_path / "consensus.json"
    written = write_consensus_artifact(report, path=path)
    assert Path(written) == path
    payload = read_consensus_artifact(path)
    assert payload["artifact"] == CONSENSUS_ARTIFACT_TAG
    assert payload["plan_hash"] == report.provenance.plan_hash
    assert payload["store_root"] == report.provenance.store_root
    assert payload["report"] == consensus_to_dict(report)


def test_artifact_no_nonfinite_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _rich_report(tmp_path, monkeypatch)
    path = tmp_path / "consensus.json"
    write_consensus_artifact(report, path=path)
    text = path.read_text(encoding="utf-8")


    assert "NaN" not in text
    assert "Infinity" not in text
    json.loads(text)


def test_write_rejects_empty_plan_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _rich_report(tmp_path, monkeypatch)
    import dataclasses

    broken = dataclasses.replace(
        report,
        provenance=dataclasses.replace(report.provenance, plan_hash=""),
    )
    with pytest.raises(ValueError, match="plan_hash"):
        write_consensus_artifact(broken, path=tmp_path / "x.json")


def test_read_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_consensus_artifact(tmp_path / "does_not_exist.json")


def test_read_rejects_non_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json at all {", encoding="utf-8")
    with pytest.raises(ConsensusArtifactError):
        read_consensus_artifact(path)


def test_read_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ConsensusArtifactError):
        read_consensus_artifact(path)


def _write_mutated(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_read_rejects_wrong_artifact_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _rich_report(tmp_path, monkeypatch)
    write_consensus_artifact(report, path=tmp_path / "good.json")
    payload = json.loads((tmp_path / "good.json").read_text(encoding="utf-8"))
    payload["artifact"] = "kd-consensus-v2"
    with pytest.raises(ConsensusArtifactError):
        read_consensus_artifact(_write_mutated(tmp_path, payload))


def test_read_rejects_unknown_envelope_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _rich_report(tmp_path, monkeypatch)
    write_consensus_artifact(report, path=tmp_path / "good.json")
    payload = json.loads((tmp_path / "good.json").read_text(encoding="utf-8"))
    payload["extra"] = 1
    with pytest.raises(ConsensusArtifactError):
        read_consensus_artifact(_write_mutated(tmp_path, payload))


def test_read_rejects_missing_provenance_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _rich_report(tmp_path, monkeypatch)
    write_consensus_artifact(report, path=tmp_path / "good.json")
    payload = json.loads((tmp_path / "good.json").read_text(encoding="utf-8"))
    del payload["report"]["provenance"]["schema"]
    with pytest.raises(ConsensusArtifactError):
        read_consensus_artifact(_write_mutated(tmp_path, payload))


def test_read_rejects_unknown_key_in_nested_lhs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _rich_report(tmp_path, monkeypatch)
    write_consensus_artifact(report, path=tmp_path / "good.json")
    payload = json.loads((tmp_path / "good.json").read_text(encoding="utf-8"))
    mutated = False
    for dataset in payload["report"]["datasets"]:
        for cls in dataset["classes"]:
            for member in cls["members"]:
                if member["native_lhs"] is not None:
                    member["native_lhs"]["unexpected"] = 1
                    mutated = True
    assert mutated, "fixture must contain a native_lhs object to mutate"
    with pytest.raises(ConsensusArtifactError):
        read_consensus_artifact(_write_mutated(tmp_path, payload))


def test_read_rejects_envelope_plan_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _rich_report(tmp_path, monkeypatch)
    write_consensus_artifact(report, path=tmp_path / "good.json")
    payload = json.loads((tmp_path / "good.json").read_text(encoding="utf-8"))
    payload["plan_hash"] = "tampered-plan-hash"
    with pytest.raises(ConsensusArtifactError, match="plan_hash"):
        read_consensus_artifact(_write_mutated(tmp_path, payload))


def test_read_rejects_envelope_store_root_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _rich_report(tmp_path, monkeypatch)
    write_consensus_artifact(report, path=tmp_path / "good.json")
    payload = json.loads((tmp_path / "good.json").read_text(encoding="utf-8"))
    payload["store_root"] = "/tampered/store/root"
    with pytest.raises(ConsensusArtifactError, match="store_root"):
        read_consensus_artifact(_write_mutated(tmp_path, payload))


def test_read_rejects_nested_provenance_schema_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _rich_report(tmp_path, monkeypatch)
    write_consensus_artifact(report, path=tmp_path / "good.json")
    payload = json.loads((tmp_path / "good.json").read_text(encoding="utf-8"))
    payload["report"]["provenance"]["schema"] = "kd-consensus-v2"
    with pytest.raises(ConsensusArtifactError, match="schema"):
        read_consensus_artifact(_write_mutated(tmp_path, payload))


def test_read_rejects_nan_numeric_token(tmp_path: Path) -> None:
    path = tmp_path / "nan.json"
    path.write_text(
        '{"artifact": "kd-consensus-v1", "plan_hash": "h", '
        '"store_root": "/r", "report": NaN}',
        encoding="utf-8",
    )
    with pytest.raises(ConsensusArtifactError, match="NaN"):
        read_consensus_artifact(path)





def test_markdown_escapes_pipes_and_is_deterministic(tmp_path: Path) -> None:
    entries = [
        PlanEntry(
            instrument="pipe|instr", dataset_ref="ds|ref", seed=0, model_kwargs={}
        )
    ]
    records = {
        0: make_record(
            "pipe|instr",
            seed=0,
            dataset_name="ds|ref",
            catalog_fit=CF_HEAT,
        )
    }
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)
    report = build_consensus(store)
    rendered = render_consensus_markdown(report)

    assert r"\|" in rendered
    assert "pipe|instr" not in rendered
    assert rendered.endswith("\n")
    assert render_consensus_markdown(report) == rendered


def test_markdown_dataset_section_shows_unsignable_count(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0), make_entry("sga", seed=1)]
    records = {
        0: make_record("sga", seed=0, catalog_fit=None),
        1: make_record("sga", seed=1, catalog_fit=None),
    }
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)
    rendered = render_consensus_markdown(build_consensus(store))
    dataset_section = rendered.split("## store")[0]
    assert "0 classes" in dataset_section
    assert "2 unsignable" in dataset_section


def test_markdown_has_title_and_store_sections(tmp_path: Path) -> None:
    entries = [make_entry("sga", seed=0)]
    records = {0: make_record("sga", seed=0, catalog_fit=CF_HEAT)}
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)
    rendered = render_consensus_markdown(build_consensus(store))
    assert rendered.startswith("# consensus ")
    assert "kd-consensus-v1" in rendered
    assert "## store" in rendered


def test_report_byte_identical_two_runs_with_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "kd.harness.consensus_verify.verify_equation",
        lambda eq, **kwargs: _fake_report(),
    )
    from kd.harness.consensus_verify import VerifyExecution

    dataset = _tiny_dataset()
    fp = compute_dataset_fingerprint(dataset)
    ref = dataset.name
    entries = [
        PlanEntry(instrument="sga", dataset_ref=ref, seed=0, model_kwargs={}),
        PlanEntry(instrument="discover", dataset_ref=ref, seed=1, model_kwargs={}),
    ]
    records = {
        0: make_record(
            "sga",
            seed=0,
            dataset_name=ref,
            dataset_cache_fingerprint=fp,
            catalog_fit=CF_HEAT,
        ),
        1: make_record(
            "discover",
            seed=1,
            dataset_name=ref,
            dataset_cache_fingerprint=fp,
            catalog_fit=CF_HEAT,
        ),
    }
    store = build_sealed_store(tmp_path / "store", entries=entries, records=records)

    def factory(ds: PDEDataset) -> Any:
        return VerifyExecution(
            executor=_SENTINEL,
            context=_SENTINEL,
            provider_kind="finite_diff",
        )

    first = build_consensus(store, datasets={ref: dataset}, context_factory=factory)
    second = build_consensus(store, datasets={ref: dataset}, context_factory=factory)
    assert json.dumps(
        consensus_to_dict(first), indent=2, allow_nan=False
    ) == json.dumps(consensus_to_dict(second), indent=2, allow_nan=False)
    assert render_consensus_markdown(first) == render_consensus_markdown(second)
