
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import torch

import kd.search.records as records
import kd.search.result as result_module
from kd.core.equation import LhsSpec, build_equation
from kd.core.equation import to_dict as equation_to_dict
from kd.core.equation.library import TermLibrarySpec
from kd.core.evaluator import EvaluationResult
from kd.search.record_assembly import assemble_run_record
from kd.search.recorder import VizRecorder
from kd.search.records import (
    EvidenceRecord,
    ResidualStats,
    RunCost,
    RunRecord,
)
from kd.search.result import ExperimentResult, invalid_evaluation_result
from kd.search.run_spec import RUN_SPEC_HASH_SCHEME, RunSpec
from tests.unit.search.test_run_spec import _run_spec



_EXPECTED_INVALID_REASON_VOCAB = frozenset(
    {
        "no_candidate",
        "evaluation_error",
        "non_finite",
        "structural_reject",
        "unclassified",
    }
)







def _minimal_cost() -> records.RunCost:
    return records.RunCost(
        wallclock_seconds=1.25,
        search_seconds=1.25,
        boundary_results=0,
        boundary_invalid_results=0,
    )


def _full_cost() -> records.RunCost:
    return records.RunCost(
        wallclock_seconds=3.5,
        search_seconds=2.0,
        boundary_results=12,
        boundary_invalid_results=2,
        preprocessing_seconds=1.5,
        surrogate_train_seconds=1.25,
        cpu_seconds=2.75,
        tokens_in=100,
        tokens_out=25,
        tokens_cached=40,
        api_cost_usd=0.0125,
    )


def _equation_payload() -> dict[str, Any]:
    equation = build_equation(
        ["mul(u, u_x)", "u_xx"],
        [-1.0, 0.1],
        LhsSpec(field="u", axis="t", order=1),
        active_indices=[0, 1],
    )
    assert equation is not None
    return equation_to_dict(equation)


def _minimal_evidence(**overrides: Any) -> records.EvidenceRecord:
    values: dict[str, Any] = {
        "instrument": "sga",
        "dataset_name": "burgers",
        "dataset_cache_fingerprint": "sha256:dataset",
        "seed": 0,
        "is_valid": True,
        "expression": "mul(u, u_x)",
        "score_kind": "AIC",
        "score_direction": "min",
        "headline_coefficient_source": "native",
    }
    values.update(overrides)
    return records.EvidenceRecord(**values)


def _full_evidence() -> records.EvidenceRecord:
    return records.EvidenceRecord(
        instrument="sga",
        dataset_name="burgers",
        dataset_cache_fingerprint="sha256:dataset",
        seed=7,
        is_valid=True,
        expression="add(mul(u, u_x), u_xx)",
        score_kind="AIC",
        score_direction="min",
        headline_coefficient_source="native",
        catalog_fit=_equation_payload(),
        support=["mul(u, u_x)", "u_xx", "u_x"],
        coefficients=[1.0, None, -2.5],
        complexity=2,
        mse=0.001,
        nmse=0.002,
        r2=0.998,
        score=-42.0,
        residual_stats=records.ResidualStats(mean=0.0, std=0.01, max_abs=0.04, n=32),
    )


def _run_record(
    *,
    evidence: records.EvidenceRecord | None = None,
    cost: records.RunCost | None = None,
    schema_version: object | None = None,
    evidence_hash_scheme: object | None = None,
    evidence_hash: str | None = None,
    run_spec: RunSpec | None = None,
    run_spec_hash: str | None = None,
    created_at: str = "2026-07-17T00:00:00+00:00",
) -> records.RunRecord:
    selected = evidence if evidence is not None else _minimal_evidence()
    selected_run_spec = run_spec or RunSpec(
        kd_version="0.1.0",
        config={"algorithm": "sga"},
        dataset_cache_fingerprint="sha256:dataset",
    )




    record = records.RunRecord(
        schema_version=(
            records.RUN_RECORD_SCHEMA_VERSION
            if schema_version is None
            else schema_version
        ),
        evidence_hash_scheme=(
            records.EVIDENCE_HASH_SCHEME
            if evidence_hash_scheme is None
            else evidence_hash_scheme
        ),
        created_at=created_at,
        cost=cost if cost is not None else _minimal_cost(),
        evidence=selected,
        evidence_hash=(
            selected.content_hash() if evidence_hash is None else evidence_hash
        ),
        run_spec=selected_run_spec,
        run_spec_hash=(
            selected_run_spec.run_spec_hash
            if run_spec_hash is None
            else run_spec_hash
        ),
        run_spec_hash_scheme=RUN_SPEC_HASH_SCHEME,
        record_hash="",
        record_hash_scheme=records.RECORD_HASH_SCHEME,
    )
    return records.seal_record_hash(record)





_record = _run_record


def _valid_record_dict() -> dict[str, Any]:
    return _run_record(evidence=_full_evidence()).to_dict()


def _experiment_result(**overrides: Any) -> ExperimentResult:
    final_eval = EvaluationResult(
        mse=0.01,
        nmse=0.02,
        r2=0.98,
        score=-50.0,
        complexity=1,
        is_valid=True,
        terms=["u"],
        expression="u",
    )
    kwargs: dict[str, Any] = {
        "best_expression": "u",
        "best_score": 0.02,
        "iterations": 1,
        "early_stopped": False,
        "final_eval": final_eval,
        "actual": torch.zeros(4),
        "predicted": torch.zeros(4),
        "dataset_name": "test",
        "algorithm_name": "sga",
        "config": {},
        "recorder": VizRecorder(),
    }
    kwargs.update(overrides)
    return ExperimentResult(**kwargs)


def _cost_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "wallclock_seconds": 1.0,
        "search_seconds": 1.0,
        "boundary_results": 0,
        "boundary_invalid_results": 0,
    }
    base.update(overrides)
    return base





_evidence_d = _minimal_evidence


def _invalid_evidence_d(**overrides: Any) -> records.EvidenceRecord:
    values: dict[str, Any] = {
        "is_valid": False,
        "invalid_reason": "evaluation_error",
        "error_detail": "Solver error: singular matrix at /x/y.py:42",
    }
    values.update(overrides)
    return _evidence_d(**values)


def _run_record_d(
    *,
    evidence: records.EvidenceRecord | None = None,
    run_spec: Any | None = None,
    run_spec_hash: str | None = None,
) -> records.RunRecord:
    selected = evidence if evidence is not None else _evidence_d()
    spec = run_spec if run_spec is not None else _run_spec()
    record = records.RunRecord(
        schema_version=records.RUN_RECORD_SCHEMA_VERSION,
        evidence_hash_scheme=records.EVIDENCE_HASH_SCHEME,
        created_at="2026-07-20T00:00:00+00:00",
        cost=_minimal_cost(),
        evidence=selected,
        evidence_hash=selected.content_hash(),
        run_spec=spec,
        run_spec_hash=(spec.run_spec_hash if run_spec_hash is None else run_spec_hash),
        run_spec_hash_scheme=RUN_SPEC_HASH_SCHEME,
        record_hash="",
        record_hash_scheme=records.RECORD_HASH_SCHEME,
    )
    return records.seal_record_hash(record)


def _manifest(**overrides: Any) -> result_module.RunManifest:
    values: dict[str, Any] = {
        "dataset_cache_fingerprint": "sha256:dataset",
        "kd_version": "0.4.0",
        "seed": 0,
    }
    values.update(overrides)
    return result_module.RunManifest(**values)


def _valid_final_eval() -> EvaluationResult:
    return EvaluationResult(
        mse=0.1,
        nmse=0.2,
        r2=0.9,
        score=-1.0,
        complexity=2,
        is_valid=True,
        coefficients=torch.tensor([1.0, -2.0]),
        selected_indices=None,
        terms=["u_x", "u_xx"],
        expression="add(u_x, u_xx)",
    )


def _assemble_d(
    *,
    run_spec: Any,
    manifest_terms: list[str] | None,
    final_eval: EvaluationResult | None = None,
) -> records.RunRecord:
    return assemble_run_record(
        instrument="sga",
        dataset_name="burgers_1d",
        dataset_cache_fingerprint="abc123",
        seed=0,
        final_eval=final_eval if final_eval is not None else _valid_final_eval(),
        equation=None,
        best_expression="add(u_x, u_xx)",
        best_score=0.5,
        score_kind="Score",
        score_direction="min",
        headline_coefficient_source="native",
        cost=_minimal_cost(),
        run_spec=run_spec,
        manifest_terms=manifest_terms,
    )







@pytest.mark.parametrize("cost", [_minimal_cost(), _full_cost()])
def test_run_cost_round_trip(cost: RunCost) -> None:
    assert RunCost.from_dict(cost.to_dict()) == cost


@pytest.mark.parametrize("evidence", [_minimal_evidence(), _full_evidence()])
def test_evidence_record_round_trip(evidence: EvidenceRecord) -> None:
    assert EvidenceRecord.from_dict(evidence.to_dict()) == evidence


def test_residual_stats_round_trip() -> None:
    stats = ResidualStats(mean=None, std=0.25, max_abs=1.5, n=8)

    assert ResidualStats.from_dict(stats.to_dict()) == stats


def test_run_record_save_load_and_plain_json(tmp_path: Path) -> None:
    record = _record(cost=_full_cost(), evidence=_full_evidence())
    path = tmp_path / "nested" / "record.json"

    record.save(path)

    assert RunRecord.load(path) == record
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload == record.to_dict()


def test_frozen_schema_key_sets() -> None:


    assert set(_minimal_cost().to_dict()) == {
        "wallclock_seconds",
        "search_seconds",
        "preprocessing_seconds",
        "surrogate_train_seconds",
        "cpu_seconds",
        "tokens_in",
        "tokens_out",
        "tokens_cached",
        "api_cost_usd",
        "boundary_results",
        "boundary_invalid_results",
    }
    assert set(_minimal_evidence().to_dict()) == {
        "instrument",
        "dataset_name",
        "dataset_cache_fingerprint",
        "seed",
        "is_valid",
        "expression",
        "score_kind",
        "score_direction",
        "headline_coefficient_source",
        "catalog_fit",
        "support",
        "coefficients",
        "complexity",
        "mse",
        "nmse",
        "r2",
        "score",
        "residual_stats",
        "invalid_reason",
        "error_detail",
    }
    assert set(_record().to_dict()) == {
        "schema_version",
        "evidence_hash_scheme",
        "created_at",
        "cost",
        "evidence",
        "evidence_hash",
        "run_spec",
        "run_spec_hash",
        "run_spec_hash_scheme",
        "record_hash",
        "record_hash_scheme",
    }


def test_cost_and_created_at_are_structurally_outside_evidence_hash() -> None:
    evidence_a = _full_evidence()
    evidence_b = EvidenceRecord.from_dict(evidence_a.to_dict())
    first = _record(
        cost=_minimal_cost(),
        evidence=evidence_a,
        created_at="2026-07-17T00:00:00+00:00",
    )
    second = _record(
        cost=_full_cost(),
        evidence=evidence_b,
        created_at="2027-01-01T12:34:56+00:00",
    )

    assert first.evidence_hash == second.evidence_hash


def test_none_is_serialized_as_json_null_not_zero() -> None:
    payload = json.loads(json.dumps(_minimal_cost().to_dict(), allow_nan=False))

    assert "tokens_in" in payload
    assert payload["tokens_in"] is None







def test_a1_run_record_schema_version_is_two() -> None:

    assert records.RUN_RECORD_SCHEMA_VERSION == 2


def test_a1_evidence_hash_scheme_constant() -> None:

    assert records.EVIDENCE_HASH_SCHEME == "kd-evidence-v1"


def test_a1_run_record_persists_scheme_field() -> None:

    record = _run_record()
    assert record.evidence_hash_scheme == records.EVIDENCE_HASH_SCHEME


def test_a1_schema_version_and_scheme_round_trip() -> None:

    record = _run_record(evidence=_full_evidence())
    payload = record.to_dict()

    assert payload["schema_version"] == 2
    assert payload["evidence_hash_scheme"] == "kd-evidence-v1"
    assert records.RunRecord.from_dict(payload) == record







def test_a4_explicit_none_matches_default_construction() -> None:

    default = _minimal_evidence()
    explicit = _minimal_evidence(
        catalog_fit=None,
        support=None,
        coefficients=None,
        complexity=None,
        mse=None,
        nmse=None,
        r2=None,
        score=None,
        residual_stats=None,
    )
    assert default.content_hash() == explicit.content_hash()


def test_a4_seed_none_differs_from_seed_zero() -> None:

    assert (
        _minimal_evidence(seed=None).content_hash()
        != _minimal_evidence(seed=0).content_hash()
    )


def test_a4_residual_stats_present_none_moments_differs_from_absent() -> None:

    present = _minimal_evidence(
        residual_stats=records.ResidualStats(mean=None, std=None, max_abs=None, n=0)
    )
    absent = _minimal_evidence(residual_stats=None)
    assert present.content_hash() != absent.content_hash()


def test_a4_interior_none_coefficient_is_position_preserving() -> None:

    assert (
        _minimal_evidence(coefficients=[1.0, None]).content_hash()
        != _minimal_evidence(coefficients=[1.0]).content_hash()
    )


def test_a4_negative_zero_coefficient_differs_from_positive_zero() -> None:

    assert (
        _minimal_evidence(coefficients=[-0.0]).content_hash()
        != _minimal_evidence(coefficients=[0.0]).content_hash()
    )


def test_a4_hash_is_key_order_independent() -> None:

    original = _full_evidence()
    rebuilt = records.EvidenceRecord.from_dict(original.to_dict())
    assert original.content_hash() == rebuilt.content_hash()


def test_a4_mutating_hash_bound_field_changes_hash() -> None:

    assert (
        _minimal_evidence(expression="a").content_hash()
        != _minimal_evidence(expression="b").content_hash()
    )







def test_a6_result_schema_version_constant_is_one() -> None:

    assert result_module.RESULT_SCHEMA_VERSION == 1


def test_a6_to_dict_emits_result_schema_version() -> None:
    payload = _experiment_result().to_dict()
    assert payload["result_schema_version"] == result_module.RESULT_SCHEMA_VERSION


def test_a6_load_rejects_future_result_schema_version(tmp_path: Path) -> None:

    payload = _experiment_result().to_dict()
    payload["result_schema_version"] = result_module.RESULT_SCHEMA_VERSION + 1
    path = tmp_path / "future.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="result_schema_version"):
        ExperimentResult.load(path)


def test_a6_load_without_result_schema_version_succeeds_legacy(
    tmp_path: Path,
) -> None:

    payload = _experiment_result().to_dict()
    payload.pop("result_schema_version", None)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = ExperimentResult.load(path)
    assert isinstance(loaded, ExperimentResult)


@pytest.mark.parametrize("bad_version", [0, -1])
def test_item9_load_rejects_below_minimum_result_schema_version(
    tmp_path: Path, bad_version: int
) -> None:




    payload = _experiment_result().to_dict()
    payload["result_schema_version"] = bad_version
    path = tmp_path / "below.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="result_schema_version"):
        ExperimentResult.load(path)







def test_a7_run_cost_is_frozen() -> None:
    cost = _minimal_cost()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cost.wallclock_seconds = 2.0


def test_a7_residual_stats_is_frozen() -> None:
    stats = records.ResidualStats(mean=0.0, std=0.1, max_abs=0.2, n=4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.n = 8


def test_a7_evidence_record_is_frozen() -> None:
    evidence = _minimal_evidence()
    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.expression = "u_xx"


def test_a7_run_record_is_frozen() -> None:
    record = _run_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.evidence_hash = "0" * 64


def test_a7_to_dict_returns_defensive_copy() -> None:

    evidence = _minimal_evidence(coefficients=[1.0, 2.0])
    before = evidence.content_hash()

    payload = evidence.to_dict()
    payload["coefficients"].append(99.0)

    assert evidence.content_hash() == before


def test_a7_constructor_copies_coefficients_list() -> None:

    coefficients = [1.0, 2.0]
    evidence = _minimal_evidence(coefficients=coefficients)
    before = evidence.content_hash()

    coefficients.append(99.0)

    assert evidence.content_hash() == before


def test_a7_constructor_deepcopies_equation_payload() -> None:

    payload = _equation_payload()
    evidence = _minimal_evidence(catalog_fit=payload)
    before = evidence.content_hash()

    payload["terms"].append(["u", {"kind": "Scalar", "value": 5.0}])

    assert evidence.content_hash() == before







def test_b_negative_boundary_results_rejected_at_construction() -> None:

    with pytest.raises(ValueError, match="boundary_results"):
        records.RunCost(
            wallclock_seconds=1.0,
            search_seconds=1.0,
            boundary_results=-1,
            boundary_invalid_results=0,
        )


def test_b_negative_boundary_invalid_results_rejected_at_construction() -> None:

    with pytest.raises(ValueError, match="boundary_invalid_results"):
        records.RunCost(
            wallclock_seconds=1.0,
            search_seconds=1.0,
            boundary_results=0,
            boundary_invalid_results=-1,
        )







@pytest.mark.parametrize("field", ["wallclock_seconds", "search_seconds"])
def test_e_required_duration_rejects_non_finite(field: str) -> None:

    with pytest.raises(ValueError, match=field):
        records.RunCost(**_cost_kwargs(**{field: float("inf")}))


@pytest.mark.parametrize("field", ["wallclock_seconds", "search_seconds"])
def test_e_required_duration_rejects_negative(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        records.RunCost(**_cost_kwargs(**{field: -1.0}))


@pytest.mark.parametrize(
    "field",
    ["preprocessing_seconds", "surrogate_train_seconds", "cpu_seconds", "api_cost_usd"],
)
def test_e_optional_float_rejects_negative(field: str) -> None:

    with pytest.raises(ValueError, match=field):
        records.RunCost(**_cost_kwargs(**{field: -0.5}))


@pytest.mark.parametrize(
    "field",
    ["preprocessing_seconds", "surrogate_train_seconds", "cpu_seconds", "api_cost_usd"],
)
def test_e_optional_float_rejects_non_finite(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        records.RunCost(**_cost_kwargs(**{field: float("nan")}))


@pytest.mark.parametrize("field", ["tokens_in", "tokens_out", "tokens_cached"])
def test_e_token_count_rejects_negative(field: str) -> None:

    with pytest.raises(ValueError, match=field):
        records.RunCost(**_cost_kwargs(**{field: -1}))


def test_e_optional_none_fields_stay_legal() -> None:

    cost = records.RunCost(**_cost_kwargs())
    assert cost.preprocessing_seconds is None
    assert cost.tokens_in is None







def test_d1_invalid_reason_vocab_is_the_five_documented_values() -> None:

    assert records.INVALID_REASON_VOCAB == _EXPECTED_INVALID_REASON_VOCAB
    assert isinstance(records.INVALID_REASON_VOCAB, frozenset)


def test_d1_invalid_evaluation_result_rejects_non_vocab_reason() -> None:

    with pytest.raises(ValueError, match="reason"):
        invalid_evaluation_result("boom", score=None, reason="not_a_reason")


def test_d1_invalid_evaluation_result_accepts_vocab_reason() -> None:

    result = invalid_evaluation_result("boom", score=None, reason="evaluation_error")
    assert result.invalid_reason == "evaluation_error"
    assert result.is_valid is False


def test_d1_invalid_evaluation_result_defaults_unclassified() -> None:

    result = invalid_evaluation_result("boom", score=None)
    assert result.invalid_reason == "unclassified"


def test_d1_evaluation_result_invalid_reason_defaults_none() -> None:


    result = EvaluationResult(mse=0.1, nmse=0.1, r2=0.9)
    assert result.invalid_reason is None


def test_d1_evidence_invalid_reason_is_hash_bound() -> None:


    reason_x = _invalid_evidence_d(invalid_reason="evaluation_error")
    reason_y = _invalid_evidence_d(invalid_reason="non_finite")
    assert reason_x.content_hash() != reason_y.content_hash()


def test_d1_valid_record_invalid_reason_none_is_omitted_from_hash() -> None:


    default_omitted = _evidence_d()
    explicit_none = _evidence_d(invalid_reason=None)
    assert default_omitted.content_hash() == explicit_none.content_hash()







def test_d2_error_detail_is_excluded_from_hash() -> None:


    detail_a = _invalid_evidence_d(error_detail="Solver error at /home/a/x.py:5")
    detail_b = _invalid_evidence_d(error_detail="Solver error at /home/b/y.py:9")
    assert detail_a.content_hash() == detail_b.content_hash()
    assert detail_a.is_valid is False and detail_b.is_valid is False


def test_d2_hash_excluded_fields_is_exactly_error_detail() -> None:





    assert frozenset({"error_detail"}) == records._HASH_EXCLUDED_FIELDS


def test_d2_error_detail_round_trips_through_codec() -> None:


    record = _invalid_evidence_d(error_detail="boom: singular theta")
    restored = records.EvidenceRecord.from_dict(record.to_dict())
    assert restored.error_detail == "boom: singular theta"







def test_d3_headline_source_bad_value_rejected_at_construction() -> None:

    with pytest.raises(ValueError, match="headline_coefficient_source"):
        _evidence_d(headline_coefficient_source="bogus_source")


def test_d3_headline_source_undeclared_accepted_and_round_trips() -> None:



    evidence = _evidence_d(headline_coefficient_source="undeclared")
    decoded = records.EvidenceRecord.from_dict(evidence.to_dict())
    assert decoded.headline_coefficient_source == "undeclared"


def test_d3_headline_source_is_hash_bound() -> None:


    native = _evidence_d(headline_coefficient_source="native")
    refit = _evidence_d(headline_coefficient_source="platform_refit")
    assert native.content_hash() != refit.content_hash()


@pytest.mark.parametrize("source", ["native", "platform_refit"])
def test_d3_headline_source_valid_values_round_trip(source: str) -> None:

    evidence = _evidence_d(headline_coefficient_source=source)
    restored = records.EvidenceRecord.from_dict(evidence.to_dict())
    assert restored.headline_coefficient_source == source







def test_d4_run_manifest_resumed_defaults_false() -> None:

    assert _manifest().resumed is False


def test_d4_to_dict_emits_resumed() -> None:

    assert _manifest().to_dict()["resumed"] is False


def test_d4_from_dict_legacy_missing_resumed_defaults_false() -> None:


    data = _manifest().to_dict()
    del data["resumed"]
    assert result_module.RunManifest.from_dict(data).resumed is False


def test_d4_resumed_round_trips_true() -> None:

    manifest = _manifest(resumed=True)
    restored = result_module.RunManifest.from_dict(manifest.to_dict())
    assert restored.resumed is True
    assert restored == manifest







def test_d5_evidence_hash_unchanged_when_run_spec_changes() -> None:


    shared_evidence = _evidence_d()
    record_a = _run_record_d(
        evidence=shared_evidence, run_spec=_run_spec(kd_version="1.0.0")
    )
    record_b = _run_record_d(
        evidence=shared_evidence, run_spec=_run_spec(kd_version="2.0.0")
    )
    assert record_a.evidence_hash == record_b.evidence_hash
    assert record_a.run_spec_hash != record_b.run_spec_hash


def test_d5_catalog_assertion_mismatch_raises() -> None:


    terms = ["u_x", "u_xx"]
    spec = _run_spec(library_fingerprint="0" * 16)
    with pytest.raises(ValueError, match="fingerprint"):
        _assemble_d(run_spec=spec, manifest_terms=terms)


def test_d5_catalog_assertion_match_succeeds() -> None:

    terms = ["u_x", "u_xx"]
    correct = TermLibrarySpec.from_terms(terms).fingerprint
    spec = _run_spec(library_fingerprint=correct)
    record = _assemble_d(run_spec=spec, manifest_terms=terms)
    assert record.run_spec.library_fingerprint == correct


def test_d5_catalog_assertion_skipped_without_declared_fingerprint() -> None:


    spec = _run_spec(library_fingerprint=None)
    record = _assemble_d(run_spec=spec, manifest_terms=["u_x", "u_xx"])
    assert record.run_spec.library_fingerprint is None







def test_item1_record_hash_scheme_constant() -> None:

    assert records.RECORD_HASH_SCHEME == "kd-record-v1"


def test_item1_record_hash_and_scheme_round_trip() -> None:

    record = _run_record_d()
    assert record.record_hash_scheme == "kd-record-v1"
    payload = record.to_dict()
    assert payload["record_hash_scheme"] == "kd-record-v1"
    assert len(payload["record_hash"]) == 64
    assert records.RunRecord.from_dict(payload) == record


def test_item1_verify_record_hash_true_for_sealed_record() -> None:

    assert _run_record_d().verify_record_hash() is True


def test_item1_record_hash_covers_cost_and_created_at() -> None:




    shared_evidence = _evidence_d()
    shared_spec = _run_spec()
    base = _run_record_d(evidence=shared_evidence, run_spec=shared_spec)
    other_cost = dataclasses.replace(base, cost=_full_cost())
    other_cost = records.seal_record_hash(
        dataclasses.replace(other_cost, record_hash="")
    )
    assert base.evidence_hash == other_cost.evidence_hash
    assert base.run_spec_hash == other_cost.run_spec_hash
    assert base.record_hash != other_cost.record_hash







def test_item3_valid_record_rejects_invalid_reason() -> None:

    with pytest.raises(records.StrictDecodeError, match="invalid_reason"):
        _minimal_evidence(is_valid=True, invalid_reason="evaluation_error")


def test_item3_valid_record_rejects_error_detail() -> None:

    with pytest.raises(records.StrictDecodeError, match="error_detail"):
        _minimal_evidence(is_valid=True, error_detail="boom")


def test_item3_invalid_record_with_both_none_stays_legal() -> None:

    evidence = _minimal_evidence(
        is_valid=False, invalid_reason=None, error_detail=None
    )
    assert evidence.is_valid is False
    assert evidence.invalid_reason is None
    assert evidence.error_detail is None







def test_item4_hasher_tables_partition_dataclass_fields_today() -> None:



    assert records._EVIDENCE_V1_SKELETON == records._EVIDENCE_REQUIRED_FIELDS
    assert (
        records._EVIDENCE_V1_OPTIONAL | records._EVIDENCE_V1_EXCLUDED
        == records._EVIDENCE_OPTIONAL_FIELDS
    )
    assert records._EVIDENCE_V1_EXCLUDED == records._HASH_EXCLUDED_FIELDS

    all_fields = {field.name for field in dataclasses.fields(records.EvidenceRecord)}
    assert all_fields == (
        records._EVIDENCE_V1_SKELETON
        | records._EVIDENCE_V1_OPTIONAL
        | records._EVIDENCE_V1_EXCLUDED
    )
    assert records._EVIDENCE_V1_SKELETON.isdisjoint(records._EVIDENCE_V1_OPTIONAL)
    assert records._EVIDENCE_V1_OPTIONAL.isdisjoint(records._EVIDENCE_V1_EXCLUDED)







def test_item5_run_spec_hash_scheme_persisted_and_round_trips() -> None:

    record = _run_record_d()
    assert record.run_spec_hash_scheme == "kd-runspec-v1"
    payload = record.to_dict()
    assert payload["run_spec_hash_scheme"] == "kd-runspec-v1"
    assert records.RunRecord.from_dict(payload) == record







def test_item8_boundary_invalid_exceeding_results_rejected() -> None:

    with pytest.raises(ValueError, match="boundary_invalid_results"):
        records.RunCost(
            **_cost_kwargs(boundary_results=2, boundary_invalid_results=3)
        )


def test_item8_boundary_invalid_equal_to_results_is_legal() -> None:

    cost = records.RunCost(
        **_cost_kwargs(boundary_results=3, boundary_invalid_results=3)
    )
    assert cost.boundary_invalid_results == 3
