
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

import kd.search.records as records
from tests.unit.search.test_records import (
    _evidence_d,
    _full_evidence,
    _invalid_evidence_d,
    _minimal_evidence,
    _run_record,
    _run_record_d,
    _valid_record_dict,
)






@pytest.mark.parametrize("bad_version", [1, 3, "2", 2.0])
def test_a2_rejects_unsupported_or_non_int_schema_version(bad_version: object) -> None:

    data = _valid_record_dict()
    data["schema_version"] = bad_version
    with pytest.raises(ValueError, match="schema_version"):
        records.RunRecord.from_dict(data)


def test_a2_rejects_missing_schema_version() -> None:

    data = _valid_record_dict()
    del data["schema_version"]
    with pytest.raises(ValueError, match="schema_version"):
        records.RunRecord.from_dict(data)


def test_a2_rejects_unknown_evidence_hash_scheme() -> None:

    data = _valid_record_dict()
    data["evidence_hash_scheme"] = "unknown-scheme-v9"
    with pytest.raises(ValueError, match="scheme"):
        records.RunRecord.from_dict(data)







def test_a3_rejects_unknown_top_level_key() -> None:
    data = _valid_record_dict()
    data["bogus_field"] = 1
    with pytest.raises(ValueError, match="bogus_field"):
        records.RunRecord.from_dict(data)


def test_a3_rejects_unknown_key_in_cost() -> None:
    data = _valid_record_dict()
    data["cost"]["bogus_cost_key"] = 1
    with pytest.raises(ValueError, match="bogus_cost_key"):
        records.RunRecord.from_dict(data)


def test_a3_rejects_unknown_key_in_evidence() -> None:
    data = _valid_record_dict()
    data["evidence"]["bogus_evidence_key"] = 1
    with pytest.raises(ValueError, match="bogus_evidence_key"):
        records.RunRecord.from_dict(data)


def test_a3_rejects_unknown_key_in_residual_stats() -> None:
    data = _valid_record_dict()
    data["evidence"]["residual_stats"]["bogus_moment"] = 1
    with pytest.raises(ValueError, match="bogus_moment"):
        records.RunRecord.from_dict(data)


def test_a3_rejects_unknown_key_in_equation_payload() -> None:

    data = _valid_record_dict()
    data["evidence"]["catalog_fit"]["bogus_eq_key"] = 1
    with pytest.raises(ValueError, match="bogus_eq_key"):
        records.RunRecord.from_dict(data)


def test_a3_rejects_missing_required_evidence_key() -> None:
    data = _valid_record_dict()
    del data["evidence"]["instrument"]
    with pytest.raises(ValueError, match="instrument"):
        records.RunRecord.from_dict(data)







def test_c_strict_decode_rejects_deleted_evidence_field() -> None:



    data = _valid_record_dict()
    data["evidence"]["recovery_regime"] = "exact"
    with pytest.raises(ValueError, match="recovery_regime"):
        records.RunRecord.from_dict(data)







def test_a5_from_dict_rejects_tampered_evidence() -> None:

    data = _valid_record_dict()
    data["evidence"]["expression"] = "tampered_expression"
    with pytest.raises(ValueError, match="hash"):
        records.RunRecord.from_dict(data)


def test_a5_load_rejects_tampered_evidence(tmp_path: Path) -> None:

    record = _run_record(evidence=_full_evidence())
    path = tmp_path / "record.json"
    record.save(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    data["evidence"]["expression"] = "tampered_expression"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        records.RunRecord.load(path)


def test_a5_save_rejects_wrong_evidence_hash(tmp_path: Path) -> None:


    record = _run_record(evidence=_minimal_evidence(), evidence_hash="0" * 64)
    with pytest.raises(ValueError, match="hash"):
        record.save(tmp_path / "bad.json")


def test_a5_verify_evidence_hash_true_for_consistent_record() -> None:

    assert _run_record(evidence=_full_evidence()).verify_evidence_hash() is True


def test_a5_verify_evidence_hash_false_for_wrong_hash() -> None:
    record = _run_record(evidence=_minimal_evidence(), evidence_hash="0" * 64)
    assert record.verify_evidence_hash() is False







def test_b_strict_decode_rejects_cost_missing_boundary_results() -> None:

    data = _valid_record_dict()
    del data["cost"]["boundary_results"]
    with pytest.raises(ValueError, match="boundary_results"):
        records.RunRecord.from_dict(data)


def test_b_strict_decode_rejects_cost_missing_boundary_invalid_results() -> None:
    data = _valid_record_dict()
    del data["cost"]["boundary_invalid_results"]
    with pytest.raises(ValueError, match="boundary_invalid_results"):
        records.RunRecord.from_dict(data)







def test_d1_decode_rejects_non_vocab_invalid_reason() -> None:



    data = _invalid_evidence_d().to_dict()
    data["invalid_reason"] = "totally_bogus_reason"
    with pytest.raises(ValueError, match="invalid_reason"):
        records.EvidenceRecord.from_dict(data)







def test_d3_headline_source_bad_value_rejected_at_decode() -> None:

    data = _evidence_d().to_dict()
    data["headline_coefficient_source"] = "bogus_source"
    with pytest.raises(ValueError, match="headline_coefficient_source"):
        records.EvidenceRecord.from_dict(data)


def test_d3_headline_source_is_required_key_on_decode() -> None:


    data = _evidence_d().to_dict()
    del data["headline_coefficient_source"]
    with pytest.raises(ValueError, match="headline_coefficient_source"):
        records.EvidenceRecord.from_dict(data)







def test_d5_run_record_strict_decode_requires_run_spec() -> None:

    data = _run_record_d().to_dict()
    del data["run_spec"]
    with pytest.raises(ValueError, match="run_spec"):
        records.RunRecord.from_dict(data)


def test_d5_run_record_strict_decode_requires_run_spec_hash() -> None:

    data = _run_record_d().to_dict()
    del data["run_spec_hash"]
    with pytest.raises(ValueError, match="run_spec_hash"):
        records.RunRecord.from_dict(data)


def test_d5_from_dict_rejects_tampered_run_spec() -> None:



    data = _run_record_d().to_dict()
    data["run_spec"]["kd_version"] = "tampered-9.9.9"
    with pytest.raises(ValueError, match="run_spec"):
        records.RunRecord.from_dict(data)


def test_d5_load_rejects_tampered_run_spec(tmp_path: Path) -> None:

    record = _run_record_d()
    path = tmp_path / "record.json"
    record.save(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    data["run_spec"]["kd_version"] = "tampered-9.9.9"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="run_spec"):
        records.RunRecord.load(path)


def test_d5_save_rejects_wrong_run_spec_hash(tmp_path: Path) -> None:

    record = _run_record_d(run_spec_hash="0" * 64)
    with pytest.raises(ValueError, match="run_spec"):
        record.save(tmp_path / "bad.json")







def test_item1_from_dict_rejects_tampered_cost() -> None:



    data = _run_record_d().to_dict()
    data["cost"]["wallclock_seconds"] = 999.0
    with pytest.raises(records.RecordHashError, match="record hash"):
        records.RunRecord.from_dict(data)


def test_item1_from_dict_rejects_tampered_created_at() -> None:

    data = _run_record_d().to_dict()
    data["created_at"] = "1999-01-01T00:00:00+00:00"
    with pytest.raises(records.RecordHashError, match="record hash"):
        records.RunRecord.from_dict(data)


def test_item1_load_rejects_tampered_cost(tmp_path: Path) -> None:

    record = _run_record_d()
    path = tmp_path / "record.json"
    record.save(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    data["cost"]["wallclock_seconds"] = 999.0
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(records.RecordHashError, match="record hash"):
        records.RunRecord.load(path)


def test_item1_save_rejects_wrong_record_hash(tmp_path: Path) -> None:


    record = dataclasses.replace(_run_record_d(), record_hash="0" * 64)
    with pytest.raises(records.RecordHashError, match="record hash"):
        record.save(tmp_path / "bad.json")


def test_item1_rejects_unknown_record_hash_scheme() -> None:

    data = _run_record_d().to_dict()
    data["record_hash_scheme"] = "kd-record-v99"
    with pytest.raises(records.RecordHashSchemeError, match="record_hash_scheme"):
        records.RunRecord.from_dict(data)


def test_item1_rejects_missing_record_hash() -> None:

    data = _run_record_d().to_dict()
    del data["record_hash"]
    with pytest.raises(ValueError, match="record_hash"):
        records.RunRecord.from_dict(data)







def test_item5_rejects_unknown_run_spec_hash_scheme() -> None:

    data = _run_record_d().to_dict()
    data["run_spec_hash_scheme"] = "kd-runspec-v99"
    with pytest.raises(records.RunSpecHashSchemeError, match="run_spec_hash_scheme"):
        records.RunRecord.from_dict(data)


def test_item5_rejects_missing_run_spec_hash_scheme() -> None:

    data = _run_record_d().to_dict()
    del data["run_spec_hash_scheme"]
    with pytest.raises(ValueError, match="run_spec_hash_scheme"):
        records.RunRecord.from_dict(data)







def test_item3_decode_rejects_valid_record_with_invalid_reason() -> None:


    data = _minimal_evidence().to_dict()
    data["invalid_reason"] = "evaluation_error"
    with pytest.raises(records.StrictDecodeError, match="invalid_reason"):
        records.EvidenceRecord.from_dict(data)


def test_item3_decode_rejects_valid_record_with_error_detail() -> None:

    data = _minimal_evidence().to_dict()
    data["error_detail"] = "boom"
    with pytest.raises(records.StrictDecodeError, match="error_detail"):
        records.EvidenceRecord.from_dict(data)
