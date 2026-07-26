
from __future__ import annotations

import re

from kd.search.mini_table import build_mini_table
from kd.search.records import (
    EVIDENCE_HASH_SCHEME,
    RECORD_HASH_SCHEME,
    RUN_RECORD_SCHEMA_VERSION,
    EvidenceRecord,
    RunCost,
    RunRecord,
    seal_record_hash,
)
from kd.search.run_spec import RUN_SPEC_HASH_SCHEME, RunSpec


def _record(
    instrument: str,
    *,
    expression: str = "u_xx",
    score: float | None = None,
) -> RunRecord:
    evidence = EvidenceRecord(
        instrument=instrument,
        dataset_name="burgers",
        dataset_cache_fingerprint="sha256:dataset",
        seed=0,
        is_valid=True,
        expression=expression,
        score_kind="AIC",
        score_direction="min",
        headline_coefficient_source="native",
        score=score,
    )
    run_spec = RunSpec(
        kd_version="0.1.0",
        config={"algorithm": instrument},
        dataset_cache_fingerprint="sha256:dataset",
    )
    return seal_record_hash(
        RunRecord(
            schema_version=RUN_RECORD_SCHEMA_VERSION,
            evidence_hash_scheme=EVIDENCE_HASH_SCHEME,
            created_at="2026-07-17T00:00:00+00:00",
            cost=RunCost(
                wallclock_seconds=1.23456,
                search_seconds=1.23456,
                boundary_results=0,
                boundary_invalid_results=0,
            ),
            evidence=evidence,
            evidence_hash=evidence.content_hash(),
            run_spec=run_spec,
            run_spec_hash=run_spec.run_spec_hash,
            run_spec_hash_scheme=RUN_SPEC_HASH_SCHEME,
            record_hash="",
            record_hash_scheme=RECORD_HASH_SCHEME,
        )
    )


def _data_rows(table: str) -> list[str]:
    lines = table.splitlines()
    return [line for line in lines[2:] if line.startswith("|")]


def _unescaped_pipe_count(row: str) -> int:
    return len(re.findall(r"(?<!\\)\|", row))


def test_header_and_one_row_per_record_in_input_order() -> None:
    table = build_mini_table([_record("sga"), _record("dlga")])
    lines = table.splitlines()

    assert lines[0] == (
        "| instrument | dataset | expression | score_kind | score | r2 | "
        "wallclock_s | preprocess_s | surrogate_s | boundary_results | "
        "evidence_hash |"
    )
    rows = _data_rows(table)
    assert len(rows) == 2
    assert "| sga |" in rows[0]
    assert "| dlga |" in rows[1]


def test_none_cells_render_as_na() -> None:
    row = _data_rows(build_mini_table([_record("sga")]))[0]



    assert row.count("n/a") >= 4


def test_expression_pipe_is_escaped_without_changing_column_count() -> None:
    table = build_mini_table([_record("sga", expression="u | u_x")])
    lines = table.splitlines()
    row = _data_rows(table)[0]

    assert r"u \| u_x" in row
    assert _unescaped_pipe_count(row) == _unescaped_pipe_count(lines[0])


def test_every_string_cell_is_escaped_and_single_line() -> None:
    record = _record("sga|v2", expression="u\nu_x")
    table = build_mini_table([record])
    lines = table.splitlines()
    rows = _data_rows(table)

    assert len(rows) == 1
    assert r"sga\|v2" in rows[0]
    assert "u u_x" in rows[0]
    assert _unescaped_pipe_count(rows[0]) == _unescaped_pipe_count(lines[0])


def test_floats_and_hash_use_compact_display_formats() -> None:
    record = _record("sga", score=12.3456)
    row = _data_rows(build_mini_table([record]))[0]

    assert "| 12.35 |" in row
    assert "| 1.235 |" in row
    assert record.evidence_hash[:10] in row
    assert record.evidence_hash[10:] not in row


def test_footnote_names_wallclock_as_primary_metric() -> None:
    table = build_mini_table([_record("sga")])

    assert (
        "wallclock_s = primary deployable-cost metric "
        "(fixed-hardware wallclock)"
    ) in table
