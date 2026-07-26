
from __future__ import annotations

from pathlib import Path

from kd.harness.report import build_plan_report, build_store_report
from kd.harness.store import EvidenceStore, environment_fingerprint

from ._helpers import (
    attempt_dict,
    completed_outcome,
    make_entry,
    make_plan,
    make_record,
)

_ENV = {"kd_version": "0.4.0", "python": "3.12.0", "platform": "linux"}


def _group_headers(report: str) -> list[str]:
    return [line for line in report.splitlines() if line.startswith("### ")]


def test_title_carries_name_hash_prefix_and_version() -> None:
    plan = make_plan([make_entry("sga", seed=0)], name="matrix_x")
    report = build_plan_report(
        plan=plan,
        plan_hash="abcdef0123456789",
        env=_ENV,
        attempts=[attempt_dict(0, instrument="sga")],
        records={0: make_record("sga")},
    )
    title = report.splitlines()[0]
    assert title.startswith("# ")
    assert "matrix_x" in title
    assert "abcdef0123" in title
    assert "0.4.0" in title


def test_summary_line_counts_statuses() -> None:
    entries = [make_entry("sga", seed=s) for s in range(3)]
    plan = make_plan(entries)
    attempts = [
        attempt_dict(0, instrument="sga", status="completed"),
        attempt_dict(1, instrument="sga", status="raised", error_type="ValueError"),
        attempt_dict(2, instrument="sga", status="no_record"),
    ]
    report = build_plan_report(
        plan=plan,
        plan_hash="h" * 16,
        env=_ENV,
        attempts=attempts,
        records={0: make_record("sga")},
    )
    assert "3 entries / 1 completed / 1 raised / 1 no_record" in report


def test_groups_in_first_appearance_order() -> None:

    entries = [
        make_entry("pysindy", seed=0),
        make_entry("sga", seed=0),
        make_entry("pysindy", seed=1),
    ]
    plan = make_plan(entries)
    records = {
        0: make_record("pysindy", seed=0),
        1: make_record("sga", seed=0),
        2: make_record("pysindy", seed=1),
    }
    attempts = [
        attempt_dict(0, instrument="pysindy"),
        attempt_dict(1, instrument="sga"),
        attempt_dict(2, instrument="pysindy", seed=1),
    ]
    report = build_plan_report(
        plan=plan, plan_hash="h" * 16, env=_ENV, attempts=attempts, records=records
    )
    assert _group_headers(report) == ["### pysindy", "### sga"]


def test_records_grouped_in_entry_order_within_instrument() -> None:
    entries = [make_entry("sga", seed=s) for s in range(2)]
    plan = make_plan(entries)
    records = {
        0: make_record("sga", seed=0, expression="first_term"),
        1: make_record("sga", seed=1, expression="second_term"),
    }
    attempts = [attempt_dict(0), attempt_dict(1, seed=1)]
    report = build_plan_report(
        plan=plan, plan_hash="h" * 16, env=_ENV, attempts=attempts, records=records
    )
    assert report.index("first_term") < report.index("second_term")


def test_no_failures_table_when_all_completed() -> None:
    plan = make_plan([make_entry("sga", seed=0)])
    report = build_plan_report(
        plan=plan,
        plan_hash="h" * 16,
        env=_ENV,
        attempts=[attempt_dict(0)],
        records={0: make_record("sga")},
    )
    assert "### failures" not in report


def test_failures_table_appears_with_non_completed_attempt() -> None:
    entries = [make_entry("sga", seed=0), make_entry("sga", seed=1)]
    plan = make_plan(entries)
    attempts = [
        attempt_dict(0, status="completed"),
        attempt_dict(
            1, seed=1, status="raised", error_type="ValueError", error_message="bad"
        ),
    ]
    report = build_plan_report(
        plan=plan,
        plan_hash="h" * 16,
        env=_ENV,
        attempts=attempts,
        records={0: make_record("sga")},
    )
    assert "### failures" in report
    assert "| entry_index | instrument | dataset_ref | seed | status | error_type |" in (
        report
    )
    assert "ValueError" in report

    assert "| 1 | sga | burgers_tiny | 1 | raised | ValueError |" in report


def test_failure_cell_escapes_pipe_and_newline() -> None:
    plan = make_plan([make_entry("sga", seed=0)])
    attempts = [
        attempt_dict(
            0,
            status="raised",
            error_type="Weird|Error\nwith newline",
        )
    ]
    report = build_plan_report(
        plan=plan, plan_hash="h" * 16, env=_ENV, attempts=attempts, records={}
    )

    assert "Weird\\|Error" in report
    assert "\nwith newline" not in report.split("### failures")[1]


def test_report_surfaces_dataset_fingerprint() -> None:

    plan = make_plan([make_entry("sga", seed=0)])
    report = build_plan_report(
        plan=plan,
        plan_hash="h" * 16,
        env=_ENV,
        attempts=[attempt_dict(0)],
        records={
            0: make_record("sga", dataset_cache_fingerprint="sha256:abcd1234ef")
        },
    )
    assert "dataset burgers_tiny @ sha256:abc" in report


def test_group_header_escapes_newline_injection() -> None:


    instrument = "sga\n### injected"
    plan = make_plan([make_entry(instrument, seed=0)])
    report = build_plan_report(
        plan=plan,
        plan_hash="h" * 16,
        env=_ENV,
        attempts=[attempt_dict(0, instrument=instrument)],
        records={},
    )


    assert "### injected" not in report.splitlines()


def test_build_store_report_delegates(tmp_path: Path) -> None:
    entry = make_entry("sga", seed=0)
    plan = make_plan([entry])
    store = EvidenceStore.create(
        tmp_path / "s", plan=plan, env=environment_fingerprint()
    )
    store.add_outcome(completed_outcome(0, entry, make_record("sga", seed=0)))

    report = build_store_report(store)
    assert report.splitlines()[0].startswith("# ")
    assert "### sga" in report

    assert report == build_plan_report(
        plan=store.plan,
        plan_hash=store.plan_hash,
        env=store.env,
        attempts=store.attempts,
        records=store.records,
    )
