
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from kd.harness.plan import ExperimentPlan
from kd.search.mini_table import build_mini_table
from kd.search.mini_table import escape_cell as _escape_cell
from kd.search.records import RunRecord

if TYPE_CHECKING:
    from pathlib import Path

    from kd.harness.store import EvidenceStore

REPORT_FILENAME = "report.md"

_COMPLETED = "completed"
_FAILURE_HEADER = (
    "| entry_index | instrument | dataset_ref | seed | status | error_type |"
)
_FAILURE_SEPARATOR = "| --- | --- | --- | --- | --- | --- |"


def _instrument_order(plan: ExperimentPlan) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for entry in plan.entries:
        if entry.instrument not in seen:
            seen.add(entry.instrument)
            order.append(entry.instrument)
    return order


def _dataset_identity_lines(
    plan: ExperimentPlan,
    records: Mapping[int, RunRecord],
    group_indices: Sequence[int],
) -> list[str]:
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for index in group_indices:
        ref = plan.entries[index].dataset_ref
        fingerprint = records[index].evidence.dataset_cache_fingerprint
        pair = (ref, fingerprint)
        if pair in seen:
            continue
        seen.add(pair)
        short = fingerprint[:10] if isinstance(fingerprint, str) else "n/a"
        lines.append(f"dataset {_escape_cell(ref)} @ {_escape_cell(short)}")
    return lines


def build_plan_report(
    *,
    plan: ExperimentPlan,
    plan_hash: str,
    env: dict[str, str],
    attempts: Sequence[dict[str, Any]],
    records: Mapping[int, RunRecord],
) -> str:
    kd_version = env.get("kd_version", "unknown")
    lines: list[str] = []


    lines.append(f"# {plan.name} ({plan_hash[:10]}, kd {kd_version})")
    lines.append("")


    completed = sum(1 for a in attempts if a.get("status") == "completed")
    raised = sum(1 for a in attempts if a.get("status") == "raised")
    no_record = sum(1 for a in attempts if a.get("status") == "no_record")
    lines.append(
        f"{len(plan.entries)} entries / {completed} completed / "
        f"{raised} raised / {no_record} no_record"
    )
    lines.append("")


    for instrument in _instrument_order(plan):
        group_indices = [
            index
            for index, entry in enumerate(plan.entries)
            if entry.instrument == instrument and index in records
        ]
        group_records = [records[index] for index in group_indices]



        lines.append(f"### {_escape_cell(instrument)}")
        lines.extend(_dataset_identity_lines(plan, records, group_indices))
        lines.append(build_mini_table(group_records))
        lines.append("")


    failures = [a for a in attempts if a.get("status") != _COMPLETED]
    if failures:
        lines.append("### failures")
        lines.append(_FAILURE_HEADER)
        lines.append(_FAILURE_SEPARATOR)
        for attempt in failures:
            error_type = attempt.get("error_type")
            cells = (
                _escape_cell(str(attempt.get("entry_index"))),
                _escape_cell(str(attempt.get("instrument"))),
                _escape_cell(str(attempt.get("dataset_ref"))),
                _escape_cell(str(attempt.get("seed"))),
                _escape_cell(str(attempt.get("status"))),
                _escape_cell("" if error_type is None else str(error_type)),
            )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_store_report(store: EvidenceStore) -> str:
    return build_plan_report(
        plan=store.plan,
        plan_hash=store.plan_hash,
        env=store.env,
        attempts=store.attempts,
        records=store.records,
    )


def write_store_report(store: EvidenceStore) -> Path:
    report_path = store.root / REPORT_FILENAME
    tmp_path = report_path.with_name(f"{report_path.name}.tmp")
    tmp_path.write_text(build_store_report(store), encoding="utf-8")
    os.replace(tmp_path, report_path)
    return report_path
