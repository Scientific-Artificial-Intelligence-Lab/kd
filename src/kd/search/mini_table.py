
from __future__ import annotations

from collections.abc import Sequence

from kd.search.records import RunRecord

_HEADER = (
    "| instrument | dataset | expression | score_kind | score | r2 | "
    "wallclock_s | preprocess_s | surrogate_s | boundary_results | evidence_hash |"
)
_SEPARATOR = "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
_FOOTNOTE = (
    "wallclock_s = primary deployable-cost metric (fixed-hardware wallclock)"
)


def _format_value(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def escape_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def _record_row(record: RunRecord) -> str:
    evidence = record.evidence
    cost = record.cost
    cells = (
        escape_cell(evidence.instrument),
        escape_cell(evidence.dataset_name),
        escape_cell(evidence.expression),
        escape_cell(evidence.score_kind),
        _format_value(evidence.score),
        _format_value(evidence.r2),
        _format_value(cost.wallclock_seconds),
        _format_value(cost.preprocessing_seconds),
        _format_value(cost.surrogate_train_seconds),
        _format_value(cost.boundary_results),
        record.evidence_hash[:10],
    )
    return "| " + " | ".join(cells) + " |"


def build_mini_table(records: Sequence[RunRecord]) -> str:
    rows = [_record_row(record) for record in records]
    return "\n".join([_HEADER, _SEPARATOR, *rows, "", _FOOTNOTE])
