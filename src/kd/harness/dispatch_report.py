
from __future__ import annotations

from typing import TYPE_CHECKING

from kd.harness._dispatch_schema import DispatchManifest
from kd.harness.dispatch_log import DispatchLog, WorkerLogRow
from kd.harness.report import build_store_report

if TYPE_CHECKING:
    from kd.harness.store import EvidenceStore

_WORKER_HEADER = (
    "| shard_id | exit_code | kill_reason | sigkill | wallclock_s | heavy |"
)
_WORKER_SEPARATOR = "| --- | --- | --- | --- | --- | --- |"


def _cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def _fmt_wallclock(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _worker_row(row: WorkerLogRow) -> str:
    cells = (
        _cell(row.shard_id),
        _cell(row.exit_code),
        _cell(row.kill_reason),
        _cell(row.sigkill_used),
        _fmt_wallclock(row.wallclock_seconds),
        _cell(row.heavy),
    )
    return "| " + " | ".join(cells) + " |"


def render_dispatch_markdown(manifest: DispatchManifest, log: DispatchLog) -> str:
    ordered = sorted(log.workers, key=lambda row: int(row.shard_id.split("-")[1]))
    lines: list[str] = [
        f"## dispatch ({log.plan_hash[:10]})",
        f"{len(manifest.shards)} shards / heavy_peak {log.heavy_peak}",
        "",
        _WORKER_HEADER,
        _WORKER_SEPARATOR,
    ]
    lines.extend(_worker_row(row) for row in ordered)
    lines.append("")
    lost = ", ".join(str(index) for index in sorted(log.lost_entries))
    lines.append(f"lost_entries: {lost if lost else 'none'}")
    return "\n".join(lines).rstrip() + "\n"


def build_batch_report(
    store: EvidenceStore, manifest: DispatchManifest, log: DispatchLog
) -> str:
    return build_store_report(store) + "\n" + render_dispatch_markdown(manifest, log)


__all__ = [
    "build_batch_report",
    "render_dispatch_markdown",
]
