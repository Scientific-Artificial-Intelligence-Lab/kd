
from __future__ import annotations

import logging
from pathlib import Path

import kd
from kdui.api.runs import UnknownRunError, resolve_run
from kdui.jobstore import JobContext
from kdui.models import (
    CompareRequest,
    CompareResult,
    ComparisonRow,
    FitRequest,
    RunDetail,
    RunOrigin,
)
from kdui.ops.inspect import load_dataset
from kdui.ops.result import finite, project_run, sealed_record

logger = logging.getLogger(__name__)

FRAME_METHOD = "finite_diff"


def _group(details: list[RunDetail]) -> tuple[str, str]:
    keys = {(detail.dataset_name, detail.loaded_fingerprint) for detail in details}
    if len(keys) != 1:
        found = "; ".join(
            f"{name} ({fingerprint})" for name, fingerprint in sorted(keys, key=str)
        )
        raise ValueError(f"runs on different data are not compared: {found}")
    ((name, fingerprint),) = keys
    if name is None or fingerprint is None:
        raise ValueError(
            "a run without a sealed record has no dataset fingerprint to compare by"
        )
    return name, fingerprint


def _frame_text(frame: kd.RefitFrame) -> str:
    return (
        f"{frame.derivative_method} derivatives of the {frame.field_source} field "
        f"as loaded, target {frame.lhs_field} order {frame.lhs_order} in "
        f"{frame.lhs_axis}, {frame.n_samples} samples"
    )


def run_compare(request: CompareRequest, job: JobContext) -> CompareResult:
    runs = []
    for run_id in request.run_ids:
        try:
            runs.append(resolve_run(job.workspace, run_id))
        except UnknownRunError as error:
            raise ValueError(f"no run {run_id}") from error
    details = [project_run(run) for run in runs]
    _, fingerprint = _group(details)
    registered = next(
        (
            detail.request
            for detail in details
            if isinstance(detail.request, FitRequest)
        ),
        None,
    )
    if registered is None:
        raise ValueError(
            "none of these runs was started from the UI, so the UI has no "
            "registered dataset to re-measure them on"
        )
    dataset = load_dataset(job.workspace, registered.dataset_id, with_reference=False)
    supports = []
    for detail in details:
        record = sealed_record(Path(detail.provenance.run_dir))
        supports.append(record.evidence.support if record is not None else None)
    measured = [support for support in supports if support]
    scores = kd.score_term_sets(dataset, measured, method=FRAME_METHOD)
    entries = iter(scores.entries)
    rows = []
    for detail, support in zip(details, supports, strict=True):
        entry = next(entries) if support else None
        fit = entry.fit if entry is not None else None
        reason = "the run has no sealed equation" if entry is None else entry.failure
        rows.append(
            ComparisonRow(
                run_id=detail.run_id,
                algorithm=detail.algorithm or "",
                seed=detail.seed,
                equation=detail.equation,
                native_score=detail.native_score,
                nmse=finite(fit.nmse) if fit is not None else None,
                complexity=detail.metrics.complexity if detail.metrics else None,
                prior_knowledge=detail.prior_knowledge,
                preprocessing=(
                    detail.preprocessing if detail.origin is RunOrigin.UI else None
                ),
                savgol_window=detail.savgol_window,
                error=reason if fit is None else None,
            )
        )
    return CompareResult(
        dataset_fingerprint=fingerprint,
        frame=_frame_text(scores.frame),
        rows=rows,
        figure_artifact_id=None,
    )
