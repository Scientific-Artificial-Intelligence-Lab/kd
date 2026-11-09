
from __future__ import annotations

import logging
from pathlib import Path

from kdui.api.runs import (
    MANIFEST_FILE,
    ManifestError,
    discover_run_dirs,
    read_manifest,
    source_runs,
    ui_runs,
)
from kdui.models import (
    HistoryScanRequest,
    HistorySource,
    RunList,
    RunListRequest,
    RunProjectionRequest,
    RunStatus,
    RunSummary,
)
from kdui.ops.result import read_run_files, summarize
from kdui.settings import Workspace

logger = logging.getLogger(__name__)


def scan_history(request: HistoryScanRequest) -> HistorySource:
    root = Path(request.path)
    walked = discover_run_dirs(root)
    for error in walked.unreadable:
        if error.directory == root:
            raise error
    run_dirs = walked.found
    unsealed = [path for path in run_dirs if not (path / MANIFEST_FILE).is_file()]
    return HistorySource(
        source_id=request.source_id,
        path=request.path,
        imported_at=request.imported_at,
        n_runs=len(run_dirs),
        n_unsealed=len(unsealed),
    )


def _summary(run: RunProjectionRequest) -> RunSummary:
    try:
        files = read_run_files(
            Path(run.run_dir), Path(run.job_dir) if run.job_dir else None
        )
        return summarize(run, files)
    except (ValueError, KeyError, TypeError) as error:
        logger.warning(
            "run %s in %s cannot be read: %s", run.run_id, run.run_dir, error
        )
    run_dir = Path(run.run_dir)
    status = RunStatus.UNSEALED
    if (run_dir / MANIFEST_FILE).is_file():
        status = RunStatus(read_manifest(run_dir)["status"])
    return RunSummary(
        run_id=run.run_id,
        origin=run.origin,
        job_id=None,
        source_id=run.source_id,
        kd_run_id=None,
        status=status,
        created_at=None,
        algorithm=None,
        seed=None,
        dataset_id=None,
        dataset_name=None,
        dataset_fingerprint=None,
        loaded_fingerprint=None,
        preprocessing=None,
        savgol_window=None,
        equation_latex=None,
        native_score=None,
        nmse=None,
        stop_reason=None,
        prior_knowledge=None,
        lineage=None,
    )


def list_runs(request: RunListRequest) -> RunList:
    runs = ui_runs(Workspace(Path(request.workspace)))
    unreadable: list[ManifestError] = []
    for source in request.sources:
        found, left_out = source_runs(source)
        runs.extend(found)
        unreadable.extend(left_out)
    summaries = []
    for run in runs:
        try:
            summaries.append(_summary(run))
        except ManifestError as error:
            unreadable.append(error)
    warnings = [
        f"Not listed: {error.directory} has a {MANIFEST_FILE} that is not valid "
        f"JSON ({error.reason}). Repair or remove that file to list its run."
        for error in unreadable
    ]
    return RunList(runs=summaries, next_cursor=None, warnings=warnings)
