
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import sympy

import kd
from kd.search import BEST_SCORE_KEY, RunRecord
from kdui.api.runs import MANIFEST_FILE, read_manifest, record_path
from kdui.models import (
    Candidate,
    CandidateList,
    CandidatesRequest,
    CheckpointInfo,
    CheckpointList,
    CheckpointsRequest,
    Convergence,
    DatasetRecord,
    Equation,
    EquationTerm,
    FitRequest,
    LaunchRecord,
    NativeScore,
    Preprocessing,
    Provenance,
    RunAvailability,
    RunCost,
    RunDetail,
    RunMetrics,
    RunOrigin,
    RunProjectionRequest,
    RunStatus,
    RunSummary,
    ScoreDirection,
    StoredJob,
)

logger = logging.getLogger(__name__)


_TOPK_EXPRESSIONS = "search_topk_expressions"
_TOPK_SCORES = "search_topk_scores"
_TOPK_TERMS = "search_topk_terms"
_TOPK_LHS = "search_topk_lhs"
_PARETO_EXPRESSIONS = "pareto_expressions"
_PARETO_COMPLEXITY = "pareto_complexity"
_PARETO_LOSS = "pareto_loss"
_PARETO_NMSE = "pareto_nmse"

ENVIRONMENT_FILE = "environment.json"
REPORT_FILE = "report/report.html"
RESULT_FILE = "result.json"
CHECKPOINTS_DIR = "checkpoints"


def finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def direction_of(direction: str) -> ScoreDirection:
    return "max" if direction == "max" else "min"


def _latex(code: str) -> str:
    return str(sympy.latex(kd.to_sympy(code.strip(), strict=True)))


def expression_latex(expression: str) -> str:
    if "=" in expression:
        lhs, rhs = expression.split("=", 1)
        return f"{_latex(lhs)} = {_latex(rhs)}"
    return _latex(expression)







def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_line(path: Path) -> dict[str, Any] | None:
    with path.open(encoding="utf-8") as handle:
        line = handle.readline()
    return json.loads(line) if line.endswith("\n") else None


@dataclass(frozen=True)
class RunFiles:

    run_dir: Path
    manifest: dict[str, Any] | None
    record: RunRecord | None
    recorder: dict[str, Any] | None
    job: StoredJob | None
    ended: bool
    environment: dict[str, Any] | None


def job_ended(job_dir: Path) -> bool:
    if (job_dir / "result.json").is_file():
        return True
    state = job_dir / "state.jsonl"
    if not state.is_file():
        return False
    return any(
        json.loads(line).get("record") == "worker_ended"
        for line in state.read_text(encoding="utf-8").splitlines(keepends=True)
        if line.endswith("\n")
    )


def sealed_record(run_dir: Path) -> RunRecord | None:
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        return None
    path = record_path(run_dir, read_manifest(run_dir))
    return RunRecord.load(path) if path.is_file() else None


def read_run_files(run_dir: Path, job_dir: Path | None) -> RunFiles:
    job = (
        StoredJob.model_validate_json(
            (job_dir / "request.json").read_text(encoding="utf-8")
        )
        if job_dir is not None
        else None
    )
    ended = job_dir is None or job_ended(job_dir)
    manifest_path = run_dir / MANIFEST_FILE
    manifest = read_manifest(run_dir) if manifest_path.is_file() else None
    record = None
    recorder = None
    if manifest is not None:
        path = record_path(run_dir, manifest)
        record = RunRecord.load(path) if path.is_file() else None
        if (run_dir / "recorder.json").is_file():
            recorder = _read_json(run_dir / "recorder.json")
    environment_path = run_dir / ENVIRONMENT_FILE
    environment = (
        _read_json(environment_path) if ended and environment_path.is_file() else None
    )
    return RunFiles(run_dir, manifest, record, recorder, job, ended, environment)







def _fit_request(files: RunFiles) -> FitRequest | None:
    if files.job is None:
        return None
    request = files.job.request
    return request if isinstance(request, FitRequest) else None


def _savgol_window(request: FitRequest | None) -> int | None:
    if request is None:
        return None
    window = request.params.get("savgol_window", 0)
    if not isinstance(window, int):
        raise TypeError(f"savgol_window of the request is {window!r}, not an integer")
    return window


def _dataset_name(files: RunFiles, request: FitRequest | None) -> str | None:
    if files.record is not None:
        return files.record.evidence.dataset_name
    if request is None:
        return None
    workspace = files.run_dir.parent.parent
    path = workspace / "datasets" / request.dataset_id / "dataset.json"
    if not path.is_file():
        return None
    return DatasetRecord.model_validate_json(path.read_text(encoding="utf-8")).name


def _created_at(files: RunFiles) -> datetime | None:
    if files.manifest is not None:
        return datetime.fromisoformat(files.manifest["created_at"])
    phases = files.run_dir / "phases.jsonl"
    if phases.is_file():
        first = _first_line(phases)
        if first is not None:
            return datetime.fromisoformat(first["created_at"])
    return files.job.created_at if files.job is not None else None


def _loaded_fingerprint(
    files: RunFiles, fingerprint: str | None, preprocessing: Preprocessing | None
) -> str | None:
    if files.environment is not None:
        return str(files.environment["loaded_fingerprint"])
    if preprocessing is None or preprocessing.smoothing is None:
        return fingerprint
    return None


def summarize(run: RunProjectionRequest, files: RunFiles) -> RunSummary:
    request = _fit_request(files)
    record = files.record
    manifest = files.manifest
    evidence = record.evidence if record is not None else None
    preprocessing = request.preprocessing if request is not None else None
    fingerprint = (
        evidence.dataset_cache_fingerprint
        if evidence is not None
        else manifest.get("dataset_cache_fingerprint")
        if manifest is not None
        else None
    )
    formatted = kd.format_record_equation(record) if record is not None else None
    return RunSummary(
        run_id=run.run_id,
        origin=run.origin,
        job_id=run.run_id if run.origin is RunOrigin.UI else None,
        source_id=run.source_id,
        kd_run_id=manifest.get("run_id") if manifest is not None else None,
        status=RunStatus(manifest["status"])
        if manifest is not None
        else RunStatus.UNSEALED,
        created_at=_created_at(files),
        algorithm=(
            manifest["instrument"]
            if manifest is not None
            else request.algorithm
            if request is not None
            else None
        ),
        seed=(
            manifest.get("seed")
            if manifest is not None
            else request.seeds[0]
            if request is not None
            else None
        ),
        dataset_id=request.dataset_id if request is not None else None,
        dataset_name=_dataset_name(files, request),
        dataset_fingerprint=fingerprint,
        loaded_fingerprint=_loaded_fingerprint(files, fingerprint, preprocessing),
        preprocessing=preprocessing,
        savgol_window=_savgol_window(request),
        equation_latex=formatted.latex if formatted is not None else None,
        native_score=(
            NativeScore(
                value=finite(evidence.score),
                kind=evidence.score_kind,
                direction=direction_of(evidence.score_direction),
            )
            if evidence is not None
            else None
        ),
        nmse=finite(evidence.nmse) if evidence is not None else None,
        stop_reason=record.cost.stop_reason if record is not None else None,
        prior_knowledge=files.job.prior_knowledge if files.job is not None else None,
        lineage=files.job.lineage if files.job is not None else None,
    )


def _equation(record: RunRecord) -> Equation | None:
    formatted = kd.format_record_equation(record)
    if formatted is None:
        return None
    evidence = record.evidence
    terms = [
        EquationTerm(latex=expression_latex(term), coefficient=coefficient)
        for term, coefficient in zip(
            evidence.support or [], evidence.coefficients or [], strict=True
        )
        if coefficient is not None
    ]
    return Equation(
        latex=formatted.latex,
        lhs_latex=str(sympy.latex(formatted.sympy_expr.lhs)),
        terms=terms,
    )


def _cost(record: RunRecord) -> RunCost:
    cost = record.cost
    return RunCost(
        wallclock_seconds=cost.wallclock_seconds,
        search_seconds=cost.search_seconds,
        preprocessing_seconds=cost.preprocessing_seconds,
        surrogate_train_seconds=cost.surrogate_train_seconds,
        cpu_seconds=cost.cpu_seconds,
        tokens_in=cost.tokens_in,
        tokens_out=cost.tokens_out,
        peak_rss_bytes=cost.peak_rss_bytes,
        device=cost.device,
        iterations=cost.iterations,
        max_iterations=cost.max_iterations,
    )


def _convergence(record: RunRecord, recorder: dict[str, Any]) -> Convergence | None:
    series = recorder.get(BEST_SCORE_KEY)
    if not series:
        return None
    return Convergence(
        score_kind=record.evidence.score_kind,
        score_direction=direction_of(record.evidence.score_direction),
        iterations=list(range(len(series))),
        best_score=[finite(value) for value in series],
    )


def _hostname(files: RunFiles) -> str | None:
    if files.job is None:
        return None
    launch = files.run_dir.parent.parent / "jobs" / files.job.job_id / "launch.json"
    if not launch.is_file():
        return None
    return LaunchRecord.model_validate_json(launch.read_text(encoding="utf-8")).hostname


def _has_checkpoints(files: RunFiles) -> bool:
    if not files.ended:
        return False
    try:
        return bool(kd.eligible_checkpoint_iterations(files.run_dir / CHECKPOINTS_DIR))
    except kd.CheckpointManifestError:


        logger.warning("unreadable checkpoint ledger in %s", files.run_dir)
        return False


def project_run(request: RunProjectionRequest) -> RunDetail:
    run_dir = Path(request.run_dir)
    files = read_run_files(run_dir, Path(request.job_dir) if request.job_dir else None)
    summary = summarize(request, files)
    record = files.record
    recorder = files.recorder
    convergence = (
        _convergence(record, recorder)
        if record is not None and recorder is not None
        else None
    )
    candidates = (
        _candidates(record, recorder, None)
        if record is not None and recorder is not None
        else None
    )
    evidence = record.evidence if record is not None else None
    environment = files.environment
    return RunDetail(
        **summary.model_dump(),
        equation=_equation(record) if record is not None else None,
        metrics=(
            RunMetrics(
                nmse=finite(evidence.nmse),
                mse=finite(evidence.mse),
                r2=finite(evidence.r2),
                complexity=evidence.complexity,
            )
            if evidence is not None
            else None
        ),
        cost=_cost(record) if record is not None else None,
        config=record.run_spec.config if record is not None else None,
        convergence=convergence,
        candidates=candidates,
        sketch_outcome=None,
        provenance=Provenance(
            kd_version=record.run_spec.kd_version if record is not None else None,
            config_hash=files.manifest.get("config_hash") if files.manifest else None,
            record_hash=record.record_hash if record is not None else None,
            hostname=_hostname(files),
            run_dir=str(run_dir),
            environment=(
                {
                    str(key): str(value)
                    for key, value in environment["environment"].items()
                }
                if environment is not None
                else None
            ),
        ),
        availability=RunAvailability(
            record=record is not None,
            convergence=convergence is not None,
            candidates=candidates is not None,
            report=files.ended and (run_dir / REPORT_FILE).is_file(),
            result_file=files.ended and (run_dir / RESULT_FILE).is_file(),
            checkpoints=_has_checkpoints(files),
        ),
        request=_fit_request(files),
    )







def _candidates(
    record: RunRecord, recorder: dict[str, Any], iteration: int | None
) -> CandidateList | None:
    evidence = record.evidence
    kind = evidence.score_kind
    direction = direction_of(evidence.score_direction)
    expressions = recorder.get(_TOPK_EXPRESSIONS)
    if expressions:
        iterations = [index for index, row in enumerate(expressions) if row]
        if not iterations:
            return None
        chosen = iterations[-1] if iteration is None else iteration
        if chosen not in iterations:
            raise ValueError(
                f"no candidates were kept for iteration {chosen}; iterations with "
                f"candidates are {iterations[0]} to {iterations[-1]}"
            )
        rows = [
            Candidate(
                rank=rank,
                equation_latex=expression_latex(f"{lhs} = {' + '.join(terms) or '0'}"),
                score=finite(score),
                complexity=len(terms),
            )
            for rank, (score, terms, lhs) in enumerate(
                zip(
                    recorder[_TOPK_SCORES][chosen],
                    recorder[_TOPK_TERMS][chosen],
                    recorder[_TOPK_LHS][chosen],
                    strict=True,
                ),
                start=1,
            )
        ]
        return CandidateList(
            form="top_k",
            score_kind=kind,
            score_direction=direction,
            iteration=chosen,
            iterations=iterations,
            rows=rows,
        )
    front = recorder.get(_PARETO_EXPRESSIONS)
    if not front:
        return None
    formatted = kd.format_record_equation(record)
    lhs = f"{formatted.lhs} = " if formatted is not None else ""


    scores = recorder.get(_PARETO_NMSE) or recorder[_PARETO_LOSS]
    rows = [
        Candidate(
            rank=rank,
            equation_latex=expression_latex(f"{lhs}{expression}"),
            score=finite(score),
            complexity=complexity,
        )
        for rank, (expression, score, complexity) in enumerate(
            zip(front[-1], scores[-1], recorder[_PARETO_COMPLEXITY][-1], strict=True),
            start=1,
        )
    ]
    return CandidateList(
        form="pareto",
        score_kind=kind,
        score_direction=direction,
        iteration=None,
        iterations=[],
        rows=rows,
    )


def project_candidates(request: CandidatesRequest) -> CandidateList:
    run_dir = Path(request.run_dir)
    record = sealed_record(run_dir)
    if record is None or not (run_dir / "recorder.json").is_file():
        raise ValueError("the run has no sealed record and recorder, so no candidates")
    recorder = _read_json(run_dir / "recorder.json")
    candidates = _candidates(record, recorder, request.iteration)
    if candidates is None:
        raise ValueError("this search kept no candidate list")
    return candidates







def _algorithm_of(run_dir: Path) -> str | None:
    manifest_path = run_dir / MANIFEST_FILE
    if manifest_path.is_file():
        return str(read_manifest(run_dir)["instrument"])
    checkpoints = run_dir / CHECKPOINTS_DIR
    if not checkpoints.is_dir():
        return None
    entries = kd.load_checkpoint_manifest(checkpoints)
    return entries[0].algorithm if entries else None


def resume_safe_params(schema: dict[str, Any]) -> list[str]:
    names = [
        field["facade_param"] or field["name"]
        for field in schema["fields"]
        if field["resume_tier"] == "resume_safe"
    ]
    budget = any(param["name"] == "generations" for param in schema["facade_params"])
    if budget and "generations" not in names and not schema["one_shot"]:
        names.append("generations")
    return names


def list_checkpoints(request: CheckpointsRequest) -> CheckpointList:
    run_dir = Path(request.run_dir)
    algorithm = _algorithm_of(run_dir)
    if algorithm is None:
        return CheckpointList(
            run_id=request.run_id,
            resumable=False,
            reason="the run wrote no manifest and no checkpoint",
            unit=None,
            checkpoints=[],
            resume_safe_params=[],
        )
    schema = next(s for s in kd.instrument_schemas() if s["algorithm"] == algorithm)
    segmentation = schema["segmentation"]

    def refused(reason: str) -> CheckpointList:
        return CheckpointList(
            run_id=request.run_id,
            resumable=False,
            reason=reason,
            unit=segmentation["unit"],
            checkpoints=[],
            resume_safe_params=[],
        )

    if segmentation["archive"] != "progress":
        return refused(
            f"{algorithm} keeps no search progress to resume: its checkpoint "
            "holds a finished fit."
        )
    checkpoint_dir = run_dir / CHECKPOINTS_DIR
    try:
        iterations = kd.eligible_checkpoint_iterations(checkpoint_dir)
        if not iterations:
            kd.resolve_checkpoint(checkpoint_dir)
    except (kd.CheckpointManifestError, kd.CheckpointSelectionError) as error:
        return refused(str(error))
    return CheckpointList(
        run_id=request.run_id,
        resumable=True,
        reason=None,
        unit=segmentation["unit"],
        checkpoints=[
            CheckpointInfo(
                iteration=iteration,
                size_bytes=kd.resolve_checkpoint(checkpoint_dir, iteration)
                .path.stat()
                .st_size,
            )
            for iteration in iterations
        ],
        resume_safe_params=resume_safe_params(schema),
    )
