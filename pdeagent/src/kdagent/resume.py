
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kd
from kd.core.equation import sketch_from_dict
from kd.search import (
    SKETCH_CLAUSES,
    RunDirPaths,
    append_catalog_row,
    catalog_row_from_record,
    used_clauses,
)
from kdagent.lineage import SegmentLedger, SegmentSummary
from kdagent.segment_report import law_from_evidence
from kdagent.session import Session
from kdagent.surrogates import file_sha256


def _failure(code: str, detail: str) -> dict[str, Any]:
    return {"code": code, "detail": detail}


def _crashed(exc: Exception) -> dict[str, Any]:
    return _failure(type(exc).__name__, str(exc))


@dataclass(frozen=True)
class _ResolvedResume:

    parent_run_id: str
    iteration: int
    checkpoint: Path
    seed: int


def _lineage_provenance(
    resume: _ResolvedResume | None,
    workspace: Path,
    *,
    requested_parent: str | None = None,
) -> dict[str, Any]:
    if resume is None:
        return {"parent_run_id": requested_parent}
    return {
        "parent_run_id": resume.parent_run_id,
        "resume_iteration": resume.iteration,
        "resume_from": os.path.relpath(resume.checkpoint, workspace),
    }


def _record_run(
    outcome: Any,
    catalog_path: Path,
    *,
    resume: _ResolvedResume | None,
    created_at: str,
) -> None:
    if outcome.run_dir is None or outcome.run_id is None:
        return
    record = RunDirPaths(root=outcome.run_dir).record
    append_catalog_row(
        catalog_path,
        catalog_row_from_record(
            outcome.record,
            run_id=outcome.run_id,
            created_at=created_at,
            instrument=outcome.entry.instrument,
            status=outcome.status,
            run_dir=os.path.relpath(outcome.run_dir, catalog_path.parent),
            dataset_name=outcome.entry.dataset_ref,
            seed=outcome.entry.seed,
            record_path=(
                os.path.relpath(record, catalog_path.parent)
                if record.is_file()
                else None
            ),
            parent_run_id=None if resume is None else resume.parent_run_id,
            resume_from=(
                None
                if resume is None
                else os.path.relpath(resume.checkpoint, catalog_path.parent)
            ),
        ),
    )


def segment_action(
    *,
    resumed: bool,
    reseed: bool,
    resume_iteration: int | None,
    switched: bool = False,
    narrowed: bool = False,
) -> str:
    if resumed:
        if reseed:
            return "fork"
        if resume_iteration is not None:
            return "rollback"
        return "continue"
    if switched:
        return "switch"
    if narrowed:
        return "narrow"
    return "fresh"


def move_facts(
    ledger: SegmentLedger,
    *,
    run_id: str,
    decision_parent: str | None,
    instrument: str,
    dataset: str,
    sketch: dict[str, Any] | None,
) -> tuple[bool, bool]:
    if decision_parent is None:
        return False, False
    try:
        parent = ledger.entry(decision_parent)
    except KeyError:
        raise ValueError(
            f"segment {run_id!r} answers to {decision_parent!r}, which no "
            "segment event recorded"
        ) from None
    same_dataset = parent.dataset == dataset
    switched = same_dataset and parent.instrument != instrument
    narrowed = sketch is not None and same_dataset and parent.instrument == instrument
    return switched, narrowed


def eligible_iterations(run_dir: Path) -> tuple[int, ...]:
    return kd.eligible_checkpoint_iterations(RunDirPaths(root=run_dir).checkpoints)


def _record_segment(
    outcome: Any,
    session: Session,
    *,
    resume: _ResolvedResume | None,
    reseed: bool,
    created_at: str,
    summary: SegmentSummary | None,
    decision_parent: str | None,
    resume_iteration: int | None,
    params: dict[str, Any] | None,
    report: dict[str, Any] | None,
    sketch: dict[str, Any] | None = None,
    surrogate: str | None = None,
    platform_nmse: float | None = None,
    platform_frame: str | None = None,
    platform_coefficients: list[float] | None = None,
    cost: dict[str, Any] | None = None,
) -> None:
    if outcome.run_dir is None or outcome.run_id is None:
        return
    switched, narrowed = (
        (False, False)
        if resume is not None
        else move_facts(
            session.ledger,
            run_id=outcome.run_id,
            decision_parent=decision_parent,
            instrument=outcome.entry.instrument,
            dataset=outcome.entry.dataset_ref,
            sketch=sketch,
        )
    )
    session.ledger.record(
        run_id=outcome.run_id,
        run_dir=os.path.relpath(outcome.run_dir, session.workspace),
        parent_run_id=None if resume is None else resume.parent_run_id,
        resume_from=(
            None
            if resume is None
            else {
                "path": os.path.relpath(resume.checkpoint, session.workspace),
                "iteration": resume.iteration,
            }
        ),
        instrument=outcome.entry.instrument,
        dataset=outcome.entry.dataset_ref,
        seed=outcome.entry.seed,
        reseed=reseed,
        status=outcome.status,
        nmse=(None if outcome.record is None else outcome.record.evidence.nmse),
        created_at=created_at,
        summary=summary,
        decision_parent=decision_parent,
        action=segment_action(
            resumed=resume is not None,
            reseed=reseed,
            resume_iteration=resume_iteration,
            switched=switched,
            narrowed=narrowed,
        ),
        params=params,
        law=(
            None
            if outcome.record is None
            else law_from_evidence(outcome.record.evidence)
        ),
        report_projection=report_projection(report),
        sketch=sketch,
        surrogate=surrogate,
        platform_nmse=platform_nmse,
        platform_frame=platform_frame,
        platform_coefficients=platform_coefficients,
        cost=cost,
    )


def segment_cost(
    outcome: Any,
    *,
    evaluation_cap: int | None,
    stopped_by_evaluation_cap: bool,
) -> dict[str, Any] | None:
    record = getattr(outcome, "record", None)
    if record is None:
        return None
    run_cost = record.cost
    return {
        "evaluations": int(run_cost.boundary_results),
        "invalid_evaluations": int(run_cost.boundary_invalid_results),
        "search_seconds": round(float(run_cost.search_seconds), 3),
        "evaluation_cap": evaluation_cap,
        "stopped_by_evaluation_cap": bool(stopped_by_evaluation_cap),
    }


def report_projection(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    blocks = {
        name: report[name]
        for name in ("validity", "diversity", "complexity")
        if name in report
    }
    return blocks or None


def _checkpoint_control_failure(
    checkpoint_every: int | None,
    checkpoint_keep_last: int | None,
) -> dict[str, Any] | None:
    if checkpoint_every is not None and (
        type(checkpoint_every) is not int or checkpoint_every < 1
    ):
        return _failure(
            "invalid_argument",
            "checkpoint_every has to be a positive integer or null; null "
            "disables checkpointing",
        )
    if checkpoint_every is None and checkpoint_keep_last is not None:
        return _failure(
            "invalid_argument",
            "checkpoint_keep_last can only be set together with "
            "checkpoint_every; set both to null to disable checkpointing",
        )
    if checkpoint_every is not None and checkpoint_keep_last is None:




        return _failure(
            "invalid_argument",
            "with checkpoint_every set, checkpoint_keep_last needs a retention "
            "bound; this tool does not offer unbounded retention. Set both to "
            "null if you do not want checkpoints",
        )
    if checkpoint_keep_last is not None and (
        type(checkpoint_keep_last) is not int or checkpoint_keep_last < 1
    ):
        return _failure(
            "invalid_argument",
            "checkpoint_keep_last has to be a positive integer or null",
        )
    return None


def _reseed_control_failure(
    algorithm: str,
    *,
    reseed: bool,
    resume_from_run_id: str | None,
    seed: int | None,
) -> dict[str, Any] | None:
    if not reseed:
        return None
    schema = schema_of(algorithm)
    if schema is not None and not schema["segmentation"]["reseed"]:
        return _failure(
            "reseed_unsupported",
            f"{algorithm} declares segmentation.reseed=False: its search stream "
            "cannot be re-derived from another seed on a restored checkpoint, "
            "so it has no branch to offer. Drop reseed to continue the segment "
            "as it stands, or switch to an algorithm that declares reseed "
            "support",
        )
    if resume_from_run_id is None:
        return _failure(
            "invalid_argument",
            "reseed requires resume_from_run_id as well: it branches a restored "
            "search onto another random stream, and a call with nothing to "
            "restore already starts from a fresh one",
        )
    if seed is None:




        return _failure(
            "invalid_argument",
            "reseed requires an explicit seed: an omitted seed inherits the "
            f"parent's, so every branch of run {resume_from_run_id!r} would "
            "re-derive one and the same stream. Pass the seed this branch "
            "should run",
        )
    return None


def _resolve_resume(
    session: Session,
    run_id: str,
    iteration: int | None,
    *,
    algorithm: str,
    dataset_id: str,
    seed: int | None,
    reseed: bool = False,
) -> tuple[_ResolvedResume | None, dict[str, Any] | None]:




    if session.ledger.is_pruned(run_id):
        return None, _failure(
            "pruned_run_id",
            f"run {run_id!r} was pruned from the policy tree (itself or an "
            "ancestor of it), so this branch is abandoned and cannot be "
            "continued. Resume from a run that is still on the tree, or start "
            "a fresh segment",
        )
    resumable = session.ledger.resumable_ids()
    if run_id not in resumable:
        return None, _failure(
            "unknown_run_id",
            f"resume_from_run_id {run_id!r} is not a resumable search from this "
            "run; "
            + (
                f"the resumable ones are {list(resumable)}"
                if resumable
                else "this run has no search with a surviving run directory yet"
            ),
        )
    source = session.ledger.entry(run_id)
    if source.instrument != algorithm:





        return None, _failure(
            "instrument_mismatch",
            f"run {run_id!r} was produced by {source.instrument} and this call "
            f"asks for {algorithm}; only the same algorithm can continue from a "
            f"checkpoint. To resume, set algorithm to {source.instrument}; to "
            f"switch algorithms, drop resume_from_run_id",
        )
    if source.dataset != dataset_id:




        return None, _failure(
            "dataset_mismatch",
            f"run {run_id!r} ran on dataset {source.dataset!r} and this call "
            f"asks for {dataset_id!r}; a resumed search has to stay on the "
            f"original dataset. To search {dataset_id!r}, drop "
            "resume_from_run_id and start a fresh segment",
        )
    if not reseed and seed is not None and seed != source.seed:
        return None, _failure(
            "seed_mismatch",
            f"run {run_id!r} was checkpointed with seed {source.seed} and this "
            f"call passes {seed}; seed is an init_only field and a resumed "
            f"search cannot change it. Omit seed to continue this segment (it "
            f"inherits {source.seed}); to use seed {seed}, pass reseed=True to "
            "branch this checkpoint onto the new seed, or drop "
            "resume_from_run_id and start a fresh segment",
        )
    schema = schema_of(source.instrument)





    if schema is not None and schema["segmentation"]["archive"] != "progress":
        archive = schema["segmentation"]["archive"]




        if archive == "conclusion":
            held = "a finished conclusion rather than search state"
        else:
            held = "no archive at all"
        return None, _failure(
            "archive_not_resumable",
            f"{source.instrument} declares a {archive!r} archive: it holds "
            f"{held}, so a segment cannot continue from it and this search "
            "cannot be extended. Run it again with different parameters, or "
            "switch to an algorithm that checkpoints progress",
        )




    try:
        selected = kd.resolve_checkpoint(
            RunDirPaths(root=session.workspace / source.run_dir).checkpoints, iteration
        )
    except kd.CheckpointManifestError as exc:





        return None, _failure(
            "resume_source_unusable",
            f"the checkpoint archive of run {run_id!r} cannot be read: {exc}",
        )
    except kd.CheckpointSelectionError as exc:


        return None, _failure("resume_iteration_unavailable", f"run {run_id!r}: {exc}")
    return (
        _ResolvedResume(
            parent_run_id=run_id,
            iteration=selected.iteration,
            checkpoint=selected.path,




            seed=source.seed if seed is None else seed,
        ),
        None,
    )


@dataclass(frozen=True)
class _ResolvedSurrogate:

    surrogate_id: str
    path: Path
    sha256: str
    key: str


def _resolve_surrogate(
    session: Session,
    surrogate_id: str,
    *,
    algorithm: str,
    dataset_id: str,
    schema: dict[str, Any] | None,
    requested: dict[str, Any],
) -> tuple[_ResolvedSurrogate | None, dict[str, Any] | None]:
    registry = session.surrogates
    try:
        entry = registry.entry(surrogate_id)
    except KeyError:
        known = list(registry.ids())
        return None, _failure(
            "unknown_surrogate",
            f"surrogate {surrogate_id!r} is not registered in this workspace; "
            + (
                f"the registered ones are {known}"
                if known
                else "nothing has been trained here yet"
            )
            + ". The id comes from primary.surrogate_id in what train_surrogate "
            "returned",
        )
    if schema is not None and not schema["config_artifact_keys"]:
        return None, _failure(
            "surrogate_unsupported",
            f"{algorithm} takes no injected surrogate (its schema declares no "
            f"config_artifact_keys), so surrogate={surrogate_id!r} has nowhere "
            f"to go; drop surrogate, or search with {entry.recipe['algorithm']} "
            "which this surrogate was trained for",
        )
    if entry.recipe["algorithm"] != algorithm:
        return None, _failure(
            "surrogate_instrument_mismatch",
            f"surrogate {surrogate_id!r} was trained for "
            f"{entry.recipe['algorithm']} and this call asks for {algorithm}; a "
            "surrogate is that instrument's own network. Train one for "
            f"{algorithm} with train_surrogate, or set algorithm to "
            f"{entry.recipe['algorithm']}",
        )
    if entry.recipe["dataset_id"] != dataset_id:
        return None, _failure(
            "surrogate_dataset_mismatch",
            f"surrogate {surrogate_id!r} was trained on dataset "
            f"{entry.recipe['dataset_id']!r} and this call asks for "
            f"{dataset_id!r}; a surrogate fits one dataset's field. Train one "
            f"on {dataset_id!r} with train_surrogate",
        )
    path = registry.path(surrogate_id)
    if not path.is_file() or file_sha256(path) != entry.file_sha256:
        return None, _failure(
            "surrogate_file_missing",
            f"the file of surrogate {surrogate_id!r} ({entry.file}) is missing "
            "or no longer hashes to its registry row; run train_surrogate again "
            "with the same recipe to rebuild it, then pass the id again",
        )
    if schema is not None and any(
        mode["provider_kind"] != "autograd" for mode in schema["modes"]
    ):
        row = next(
            (row for row in schema["facade_params"] if row["name"] == "derivatives"),
            None,
        )
        effective = requested.get(
            "derivatives", None if row is None else row["default"]
        )
        if effective != "autograd":
            return None, _failure(
                "invalid_argument",
                f"{algorithm} consults an injected surrogate only under "
                f"derivatives=autograd, and this call runs derivatives="
                f'{effective!r}; pass params={{"derivatives": "autograd"}} '
                "(restating the rest of params) or drop surrogate",
            )
    return (
        _ResolvedSurrogate(
            surrogate_id=surrogate_id,
            path=path,
            sha256=entry.file_sha256,
            key=entry.key,
        ),
        None,
    )


def _sketch_failure(
    algorithm: str,
    sketch: dict[str, Any],
    schema: dict[str, Any] | None,
) -> dict[str, Any] | None:
    try:
        decoded = sketch_from_dict(sketch)
    except ValueError as exc:
        return _failure(
            "invalid_sketch",
            f"sketch is not a kd-sketch-v1 payload kd can decode: {exc}",
        )
    if schema is None:
        return None
    used = used_clauses(decoded)
    offending = [
        (mode["name"], clause, mode["sketch"][clause])
        for mode in schema["modes"]
        for clause in SKETCH_CLAUSES
        if clause in used
        if mode["sketch"][clause] == "unsupported"
    ]
    if not offending:
        return None
    details = ", ".join(
        f"(mode={mode!r}, clause={clause!r}, level={level!r})"
        for mode, clause, level in offending
    )
    return _failure(
        "sketch_unsupported",
        f"algorithm {algorithm!r} cannot accept this sketch: {details}; "
        "this algorithm declares no sketch support for these clauses; "
        "drop the listed clauses from sketch, or use an instrument whose "
        "modes support them",
    )


def baseline(
    ledger: SegmentLedger, parent_run_id: str
) -> tuple[str, SegmentSummary] | None:
    for entry in reversed(ledger.chain(parent_run_id)):
        if entry.summary is not None:
            return entry.run_id, entry.summary
    return None


def schema_of(algorithm: str) -> dict[str, Any] | None:
    return next(
        (item for item in kd.instrument_schemas() if item["algorithm"] == algorithm),
        None,
    )
