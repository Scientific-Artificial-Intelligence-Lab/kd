"""kd's public surface -> tool objects. Narrowing and shaping both happen here.

Narrowing = kd's three public surfaces add up to well over a hundred names, and
this layer admits sixteen verbs (`list_datasets`, `run_discovery`,
`train_surrogate`, `prune_node`, `select_node`, `submit_answer`,
`refine_candidate`, `smooth_input`, `ablate_candidate`, `score_term_sets`,
`verify_candidate`, `verify_system`, `vote_terms`, `make_sketch`,
`set_sketch`, `finish_without_answer`).

Beside the envelopes, three kinds of record a later call can cite live in the
`Session`: refinement receipts (`refinement_id`, bound by `submit_answer` on
exact equality only), evidence records (`evidence_id`, from votes and
measurements, resolved by `make_sketch`) and registered Sketches
(`sketch_id`, built by kd from cited anchors, bound from the researcher's
terms by `set_sketch`, or the user's, each arrival of a payload one more
registration of it). Each is written
once; the copies under the workspace (`refinements.jsonl`, `evidence.jsonl`)
are for reporting and are never read back as authority.
Shaping = every tool returns the same envelope, with fixed field names:

    status success | partial | failure
    primary display (the human-readable expression) + law (structured
                 terms, coefficients and the LHS the record names) + equation
                 (that law term by term on one line with signed coefficients,
                 present only when its LHS and every coefficient are known)
    diagnostics quantities the fit produced
    provenance which dataset, which instrument, which parameters
    failures fixable failures, with the fix
    budget what is left of this run (`Session.budget()`)

During the skeleton phase the envelope is a convention, not a type.

`budget` is the one field **unrelated to this call**: the first five describe
what just happened, and it describes what is left of the run as a whole.
`diagnostics.search_seconds` reports how long this one call took afterwards; do
not confuse the two. `envelope()` makes it a required keyword-only argument,
because the consequence of missing one is that the model cannot see the budget
on that return while nothing reports it -- one field silently absent, and only
mypy catches it.

Every `run_discovery` call leaves a sealed run directory under
`<workspace>/runs/` and appends one catalog row to `runs/catalog.jsonl`. Both
layers are kd's own recording formats (`kd-rundir-v1` / `kd-runcat-v1`), not
inventions of ours; the query side is the model's `execute` (DuckDB, or reading
the JSONL directly). Why not `EvidenceStore`: it requires the plan to be fixed
up front and cannot be reopened for appends, which does not fit an exploratory
session.

`run_discovery`, `submit_answer` and the other verbs that move the answer
(`prune_node`, `select_node`, `verify_system`) return a `Command` on success, writing the
answer channel's keys straight into graph state (`Command` is the only way a
tool writes state in this framework; `ToolRuntime` grants reads only). The
failure paths keep returning envelope dicts: there is no state to write, and
ToolNode supports both return kinds in one turn. `InjectedToolCallId` and
`Command` must be module-level imports -- this file has
`from __future__ import annotations`, so the closures' annotations are strings
and pydantic resolves them against the module dict with `get_type_hints` when
the tools are built, where a name imported inside a function body is invisible.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import textwrap
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final, Literal, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command

import kd
from kd import (
    extension_source,
    get_dataset,
    instrument_schemas,
    list_datasets_answer_blind,
    preview_report,
)
from kd.models import DLGA_NN_1, FieldRecipe

from kd.harness import SKETCH_SIDECAR_FILENAME, PlanEntry
from kd.search import (
    CATALOG_FILENAME,
    DEFAULT_RUNS_ROOT,
    RunDirPaths,
    RunRecord,
    new_run_id,
)
from kdagent import action_memory, action_memory_text
from kdagent import criteria, experience, priors, surrogate_training
from kdagent.episode_worker import (
    ABLATE_TASK,
    FORWARD_WORKER_SECONDS,
    NETWORK_TASK,
    STORED_SURROGATE_KEY,
    SCORE_TASK,
    SELECT_LOWPASS_TASK,
    SKETCH_TASK,
    SYSTEM_TASK,
    TASK_KEY,
    VERIFY_TASK,
    IsolatedEpisode,
    IsolatedRefine,
    IsolatedTask,
    finite_tree,
    frame_request,
    measure_selection_forward,
    prepare_file,
    run_isolated_episode,
    run_isolated_pinn,
    run_isolated_refine,
    run_isolated_smooth,
    run_isolated_task,
)
from kdagent.cards import DEFAULT_GENERATIONS, DEFAULT_SEED
from kdagent.config import LLM4ED_LIMITS, Llm4edConfig
from kdagent.prompt_text import (
    LLM4ED_OFF,
    combined_name_refusal,
    registered_algorithms,
)
from kdagent.lineage import (
    PLATFORM_FRAME,
    SYSTEM_FRAME,
    SegmentEntry,
    SegmentLedger,
    SegmentSummary,
    finite_or_none,
    sketch_identity,
    standing_verification,
    system_identity,
)
from kdagent.resume import (
    _ResolvedResume,
    _ResolvedSurrogate,
    baseline,
    _checkpoint_control_failure,
    _crashed,
    _failure,
    _lineage_provenance,
    _record_run,
    _record_segment,
    segment_cost,
    _reseed_control_failure,
    _resolve_resume,
    _resolve_surrogate,
    _sketch_failure,
    schema_of,
    sketch_lineage_refusal,
)
from kdagent.segment_report import (
    build_segment_report,
    law_from_evidence,
    sealed_equation_line,
)
from kdagent.session import Session, effective_training_dtype
from kdagent.data_source import (
    derivation_chain,
    derived_ref,
    field_ref,
    input_record,
    input_ref,
    is_derived,
    is_file_ref,
    is_pinn_derived,
    is_table,
    pinn_ref,
    projection_recipe,
    raw_fingerprint,
    resolve_input,
    root_ref,
    save_input,
    sealed_input_failure,
    spell_modes,
)
from kdagent.state import ANSWER, Answer
from kdagent.surrogates import (
    NETWORK_ALGORITHM,
    SurrogateEntry,
    file_sha256,
    stopped_at_deadline,
    surrogate_id_of,
    trained_surrogate_id,
)


def _law_shape_failure(law: dict[str, Any]) -> dict[str, Any] | None:
    """Say what is wrong when `law` is not the shape `run_discovery` hands out;
    `None` when it is.

    The shape is defined by `run_discovery`'s `success` envelope: `support` is a
    non-empty list of term names; `lhs` (the sealed record's LHS, or null) is
    not judged here but against the search it names (`_lhs_failure`);
    `coefficients` is either absent (kd drops it entirely when coefficients and
    support do not line up, with `is_valid` still true --
    `record_assembly.active_support_and_coefficients`) or the same length as
    `support`, each position a finite number or null. This is tighter
    than kd's `EvidenceRecord` in exactly one place: kd allows `support` to be
    None or empty and this does not -- but `_outcome_envelope` reports such a
    record as `partial`, so "a law handed out under `success` always passes
    here" holds by construction. One notch tighter would not: the model would
    receive a failure it cannot fix.

    The signature is `dict[str, Any]`, so pydantic guarantees only that it is a
    dict; the shape of the values is visible only here. In exp-08's first wave
    the kdv run submitted `{equation, terms, lhs, form}`, the channel took it,
    and that is what stood in the `answer` column -- an answer submitted whose
    structure cannot be judged. This matters more since the terminating
    middleware landed: however wrong the submission, the run is over and there is
    no next turn to fix it in.
    """
    support = law.get("support")
    if not (
        isinstance(support, list)
        and support
        and all(isinstance(term, str) for term in support)
    ):
        return _failure(
            "unstructured_answer",
            "law.support has to be a non-empty list of term names, the one "
            "under primary.law when run_discovery succeeds; this law does not "
            "have it (or it is empty). An empty support means that search "
            "produced no structured result, so run another search before "
            "submitting",
        )
    coefficients = law.get("coefficients")
    if coefficients is None:
        return None
    aligned = isinstance(coefficients, list) and len(coefficients) == len(support)
    if not aligned or not all(
        value is None
        or (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
        for value in coefficients
    ):
        return _failure(
            "unstructured_answer",
            f"law.coefficients has to be the same length as support "
            f"({len(support)} entries), each position a finite number or null; "
            "passing run_discovery's primary.law through unchanged is enough",
        )
    return None


def _lhs_failure(
    law: dict[str, Any], sealed: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Say what is wrong when `law` does not carry `sealed`, the left-hand side
    of the search it is submitted on; `None` when it does.

    The left-hand side is a fact of the sealed search, not a choice of the
    model's: `law_from_evidence` puts it on every law (null for a homogeneous
    form, an empty axis and order 0 for a regression), and whoever reads the
    answer takes the law's form from it. A submission that drops the key or
    retypes the value (`axis: null` for a table's empty axis, the target's
    name as a string) ends the run with an answer whose form cannot be read,
    and there is no next turn to fix it in.
    """
    if "lhs" in law and law["lhs"] == sealed:
        return None
    given = json.dumps(law["lhs"]) if "lhs" in law else "no lhs key"
    return _failure(
        "unstructured_answer",
        f"law.lhs has to be the left-hand side that search sealed, "
        f"{json.dumps(sealed)}; this law has {given}. Pass primary.law.lhs "
        "through unchanged",
    )


def _entry_provenance(entry: SegmentEntry) -> dict[str, Any]:
    """The five keys every answer carries, read off a ledger row.

    The same five the submitted path publishes (`Answer.provenance`), computed
    from the tree rather than read back out of the run directory.
    """
    return {
        "instrument": entry.instrument,
        "dataset": entry.dataset,
        "run_id": entry.run_id,
        "run_dir": entry.run_dir,
        "nmse": entry.nmse,
    }


def _node_refusal(
    session: Session, run_id: str, *, action: str
) -> dict[str, Any] | None:
    """Why this run_id cannot be acted on -- the two refusals `prune_node` and
    `select_node` share, in their order; None when the node can be.

    The order is the contract, so it is written once: a node this workspace's
    tree never recorded is `unknown_run_id`, and one that is already abandoned
    -- itself or through an ancestor -- is `pruned_run_id`. Both verbs ask the
    tree rather than `Session.successful_runs`: a branch an earlier run left on
    this workspace is a legitimate target, and it never succeeded this run.
    """
    recorded = [entry.run_id for entry in session.ledger.segments()]
    if run_id not in recorded:
        return _failure(
            "unknown_run_id",
            f"run_id {run_id!r} is not a search on this workspace's policy "
            + (
                f"tree, the recorded ones are {recorded}"
                if recorded
                else "tree, which has no searches on it yet; run run_discovery first"
            )
            + "; the value comes from provenance.run_id in what run_discovery "
            "returned",
        )
    if session.ledger.is_pruned(run_id):
        return _failure(
            "pruned_run_id",
            f"run {run_id!r} was already abandoned (itself or an ancestor of "
            f"it) and cannot be {action}; the tree records the reason it was "
            "given up on",
        )
    return None


_Selected = tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]
"""Law, provenance, the default rule's decision and the answer's system."""


def _selected_answer(session: Session, dataset_id: str) -> _Selected:
    """The law and provenance of the node the policy holds right now, the
    default rule's decision when that rule chose it (None when an explicit
    selection is in force), and the ODE system the law was chosen as a
    member of (None when no system chose it).

    The answer channel is written on the **write** side with whichever node
    would be selected at this moment, so that exploring a worse branch cannot
    make that worse branch the run's public result. Teardown then only labels
    the channel, and all five termination paths are covered without any of them
    knowing about this rule.

    An explicit `select` wins over the default rule, except on the two facts
    that make a node unpublishable as this run's answer. A row whose `law` is
    None is a partial, and publishing it would put a malformed answer in the
    channel. A row on **another dataset** is the very thing the dataset axis of
    `default_selection` exists to keep out: one workspace's tree outlives a
    single `run()`, so the selected node may have been named out of band
    (`python -m kdagent.tree select ...`) or by an earlier run that searched
    something else, and publishing that law here would put another dataset's
    equation, and its provenance, under this run's question. Either way the
    selection falls through to the default rule, whose domain excludes both by
    construction.

    "Another dataset" means another input: a derived copy of the same input
    (`smooth_input`, a PINN field; `_same_input`) is a version of it, not
    another dataset, so an explicit selection on one version stays in force
    when a search on, or a prune of, a sibling version recomputes the channel
    (review 2026-10-20). The default rule's domain stays the one version
    `dataset_id` names, because scores are in that input's frame.

    A search on a derived input with no launch record from this run
    (`_unbound_derived_runs`: an earlier run's node on a reopened workspace)
    is never published: an explicit selection of one falls through to the
    default rule, and the default rule's domain leaves them out (review
    2026-10-20).

    Law and provenance are None when the default rule has no winner
    (`criteria.selection_decision`: an empty domain, several targets, or no
    forward value across instruments); the caller then writes that empty
    selection into the channel, so an earlier automatic answer does not stay
    behind in it, and never falls back to the search it just ran.

    On an input that projects one state of raw ODE trajectories
    (`_raw_target`) the systems of those trajectories come first (default
    answer rule (b) of the ODE system design): `select_node(system_id=...)`
    is an explicit selection of the system's member for the input's target,
    and the default rule ranks the systems (`criteria.system_decision`) as
    soon as one has a current result, publishing the winner's member or,
    when none has a finite error, nothing (`no_forward_value`). With no such
    system the single-target rule answers, and its decision says so
    (`system_not_assembled`).
    """
    raw = _raw_target(session, dataset_id)
    explicit = _explicit_answer(session, dataset_id, raw)
    if explicit is not None:
        return explicit
    return _default_answer(session, dataset_id, raw)


def _explicit_answer(
    session: Session, dataset_id: str, raw: tuple[str, str] | None
) -> _Selected | None:
    """The answer the explicit selection holds for `dataset_id`, or None
    when none applies: a selected system of the input's raw trajectories
    publishes its member for the input's target; a selected run applies
    unless it is a partial, on another input, or an unbound derived one
    (`_selected_answer`)."""
    ledger = session.ledger
    system_id = ledger.selected_system
    if system_id is not None and raw is not None:
        system = ledger.system(system_id)
        if system["raw_fingerprint"] == raw[0]:
            entry = ledger.entry(system["members"][raw[1]])
            return (
                entry.law,
                _entry_provenance(entry),
                None,
                _answer_system(ledger, system_id),
            )
    selected = ledger.selected
    explicit = None if selected is None else ledger.entry(selected)
    if (
        explicit is None
        or explicit.law is None
        or not _same_input(explicit.dataset, dataset_id)
        or explicit.run_id in _unbound_derived_runs(session)
    ):
        return None
    return explicit.law, _entry_provenance(explicit), None, None


def _default_answer(
    session: Session, dataset_id: str, raw: tuple[str, str] | None
) -> _Selected:
    """The default rule's answer for `dataset_id`: the system rule first on
    raw ODE trajectories, else the single-target rule (`_selected_answer`)."""
    ledger = session.ledger
    if raw is not None:
        decision = criteria.system_decision(
            ledger, raw_fingerprint=raw[0], target=raw[1]
        )
        if decision is not None:
            if decision["run_id"] is None:
                return None, None, decision, None
            entry = ledger.entry(decision["run_id"])
            system = _answer_system(ledger, decision["system_id"])
            return entry.law, _entry_provenance(entry), decision, system
    decision = criteria.selection_decision(
        ledger, dataset_id=dataset_id, exclude=_unbound_derived_runs(session)
    )
    if decision["run_id"] is None:
        return None, None, decision, None
    if raw is not None:
        decision = {**decision, "reason": "system_not_assembled"}
    entry = ledger.entry(decision["run_id"])
    return entry.law, _entry_provenance(entry), decision, None


def _channel_answer(
    law: dict[str, Any] | None,
    provenance: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    system: dict[str, Any] | None,
) -> Answer:
    """The `selected` answer a search, a prune or a system verification
    writes into the channel: the node `_selected_answer` holds, with the
    default rule's decision when that rule chose it (`selection`) and the
    system it was chosen as a member of (`system`), or the rule's empty
    choice (law and provenance None), which is not terminal and outranks
    nothing."""
    answer = Answer(law=law, source="selected", provenance=provenance)
    if decision is not None:
        answer["selection"] = decision
    if system is not None:
        answer["system"] = system
    return answer


def _listed(description: dict[str, Any]) -> dict[str, Any]:
    """A workspace input's `list_datasets` row: its data summary under the
    verified reference (the descriptor's `ref`, checked against its file
    name and its content or derivation), and for a derived input the
    verified parent as `derived_from`; the summary's own `id` and
    `derived_from` are not checked by verification (review 2026-10-20)."""
    row = {**description["summary"], "id": description["ref"]}
    if "derived" in description:
        row["derived_from"] = description["derived"]["from"]
    return row


def _unbound_derived_runs(session: Session) -> set[str]:
    """The ledger's searches on a derived input that have no launch record
    from this run (`Session.search_inputs`): their input cannot be bound to
    what they ran on, so they are neither selectable nor published."""
    return {
        entry.run_id
        for entry in session.ledger.segments()
        if entry.dataset
        and is_derived(entry.dataset)
        and entry.run_id not in session.search_inputs
    }


def _unbound_search_refusal(run_id: str, dataset_id: str) -> dict[str, Any]:
    return _failure(
        "input_unverified",
        f"search {run_id!r} has no record from this run of the input it ran "
        f"on, so the derived input {dataset_id} cannot be bound to it; search "
        "that input again in this run",
    )


def _segment_report(
    outcome: Any,
    session: Session,
    lineage: dict[str, Any],
) -> tuple[dict[str, Any] | None, SegmentSummary | None, list[dict[str, Any]]]:
    """The report is a side channel: report it when it computes badly, but never
    lose the sealed scientific result with it.

    The baseline comes from this run's ledger (`_resolve_resume` has already
    confirmed the parent id is on it), and when the parent has no baseline of
    its own the lineage is walked upward (`baseline`); the report's
    `improvement.baseline_run_id` names which segment was actually used.
    Reaching the root without a baseline is handled as "no parent": this segment
    still has its own quantities, and only the two parent-relative numbers and
    the structure diff have nothing to stand on, so they are omitted.
    """
    if outcome.run_dir is None:
        return None, None, []
    parent_run_id = lineage["parent_run_id"]
    try:
        inherited = (
            None if parent_run_id is None else baseline(session.ledger, parent_run_id)
        )
        baseline_run_id, parent = (None, None) if inherited is None else inherited
        report, summary = build_segment_report(
            Path(outcome.run_dir), parent, baseline_run_id=baseline_run_id
        )
    except Exception as exc:
        return None, None, [_crashed(exc)]
    return report, summary, []


def _search_forward(
    session: Session,
    result: dict[str, Any],
    *,
    run_id: str | None,
    run_dir: Path | None,
    dataset_id: str,
    input_source: dict[str, Any] | None,
    launched_on: dict[str, Any],
) -> dict[str, Any] | None:
    """The forward measurement this search's tree row carries, also put
    under `diagnostics.selection_forward`; None when the search left no row.

    A success is measured once on its own sealed law and the input it ran on
    (`episode_worker.measure_selection_forward`), under what the run has
    left at this moment, and the measurement names the hash of the record
    sealed in its run directory; any other outcome has no law (`no_law`).
    The search sealed before its measurement starts, so a measurement that
    raises on the parent side is reported in `failures` and costs the row
    its measurement, never the sealed result.
    """
    if run_id is None:
        return None
    law = result["primary"]["law"] if result["status"] == "success" else None
    try:
        record_hash = (
            None
            if law is None or run_dir is None
            else RunRecord.load(RunDirPaths(root=run_dir).record).record_hash
        )
        measured = measure_selection_forward(
            run_id=run_id,
            law=law,
            record_hash=record_hash,
            dataset_id=dataset_id,
            input_source=input_source,
            input_record=launched_on,
            remaining_seconds=session.budget()["remaining_seconds"],
        )
    except Exception as exc:
        result["failures"].append(_crashed(exc))
        return None
    result["diagnostics"]["selection_forward"] = measured
    return measured


def _equation_line(
    law: dict[str, Any], catalog_fit: dict[str, Any] | None
) -> tuple[str | None, list[dict[str, Any]]]:
    """`primary.equation` is a side channel like the segment report: a law kd's
    parser cannot render is reported in `failures`, and the sealed law is still
    handed out and recorded.

    `_outcome_envelope` runs inside the same `try` as the search in
    `run_discovery`, so an exception escaping here would turn a sealed valid
    result into a crashed envelope, returned before the ledger row and the
    catalog row are written and without registering the run for an answer.
    The detail names the field, because the envelope's status stays `success`.
    """
    try:
        return sealed_equation_line(law, catalog_fit), []
    except Exception as exc:
        return None, [
            _failure(type(exc).__name__, f"primary.equation was not rendered: {exc}")
        ]


def _outcome_envelope(
    outcome: Any,
    params: dict[str, Any],
    session: Session,
    *,
    lineage: dict[str, Any],
    time_cap_seconds: float | None,
    stopped_by_time_cap: bool,
    post_seal_error: str | None,
    evaluation_cap: int | None,
    stopped_by_evaluation_cap: bool,
    platform_nmse: float | None,
    platform_coefficients: list[float] | None,
    platform_eval_error: str | None,
    platform_eval_dropped: int | None,
    unavailable_reason: str | None,
    sketch: dict[str, Any] | None,
    surrogate: str | None,
) -> tuple[dict[str, Any], SegmentSummary | None]:
    """`EpisodeOutcome` -> (return envelope, baseline for child segments).

    Three outcomes are reported apart, because they mean three different things
    to the model: it crashed (retry with different parameters), it finished but
    sealed no record (an anomaly, not something the model can fix), or it
    finished with an invalid final evaluation (the result cannot be submitted as
    an answer, and `best_expression` may by then have degraded into a genotype).

    The second return value is a `SegmentSummary`: the baseline the segment
    report computes on the side, stored on the ledger by the caller and read
    back when the next segment resumes from this run. It is None when there is
    no sealed record, or when the report did not compute.

    It takes a `Session` rather than a `workspace` because it needs two things
    (the workspace root and the budget), and the budget has to be read **after**
    the search finishes -- the search is the most time-consuming action of the
    run, and a snapshot taken on entry would erase the tens of minutes just
    spent.

    `platform_coefficients` is the neutral re-fit's coefficient vector, aligned
    with the support, published under `diagnostics` beside `platform_nmse`;
    `primary.law.coefficients` stays the instrument's own, and
    `provenance.headline_coefficient_source` says so.

    `sketch` and `surrogate` are the ones the segment effectively ran under
    (inherited on a resume) and are echoed in `provenance`. The sketch verdict
    goes out as `diagnostics.sketch`, projected from the `sketch.json` sidecar
    kd writes at the run-dir root -- the parent reads the file rather than the
    outcome object, since only JSON crosses the subprocess boundary -- and only
    the verdict and scalar certification are projected (`satisfied`,
    `certified` plus the per-clause results), never the sidecar's `solution` /
    `best_candidate` / `full_verify`. When kd sealed
    the record but the sidecar write itself failed, the envelope is still the
    completed record, `diagnostics.sketch` is absent and
    `diagnostics.post_seal_error` carries the failure.
    """
    budget = session.budget()
    entry = outcome.entry
    provenance: dict[str, Any] = {
        "instrument": entry.instrument,
        "dataset": entry.dataset_ref,
        "seed": entry.seed,
        "params": params,
        "sketch": sketch,
        "surrogate": surrogate,
        "lineage": lineage,
        "wallclock_seconds": round(outcome.wallclock_seconds, 3),
        "run_id": outcome.run_id,



        "run_dir": (
            None
            if outcome.run_dir is None
            else os.path.relpath(outcome.run_dir, session.workspace)
        ),
    }
    if outcome.status == "raised":
        return envelope(
            "failure",
            budget=budget,
            provenance=provenance,
            failures=[_failure(str(outcome.error_type), str(outcome.error_message))],
        ), None
    if outcome.record is None:
        return envelope(
            "failure",
            budget=budget,
            provenance=provenance,
            failures=[
                _failure(
                    "no_record",
                    "the search finished but sealed no result record",
                )
            ],
        ), None





    provenance["params"]["effective"] = deepcopy(outcome.record.run_spec.config)
    evidence = outcome.record.evidence
    cost = outcome.record.cost
    report, summary, report_failures = _segment_report(outcome, session, lineage)
    diagnostics: dict[str, Any] = {
        "score": evidence.score,
        "score_kind": evidence.score_kind,
        "score_direction": evidence.score_direction,
        "nmse": evidence.nmse,
        "platform_nmse": platform_nmse,


        "term_count": None if evidence.support is None else len(evidence.support),
        "platform_frame": PLATFORM_FRAME,
        "platform_coefficients": platform_coefficients,
        "platform_eval_dropped": platform_eval_dropped,
        "unavailable_reason": unavailable_reason,
        "mse": evidence.mse,
        "complexity": evidence.complexity,
        "search_seconds": round(cost.search_seconds, 3),



        "time_cap_seconds": time_cap_seconds,
        "stopped_by_time_cap": stopped_by_time_cap,







        "evaluations": cost.boundary_results,
        "invalid_evaluations": cost.boundary_invalid_results,
        "evaluation_cap": evaluation_cap,
        "stopped_by_evaluation_cap": stopped_by_evaluation_cap,
    }
    if report is not None:
        diagnostics["segment_report"] = report
    verdict = (
        None if outcome.run_dir is None else _sketch_verdict(Path(outcome.run_dir))
    )
    if verdict is not None:
        diagnostics["sketch"] = verdict
    if post_seal_error is not None:
        diagnostics["post_seal_error"] = post_seal_error
    if platform_eval_error is not None:
        diagnostics["platform_eval_error"] = platform_eval_error
    provenance["headline_coefficient_source"] = evidence.headline_coefficient_source
    if not evidence.is_valid:




        return envelope(
            "partial",
            budget=budget,
            primary={"display": evidence.expression, "law": None},
            diagnostics=diagnostics,
            provenance=provenance,
            failures=[
                _failure(
                    "invalid_result",
                    str(evidence.invalid_reason or evidence.error_detail),
                ),
                *report_failures,
            ],
        ), summary
    if not evidence.support:








        return envelope(
            "partial",
            budget=budget,
            primary={"display": evidence.expression, "law": None},
            diagnostics=diagnostics,
            provenance=provenance,
            failures=[
                _failure(
                    "no_structured_result",
                    "the search finished and the evaluation was valid, but no "
                    "term was selected (support is empty); this is not a "
                    "submittable answer, so search again with different "
                    "parameters or a different instrument",
                ),
                *report_failures,
            ],
        ), summary
    if evidence.catalog_fit is None:
        return envelope(
            "partial",
            budget=budget,
            primary={"display": evidence.expression, "law": None},
            diagnostics=diagnostics,
            provenance=provenance,
            failures=[
                _failure(
                    "uncatalogued_law",
                    f"Expression {evidence.expression!r} has no catalog_fit: "
                    "kd's expression IR does not accept numeric constants inside "
                    "terms. Search with another expression or instrument to "
                    "obtain a submittable law",
                ),
                *report_failures,
            ],
        ), summary
    law = law_from_evidence(evidence)
    primary: dict[str, Any] = {
        "display": evidence.expression,




        "law": law,
    }





    equation, equation_failures = _equation_line(


        cast(dict[str, Any], law),
        evidence.catalog_fit,
    )
    if equation is not None:
        primary["equation"] = equation
    return envelope(
        "success",
        budget=budget,
        primary=primary,
        diagnostics=diagnostics,
        provenance=provenance,
        failures=[*report_failures, *equation_failures],
    ), summary


def envelope(
    status: str,
    *,
    budget: dict[str, Any],
    primary: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The one return envelope shared by every tool.

    `budget` is required and keyword-only: any other missing field comes out as
    an empty dict and the model can see that this call had no such thing, while
    a missing budget is silent -- on that one return the model simply does not
    know what is left, with nothing to hint at it. Requiring it turns a missed
    call site red in mypy on the spot.
    """
    return {
        "status": status,
        "primary": primary or {},
        "diagnostics": diagnostics or {},
        "provenance": provenance or {},
        "failures": failures or [],
        "budget": budget,
    }


def _sketch_verdict(run_dir: Path) -> dict[str, Any] | None:
    """The sketch verdict of a sealed run, projected from its `sketch.json`.

    None when the run has no sidecar (no sketch was given, or the fit reported
    no outcome). The projection is a named allowlist: `satisfied` (the
    verdict's overall bit), `certified` (`solution is not None`), the LHS
    match, the per-clause results (pinned /
    anchored / holes), the law terms outside every clause, kd's failure
    string when it produced no verdict, and the certificate
    (`_sketch_certificate`: the sealed candidate, its residual NMSE, the
    threshold and where it was measured). The sidecar's `solution` /
    `best_candidate` / `full_verify` objects are projected, not poured into
    the envelope by dumping the file; a certificate is about the search
    that sealed it, and no later refit inherits it.
    """
    path = run_dir / SKETCH_SIDECAR_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = payload["verdict"]
    certified = payload["solution"] is not None
    certificate = _sketch_certificate(payload)
    if verdict is None:
        return {
            "satisfied": False,
            "certified": certified,
            **certificate,
            "failure": payload["failure"],
        }
    return {
        "satisfied": verdict["overall"],
        "certified": certified,
        **certificate,
        "lhs_matched": verdict["lhs_matched"],
        "pinned": [
            {
                "term_ir": clause["term_ir"],
                "expected": clause["expected"],
                "observed": clause["observed"],
                "matched": clause["matched"],
            }
            for clause in verdict["pinned"]
        ],
        "anchored": [
            {
                "term_ir": clause["term_ir"],
                "observed": clause["observed"],
                "matched": clause["matched"],
            }
            for clause in verdict["anchored"]
        ],
        "holes": [
            {
                "hole_id": hole["hole_id"],
                "assigned": list(hole["assigned"]),
                "matched": hole["matched"],
            }
            for hole in verdict["holes"]
        ],
        "unassigned": [term["law_key"] for term in verdict["unassigned"]],
        "failure": payload["failure"],
    }


def _equation_law(equation: dict[str, Any]) -> dict[str, Any]:
    """The active law of a kd equation in its JSON form (`kd.core.equation`'s
    `to_dict`: the terms in kd's IR, each with its coefficient, and the
    active indices) as `{"support", "coefficients", "lhs"}`; a coefficient
    that is not a scalar is null."""
    active = equation["active_indices"]
    terms = (
        equation["terms"]
        if active is None
        else [equation["terms"][index] for index in active]
    )
    return {
        "support": [term for term, _ in terms],
        "coefficients": [
            value["value"] if value["kind"] == "Scalar" else None for _, value in terms
        ],
        "lhs": equation["lhs_spec"],
    }


def _sketch_certificate(payload: dict[str, Any]) -> dict[str, Any]:
    """The sealed candidate and the residual gate of a Sketch outcome: the
    candidate law kd judged (support, coefficients, left-hand side; null
    when it lifted none), the residual NMSE it measured, the threshold it
    applied and whether the candidate passed, and where it measured (the
    input's name and fingerprint, the sample count and the normalizing
    term). A later refit does not change any of these."""
    report = payload["full_verify"]
    candidate = payload["best_candidate"]
    return {
        "candidate": None if candidate is None else _equation_law(candidate),
        "residual_nmse": None if report is None else report["nmse"],
        "nmse_max": None if report is None else report["policy"]["nmse_max"],
        "verify_passed": None if report is None else report["passed"],
        "measured_on": (
            None
            if report is None
            else {
                key: report[key]
                for key in (
                    "dataset_name",
                    "dataset_fingerprint",
                    "n_samples",
                    "normalizer_term",
                )
            }
        ),
    }


REFINED_FROM = "platform_refit"
"""`provenance.refined_from` of a refined law: the coefficient-source word
`headline_coefficient_source` uses for coefficients the platform refitted."""

REFIT_TABLE_FRAME = "kd-refit-constants-full-table-v1"
"""The frame of a table refit's `nmse`: `kd.refit_constants` on every row."""

PINN_REFINED_FROM = "pinn_refit"
"""`provenance.refined_from` of a law whose coefficients a PINN fit trained."""

PINN_PRETRAIN_EPOCHS = 5000
"""`refine_candidate`'s default data-only epochs before a PINN fit's joint ones."""

PINN_SEED = 0
"""The seed of every PINN fit `refine_candidate` makes (`kd.refine_pinn`)."""


def _pinn_frame(epochs: int, pretrain_epochs: int, residual_weight: float) -> str:
    """The frame a PINN fit's coefficients are measured in: the network's
    autograd derivatives after these epochs at this residual weight, not a
    finite-difference refit. `v2` (2026-10-23): `kd.refine_pinn`'s
    pretraining runs its requested epochs up to 5000, where a `v1` fit with
    the same settings took its patience from the joint epochs and could stop
    after `max(50, 10 * (epochs + 1))`."""
    return f"kd-refine-pinn-e{epochs}-p{pretrain_epochs}-w{residual_weight:g}-v2"


def _refit_frame(
    savgol_window: int,
    lowpass_modes: int | dict[str, int],
    time_window: int,
    *,
    method: str = "finite_diff",
    trim: Trim = 0,
) -> str:
    """The frame tag of a grid refit's numbers.

    Unsmoothed, untrimmed finite differences give `PLATFORM_FRAME`:
    `kd.refit_terms` without pruning is the same fit as the search's platform
    re-evaluation, so the two numbers compare directly. A smoothed field is
    another target for every window or mode count, so the parameters are part
    of the tag; the two smoothers are never combined (`_refine_refusal`), so a
    tag names one of them. Another derivative method or a trimmed row set is
    another frame again, `kd-refit-terms-<method>[-trim<rows>][<smoothing>]-v1`.
    A refit measured on a derived input carries that input's frame. The tag
    is a label; the receipt and `diagnostics.refit_frame` hold kd's resolved
    `kd.RefitFrame`.
    """
    smoothing = ""
    if lowpass_modes:
        smoothing = f"-lowpass{spell_modes(lowpass_modes)}-t{time_window}"
    elif savgol_window != 0:
        smoothing = f"-savgol{savgol_window}"
    if method != "finite_diff" or trim != 0:
        rows = (
            ""
            if trim == 0
            else "-trim" + (spell_modes(trim) if isinstance(trim, dict) else str(trim))
        )
        return f"kd-refit-terms-{method}{rows}{smoothing}-v1"
    if smoothing:
        return f"kd-evaluate-terms-full-grid{smoothing}-v1"
    return PLATFORM_FRAME


SAVGOL_POLYORDER = 3
"""The polynomial order `kd.data.savgol_smooth` and the time pass of
`kd.data.lowpass_smooth` fit by default; a window has to be longer than it."""


def _window_failure(name: str, window: int) -> str | None:
    """Why `window` is not a Savitzky-Golay window this layer passes on: an
    even window is refused although scipy takes one, because it centres the
    fit halfway between two samples and shifts the field by half a step."""
    if window != 0 and (window <= SAVGOL_POLYORDER or window % 2 == 0):
        return (
            f"{name} has to be 0 (no smoothing) or an odd window longer than "
            f"{SAVGOL_POLYORDER}, got {window!r}"
        )
    return None


def _positive_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _lowpass_failure(
    lowpass_modes: int | dict[str, int], time_window: int, *, optional: bool
) -> dict[str, Any] | None:
    """Why these low-pass arguments cannot be passed to
    `kd.data.lowpass_smooth`; None when they can. With `optional`, the
    integer 0 means no low-pass (`refine_candidate`'s default) and is
    allowed. kd's own refusals (a count above the axis length, an axis the
    data lacks, a NaN field) come back from the worker; this repeats only
    what needs no data."""
    off = optional and lowpass_modes == 0 and not isinstance(lowpass_modes, bool)
    if isinstance(lowpass_modes, dict):
        well_formed = bool(lowpass_modes) and all(
            isinstance(axis, str) and _positive_count(count)
            for axis, count in lowpass_modes.items()
        )
    else:
        well_formed = off or _positive_count(lowpass_modes)
    if not well_formed:
        return _failure(
            "invalid_argument",
            "lowpass_modes has to be a positive integer"
            + (" (0 for no low-pass)" if optional else "")
            + f", or a mapping from axis name to a positive integer, got "
            f"{lowpass_modes!r}",
        )
    window = _window_failure("time_window", time_window)
    if window is not None:
        return _failure("invalid_argument", window)
    if time_window != 0 and off:
        return _failure(
            "invalid_argument",
            "time_window applies after the low-pass, so it needs lowpass_modes "
            f"as well; got time_window={time_window!r} with lowpass_modes 0",
        )
    return None


def _same_input(dataset_id: str, other: str) -> bool:
    """Whether two ids name one input or inputs derived from the same one:
    the pin and the selection domain are about the input, and a derived copy
    (`smooth_input`) is that input smoothed, not another dataset. Two ids
    whose chains start from the same root are the same input."""
    return root_ref(dataset_id) == root_ref(other)


def _same_raw_set(session: Session, dataset_id: str, pinned: str) -> bool:
    """Whether `dataset_id` projects another state of the raw trajectories
    the pinned input projects: both are file inputs whose verified
    descriptions record a projection and the same raw fingerprint (the file's
    bytes alone do not decide it: the same file read with its parameters in
    another order is another raw set). An id that names no verified
    description is not one.

    `run_discovery` accepts such an input on a pinned run. A search on it is
    a search of its own projection: `_same_input` stays false for it, so it
    is neither selected nor published for the pinned target, and a
    measurement of it stays on its own input (`_measured_input`).
    """
    if not (is_file_ref(dataset_id) and is_file_ref(pinned)) or is_derived(dataset_id):
        return False
    try:
        candidate = raw_fingerprint(_resolve(session, dataset_id))
        own = raw_fingerprint(_resolve(session, pinned))
    except (ValueError, KeyError, TypeError, OSError):
        return False
    return candidate is not None and candidate == own


def _projection_input(
    session: Session,
    dataset_id: str,
    target_state: str,
    time_cap_seconds: float | None,
) -> tuple[str, dict[str, Any] | None]:
    """The input that projects `target_state` of the raw trajectories
    `dataset_id` projects, or `dataset_id` with a refusal.

    The input's own target state is `dataset_id` itself. Another state's
    projection is the description of the same file under the same load
    options and the projection of that state; its reference is computed
    here, and when the workspace has no description under it the file is
    prepared again by a worker (`episode_worker.prepare_file`, capped like
    the search) and the description saved.
    """
    refusal = _failure(
        "invalid_argument",
        f"target_state chooses a state of raw trajectories, and {dataset_id!r} "
        "is not a projection of raw trajectories; omit target_state",
    )
    if not is_file_ref(dataset_id) or is_derived(dataset_id):
        return dataset_id, refusal
    description = cast(dict[str, Any], _resolve(session, dataset_id))
    if raw_fingerprint(description) is None:
        return dataset_id, refusal
    states = list(description["load_options"]["fields"])
    if target_state not in states:
        return dataset_id, _failure(
            "invalid_argument",
            f"target_state {target_state!r} is not a state of the trajectories "
            f"{dataset_id!r} projects; its states are {states}",
        )
    if target_state == description["projection"]["target_state"]:
        return dataset_id, None
    ref = input_ref({**description, "projection": projection_recipe(target_state)})
    if (session.workspace / "inputs" / f"{ref}.json").exists():
        return ref, None
    prepared = prepare_file(
        Path(description["path"]),
        description["load_options"],
        target_state=target_state,
        time_cap_seconds=time_cap_seconds,
    )
    save_input(session.workspace, prepared)
    return prepared["ref"], None


def _answer_version(session: Session) -> str | None:
    """The input version the answer channel is on: the input of the last
    search or selection that wrote it, else the pinned input (also when that
    pointer replayed an earlier run's selection on another dataset)."""
    pinned = session.pinned_dataset_id
    current = session.answer_dataset_id
    if current is not None and (pinned is None or _same_input(current, pinned)):
        return current
    return pinned


def _resolve(session: Session, dataset_id: str) -> dict[str, Any] | None:
    """`resolve_input` with the sha256 of every derived file this run wrote,
    so a derived input's file is checked against what the controller saw as
    well as against its own record."""
    return resolve_input(session.workspace, dataset_id, written=session.derived_inputs)


def _register_derived(session: Session, description: dict[str, Any]) -> dict[str, Any]:
    """Register a derived input the adapter's worker has just written and
    return its verified description.

    The id is registered with the sha256 the worker reported for the file it
    wrote (`description`, the worker's), not with a hash read again from the
    workspace, and then the whole chain is verified against that
    registration. An id is registered once: one already registered (two
    identical calls in one turn) is kept, and a different file under it does
    not verify. When verification fails the new registration is withdrawn and
    `InputVerificationError` propagates.
    """
    ref = description["ref"]
    fresh = ref not in session.derived_inputs
    if fresh:
        session.derived_inputs[ref] = description["source"]["sha256"]
    try:
        return cast(dict[str, Any], _resolve(session, ref))
    except Exception:
        if fresh:
            del session.derived_inputs[ref]
        raise


def _reusable(
    session: Session, dataset_id: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """What a derivation call does with the id it names: `(description, None)`
    to reuse a registered input that verifies, `(None, refusal)` for a
    registered input that does not (`input_unverified`; nothing is derived
    again and nothing overwrites a registered id), and `(None, None)` to
    derive it, when the id is not registered this run (a file an earlier run
    or anything else left under that name is overwritten by the derivation).
    """
    if dataset_id not in session.derived_inputs:
        return None, None
    refusal = _input_refusal(session, dataset_id)
    if refusal is not None:
        return None, refusal
    return cast(dict[str, Any], _resolve(session, dataset_id)), None


def _input_refusal(session: Session, dataset_id: str) -> dict[str, Any] | None:
    """Why a derived input cannot be used as data or published, or None.

    A derived id (any id with a `+` segment) is used only when this run
    derived and registered it (`Session.derived_inputs`); the whole chain is
    then checked down to the root (`data_source.derivation_chain`): its
    descriptor can be rewritten through the file tools while its id still
    names the pinned input, so the pin test on the id (`_same_input`) is not
    enough. Inputs that are not derived are left to the existing checks."""
    if not is_derived(dataset_id):
        return None
    try:
        derivation_chain(session.workspace, dataset_id, written=session.derived_inputs)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return _failure(
            "input_unverified",
            f"derived input {dataset_id!r} does not verify against the input it "
            f"names, so it is refused: {exc}",
        )
    return None


def _search_input_refusal(session: Session, run_id: str) -> dict[str, Any] | None:
    """Why the input a search of this run ran on can no longer be used, or
    None: the derived-input checks of `_input_refusal`, then the comparison
    of the workspace with what the search was launched on
    (`Session.search_inputs`, `data_source.sealed_input_failure`)."""
    dataset_id = session.successful_runs[run_id]["dataset"]
    refusal = _input_refusal(session, dataset_id)
    if refusal is not None:
        return refusal
    record = session.search_inputs.get(run_id)
    if record is None:
        return None
    failure = sealed_input_failure(
        session.workspace, record, written=session.derived_inputs
    )
    if failure is None:
        return None
    return _failure(
        "input_unverified",
        f"the input {dataset_id!r} that search {run_id!r} ran on is not what the "
        f"workspace now holds, so it is refused: {failure}",
    )


def _off_pin_refusal(
    session: Session, run_id: str, dataset_id: str
) -> dict[str, Any] | None:
    """Why search `run_id`, which ran on `dataset_id`, cannot be this run's
    answer, or None: the run is pinned and the search ran on another input,
    which on a pinned run is the projection of another state of the raw
    trajectories (`_same_raw_set`). `select_node` and `submit_answer` both
    refuse it, so a pinned run publishes only a law of its own target."""
    pinned = session.pinned_dataset_id
    if pinned is None or _same_input(dataset_id, pinned):
        return None
    return _failure(
        "dataset_not_pinned",
        f"run {run_id!r} searched dataset {dataset_id!r} while this run is "
        f"pinned to {pinned!r}; publishing it would answer this run's question "
        "with another dataset's equation",
    )


def _unknown_run_failure(session: Session, run_id: str) -> dict[str, Any]:
    """The refusal for a `run_id` that is not a search that succeeded this
    run, naming the ones that did; `submit_answer` and `refine_candidate`
    take only those."""
    eligible = sorted(session.successful_runs)
    return _failure(
        "unknown_run_id",
        f"run_id {run_id!r} is not a search that succeeded this run, "
        + (
            f"the usable ones are {eligible}"
            if eligible
            else "no search has succeeded this run yet, so run run_discovery first"
        )
        + "; the value comes from provenance.run_id in what run_discovery "
        "returned",
    )


def _refine_refusal(
    session: Session,
    run_id: str,
    *,
    law: dict[str, Any] | None,
    terms: list[str] | None,
    small_term_rtol: float,
    savgol_window: int,
    lowpass_modes: int | dict[str, int],
    time_window: int,
    pinn_epochs: int,
    pinn_pretrain_epochs: int,
    pinn_residual_weight: float,
) -> dict[str, Any] | None:
    """Why `refine_candidate` cannot run on these arguments; None when it can.

    The refined law is meant for `submit_answer` on the same `run_id`, so the
    source has to be one `submit_answer` accepts, in its order (pruned, then
    unknown), and a given `law` is held to the shape and the left-hand side
    `submit_answer` checks. The argument checks repeat kd's and scipy's rules
    so that a wrong argument costs no worker start and no data load
    (`_window_failure` says why an even window is refused). The two smoothers
    are not combined: each is one frame tag, and a field smoothed twice
    would be measured in neither. A PINN fit replaces the least-squares
    refit, so it takes none of the refit's smoothing or pruning, and it
    starts from the candidate's coefficients, so it takes no bare `terms`.
    """
    if session.ledger.is_pruned(run_id):
        return _failure(
            "pruned_run_id",
            f"run {run_id!r} was pruned from the policy tree (itself or an "
            "ancestor of it), and a law refined from it cannot be submitted",
        )
    if run_id not in session.successful_runs:
        return _unknown_run_failure(session, run_id)
    if law is not None and terms is not None:
        return _failure(
            "invalid_argument",
            "pass either law or terms, not both: terms replaces the whole "
            "candidate, coefficients included",
        )
    if law is not None:
        shape = _law_shape_failure(law) or _lhs_failure(law, session.sealed_lhs[run_id])
        if shape is not None:
            return shape
    window = _window_failure("savgol_window", savgol_window)
    checks = (
        (
            not 0.0 <= small_term_rtol < 1.0,
            f"small_term_rtol has to be in [0, 1), got {small_term_rtol!r}",
        ),
        (terms is not None and not terms, "terms has to name at least one term"),
        (window is not None, str(window)),
    )
    for failed, detail in checks:
        if failed:
            return _failure("invalid_argument", detail)
    lowpass = _lowpass_failure(lowpass_modes, time_window, optional=True)
    if lowpass is not None:
        return lowpass
    if savgol_window != 0 and lowpass_modes:
        return _failure(
            "invalid_argument",
            "savgol_window and lowpass_modes are two smoothers measured in "
            "two frames; pass one of them and 0 for the other",
        )
    return _pinn_refusal(
        pinn_epochs,
        pinn_pretrain_epochs,
        pinn_residual_weight,
        bare_terms=terms is not None,
        others=small_term_rtol != 0 or savgol_window != 0 or bool(lowpass_modes),
    )


def _pinn_refusal(
    pinn_epochs: int,
    pinn_pretrain_epochs: int,
    pinn_residual_weight: float,
    *,
    bare_terms: bool,
    others: bool,
) -> dict[str, Any] | None:
    """Why the PINN arguments cannot be passed on; None when they can."""
    if (
        isinstance(pinn_residual_weight, bool)
        or not isinstance(pinn_residual_weight, int | float)
        or not 0.0 <= pinn_residual_weight < math.inf
    ):
        return _failure(
            "invalid_argument",
            "pinn_residual_weight has to be a finite number >= 0, got "
            f"{pinn_residual_weight!r}",
        )
    if not (isinstance(pinn_epochs, int) and pinn_epochs >= 0) or isinstance(
        pinn_epochs, bool
    ):
        return _failure(
            "invalid_argument",
            f"pinn_epochs has to be 0 or a positive integer, got {pinn_epochs!r}",
        )
    if not (
        isinstance(pinn_pretrain_epochs, int) and pinn_pretrain_epochs >= 0
    ) or isinstance(pinn_pretrain_epochs, bool):
        return _failure(
            "invalid_argument",
            "pinn_pretrain_epochs has to be 0 or a positive integer, got "
            f"{pinn_pretrain_epochs!r}",
        )
    if pinn_epochs == 0:
        return None
    if bare_terms:
        return _failure(
            "invalid_argument",
            "a PINN fit starts from the candidate's coefficients, and terms "
            "carry none; pass a law with coefficients instead",
        )
    if others:
        return _failure(
            "invalid_argument",
            "pinn_epochs replaces the least-squares refit, so small_term_rtol, "
            "savgol_window, lowpass_modes and time_window must be 0 with it; to "
            "drop a term, pass a law without it",
        )
    return None


def _refine_input_refusal(
    *,
    table: bool,
    catalog: bool,
    lhs: dict[str, Any] | None,
    coefficients: list[float | None] | None,
    small_term_rtol: float,
    savgol_window: int,
    lowpass_modes: int | dict[str, int],
    pinn_epochs: int,
    framed: bool = False,
) -> dict[str, Any] | None:
    """The refusals that need the input and the candidate resolved: the grid
    parameters on a table (`framed`: a method other than finite differences,
    or a trim), a grid law with no time-derivative left-hand side (the
    `unsupported_form` of `platform_evaluation`), pruning with no
    coefficients to prune by, and a PINN fit that cannot run or whose field
    cannot become a derived input. None when the refit can be asked for."""
    if table and (
        small_term_rtol != 0
        or savgol_window != 0
        or lowpass_modes
        or pinn_epochs
        or framed
    ):
        return _failure(
            "invalid_argument",
            "a table refits the constants inside its terms; small_term_rtol, "
            "savgol_window, lowpass_modes, time_window, pinn_epochs, method, "
            "trim, weak and surrogate_id apply to grid data, so leave them at "
            "their defaults",
        )
    if pinn_epochs:
        pinn = _pinn_input_refusal(catalog=catalog, lhs=lhs, coefficients=coefficients)
        if pinn is not None:
            return pinn
    if not table and (lhs is None or lhs["order"] == 0):
        return _failure(
            "unsupported_form",
            "this search sealed a law with no time-derivative left-hand side "
            "(a homogeneous or zero-order form); a grid refit fits a first- or "
            "second-order one only",
        )
    if small_term_rtol > 0 and coefficients is None:
        return _failure(
            "invalid_argument",
            "small_term_rtol prunes by the candidate's coefficients, and this "
            "candidate has none (terms carry none); pass small_term_rtol=0",
        )
    return None


def _pinn_input_refusal(
    *,
    catalog: bool,
    lhs: dict[str, Any] | None,
    coefficients: list[float | None] | None,
) -> dict[str, Any] | None:
    """Why a PINN fit cannot run on this input and candidate; None when it can."""
    if catalog:
        return _failure(
            "invalid_argument",
            "pinn_epochs writes the network's field as a derived input, and "
            "only a file input (a file-v1- id) can have one; this search ran "
            "on a catalog dataset",
        )
    if lhs is None or lhs["order"] != 1:
        return _failure(
            "unsupported_form",
            "a PINN fit takes a law whose left-hand side is a first time derivative",
        )
    if coefficients is None or any(value is None for value in coefficients):
        return _failure(
            "invalid_argument",
            "a PINN fit starts from the candidate's coefficients, and this "
            "candidate has an unknown (null) one; pass a law with a number for "
            "every term",
        )
    return None


def _refined_envelope(
    refined: IsolatedRefine,
    *,
    lhs: dict[str, Any] | None,
    smoothing: dict[str, Any],
    method: str,
    trim: Trim,
    original_coefficients: list[float | None] | None,
    provenance: dict[str, Any],
    budget: dict[str, Any],
) -> dict[str, Any]:
    """The worker's refit -> the verb's envelope, every number finite or null.

    `lhs` is the source search's sealed left-hand side, carried through so the
    law passes `submit_answer` unchanged. The worker's `refit_kind` says which
    kd call it made, and the frame and the diagnostics follow it: a table
    refit carries `constants`; a grid refit carries the fit statistics, the
    pruned terms, kd's resolved frame and, for a weak fit, its window
    diagnostics. `smoothing` holds the refit's `savgol_window`,
    `lowpass_modes` and `time_window` (and its `small_term_rtol`).
    """
    refit = refined.refit
    if refit is None:
        failure = (
            _failure(
                "refine_timeout",
                "the refit was terminated at its wall-clock cap (the run's "
                "remaining time, at least a minute) and nothing was kept",
            )
            if refined.error_code == "SearchTimeout"
            else _failure(str(refined.error_code), str(refined.error_message))
        )
        return envelope(
            "failure", budget=budget, provenance=provenance, failures=[failure]
        )
    law = {
        "support": refit["terms"],
        "coefficients": [finite_or_none(value) for value in refit["coefficients"]],
        "lhs": deepcopy(lhs),
    }
    diagnostics: dict[str, Any] = {
        "nmse": finite_or_none(refit["nmse"]),
        "frame": (
            REFIT_TABLE_FRAME
            if refit["refit_kind"] == "constants"
            else _refit_frame(
                smoothing["savgol_window"],
                smoothing["lowpass_modes"],
                smoothing["time_window"],
                method=method,
                trim=trim,
            )
        ),
        "original_coefficients": original_coefficients,
        "wallclock_seconds": round(refined.wallclock_seconds, 3),
    }
    if refit["refit_kind"] == "constants":
        diagnostics["constants"] = {
            name: finite_or_none(value) for name, value in refit["constants"].items()
        }
        diagnostics["initial"] = refit["initial"]
    else:
        diagnostics["r2"] = finite_or_none(refit["r2"])
        diagnostics["condition_number"] = finite_or_none(refit["condition_number"])
        diagnostics["terms_dropped"] = [
            {"term": term, "coefficient": finite_or_none(value)}
            for term, value in refit["dropped"]
        ]
        diagnostics["derivative_source"] = refit["derivative_source"]
        diagnostics["mse"] = finite_or_none(refit["mse"])
        diagnostics["refit_frame"] = finite_tree(refit["frame"])
        diagnostics["weak"] = finite_tree(refit["weak"])
    return envelope(
        "success",
        budget=budget,
        primary={"law": law},
        diagnostics=diagnostics,
        provenance=provenance,
    )


def _smoothed_envelope(
    description: dict[str, Any],
    *,
    wallclock_seconds: float | None,
    selection: dict[str, Any] | None,
    provenance: dict[str, Any],
    budget: dict[str, Any],
) -> dict[str, Any]:
    """A derived input's description -> `smooth_input`'s success envelope.
    The same description serves a fresh write and a reuse; only a fresh one
    has a wall-clock to report, and `reused` says which this was. With
    `lowpass_modes="auto"`, `selection` is kd's choice of the counts with
    its complete trace."""
    derived = description["derived"]
    spectrum = derived["spectrum"]
    diagnostics: dict[str, Any] = {"spectrum": spectrum}
    if selection is not None:
        diagnostics["selection"] = selection
    if wallclock_seconds is not None:
        diagnostics["wallclock_seconds"] = round(wallclock_seconds, 3)
    return envelope(
        "success",
        budget=budget,
        primary={
            "dataset_id": description["ref"],
            "derived_from": derived["from"],
            "lowpass_modes": derived["lowpass_modes"],
            "time_window": derived["time_window"],
            "reused": wallclock_seconds is None,
            "energy_kept": {
                axis: entry["energy_kept"] for axis, entry in spectrum.items()
            },
        },
        diagnostics=diagnostics,
        provenance={
            **provenance,
            "file_sha256": description["source"]["sha256"],
            "path": f"inputs/{description['ref']}.npz",
        },
    )


def _pinn_refinement(
    session: Session,
    *,
    dataset_id: str,
    candidate: dict[str, Any],
    lhs: dict[str, Any],
    epochs: int,
    pretrain_epochs: int,
    residual_weight: float,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Fit a network to the input `dataset_id` derives from and to the
    candidate, register its field as a derived input, and return the verb's
    envelope.

    The fit runs on the root of `dataset_id`'s chain (the input the run is
    pinned to; a search on a smoothed copy is fitted on the data it was
    smoothed from), in a worker on the run's device, under the run's
    remaining time with a one-minute floor. The same settings name the same
    derived input (`pinn_ref`), so a repeated call reuses the description it
    registered and fits nothing, provided that input still verifies; one
    that does not is refused with `input_unverified` and never fitted again
    (`_reusable`): the deadline is not in the id, so a second fit under the
    same id could be another field.
    """
    parent = root_ref(dataset_id)
    pinn = {
        "terms": list(candidate["support"]),
        "coefficients": [float(value) for value in candidate["coefficients"]],
        "epochs": epochs,
        "pretrain_epochs": pretrain_epochs,
        "residual_weight": float(residual_weight),
        "seed": PINN_SEED,
    }
    derived_id = pinn_ref(parent, pinn)
    inputs = session.workspace / "inputs"
    wallclock: float | None = None
    resources: dict[str, Any] | None = None
    reused, refusal = _reusable(session, derived_id)
    if refusal is not None:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance={**provenance, "derived_id": derived_id},
            failures=[refusal],
        )
    if reused is not None:
        description = reused
    else:
        parent_source = cast(dict[str, Any], _resolve(session, parent))
        remaining = session.budget()["remaining_seconds"]
        session.workers_started += 1
        fitted = run_isolated_pinn(
            dataset_id=parent,
            input_source=parent_source,
            pinn=pinn,
            output_path=inputs / f"{derived_id}.npz",
            device=session.device,
            time_cap_seconds=(
                None if remaining is None else max(60.0, float(remaining))
            ),
        )
        if fitted.description is None:
            failure = (
                _failure(
                    "refine_timeout",
                    "the PINN fit was terminated at its wall-clock cap (the "
                    "run's remaining time, at least a minute) and nothing was "
                    "kept",
                )
                if fitted.error_code == "SearchTimeout"
                else _failure(str(fitted.error_code), str(fitted.error_message))
            )
            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=(
                    None
                    if fitted.resources is None
                    else {"resources": fitted.resources}
                ),
                provenance=provenance,
                failures=[failure],
            )
        save_input(session.workspace, fitted.description)
        try:
            description = _register_derived(session, fitted.description)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance={**provenance, "derived_id": derived_id},
                failures=[
                    _failure(
                        "input_unverified",
                        f"the PINN field written as {derived_id!r} does not "
                        f"verify, so it is not registered: {exc}",
                    )
                ],
            )
        wallclock = fitted.wallclock_seconds
        resources = fitted.resources
    result = description["derived"]["pinn_result"]
    diagnostics: dict[str, Any] = {
        "nmse": None,
        "frame": _pinn_frame(epochs, pretrain_epochs, residual_weight),
        "original_coefficients": candidate["coefficients"],
        "derivative_source": "autograd",
        **{
            key: finite_or_none(result[key])
            for key in (
                "initial_data_loss",
                "initial_residual_loss",
                "data_loss",
                "residual_loss",
            )
        },
        "epochs_run": result["epochs_run"],
        "stopped_at_deadline": result["stopped_at_deadline"],
        "device": result["device"],
    }
    if wallclock is not None:
        diagnostics["wallclock_seconds"] = round(wallclock, 3)
        diagnostics["resources"] = resources
    return envelope(
        "success",
        budget=session.budget(),
        primary={
            "law": {
                "support": pinn["terms"],
                "coefficients": [
                    finite_or_none(value) for value in result["coefficients"]
                ],
                "lhs": deepcopy(lhs),
            },
            "dataset_id": derived_id,
            "derived_from": parent,
            "reused": wallclock is None,
        },
        diagnostics=diagnostics,
        provenance={
            **provenance,
            "refined_from": PINN_REFINED_FROM,
            "file_sha256": description["source"]["sha256"],
            "path": f"inputs/{derived_id}.npz",
        },
    )


def _tool_message(payload: dict[str, Any], name: str, tool_call_id: str) -> ToolMessage:
    """A hand-built ToolMessage whose content must be byte-for-byte what
    ToolNode produces today.

    `msg_content_output` (`tool_node.py`) uses
    `json.dumps(output, ensure_ascii=False)`. Writing `str(payload)` would turn
    what the model sees from JSON into a Python repr, and dropping
    `ensure_ascii=False` would escape every non-ASCII character into \\uXXXX --
    both are silent, model-visible changes that nothing else would report.
    `name` is set the way ToolNode sets it too: the model does not see it (a
    tool-role message carries no name), but trace.jsonl and LangSmith align on
    it.
    """
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        name=name,
        tool_call_id=tool_call_id,
    )


def _spends_generations(schema: dict[str, Any] | None) -> bool:
    """Whether this algorithm consumes ``generations`` anywhere.

    Read off the schema rather than branched on the algorithm name, so adding an
    instrument to kd still needs no change here. An unregistered name answers
    ``True``: the facade's own algorithm check is the one that should report it.
    """
    if schema is None:
        return True
    return any(
        row["name"] == "generations" and row["effect"] != "unused"
        for row in schema["facade_params"]
    )


def _parameter_provenance(
    schema: dict[str, Any] | None,
    requested: dict[str, Any],
    defaults_applied: dict[str, Any],
) -> dict[str, Any]:
    """Describe requested/defaulted fields and their resume-identity effect."""
    identity_breaking: list[str] = []
    if schema is not None:
        config_names = {
            row["facade_param"] or row["name"]: row["name"] for row in schema["fields"]
        }
        translated = {config_names.get(name, name) for name in requested}
        identity_breaking = sorted(translated & set(schema["identity_breaking_fields"]))
    return {
        "requested": requested,
        "defaults_applied": defaults_applied,
        "identity_breaking": identity_breaking,
    }


def prime_prior(session: Session, prior_path: Path) -> dict[str, str]:
    """Load one prior document for this run and post its annotation to `session`.

    The four refusals, then the projection, then the mailbox. This is the whole
    of what `--prior` does before a search starts, and it is one function
    because the C4 probe drives the same sequence: a probe checking a
    hand-rolled copy of these steps would be checking a path the run lane does
    not take. The return value is what the run records about the document
    (`prior_path`, `prior_type`, `prior_sha256`), so the word "eval" or "usage"
    is decided in the one place that also acts on the distinction. `run` splits
    that record in two: the trace metadata takes the type and the digest, while
    the absolute path stays on the teardown result, which is never written into
    the workspace (`run._TRACE_PRIOR_KEYS`).

    The refusals, in order:

    1. **A pinned dataset is required.** A prior is read against one dataset's
       features, and on an unpinned run the model picks the dataset later and
       from several.
    2. **The document may not sit under the workspace root.** The workspace is
       the one directory this run tells the model to browse -- the skills, the
       trace and the sealed run directories all land there -- so a document
       filed inside it is one the model is pointed at, readable in full (every
       row, every basis sentence) on a channel meant to carry two fields about
       one instrument (ruling 8-D5). It is containment and not access control:
       `root_dir` bounds the file tools without constraining the shell
       (`orchestration.build_agent` says so), so a path outside the workspace is
       still readable by a model that goes looking for it.
    3. **`priors.assert_serves`**, fail-loud. One run pins one dataset, so the
       document either serves it or the run has nothing to say with it; the tree
       lane refuses the same document per leaf and soft, and both read the one
       judgement in `priors` (ruling 8-D6).
    4. **A usage build is refused on a dataset kd's catalog carries** (ruling
       8-D7): it withheld nothing, so on a catalog dataset it can be feeding a
       measurement run the observations of the very dataset being measured.
       "Is this a measurement run" is undecidable in general and decidable here,
       because `--prior` requires a pinned dataset and the catalog is
       enumerable. This lane reaches catalog ids only today -- an id off the
       catalog gets past this gate and dies one step later at `get_dataset` --
       so the `usage` value the record can carry is reserved rather than
       reachable, and the surface a usage build serves is the tree lane's
       `status --prior`.

    The mailbox is written last, so a refused document leaves the run with no
    annotation at all rather than half of one.
    """
    pinned = session.pinned_dataset_id
    if pinned is not None and is_file_ref(pinned):
        raise ValueError("prior supports catalog datasets only, not file inputs")
    if pinned is None:
        raise ValueError(
            f"{prior_path} was passed to a run that pinned no dataset; a prior "
            "is read against one dataset's features, and an unpinned run picks "
            "its dataset later and from several"
        )
    _refuse_document_in_workspace(
        session,
        prior_path,
        "a prior reaches this run as one annotated row per instrument",
    )
    prior = priors.load_prior(prior_path)
    priors.assert_serves(prior, pinned)
    excluded = prior.get("excluded")
    if not excluded and pinned in {row["id"] for row in list_datasets_answer_blind()}:
        raise ValueError(
            f"{prior_path} withheld nothing and {pinned!r} is in kd's catalog: "
            "a usage build is a quality filter over the whole catalog, so read "
            "on a dataset the catalog carries it can put that dataset's own "
            "observations into the run measuring it. An evaluation build (one "
            "naming what it withheld) is the kind that serves a catalog dataset"
        )
    report = preview_report(get_dataset(pinned).loader()).to_dict()
    session.prior_annotation = priors.annotate(
        prior,
        priors.features_from_report(report),
        [schema["algorithm"] for schema in instrument_schemas()],
    )
    return {
        "prior_path": str(prior_path),
        "prior_type": "eval" if excluded else "usage",



        "prior_sha256": file_sha256(prior_path),
    }


def _refuse_document_in_workspace(session: Session, path: Path, reach: str) -> None:
    """Refuse a document filed under the workspace root (ruling 8-D5).

    The workspace is the one directory a run points the model at, so a
    document inside it is readable in full; `reach` names the narrower channel
    the document is meant to arrive on. `prime_prior` and `prime_experience`
    share this one judgement, and each passes its own channel.
    """
    if path.resolve().is_relative_to(session.workspace.resolve()):
        raise ValueError(
            f"{path} sits under the workspace {session.workspace}, the "
            f"one directory this run points the model at; {reach}, and keeping "
            "the document outside the workspace is what holds it to that"
        )


def prime_experience(
    session: Session, memory_path: Path
) -> tuple[dict[str, str], experience.Retrieval | None]:
    """Load a v2 document and retrieve rows plus their input readings.

    The document stays outside the model-readable workspace. Its declaration
    digest must match installed kd; the stored input summary supplies the
    report and task kind, without loading the dataset again.
    """
    import kd

    _refuse_document_in_workspace(
        session,
        memory_path,
        "an experience document reaches this run as the rows retrieved for its input",
    )
    document = experience.load_experience(memory_path)
    if document["instrument_schema_sha256"] != kd.instrument_schema_digest():
        raise ValueError(
            "instrument_schema_sha256 differs from installed kd; rebuild the experience document"
        )
    summary = cast(dict[str, Any], session.input_source)["summary"]
    report, kind = summary["report"], summary["task_kind"]
    vocabulary = document["features_vocabulary"][kind]
    features = experience.features_of(report, kind, vocabulary, document["bins"])
    rows = experience.retrieve(document, kind, features, k=2)
    record = {
        "memory_path": str(memory_path),
        "memory_sha256": file_sha256(memory_path),
    }
    if not rows:
        return record, None
    return record, {
        "rows": rows,
        "readings": experience.readings(report, kind, vocabulary, document["bins"]),
        "vocabulary": vocabulary,
        "templates": document["templates"][kind],
        "bins": document["bins"],
        "min_support": document["min_support"],
        "rules_version": document["rules_version"],
        "task_kind": kind,
    }


def prime_action_experience(
    session: Session, action_memory_path: Path
) -> dict[str, str]:
    """Freeze one compatible action document before constructing the model."""
    import kd

    if session.input_source is None:
        raise ValueError("action_memory supports file inputs only")
    _refuse_document_in_workspace(session, action_memory_path, "action experience")
    raw = action_memory_path.read_bytes()
    document = json.loads(raw)
    action_memory.validate_document(document)
    if document["instrument_schema_sha256"] != kd.instrument_schema_digest():
        raise ValueError("instrument_schema_sha256 differs from installed kd")
    if document["implementation_identity"] != action_memory.implementation_identity():
        raise ValueError("implementation_identity differs from installed source")
    if (
        document["version"] == 2
        and document["effect_protocol"]["selector_sha256"]
        != action_memory.selector_identity()
    ):
        raise ValueError("selector_sha256 differs from installed selector")
    summary = session.input_source["summary"]
    kind = summary["task_kind"]
    vocabulary = [
        key
        for key in document["features_vocabulary"][kind]
        if key in experience.FEATURES[kind]
    ]
    session.action_features = experience.features_of(
        summary["report"], kind, vocabulary, document["bins"]
    )
    session.action_experience = document
    session.action_memory_sha256 = hashlib.sha256(raw).hexdigest()
    session.action_presented.clear()
    return {
        "action_memory_path": str(action_memory_path),
        "action_memory_sha256": session.action_memory_sha256,
    }


def _action_transition(
    session: Session, entry: SegmentEntry, pipeline: dict[str, Any]
) -> str | None:
    parent_id = entry.decision_parent
    if parent_id is None or entry.action in {"fork", "rollback", "narrow"}:
        return None
    if entry.parent_run_id is not None and entry.parent_run_id != parent_id:
        return None
    previous = session.action_pipelines.get(parent_id)
    if previous is None:
        return None
    if previous["algorithm"] != pipeline["algorithm"]:
        return "switch" if entry.parent_run_id is None else None
    if previous != pipeline:
        return "retune"
    return "continue" if entry.parent_run_id == parent_id else "restart"


def _action_query(
    session: Session, result: dict[str, Any]
) -> tuple[dict[str, Any], SegmentEntry] | None:
    provenance = result["provenance"]
    run_id = provenance.get("run_id")
    entries = {entry.run_id: entry for entry in session.ledger.segments()}
    params = provenance.get("params") or {}
    if (
        run_id not in entries
        or "effective" not in params
        or "facade_effective" not in params
        or "execution_conditions" not in provenance
    ):
        return None
    if provenance.get("sketch") is not None or provenance.get("surrogate") is not None:
        return None
    entry = entries[run_id]
    summary = cast(dict[str, Any], session.input_source)["summary"]
    try:
        pipeline = action_memory.normalize_pipeline(
            provenance["instrument"],
            effective_config=params["effective"],
            facade_params=params["facade_effective"],
            report=summary["report"],
        )
    except ValueError:
        return None
    diagnostics = result["diagnostics"]
    metric = diagnostics.get("platform_nmse")
    status = (
        "measured" if metric is not None and math.isfinite(metric) else "unavailable"
    )
    if diagnostics.get("platform_eval_error"):
        status = "failed"
    remaining = session.budget()["remaining_seconds"]
    state = action_memory.project_state(
        features=session.action_features,
        entry=entry,
        pipeline=pipeline,
        metric_status=status,
        last_action=_action_transition(session, entry, pipeline),
        conditions=provenance["execution_conditions"],
        remaining_budget={
            "seconds": None if remaining is None else max(0, remaining),
            "evaluations": None,
        },
    )
    session.action_pipelines[run_id] = pipeline
    return state, entry


def _action_memory_diagnostics(
    session: Session, result: dict[str, Any]
) -> dict[str, Any]:
    """Retrieve after ledger registration, including sealed failed segments."""
    document = session.action_experience
    if document is None:
        return {}
    digest = cast(str, session.action_memory_sha256)
    query = _action_query(session, result)
    if query is None:
        return action_memory_text.diagnostic("unavailable", digest, [])
    state, entry = query
    report = cast(dict[str, Any], session.input_source)["summary"]["report"]
    legal = action_memory.legal_actions(
        document,
        state,
        entry=entry,
        ledger=session.ledger,
        workspace=session.workspace,
        report=report,
    )
    hits = action_memory.retrieve_document(document, state, legal_actions=legal)
    if not hits:
        return action_memory_text.diagnostic("no_match", digest, [])
    matching_state = {
        key: value for key, value in state.items() if key != "remaining_budget"
    }
    key = json.dumps(
        [digest, matching_state, legal, [h["row_index"] for h in hits]],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if key in session.action_presented:
        return action_memory_text.diagnostic("unchanged", digest, [])
    session.action_presented.add(key)
    return action_memory_text.diagnostic("matched", digest, hits)


def _collected_controls(
    schema: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Record consumed facade controls at invocation, outside plugin settings."""
    return {
        param["name"]: deepcopy(kwargs.get(param["name"], param["default"]))
        for param in schema["facade_params"]
        if param["config_field"] is None
    }


def _execution_conditions(device: str) -> dict[str, Any]:
    """The run's device policy (`Session.device`, ``"cpu"`` or ``"cuda"``) and
    the worker's existing environment pins.

    The policy is the one the search was launched under; ``model_kwargs``
    cannot carry a device (`kd.harness.PlanEntry` reserves the key).
    """
    seed = os.environ.get("PYTHONHASHSEED")
    return {
        "device": device,
        "blas_threads": 1,
        "python_hash_seed": None if seed is None or seed == "random" else int(seed),
    }


def _search_precision_failure(
    session: Session,
    schema: dict[str, Any] | None,
    model_kwargs: dict[str, Any],
    defaults_applied: dict[str, Any],
    *,
    table: bool,
    surrogate: _ResolvedSurrogate | None,
) -> dict[str, Any] | None:
    """Resolve a search's ``training_dtype`` in place, or refuse it.

    When the call's mode trains a field network
    (`surrogate_training.takes_training_dtype`), its value is resolved under
    the run's choice (`session.effective_training_dtype`), and an inherited
    value is written into ``model_kwargs`` and ``defaults_applied`` (the
    adapter filled it in). Any other mode never receives the run's choice;
    an explicit value there is only checked against kd's values and kd's
    mode refusal stays the answer. An injected surrogate has to have been
    trained in the precision the search runs in: kd verifies a dlga file's
    identity, but an sga surrogate is a bare module, so the registry recipe
    is compared here, before anything spawns.
    """
    takes = schema is not None and surrogate_training.takes_training_dtype(
        schema, model_kwargs, table=table
    )
    given = model_kwargs.get("training_dtype")
    try:
        value = effective_training_dtype(
            session.training_dtype if takes else None, given
        )
    except ValueError as exc:
        return _failure("invalid_argument", str(exc))
    if value is not None and given is None:
        model_kwargs["training_dtype"] = value
        defaults_applied["training_dtype"] = value
    if surrogate is None:
        return None
    entry = session.surrogates.entry(surrogate.surrogate_id)
    trained = entry.recipe["params"].get("training_dtype")
    if trained == value:
        return None
    return _failure(
        "surrogate_training_dtype_mismatch",
        f"surrogate {surrogate.surrogate_id!r} was trained under training_dtype "
        f"{trained!r} and this search runs under {value!r}; a search injects "
        "only a surrogate trained in its own precision. Train one with "
        "train_surrogate under this run (it takes the run's precision) and "
        "pass that id",
    )


def _network_target(
    session: Session,
    schema: dict[str, Any] | None,
    *,
    dataset_id: str,
    seed: int,
    model_kwargs: dict[str, Any],
    run_dir: Path,
    injected: bool,
    resumed: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """The recipe and registry id of the network this search trains, or why
    it gets no save target.

    Only an instrument whose surrogate kd stores with its identity
    (:data:`STORED_SURROGATE_KEY`, dlga) has one; any other answers
    `(None, None)`. A search that injects a registered network trains none
    (`"injected"`), and a resumed or forked segment restores its network from
    the parent's checkpoint (`"resumed"`): neither gets a target, so no file
    is saved that no row would name. Otherwise the search gets a target
    whatever the registry already holds for its recipe: the network it
    trains is its own, registered under an id of its own
    (`surrogates.trained_surrogate_id`), and no stored network is injected
    in its place. The recipe is `surrogate_training.search_recipe`'s.
    """
    if (
        schema is None
        or not schema["config_artifact_keys"]
        or surrogate_training.artifact_key(schema) != STORED_SURROGATE_KEY
    ):
        return None, None
    if injected:
        return None, "injected"
    if resumed:
        return None, "resumed"
    recipe = surrogate_training.search_recipe(
        schema, dataset_id, seed, model_kwargs, session.training_dtype
    )
    return {
        "recipe": recipe,
        "surrogate_id": trained_surrogate_id(recipe, run_dir.name),
    }, None


def _register_network(
    session: Session,
    target: dict[str, Any] | None,
    reason: str | None,
    isolated: IsolatedEpisode,
) -> tuple[str | None, dict[str, Any] | None]:
    """Register the network this search trained, and return its id and the
    reply's `diagnostics.surrogate` (None for an instrument with no stored
    network).

    A registered network reports its id, whether the fit's deadline cut its
    training short, the `savgol_window` its input was smoothed with (None
    for the input as loaded) and the training facts. Without one the block
    says why: `injected`, `resumed`, or `not_trained` with the worker's
    detail (the fit stopped before it trained a network, the export
    failed, the parent could not read the saved file, or the worker wrote
    no result).
    """
    if target is None:
        return None, None if reason is None else {
            "surrogate_id": None,
            "reason": reason,
        }
    network = isolated.trained_surrogate
    if network is None:
        detail = isolated.trained_surrogate_error
        return None, {
            "surrogate_id": None,
            "reason": "not_trained",
            "detail": "the search's worker wrote no result, so it saved no network"
            if detail is None
            else detail,
        }
    entry = surrogate_training.register_search_network(
        registry=session.surrogates,
        recipe=target["recipe"],
        key=STORED_SURROGATE_KEY,
        surrogate_id=target["surrogate_id"],
        network=network,
        source_run=isolated.outcome.run_id,
    )
    return entry.surrogate_id, {
        "surrogate_id": entry.surrogate_id,
        "stopped_at_deadline": stopped_at_deadline(entry),
        "savgol_window": entry.recipe["params"].get("savgol_window"),
        "train_seconds": entry.train_seconds,
        "epochs": entry.epochs,
        "final_loss": entry.final_loss,
    }


def _prior_diagnostics(session: Session, algorithm: str) -> dict[str, Any]:
    """The `prior` diagnostics key for one instrument, or no key at all.

    A conditional key of the same family as `segment_report` and `sketch`:
    written when there is one, absent when there is not. It rides every envelope
    `run_discovery` returns, refusals included -- the annotation is about the
    instrument the call names, and a refusal is exactly where knowing what the
    prior says about that instrument can change the next call.

    An instrument the annotation does not carry gets no key either. `algorithm`
    is a model-supplied string and the refusals ahead of `schema_of` are reached
    before anything validates it, so an unregistered name arrives here; a
    KeyError raised on it would escape the tool and take the run's output with
    it.
    """
    if session.prior_annotation is None:
        return {}
    row = session.prior_annotation.get(algorithm)
    return {} if row is None else {"prior": row}


Trim = int | dict[str, int] | Literal["by_order"]
"""A measurement's row trimming, as `kd.refit_terms` takes it: grid points
removed at both ends of every axis, per axis, or `"by_order"` (the stencil
footprint of the terms)."""

METHODS: Final = ("finite_diff", "autograd", "weak")
"""The derivative methods of a framed measurement (`kd.refit_terms`)."""

WEAK_OPTIONS: Final = frozenset(
    {"half_widths", "spectrum_tau", "noise_correction", "noise_sigma"}
)
"""The keys of `weak`: the fields of `kd.WeakFormOptions`."""

EVIDENCE_FILENAME: Final = "evidence.jsonl"
"""The reporting copy of this run's evidence records, under the workspace."""

RECEIPTS_FILENAME: Final = "refinements.jsonl"
"""The reporting copy of this run's refinement receipts, under the workspace."""

RECEIPT_FORMAT: Final = "kdagent-refinement-receipt-v1"
EVIDENCE_FORMAT: Final = "kdagent-evidence-v1"


def _new_id(prefix: str) -> str:
    """A fresh id for a receipt, an evidence record or a sketch request: the
    prefix and twelve random hex digits, so a reopened workspace's earlier
    ids never collide with this run's."""
    return f"{prefix}-{secrets.token_hex(6)}"


def _canonical_sha256(value: Any) -> str:
    """The sha256 of `value` as canonical JSON (sorted keys, no spaces)."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_PYSR_OPERATOR: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[-+*/^]")
"""A PySR operator named, not defined: PySR hands an operator string to
Julia, and one with a definition in it (`"inv(x) = 1/x"`) is Julia code."""

_PYSR_PLAIN_OPTIONS: Final[dict[str, tuple[object, ...]]] = {
    "deterministic": (True, False),
    "parallelism": ("serial", "multithreading", "multiprocessing"),
}
"""The `extra_pysr_kwargs` entries this controller passes on, with the values
each takes: settings, not code. The rest of that pass-through reaches the
`PySRRegressor` constructor unchanged, and several of its options (a loss,
operator mappings, output paths) are code or files."""


def _untrusted_asset_failure(
    schema: dict[str, Any] | None, params: dict[str, Any]
) -> dict[str, Any] | None:
    """The refusal of search parameters that would make kd execute something
    the model chose, or None (correction 1 of the 2026-10-24 review).

    A parameter whose declared type is a path names a file kd loads, and
    for EqGPT that is a pickle or a torch checkpoint (`wave_pkl_path`
    reaches `pickle.load` before any data check): a file the model wrote
    through its file tools would be deserialized. kd resolves its own
    assets when the parameter is omitted. PySR's `extra_pysr_kwargs` and an
    operator string with a definition reach Julia unchanged, which compiles
    them (`_PYSR_PLAIN_OPTIONS`, `_PYSR_OPERATOR`). Checked before anything
    starts, so nothing is loaded. A null path is kd's own default, which a
    resume restates from the parent's effective parameters, and passes.
    """
    if schema is None:
        return None
    paths = [
        field["name"]
        for field in schema["fields"]
        if "Path" in field["type"] and params.get(field["name"]) is not None
    ]
    if paths:
        return _failure(
            "untrusted_asset",
            f"{paths} name files for kd to load (a pickle or a torch "
            "checkpoint), and this controller does not load a file a model "
            "names; omit them and kd uses its own assets",
        )
    if schema["algorithm"] != "pysr":
        return None


    extra = params.get("extra_pysr_kwargs")
    code = sorted(
        name
        for name, value in (extra.items() if isinstance(extra, dict) else ())
        if name not in _PYSR_PLAIN_OPTIONS or value not in _PYSR_PLAIN_OPTIONS[name]
    )
    if code:
        return _failure(
            "untrusted_asset",
            f"extra_pysr_kwargs {code} would reach PySR unchanged, and PySR "
            "compiles some of its options as Julia code; only "
            + ", ".join(
                f"{name} ({', '.join(json.dumps(value) for value in values)})"
                for name, values in _PYSR_PLAIN_OPTIONS.items()
            )
            + " are accepted here",
        )
    defined = [
        operator
        for name in ("binary_operators", "unary_operators")
        if isinstance(params.get(name), list)
        for operator in params[name]
        if not _PYSR_OPERATOR.fullmatch(str(operator))
    ]
    if defined:
        return _failure(
            "untrusted_asset",
            f"operators {defined} are definitions, which PySR runs as Julia "
            'code; name built-in operators only (such as "+", "*", "sin")',
        )
    return None


_LLM4ED_RUN_MANAGED: Final = ("base_url", "model", "extra_body")
"""The LLM4ED parameters that are this run's endpoint, set by its
configuration (`Session.llm4ed`): a search may not pass them."""

_LLM4ED_CALL_LIMITS: Final = tuple(
    name for name in LLM4ED_LIMITS if name != "max_search_seconds"
)
"""The LLM4ED parameters bounded by the run's configured limits."""


def _llm4ed_refusal(
    llm4ed: Llm4edConfig | None, params: dict[str, Any]
) -> dict[str, Any] | None:
    """The refusal of an LLM4ED search before anything starts, or None: the
    run has no LLM4ED endpoint (`LLM4ED_OFF`), the call names a file for kd
    to write (`tape_record_path`, a path the model would choose outside the
    write fence of its file tools; null, kd's default, passes), gives a
    run-managed endpoint parameter another value than the run's (the same
    value, which a resume restates from the parent's effective parameters,
    is accepted), or asks for more than a configured limit."""
    if llm4ed is None:
        return _failure("instrument_unavailable", LLM4ED_OFF)
    if params.get("tape_record_path") is not None:
        return _failure(
            "untrusted_asset",
            "tape_record_path names a file for kd to write, and this "
            "controller does not let a model choose one; omit it",
        )
    managed = [
        name
        for name in _LLM4ED_RUN_MANAGED
        if name in params and params[name] != getattr(llm4ed, name)
    ]
    if managed:
        return _failure(
            "invalid_argument",
            f"{managed} differ from this run's LLM4ED endpoint, which its "
            "configuration sets; omit them",
        )
    over = [
        f"{name} {params[name]!r} (at most {getattr(llm4ed, name)})"
        for name in _LLM4ED_CALL_LIMITS
        if name in params
        and isinstance(params[name], int)
        and params[name] > getattr(llm4ed, name)
    ]
    if over:
        return _failure(
            "invalid_argument",
            f"{', '.join(over)} exceed this run's LLM4ED limits; ask for "
            "less or omit them",
        )
    return None


def _apply_llm4ed(
    llm4ed: Llm4edConfig,
    model_kwargs: dict[str, Any],
    defaults_applied: dict[str, Any],
) -> None:
    """Write the run's LLM4ED endpoint, and each configured limit the call
    did not lower, into the search's parameters; each is recorded as
    supplied by the controller. The key is not among them."""
    supplied: dict[str, Any] = {"base_url": llm4ed.base_url, "model": llm4ed.model}
    if llm4ed.extra_body is not None:
        supplied["extra_body"] = deepcopy(llm4ed.extra_body)
    for name in _LLM4ED_CALL_LIMITS:
        if name not in model_kwargs:
            supplied[name] = getattr(llm4ed, name)
    model_kwargs.update(supplied)
    defaults_applied.update(deepcopy(supplied))


def _llm4ed_cap(
    session: Session, max_search_seconds: int | None
) -> tuple[float | None, dict[str, Any] | None]:
    """An LLM4ED search's wall-clock cap: the configured limit, the call's
    `max_search_seconds` and the run's remaining time, whichever is
    smallest, with no floor (every second may be billed requests); a run
    whose time is spent is refused."""
    llm4ed = cast(Llm4edConfig, session.llm4ed)
    remaining = session.budget()["remaining_seconds"]
    if remaining is not None and remaining <= 0:
        return None, _failure(
            "budget_exhausted",
            "this run's time budget is spent, and an LLM4ED search gets no "
            "minimum time",
        )
    limits = [float(llm4ed.max_search_seconds)]
    if max_search_seconds is not None:
        limits.append(float(max_search_seconds))
    if remaining is not None:
        limits.append(float(remaining))
    return min(limits), None


def _llm4ed_description(llm4ed: Llm4edConfig | None) -> str:
    """The `run_discovery` description's LLM4ED paragraph for this run."""
    if llm4ed is None:
        text = f"LLM4ED: {LLM4ED_OFF}"
    else:
        text = (
            f"LLM4ED: this run's LLM4ED endpoint is `{llm4ed.model}` at "
            f"`{llm4ed.base_url}`, configured by the operator with its own key. "
            "The run supplies `base_url`, `model` and `extra_body`, and a call "
            "that passes them is refused. "
            + ", ".join(
                f"`{name}` (at most {getattr(llm4ed, name)})"
                for name in _LLM4ED_CALL_LIMITS
            )
            + " may be lowered, not raised; omitted, each is that limit. An "
            f"LLM4ED search is capped at {llm4ed.max_search_seconds} s, or at "
            "`max_search_seconds` or the run's remaining time when smaller, "
            "and its requests are charged to that key's account."
        )
    return _description_paragraph(text)


def _process_only_refusal(algorithm: str) -> dict[str, Any] | None:
    """The refusal of a registered algorithm that exists in this interpreter
    only, or None: every search and training runs in a worker process that
    imports kd afresh, which registers kd's built-ins and runs the installed
    `kd.extensions` hooks, so a `kd.register_algorithm` call made in this
    process is not there (`kd.extension_source` is `process`)."""
    if extension_source("algorithm", algorithm) != "process":
        return None
    return _failure(
        "instrument_unavailable",
        f"{algorithm!r} is registered in this process only, and every search "
        "and training runs in a worker process where it is not registered. "
        "Register it from a `kd.extensions` entry-point hook of an installed "
        "distribution, which every interpreter that imports kd runs",
    )


def _description_paragraph(text: str) -> str:
    """`text` wrapped as a paragraph of a tool description."""
    return textwrap.fill(text, width=72, subsequent_indent=" " * 8)


def _reason_refusal(reason: str | None) -> dict[str, Any] | None:
    """The refusal of a `reason` that is given and blank, or None: a reason
    is optional, and an empty one says nothing the tree could record."""
    if reason is not None and not reason.strip():
        return _failure(
            "invalid_argument",
            "reason has to say why this call is made; omit it rather than "
            "pass an empty one",
        )
    return None


def _note_action(
    session: Session,
    result: dict[str, Any],
    *,
    verb: str,
    ref: dict[str, Any],
    run_id: str | None,
    reason: str | None,
    status: str,
) -> None:
    """Append the call's `action` event to the tree. The tree is a side
    channel of the call's own record (the registry row, the receipt, the
    evidence line): an append that raises is added to `failures` and the
    result stands."""
    try:
        session.ledger.note(
            verb=verb,
            ref=ref,
            run_id=run_id,
            reason=reason,
            status=status,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
    except Exception as exc:
        result["failures"].append(_crashed(exc))


_ACTION_REFS: Final[dict[str, tuple[str, ...]]] = {
    "train_surrogate": ("surrogate_id",),
    "smooth_input": ("dataset_id", "surrogate_id"),
    "refine_candidate": ("refinement_id", "dataset_id"),
    "ablate_candidate": ("evidence_id",),
    "score_term_sets": ("evidence_id",),
    "verify_candidate": ("evidence_id",),
    "vote_terms": ("evidence_id",),
    "make_sketch": ("sketch_id",),
    "set_sketch": ("sketch_id",),
    "verify_system": ("system_id", "verification_id"),
}
"""The ids a verb's `primary` names that its `action` event records as
`ref`: what the call produced."""


def _logged(
    session: Session,
    result: dict[str, Any],
    *,
    verb: str,
    reason: str | None,
    started: int,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Note the call on the tree and return its envelope.

    A call that produced its result is noted with the envelope's status and
    the ids it produced. A call that started a worker
    (`Session.workers_started` moved past `started`) and failed is noted
    with status `failure` and no id. A call refused before it started
    anything is not an action and is not noted; its envelope is the record.
    """
    if result["status"] != "failure":
        ref = {
            key: result["primary"][key]
            for key in _ACTION_REFS[verb]
            if key in result["primary"]
        }
        status = result["status"]
    elif session.workers_started > started:
        ref, status = {}, "failure"
    else:
        return result
    _note_action(
        session, result, verb=verb, ref=ref, run_id=run_id, reason=reason, status=status
    )
    return result


def _append_record(session: Session, filename: str, record: dict[str, Any]) -> None:
    """Append one JSON line to a reporting file under the workspace."""
    path = session.workspace / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _law_keys(terms: list[str]) -> list[str]:
    """kd's law-term identity of each term (the alias-normalized key
    `kd.vote_terms` counts by, which folds argument order of commutative
    operators), read off a vote over one-term supports so the controller
    never computes a key of its own. A term may be written in kd's product
    notation (`u*u_x`), which kd lowers to its IR before keying. Raises
    ValueError for a term kd cannot read."""
    if not terms:
        return []
    keys = [""] * len(terms)
    for row in kd.vote_terms([[term] for term in terms], min_count=1).rows:
        for index in row.support_indices:
            keys[index] = row.law_key
    return keys


def _frame_refusal(
    method: str,
    trim: Trim,
    weak: dict[str, Any] | None,
    surrogate_id: str | None,
    *,
    inline: bool = False,
) -> dict[str, Any] | None:
    """Why this measurement frame cannot be asked for; None when it can.

    The combinations refused here need no data: an unknown method, weak
    options outside `kd.WeakFormOptions` or without the weak method, a weak
    fit with trimming, a model or inline smoothing or PINN (`inline`), an
    autograd fit without a model, or with inline smoothing or a PINN fit (a
    model is bound to the input as it is), and a model without autograd.
    Nothing substitutes another method. kd's own refusals (a weak option
    value, `trim="by_order"` with autograd, a term outside the weak class)
    come back from the worker.
    """
    if method not in METHODS:
        return _failure(
            "invalid_argument",
            f"method has to be one of {list(METHODS)}, got {method!r}",
        )
    if weak is not None and method != "weak":
        return _failure(
            "invalid_argument",
            f"weak options apply to method='weak' only; this call has method={method!r}",
        )
    unknown = sorted(set(weak or {}) - WEAK_OPTIONS)
    if unknown:
        return _failure(
            "invalid_argument",
            f"weak takes {sorted(WEAK_OPTIONS)}; {unknown} are not weak options",
        )
    if method == "weak" and (trim != 0 or surrogate_id is not None or inline):
        return _failure(
            "invalid_argument",
            "a weak fit integrates the input it is given over test-function "
            "windows: trim, surrogate_id, savgol_window, lowpass_modes and "
            "pinn_epochs are refused with it. Derive a smoothed input first "
            "and pass it as the measurement input",
        )
    if method == "autograd" and (surrogate_id is None or inline):
        return _failure(
            "invalid_argument",
            "method='autograd' takes field values and derivatives from the "
            "model surrogate_id names, bound to the input as it is, so it "
            "needs surrogate_id and takes no savgol_window, lowpass_modes or "
            "pinn_epochs",
        )
    if surrogate_id is not None and method != "autograd":
        return _failure(
            "invalid_argument",
            "surrogate_id is the model an autograd fit differentiates; pass "
            "method='autograd' with it, or drop it",
        )
    return None


def _measured_input(
    session: Session, run_id: str, input_id: str | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The input a measurement of search `run_id` is made on, or a refusal.

    The search must have succeeded this run, and the input it ran on must
    still be what it was launched on. An omitted `input_id` is that input;
    an explicit one must be its root or an input derived from that root and
    registered this run (a derived input keeps its parent's left-hand side
    at every link, so the same root is the same left-hand side). Nothing
    substitutes another input. Returns the search and measurement dataset
    ids, the measurement input's description and record, and the search's
    sealed left-hand side.
    """
    if run_id not in session.successful_runs:
        return None, _unknown_run_failure(session, run_id)
    refusal = _search_input_refusal(session, run_id)
    if refusal is not None:
        return None, refusal
    searched = session.successful_runs[run_id]["dataset"]
    dataset_id = searched if input_id is None else input_id
    if root_ref(dataset_id) != root_ref(searched):
        return None, _failure(
            "measurement_input_mismatch",
            f"{dataset_id!r} is not the input search {run_id!r} ran on "
            f"({searched!r}), its root or an input derived from that root; a "
            "measurement of this search is made on one of those",
        )
    refusal = _input_refusal(session, dataset_id)
    if refusal is not None:
        return None, refusal
    chain = (
        derivation_chain(session.workspace, dataset_id, written=session.derived_inputs)
        if is_file_ref(dataset_id)
        else []
    )
    return {
        "run_id": run_id,
        "searched": searched,
        "dataset_id": dataset_id,
        "input_source": chain[-1] if chain else _resolve(session, dataset_id),
        "record": input_record(dataset_id, chain),
        "lhs": session.sealed_lhs[run_id],
    }, None


def _candidate_law(
    session: Session, run_id: str, law: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The law a measurement of search `run_id` measures: its sealed law, or
    a given one held to the shape and left-hand side `submit_answer`
    checks."""
    if law is None:
        return deepcopy(session.sealed_laws[run_id]), None
    refusal = _law_shape_failure(law) or _lhs_failure(law, session.sealed_lhs[run_id])
    if refusal is not None:
        return None, refusal
    return deepcopy(law), None


def _run_precision_failure(
    session: Session, surrogate_id: str, entry: SurrogateEntry, remedy: str
) -> dict[str, Any] | None:
    """The refusal of a registered model trained under another precision
    than the run's (`Session.training_dtype`), or None.

    A run that made no choice uses a model of any precision. A run that made
    one uses only a model trained in it, and a stored null (trained with no
    stated precision) differs from it. Sampling a network (`smooth_input`)
    and an autograd measurement frame (`_model_binding`) apply this rule;
    `remedy` says what the refused call can do instead. A search compares
    the surrogate it injects with its own resolved precision instead
    (`_search_precision_failure`).
    """
    trained = entry.recipe["params"].get("training_dtype")
    if session.training_dtype is None or trained == session.training_dtype:
        return None
    return _failure(
        "surrogate_training_dtype_mismatch",
        f"surrogate {surrogate_id!r} was trained under training_dtype "
        f"{trained!r} and this run trains under {session.training_dtype!r}; "
        f"{remedy}",
    )


def _model_binding(
    session: Session, surrogate_id: str, dataset_id: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The registered model file an autograd measurement on `dataset_id`
    differentiates, as `{"path", "sha256"}`, or a refusal.

    The model must be registered in this workspace, still hash to its row,
    have been trained on `dataset_id` itself (it is bound to that input, and
    an input sampled from it, or any other, is refused rather than bound),
    and have been trained in the run's precision (`_run_precision_failure`).
    kd checks the stored binding again when it loads the file.
    """
    registry = session.surrogates
    try:
        entry = registry.entry(surrogate_id)
    except KeyError:
        return None, _failure(
            "unknown_surrogate",
            f"surrogate {surrogate_id!r} is not registered in this workspace; "
            f"the registered ones are {list(registry.ids())}",
        )
    path = registry.path(surrogate_id)
    if not path.is_file() or file_sha256(path) != entry.file_sha256:
        return None, _failure(
            "surrogate_file_changed",
            f"the file of surrogate {surrogate_id!r} ({entry.file}) is missing or "
            "no longer hashes to its registry row, so it is not the model that "
            "was registered and is not used",
        )
    if entry.recipe["dataset_id"] != dataset_id:
        return None, _failure(
            "surrogate_binding_mismatch",
            f"surrogate {surrogate_id!r} was trained on {entry.recipe['dataset_id']!r} "
            f"and is bound to it; an autograd measurement on {dataset_id!r} (an "
            "input sampled from it, or any other) is refused. Measure that input "
            "with method='finite_diff' or 'weak'",
        )
    refusal = _run_precision_failure(
        session,
        surrogate_id,
        entry,
        "an autograd measurement differentiates only a model trained in the "
        "run's precision. Train one with train_surrogate under this run (it "
        "takes the run's precision) and pass that id, or measure with "
        "method='finite_diff' or 'weak'",
    )
    if refusal is not None:
        return None, refusal
    return {"path": str(path), "sha256": entry.file_sha256}, None


def _task_failure(task: IsolatedTask, verb: str) -> dict[str, Any]:
    """A failed worker task as one failure: a stop at the hard limit is
    `<verb>_timeout` (unmeasured, never an infinite error), anything else
    keeps the worker's code."""
    if task.error_code == "SearchTimeout":
        return _failure(f"{verb}_timeout", str(task.error_message))
    return _failure(str(task.error_code), str(task.error_message))


def _remaining_cap(session: Session) -> float | None:
    """The cap of a short worker call: the run's remaining time with a
    one-minute floor, or none without a budget."""
    remaining = session.budget()["remaining_seconds"]
    return None if remaining is None else max(60.0, float(remaining))


def _positive_seconds(name: str, value: Any) -> dict[str, Any] | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        return _failure(
            "invalid_argument",
            f"{name} has to be a positive number of seconds, got {value!r}",
        )
    return None


def _register_evidence(
    session: Session,
    kind: str,
    *,
    run_ids: list[str],
    inputs: list[str],
    terms: list[str],
    result: dict[str, Any],
) -> str:
    """Record one citable result and return its `evidence_id`.

    The record names the searches and inputs it was made on, their common
    root, the terms it covers with kd's law-term keys (what `make_sketch`
    checks anchors against) and the result itself. The reporting copy is
    appended first; the session record, which nothing in the workspace can
    change, is the authority.
    """
    evidence_id = _new_id("evidence")
    record = {
        "format": EVIDENCE_FORMAT,
        "evidence_id": evidence_id,
        "kind": kind,
        "run_ids": list(run_ids),
        "inputs": sorted(set(inputs)),
        "root": root_ref(inputs[0]),
        "terms": list(terms),
        "law_keys": _law_keys(list(terms)),
        "result": result,
    }
    _append_record(session, EVIDENCE_FILENAME, record)
    session.evidence[evidence_id] = deepcopy(record)
    return evidence_id


def _support_origin(session: Session, run_id: str, support: list[str]) -> str:
    """`"search-derived"` when every term a refinement was given is a term of
    the source search's sealed law, so the support came from that search
    (pruned or not); `"agent-proposed"` when it holds a term the search did
    not find, whatever search the call cites. Terms are compared by
    spelling, then by kd's law-term identity; a term with a numeric constant
    inside (a table's) has no law-term key, so only its spelling places it.
    """
    found = list(session.sealed_laws[run_id]["support"])
    if set(support) <= set(found):
        return "search-derived"
    try:
        derived = set(_law_keys(support)) <= set(_law_keys(found))
    except ValueError:
        derived = False
    return "search-derived" if derived else "agent-proposed"


def _record_receipt(
    session: Session,
    *,
    run_id: str,
    law: dict[str, Any],
    support: list[str],
    method: str,
    parameters: dict[str, Any],
    evaluation: dict[str, Any],
    measurement: dict[str, Any],
    output_record: dict[str, Any] | None = None,
    artifact_sha256: str | None = None,
) -> str:
    """Store the immutable receipt of one successful refinement and return
    its `refinement_id`.

    The receipt names the operation that produced the coefficients: the
    exact output law and its identity, the source search (the parent
    reference and the contextual citation, which are the same search here),
    whether the support it was given (`support`) came from that search, the
    method and the call's
    parameters, the search's and the measurement's input records (and the
    input a PINN fit wrote), the measurement frame and metrics, and the
    hash of the file the coefficients came out of. The reporting copy goes
    to `<workspace>/refinements.jsonl` first; the session copy is the
    authority and is never rewritten.
    """
    refinement_id = _new_id("refinement")
    receipt = {
        "format": RECEIPT_FORMAT,
        "refinement_id": refinement_id,
        "parent_ref": {"kind": "search", "run_id": run_id},
        "source_run_id": run_id,
        "support_origin": _support_origin(session, run_id, support),
        "law": deepcopy(law),
        "law_sha256": _canonical_sha256(law),
        "method": method,
        "parameters": deepcopy(parameters),
        "search_input_record": deepcopy(session.search_inputs[run_id]),
        "evaluation_input_record": deepcopy(evaluation),
        "output_input_record": deepcopy(output_record),
        "measurement": deepcopy(measurement),
        "artifact_sha256": artifact_sha256,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _append_record(session, RECEIPTS_FILENAME, receipt)
    session.receipts[refinement_id] = receipt
    return refinement_id


def _producer(
    session: Session, law: dict[str, Any], run_id: str, refinement_id: str | None
) -> tuple[dict[str, Any] | None, float | None, dict[str, Any] | None]:
    """The operation a submitted law's coefficients come from, its
    measurement, or a refusal.

    An explicit `refinement_id` must name a receipt of this run whose source
    search is `run_id` and whose law equals `law` exactly. Without one, the
    latest such receipt produces it; otherwise the search does, when `law`
    is exactly its sealed law; otherwise the law is edited and borrows no
    measurement. No approximate comparison transfers a measurement.
    """
    if refinement_id is not None:
        receipt = session.receipts.get(refinement_id)
        if receipt is None:
            return (
                None,
                None,
                _failure(
                    "unknown_refinement_id",
                    f"refinement_id {refinement_id!r} is not a refinement of this run; "
                    f"the recorded ones are {list(session.receipts)}",
                ),
            )
        if receipt["source_run_id"] != run_id:
            return (
                None,
                None,
                _failure(
                    "refinement_source_mismatch",
                    f"refinement {refinement_id!r} refined search "
                    f"{receipt['source_run_id']!r}, not {run_id!r}; submit it with "
                    "that run_id",
                ),
            )
        if receipt["law"] != law:
            return (
                None,
                None,
                _failure(
                    "refinement_law_mismatch",
                    f"law is not exactly the law refinement {refinement_id!r} "
                    "produced; pass its primary.law unchanged, or omit "
                    "refinement_id to submit an edited law",
                ),
            )
        return (
            {"kind": "refinement", "refinement_id": refinement_id},
            receipt["measurement"]["nmse"],
            None,
        )
    matching = [
        receipt
        for receipt in session.receipts.values()
        if receipt["source_run_id"] == run_id and receipt["law"] == law
    ]
    if matching:
        latest = matching[-1]
        return (
            {"kind": "refinement", "refinement_id": latest["refinement_id"]},
            latest["measurement"]["nmse"],
            None,
        )
    if session.sealed_laws[run_id] == law:
        return (
            {"kind": "search", "run_id": run_id},
            session.successful_runs[run_id]["nmse"],
            None,
        )
    return {"kind": "edited"}, None, None


def _verify_refusal(verify: dict[str, Any] | None) -> dict[str, Any] | None:
    """Why `verify` is not a verification threshold; None when it is (or is
    omitted). Exactly `{"nmse_max": <finite number >= 0>}`: the report-only
    opt-out and kd's other policy fields are not offered."""
    if verify is None:
        return None
    value = verify.get("nmse_max") if set(verify) == {"nmse_max"} else None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not (math.isfinite(value) and value >= 0)
    ):
        return _failure(
            "invalid_argument",
            'verify has to be exactly {"nmse_max": <a finite number >= 0>}, '
            f"got {verify!r}; omit it for kd's default threshold "
            f"{kd.SKETCH_EXIT_VERIFY.nmse_max}",
        )
    return None


def _threshold(verify: dict[str, Any] | None) -> float:
    """The residual threshold a search runs under: the explicit one, or kd's
    default sketch-exit threshold."""
    default = cast(float, kd.SKETCH_EXIT_VERIFY.nmse_max)
    return default if verify is None else float(verify["nmse_max"])


_ACTIVATING_ORIGINS: Final = ("user", "conversation")
"""The registrations that make their Sketch the active one: the run's
Sketch file and `set_sketch`. A `make_sketch` registration does not."""


def _bound_sketch_task(
    session: Session, dataset_id: str, form: dict[str, Any]
) -> IsolatedTask:
    """Bind a Sketch on `dataset_id` in a worker (`SKETCH_TASK` with `form`:
    a Sketch file's content as `payload`, or `set_sketch`'s shorthand terms
    as `clauses`): the value is kd's complete `kd-sketch-v2` payload on the
    input the worker loaded."""
    return run_isolated_task(
        {
            TASK_KEY: SKETCH_TASK,
            "dataset_id": dataset_id,
            "input_source": _resolve(session, dataset_id),
            **form,
        },
        time_cap_seconds=_remaining_cap(session),
        cooperative=False,
    )


def _register_sketch(
    session: Session, payload: dict[str, Any], registration: dict[str, Any]
) -> dict[str, Any]:
    """Register one more arrival of a bound Sketch payload and return what
    its envelope echoes: the Sketch's `sketch_id`, `sketch` and
    `sketch_sha256` beside the fields of this registration.

    One payload is one record (`lineage.sketch_identity`), and every
    registration of it is appended to the record's `registrations`, so the
    origin, evidence, input and reason of one arrival are never merged with
    or replaced by those of another arrival of the same payload."""
    sketch_id, digest = sketch_identity(payload)
    record = session.sketches.setdefault(
        sketch_id,
        {
            "sketch_id": sketch_id,
            "sketch": payload,
            "sketch_sha256": digest,
            "registrations": [],
        },
    )
    record["registrations"].append(registration)
    return {
        "sketch_id": sketch_id,
        "sketch": deepcopy(record["sketch"]),
        "sketch_sha256": digest,
        **deepcopy(registration),
    }


def register_user_sketch(session: Session, payload: dict[str, Any]) -> dict[str, Any]:
    """Bind a user's Sketch file content to the run's pinned input and
    register it as this run's user Sketch, which is also the active one;
    return the Sketch (`sketch_id`, the bound `sketch`, `sketch_sha256`) with
    its registration (`origin="user"`, no `evidence_ids`, `input_id`,
    `reason` None, `created_at`).

    The run must be pinned to an input. The payload is what `kd.save_sketch`
    writes, a `kd-sketch-spec-v1` shorthand or a complete `kd-sketch-v2`
    Sketch (a released `kd-sketch-v1` one is read too); a worker binds it to
    the pinned input with `kd.load_sketch(payload, dataset=)`, so a shorthand
    is resolved on the data and a complete Sketch whose left-hand side or
    vocabulary does not fit the input is refused. What is registered is the
    bound `kd-sketch-v2` payload, and its digest is over canonical JSON: a
    shorthand file and the complete file kd resolves it to register one
    Sketch. A search that names no Sketch then runs under it, a search
    naming another one is refused, and `set_sketch` cannot replace it.
    Raises ValueError for an unpinned run or a payload the worker refuses.
    """
    pinned = session.pinned_dataset_id
    if pinned is None:
        raise ValueError(
            "a user Sketch requires a pinned input: it constrains every search "
            "of the run on one input's left-hand side and vocabulary"
        )
    task = _bound_sketch_task(session, pinned, {"payload": payload})
    if task.value is None:
        raise ValueError(
            f"the user Sketch cannot be bound to input {pinned!r}: "
            f"{task.error_code}: {task.error_message}"
        )
    primary = _register_sketch(
        session,
        task.value["sketch"],
        {
            "origin": "user",
            "evidence_ids": [],
            "input_id": pinned,
            "reason": None,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    session.user_sketch_id = session.active_sketch_id = primary["sketch_id"]
    return primary


def _requested_sketch(
    session: Session,
    sketch: dict[str, Any] | None,
    sketch_id: str | None,
    dataset_id: str,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """The Sketch a search on `dataset_id` is asked to run under, its
    registered id, or a refusal: a registered `sketch_id`, a raw payload,
    or (neither given) the active Sketch.

    A raw payload is taken as kd's canonical form of it (decoded and
    encoded again, so a released `kd-sketch-v1` payload is a `kd-sketch-v2`
    one from here on: that form is checked, launched and recorded, and the
    search's worker binds it to the input it loads); one kd cannot decode
    is `invalid_sketch`. Then the active Sketch's precedence
    (`_active_sketch_refusal`): a Sketch that differs from it is refused,
    not merged, and so is a search on an input of another root."""
    if sketch is not None and sketch_id is not None:
        return (
            None,
            None,
            _failure("invalid_argument", "pass sketch_id or sketch, not both"),
        )
    if sketch_id is not None and sketch_id not in session.sketches:
        return (
            None,
            None,
            _failure(
                "unknown_sketch_id",
                f"sketch_id {sketch_id!r} was not registered this run; the registered "
                f"ones are {list(session.sketches)}. make_sketch and set_sketch "
                "return one",
            ),
        )
    if sketch_id is not None:
        requested, requested_id = session.sketches[sketch_id]["sketch"], sketch_id
    elif sketch is not None:
        try:
            requested = dict(kd.sketch_to_dict(kd.sketch_from_dict(sketch)))
        except ValueError as exc:
            return None, None, _failure("invalid_sketch", _undecodable(exc))
        requested_id = None
    elif session.active_sketch_id is not None:
        active = session.sketches[session.active_sketch_id]
        requested, requested_id = active["sketch"], active["sketch_id"]
    else:
        return None, None, None
    refusal = _active_sketch_refusal(session, requested, dataset_id)
    if refusal is not None:
        return None, None, refusal
    return deepcopy(requested), requested_id, None


def _undecodable(exc: ValueError) -> str:
    """The detail of a `run_discovery(sketch=)` payload kd cannot decode."""
    return (
        f"sketch is not a Sketch payload kd can decode: {exc}. It takes a "
        "complete kd-sketch-v2 payload (a released kd-sketch-v1 one is read "
        "as well); shorthand terms are bound with set_sketch"
    )


def _active_sketch_refusal(
    session: Session, payload: dict[str, Any], dataset_id: str
) -> dict[str, Any] | None:
    """Why a search on `dataset_id` under `payload` breaks the precedence of
    the active Sketch, or None (no active Sketch, or `payload` is it on its
    input root).

    A different Sketch is `sketch_conflict`: the active one is replaced only
    by `set_sketch`, and the user's not at all. An input of another root is
    `sketch_input_mismatch`: a Sketch is bound to the input it was set on
    and the inputs derived from it (whose left-hand side and vocabulary kd's
    facade still checks at the fit), and on another input it applies only
    once `set_sketch(input_id=...)` binds one there."""
    active_id = session.active_sketch_id
    if active_id is None:
        return None
    if _differs_from_active(session, payload):
        return _failure(
            "sketch_conflict",
            _active_sketch_detail(session)
            + "; the Sketch this call names differs from it. Omit sketch_id and "
            "sketch to search under it",
        )
    bound_on = _active_input(session)
    if root_ref(dataset_id) != root_ref(bound_on):
        return _failure(
            "sketch_input_mismatch",
            f"the active Sketch {active_id} is bound to input {bound_on!r}, and "
            f"{dataset_id!r} is neither that input nor one derived from it. To "
            f"search {dataset_id!r} under a Sketch, bind one on it with "
            f"set_sketch(input_id={dataset_id!r}, ...), which makes it the "
            "active Sketch",
        )
    return None


def _active_sketch_detail(session: Session) -> str:
    """What the active Sketch is and how it is replaced, for a refusal."""
    active_id = session.active_sketch_id
    if active_id == session.user_sketch_id:
        return (
            f"this run's user Sketch ({active_id}) constrains every search, and "
            "set_sketch cannot replace it"
        )
    return (
        f"the active Sketch ({active_id}) constrains every search that names "
        "none, and set_sketch replaces it: to search under another Sketch, "
        "make that one active with set_sketch first"
    )


def _active_input(session: Session) -> str:
    """The input the active Sketch was bound on: that of the registration
    that made it active (the user's Sketch file or the last `set_sketch`)."""
    record = session.sketches[cast(str, session.active_sketch_id)]
    return cast(
        str,
        next(
            entry["input_id"]
            for entry in reversed(record["registrations"])
            if entry["origin"] in _ACTIVATING_ORIGINS
        ),
    )


def _differs_from_active(session: Session, payload: dict[str, Any] | None) -> bool:
    """Whether a search under `payload` (None: no Sketch) leaves this run's
    active Sketch unmet: a Sketch is active and `payload` is none or
    another one."""
    if session.active_sketch_id is None:
        return False
    if payload is None:
        return True
    active = session.sketches[session.active_sketch_id]
    return _sketch_digest(payload) != cast(str, active["sketch_sha256"])


def _sketch_digest(payload: dict[str, Any]) -> str:
    """The digest of kd's canonical form of a payload kd decodes (a call's
    payload once `_requested_sketch` took it, or a segment row's, decoded
    when the row was recorded)."""
    return sketch_identity(dict(kd.sketch_to_dict(kd.sketch_from_dict(payload))))[1]


def _vote(
    session: Session, run_ids: list[str], min_count: int, small_term_rtol: float
) -> dict[str, Any]:
    """`vote_terms`: count the law terms of searches this run recorded.

    Each search must have succeeded this run, and its active law is read off
    the equation its record sealed (`Session.sealed_fits`, kd's IR), and sit
    on the policy tree, whose rows hold the lineage; a repeated id, two
    segments of one
    lineage (one root on the tree's resume edges) and searches of two roots
    are refused, so every vote is an independent search of one input.
    Coefficients are passed to kd when every law has a number for every
    term, so kd leaves out exact zeros and small terms; a positive cutoff
    with an unknown coefficient is refused.
    """
    provenance = {
        "run_ids": list(run_ids),
        "min_count": min_count,
        "small_term_rtol": small_term_rtol,
    }

    def refused(failure: dict[str, Any]) -> dict[str, Any]:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[failure],
        )

    repeated = sorted({run for run in run_ids if run_ids.count(run) > 1})
    if repeated:
        return refused(
            _failure(
                "invalid_argument",
                f"run_ids {repeated} are listed more than once; one search is one vote",
            )
        )
    unknown = [run for run in run_ids if run not in session.successful_runs]
    if unknown:
        return refused(
            _failure(
                "unknown_run_id",
                f"run_ids {unknown} are not searches that succeeded this run; the "
                f"usable ones are {sorted(session.successful_runs)}",
            )
        )
    entries = {entry.run_id: entry for entry in session.ledger.segments()}
    missing = [run for run in run_ids if run not in entries]
    if missing:
        return refused(
            _failure(
                "ledger_unavailable",
                f"searches {missing} are not on the policy tree, whose rows hold "
                "the lineage a vote checks",
            )
        )
    lineages = [session.ledger.chain(run)[0].run_id for run in run_ids]
    shared = sorted(
        {
            run
            for run, root in zip(run_ids, lineages, strict=True)
            if lineages.count(root) > 1
        }
    )
    if shared:
        return refused(
            _failure(
                "invalid_argument",
                f"run_ids {shared} are segments of one lineage (resumed from one "
                "another), which are not independent votes; keep one of them",
            )
        )
    inputs = [session.successful_runs[run]["dataset"] for run in run_ids]
    if len({root_ref(dataset) for dataset in inputs}) > 1:
        return refused(
            _failure(
                "measurement_input_mismatch",
                f"these searches ran on inputs of different roots ({sorted(set(inputs))}); "
                "a vote counts the laws of one input and its derived inputs",
            )
        )
    laws = [_equation_law(session.sealed_fits[run]) for run in run_ids]
    known = all(
        law.get("coefficients") is not None
        and all(value is not None for value in law["coefficients"])
        for law in laws
    )
    if small_term_rtol != 0 and not known:
        return refused(
            _failure(
                "invalid_argument",
                "small_term_rtol judges terms by their coefficients, and a law in "
                "this vote has an unknown one; pass small_term_rtol=0",
            )
        )
    try:
        result = kd.vote_terms(
            [list(law["support"]) for law in laws],
            min_count=min_count,
            coefficients=[list(law["coefficients"]) for law in laws] if known else None,
            small_term_rtol=small_term_rtol,
        )
    except ValueError as exc:
        return refused(_failure("invalid_argument", str(exc)))
    rows = [
        {
            "law_key": row.law_key,
            "term": row.term,
            "count": row.count,
            "run_ids": [run_ids[index] for index in row.support_indices],
            "seeds": [entries[run_ids[index]].seed for index in row.support_indices],
            "lineages": [lineages[index] for index in row.support_indices],
        }
        for row in result.rows
    ]
    primary = {
        "selected": list(result.selected),
        "rows": rows,
        "min_count": result.min_count,
        "n_runs": result.n_supports,
    }
    try:
        evidence_id = _register_evidence(
            session,
            "vote",
            run_ids=list(run_ids),
            inputs=inputs,
            terms=[row.term for row in result.rows],
            result=primary,
        )
    except Exception as exc:
        return refused(_crashed(exc))
    return envelope(
        "success",
        budget=session.budget(),
        primary={"evidence_id": evidence_id, **primary},
        provenance={**provenance, "inputs": sorted(set(inputs))},
    )


def _sketch_target(
    session: Session, records: list[dict[str, Any]], input_id: str | None
) -> tuple[str | None, dict[str, Any] | None]:
    """The input a Sketch from these evidence records is built on: the one
    input they were all measured on, or the explicit `input_id` (required
    when they span several inputs), which must be of their common root."""
    roots = sorted({record["root"] for record in records})
    if len(roots) > 1:
        return None, _failure(
            "evidence_root_mismatch",
            f"the cited evidence comes from inputs of different roots {roots}; "
            "a Sketch is built for one input and its derived inputs",
        )
    inputs = sorted({dataset for record in records for dataset in record["inputs"]})
    if input_id is None:
        if len(inputs) != 1:
            return None, _failure(
                "ambiguous_input",
                f"the cited evidence was measured on {inputs}; pass input_id to "
                "say which input the Sketch is built and checked on",
            )
        return inputs[0], None
    if root_ref(input_id) != roots[0]:
        return None, _failure(
            "measurement_input_mismatch",
            f"input_id {input_id!r} is not of the root {roots[0]!r} the cited "
            "evidence was measured on",
        )
    refusal = _input_refusal(session, input_id)
    return (None, refusal) if refusal is not None else (input_id, None)


def _make_sketch(
    session: Session,
    *,
    evidence_ids: list[str],
    anchors: list[str],
    bounds: dict[str, int],
    input_id: str | None,
    reason: str | None,
) -> dict[str, Any]:
    """`make_sketch`: resolve the evidence, check the anchors against it,
    have kd build the Sketch on the chosen input in a worker, register it
    (origin `agent`, with the evidence, the input and the reason). It is not
    made the active Sketch."""
    provenance = {
        "evidence_ids": list(evidence_ids),
        "anchor_term_keys": list(anchors),
        **bounds,
        "input_id": input_id,
    }

    def refused(failure: dict[str, Any]) -> dict[str, Any]:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[failure],
        )

    unknown = [eid for eid in evidence_ids if eid not in session.evidence]
    if unknown or not evidence_ids:
        return refused(
            _failure(
                "unknown_evidence_id",
                f"evidence ids {unknown} were not returned by a vote or a "
                f"measurement this run; the recorded ones are {list(session.evidence)}",
            )
        )
    records = [session.evidence[eid] for eid in evidence_ids]
    target, refusal = _sketch_target(session, records, input_id)
    if refusal is not None:
        return refused(refusal)
    try:
        keys = _law_keys(list(anchors))
    except ValueError as exc:
        return refused(_failure("invalid_term", str(exc)))
    cited = {key for record in records for key in record["law_keys"]}
    absent = [
        anchor for anchor, key in zip(anchors, keys, strict=True) if key not in cited
    ]
    if absent:
        return refused(
            _failure(
                "anchor_not_in_evidence",
                f"anchors {absent} occur in none of the cited evidence {evidence_ids}; "
                "an anchor has to be a term the cited votes or measurements cover",
            )
        )
    target_id = cast(str, target)
    try:
        session.workers_started += 1
        task = run_isolated_task(
            {
                TASK_KEY: SKETCH_TASK,
                "dataset_id": target_id,
                "input_source": _resolve(session, target_id),
                "anchors": list(anchors),
                **bounds,
            },
            time_cap_seconds=_remaining_cap(session),
            cooperative=False,
        )
    except Exception as exc:
        return refused(_crashed(exc))
    if task.value is None:
        return refused(_task_failure(task, "make_sketch"))
    primary = _register_sketch(
        session,
        task.value["sketch"],
        {
            "origin": "agent",
            "evidence_ids": list(evidence_ids),
            "input_id": target_id,
            "reason": reason,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    return envelope(
        "success",
        budget=session.budget(),
        primary=primary,
        diagnostics={"wallclock_seconds": round(task.wallclock_seconds, 3)},
        provenance=provenance,
    )


def _prior_target(
    session: Session, input_id: str | None
) -> tuple[str | None, dict[str, Any] | None]:
    """The input `set_sketch` binds on, or a refusal: `input_id`, which on a
    pinned run has to be the pinned input or one derived from it, or, when
    it is omitted, the pinned input; an unpinned run has to name it."""
    pinned = session.pinned_dataset_id
    if input_id is None:
        if pinned is None:
            return None, _failure(
                "invalid_argument",
                "this run is not pinned to an input, so set_sketch needs "
                "input_id: the input whose left-hand side and vocabulary the "
                "Sketch is bound to (list_datasets names them); searches on it "
                "and on the inputs derived from it then run under the Sketch",
            )
        return pinned, None
    if pinned is not None and not _same_input(input_id, pinned):
        return None, _failure(
            "dataset_not_pinned",
            f"this run is pinned to dataset {pinned!r}; input_id has to be it "
            "or an input smooth_input derived from it",
        )
    refusal = _input_refusal(session, input_id)
    return (None, refusal) if refusal is not None else (input_id, None)


def _user_sketch_conflict(
    session: Session, payload: dict[str, Any]
) -> dict[str, Any] | None:
    """The refusal of a `set_sketch` whose bound payload is not this run's
    user Sketch, or None (no user Sketch, or that one)."""
    user = session.user_sketch_id
    if user is None or sketch_identity(payload)[0] == user:
        return None
    return _failure(
        "sketch_conflict",
        _active_sketch_detail(session)
        + "; the Sketch these terms bind to differs from it",
    )


def _set_sketch(
    session: Session,
    *,
    clauses: dict[str, Any],
    input_id: str | None,
    reason: str | None,
) -> dict[str, Any]:
    """`set_sketch`: have kd bind the shorthand terms on the chosen input in
    a worker, register the bound Sketch (origin `conversation`, with the
    input and the reason) and make it the active Sketch. Binding that fails,
    or a Sketch other than the user's on a run that has one, leaves the
    active Sketch as it was."""
    provenance = {**deepcopy(clauses), "input_id": input_id}

    def refused(failure: dict[str, Any]) -> dict[str, Any]:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[failure],
        )

    target, refusal = _prior_target(session, input_id)
    if refusal is not None:
        return refused(refusal)
    target_id = cast(str, target)
    try:
        session.workers_started += 1
        task = _bound_sketch_task(session, target_id, {"clauses": clauses})
    except Exception as exc:
        return refused(_crashed(exc))
    if task.value is None:
        return refused(_task_failure(task, "set_sketch"))
    payload = task.value["sketch"]
    conflict = _user_sketch_conflict(session, payload)
    if conflict is not None:
        return refused(conflict)
    primary = _register_sketch(
        session,
        payload,
        {
            "origin": "conversation",
            "evidence_ids": [],
            "input_id": target_id,
            "reason": reason,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    session.active_sketch_id = primary["sketch_id"]
    return envelope(
        "success",
        budget=session.budget(),
        primary=primary,
        diagnostics={"wallclock_seconds": round(task.wallclock_seconds, 3)},
        provenance=provenance,
    )


def _measurement_setup(
    session: Session,
    *,
    run_id: str,
    law: dict[str, Any] | None,
    input_id: str | None,
    max_seconds: Any,
    frame: tuple[str, Trim, dict[str, Any] | None, str | None] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """What every measurement of a search's law checks before a worker
    starts, in order: the frame (`frame` is `(method, trim, weak,
    surrogate_id)`, None for the verification, which has none), the cap,
    the search and the measurement input, the law, and the model an autograd
    frame differentiates. Returns the input, the law and the frame's worker
    keys, or the first refusal."""
    if frame is not None:
        refusal = _frame_refusal(*frame)
        if refusal is not None:
            return None, refusal
    refusal = _positive_seconds("max_seconds", max_seconds)
    if refusal is not None:
        return None, refusal
    measured, refusal = _measured_input(session, run_id, input_id)
    if refusal is not None:
        return None, refusal
    measured = cast(dict[str, Any], measured)
    candidate, refusal = _candidate_law(session, run_id, law)
    if refusal is not None:
        return None, refusal
    keys: dict[str, Any] = {}
    if frame is not None:
        method, trim, weak, surrogate_id = frame
        model = None
        if surrogate_id is not None:
            model, refusal = _model_binding(
                session, surrogate_id, measured["dataset_id"]
            )
            if refusal is not None:
                return None, refusal
        keys = frame_request(method=method, trim=trim, weak=weak, surrogate=model)
    lhs = measured["lhs"]
    return {
        **measured,
        "law": candidate,
        "frame_keys": keys,
        "lhs_order": 0 if lhs is None else lhs["order"],
    }, None


def _measurement_request(
    task: str, setup: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    request: dict[str, Any] = {
        TASK_KEY: task,
        "dataset_id": setup["dataset_id"],
        "lhs_order": setup["lhs_order"],
        **setup["frame_keys"],
        **extra,
    }
    if setup["input_source"] is not None:
        request["input_source"] = setup["input_source"]
    return request


def _ablate_candidate(
    session: Session,
    *,
    run_id: str,
    law: dict[str, Any] | None,
    input_id: str | None,
    frame: tuple[str, Trim, dict[str, Any] | None, str | None],
    max_seconds: Any,
) -> dict[str, Any]:
    """`ablate_candidate`: one worker, one `kd.ablate_terms` call, one frame."""
    method, trim, weak, surrogate_id = frame
    provenance: dict[str, Any] = {
        "run_id": run_id,
        "input_id": input_id,
        "method": method,
        "trim": trim,
        "weak": weak,
        "surrogate_id": surrogate_id,
        "max_seconds": max_seconds,
    }

    def refused(failure: dict[str, Any]) -> dict[str, Any]:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[failure],
        )

    try:
        setup, refusal = _measurement_setup(
            session,
            run_id=run_id,
            law=law,
            input_id=input_id,
            max_seconds=max_seconds,
            frame=frame,
        )
        if refusal is not None:
            return refused(refusal)
        setup = cast(dict[str, Any], setup)
        provenance["input"] = setup["record"]
        terms = list(setup["law"]["support"])
        session.workers_started += 1
        task = run_isolated_task(
            _measurement_request(ABLATE_TASK, setup, terms=terms),
            time_cap_seconds=float(max_seconds),
            cooperative=False,
        )
        if task.value is None:
            return refused(_task_failure(task, "ablate"))
        value = finite_tree(task.value)
        baseline = value["baseline"]
        frame_record, weak_record = baseline.pop("frame"), baseline.pop("weak")
        result = {
            "law": setup["law"],
            "baseline": baseline,
            "rows": value["rows"],
            "relative_rise_reason": value["relative_rise_reason"],
        }
        evidence_id = _register_evidence(
            session,
            "ablation",
            run_ids=[run_id],
            inputs=[setup["dataset_id"]],
            terms=terms,
            result={**result, "frame": frame_record, "weak": weak_record},
        )
    except Exception as exc:
        return refused(_crashed(exc))
    return envelope(
        "success",
        budget=session.budget(),
        primary={"evidence_id": evidence_id, **result},
        diagnostics={
            "frame": frame_record,
            "weak": weak_record,
            "wallclock_seconds": round(task.wallclock_seconds, 3),
        },
        provenance=provenance,
    )


def _score_term_sets(
    session: Session,
    *,
    run_id: str,
    term_sets: list[list[str]],
    input_id: str | None,
    frame: tuple[str, Trim, dict[str, Any] | None, str | None],
    max_seconds: Any,
) -> dict[str, Any]:
    """`score_term_sets`: one worker, one `kd.score_term_sets` call, one
    frame for the union of the sets."""
    method, trim, weak, surrogate_id = frame
    provenance: dict[str, Any] = {
        "run_id": run_id,
        "input_id": input_id,
        "method": method,
        "trim": trim,
        "weak": weak,
        "surrogate_id": surrogate_id,
        "max_seconds": max_seconds,
    }

    def refused(failure: dict[str, Any]) -> dict[str, Any]:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[failure],
        )

    try:
        setup, refusal = _measurement_setup(
            session,
            run_id=run_id,
            law=None,
            input_id=input_id,
            max_seconds=max_seconds,
            frame=frame,
        )
        if refusal is not None:
            return refused(refusal)
        setup = cast(dict[str, Any], setup)
        provenance["input"] = setup["record"]
        session.workers_started += 1
        task = run_isolated_task(
            _measurement_request(
                SCORE_TASK, setup, term_sets=[list(terms) for terms in term_sets]
            ),
            time_cap_seconds=float(max_seconds),
            cooperative=False,
        )
        if task.value is None:
            return refused(_task_failure(task, "score_term_sets"))
        value = finite_tree(task.value)
        entries = value["entries"]


        for entry in entries:
            if entry["fit"] is not None:
                entry["fit"].pop("frame")
        fitted = [
            term
            for entry in entries
            if entry["fit"] is not None
            for term in entry["terms"]
        ]
        evidence_id = _register_evidence(
            session,
            "term_sets",
            run_ids=[run_id],
            inputs=[setup["dataset_id"]],
            terms=list(dict.fromkeys(fitted)),
            result={"entries": entries, "frame": value["frame"]},
        )
    except Exception as exc:
        return refused(_crashed(exc))
    return envelope(
        "success",
        budget=session.budget(),
        primary={"evidence_id": evidence_id, "entries": entries},
        diagnostics={
            "frame": value["frame"],
            "wallclock_seconds": round(task.wallclock_seconds, 3),
        },
        provenance=provenance,
    )


def _verify_candidate(
    session: Session,
    *,
    run_id: str,
    law: dict[str, Any] | None,
    input_id: str | None,
    max_seconds: Any,
) -> dict[str, Any]:
    """`verify_candidate`: three separate measurements in one worker, under
    kd's cooperative forward deadline and the worker's hard limit."""
    provenance: dict[str, Any] = {
        "run_id": run_id,
        "input_id": input_id,
        "max_seconds": max_seconds,
    }

    def refused(failure: dict[str, Any]) -> dict[str, Any]:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[failure],
        )

    try:
        setup, refusal = _measurement_setup(
            session,
            run_id=run_id,
            law=law,
            input_id=input_id,
            max_seconds=max_seconds,
            frame=None,
        )
        if refusal is not None:
            return refused(refusal)
        setup = cast(dict[str, Any], setup)
        provenance["input"] = setup["record"]
        candidate = setup["law"]
        coefficients = candidate.get("coefficients")
        if coefficients is None or any(value is None for value in coefficients):
            return refused(
                _failure(
                    "invalid_argument",
                    "a forward check integrates the law with its coefficients, and "
                    "this law has an unknown one; pass a law with a number for every term",
                )
            )
        session.workers_started += 1
        task = run_isolated_task(
            _measurement_request(
                VERIFY_TASK,
                setup,
                terms=list(candidate["support"]),
                coefficients=[float(value) for value in coefficients],
                lhs=setup["lhs"],
                max_seconds=float(max_seconds),
            ),
            time_cap_seconds=float(max_seconds),
            cooperative=True,
        )
        if task.value is None:
            return refused(_task_failure(task, "verify"))
        value = finite_tree(task.value)
        result = {
            "law": candidate,
            "validation": value["validation"],
            "dispersion": value["dispersion"],
            "forward": value["forward"],
        }
        evidence_id = _register_evidence(
            session,
            "verification",
            run_ids=[run_id],
            inputs=[setup["dataset_id"]],
            terms=list(candidate["support"]),
            result=result,
        )
    except Exception as exc:
        return refused(_crashed(exc))
    failures = [
        _failure(item["code"], f"{item['measurement']}: {item['message']}")
        for item in value["failures"]
    ]
    return envelope(
        "partial" if failures else "success",
        budget=session.budget(),
        primary={"evidence_id": evidence_id, **result},
        diagnostics={
            "forward_max_seconds": value["forward_max_seconds"],
            "wallclock_seconds": round(task.wallclock_seconds, 3),
        },
        provenance=provenance,
        failures=failures,
    )


def _raw_target(session: Session, dataset_id: str) -> tuple[str, str] | None:
    """The raw fingerprint and the regression target (`dx_dt`) of an input
    that projects one state of raw ODE trajectories, or None for any other
    input, and for one whose description no longer resolves (the
    `_same_raw_set` posture: it is then no projection the systems answer
    for)."""
    if not is_file_ref(dataset_id) or is_derived(dataset_id):
        return None
    if dataset_id == session.pinned_dataset_id:


        description = session.input_source
    else:
        try:
            description = _resolve(session, dataset_id)
        except (ValueError, KeyError, TypeError, OSError):
            return None
    fingerprint = raw_fingerprint(description)
    if fingerprint is None:
        return None
    projection = cast(dict[str, Any], description)["projection"]
    return fingerprint, kd.derivative_target(projection["target_state"])


def _answer_system(ledger: SegmentLedger, system_id: str) -> dict[str, Any]:
    """The `system` an answer carries: the system's id, members and laws,
    and the verification its current result stands on (None when every
    verification of it timed out or failed)."""
    system = ledger.system(system_id)
    standing = standing_verification(system)
    return {
        "system_id": system_id,
        "members": system["members"],
        "laws": system["laws"],
        "verification_id": None if standing is None else standing["verification_id"],
    }


def _system_input(
    session: Session, input_id: str | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The input a system is verified on, or a refusal: the pinned input
    (an explicit `input_id` must be it), or on an unpinned run the
    `input_id` given, which has to project a state of raw trajectories.
    Returns its id, description, raw fingerprint and the regression target
    of every state in state order."""
    pinned = session.pinned_dataset_id
    if input_id is None and pinned is None:
        return None, _failure(
            "invalid_argument",
            "this run is pinned to no input, so input_id has to name the input "
            "the system is verified on: a dataset_id one of the members "
            "searched, whose raw trajectories are integrated and whose target "
            "the answer is about",
        )
    dataset_id = cast(str, input_id if input_id is not None else pinned)
    if pinned is not None and dataset_id != pinned:
        return None, _failure(
            "dataset_not_pinned",
            f"input_id {dataset_id!r} is not the input this run is pinned to "
            f"({pinned!r}); omit input_id to verify on it",
        )
    description = None
    if is_file_ref(dataset_id) and not is_derived(dataset_id):
        try:
            description = _resolve(session, dataset_id)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return None, _failure(
                "unknown_dataset",
                f"{dataset_id!r} is not an input of this workspace that "
                f"verifies: {exc}",
            )
    fingerprint = raw_fingerprint(description)
    if fingerprint is None:
        return None, _failure(
            "raw_trajectories_unavailable",
            f"{dataset_id!r} holds no raw ODE trajectories (a table, a grid or "
            "another input), so there is no time series to integrate the laws "
            "along; verify_system needs an input prepared from a trajectory "
            "file",
        )
    description = cast(dict[str, Any], description)
    return {
        "input_id": dataset_id,
        "description": description,
        "raw_fingerprint": fingerprint,
        "targets": [
            kd.derivative_target(state)
            for state in description["load_options"]["fields"]
        ],
    }, None


def _member_refusal(
    session: Session, target: str, run_id: str, fingerprint: str
) -> dict[str, Any] | None:
    """Why search `run_id` cannot be the law for `target` of a system on the
    raw trajectories `fingerprint` names, or None: it must be on the tree
    and not pruned, a search that succeeded this run on the input it was
    launched on, a search of a projection of those trajectories, sealed on
    `target` as its regression left-hand side, with every coefficient
    known."""
    if session.ledger.is_pruned(run_id):
        return _failure(
            "pruned_run_id",
            f"member {run_id!r} was pruned from the policy tree (itself or an "
            "ancestor of it), so a system with it is not a candidate",
        )
    if run_id not in session.successful_runs:
        if run_id in {entry.run_id for entry in session.ledger.segments()}:
            return _failure(
                "input_unverified",
                f"member {run_id!r} is a search an earlier run left on this "
                "workspace; this run has no record of the input it ran on, so "
                "search that state again in this run",
            )
        return _unknown_run_failure(session, run_id)
    refusal = _search_input_refusal(session, run_id)
    if refusal is not None:
        return refusal
    dataset_id = session.successful_runs[run_id]["dataset"]
    if raw_fingerprint(_resolve(session, dataset_id)) != fingerprint:
        return _failure(
            "measurement_input_mismatch",
            f"member {run_id!r} searched {dataset_id!r}, which is not a "
            "projection of the raw trajectories the system is verified on",
        )
    if session.sealed_lhs[run_id] != {"field": target, "axis": "", "order": 0}:
        return _failure(
            "invalid_argument",
            f"member {run_id!r} sealed the left-hand side "
            f"{json.dumps(session.sealed_lhs[run_id])}, not {target!r}; each key "
            "of members is the target its search fitted",
        )
    coefficients = session.sealed_laws[run_id].get("coefficients")
    if coefficients is None or any(value is None for value in coefficients):
        return _failure(
            "invalid_argument",
            f"member {run_id!r} sealed a law with an unknown coefficient, and the "
            "system is integrated with a number for every term",
        )
    return None


def _system_setup(
    session: Session, members: dict[str, str], input_id: str | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """What `verify_system` checks before a worker starts, in order: the
    input, the members' keys (every state's target once, no other key),
    each member (`_member_refusal`) and the run's time. Returns the input
    with the members in state order, or the refusal and its diagnostics.
    The missing-state check repeats kd's own refusal to integrate an
    incomplete system so that no worker starts for one."""
    setup, refusal = _system_input(session, input_id)
    if refusal is not None:
        return None, refusal, None
    setup = cast(dict[str, Any], setup)
    targets = setup["targets"]
    extra = sorted(set(members) - set(targets))
    if extra:
        return (
            None,
            _failure(
                "invalid_argument",
                f"members has the keys {extra}, which are no state's target; the "
                f"targets of these trajectories are {targets}",
            ),
            None,
        )
    missing = [target for target in targets if target not in members]
    if missing:
        detail = f"a system has one law per state; members gives none for {missing}"
        diagnostics = {"required_targets": targets, "missing_targets": missing}
        return None, _failure("incomplete_system", detail), diagnostics
    ordered = {target: members[target] for target in targets}
    for target, run_id in ordered.items():
        refusal = _member_refusal(session, target, run_id, setup["raw_fingerprint"])
        if refusal is not None:
            return None, refusal, None
    remaining = session.budget()["remaining_seconds"]
    if remaining is not None and remaining <= 0.0:
        return (
            None,
            _failure(
                "budget_exhausted",
                "this run has no time left, so no integration is started",
            ),
            None,
        )
    cap = (
        FORWARD_WORKER_SECONDS
        if remaining is None
        else min(FORWARD_WORKER_SECONDS, float(remaining))
    )
    return {**setup, "members": ordered, "cap": cap}, None, None


def _system_verification(
    session: Session, setup: dict[str, Any], task: IsolatedTask, reason: str | None
) -> dict[str, Any]:
    """The `system_verify` record of one attempt: the system it verifies,
    the outcome (`measured` when kd classified it, a finite value or
    `diverged`, `solve-failed` or `refused`; `not_measured` for a timeout,
    no time left after the load, or a failed worker), the budgets and the
    seconds."""
    members = setup["members"]
    value = task.value or {}
    forward = value.get("forward")
    measured = forward is not None and forward["reason"] != "timeout"
    return {
        "system_id": system_identity(setup["raw_fingerprint"], members),
        "verification_id": _new_id("verification"),
        "raw_fingerprint": setup["raw_fingerprint"],
        "input_id": setup["input_id"],
        "members": members,
        "laws": {
            target: deepcopy(session.sealed_laws[run_id])
            for target, run_id in members.items()
        },
        "frame": SYSTEM_FRAME,
        "status": "measured" if measured else "not_measured",
        "forward": forward,
        "error": None
        if task.value is not None
        else {"code": task.error_code, "message": task.error_message},
        "budget": {
            "worker_seconds": setup["cap"],
            "forward_seconds": value.get("forward_max_seconds"),
        },
        "seconds": {
            "load": value.get("load_seconds"),
            "forward": value.get("forward_seconds"),
            "worker": round(task.wallclock_seconds, 3),
        },
        "reason": reason,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _verify_system(
    session: Session,
    *,
    members: dict[str, str],
    input_id: str | None,
    reason: str | None,
) -> tuple[dict[str, Any], bool]:
    """`verify_system`: one worker integrates the members' sealed laws
    together on the raw trajectories (`episode_worker._verify_system`), and
    the attempt is recorded on the tree (`SegmentLedger.verify`). Returns
    the envelope and whether the tree now holds the attempt, which is when
    the answer channel is recomputed."""
    provenance: dict[str, Any] = {"members": dict(members), "input_id": input_id}
    setup, refusal, diagnostics = _system_setup(session, members, input_id)
    if refusal is not None:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            diagnostics=diagnostics,
            failures=[refusal],
        ), False
    setup = cast(dict[str, Any], setup)
    provenance["input_id"] = setup["input_id"]
    try:
        session.workers_started += 1
        task = run_isolated_task(
            {
                TASK_KEY: SYSTEM_TASK,
                "dataset_id": setup["input_id"],
                "input_source": setup["description"],
                "laws": {
                    target: [
                        session.sealed_laws[run_id]["support"],
                        session.sealed_laws[run_id]["coefficients"],
                    ]
                    for target, run_id in setup["members"].items()
                },
                "max_seconds": setup["cap"],
            },
            time_cap_seconds=setup["cap"],
            cooperative=False,
        )
        verification = _system_verification(session, setup, task, reason)
    except Exception as exc:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[_crashed(exc)],
        ), False
    return _system_envelope(session, verification, task)


def _system_envelope(
    session: Session, verification: dict[str, Any], task: IsolatedTask
) -> tuple[dict[str, Any], bool]:
    """Record one verification on the tree and shape its envelope. A
    failed append is one more failure: the result stands, the attempt is
    not in the tree and takes no part in the selection."""
    failures = [] if task.value is not None else [_task_failure(task, "verify_system")]
    recorded = True
    try:
        session.ledger.verify(verification)
    except Exception as exc:
        failures.append(_crashed(exc))
        recorded = False
    standing = (
        standing_verification(session.ledger.system(verification["system_id"]))
        if recorded
        else None
    )
    primary = {
        key: verification[key]
        for key in (
            "system_id",
            "verification_id",
            "members",
            "laws",
            "raw_fingerprint",
        )
    }
    status = "failure" if task.value is None else "partial" if failures else "success"
    return envelope(
        status,
        budget=session.budget(),
        primary={
            **primary,
            "forward": verification["forward"],
            "standing_verification_id": None
            if standing is None
            else standing["verification_id"],
        },
        diagnostics={
            "frame": verification["frame"],
            "worker_seconds": verification["budget"]["worker_seconds"],
            "forward_max_seconds": verification["budget"]["forward_seconds"],
            "load_seconds": verification["seconds"]["load"],
            "forward_seconds": verification["seconds"]["forward"],
            "wallclock_seconds": verification["seconds"]["worker"],
        },
        provenance={"input_id": verification["input_id"]},
        failures=failures,
    ), recorded


def _system_refusal(session: Session, system_id: str) -> dict[str, Any] | None:
    """Why a system cannot be named by `select_node` or `submit_answer`, or
    None: it must be one a verification recorded on this workspace's tree,
    with no abandoned member."""
    recorded = [system["system_id"] for system in session.ledger.systems()]
    if system_id not in recorded:
        return _failure(
            "unknown_system_id",
            f"system_id {system_id!r} is not a system on this workspace's tree; "
            f"the recorded ones are {recorded}. The value comes from "
            "primary.system_id in what verify_system returned",
        )
    if session.ledger.system_pruned(system_id):
        return _failure(
            "pruned_run_id",
            f"a member of system {system_id!r} was abandoned (itself or an "
            "ancestor of it), so the system cannot be published",
        )
    return None


def _system_member(
    session: Session, system_id: str
) -> tuple[SegmentEntry | None, dict[str, Any] | None]:
    """The member of system `system_id` for this run's target, or a refusal.
    The target is the pinned input's (a system of other trajectories is
    refused); an unpinned run takes the input of the system's latest
    verification. A member this run searched must still have the input it
    was launched on."""
    refusal = _system_refusal(session, system_id)
    if refusal is not None:
        return None, refusal
    system = session.ledger.system(system_id)
    pinned = session.pinned_dataset_id
    dataset_id = (
        pinned if pinned is not None else system["verifications"][-1]["input_id"]
    )
    raw = _raw_target(session, dataset_id)
    if raw is None or raw[0] != system["raw_fingerprint"]:
        return None, _failure(
            "dataset_not_pinned",
            f"system {system_id!r} integrates other trajectories than the input "
            f"{dataset_id!r} this run answers for",
        )
    entry = session.ledger.entry(system["members"][raw[1]])
    if entry.run_id in session.successful_runs:
        refusal = _search_input_refusal(session, entry.run_id)
        if refusal is not None:
            return None, refusal
    return entry, None


def _select_system(
    session: Session, system_id: str, reason: str, tool_call_id: str
) -> dict[str, Any] | Command[Any]:
    """`select_node(system_id=...)`: record the system selection on the tree
    and publish the system's member for this run's target with the system
    beside it."""
    entry, refusal = _system_member(session, system_id)
    if refusal is not None:
        return envelope("failure", budget=session.budget(), failures=[refusal])
    entry = cast(SegmentEntry, entry)
    provenance = _entry_provenance(entry)
    try:
        session.ledger.select_system(system_id, reason=reason)
    except Exception as exc:


        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[_crashed(exc)],
        )
    session.answer_dataset_id = entry.dataset
    result = envelope(
        "success",
        budget=session.budget(),
        primary={
            "system_id": system_id,
            "run_id": entry.run_id,
            "reason": reason,
            "now_selected": entry.run_id,
        },
        provenance=provenance,
    )
    answer = Answer(law=entry.law, source="selected", provenance=provenance)
    answer["system"] = _answer_system(session.ledger, system_id)
    return Command(
        update={
            "messages": [_tool_message(result, "select_node", tool_call_id)],
            ANSWER: answer,
        }
    )


def _submitted_system(
    session: Session, system_id: str, run_id: str, law: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The `system` a submission with `system_id` carries, or a refusal:
    `run_id` must be the system's member for this run's target and `law`
    exactly that member's sealed law."""
    entry, refusal = _system_member(session, system_id)
    if refusal is not None:
        return None, refusal
    entry = cast(SegmentEntry, entry)
    if entry.run_id != run_id or entry.law != law:
        return None, _failure(
            "invalid_argument",
            f"with system_id {system_id!r} the submission is that system's member "
            f"for this run's target, run {entry.run_id!r} with its sealed law "
            "unchanged; submit without system_id to answer with another run or "
            "an edited law",
        )
    return _answer_system(session.ledger, system_id), None


def _network_recipe(
    network_recipe: dict[str, Any] | None, run_choice: str | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The resolved recipe of a network preparation as JSON
    (`kd.models.FieldRecipe.to_dict()`), or a refusal. None selects kd's
    `DLGA_NN_1`; a given recipe names every `FieldRecipe` field, with
    `training_dtype` optional, and kd's `FieldRecipe` judges the values.
    `training_dtype` is resolved under the run's choice (`run_choice`,
    `session.effective_training_dtype`) before the recipe exists, so the
    resolved recipe, and with it the reuse key, carries the precision the
    network is trained in."""
    given = DLGA_NN_1.to_dict() if network_recipe is None else network_recipe
    optional = {"training_dtype"}
    required = {field.name for field in fields(FieldRecipe)} - optional
    names = set(given)
    if not required <= names <= required | optional:
        return None, _failure(
            "invalid_argument",
            f"network_recipe names the fields {sorted(required)} and optionally "
            f"{sorted(optional)}; missing {sorted(required - names)}, unknown "
            f"{sorted(names - required - optional)}. Omit it for kd's "
            f"{DLGA_NN_1.name}",
        )
    try:
        recipe = FieldRecipe(
            **{
                **given,
                "hidden_sizes": tuple(given["hidden_sizes"]),
                "training_dtype": effective_training_dtype(
                    run_choice, given.get("training_dtype")
                ),
            }
        )
    except (TypeError, ValueError) as exc:
        return None, _failure("invalid_argument", f"network_recipe: {exc}")
    return recipe.to_dict(), None


def _training_cap(
    session: Session, max_seconds: int | None
) -> tuple[float | None, dict[str, Any] | None]:
    """The wall-clock cap of a network training, which has no cooperative
    stop: an explicit `max_seconds`, else the run's remaining time with no
    floor (a floor would start a training that cannot finish); an exhausted
    budget is refused before anything starts, and no budget is no cap."""
    if max_seconds is not None:
        return float(max_seconds), None
    remaining = session.budget()["remaining_seconds"]
    if remaining is None:
        return None, None
    if remaining <= 0:
        return None, _failure(
            "budget_exhausted",
            f"this run's time budget is exhausted ({remaining} s remaining); a "
            "training cut short at the cap keeps nothing, so none was started",
        )
    return float(remaining), None


def _field_envelope(
    session: Session,
    description: dict[str, Any],
    *,
    reused: bool,
    task: IsolatedTask | None,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """A sampled field network's derived input -> `smooth_input`'s envelope.
    The description holds the model file, its sha256 and, for a network
    trained here, kd's native preparation identity and quality block; a
    fresh call adds its wall-clock and the worker's resources."""
    network = description["derived"]["network"]
    preparation = network.get("preparation")
    diagnostics: dict[str, Any] = {
        "preparation": preparation,
        "quality": network.get("quality"),
    }
    if task is not None:
        diagnostics["wallclock_seconds"] = round(task.wallclock_seconds, 3)
        diagnostics["resources"] = task.resources
    return envelope(
        "success",
        budget=session.budget(),
        primary={
            "dataset_id": description["ref"],
            "derived_from": description["derived"]["from"],
            "surrogate_id": network["surrogate_id"],
            "method": "network",
            "reused": reused,
        },
        diagnostics=diagnostics,
        provenance={
            **provenance,
            "artifact": {
                "path": network["artifact"],
                "sha256": network["artifact_sha256"],
            },
            "binding": (
                None
                if preparation is None
                else {
                    "dataset": preparation["dataset"],
                    "target_sha256": preparation["field_sha256"][
                        preparation["field_order"][0]
                    ],
                }
            ),
            "file_sha256": description["source"]["sha256"],
            "path": f"inputs/{description['ref']}.npz",
        },
    )


def _network_field(
    session: Session,
    *,
    dataset_id: str,
    description: dict[str, Any],
    surrogate_id: str,
    recipe: dict[str, Any] | None,
    seed: int,
    cap: float | None,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Train (`recipe` given) or sample the network `surrogate_id` names in
    a worker, register a trained one, register its field as a derived input
    and return the envelope. The derived id is `field_ref(dataset_id,
    surrogate_id)`: a registered one that verifies is reused with no worker,
    and one that does not is refused (`_reusable`)."""
    derived_id = field_ref(dataset_id, surrogate_id)
    existing, refusal = _reusable(session, derived_id)
    if refusal is not None:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[refusal],
        )
    if existing is not None:
        return _field_envelope(
            session, existing, reused=True, task=None, provenance=provenance
        )
    registry = session.surrogates
    model_path = registry.file_for(surrogate_id)
    request: dict[str, Any] = {
        TASK_KEY: NETWORK_TASK,
        "dataset_id": dataset_id,
        "input_source": description,
        "output_path": str(session.workspace / "inputs" / f"{derived_id}.npz"),
        "model_path": str(model_path),
        "network": {
            "surrogate_id": surrogate_id,
            "artifact": model_path.relative_to(session.workspace).as_posix(),
        },
    }
    if recipe is not None:
        request.update(recipe=recipe, seed=seed)
        if session.device == "cuda":
            request["device"] = "cuda"
    else:
        request["surrogate"] = {
            "path": str(model_path),
            "sha256": registry.entry(surrogate_id).file_sha256,
        }
    session.workers_started += 1
    task = run_isolated_task(request, time_cap_seconds=cap, cooperative=False)
    if task.value is None:
        failure = _task_failure(task, "smooth")
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            diagnostics=None
            if task.resources is None
            else {"resources": task.resources},
            failures=[failure],
        )
    written = task.value["description"]
    if recipe is not None:
        training = task.value["training"]
        registry.register(
            recipe={
                "dataset_id": dataset_id,
                "algorithm": NETWORK_ALGORITHM,
                "seed": seed,
                "params": recipe,
            },
            key=None,
            file_sha256=file_sha256(model_path),
            train_seconds=training["train_seconds"],
            epochs=training["epochs"],
            final_loss=training["final_loss"],
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            identity=written["derived"]["network"]["preparation"],
        )
    save_input(session.workspace, written)
    try:
        registered = _register_derived(session, written)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[
                _failure(
                    "input_unverified",
                    f"the field written as {derived_id!r} does not verify, "
                    f"so it is not registered: {exc}",
                )
            ],
        )
    return _field_envelope(
        session, registered, reused=False, task=task, provenance=provenance
    )


def _network_preparation(
    session: Session,
    *,
    dataset_id: str,
    description: dict[str, Any],
    surrogate_id: str | None,
    network_recipe: dict[str, Any] | None,
    seed: int,
    max_seconds: int | None,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """`smooth_input(method="network")`: sample a registered model, or train
    one under its recipe (once: the registry's recipe row is the reuse key),
    and register its field as a derived input.

    The reuse key is the request (the input, the resolved recipe, the seed),
    which determines kd's native preparation identity apart from the model
    hash: the identity's dataset fingerprint carries the input's id (a file
    input loads under it), and that id binds the bytes, the loading and the
    dtype. A registered model whose file changed is refused and never
    trained again; so is an existing model asked to train. When the run has
    chosen a training precision (`Session.training_dtype`), a registered
    model trained under another one is refused, whether found by recipe
    (the resolved recipe keys it) or named by `surrogate_id`.
    """

    def refused(failure: dict[str, Any]) -> dict[str, Any]:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[failure],
        )

    registry = session.surrogates
    recipe: dict[str, Any] | None = None
    if surrogate_id is None:
        recipe, refusal = _network_recipe(network_recipe, session.training_dtype)
        if refusal is not None:
            return refused(refusal)
        recipe = cast(dict[str, Any], recipe)
        row = {
            "dataset_id": dataset_id,
            "algorithm": NETWORK_ALGORITHM,
            "seed": seed,
            "params": recipe,
        }
        provenance.update(recipe=recipe, seed=seed)
        surrogate_id = surrogate_id_of(row)
        hit = registry.find_by_recipe(row)
        if hit is not None:
            recipe = None
    try:
        entry = registry.entry(surrogate_id) if recipe is None else None
    except KeyError:
        return refused(
            _failure(
                "unknown_surrogate",
                f"surrogate {surrogate_id!r} is not registered in this workspace; the "
                f"registered ones are {list(registry.ids())}",
            )
        )
    if entry is not None:
        path = registry.path(surrogate_id)
        if not path.is_file() or file_sha256(path) != entry.file_sha256:
            return refused(
                _failure(
                    "surrogate_file_changed",
                    f"the file of {surrogate_id!r} ({entry.file}) is missing or no "
                    "longer hashes to its registry row; it is refused and not trained "
                    "again under the same id",
                )
            )
        if entry.recipe["dataset_id"] != dataset_id:
            return refused(
                _failure(
                    "surrogate_binding_mismatch",
                    f"surrogate {surrogate_id!r} was trained on "
                    f"{entry.recipe['dataset_id']!r}; it is sampled on that input only",
                )
            )
        refusal = _run_precision_failure(
            session,
            surrogate_id,
            entry,
            "a run samples only a network trained in its own precision. Prepare "
            "one with smooth_input under this run (it takes the run's precision)",
        )
        if refusal is not None:
            return refused(refusal)
        cap = float(max_seconds) if max_seconds is not None else _remaining_cap(session)
    else:
        cap, refusal = _training_cap(session, max_seconds)
        if refusal is not None:
            return refused(refusal)
    return _network_field(
        session,
        dataset_id=dataset_id,
        description=description,
        surrogate_id=surrogate_id,
        recipe=recipe,
        seed=seed,
        cap=cap,
        provenance=provenance,
    )


def _smooth_argument_refusal(
    *,
    method: str,
    lowpass_modes: int | dict[str, int] | str | None,
    time_window: int,
    surrogate_id: str | None,
    network_recipe: dict[str, Any] | None,
    seed: int,
    terms: list[str] | None,
    tolerance: float,
    max_counts: int,
    max_seconds: int | None,
) -> dict[str, Any] | None:
    """Why `smooth_input`'s arguments cannot be combined; None when they can.
    Each method takes its own arguments and refuses the other's: a
    parameter that would be silently ignored is refused instead."""
    auto = lowpass_modes == "auto"
    selector = terms is not None or tolerance != 0.02 or max_counts != 8
    network_args = surrogate_id is not None or network_recipe is not None or seed != 0
    checks = (
        (
            method not in ("lowpass", "network"),
            f"method has to be 'lowpass' or 'network', got {method!r}",
        ),
        (
            method == "lowpass" and lowpass_modes is None,
            "method='lowpass' needs lowpass_modes (a count, a mapping or 'auto')",
        ),
        (
            method == "lowpass" and network_args,
            "surrogate_id, network_recipe and seed belong to method='network'",
        ),
        (
            method == "network" and (lowpass_modes is not None or time_window != 0),
            "method='network' samples a field network and takes no lowpass_modes "
            "or time_window",
        ),
        (
            selector and not auto,
            "terms, tolerance and max_counts belong to lowpass_modes='auto'",
        ),
        (
            auto and not terms,
            "lowpass_modes='auto' needs terms: the candidate terms the selector refits",
        ),
        (
            method == "network"
            and surrogate_id is not None
            and (network_recipe is not None or seed != 0),
            "an existing model (surrogate_id) is sampled, not trained, so "
            "network_recipe and seed are refused with it",
        ),
        (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0,
            f"seed has to be a nonnegative integer, got {seed!r}",
        ),
    )
    for failed, detail in checks:
        if failed:
            return _failure("invalid_argument", detail)
    if max_seconds is not None:
        return _positive_seconds("max_seconds", max_seconds)
    return None


def _auto_lowpass(
    session: Session,
    *,
    dataset_id: str,
    description: dict[str, Any],
    terms: list[str],
    time_window: int,
    tolerance: float,
    max_counts: int,
    max_seconds: float | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """kd's automatic choice of the low-pass mode counts for these terms on
    this input, made in a worker (`kd.select_lowpass_modes` with
    `kd.refit_terms` at the input's own left-hand-side order) stopped at
    `max_seconds`, or without it at the run's remaining time: the
    selection with its complete trace, or the refusal. A selector failure
    is reported, never replaced by a guessed count."""
    session.workers_started += 1
    task = run_isolated_task(
        {
            TASK_KEY: SELECT_LOWPASS_TASK,
            "dataset_id": dataset_id,
            "input_source": description,
            "terms": list(terms),
            "time_window": time_window,
            "tolerance": tolerance,
            "max_counts": max_counts,
        },
        time_cap_seconds=(
            float(max_seconds) if max_seconds is not None else _remaining_cap(session)
        ),
        cooperative=False,
    )
    if task.value is None:
        return None, _task_failure(task, "smooth")
    return finite_tree(task.value), None


def _finish_without_answer(
    session: Session, reason: str, evidence_run_ids: list[str], tool_call_id: str
) -> dict[str, Any] | Command[Any]:
    """`finish_without_answer`: check the reason and the cited searches and
    write the abstained answer; no numerical call."""
    if not reason.strip():
        return envelope(
            "failure",
            budget=session.budget(),
            failures=[
                _failure(
                    "invalid_argument",
                    "reason has to say why the route ends without an answer",
                )
            ],
        )
    recorded = {entry.run_id for entry in session.ledger.segments()}
    unknown = [run for run in evidence_run_ids if run not in recorded]
    if unknown:
        return envelope(
            "failure",
            budget=session.budget(),
            failures=[
                _failure(
                    "unknown_run_id",
                    f"evidence_run_ids {unknown} are not searches on this workspace's "
                    "policy tree; cite the provenance.run_id of the searches the reason "
                    "rests on",
                )
            ],
        )
    result = envelope(
        "success",
        budget=session.budget(),
        primary={
            "answer": None,
            "reason": reason,
            "evidence_run_ids": list(evidence_run_ids),
        },
        diagnostics={
            "recorded": True,
            "note": ("This run's result is recorded as unresolved, with no answer."),
        },
    )
    _note_action(
        session,
        result,
        verb="finish_without_answer",
        ref={"evidence_run_ids": list(evidence_run_ids)},
        run_id=None,
        reason=reason,
        status="success",
    )
    return Command(
        update={
            "messages": [_tool_message(result, "finish_without_answer", tool_call_id)],
            ANSWER: Answer(
                law=None,
                source="abstained",
                provenance=None,
                reason=reason,
                evidence_run_ids=list(evidence_run_ids),
            ),
        }
    )


def _smooth_input(
    session: Session,
    dataset_id: str,
    *,
    lowpass_modes: int | dict[str, int] | str | None,
    time_window: int,
    method: str,
    surrogate_id: str | None,
    network_recipe: dict[str, Any] | None,
    seed: int,
    terms: list[str] | None,
    tolerance: float,
    max_counts: int,
    max_seconds: int | None,
) -> dict[str, Any]:
    """`smooth_input`: the refusals that need no data, then the parent's
    chain, then the low-pass (with kd's automatic choice first for
    `"auto"`) or the network preparation."""
    provenance: dict[str, Any] = {
        "dataset": dataset_id,
        "method": method,
        "lowpass_modes": lowpass_modes,
        "time_window": time_window,
    }

    def refused(failure: dict[str, Any]) -> dict[str, Any]:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[failure],
        )

    refusal = _smooth_argument_refusal(
        method=method,
        lowpass_modes=lowpass_modes,
        time_window=time_window,
        surrogate_id=surrogate_id,
        network_recipe=network_recipe,
        seed=seed,
        terms=terms,
        tolerance=tolerance,
        max_counts=max_counts,
        max_seconds=max_seconds,
    )
    if refusal is not None:
        return refused(refusal)
    pinned = session.pinned_dataset_id
    if pinned is not None and not _same_input(dataset_id, pinned):
        provenance["pinned"] = pinned
        return refused(
            _failure(
                "dataset_not_pinned",
                f"this run is pinned to dataset {pinned!r} and can only smooth that "
                "one or an input derived from it",
            )
        )
    if not is_file_ref(dataset_id):
        return refused(
            _failure(
                "invalid_argument",
                f"smooth_input takes a file input (a file-v1- id); {dataset_id!r} is "
                "a catalog dataset",
            )
        )
    if is_pinn_derived(dataset_id):
        return refused(
            _failure(
                "unsupported_form",
                f"{dataset_id!r} is the field of a PINN fit (or derived from one), "
                "which smooth_input does not take; smooth the input the fit was made on",
            )
        )
    if method == "lowpass":
        failure = (
            _lowpass_failure(
                cast("int | dict[str, int]", lowpass_modes), time_window, optional=False
            )
            if lowpass_modes != "auto"
            else _window_refusal(time_window)
        )
        if failure is not None:
            return refused(failure)
    unverified = _input_refusal(session, dataset_id)
    if unverified is not None:
        return refused(unverified)
    try:
        chain = derivation_chain(
            session.workspace, dataset_id, written=session.derived_inputs
        )
        description = chain[-1]
        report = description["summary"]["report"]
        if is_table(description) or report["topology"] != "grid":
            return refused(
                _failure(
                    "invalid_argument",
                    f"{dataset_id!r} is not a grid input, and smoothing runs over "
                    "grid axes",
                )
            )
        if report["lhs_order"] == 0:
            return refused(
                _failure(
                    "unsupported_form",
                    f"{dataset_id!r} has no time-derivative left-hand side, and the "
                    "derived input would carry none either",
                )
            )
        provenance["parent_sha256"] = description["source"]["sha256"]
        provenance["root"] = root_ref(dataset_id)
        provenance["chain"] = [
            {"dataset_id": link["ref"], "file_sha256": link["source"]["sha256"]}
            for link in chain
        ]
        if method == "network":
            return _network_preparation(
                session,
                dataset_id=dataset_id,
                description=description,
                surrogate_id=surrogate_id,
                network_recipe=network_recipe,
                seed=seed,
                max_seconds=max_seconds,
                provenance=provenance,
            )
        selection: dict[str, Any] | None = None


        cap = None if max_seconds is None else float(max_seconds)
        if lowpass_modes == "auto":
            started = time.monotonic()
            selection, refusal = _auto_lowpass(
                session,
                dataset_id=dataset_id,
                description=description,
                terms=cast(list[str], terms),
                time_window=time_window,
                tolerance=tolerance,
                max_counts=max_counts,
                max_seconds=cap,
            )
            if refusal is not None:
                return refused(refusal)
            lowpass_modes = cast(dict[str, Any], selection)["modes"]
            if cap is not None:
                cap -= time.monotonic() - started
        return _lowpass_derivation(
            session,
            dataset_id=dataset_id,
            description=description,
            report=report,
            lowpass_modes=cast("int | dict[str, int]", lowpass_modes),
            time_window=time_window,
            selection=selection,
            max_seconds=cap,
            provenance=provenance,
        )
    except Exception as exc:



        return refused(_crashed(exc))


def _window_refusal(time_window: int) -> dict[str, Any] | None:
    window = _window_failure("time_window", time_window)
    return None if window is None else _failure("invalid_argument", window)


def _lowpass_derivation(
    session: Session,
    *,
    dataset_id: str,
    description: dict[str, Any],
    report: dict[str, Any],
    lowpass_modes: int | dict[str, int],
    time_window: int,
    selection: dict[str, Any] | None,
    max_seconds: float | None,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """The low-passed copy of `dataset_id` under these counts: the axis
    checks the summary allows, then reuse of a registered copy or a worker
    that writes it. `selection` is kd's automatic choice the counts came
    from, reported with the result."""

    def refused(failure: dict[str, Any]) -> dict[str, Any]:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[failure],
        )

    provenance["lowpass_modes"] = lowpass_modes
    lengths = {axis["name"]: axis["n"] for axis in report["axes"]}
    smoothed_axes = (
        [name for name in lengths if name != report["lhs_axis"]]
        if isinstance(lowpass_modes, int)
        else list(lowpass_modes)
    )
    unknown = sorted(set(smoothed_axes) - set(lengths))
    if unknown:
        return refused(
            _failure(
                "invalid_argument",
                f"lowpass_modes names axes {unknown} that {dataset_id!r} does not "
                f"have; its axes are {list(lengths)}",
            )
        )
    counts = {
        axis: lowpass_modes if isinstance(lowpass_modes, int) else lowpass_modes[axis]
        for axis in smoothed_axes
    }
    too_many = {axis: count for axis, count in counts.items() if count > lengths[axis]}
    if too_many:
        return refused(
            _failure(
                "invalid_argument",
                f"lowpass_modes {too_many} exceed the axis lengths "
                f"{ {axis: lengths[axis] for axis in too_many} }; a count is at most "
                "the axis length",
            )
        )
    derived_id = derived_ref(dataset_id, lowpass_modes, time_window)
    existing, refusal = _reusable(session, derived_id)
    if refusal is not None:
        return refused(refusal)
    if existing is not None:
        return _smoothed_envelope(
            existing,
            wallclock_seconds=None,
            selection=selection,
            provenance=provenance,
            budget=session.budget(),
        )
    session.workers_started += 1
    smoothed = run_isolated_smooth(
        dataset_id=dataset_id,
        input_source=description,
        lowpass_modes=lowpass_modes,
        time_window=time_window,
        output_path=session.workspace / "inputs" / f"{derived_id}.npz",
        time_cap_seconds=(
            float(max_seconds) if max_seconds is not None else _remaining_cap(session)
        ),
    )
    if smoothed.description is None:
        return refused(
            _failure(
                "smooth_timeout"
                if smoothed.error_code == "SearchTimeout"
                else str(smoothed.error_code),
                str(smoothed.error_message),
            )
        )
    save_input(session.workspace, smoothed.description)
    try:
        written = _register_derived(session, smoothed.description)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return refused(
            _failure(
                "input_unverified",
                f"the low-passed copy written as {derived_id!r} does not verify, so "
                f"it is not registered: {exc}",
            )
        )
    return _smoothed_envelope(
        written,
        wallclock_seconds=smoothed.wallclock_seconds,
        selection=selection,
        provenance=provenance,
        budget=session.budget(),
    )


def _refine_candidate(
    session: Session,
    run_id: str,
    *,
    law: dict[str, Any] | None,
    terms: list[str] | None,
    smoothing: dict[str, Any],
    pinn: dict[str, Any],
    evaluation_input_id: str | None,
    frame: tuple[str, Trim, dict[str, Any] | None, str | None],
) -> dict[str, Any]:
    """`refine_candidate`: the refusals that need no data, the measurement
    input, then a PINN fit or one framed refit in a worker, and the receipt
    of a successful one. `smoothing` holds `small_term_rtol`,
    `savgol_window`, `lowpass_modes` and `time_window`; `pinn` holds
    `pinn_epochs`, `pinn_pretrain_epochs` and `pinn_residual_weight`;
    `frame` is `(method, trim, weak, surrogate_id)`."""
    method, trim, weak, surrogate_id = frame
    parameters = {
        "law": law,
        "terms": terms,
        **smoothing,
        **pinn,
        "evaluation_input_id": evaluation_input_id,
        "method": method,
        "trim": trim,
        "surrogate_id": surrogate_id,
        "weak": weak,
    }
    provenance: dict[str, Any] = {
        "run_id": run_id,
        "refined_from": REFINED_FROM,
        "params": parameters,
    }

    def refused(failure: dict[str, Any]) -> dict[str, Any]:
        return envelope(
            "failure",
            budget=session.budget(),
            provenance=provenance,
            failures=[failure],
        )

    inline = bool(
        smoothing["savgol_window"] or smoothing["lowpass_modes"] or pinn["pinn_epochs"]
    )
    refusal = (
        _refine_refusal(session, run_id, law=law, terms=terms, **smoothing, **pinn)
        or _frame_refusal(method, trim, weak, surrogate_id, inline=inline)
        or (
            _failure(
                "invalid_argument",
                "a PINN fit is made on the input the search's chain starts from; "
                "evaluation_input_id is refused with pinn_epochs",
            )
            if pinn["pinn_epochs"] and evaluation_input_id is not None
            else None
        )
    )
    if refusal is not None:
        return refused(refusal)
    source = session.successful_runs[run_id]
    provenance["instrument"] = source["instrument"]
    provenance["dataset"] = source["dataset"]
    lhs = session.sealed_lhs[run_id]
    try:
        measured, refusal = _measured_input(session, run_id, evaluation_input_id)
        if refusal is not None:
            return refused(refusal)
        measured = cast(dict[str, Any], measured)
        provenance["evaluation_input"] = measured["record"]
        candidate = cast(
            dict[str, Any], session.ledger.entry(run_id).law if law is None else law
        )
        original = None if terms is not None else candidate.get("coefficients")
        refusal = _refine_input_refusal(
            table=is_table(measured["input_source"]),
            catalog=measured["input_source"] is None,
            lhs=lhs,
            coefficients=original,
            small_term_rtol=smoothing["small_term_rtol"],
            savgol_window=smoothing["savgol_window"],
            lowpass_modes=smoothing["lowpass_modes"],
            pinn_epochs=pinn["pinn_epochs"],
            framed=method != "finite_diff" or trim != 0,
        )
        if refusal is not None:
            return refused(refusal)
        if pinn["pinn_epochs"]:
            return _pinn_receipt(
                session,
                run_id,
                candidate=candidate,
                lhs=cast(dict[str, Any], lhs),
                pinn=pinn,
                parameters=parameters,
                provenance=provenance,
            )
        model = None
        if surrogate_id is not None:
            model, refusal = _model_binding(
                session, surrogate_id, measured["dataset_id"]
            )
            if refusal is not None:
                return refused(refusal)
        given = list(candidate["support"] if terms is None else terms)
        session.workers_started += 1
        refined = run_isolated_refine(
            dataset_id=measured["dataset_id"],
            input_source=measured["input_source"],
            terms=given,
            coefficients=(
                None
                if original is None
                else [math.nan if value is None else value for value in original]
            ),
            lhs_order=0 if lhs is None else lhs["order"],
            **smoothing,
            method=method,
            trim=trim,
            weak=weak,
            surrogate=model,
            time_cap_seconds=_remaining_cap(session),
        )
        result = _refined_envelope(
            refined,
            lhs=lhs,
            smoothing=smoothing,
            method=method,
            trim=trim,
            original_coefficients=original,
            provenance=provenance,
            budget=session.budget(),
        )
        if result["status"] == "success":
            diagnostics = result["diagnostics"]
            result["primary"]["refinement_id"] = _record_receipt(
                session,
                run_id=run_id,
                law=result["primary"]["law"],
                support=given,
                method=(
                    "constants" if diagnostics["frame"] == REFIT_TABLE_FRAME else method
                ),
                parameters=parameters,
                evaluation=measured["record"],
                measurement={
                    key: diagnostics.get(key)
                    for key in (
                        "nmse",
                        "frame",
                        "refit_frame",
                        "weak",
                        "r2",
                        "mse",
                        "condition_number",
                    )
                },
                artifact_sha256=None if model is None else model["sha256"],
            )
    except Exception as exc:



        return refused(_crashed(exc))
    return result


def _pinn_receipt(
    session: Session,
    run_id: str,
    *,
    candidate: dict[str, Any],
    lhs: dict[str, Any],
    pinn: dict[str, Any],
    parameters: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """A PINN refinement (`_pinn_refinement`) with the receipt of a
    successful one: the fit is measured on the root of the search's input,
    and the field it wrote is the receipt's output input."""
    dataset_id = session.successful_runs[run_id]["dataset"]
    result = _pinn_refinement(
        session,
        dataset_id=dataset_id,
        candidate=candidate,
        lhs=lhs,
        epochs=pinn["pinn_epochs"],
        pretrain_epochs=pinn["pinn_pretrain_epochs"],
        residual_weight=pinn["pinn_residual_weight"],
        provenance=provenance,
    )
    if result["status"] != "success":
        return result
    root = root_ref(dataset_id)
    derived = result["primary"]["dataset_id"]
    written = session.derived_inputs
    result["primary"]["refinement_id"] = _record_receipt(
        session,
        run_id=run_id,
        law=result["primary"]["law"],
        support=list(candidate["support"]),
        method="pinn",
        parameters=parameters,
        evaluation=input_record(
            root, derivation_chain(session.workspace, root, written=written)
        ),
        output_record=input_record(
            derived, derivation_chain(session.workspace, derived, written=written)
        ),
        measurement={
            key: result["diagnostics"].get(key)
            for key in (
                "nmse",
                "frame",
                "data_loss",
                "residual_loss",
                "epochs_run",
                "stopped_at_deadline",
            )
        },
        artifact_sha256=result["provenance"]["file_sha256"],
    )
    return result


def make_tools(session: Session) -> list[Callable[..., Any]]:
    """Wrap kd into the sixteen verbs admitted for this run, in a fixed
    order: `[list_datasets, run_discovery, train_surrogate, prune_node,
    select_node, submit_answer, refine_candidate, smooth_input,
    ablate_candidate, score_term_sets, verify_candidate, verify_system,
    vote_terms, make_sketch, set_sketch, finish_without_answer]`.

    The order is part of the surface (callers unpack it positionally), so it is
    stated once here. No exception may escape a tool: one that does loses the
    whole run. Every tool body touching real kd has to catch and turn it into a
    ``failures`` entry.
    """

















    def list_datasets() -> dict[str, Any]:
        """List the datasets available here. A file input is listed as id, task
        kind, target, the data report and the file's array inventory. A catalog
        dataset is listed as id, axes, lhs, fmt, tier and tags, with no report.
        The governing equation is never included.

        When this run is pinned to one dataset, only that dataset is listed,
        with the inputs `smooth_input` derived from it (each with its
        `derived_from`), and those are the only ids the other tools accept;
        on raw trajectories `run_discovery(target_state=...)` also searches
        the projection of another of their states.
        A derived input that this run did not write, or whose file or
        derivation no longer verifies against the input it names, is left
        out.
        """
        pinned = session.pinned_dataset_id
        try:
            if pinned is not None and is_file_ref(pinned):
                source = cast(dict[str, Any], _resolve(session, pinned))
                rows = [_listed(source)]
                inputs = session.workspace / "inputs"
                for path in sorted(inputs.glob(f"{pinned}+*.json")):
                    if _input_refusal(session, path.stem) is None:
                        derived = cast(dict[str, Any], _resolve(session, path.stem))
                        rows.append(_listed(derived))
            else:
                rows = list_datasets_answer_blind()
                if pinned is None:
                    for path in sorted((session.workspace / "inputs").glob("*.json")):
                        if _input_refusal(session, path.stem) is None:
                            source = cast(dict[str, Any], _resolve(session, path.stem))
                            rows.append(_listed(source))
        except Exception as exc:
            return envelope(
                "failure", budget=session.budget(), failures=[_crashed(exc)]
            )
        provenance = {"pinned": pinned}
        if pinned is None:
            return envelope(
                "success",
                budget=session.budget(),
                primary={"datasets": rows},
                provenance=provenance,
            )
        shown = [row for row in rows if _same_input(row["id"], pinned)]
        if not shown:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance=provenance,
                failures=[
                    _failure(
                        "unknown_dataset",
                        f"the built-in catalog has no {pinned!r}",
                    )
                ],
            )
        return envelope(
            "success",
            budget=session.budget(),
            primary={"datasets": shown},
            provenance=provenance,
        )












































































    def run_discovery(
        dataset_id: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        algorithm: str,
        params: dict[str, Any] | None = None,
        seed: int | None = None,
        max_search_seconds: int | None = None,
        resume_from_run_id: str | None = None,
        resume_iteration: int | None = None,
        reseed: bool = False,
        decision_parent: str | None = None,
        checkpoint_every: int | None = 1,
        checkpoint_keep_last: int | None = 5,
        sketch: dict[str, Any] | None = None,
        surrogate: str | None = None,
        max_evaluations: int | None = None,
        sketch_id: str | None = None,
        verify: dict[str, float] | None = None,
        reason: str | None = None,
        target_state: str | None = None,
    ) -> dict[str, Any] | Command[Any]:
        """Run one search on one dataset and wait for it to finish.

        Args:
            dataset_id: Which dataset to search. When this run is pinned to a
                dataset, any id other than it, an input `smooth_input`
                derived from it, or a projection of another state of the
                same raw trajectories is refused rather than corrected.
            algorithm: Which instrument to run: the algorithm name alone,
                such as `"sga"`; the input and `params` select its mode, as
                the Availability table shows. Its skill card lists the
                parameters it accepts and what they cost.
            params: Instrument parameters as plain JSON, passed through
                unchanged. Omitting the instrument's generation count gets you
                a small default, which is smaller than kd's own; pass one
                explicitly when you want a different one.
            seed: Random seed. Omitting it on a fresh call runs seed 0.
                Omitting it on a resumed call inherits the parent segment's
                seed, so a zero-change continuation needs nothing here; an
                explicit seed that differs from the parent's is refused,
                because kd classifies `seed` `init_only` -- unless the call
                also passes `reseed`, which is what a differing seed is for.
            max_search_seconds: Wall-clock cap for this one search. When
                omitted, a cap is derived from this run's remaining time
                budget if one was set; otherwise the search is uncapped.
            max_evaluations: Candidate-evaluation cap for this one search:
                the number of candidates the instrument may score, every
                instrument counted the same way (`diagnostics.evaluations`
                reports what was actually spent). The search stops at the
                first generation boundary at or past the cap, so the spend
                may overshoot it by one generation. Omitted means no cap;
                `params.generations` still bounds the run either way.
            resume_from_run_id: A sealed run from this session to continue or
                branch from. Omit `resume_iteration` to continue from its
                latest eligible checkpoint; provide an iteration to roll back
                and fork from that point.
            resume_iteration: An exact eligible checkpoint iteration in the
                parent run. It requires `resume_from_run_id`.
            reseed: Turn the resume into a branch: the search state restored
                from the checkpoint is kept, while the random stream is
                re-derived from the seed passed here. It requires
                `resume_from_run_id`, an explicit `seed`, and an instrument
                that declares it can do this; each missing one is its own
                refusal.
            decision_parent: The search whose result made you decide to run
                this one, taken from that search's `provenance.run_id`. It is
                not where this search resumes from -- that is
                `resume_from_run_id`, and on a rollback the two are different
                runs on purpose. Omit it and a resumed call records the run it
                resumed from, while a fresh call records nothing. It must be a
                search from this run; an abandoned one is allowed, since an
                abandoned result is a perfectly good reason to try something
                else.
            checkpoint_every: Positive checkpoint cadence. The default is 1.
                Set this and `checkpoint_keep_last` both to null to disable
                checkpoints.
            checkpoint_keep_last: Positive bound on retained periodic
                checkpoints, required whenever `checkpoint_every` is set:
                unbounded retention is not offered. The default is 5, so
                rollback normally reaches only the retained window; raise it
                when deeper rollback is worth the storage cost.
            sketch: A complete `kd-sketch-v2` payload that narrows the
                search to a structure: terms pinned with a coefficient, terms
                anchored without one, and holes that must be filled from a
                constrained pool, over a declared vocabulary. A released
                `kd-sketch-v1` payload is read too and recorded as
                `kd-sketch-v2`; the search's worker binds it to the input.
                Only an instrument whose
                skill card lists sketch clause levels takes one; the verdict
                comes back under `diagnostics.sketch` (absent, with the
                failure under `diagnostics.post_seal_error`, if kd sealed the
                record but could not write the verdict). On a resumed call it is
                inherited from the parent segment, so omit it; a different
                sketch is a new lineage (drop `resume_from_run_id`, name the
                reason with `decision_parent`), and a segment recorded under a
                `kd-sketch-v1` Sketch is not resumed. A wrong pin is not repaired:
                a hole candidate whose expansion shares a monomial with a
                pinned term is refused before scoring, and the law then fails
                the exit residual gate. No hole clause
                excludes a coordinate factor along an axis the hole
                differentiates: `axes` limits derivative axes and coordinate
                factors together, so `axes: ["x"]` still admits a factor of
                x. The exit gate is NMSE at most 0.05 on the search's own
                derivatives of the data it sees, unless `verify` sets
                another threshold. `params.small_term_rtol` is not applied
                on a sketch run. Prefer `sketch_id`; pass one of the two.
            surrogate: The `surrogate_id` a `train_surrogate` call returned,
                to inject that trained derivative surrogate instead of having
                this search train its own. It must be for this instrument and
                dataset, and the instrument must run in the mode that consults
                it (sga: `params={"derivatives": "autograd"}`). Inherited on a
                resumed call like `sketch`; a different id is a new lineage.
                A field network `smooth_input` trained is not a surrogate any
                instrument takes; search its derived input instead.
            sketch_id: The `sketch_id` a `make_sketch` or `set_sketch` call
                returned: the search runs under that registered Sketch. When
                a Sketch is active (the run's user Sketch, or the last one
                `set_sketch` bound), a search that names none runs under it,
                a different one is refused, and so is a search on an input
                of another root. Inherited on a resumed call.
            verify: The Sketch exit's residual threshold,
                `{"nmse_max": <a finite number >= 0>}`; omitted, kd's 0.05.
                A looser threshold admits exploratory candidates; a
                certificate under it does not establish a correct law. It
                gates only a search with a Sketch, and a resumed segment
                keeps its lineage's threshold (a different one is refused:
                start a fresh segment).
            reason: Why you run this search, in your own words. It is
                stored on the search's segment row in `tree.jsonl` and shown
                in the report. Optional; an empty one is refused.
            target_state: On an input projected from raw trajectories (its
                summary has a `projection`), search the time derivative of
                this state instead: the search runs on the projection of the
                same trajectories to that state, whose id the envelope's
                `provenance.dataset` names. That search is a search of its
                own projection: it is not published as this run's answer,
                and a resume stays on it. Omitted, the input's own target.

        On a resumed call, the generation count in `params` is the additional
        budget for this new segment, not a cumulative target. Parameters kd
        classifies `resume_safe` may change; every other `params` entry has to
        be restated verbatim from the parent segment's
        `provenance.params.effective`, and `dataset_id` has to stay the parent's
        too. kd's resume gate refuses an `init_only` or `identity_breaking`
        change and names the field and its tier, but it gets there only after
        the worker has spawned and loaded the data; `dataset_id` and `seed` are
        refused here instead, before anything spawns.

        Returns an envelope whose `primary.law` holds the discovered structure
        as `support` (coefficient-free terms), `coefficients` and `lhs` (the
        left-hand side the sealed record names, or null), whose
        `primary.equation` is that same law written out term by term on one
        line, each support term once and in order with its signed coefficient
        and nothing combined or cancelled (for example
        `u_t = -0.5*u*u_x + 0.1*u_xx`), present only when its left-hand side
        (`0` for a homogeneous law, the target column for a regression) and
        every coefficient are known -- it is for reading, and `submit_answer`
        takes `primary.law`, which holds the coefficients at full precision --
        whose `diagnostics` holds `nmse`, `score`, `search_seconds`, a
        `segment_report` block (score trend, two improvement numbers, plateau
        counters, diversity, validity, structure diff, cost) computed from the
        sealed run and a `resources` block measured in the search's worker
        (`device`, `"cuda"` when it allocated on the card, and
        `cuda_max_memory_allocated_bytes`), and whose `provenance` holds
        `run_id` -- the value
        `submit_answer` asks for. `primary.now_selected` is the `run_id` whose
        law this run publishes if you submit nothing: your `select_node`
        choice, otherwise the default rule over every search on the tree (on
        raw ODE trajectories, once a system `verify_system` measured has a
        result, the rule over the systems, and the `run_id` is the winning
        system's member for this run's target), or null when that rule has
        no answer. `diagnostics.selection_forward` is
        the forward error of this search's sealed law, with its own
        coefficients, that the default rule ranks searches of different
        algorithms on: measured once after the search in a worker of at most
        30 s (10 s of integration), charged to this run's time.

        A dlga search trains its own derivative network unless it injects
        one (`surrogate`) or resumes a segment, which restores its parent's.
        That network is registered under an id of its own:
        `provenance.trained_surrogate` (null when none was registered;
        `provenance.surrogate` names an injected one) and
        `diagnostics.surrogate` (`surrogate_id`, `stopped_at_deadline`,
        `savgol_window`, `train_seconds`, `epochs`, `final_loss`, or a
        `reason` when there is none). The id works where a `train_surrogate`
        id does: `surrogate` on a fresh lineage, `method="autograd"` with
        `surrogate_id` in `refine_candidate`, `ablate_candidate` and
        `score_term_sets`, and `smooth_input(surrogate_id=)`. A network the
        cap cut short (`stopped_at_deadline` true) is not a complete
        training: kd refuses it as a search's surrogate and `train_surrogate`
        does not reuse it, while the autograd measurements take it. A network
        trained on an input smoothed with `savgol_window` is that smoothed
        field's: only a search with the same window takes it, and the calls
        bound to the input as loaded refuse it.

        This call waits for the search, which runs in its own worker process
        under the cap. The cap is kd's deadline for the whole fit, so it also
        cuts a derivative network trained before the search (dlga, sga
        under autograd) short. `diagnostics.stopped_by_time_cap` true means
        the cap elapsed before the search finished; the search then stopped
        at an iteration boundary and its result is still a sealed,
        submittable record. An instrument with no iteration boundaries to
        stop at (a one-shot fit, or a preparation phase the deadline does not
        cut, such as DISCOVER's pretraining) is killed instead and returns a
        failure whose detail may quote the best expression seen so far --
        that quote is a plain string, not a submittable law. pysr is the
        exception among one-shot fits: kd passes
        the cap to PySR as its own time limit, so its fit ends near the cap
        and returns a sealed record. That fit can end a few seconds before the
        cap, so for pysr `stopped_by_time_cap` false does not mean the search
        ran all its iterations. Every envelope also carries a `budget` block
        reporting the wall-clock this whole run has used and, when a limit was
        set for it, how much of that limit is left.

        {algorithms}

        {llm4ed}
        """
        lineage = _lineage_provenance(
            None, session.workspace, requested_parent=resume_from_run_id
        )




        prior_diagnostics = _prior_diagnostics(session, algorithm)
        reason_failure = _reason_refusal(reason)
        if reason_failure is not None:
            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance={
                    "instrument": algorithm,
                    "dataset": dataset_id,
                    "seed": seed,
                    "lineage": lineage,
                },
                failures=[reason_failure],
            )
        asset_failure = _untrusted_asset_failure(schema_of(algorithm), params or {})
        if asset_failure is None and algorithm == "llm4ed":
            asset_failure = _llm4ed_refusal(session.llm4ed, params or {})
        if asset_failure is not None:
            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance={
                    "instrument": algorithm,
                    "dataset": dataset_id,
                    "seed": seed,
                    "lineage": lineage,
                },
                failures=[asset_failure],
            )
        combined = combined_name_refusal(algorithm, instrument_schemas())
        if combined is not None:



            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance={
                    "instrument": algorithm,
                    "dataset": dataset_id,
                    "seed": seed,
                    "lineage": lineage,
                },
                failures=[_failure("unknown_instrument", combined)],
            )
        process_only = (
            None if schema_of(algorithm) is None else _process_only_refusal(algorithm)
        )
        if process_only is not None:
            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance={
                    "instrument": algorithm,
                    "dataset": dataset_id,
                    "seed": seed,
                    "lineage": lineage,
                },
                failures=[process_only],
            )
        pinned = session.pinned_dataset_id
        if (
            pinned is not None
            and not _same_input(dataset_id, pinned)
            and not _same_raw_set(session, dataset_id, pinned)
        ):
            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance={
                    "dataset": dataset_id,
                    "pinned": pinned,
                    "lineage": lineage,
                },
                failures=[
                    _failure(
                        "dataset_not_pinned",
                        f"this run is pinned to dataset {pinned!r} and can only "
                        "search that one, an input smooth_input derived from "
                        "it, or, for raw trajectories, the projection of "
                        "another of their states (target_state)",
                    )
                ],
            )
        unverified = _input_refusal(session, dataset_id)
        if unverified is not None:
            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance={
                    "dataset": dataset_id,
                    "pinned": pinned,
                    "lineage": lineage,
                },
                failures=[unverified],
            )
        if max_search_seconds is not None and max_search_seconds <= 0:
            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance={
                    "instrument": algorithm,
                    "dataset": dataset_id,
                    "seed": seed,
                    "lineage": lineage,
                },
                failures=[
                    _failure(
                        "invalid_argument",
                        f"max_search_seconds has to be a positive number of "
                        f"seconds, got {max_search_seconds!r}; omit it if you "
                        "do not want a cap",
                    )
                ],
            )
        if max_evaluations is not None and (
            isinstance(max_evaluations, bool)
            or not isinstance(max_evaluations, int)
            or max_evaluations <= 0
        ):
            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance={
                    "instrument": algorithm,
                    "dataset": dataset_id,
                    "seed": seed,
                    "lineage": lineage,
                },
                failures=[
                    _failure(
                        "invalid_argument",
                        f"max_evaluations has to be a positive integer number "
                        f"of candidate evaluations, got {max_evaluations!r}; "
                        "omit it if you do not want a cap",
                    )
                ],
            )
        if resume_iteration is not None and resume_from_run_id is None:
            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance={
                    "instrument": algorithm,
                    "dataset": dataset_id,
                    "seed": seed,
                    "lineage": lineage,
                },
                failures=[
                    _failure(
                        "invalid_argument",
                        "resume_iteration requires resume_from_run_id as well",
                    )
                ],
            )
        reseed_failure = _reseed_control_failure(
            algorithm,
            reseed=reseed,
            resume_from_run_id=resume_from_run_id,
            seed=seed,
        )
        if reseed_failure is not None:
            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance={
                    "instrument": algorithm,
                    "dataset": dataset_id,
                    "seed": seed,
                    "lineage": lineage,
                },
                failures=[reseed_failure],
            )
        recorded = {entry.run_id for entry in session.ledger.segments()}
        if decision_parent is not None and decision_parent not in recorded:




            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance={
                    "instrument": algorithm,
                    "dataset": dataset_id,
                    "seed": seed,
                    "lineage": lineage,
                },
                failures=[
                    _failure(
                        "unknown_decision_parent",
                        f"decision_parent {decision_parent!r} is not a search "
                        "from this run; the value comes from provenance.run_id "
                        "in what run_discovery returned. Omit it on a search "
                        "that answers to nothing earlier",
                    )
                ],
            )
        checkpoint_failure = _checkpoint_control_failure(
            checkpoint_every, checkpoint_keep_last
        )
        if checkpoint_failure is not None:
            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance={
                    "instrument": algorithm,
                    "dataset": dataset_id,
                    "seed": seed,
                    "lineage": lineage,
                },
                failures=[checkpoint_failure],
            )
        ingress_failure = _verify_refusal(verify)
        if ingress_failure is None and resume_from_run_id is not None:


            ingress_failure = sketch_lineage_refusal(session.ledger, resume_from_run_id)
        requested_sketch: dict[str, Any] | None = None
        requested_sketch_id: str | None = None
        if ingress_failure is None and (
            resume_from_run_id is None or sketch is not None or sketch_id is not None
        ):


            requested_sketch, requested_sketch_id, ingress_failure = _requested_sketch(
                session, sketch, sketch_id, dataset_id
            )
        if ingress_failure is not None:
            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance={
                    "instrument": algorithm,
                    "dataset": dataset_id,
                    "seed": seed,
                    "lineage": lineage,
                },
                failures=[ingress_failure],
            )
        if requested_sketch is not None:




            sketch_failure = _sketch_failure(
                algorithm, requested_sketch, schema_of(algorithm), params=params or {}
            )
            if sketch_failure is not None:
                return envelope(
                    "failure",
                    budget=session.budget(),
                    diagnostics=prior_diagnostics,
                    provenance={
                        "instrument": algorithm,
                        "dataset": dataset_id,
                        "seed": seed,
                        "lineage": lineage,
                    },
                    failures=[sketch_failure],
                )
        if algorithm == "llm4ed":
            time_cap_seconds, exhausted = _llm4ed_cap(session, max_search_seconds)
            if exhausted is not None:
                return envelope(
                    "failure",
                    budget=session.budget(),
                    diagnostics=prior_diagnostics,
                    provenance={
                        "instrument": algorithm,
                        "dataset": dataset_id,
                        "seed": seed,
                        "lineage": lineage,
                    },
                    failures=[exhausted],
                )
        elif max_search_seconds is not None:
            time_cap_seconds = float(max_search_seconds)
        else:





            remaining = session.budget()["remaining_seconds"]
            time_cap_seconds = (
                None if remaining is None else max(60.0, float(remaining))
            )
        runs_root = session.workspace / DEFAULT_RUNS_ROOT
        provenance_params: dict[str, Any] | None = None
        resolved_resume: _ResolvedResume | None = None
        resolved_surrogate: _ResolvedSurrogate | None = None
        requested: dict[str, Any] = {}






        effective_sketch = requested_sketch
        effective_surrogate = surrogate
        effective_verify = verify
        try:
            if target_state is not None:


                dataset_id, target_failure = _projection_input(
                    session, dataset_id, target_state, time_cap_seconds
                )
                if target_failure is not None:
                    return envelope(
                        "failure",
                        budget=session.budget(),
                        diagnostics=prior_diagnostics,
                        provenance={
                            "instrument": algorithm,
                            "dataset": dataset_id,
                            "seed": seed,
                            "lineage": lineage,
                        },
                        failures=[target_failure],
                    )




            requested = deepcopy(params) if params else {}
            model_kwargs = dict(requested)
            defaults_applied: dict[str, Any] = {}
            schema = schema_of(algorithm)




            if "generations" not in model_kwargs and _spends_generations(schema):
                model_kwargs["generations"] = DEFAULT_GENERATIONS
                defaults_applied["generations"] = DEFAULT_GENERATIONS
            if algorithm == "llm4ed":
                _apply_llm4ed(
                    cast(Llm4edConfig, session.llm4ed), model_kwargs, defaults_applied
                )
            provenance_params = _parameter_provenance(
                schema, requested, defaults_applied
            )
            if resume_from_run_id is not None:
                resolved_resume, resume_failure = _resolve_resume(
                    session,
                    resume_from_run_id,
                    resume_iteration,
                    algorithm=algorithm,
                    dataset_id=dataset_id,
                    seed=seed,
                    reseed=reseed,
                )
                if resume_failure is not None:
                    return envelope(
                        "failure",
                        budget=session.budget(),
                        diagnostics=prior_diagnostics,
                        provenance={
                            "instrument": algorithm,
                            "dataset": dataset_id,
                            "seed": seed,
                            "params": provenance_params,
                            "lineage": lineage,
                        },
                        failures=[resume_failure],
                    )
                lineage = _lineage_provenance(resolved_resume, session.workspace)
                source = session.ledger.entry(resume_from_run_id)



                parent_verify = session.search_verify.get(resume_from_run_id)
                if verify is None:
                    effective_verify = parent_verify
                elif _threshold(verify) != _threshold(parent_verify):
                    return envelope(
                        "failure",
                        budget=session.budget(),
                        diagnostics=prior_diagnostics,
                        provenance={
                            "instrument": algorithm,
                            "dataset": dataset_id,
                            "seed": seed,
                            "params": provenance_params,
                            "lineage": lineage,
                        },
                        failures=[
                            _failure(
                                "invalid_argument",
                                f"run {resume_from_run_id!r} ran under the "
                                f"verification threshold {_threshold(parent_verify)} "
                                f"and this call asks for {_threshold(verify)}; a "
                                "resumed segment keeps its lineage's threshold. "
                                "Omit verify to inherit it, or start a fresh "
                                "segment under the new one",
                            )
                        ],
                    )
                if requested_sketch is None:
                    effective_sketch = source.sketch
                    if _differs_from_active(session, source.sketch):
                        return envelope(
                            "failure",
                            budget=session.budget(),
                            diagnostics=prior_diagnostics,
                            provenance={
                                "instrument": algorithm,
                                "dataset": dataset_id,
                                "seed": seed,
                                "params": provenance_params,
                                "lineage": lineage,
                            },
                            failures=[
                                _failure(
                                    "sketch_conflict",
                                    _active_sketch_detail(session)
                                    + f"; run {resume_from_run_id!r} ran under "
                                    "another Sketch (or none), and a resumed "
                                    "segment keeps its lineage's Sketch. Drop "
                                    "resume_from_run_id to start a fresh search, "
                                    "which runs under the active Sketch",
                                )
                            ],
                        )
                elif requested_sketch != source.sketch:
                    return envelope(
                        "failure",
                        budget=session.budget(),
                        diagnostics=prior_diagnostics,
                        provenance={
                            "instrument": algorithm,
                            "dataset": dataset_id,
                            "seed": seed,
                            "params": provenance_params,
                            "lineage": lineage,
                        },
                        failures=[
                            _failure(
                                "invalid_argument",
                                f"run {resume_from_run_id!r} ran under a "
                                "different sketch (or none), and a resumed "
                                "segment keeps its lineage's sketch. Omit "
                                "sketch to inherit it; to search under this "
                                "sketch, drop resume_from_run_id, record the "
                                "reason with decision_parent, and carry the "
                                "surrogate with surrogate=",
                            )
                        ],
                    )
                if surrogate is None:
                    effective_surrogate = source.surrogate
                elif surrogate != source.surrogate:
                    return envelope(
                        "failure",
                        budget=session.budget(),
                        diagnostics=prior_diagnostics,
                        provenance={
                            "instrument": algorithm,
                            "dataset": dataset_id,
                            "seed": seed,
                            "params": provenance_params,
                            "lineage": lineage,
                        },
                        failures=[
                            _failure(
                                "invalid_argument",
                                f"run {resume_from_run_id!r} ran with surrogate "
                                f"{source.surrogate!r}, and a resumed segment "
                                "keeps its lineage's surrogate (kd refuses the "
                                "resume as a changed run otherwise). Omit "
                                "surrogate to inherit it; to search with "
                                f"{surrogate!r}, drop resume_from_run_id and "
                                "record the reason with decision_parent",
                            )
                        ],
                    )
            if effective_surrogate is not None:
                resolved_surrogate, surrogate_failure = _resolve_surrogate(
                    session,
                    effective_surrogate,
                    algorithm=algorithm,
                    dataset_id=dataset_id,
                    schema=schema,
                    requested=requested,
                )
                if surrogate_failure is not None:
                    return envelope(
                        "failure",
                        budget=session.budget(),
                        diagnostics=prior_diagnostics,
                        provenance={
                            "instrument": algorithm,
                            "dataset": dataset_id,
                            "seed": seed,
                            "params": provenance_params,
                            "lineage": lineage,
                        },
                        failures=[surrogate_failure],
                    )
            if resolved_resume is None:
                run_seed = DEFAULT_SEED if seed is None else seed
            else:






                run_seed = resolved_resume.seed





            if is_file_ref(dataset_id):
                launch_chain = derivation_chain(
                    session.workspace, dataset_id, written=session.derived_inputs
                )
                input_source: dict[str, Any] | None = launch_chain[-1]
            else:
                launch_chain = []
                input_source = _resolve(session, dataset_id)
            launched_on = input_record(dataset_id, launch_chain)
            if effective_sketch is not None and is_table(input_source):
                return envelope(
                    "failure",
                    budget=session.budget(),
                    diagnostics=prior_diagnostics,
                    failures=[
                        _failure(
                            "sketch_unsupported",
                            "Tabular input does not accept sketch; omit sketch",
                        )
                    ],
                )
            precision_failure = _search_precision_failure(
                session,
                schema,
                model_kwargs,
                defaults_applied,
                table=is_table(input_source),
                surrogate=resolved_surrogate,
            )
            if precision_failure is not None:
                return envelope(
                    "failure",
                    budget=session.budget(),
                    diagnostics=prior_diagnostics,
                    provenance={
                        "instrument": algorithm,
                        "dataset": dataset_id,
                        "seed": seed,
                        "params": provenance_params,
                        "lineage": lineage,
                    },
                    failures=[precision_failure],
                )
            entry = PlanEntry(
                instrument=algorithm,
                dataset_ref=dataset_id,
                seed=run_seed,
                model_kwargs=model_kwargs,
            )
            if schema is not None:
                provenance_params["facade_effective"] = _collected_controls(
                    schema, model_kwargs
                )













            run_dir = runs_root / new_run_id(algorithm)
            network_target, network_reason = _network_target(
                session,
                schema,
                dataset_id=dataset_id,
                seed=run_seed,
                model_kwargs=model_kwargs,
                run_dir=run_dir,
                injected=resolved_surrogate is not None,
                resumed=resolved_resume is not None,
            )
            isolated = run_isolated_episode(
                entry,
                run_dir=run_dir,
                time_cap_seconds=time_cap_seconds,
                evaluation_cap=max_evaluations,
                input_source=input_source,
                resume_from=(
                    None if resolved_resume is None else resolved_resume.checkpoint
                ),
                reseed=reseed,
                checkpoint_every=checkpoint_every,
                checkpoint_keep_last=checkpoint_keep_last,
                sketch=effective_sketch,
                surrogate=(
                    None
                    if resolved_surrogate is None
                    else {
                        "path": str(resolved_surrogate.path),
                        "sha256": resolved_surrogate.sha256,
                        "key": resolved_surrogate.key,
                    }
                ),

                device=session.device,
                verify=effective_verify,

                llm4ed_api_key=(
                    cast(Llm4edConfig, session.llm4ed).api_key
                    if algorithm == "llm4ed"
                    else None
                ),
                surrogate_target=(
                    None
                    if network_target is None
                    else session.surrogates.file_for(network_target["surrogate_id"])
                ),
            )
            outcome = isolated.outcome
            if outcome.run_id is not None:
                session.search_verify[outcome.run_id] = deepcopy(effective_verify)
            result, summary = _outcome_envelope(
                outcome,
                provenance_params,
                session,
                lineage=lineage,
                time_cap_seconds=time_cap_seconds,
                stopped_by_time_cap=isolated.stopped_by_time_cap,
                post_seal_error=isolated.post_seal_error,
                evaluation_cap=max_evaluations,
                stopped_by_evaluation_cap=isolated.stopped_by_evaluation_cap,
                platform_nmse=isolated.platform_nmse,
                platform_coefficients=isolated.platform_coefficients,
                platform_eval_error=isolated.platform_eval_error,
                platform_eval_dropped=isolated.platform_eval_dropped,
                unavailable_reason=isolated.unavailable_reason,
                sketch=effective_sketch,
                surrogate=effective_surrogate,
            )



            result["diagnostics"].update(prior_diagnostics)
            if isolated.resources is not None:
                result["diagnostics"]["resources"] = isolated.resources
            result["provenance"]["execution_conditions"] = _execution_conditions(
                session.device
            )
            result["provenance"]["sketch_id"] = requested_sketch_id
            result["provenance"]["verify"] = deepcopy(effective_verify)
        except Exception as exc:







            provenance = {
                "instrument": algorithm,
                "dataset": dataset_id,
                "seed": seed,
                "lineage": lineage,
            }
            if provenance_params is not None:
                provenance["params"] = provenance_params
            return envelope(
                "failure",
                budget=session.budget(),
                diagnostics=prior_diagnostics,
                provenance=provenance,
                failures=[_crashed(exc)],
            )


        trained_surrogate: str | None = None
        try:
            trained_surrogate, network_diagnostics = _register_network(
                session, network_target, network_reason, isolated
            )
        except Exception as exc:
            result["failures"].append(_crashed(exc))
            network_diagnostics = {
                "surrogate_id": None,
                "reason": "not_registered",
                "detail": "the network was saved but its registration failed; "
                "the failure is reported above",
            }
        result["provenance"]["trained_surrogate"] = trained_surrogate
        if network_diagnostics is not None:
            result["diagnostics"]["surrogate"] = network_diagnostics
        selection_forward = _search_forward(
            session,
            result,
            run_id=outcome.run_id,
            run_dir=None if outcome.run_dir is None else Path(outcome.run_dir),
            dataset_id=dataset_id,
            input_source=input_source,
            launched_on=launched_on,
        )


        result["budget"] = session.budget()
        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        try:
            _record_segment(
                outcome,
                session,
                resume=resolved_resume,
                reseed=reseed,
                created_at=created_at,
                summary=summary,




                decision_parent=(
                    resume_from_run_id if decision_parent is None else decision_parent
                ),
                resume_iteration=resume_iteration,
                params=requested,
                report=result["diagnostics"].get("segment_report"),
                platform_nmse=isolated.platform_nmse,
                platform_frame=(
                    PLATFORM_FRAME if isolated.platform_nmse is not None else None
                ),
                platform_coefficients=isolated.platform_coefficients,
                sketch=effective_sketch,
                surrogate=effective_surrogate,
                cost=segment_cost(
                    outcome,
                    time_cap_seconds=time_cap_seconds,
                    stopped_by_time_cap=isolated.stopped_by_time_cap,
                    evaluation_cap=max_evaluations,
                    stopped_by_evaluation_cap=isolated.stopped_by_evaluation_cap,
                ),
                selection_forward=selection_forward,
                reason=reason,
                trained_surrogate=trained_surrogate,
            )
        except Exception as exc:
            result["failures"].append(_crashed(exc))
        try:
            _record_run(
                outcome,
                runs_root / CATALOG_FILENAME,
                resume=resolved_resume,
                created_at=created_at,
            )
        except Exception as exc:


            result["failures"].append(_crashed(exc))
        try:
            result["diagnostics"].update(_action_memory_diagnostics(session, result))
        except Exception as exc:
            result["failures"].append(_crashed(exc))
        if result["status"] != "success":
            return result









        auxiliary = pinned is not None and not _same_input(dataset_id, pinned)
        if not auxiliary:
            session.answer_dataset_id = dataset_id




        provenance = result["provenance"]
        own_provenance = {
            "instrument": provenance["instrument"],
            "dataset": provenance["dataset"],
            "run_id": provenance["run_id"],
            "run_dir": provenance["run_dir"],
            "nmse": result["diagnostics"]["nmse"],
        }
        session.successful_runs[provenance["run_id"]] = own_provenance
        session.sealed_lhs[provenance["run_id"]] = result["primary"]["law"]["lhs"]
        session.sealed_laws[provenance["run_id"]] = deepcopy(result["primary"]["law"])


        sealed_evidence = cast(Any, outcome.record).evidence
        session.sealed_fits[provenance["run_id"]] = deepcopy(
            cast(dict[str, Any], sealed_evidence.catalog_fit)
        )
        session.search_inputs[provenance["run_id"]] = launched_on
        provenance["input"] = {
            key: deepcopy(launched_on[key])
            for key in ("dataset_id", "file_sha256", "chain")
        }
        answer_law, answer_provenance, decision, answer_system = _selected_answer(
            session, cast(str, _answer_version(session)) if auxiliary else dataset_id
        )
        if answer_provenance is not None:




            session.answer_dataset_id = answer_provenance["dataset"]




        result["primary"]["now_selected"] = (
            None if answer_provenance is None else answer_provenance["run_id"]
        )
        if provenance["run_id"] not in {
            entry.run_id for entry in session.ledger.segments()
        }:







            result["failures"].append(
                _failure(
                    "ledger_unavailable",
                    "this search is not on the policy tree, so the default "
                    "selection was made without it; it can still be submitted. "
                    "The underlying ledger failure is reported above",
                )
            )


        return Command(
            update={
                "messages": [_tool_message(result, "run_discovery", tool_call_id)],
                ANSWER: _channel_answer(
                    answer_law, answer_provenance, decision, answer_system
                ),
            }
        )














































    def train_surrogate(
        dataset_id: str,
        algorithm: str,
        params: dict[str, Any] | None = None,
        seed: int | None = None,
        max_seconds: int | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Train the derivative surrogate an instrument fits before it searches,
        once per recipe, and register the file for `run_discovery(surrogate=)`.

        Args:
            dataset_id: Which dataset to train on. When this run is pinned to
                a dataset, any id other than it or an input `smooth_input`
                derived from it is refused.
            algorithm: Which instrument the surrogate is for. Only an
                instrument whose skill card has a "Surrogate" section takes
                one; the refusal names the instruments that do.
            params: The instrument's surrogate parameters, by these exact
                names: {surrogate_parameters}. The other instruments train no
                surrogate here: DISCOVER's network settings go inside
                `run_discovery(params={"pinn": {...}})`, and
                `refine_candidate` takes its own `pinn_epochs` and
                `pinn_pretrain_epochs`. For u_tt with dlga, pass
                target_lhs_order=2. Search parameters are refused here: they
                do not shape the network.
            seed: Random seed of the training. Omitted, it is 0. The seed is
                part of the recipe, so two seeds are two surrogates.
            max_seconds: Wall-clock cap for this training. A training has no
                iteration boundary to stop at: at the cap it is killed and
                nothing is kept, so give it what the epoch count needs, or
                lower the epoch count. Omitted, the run's remaining time
                budget is the cap (an exhausted budget is refused before
                anything starts); with no budget the training is uncapped.
            reason: Why you make this call, in your own words. It is
                recorded with the call on this workspace's policy tree
                (`tree.jsonl`) and shown in the report. Optional; an empty
                one is refused.

        The same recipe (dataset, instrument, seed, params) is trained once: a
        second call with the file still in place returns `primary.reused:
        true` and trains nothing, whatever the remaining budget (the cap
        applies to a training, not to a lookup). A network a dlga search
        trained to completion on the input as loaded, under the same
        settings and seed, counts as that training and is the id returned.
        Pass the returned
        `primary.surrogate_id` to
        `run_discovery(surrogate=...)`; for sga that search has to run with
        `params={"derivatives": "autograd"}`, the mode that consults it. A
        lineage keeps one surrogate: a resumed segment inherits it.

        Returns an envelope whose `primary` holds `surrogate_id` and `reused`,
        whose `diagnostics` hold `train_seconds`, `epochs`, `final_loss` and,
        for a training that ran in this call, a `resources` block measured in
        its worker (`device` and `cuda_max_memory_allocated_bytes`), and whose `provenance` names the instrument, dataset, seed, params,
        the config key the surrogate is injected under, and the file's sha256.
        Every envelope also carries the run's `budget` block.
        """
        reason_failure = _reason_refusal(reason)
        if reason_failure is not None:
            return envelope(
                "failure", budget=session.budget(), failures=[reason_failure]
            )
        started = session.workers_started
        return _logged(
            session,
            _train_surrogate(dataset_id, algorithm, params, seed, max_seconds),
            verb="train_surrogate",
            reason=reason,
            started=started,
        )

    def _train_surrogate(
        dataset_id: str,
        algorithm: str,
        params: dict[str, Any] | None,
        seed: int | None,
        max_seconds: int | None,
    ) -> dict[str, Any]:
        """The body of `train_surrogate`, every return an envelope."""
        requested = deepcopy(params) if params else {}
        provenance: dict[str, Any] = {
            "instrument": algorithm,
            "dataset": dataset_id,
            "seed": seed,
            "params": requested,
        }
        pinned = session.pinned_dataset_id
        if pinned is not None and not _same_input(dataset_id, pinned):
            return envelope(
                "failure",
                budget=session.budget(),
                provenance={**provenance, "pinned": pinned},
                failures=[
                    _failure(
                        "dataset_not_pinned",
                        f"this run is pinned to dataset {pinned!r} and can only "
                        "train on that one or an input smooth_input derived "
                        "from it",
                    )
                ],
            )
        unverified = _input_refusal(session, dataset_id)
        if unverified is not None:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance={**provenance, "pinned": pinned},
                failures=[unverified],
            )
        schema = schema_of(algorithm)
        if schema is None:
            known = [item["algorithm"] for item in instrument_schemas()]
            return envelope(
                "failure",
                budget=session.budget(),
                provenance=provenance,
                failures=[
                    _failure(
                        "unknown_instrument",
                        f"{algorithm!r} is not an instrument kd registers; the "
                        f"registered ones are {known}",
                    )
                ],
            )
        process_only = _process_only_refusal(algorithm)
        if process_only is not None:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance=provenance,
                failures=[process_only],
            )
        keys = schema["config_artifact_keys"]
        if not keys:
            injectable = [
                item["algorithm"]
                for item in instrument_schemas()
                if item["config_artifact_keys"]
            ]
            return envelope(
                "failure",
                budget=session.budget(),
                provenance=provenance,
                failures=[
                    _failure(
                        "surrogate_unsupported",
                        f"{algorithm} fits no derivative surrogate (its schema "
                        "declares no config_artifact_keys), so there is nothing "
                        f"to train; the instruments that take one are {injectable}",
                    )
                ],
            )
        key = surrogate_training.artifact_key(schema)
        fields = surrogate_training.surrogate_parameter_names(schema)
        extra = surrogate_training.unsupported_params(schema, requested)
        if extra:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance=provenance,
                failures=[
                    _failure(
                        "invalid_argument",
                        f"params {extra} are not surrogate parameters of "
                        f"{algorithm}; train_surrogate takes {fields} (and the "
                        "separate seed argument), the parameters that shape "
                        "the network. Search parameters go to run_discovery",
                    )
                ],
            )
        try:
            requested = surrogate_training.resolve_training_params(
                schema, requested, session.training_dtype
            )
        except ValueError as exc:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance=provenance,
                failures=[_failure("invalid_argument", str(exc))],
            )
        if max_seconds is not None and max_seconds <= 0:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance=provenance,
                failures=[
                    _failure(
                        "invalid_argument",
                        f"max_seconds has to be a positive number of seconds, "
                        f"got {max_seconds!r}; omit it to cap the training by "
                        "the run's remaining budget",
                    )
                ],
            )
        run_seed = surrogate_training.DEFAULT_SEED if seed is None else seed
        recipe = {
            "dataset_id": dataset_id,
            "algorithm": algorithm,
            "seed": run_seed,
            "params": requested,
        }
        provenance = {
            "instrument": algorithm,
            "dataset": dataset_id,
            "seed": run_seed,
            "params": requested,
            "key": key,
        }
        registry = session.surrogates
        try:




            hit, retrained_reason = surrogate_training.reuse_hit(
                registry, recipe, written=session.derived_inputs
            )
            if hit is not None:


                return envelope(
                    "success",
                    budget=session.budget(),
                    primary={"surrogate_id": hit.surrogate_id, "reused": True},
                    diagnostics={
                        "train_seconds": hit.train_seconds,
                        "epochs": hit.epochs,
                        "final_loss": hit.final_loss,
                    },
                    provenance={**provenance, "file_sha256": hit.file_sha256},
                )


            time_cap_seconds, exhausted = _training_cap(session, max_seconds)
            if exhausted is not None:
                return envelope(
                    "failure",
                    budget=session.budget(),
                    provenance=provenance,
                    failures=[exhausted],
                )
            training = surrogate_training.run_training(
                registry=registry,
                schema=schema,
                recipe=recipe,
                key=key,
                time_cap_seconds=time_cap_seconds,


                device=session.device,
                written=session.derived_inputs,
            )


            session.workers_started += 1
            if training.error_type is not None:
                if training.error_type == "SearchTimeout":
                    failure = _failure(
                        "training_timeout",
                        f"The training was terminated after exceeding the "
                        f"wall-clock cap of {time_cap_seconds} s. A surrogate "
                        "training has no cooperative stop, so nothing was "
                        "cached: raise max_seconds, or lower the epoch count in "
                        f"params (the surrogate parameters are {fields}), and "
                        "call train_surrogate again",
                    )
                else:
                    failure = _failure(training.error_type, str(training.error_message))


                return envelope(
                    "failure",
                    budget=session.budget(),
                    diagnostics=(
                        None
                        if training.resources is None
                        else {"resources": training.resources}
                    ),
                    provenance=provenance,
                    failures=[failure],
                )
            registered = surrogate_training.register_training(
                registry=registry, recipe=recipe, key=key, training=training
            )
        except Exception as exc:




            return envelope(
                "failure",
                budget=session.budget(),
                provenance=provenance,
                failures=[_crashed(exc)],
            )
        diagnostics: dict[str, Any] = {
            "train_seconds": registered.train_seconds,
            "epochs": registered.epochs,
            "final_loss": registered.final_loss,
            "resources": training.resources,
        }
        if retrained_reason is not None:
            diagnostics["retrained_reason"] = retrained_reason
        return envelope(
            "success",
            budget=session.budget(),
            primary={"surrogate_id": registered.surrogate_id, "reused": False},
            diagnostics=diagnostics,
            provenance={**provenance, "file_sha256": registered.file_sha256},
        )









































    def prune_node(
        run_id: str,
        reason: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> dict[str, Any] | Command[Any]:
        """Abandon a branch of the policy tree: this search, and everything
        resumed from it, stop being candidates for this run's answer.

        Args:
            run_id: The search to abandon, taken from that search's
                `provenance.run_id`. Any search on this workspace's tree can be
                abandoned, including one an earlier run left behind; one that
                is already abandoned, itself or through an ancestor, is refused
                rather than abandoned twice.
            reason: Why this branch is being given up, recorded on the tree
                beside the prune. One sentence, for whoever reads the tree
                afterwards.

        Abandoning is not deleting: the sealed run directory stays where it is
        and the row stays on the tree, marked. What changes is what may be
        published. `primary.now_selected` reports which search the policy holds
        once this one is gone, and is null when the default rule then has no
        answer (no publishable search left on this dataset, or none it can
        rank) -- this run then ends with no answer unless you submit one,
        select one or run another search.
        """
        ledger = session.ledger
        refusal = _node_refusal(session, run_id, action="abandoned again")
        if refusal is not None:
            return envelope("failure", budget=session.budget(), failures=[refusal])
        node = ledger.entry(run_id)
        standing = [
            system_row["system_id"]
            for system_row in ledger.systems()
            if not ledger.system_pruned(system_row["system_id"])
        ]
        try:
            ledger.prune(run_id, reason=reason)
        except Exception as exc:





            return envelope(
                "failure",
                budget=session.budget(),
                provenance=_entry_provenance(node),
                failures=[_crashed(exc)],
            )
        law: dict[str, Any] | None = None
        provenance: dict[str, Any] | None = None
        decision: dict[str, Any] | None = None
        system: dict[str, Any] | None = None
        recomputed = False











        answer_version = _answer_version(session)







        withdrew = any(ledger.system_pruned(system_id) for system_id in standing)
        if (
            node.dataset
            and answer_version is not None
            and (withdrew or _same_input(node.dataset, answer_version))
        ):
            law, provenance, decision, system = _selected_answer(
                session, answer_version
            )
            recomputed = True
        result = envelope(
            "success",
            budget=session.budget(),
            primary={
                "run_id": run_id,
                "reason": reason,
                "now_selected": None if provenance is None else provenance["run_id"],
            },
            provenance=_entry_provenance(node),
        )
        if not recomputed:
            return result
        return Command(
            update={
                "messages": [_tool_message(result, "prune_node", tool_call_id)],
                ANSWER: _channel_answer(law, provenance, decision, system),
            }
        )




























    def select_node(
        reason: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        run_id: str | None = None,
        system_id: str | None = None,
    ) -> dict[str, Any] | Command[Any]:
        """Name the search whose law is published as this run's answer if no
        answer is submitted, or the ODE system whose law for this run's target
        is.

        Args:
            reason: Why this search is the one to publish, recorded on the tree
                beside the selection. One sentence, for whoever reads the tree
                afterwards.
            run_id: The search to publish, taken from that search's
                `provenance.run_id`. It must be a search on this workspace's
                tree that sealed a structured law and, when this run is pinned
                to a dataset, one that searched that dataset. An abandoned
                search is refused.
            system_id: Instead of `run_id`, a system `verify_system` returned
                (`primary.system_id`): its member for this run's target is
                published, with the system beside it. A system with an
                abandoned member is refused. Give exactly one of the two.

        Use this when the searches disagree and the better score is not the
        better equation: without it, the run publishes whichever node the
        default rule holds. It does not end the run, and it is not a
        submission -- `submit_answer` still outranks it, and a later prune of
        the selected node (or of a member of the selected system) takes it
        back out of the running.
        """
        if (run_id is None) == (system_id is None):
            return envelope(
                "failure",
                budget=session.budget(),
                failures=[
                    _failure(
                        "invalid_argument",
                        "name exactly one of run_id (a search) and system_id (a "
                        "system verify_system returned)",
                    )
                ],
            )
        if system_id is not None:
            return _select_system(session, system_id, reason, tool_call_id)
        run_id = cast(str, run_id)
        ledger = session.ledger
        refusal = _node_refusal(
            session, run_id, action="published as this run's answer"
        )
        if refusal is not None:
            return envelope("failure", budget=session.budget(), failures=[refusal])
        entry = ledger.entry(run_id)
        law = entry.law
        if law is None:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance=_entry_provenance(entry),
                failures=[
                    _failure(
                        "no_structured_result",
                        f"run {run_id!r} sealed no structured law: it finished, "
                        "but its final evaluation produced no support to "
                        "publish, so there is nothing here to be this run's "
                        "answer. Select a search that returned status success, "
                        "or run another one",
                    )
                ],
            )
        off_pin = _off_pin_refusal(session, run_id, entry.dataset)
        if off_pin is not None:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance=_entry_provenance(entry),
                failures=[off_pin],
            )




        if run_id in session.successful_runs:
            unverified = _search_input_refusal(session, run_id)
        elif is_derived(entry.dataset):
            unverified = _unbound_search_refusal(run_id, entry.dataset)
        else:
            unverified = _input_refusal(session, entry.dataset)
        if unverified is not None:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance=_entry_provenance(entry),
                failures=[unverified],
            )
        try:
            ledger.select(run_id, reason=reason)
        except Exception as exc:



            return envelope(
                "failure",
                budget=session.budget(),
                provenance=_entry_provenance(entry),
                failures=[_crashed(exc)],
            )



        session.answer_dataset_id = entry.dataset
        provenance = _entry_provenance(entry)
        result = envelope(
            "success",
            budget=session.budget(),
            primary={"run_id": run_id, "reason": reason, "now_selected": run_id},
            provenance=provenance,
        )
        return Command(
            update={
                "messages": [_tool_message(result, "select_node", tool_call_id)],
                ANSWER: Answer(law=law, source="selected", provenance=provenance),
            }
        )

























































    def submit_answer(
        law: dict[str, Any],
        run_id: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        refinement_id: str | None = None,
        reason: str | None = None,
        system_id: str | None = None,
    ) -> dict[str, Any] | Command[Any]:
        """Submit the answer for this run.

        Args:
            law: The structured law, exactly the object `run_discovery` returns
                under `primary.law`: `support` (coefficient-free terms),
                `coefficients` and `lhs`. Not the human-readable expression
                string, which degrades into internal names when a search fails.
            run_id: The search this answer is based on, taken from that
                search's `provenance.run_id`. It must be a search from this
                run that returned `status: success`; any other id is refused
                and the eligible ones are listed. The law itself need not match
                that search's output term for term -- dropping a negligible
                term is fine -- but the answer must rest on a search that
                actually ran and succeeded, and `lhs` must be that search's
                `primary.law.lhs`, passed through unchanged.
            refinement_id: The `primary.refinement_id` a `refine_candidate`
                call on this same `run_id` returned, when `law` is the law it
                returned: the answer then carries that refinement's
                measurement. It must be that refinement's law exactly, term
                spellings included (`mul(u,u_x)` for a returned `u*u_x` is
                another law); anything else is refused.
            reason: Why you make this call, in your own words. It is
                recorded with the call on this workspace's policy tree
                (`tree.jsonl`) and shown in the report. Optional; an empty
                one is refused.
            system_id: A system `verify_system` returned, when the answer is
                that system's law for this run's target: `run_id` must then
                be the system's member for the target and `law` its sealed
                law unchanged, and the answer carries the system. Without
                it the answer names no system, even when a verified one
                holds the same law.

        Without `refinement_id`, a law exactly equal to one this run refined
        from that search takes the latest such refinement's measurement, and
        the search's own sealed law takes the search's; any other law, one
        that only spells a term another way included, is recorded as edited,
        with no measurement borrowed from either.
        `provenance.producer_ref` says which (`kind` `refinement`, `search` or
        `edited`), beside `cited_run_id`.

        A successful submission ends this run: no further model turn happens
        after it. Any other tool you called in the same turn still runs, because
        those calls were already issued. A refused submission does not end the
        run, so a failure can be fixed and submitted again.
        """
        shape_failure = _reason_refusal(reason) or _law_shape_failure(law)
        if shape_failure is not None:
            return envelope(
                "failure", budget=session.budget(), failures=[shape_failure]
            )
        if session.ledger.is_pruned(run_id):




            return envelope(
                "failure",
                budget=session.budget(),
                failures=[
                    _failure(
                        "pruned_run_id",
                        f"run {run_id!r} was pruned from the policy tree "
                        "(itself or an ancestor of it) and cannot be submitted "
                        "as the answer; submit a run that is still on the tree",
                    )
                ],
            )
        provenance = session.successful_runs.get(run_id)
        if provenance is None:
            return envelope(
                "failure",
                budget=session.budget(),
                failures=[_unknown_run_failure(session, run_id)],
            )
        off_pin = _off_pin_refusal(session, run_id, provenance["dataset"])
        if off_pin is not None:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance=provenance,
                failures=[off_pin],
            )
        unverified = _search_input_refusal(session, run_id)
        if unverified is not None:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance=provenance,
                failures=[unverified],
            )
        lhs_failure = _lhs_failure(law, session.sealed_lhs[run_id])
        if lhs_failure is not None:
            return envelope("failure", budget=session.budget(), failures=[lhs_failure])
        system: dict[str, Any] | None = None
        if system_id is not None:
            system, system_failure = _submitted_system(session, system_id, run_id, law)
            if system_failure is not None:
                return envelope(
                    "failure", budget=session.budget(), failures=[system_failure]
                )
        producer, nmse, producer_failure = _producer(
            session, law, run_id, refinement_id
        )
        if producer_failure is not None:
            return envelope(
                "failure", budget=session.budget(), failures=[producer_failure]
            )
        provenance = {
            **provenance,
            "nmse": nmse,
            "producer_ref": producer,
            "cited_run_id": run_id,
        }
        result = envelope(
            "success",
            budget=session.budget(),
            primary={"law": law},
            provenance=provenance,







            diagnostics={
                "recorded": True,
                "note": (
                    "This answer is recorded as the answer for this run, and "
                    "this run ends here."
                ),
            },
        )
        _note_action(
            session,
            result,
            verb="submit_answer",
            ref={
                "producer_ref": deepcopy(producer),
                **({} if system is None else {"system_id": system["system_id"]}),
            },
            run_id=run_id,
            reason=reason,
            status="success",
        )
        answer = Answer(law=law, source="submitted", provenance=provenance)
        if system is not None:
            answer["system"] = system
        return Command(
            update={
                "messages": [_tool_message(result, "submit_answer", tool_call_id)],
                ANSWER: answer,
            }
        )






















    def refine_candidate(
        run_id: str,
        law: dict[str, Any] | None = None,
        terms: list[str] | None = None,
        small_term_rtol: float = 0.0,
        savgol_window: int = 0,
        lowpass_modes: int | dict[str, int] = 0,
        time_window: int = 0,
        pinn_epochs: int = 0,
        pinn_pretrain_epochs: int = PINN_PRETRAIN_EPOCHS,
        pinn_residual_weight: float = 1.0,
        evaluation_input_id: str | None = None,
        method: str = "finite_diff",
        trim: Trim = 0,
        surrogate_id: str | None = None,
        weak: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Refit a candidate law's coefficients on the data of the search it
        comes from with its terms held fixed, and return a law that
        `submit_answer` accepts.

        Args:
            run_id: The search the candidate comes from, taken from that
                search's `provenance.run_id`; it must be a search that
                succeeded this run. Its `primary.law` is the candidate unless
                `law` or `terms` replaces it, its data are refitted, and its
                `lhs` is kept, so the refined law can be submitted with this
                same `run_id`.
            law: A law to refit instead, in the shape of `primary.law`
                (`support`, `coefficients`, `lhs`), for example that law with
                a term removed. Its `lhs` must be the search's
                `primary.law.lhs`, unchanged. Its terms are written as
                `terms` are.
            terms: A list of terms to refit instead, with no coefficients, so
                `small_term_rtol` has nothing to prune by and must stay 0.
                Each term is in KD's term notation: a name (`u_xx`), a
                product (`u*u_x`), a power (`u**2`) or KD's IR
                (`mul(u,u_x)`). The refined law keeps the terms as written
                and in this order, and a term counts as one of the search's
                when kd's law-term identity says so, whatever its spelling.
                Pass either `law` or `terms`, not both.
            small_term_rtol: Grid data only, in [0, 1). Before the refit,
                drop each term whose coefficient is smaller in magnitude than
                this fraction of the largest one; the largest term always
                stays, and 0 drops nothing.
            savgol_window: Grid data only. An odd window of 5 or more smooths
                every field with a Savitzky-Golay filter (polynomial order 3)
                along each axis before the refit; 0 refits the data as
                loaded. Not combined with `lowpass_modes`.
            lowpass_modes: Grid data only. 0, or a mode count in the form
                `smooth_input` takes (one positive integer, or a mapping from
                axis name to one): before the refit, every field is low-passed
                along each axis other than the left-hand side's to that many
                lowest spectral modes, the transform `smooth_input` applies.
                0 applies no low-pass. Not combined with `savgol_window`.
            time_window: Grid data only, with `lowpass_modes`. 0, or an odd
                window of 5 or more: a Savitzky-Golay filter (polynomial
                order 3) along the left-hand side's axis after the low-pass.
            pinn_epochs: Grid file inputs and a first-order time-derivative
                left-hand side only. 0, or a positive number of epochs:
                instead of the least-squares refit, a neural network is
                fitted to the data and to the candidate law at once, with the
                law's coefficients trained together with the network, and
                the coefficients it ends with are returned. The data are the
                input the search ran on, or the input its chain of derived
                ids starts from when that is a derived input. The network is
                first trained on the data alone for `pinn_pretrain_epochs`
                epochs, then for `pinn_epochs` epochs on the data misfit plus
                the mean squared residual of the law, with derivatives taken
                from the network (`kd.refine_pinn` at its other defaults, seed
                0, on the run's device). The candidate needs a number for
                every coefficient.
                Not combined with `terms`, `small_term_rtol`,
                `savgol_window` or `lowpass_modes`.
            pinn_pretrain_epochs: With `pinn_epochs`, the number of
                data-only epochs before the joint ones; default 5000.
            pinn_residual_weight: With `pinn_epochs`, the weight of the
                law's mean squared residual against the data misfit in the
                joint epochs, a finite number >= 0; default 1.0. At 0 the
                network is fitted to the data alone and the coefficients
                stay as given.
            evaluation_input_id: The input the refit is measured on.
                Omitted, the input the search ran on. Otherwise the root of
                that input or an input derived from that root this run (a
                low-passed copy, a network field); any other is refused.
                Not with `pinn_epochs`.
            method: `"finite_diff"` (default), `"autograd"` or `"weak"`.
                `"autograd"` takes field values and derivatives from the
                model `surrogate_id` names, which has to be trained on the
                measurement input itself (not on an input sampled from it).
                `"weak"` integrates the input against test-function windows
                (`kd-weak-v1`: one field, one spatial axis, the terms one,
                the field, its square and cube as `n2`/`n3` of it or as the
                raw products `mul(u,u)` and `mul(mul(u,u),u)`, its spatial
                derivatives of order 1 to 3 and `mul` of it with its first
                derivative); a term outside that class is refused, never
                fitted another way.
                Neither takes `savgol_window`, `lowpass_modes` or
                `pinn_epochs`; derive a smoothed input and measure that.
            trim: Finite differences only: 0 (default), a number of grid
                points to drop at both ends of every axis, a mapping from
                axis name to one, or `"by_order"` (the stencil footprint of
                the terms and the left-hand side). Derivatives are taken on
                the whole input first.
            surrogate_id: With `method="autograd"`, the model to
                differentiate: a `smooth_input(method="network")` or
                `train_surrogate` id.
            weak: With `method="weak"`, any of `half_widths` (axis name to
                a support half-width in grid points), `spectrum_tau`
                (default 0.03), `noise_correction` (`"none"` or
                `"gaussian"`) and `noise_sigma`.
            reason: Why you make this call, in your own words. It is
                recorded with the call on this workspace's policy tree
                (`tree.jsonl`) and shown in the report. Optional; an empty
                one is refused.

        On grid data the kept terms are fitted once on the measurement input
        in the frame `method` and `trim` choose (by default least squares on
        the whole grid with finite-difference derivatives, the frame of a
        search's `platform_nmse`). On a table the numeric constants inside
        the terms are refitted and the outer coefficients solved again;
        nothing is pruned, smoothed or framed, so those parameters keep
        their defaults. Nothing is searched, and the call counts against no
        evaluation cap. It runs in its own worker process under the run's
        remaining time.

        `run_discovery`'s `params.small_term_rtol` drops the same terms from
        the algorithm's own result without a refit; this tool's
        `small_term_rtol` drops them and then refits the rest.

        Returns an envelope whose `primary.law` is the refitted law
        (`support`, `coefficients` and the search's `lhs`), ready for
        `submit_answer(law=..., run_id=<the same run_id>,
        refinement_id=<primary.refinement_id>)`; `primary.refinement_id`
        names this refinement's receipt, which records the law, the source
        search, whether its support came from that search
        (`search-derived`) or not (`agent-proposed`), both inputs and the
        frame. `diagnostics` holds `nmse`, `frame` (a tag of what the number
        measures: untrimmed finite differences on unsmoothed grid data give
        the frame of a search's `platform_nmse`, a Savitzky-Golay window
        `kd-evaluate-terms-full-grid-savgol<window>-v1`, a low-pass
        `kd-evaluate-terms-full-grid-lowpass<modes>-t<time_window>-v1` with
        `<modes>` spelled as in a derived input's id, another method or a
        trim `kd-refit-terms-<method>[-trim<rows>]...-v1`),
        `original_coefficients` and `wallclock_seconds`; on grid data also
        `r2`, `mse`, `condition_number`, `terms_dropped` (each pruned term
        with its coefficient), `derivative_source`, `refit_frame` (kd's
        resolved frame: the input, the field source, the method, the trim
        per axis, the sample count) and `weak` (the windows and correction
        of a weak fit, null otherwise); on a table also `constants`
        (placeholder to fitted value) and `initial` (placeholder to the
        number it replaced). A number that is not finite is null.
        `provenance` names the source `run_id`, `refined_from`, the
        parameters and the `evaluation_input`.

        With `pinn_epochs`, the network's prediction of every field on the
        input's grid is also written as a new file input, with no axis
        flagged periodic (the network does not wrap an axis), `primary.dataset_id`
        `<input id>+pinn<pinn_epochs>-<digest>` (the digest covers the terms,
        the starting coefficients, both epoch counts, the residual weight and
        the seed), with
        `primary.derived_from` the input it was fitted to. It is accepted
        wherever that input is, like an input `smooth_input` writes, and the
        same arguments return the same id with `primary.reused` true and fit
        nothing; when that input no longer verifies the call is refused with
        `input_unverified` and nothing is fitted again. `diagnostics` then
        holds `frame`
        `kd-refine-pinn-e<pinn_epochs>-p<pinn_pretrain_epochs>-w<pinn_residual_weight>-v2`,
        `nmse`
        null, `initial_data_loss` and `initial_residual_loss` (the mean
        squared misfit to the data and the mean squared residual of the law
        with the starting coefficients, after the data-only epochs),
        `data_loss` and `residual_loss` (the same for the returned state),
        `epochs_run`, `stopped_at_deadline` (whether the run's remaining time
        ended the joint epochs early), `device`, `derivative_source`
        `autograd`, `original_coefficients` and, for a fresh fit,
        `wallclock_seconds` and the `resources` block measured in the fit's
        worker (`device` and `cuda_max_memory_allocated_bytes`); `provenance`
        adds the new file's path and sha256.
        """
        reason_failure = _reason_refusal(reason)
        if reason_failure is not None:
            return envelope(
                "failure", budget=session.budget(), failures=[reason_failure]
            )
        started = session.workers_started
        result = _refine_candidate(
            session,
            run_id,
            law=law,
            terms=terms,
            smoothing={
                "small_term_rtol": small_term_rtol,
                "savgol_window": savgol_window,
                "lowpass_modes": lowpass_modes,
                "time_window": time_window,
            },
            pinn={
                "pinn_epochs": pinn_epochs,
                "pinn_pretrain_epochs": pinn_pretrain_epochs,
                "pinn_residual_weight": pinn_residual_weight,
            },
            evaluation_input_id=evaluation_input_id,
            frame=(method, trim, weak, surrogate_id),
        )
        return _logged(
            session,
            result,
            verb="refine_candidate",
            reason=reason,
            started=started,
            run_id=run_id,
        )





























    def smooth_input(
        dataset_id: str,
        lowpass_modes: int | dict[str, int] | Literal["auto"] | None = None,
        time_window: int = 0,
        method: Literal["lowpass", "network"] = "lowpass",
        surrogate_id: str | None = None,
        network_recipe: dict[str, Any] | None = None,
        seed: int = 0,
        terms: list[str] | None = None,
        tolerance: float = 0.02,
        max_counts: int = 8,
        max_seconds: int | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Write a smoothed copy of a grid file input as a new file input and
        register it under an id derived from the parent's: a low-passed copy
        (`method="lowpass"`), or the field of a neural network fitted to the
        data (`method="network"`).

        Args:
            dataset_id: The file input to smooth, a `file-v1-...` id with a
                time-derivative left-hand side, or an id `smooth_input`
                derived from one. A catalog dataset is refused, and so is an
                input `refine_candidate` derived with `pinn_epochs` or one
                derived from it. When this run is pinned, it has to be the
                pinned input or an input derived from it.
            lowpass_modes: Low-pass only. How many of the lowest spectral
                modes to keep along each axis other than the left-hand
                side's: one positive integer for every such axis, or a
                mapping from axis name to a positive integer, at most the
                axis length; or `"auto"`, which lets kd choose the counts
                from the spectrum and from refits of `terms`. Along a
                periodic axis the transform is the real Fourier transform,
                otherwise the discrete cosine transform; the coefficients
                above the count are set to zero and the field is transformed
                back.
            time_window: Low-pass only. 0, or an odd window of 5 or more: a
                Savitzky-Golay filter (polynomial order 3) applied along the
                left-hand side's axis after the spectral pass.
            method: `"lowpass"` (default) or `"network"`.
            surrogate_id: Network only. A model already registered for this
                input (a `smooth_input` network or a `train_surrogate`
                surrogate trained on it): its field is sampled and nothing
                is trained. Not with `network_recipe` or `seed`.
            network_recipe: Network only. The training recipe, every field
                of kd's `FieldRecipe`: `name`, `version`, `hidden_sizes`
                (a list of layer widths), `activation` (`"tanh"`, `"sin"` or
                `"relu"`), `lr`, `max_epochs`, `patience` (null: none),
                `val_ratio`, `restore_best`. Omitted, kd's `dlga_nn_1`
                (five sine layers of 50, 50,000 epochs), which takes a long
                time on a CPU; give `max_seconds` what the epochs need.
            seed: Network only. The training seed; another seed is another
                network.
            terms: With `lowpass_modes="auto"`: the candidate terms whose
                refitted coefficients the choice watches, in KD's term
                notation (as in `refine_candidate`).
            tolerance: With `"auto"`: the largest relative change of those
                coefficients between two counts at which a count is chosen.
            max_counts: With `"auto"`: the most counts tried per axis.
            max_seconds: Wall-clock cap of this call. A network training is
                killed at it and keeps nothing; omitted, the run's remaining
                time.
            reason: Why you make this call, in your own words. It is
                recorded with the call on this workspace's policy tree
                (`tree.jsonl`) and shown in the report. Optional; an empty
                one is refused.

        A low-passed copy keeps the parent's axes, field names, shape,
        left-hand side and periodic flags; its id is
        `<dataset_id>+lowpass<modes>-t<time_window>`, where `<modes>` is the
        integer, or `axis=count` pairs in axis-name order joined by commas
        (with `"auto"`, the counts kd chose). A network field keeps the
        axes, field and left-hand side but flags no axis periodic (the
        network does not wrap an axis; the source's periodicity is in the
        preparation identity); its id is `<dataset_id>+field-<surrogate_id>`.
        A network is trained once per input, recipe and seed: the same
        request returns the same ids with `primary.reused` true and trains
        nothing; a model file that changed is refused and not trained again.
        The trained model is bound to the input it was fitted to, and
        `refine_candidate(method="autograd", surrogate_id=...)` measures
        that input with it; it is not a surrogate any search instrument
        takes. The same arguments name the same input: a second call returns
        the same id with `primary.reused` true and writes nothing. The
        derived id is accepted wherever the parent's is (`run_discovery`,
        `train_surrogate`, and through a search's `run_id` `refine_candidate`
        and `submit_answer`), and `list_datasets` lists it with its
        `derived_from`.

        Returns an envelope whose `primary` holds `dataset_id` (the derived
        id), `derived_from` and `reused`. A low-pass adds `lowpass_modes`,
        `time_window` and `energy_kept`: for each smoothed axis, the fraction
        of the left-hand side field's spectral energy held by the kept modes,
        averaged over the other axes; `diagnostics.spectrum` gives, for each
        of those axes, `n` (the axis length), `energy_kept` again,
        `modes_at_0.99` and `modes_at_0.999` (the smallest mode count whose
        kept fraction reaches that value) and `modes_at_knee` (the first
        count whose remaining fraction, 1 minus the kept one, is within 1.5
        times the remaining fraction at n/8 modes), and with `"auto"`
        `diagnostics.selection` the counts chosen per axis with kd's whole
        trace (every count tried, its refitted coefficients and their
        change). A network adds `surrogate_id` and `method`;
        `diagnostics.preparation` is kd's preparation identity of a network
        trained here (the input's fingerprint and field hash, axis and field
        order, dtype, source periodicity, recipe, seed, model hash) and
        `diagnostics.quality` its training report (requested and executed
        epochs, stopping reason, the losses and their scale, a noise
        estimate per field and the residual over it, the device and its
        CUDA peak); none of these says the derivatives are accurate. A fresh
        call also reports `wallclock_seconds`. `provenance` holds the parent
        id and file sha256, `root` (the input the chain of derived ids
        starts from) and `chain` (the ids from the root to the parent, each
        with its file sha256), and the new file's path and sha256 (for a
        network also the model file's). A derived input is accepted only in
        the run that wrote it, and it is checked against the file written,
        the model it was sampled from and the input it names on every use;
        one that no longer matches is refused with `input_unverified`, and so
        is a repeated call on it, which writes nothing. It runs in worker
        processes stopped at `max_seconds`, or without it at the run's
        remaining time.
        """
        reason_failure = _reason_refusal(reason)
        if reason_failure is not None:
            return envelope(
                "failure", budget=session.budget(), failures=[reason_failure]
            )
        started = session.workers_started
        result = _smooth_input(
            session,
            dataset_id,
            lowpass_modes=lowpass_modes,
            time_window=time_window,
            method=method,
            surrogate_id=surrogate_id,
            network_recipe=network_recipe,
            seed=seed,
            terms=terms,
            tolerance=tolerance,
            max_counts=max_counts,
            max_seconds=max_seconds,
        )
        return _logged(
            session, result, verb="smooth_input", reason=reason, started=started
        )




    def ablate_candidate(
        run_id: str,
        law: dict[str, Any] | None = None,
        input_id: str | None = None,
        method: str = "finite_diff",
        weak: dict[str, Any] | None = None,
        max_seconds: int = 60,
        trim: Trim = 0,
        surrogate_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Refit a search's law once with all its terms and once without
        each term, in one measurement frame, to see how much each term
        carries.

        Args:
            run_id: The search whose law is measured, a search that
                succeeded this run (its `provenance.run_id`).
            law: A law to measure instead, in the shape of `primary.law`,
                with that search's `lhs`; its terms are in KD's term notation
                (as in `refine_candidate`), and each row names its term as
                written.
            input_id: The input to measure on: omitted, the input the search
                ran on; otherwise its root or an input derived from that
                root this run. Any other is refused.
            method: `"finite_diff"` (default), `"autograd"` (with
                `surrogate_id`, a model trained on this input) or `"weak"`
                (test-function windows; a term outside kd's weak class is
                refused, never fitted another way).
            weak: With `method="weak"`: `half_widths`, `spectrum_tau`,
                `noise_correction`, `noise_sigma`.
            max_seconds: Wall-clock cap of the call; it is stopped at it and
                keeps nothing.
            trim: Finite differences only: 0, an integer, a per-axis
                mapping or `"by_order"`.
            surrogate_id: With `method="autograd"`, the model to
                differentiate.
            reason: Why you make this call, in your own words. It is
                recorded with the call on this workspace's policy tree
                (`tree.jsonl`) and shown in the report. Optional; an empty
                one is refused.

        Every reduced fit reuses the baseline's rows or windows, target and
        normalization, so the rises compare. Returns an envelope whose
        `primary` holds `evidence_id` (citable by `make_sketch`), `law`,
        `baseline` (the full fit: coefficients, `mse`, `nmse`, `r2`,
        `condition_number`), `rows` (per dropped term: the retained terms,
        their coefficients, `mse`, `nmse`, `delta_mse` and `relative_rise`,
        the rise over the baseline MSE) and `relative_rise_reason`
        (`zero_baseline` when the baseline fits exactly and no relative
        rise is defined, so every `relative_rise` is null). A reduced fit
        that fails fails the call, naming the term; it is never a zero
        importance. `diagnostics.frame` is kd's resolved frame and
        `diagnostics.weak` the weak windows. A large rise says a term
        carries the fit in this frame, not that it is physically present.
        """
        reason_failure = _reason_refusal(reason)
        if reason_failure is not None:
            return envelope(
                "failure", budget=session.budget(), failures=[reason_failure]
            )
        started = session.workers_started
        result = _ablate_candidate(
            session,
            run_id=run_id,
            law=law,
            input_id=input_id,
            frame=(method, trim, weak, surrogate_id),
            max_seconds=max_seconds,
        )
        return _logged(
            session,
            result,
            verb="ablate_candidate",
            reason=reason,
            started=started,
            run_id=run_id,
        )

    def score_term_sets(
        run_id: str,
        term_sets: list[list[str]],
        input_id: str | None = None,
        method: str = "finite_diff",
        trim: Trim = 0,
        surrogate_id: str | None = None,
        weak: dict[str, Any] | None = None,
        max_seconds: int = 60,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Fit several fixed term sets on the input of a search, all in one
        measurement frame resolved for their union, in one call.

        Args:
            run_id: The search whose input is measured, a search that
                succeeded this run.
            term_sets: The supports to fit, each a non-empty list of terms
                in KD's term notation (as in `refine_candidate`). Each entry
                keeps its terms as written; a set with a term kd cannot read
                fails alone, and the other sets are fitted as they would be
                on their own.
            input_id: As in `ablate_candidate`.
            method: As in `ablate_candidate`.
            trim: As in `ablate_candidate`.
            surrogate_id: As in `ablate_candidate`.
            weak: As in `ablate_candidate`.
            max_seconds: Wall-clock cap of the call.
            reason: As in `ablate_candidate`.

        Returns an envelope whose `primary` holds `evidence_id` and
        `entries`, one per set in order: `terms`, `fit` (coefficients,
        `mse`, `nmse`, `r2`, `condition_number` and `weak`, a weak fit's
        windows, rank and column count, null otherwise) or null, `failure` and
        `failure_kind` (`refused`: a term the frame rejects; `fit_failed`:
        the fit failed). A set with no fit has no score; it is not a zero.
        `diagnostics.frame` is the one frame every set was fitted in.
        """
        reason_failure = _reason_refusal(reason)
        if reason_failure is not None:
            return envelope(
                "failure", budget=session.budget(), failures=[reason_failure]
            )
        started = session.workers_started
        result = _score_term_sets(
            session,
            run_id=run_id,
            term_sets=term_sets,
            input_id=input_id,
            frame=(method, trim, weak, surrogate_id),
            max_seconds=max_seconds,
        )
        return _logged(
            session,
            result,
            verb="score_term_sets",
            reason=reason,
            started=started,
            run_id=run_id,
        )

    def verify_candidate(
        run_id: str,
        law: dict[str, Any] | None = None,
        input_id: str | None = None,
        max_seconds: int = 30,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Check a search's law three separate ways on one input: whether
        each term can be evaluated, how much its coefficients spread when it
        is refitted on separate grid blocks, and how far integrating it
        forward from the first time slice ends from the data.

        Args:
            run_id: The search whose law is checked, a search that succeeded
                this run.
            law: A law to check instead, in the shape of `primary.law`, with
                that search's `lhs` and a number for every coefficient; its
                terms are in KD's term notation (as in `refine_candidate`).
            input_id: As in `ablate_candidate`.
            max_seconds: The time the call may take. The forward
                integration stops at what is left of it and reports
                `reason: "timeout"` (not measured, which is not an infinite
                error); the call is killed at a hard limit above it.
            reason: As in `ablate_candidate`.

        Returns an envelope whose `primary` holds `evidence_id`, the `law`
        checked, `validation` (per term `ok` and the reason a term is
        rejected), `dispersion` (`coefficient_of_variation` across
        `n_blocks` blocks) and `forward` (`value`, the relative L2 error of
        the integrated field, or null with `reason` `timeout`, `diverged`,
        `solve-failed` or `refused`; `interval`, `boundary`, the
        coefficients and the input it integrated). A measurement that
        could not be made (out of memory, an input it does not support) is
        listed under `failures` and the status is `partial`. These are
        discrepancy measurements: KD gives no rule on when they establish
        a law.
        """
        reason_failure = _reason_refusal(reason)
        if reason_failure is not None:
            return envelope(
                "failure", budget=session.budget(), failures=[reason_failure]
            )
        started = session.workers_started
        result = _verify_candidate(
            session, run_id=run_id, law=law, input_id=input_id, max_seconds=max_seconds
        )
        return _logged(
            session,
            result,
            verb="verify_candidate",
            reason=reason,
            started=started,
            run_id=run_id,
        )














    def verify_system(
        members: dict[str, str],
        tool_call_id: Annotated[str, InjectedToolCallId],
        input_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any] | Command[Any]:
        """Integrate the laws found for every state of raw ODE trajectories
        together, forward in time from each trajectory's first observation
        to its last, and compare the result with the observations.

        Args:
            members: One search per state, keyed by the target that search
                fitted (`dx_dt` for state `x`), each value a `run_id` of a
                search that succeeded this run on a projection of these
                trajectories (`run_discovery` with `target_state`). Every
                state needs exactly one; the searches' sealed laws are
                integrated with their sealed coefficients, nothing is
                refitted.
            input_id: The input whose raw trajectories are integrated. On a
                pinned run it is the pinned input, and may be omitted; on an
                unpinned run it is required, and the answer is then about
                that input's target.
            reason: As in `ablate_candidate`.

        Returns an envelope whose `primary` holds `system_id` (the same
        searches are always the same system; another search for any state
        is another system), `verification_id` (this attempt), `members`,
        `laws`, `raw_fingerprint` and `forward`: kd's `value` (the relative
        L2 error over every trajectory, state and time point), `per_target`,
        `per_trajectory`, `intervals` (each trajectory's integration
        interval), `solver`, and `reason` with null values when kd could not
        integrate (`refused`, `diverged`, `solve-failed`) or ran out of time
        (`timeout`, not measured). `standing_verification_id` is the
        verification the system's current result stands on: a timeout or a
        failed worker does not replace an earlier result. The integration
        has 10 s within a 30 s worker, never more than the run has left.

        Once a system of this run's raw trajectories has a result, the
        systems decide the default answer: among the systems whose error is
        at most twice the smallest, the one with the fewest terms over all
        states wins (then the smaller error, then the earlier system), and
        its member for this run's target is published with the system; when
        none of them has a finite error there is no default answer. Until
        then the searches of the target decide it as before. The error
        compares the laws with the observations they were found on; it is
        not a held-out test.
        """
        reason_failure = _reason_refusal(reason)
        if reason_failure is not None:
            return envelope(
                "failure", budget=session.budget(), failures=[reason_failure]
            )
        started = session.workers_started
        result, recorded = _verify_system(
            session, members=members, input_id=input_id, reason=reason
        )
        _logged(session, result, verb="verify_system", reason=reason, started=started)
        if not recorded or result["status"] == "failure":
            return result
        if session.pinned_dataset_id is None:
            session.answer_dataset_id = result["provenance"]["input_id"]
        law, provenance, decision, system = _selected_answer(
            session, cast(str, _answer_version(session))
        )
        result["primary"]["now_selected"] = (
            None if provenance is None else provenance["run_id"]
        )
        return Command(
            update={
                "messages": [_tool_message(result, "verify_system", tool_call_id)],
                ANSWER: _channel_answer(law, provenance, decision, system),
            }
        )

    def vote_terms(
        run_ids: list[str],
        min_count: int,
        small_term_rtol: float = 0.0,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Count how many independent searches found each term.

        Args:
            run_ids: Searches that succeeded this run, on one input or
                inputs derived from one root. One search is one vote: a
                repeated id, or two segments of one lineage (one resumed
                from the other), is refused.
            min_count: How many searches a term needs to be selected,
                between 1 and the number of searches.
            small_term_rtol: In [0, 1). A term whose recorded coefficient
                is smaller in magnitude than this fraction of the largest in
                its law is not counted for that search (0 counts every
                term); a cutoff in the data's units, so it means different
                things on differently scaled data.
            reason: Why you make this call, in your own words. It is
                recorded with the call on this workspace's policy tree
                (`tree.jsonl`) and shown in the report. Optional; an empty
                one is refused.

        Terms are counted by kd's law-term identity, so two spellings of one
        term (`mul(u,u_x)`, `mul(u_x,u)`) are one term. Returns an envelope
        whose `primary` holds `evidence_id` (citable by `make_sketch`),
        `selected` (the terms that reached `min_count`, possibly none) and
        `rows`: per term `term`, `count`, and the `run_ids`, `seeds` and
        `lineages` behind the count. A count says a term recurs; it is not
        evidence that the term is physically present.
        """
        reason_failure = _reason_refusal(reason)
        if reason_failure is not None:
            return envelope(
                "failure", budget=session.budget(), failures=[reason_failure]
            )
        started = session.workers_started
        return _logged(
            session,
            _vote(session, run_ids, min_count, small_term_rtol),
            verb="vote_terms",
            reason=reason,
            started=started,
        )

    def make_sketch(
        evidence_ids: list[str],
        anchor_term_keys: list[str],
        min_additional_terms: int,
        max_additional_terms: int,
        max_derivative_order: int,
        input_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Have KD build a Sketch that requires the terms you choose, and
        register it for `run_discovery(sketch_id=...)`.

        Args:
            evidence_ids: The `evidence_id`s of votes or measurements this
                run returned that support the anchors.
            anchor_term_keys: The terms every law of the narrowed search
                must contain, each one a term of the cited evidence, in KD's
                term notation (as in `refine_candidate`). An anchor matches
                an evidence term by kd's law-term identity, so `u*u_x` cites
                evidence that recorded `mul(u,u_x)`, and the reverse.
            min_additional_terms: Fewest terms beyond the anchors.
            max_additional_terms: Most terms beyond the anchors.
            max_derivative_order: Highest total derivative order of those
                additional terms.
            input_id: The input the Sketch is built and checked on. Omitted,
                the one input the cited evidence was measured on; required
                when it spans several inputs of one root.
            reason: Why you make this call, in your own words. It is
                recorded with the call on this workspace's policy tree
                (`tree.jsonl`) and shown in the report. Optional; an empty
                one is refused.

        Returns an envelope whose `primary` holds `sketch_id`, the `sketch`
        payload KD built, its `sketch_sha256` and this registration
        (`origin` `agent`, the `evidence_ids` it cites, `input_id`, `reason`,
        `created_at`). A different choice of anchors or bounds is a
        different `sketch_id`, so a Sketch is revised by making another.
        It is not made active: name it with `run_discovery(sketch_id=...)`,
        which is refused while another Sketch is active (`set_sketch`).
        Anchors are a hypothesis to search under, not established terms.
        """
        reason_failure = _reason_refusal(reason)
        if reason_failure is not None:
            return envelope(
                "failure", budget=session.budget(), failures=[reason_failure]
            )
        started = session.workers_started
        result = _make_sketch(
            session,
            evidence_ids=evidence_ids,
            anchors=anchor_term_keys,
            bounds={
                "min_additional_terms": min_additional_terms,
                "max_additional_terms": max_additional_terms,
                "max_derivative_order": max_derivative_order,
            },
            input_id=input_id,
            reason=reason,
        )
        return _logged(
            session, result, verb="make_sketch", reason=reason, started=started
        )

    def set_sketch(
        anchored: list[str],
        pinned: dict[str, float],
        holes: list[dict[str, Any]],
        input_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Make the Sketch of the terms the researcher states the constraint
        of every later search: KD binds them on the input and registers the
        bound Sketch, and each search that names no Sketch runs under it.

        Args:
            anchored: Terms every law must contain, with fitted
                coefficients, in KD's term notation: a name (`u`, `u_xx`),
                a product (`u*u_x`), a power (`u**2`) or KD's IR
                (`mul(u,u_x)`). Pass an empty list for none.
            pinned: Terms with a fixed coefficient, as term to coefficient.
                Pass an empty mapping for none.
            holes: Room for terms you do not name, each an object with
                `max_terms` and optionally `min_terms` (default 1),
                `max_deriv_order`, `operators` (law feature names such as
                `mul`) and `fields`. Pass an empty list for none. Not every
                list may be empty.
            input_id: The input the Sketch is bound to, whose left-hand side
                and vocabulary it takes. Omitted, the pinned input; a run
                pinned to none has to name it.
            reason: What the researcher said that this Sketch states, in
                your own words. It is recorded with the call on this
                workspace's policy tree (`tree.jsonl`) and shown in the
                report. Optional; an empty one is refused.

        Returns an envelope whose `primary` holds `sketch_id`, the `sketch`
        payload KD bound, its `sketch_sha256` and this registration
        (`origin` `conversation`, no `evidence_ids`, `input_id`, `reason`,
        `created_at`). That
        Sketch is then active: a search that names none, on its input or one
        derived from it, runs under it, a search naming another Sketch is
        refused (`sketch_conflict`), a search on another input is refused
        until a Sketch is set there, and a resumed search whose lineage ran
        under another Sketch is refused. The next `set_sketch` replaces it;
        one that fails leaves it as it was. Searches that ran before keep
        the Sketch they ran under, and so does an earlier answer. When the
        run was given a user Sketch, only that Sketch can be set.
        """
        reason_failure = _reason_refusal(reason)
        if reason_failure is not None:
            return envelope(
                "failure", budget=session.budget(), failures=[reason_failure]
            )
        started = session.workers_started
        result = _set_sketch(
            session,
            clauses={"anchored": anchored, "pinned": pinned, "holes": holes},
            input_id=input_id,
            reason=reason,
        )
        return _logged(
            session, result, verb="set_sketch", reason=reason, started=started
        )

    def finish_without_answer(
        reason: str,
        evidence_run_ids: list[str],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> dict[str, Any] | Command[Any]:
        """End this run with no answer, saying why: the result is recorded
        as unresolved rather than as a law the evidence does not support.

        Args:
            reason: Why the searches and checks do not support an answer.
            evidence_run_ids: The searches the reason rests on, from their
                `provenance.run_id`; each must be on this workspace's
                policy tree.
        """
        return _finish_without_answer(session, reason, evidence_run_ids, tool_call_id)

    run_discovery.__doc__ = (
        cast(str, run_discovery.__doc__)
        .replace("{llm4ed}", _llm4ed_description(session.llm4ed))
        .replace(
            "{algorithms}",
            _description_paragraph(registered_algorithms(instrument_schemas())),
        )
    )


    train_surrogate.__doc__ = cast(str, train_surrogate.__doc__).replace(
        "{surrogate_parameters}",
        "; ".join(
            f"{schema['algorithm']}: "
            + ", ".join(
                f"`{name}`"
                for name in surrogate_training.surrogate_parameter_names(schema)
            )
            for schema in instrument_schemas()
            if schema["config_artifact_keys"]
        ),
    )
    return [
        list_datasets,
        run_discovery,
        train_surrogate,
        prune_node,
        select_node,
        submit_answer,
        refine_candidate,
        smooth_input,
        ablate_candidate,
        score_term_sets,
        verify_candidate,
        verify_system,
        vote_terms,
        make_sketch,
        set_sketch,
        finish_without_answer,
    ]
