"""kd's public surface -> tool objects. Narrowing and shaping both happen here.

Narrowing = kd's three public surfaces add up to well over a hundred names, and
this layer admits six verbs (`list_datasets`, `run_discovery`,
`train_surrogate`, `prune_node`, `select_node`, `submit_answer`).
Shaping = every tool returns the same envelope, with fixed field names:

    status success | partial | failure
    primary display (the human-readable expression) + law (structured
                 terms, coefficients and the LHS the record names)
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

`run_discovery` and `submit_answer` return a `Command` on success, writing the
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

import json
import math
import os
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command

from kd import (
    get_dataset,
    instrument_schemas,
    list_datasets_answer_blind,
    preview_report,
)

from kd.harness import SKETCH_SIDECAR_FILENAME, PlanEntry
from kd.search import (
    CATALOG_FILENAME,
    DEFAULT_RUNS_ROOT,
    RunDirPaths,
    new_run_id,
)
from kdagent import criteria, priors, surrogate_training
from kdagent.episode_worker import run_isolated_episode
from kdagent.lineage import PLATFORM_FRAME, TREE_FILENAME, SegmentEntry, SegmentSummary
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
)
from kdagent.segment_report import build_segment_report, law_from_evidence
from kdagent.session import Session
from kdagent.data_source import is_file_ref, resolve_input
from kdagent.state import ANSWER, Answer
from kdagent.surrogates import file_sha256

DEFAULT_SEED = 0
DEFAULT_GENERATIONS = 10
"""The budget this tool applies for a caller that omitted ``generations``.

Smaller than kd's own facade default of 50, to bound the cost of one call.
**The card must report this number**: that table is about `run_discovery`'s
``params``, and reporting 50 would have the model planning against five times
the budget.
"""


def _law_shape_failure(law: dict[str, Any]) -> dict[str, Any] | None:
    """Say what is wrong when `law` is not the shape `run_discovery` hands out;
    `None` when it is.

    The shape is defined by `run_discovery`'s `success` envelope: `support` is a
    non-empty list of term names; `lhs` (the sealed record's LHS, or null) is
    passed through and not checked; `coefficients` is either absent (kd drops it
    entirely when coefficients and support do not line up, with `is_valid` still
    true -- `record_assembly.active_support_and_coefficients`) or the same
    length as `support`, each position a finite number or null. This is tighter
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


def _selected_answer(
    session: Session, dataset_id: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The law and provenance of the node the policy holds right now.

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

    Both values are None when the ledger has nothing to offer -- the caller
    degrades to this search's own result and says so.
    """
    ledger = session.ledger
    selected = ledger.selected
    explicit = None if selected is None else ledger.entry(selected)
    if explicit is None or explicit.law is None or explicit.dataset != dataset_id:
        selected = criteria.default_selection(ledger, dataset_id=dataset_id)
    if selected is None:
        return None, None
    entry = ledger.entry(selected)
    return entry.law, _entry_provenance(entry)


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
        "platform_frame": PLATFORM_FRAME,
        "platform_coefficients": platform_coefficients,
        "platform_eval_dropped": platform_eval_dropped,
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
    return envelope(
        "success",
        budget=budget,
        primary={
            "display": evidence.expression,




            "law": law_from_evidence(evidence),
        },
        diagnostics=diagnostics,
        provenance=provenance,
        failures=report_failures,
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
    anchored / holes), the law terms outside every clause, and kd's failure
    string when it produced no verdict. The sidecar also carries the found
    equation as `solution` / `best_candidate` / `full_verify`; that is a
    discovered law, not ground truth, but it is not poured into the envelope
    by dumping the file either.
    """
    path = run_dir / SKETCH_SIDECAR_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = payload["verdict"]
    certified = payload["solution"] is not None
    if verdict is None:
        return {
            "satisfied": False,
            "certified": certified,
            "failure": payload["failure"],
        }
    return {
        "satisfied": verdict["overall"],
        "certified": certified,
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


_PARAM_TABLE_HEAD = (
    "| Parameter | Type | Default | Allowed | Resume | Notes |",
    "|---|---|---|---|---|---|",
)
_EFFECT_NOTES = {
    "max_iterations": "search-loop length (the runner's max_iterations)",
    "unused": "**this algorithm consumes it nowhere** -- do not pass it",
}


def _cell(value: Any) -> str:
    """Render one JSON value for a markdown cell."""
    return "" if value is None else f"`{json.dumps(value)}`"


def _tier_cell(tier: str | None) -> str:
    """Render a knob's resume tier; empty when kd classifies that name nowhere.

    Empty is the honest cell, not a gap to be filled here: only some of each
    algorithm's parameter surface is a descriptor knob (9 of sga's 20 rows), and
    inventing a tier for the rest would be this layer asserting something kd has
    not decided.
    """
    return "" if tier is None else f"`{tier}`"


def _text_cell(text: str) -> str:
    """Escape a markdown cell's separator.

    Union annotations carry a literal ``|`` (``dict[str, Any] | None``), which
    silently splits the row into an extra column.
    """
    return text.replace("|", r"\|")


def _parameters_markdown(schema: dict[str, Any]) -> str:
    """Render one algorithm's ``params`` surface as a markdown table.

    Two halves, because config field names alone do not describe the surface:
    the facade's own parameters (``generations`` is not a config field for six
    of seven algorithms; ``population`` is spelled ``num`` in the config), then
    the config fields JSON can reach. ``seed`` is dropped -- it is a separate
    ``run_discovery`` argument and the plan layer rejects it inside
    ``model_kwargs``, so listing it here would only invite a hard failure.

    The ``Resume`` column is kd's own per-knob classification, read off the
    schema like the rest of the table. Half of it already reached the model
    after the fact -- ``provenance.params.identity_breaking`` is computed from
    the same schema on every ``run_discovery`` -- so withholding the column only
    meant the model learned which parameters break a run's identity *after*
    spending a search on it.
    """
    lines = list(_PARAM_TABLE_HEAD)



    tiers = {knob["name"]: knob["resume_tier"] for knob in schema["knobs"]}









    for name in schema["identity_breaking_fields"]:
        tiers.setdefault(name, "identity_breaking")
    for row in schema["facade_params"]:
        if row["name"] == "seed":
            continue
        note = _EFFECT_NOTES.get(row["effect"], f"config field `{row['config_field']}`")
        default = row["default"]
        if row["name"] == "generations" and row["effect"] != "unused":


            default = DEFAULT_GENERATIONS
            note = f"{note}; this tool applies {DEFAULT_GENERATIONS} if you omit it"
        lines.append(
            f"| `{row['name']}` | `{row['kind']}` | {_cell(default)} "
            f"| {_cell(row['literal_values'])} "
            f"| {_tier_cell(tiers.get(row['config_field']))} | {note} |"
        )
    for row in schema["fields"]:
        if row["settable_from"] != "json":
            continue
        default = "**required**" if row["required"] else _cell(row["default"])
        lines.append(
            f"| `{row['name']}` | `{_text_cell(row['type'])}` | {default} "
            f"| {_cell(row['literal_values'])} "
            f"| {_tier_cell(tiers.get(row['name']))} "
            f"| {_text_cell(row['description'])} |"
        )
    return "\n".join(lines)


_SKETCH_LEVEL_LEGEND = (
    "What the levels mean. `lowered`: the clause is compiled into the search "
    "arithmetic itself, as a pinned term deducted from the regression target "
    "before the fit. `generation_enforced`: the clause filters which "
    "candidates the search may generate, which decides how fast a compliant "
    "law is found rather than whether it is compliant. `exit_checked`: the "
    "search itself is not constrained by that "
    "clause at all, and compliance is judged only at the exit.\n\n"
    "There are two exit gates. First, every used clause is judged whatever "
    "its level; `diagnostics.sketch.satisfied` reports that clause verdict. "
    "Second, the full law must pass the exit residual verification policy "
    "on the platform footprint (by default NMSE <= 0.05). "
    "`diagnostics.sketch.certified` reports whether kd delivered a solution "
    "after both gates. A law can satisfy every clause yet fail the residual "
    "gate: `satisfied` true, `certified` false and a null `primary.law`; "
    "`diagnostics.sketch.failure` explains the refusal. kd never delivers "
    "a law that fails either gate, but the search may find no certified law. "
    "A false `certified` and a null `primary.law` are expected outcomes, and "
    "clause failure is likelier when more used clauses sit at `exit_checked`."
)


def _sketch_markdown(schema: dict[str, Any]) -> str:
    """Render one algorithm's sketch support: the clause levels per mode, read
    off `tool_schema`'s per-mode `sketch` block rather than written by hand.

    An instrument that declares every clause `unsupported` on every mode gets
    the sentence saying so, since `run_discovery(sketch=...)` refuses it and
    the model should learn that from the card rather than from the refusal.
    """
    modes = schema["modes"]
    if not criteria.sketch_supported(modes):
        return (
            "This instrument declares every sketch clause `unsupported` on every "
            "mode, so it cannot narrow: `run_discovery(sketch=...)` refuses it "
            "(`sketch_unsupported`)."
        )
    clauses = list(modes[0]["sketch"])
    lines = [
        "| Clause | " + " | ".join(f"`{mode['name']}`" for mode in modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    for clause in clauses:
        cells = " | ".join(f"`{mode['sketch'][clause]}`" for mode in modes)
        lines.append(f"| `{clause}` | {cells} |")
    return (
        "Pass a `kd-sketch-v1` payload as `run_discovery(sketch=...)` to narrow "
        "the search (pinned terms, anchored terms, and holes over a vocabulary; "
        "the tool description shows the payload). Each clause is enforced at "
        "the level kd declares for the mode the search runs in; a clause the "
        "sketch actually uses must be at least `exit_checked` on that mode, "
        "or kd refuses the fit naming the clause.\n\n"
        + "\n".join(lines)
        + "\n\n"
        + _SKETCH_LEVEL_LEGEND
    )


def _surrogate_markdown(schema: dict[str, Any]) -> str:
    """Render how a trained surrogate reaches this algorithm, or "" when it
    takes none.

    The parameter table never lists the injection key (`field_model` /
    `surrogate_model` are not settable from JSON), so the channel has to be
    said in prose: trained by `train_surrogate` with only the instrument's
    `surrogate_fields`, injected by `run_discovery(surrogate=<id>)`, under
    which key, and under which mode when the instrument has one that does not
    consult a surrogate (read off the modes' `provider_kind`, not the name).
    """
    keys = schema["config_artifact_keys"]
    if not keys:
        return ""
    (key,) = keys
    fields = ", ".join(f"`{name}`" for name in schema["surrogate_fields"])
    mode_note = ""
    if any(mode["provider_kind"] != "autograd" for mode in schema["modes"]):
        mode_note = (
            ' under `params={"derivatives": "autograd"}`, the only mode that '
            "consults it (the call is refused otherwise)"
        )
    return (
        "This instrument fits a neural derivative surrogate before it searches, "
        "a long stretch with no progress and no early stop. Train it once with "
        f"`train_surrogate` (which accepts only {fields}, plus `seed`; the same "
        "recipe is trained once and reused) and inject the registered file with "
        f"`run_discovery(surrogate=<surrogate_id>)`{mode_note}. It enters under "
        f"the config key `{key}`, which the table above does not list because "
        "it is not settable from JSON. A lineage keeps one surrogate: a resumed "
        "segment inherits it, and a different id is a new lineage."
    )


def run_record_layout() -> dict[str, str]:
    """What the records a search leaves behind are called and where they live --
    the run directory and catalog names taken from kd, the lineage ledger name
    from `lineage`, none written out by hand in the prompt.

    `orchestration` has to put these into the system prompt, but by the division
    of labor it does not touch kd's surface (the mechanical gate only blocks the
    `kd` imports `kdagent` direction; this one is a convention). The cost of a
    hand-written copy is a prompt that expires silently when kd renames
    something, while the model, told "there is no such file", just works around
    it -- it has no second channel to learn that we wrote it wrong.
    """
    runs_root = Path(DEFAULT_RUNS_ROOT)
    paths = RunDirPaths(root=runs_root)
    return {
        "runs_root": str(runs_root),
        "catalog": str(runs_root / CATALOG_FILENAME),
        "tree": TREE_FILENAME,
        "record": paths.record.name,
        "events": paths.events.name,
        "phases": paths.phases.name,
    }


def _capabilities_markdown(schema: dict[str, Any]) -> str:
    """Render the mode input contracts removed from the handwritten cards."""
    lines = [
        "| Mode | Form | Data layout | Derivatives | LHS orders |",
        "|---|---|---|---|---|",
    ]
    for mode in schema["modes"]:
        cells = [
            f"`{mode['name']}`",
            ", ".join(mode["forms"]),
            ", ".join(mode["topologies"]),
            mode["provider_kind"],
            ", ".join(str(order) for order in mode["lhs_orders"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append(
        f"\nNative score: `{schema['score_kind']}` "
        f"(direction: `{schema['score_direction']}`). "
        f"Cost class: `{schema['cost_class']}`."
    )
    return "\n".join(lines)


def instrument_briefs() -> list[dict[str, str]]:
    """One entry per algorithm: name, one-line summary, rendered parameter table.

    The table is rendered from kd's schema at call time rather than written into
    the card by hand. A hand-written copy goes stale the next time a config
    gains a field, and the model has no second channel telling it so.
    """
    return [
        {
            "algorithm": schema["algorithm"],
            "summary": schema["summary"],







            "cost_class": schema["cost_class"],






            "segmentation_archive": schema["segmentation"]["archive"],
            "capabilities": _capabilities_markdown(schema),
            "parameters": _parameters_markdown(schema),



            "narrowing": _sketch_markdown(schema),
            "surrogate": _surrogate_markdown(schema),
        }
        for schema in instrument_schemas()
    ]


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
    if prior_path.resolve().is_relative_to(session.workspace.resolve()):
        raise ValueError(
            f"{prior_path} sits under the workspace {session.workspace}, the "
            "one directory this run points the model at; a prior reaches this "
            "run as one annotated row per instrument, and keeping the document "
            "outside the workspace is what holds it to that"
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


def make_tools(session: Session) -> list[Callable[..., Any]]:
    """Wrap kd into the six verbs admitted for this run, in a fixed order:
    `[list_datasets, run_discovery, train_surrogate, prune_node, select_node,
    submit_answer]`.

    The order is part of the surface (callers unpack it positionally), so it is
    stated once here. No exception may escape a tool: one that does loses the
    whole run. Every tool body touching real kd has to catch and turn it into a
    ``failures`` entry.
    """

















    def list_datasets() -> dict[str, Any]:
        """List the datasets available here: id, axes, left-hand side, layout,
        tier and tags. The governing equation is never included.

        When this run is pinned to one dataset, only that dataset is listed, and
        it is the only id the other tools accept.
        """
        pinned = session.pinned_dataset_id
        try:
            if pinned is not None and is_file_ref(pinned):
                source = cast(dict[str, Any], resolve_input(session.workspace, pinned))
                rows = [source["summary"]]
            else:
                rows = list_datasets_answer_blind()
                if pinned is None:
                    for path in sorted((session.workspace / "inputs").glob("*.json")):
                        source = cast(
                            dict[str, Any], resolve_input(session.workspace, path.stem)
                        )
                        rows.append(source["summary"])
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
        shown = [row for row in rows if row["id"] == pinned]
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
        algorithm: str = "sga",
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
    ) -> dict[str, Any] | Command[Any]:
        """Run one search on one dataset and wait for it to finish.

        Args:
            dataset_id: Which dataset to search. When this run is pinned to a
                dataset, any other id is refused rather than corrected.
            algorithm: Which instrument to run. Its skill card lists the
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
            sketch: A `kd-sketch-v1` payload that narrows the search to a
                structure: terms pinned with a coefficient, terms anchored
                without one, and holes that must be filled from a constrained
                pool, over a declared vocabulary. Only an instrument whose
                skill card lists sketch clause levels takes one; the verdict
                comes back under `diagnostics.sketch` (absent, with the
                failure under `diagnostics.post_seal_error`, if kd sealed the
                record but could not write the verdict). On a resumed call it is
                inherited from the parent segment, so omit it; a different
                sketch is a new lineage (drop `resume_from_run_id`, name the
                reason with `decision_parent`).
            surrogate: The `surrogate_id` a `train_surrogate` call returned,
                to inject that trained derivative surrogate instead of having
                this search train its own. It must be for this instrument and
                dataset, and the instrument must run in the mode that consults
                it (sga: `params={"derivatives": "autograd"}`). Inherited on a
                resumed call like `sketch`; a different id is a new lineage.

        The sketch payload, for a field `h(t, x)` (the field and coordinate
        names are the dataset's own, the ones `list_datasets` reports as
        `lhs` and `axes`): pin one third-order term and leave one hole for a
        term of order two or less, e.g. `{"schema": "kd-sketch-v1",
        "lhs_spec": {"field": "h", "axis": "t", "order": 1}, "vocabulary":
        {"fields": ["h"], "coordinates": ["t", "x"]}, "pinned": [{"term_ir":
        "mul(h,h_xxx)", "value": -0.5}], "anchored": [], "holes": [{"id":
        "rest", "min_count": 1, "max_count": 1, "constraint":
        {"max_deriv_order": 2, "operators": null, "fields": null, "axes":
        null}}], "match_policy": {"coeff_atol": 1e-9, "coeff_rtol": 1e-9,
        "support_threshold": 0.0, "term_identity": "kd-lawsig-v2",
        "hole_assignment": "disjoint", "derivative_order":
        "total-effective"}}`. A payload kd cannot decode is refused with kd's
        own message.

        A hole caps, it never requires. Its four `constraint` dimensions are a
        ceiling (`max_deriv_order`) and whitelists (`operators`, `fields`,
        `axes`), so the hole above admits any term of order two or less and
        cannot ask that a second derivative appear; `min_count` bounds how
        many terms land in the hole, not which kind. The `id` is an unchecked
        label: `h_t = -0.5*h*h_xxx + 0.3*h` satisfies that same sketch, and
        naming the hole `diffusion` instead of `rest` would not make a second
        derivative appear.

        A sketch describes an evolution law: `lhs_spec` is the time-derivative
        side and has to equal the dataset's resolved LHS, and kd refuses a
        sketch on a homogeneous (`lhs_order=0`) dataset. Every instrument that
        declares sketch support runs evolution-form modes only, so this tool
        adds no separate refusal for that combination.

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
        `diagnostics` holds `nmse`, `score`, `search_seconds` and a
        `segment_report` block (score trend, two improvement numbers, plateau
        counters, diversity, validity, structure diff, cost) computed from the
        sealed run, and whose `provenance` holds `run_id` -- the value
        `submit_answer` asks for.

        The numbers in that block do not share one basis, so take the basis
        with the number. `improvement.vs_parent` measures this segment's
        best against the baseline segment's own best, while
        `improvement.vs_lineage_best` measures it against the best reached
        anywhere in the lineage; `improvement.baseline_run_id` names the
        segment both were measured from, which is the direct parent unless
        that parent sealed no report of its own, in which case it is the
        nearest ancestor that did. The `plateau` counters are cumulative along
        the whole lineage, not this segment's own. Every checkpoint iteration
        number is the opposite -- within-segment, counted from 0 at the start
        of each segment -- which covers `resume_iteration`, the available
        iterations quoted in a refusal, and the one in `provenance.lineage`.
        `cost` is this segment's alone. `diagnostics.evaluations` is this
        segment's candidate-evaluation spend (the compute-neutral unit; the
        inherited prefix of a resumed run is summed on the tree, not here),
        and `diagnostics.stopped_by_evaluation_cap` says whether
        `max_evaluations` ended it.

        This call waits for the search, which runs in its own worker process
        under the cap. `diagnostics.stopped_by_time_cap` true means the cap
        elapsed before the search finished; the search then stopped at an
        iteration boundary and its result is still a sealed, submittable
        record. An instrument with no iteration boundaries to stop at (a
        one-shot fit, or one whose preparation phase eats the whole cap) is
        killed instead and returns a failure whose detail may quote the best
        expression seen so far -- that quote is a plain string, not a
        submittable law. Every envelope also carries a `budget` block
        reporting the wall-clock this whole run has used and, when a limit was
        set for it, how much of that limit is left.

        `diagnostics.prior` is there only when this run was started with a
        prior document, and it says what that document holds about **this**
        instrument on data shaped like this dataset's. `capability` is one of
        four words: `capable` (measured to work on data of this shape),
        `unproven` (the silence word, meaning nobody measured it, never a
        guess), `topology_mismatch` and `form_mismatch` (kd's own declarations
        say this instrument does not take this layout, or this left-hand side).
        `matched` lists the features of the data that put that word there. The
        annotation is a fact about the menu, not a verdict: the call stays
        legal whatever it says, nothing here filters on it, and kd's own
        refusal at fit time is unchanged.
        """
        lineage = _lineage_provenance(
            None, session.workspace, requested_parent=resume_from_run_id
        )




        prior_diagnostics = _prior_diagnostics(session, algorithm)
        pinned = session.pinned_dataset_id
        if pinned is not None and dataset_id != pinned:
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
                        "search that one",
                    )
                ],
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
        if sketch is not None:



            sketch_failure = _sketch_failure(algorithm, sketch, schema_of(algorithm))
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
        if max_search_seconds is not None:
            time_cap_seconds: float | None = float(max_search_seconds)
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




        effective_sketch = sketch
        effective_surrogate = surrogate
        try:




            requested = deepcopy(params) if params else {}
            model_kwargs = dict(requested)
            defaults_applied: dict[str, Any] = {}
            schema = schema_of(algorithm)




            if "generations" not in model_kwargs and _spends_generations(schema):
                model_kwargs["generations"] = DEFAULT_GENERATIONS
                defaults_applied["generations"] = DEFAULT_GENERATIONS
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
                if sketch is None:
                    effective_sketch = source.sketch
                elif sketch != source.sketch:
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


            input_source = resolve_input(session.workspace, dataset_id)
            entry = PlanEntry(
                instrument=algorithm,
                dataset_ref=dataset_id,
                seed=run_seed,
                model_kwargs=model_kwargs,
            )
            isolated = run_isolated_episode(
                entry,













                run_dir=runs_root / new_run_id(algorithm),
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
            )
            outcome = isolated.outcome
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
                sketch=effective_sketch,
                surrogate=effective_surrogate,
            )



            result["diagnostics"].update(prior_diagnostics)
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
                    evaluation_cap=max_evaluations,
                    stopped_by_evaluation_cap=isolated.stopped_by_evaluation_cap,
                ),
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
        if result["status"] != "success":
            return result






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
        answer_law, answer_provenance = _selected_answer(session, dataset_id)
        if answer_law is None:




            answer_law, answer_provenance = result["primary"]["law"], own_provenance
        if provenance["run_id"] not in {
            entry.run_id for entry in session.ledger.segments()
        }:







            result["failures"].append(
                _failure(
                    "ledger_unavailable",
                    "this search is not on the policy tree, so the answer was "
                    "selected without it; the underlying ledger failure is "
                    "reported above",
                )
            )


        return Command(
            update={
                "messages": [_tool_message(result, "run_discovery", tool_call_id)],
                ANSWER: Answer(
                    law=answer_law,
                    source="selected",
                    provenance=answer_provenance,
                ),
            }
        )



































    def train_surrogate(
        dataset_id: str,
        algorithm: str,
        params: dict[str, Any] | None = None,
        seed: int | None = None,
        max_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Train the derivative surrogate an instrument fits before it searches,
        once per recipe, and register the file for `run_discovery(surrogate=)`.

        Args:
            dataset_id: Which dataset to train on. When this run is pinned to
                a dataset, any other id is refused.
            algorithm: Which instrument the surrogate is for. Only an
                instrument whose skill card has a "Surrogate" section takes
                one; the refusal names the instruments that do.
            params: Only the instrument's surrogate parameters, the ones its
                skill card's "Surrogate" section lists (epochs, learning rate,
                patience, validation ratio; the network shape for dlga).
                Search parameters are refused here: they do not shape the
                network.
            seed: Random seed of the training. Omitted, it is 0. The seed is
                part of the recipe, so two seeds are two surrogates.
            max_seconds: Wall-clock cap for this training. A training has no
                iteration boundary to stop at: at the cap it is killed and
                nothing is kept, so give it what the epoch count needs, or
                lower the epoch count. Omitted, the run's remaining time
                budget is the cap (an exhausted budget is refused before
                anything starts); with no budget the training is uncapped.

        The same recipe (dataset, instrument, seed, params) is trained once: a
        second call with the file still in place returns `primary.reused:
        true` and trains nothing, whatever the remaining budget (the cap
        applies to a training, not to a lookup). Pass the returned
        `primary.surrogate_id` to
        `run_discovery(surrogate=...)`; for sga that search has to run with
        `params={"derivatives": "autograd"}`, the mode that consults it. A
        lineage keeps one surrogate: a resumed segment inherits it.

        Returns an envelope whose `primary` holds `surrogate_id` and `reused`,
        whose `diagnostics` hold `train_seconds`, `epochs` and `final_loss`,
        and whose `provenance` names the instrument, dataset, seed, params,
        the config key the surrogate is injected under, and the file's sha256.
        Every envelope also carries the run's `budget` block.
        """
        requested = deepcopy(params) if params else {}
        provenance: dict[str, Any] = {
            "instrument": algorithm,
            "dataset": dataset_id,
            "seed": seed,
            "params": requested,
        }
        pinned = session.pinned_dataset_id
        if pinned is not None and dataset_id != pinned:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance={**provenance, "pinned": pinned},
                failures=[
                    _failure(
                        "dataset_not_pinned",
                        f"this run is pinned to dataset {pinned!r} and can only "
                        "train on that one",
                    )
                ],
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
        fields = schema["surrogate_fields"]
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
                        f"{algorithm}; train_surrogate takes only {fields} (plus "
                        "seed), the parameters that shape the network. Search "
                        "parameters go to run_discovery",
                    )
                ],
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




            hit, retrained_reason = surrogate_training.reuse_hit(registry, recipe)
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


            if max_seconds is not None:
                time_cap_seconds: float | None = float(max_seconds)
            else:
                remaining = session.budget()["remaining_seconds"]
                if remaining is None:
                    time_cap_seconds = None
                elif remaining <= 0:
                    return envelope(
                        "failure",
                        budget=session.budget(),
                        provenance=provenance,
                        failures=[
                            _failure(
                                "budget_exhausted",
                                f"this run's time budget is exhausted "
                                f"({remaining} s remaining); a training cut "
                                "short at the cap keeps nothing, so none was "
                                "started",
                            )
                        ],
                    )
                else:
                    time_cap_seconds = float(remaining)
            training = surrogate_training.run_training(
                registry=registry,
                schema=schema,
                recipe=recipe,
                key=key,
                time_cap_seconds=time_cap_seconds,
            )
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
        once this one is gone, and is null when this dataset has no publishable
        search left -- prune the last one and this run ends with no answer
        unless you submit one or run another search.
        """
        ledger = session.ledger
        refusal = _node_refusal(session, run_id, action="abandoned again")
        if refusal is not None:
            return envelope("failure", budget=session.budget(), failures=[refusal])
        node = ledger.entry(run_id)
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
        own_dataset = (
            session.pinned_dataset_id
            if session.pinned_dataset_id is not None
            else session.answer_dataset_id
        )


        if node.dataset and node.dataset == own_dataset:
            law, provenance = _selected_answer(session, node.dataset)
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
        if law is None or provenance is None:
            return result
        return Command(
            update={
                "messages": [_tool_message(result, "prune_node", tool_call_id)],
                ANSWER: Answer(law=law, source="selected", provenance=provenance),
            }
        )




























    def select_node(
        run_id: str,
        reason: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> dict[str, Any] | Command[Any]:
        """Name the search whose law is published as this run's answer if no
        answer is submitted.

        Args:
            run_id: The search to publish, taken from that search's
                `provenance.run_id`. It must be a search on this workspace's
                tree that sealed a structured law and, when this run is pinned
                to a dataset, one that searched that dataset. An abandoned
                search is refused.
            reason: Why this search is the one to publish, recorded on the tree
                beside the selection. One sentence, for whoever reads the tree
                afterwards.

        Use this when the searches disagree and the better score is not the
        better equation: without it, the run publishes whichever node the
        default rule holds. It does not end the run, and it is not a
        submission -- `submit_answer` still outranks it, and a later prune of
        the selected node takes it back out of the running.
        """
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
        pinned = session.pinned_dataset_id
        if pinned is not None and entry.dataset != pinned:
            return envelope(
                "failure",
                budget=session.budget(),
                provenance=_entry_provenance(entry),
                failures=[
                    _failure(
                        "dataset_not_pinned",
                        f"run {run_id!r} searched dataset {entry.dataset!r} "
                        f"while this run is pinned to {pinned!r}; publishing it "
                        "would answer this run's question with another "
                        "dataset's equation",
                    )
                ],
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
                actually ran and succeeded.

        A successful submission ends this run: no further model turn happens
        after it. Any other tool you called in the same turn still runs, because
        those calls were already issued. A refused submission does not end the
        run, so a failure can be fixed and submitted again.
        """
        shape_failure = _law_shape_failure(law)
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
            eligible = sorted(session.successful_runs)
            return envelope(
                "failure",
                budget=session.budget(),
                failures=[
                    _failure(
                        "unknown_run_id",
                        f"run_id {run_id!r} is not a search that succeeded this "
                        "run, "
                        + (
                            f"the usable ones are {eligible}"
                            if eligible
                            else "no search has succeeded this run yet, so run "
                            "run_discovery first"
                        )
                        + "; the value comes from provenance.run_id in what "
                        "run_discovery returned",
                    )
                ],
            )
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
        return Command(
            update={
                "messages": [_tool_message(result, "submit_answer", tool_call_id)],
                ANSWER: Answer(law=law, source="submitted", provenance=provenance),
            }
        )

    return [
        list_datasets,
        run_discovery,
        train_surrogate,
        prune_node,
        select_node,
        submit_answer,
    ]
