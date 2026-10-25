
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from kd.search import RunDirPaths, RunRecord

from kdagent.lineage import (
    SegmentEntry,
    SegmentLedger,
    sketch_identity,
    standing_verification,
)
from kdagent.segment_report import law_from_evidence, sealed_equation, sealed_lhs_label
from kdagent.surrogates import SurrogateRegistry, stopped_at_deadline


def _cell(value: Any) -> str:
    if value is None:
        return "not reported"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _code(text: str, language: str = "") -> str:

    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{language}\n{text}\n{fence}\n"


def _json(value: Any) -> str:
    return _code(json.dumps(value, indent=2, ensure_ascii=False), "json")


def _link(path: Path, workspace: Path) -> str:
    relative = os.path.relpath(path, workspace)
    return f"[{_cell(path.name)}]({quote(relative, safe='/')})"


def _law(
    law: dict[str, Any], *, catalog_fit: dict[str, Any] | None, sealed: bool
) -> str:
    terms = law["support"]
    coefficients = law.get("coefficients")
    lhs = law.get("lhs")
    if sealed or lhs is None:
        label = sealed_lhs_label(lhs, catalog_fit)
    else:
        label = json.dumps(lhs)
    lines = [f"LHS: `{label}`" if label is not None else "LHS: not reported."]



    equation = sealed_equation(law, catalog_fit) if sealed else None
    if equation is not None:
        lines.append(f"\n$$\n{equation.latex}\n$$\n")
    elif sealed:
        lines.append(
            "\nEquation display unavailable: LHS or coefficients not reported.\n"
        )
    else:
        lines.append(
            "\nTerms are listed as submitted; the equation image belongs to the sealed law.\n"
        )
    lines.extend(["| Term | Coefficient (full precision) |", "|---|---|"])
    for index, term in enumerate(terms):
        coefficient = None if coefficients is None else coefficients[index]
        lines.append(f"| `{_cell(term)}` | {_cell(coefficient)} |")
    return "\n".join(lines) + "\n"


def _sealed_section(record: RunRecord, run_dir: Path, workspace: Path) -> str:
    evidence = record.evidence
    law = law_from_evidence(evidence)
    lines = ["## Cited sealed law\n"]
    lines.append(
        "No submittable sealed law.\n"
        if law is None
        else _law(law, catalog_fit=evidence.catalog_fit, sealed=True)
    )
    lines.extend(
        [
            "\nNative measurements of this sealed law:\n",
            "| Score kind | Direction | Score | NMSE | Coefficient source |",
            "|---|---|---|---|---|",
            f"| {_cell(evidence.score_kind)} | {_cell(evidence.score_direction)} "
            f"| {_cell(evidence.score)} | {_cell(evidence.nmse)} "
            f"| {_cell(evidence.headline_coefficient_source)} |\n",
            "### Search provenance\n",
            _json(
                {
                    "instrument": evidence.instrument,
                    "dataset": evidence.dataset_name,
                    "seed": evidence.seed,
                    "created_at": record.created_at,
                    "kd_version": record.run_spec.kd_version,
                    "dataset_cache_fingerprint": evidence.dataset_cache_fingerprint,
                    "record_hash": record.record_hash,
                    "evidence_hash": record.evidence_hash,
                    "run_spec_hash": record.run_spec_hash,
                    "config": record.run_spec.config,
                    "cost": record.cost.to_dict(),
                }
            ),
            "Artifacts: "
            + ", ".join(
                _link(path, workspace)
                for path in (
                    RunDirPaths(root=run_dir).record,
                    RunDirPaths(root=run_dir).manifest,
                    RunDirPaths(root=run_dir).events,
                    RunDirPaths(root=run_dir).phases,
                    RunDirPaths(root=run_dir).recorder,
                )
                if path.exists()
            )
            + "\n",
        ]
    )
    return "\n".join(lines)


def _answer_sections(
    workspace: Path,
    result: dict[str, Any],
    ledger: SegmentLedger,
) -> str:
    answer = result["answer"]
    if answer is None and result["answer_source"] == "abstained":
        return _abstention_section(result)
    if answer is None:
        return "## Answer\n\nNo answer was produced.\n" + _selection_section(result)
    provenance = result["answer_provenance"]
    run_dir = workspace / provenance["run_dir"]
    record = RunRecord.load(RunDirPaths(root=run_dir).record)
    sealed = law_from_evidence(record.evidence)
    catalog_fit = record.evidence.catalog_fit
    title = (
        "Submitted law" if result["answer_source"] == "submitted" else "Selected law"
    )


    entry = next(
        (entry for entry in ledger.segments() if entry.run_id == provenance["run_id"]),
        None,
    )
    lines = [
        f"## {title}\n",
        _attributed_law(answer, result, catalog_fit=catalog_fit),
        f"Cited run: `{provenance['run_id']}`\n",
        _answer_system(result.get("answer_system")),
        _answer_sketch(entry),
        _selection_section(result),
        _sealed_section(record, run_dir, workspace),
    ]
    lines.append("## Platform refit of cited support\n")
    if entry is None:
        lines.append(
            "No ledger entry for the cited search; platform refit and lineage cost are not reported.\n"
        )
    else:
        lines.append(_platform_section(entry, sealed, catalog_fit=catalog_fit))
        lines.extend(
            [
                "## Forward measurement of the cited sealed law\n",
                "The measurement the default selection ranks on, recorded on the "
                "ledger row when the search sealed; nothing is measured here.\n",
                _json(entry.selection_forward),
                "### Cited archive lineage cost\n",
                _json(ledger.lineage_cost(entry.run_id)),
            ]
        )
    return "\n".join(lines)


def _answer_system(system: dict[str, Any] | None) -> str:
    if system is None:
        return ""
    standing = system["verification_id"]
    return (
        f"The law is published as a member of system `{system['system_id']}` "
        + (
            f"(verification `{standing}`)"
            if standing is not None
            else "(no verification of it has a result)"
        )
        + ", whose members are "
        + ", ".join(
            f"`{run_id}` for `{target}`" for target, run_id in system["members"].items()
        )
        + "; its verifications are under ODE systems.\n"
    )


def _answer_sketch(entry: SegmentEntry | None) -> str:
    if entry is None:
        return ""
    if entry.sketch is None:
        return "The cited search ran under no Sketch.\n"
    sketch_id, digest = sketch_identity(entry.sketch)
    return (
        f"The cited search ran under Sketch `{sketch_id}` (sha256 `{digest}`, "
        "its segment's payload below); the active Sketch is under Sketches.\n"
    )


def _sketch_section(sketches: dict[str, Any]) -> str:
    active = sketches["active_sketch_id"]
    lines = [
        "## Sketches\n",
        (
            f"Active Sketch: `{active}`; a search that names no Sketch runs under it."
            if active is not None
            else "No Sketch is active."
        )
        + " Each search keeps the Sketch it ran under (its segment row), "
        "and the answer names its own.\n",
        "| Sketch | sha256 | Origin | Input | Evidence | Reason | Registered at |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in sketches["registered"]:
        for entry in record["registrations"]:
            evidence = ", ".join(entry["evidence_ids"])
            lines.append(
                f"| `{record['sketch_id']}` | `{record['sketch_sha256']}` "
                f"| {_cell(entry['origin'])} | `{_cell(entry['input_id'])}` "
                f"| {_cell(evidence or 'none')} | {_cell(entry['reason'] or 'none')} "
                f"| {_cell(entry['created_at'])} |"
            )
    return "\n".join(lines) + "\n"


def _selection_section(result: dict[str, Any]) -> str:
    if "answer_selection" not in result:
        return ""
    return "\n".join(["### Default selection\n", _json(result["answer_selection"])])


def _abstention_section(result: dict[str, Any]) -> str:
    evidence = ", ".join(f"`{_cell(run_id)}`" for run_id in result["evidence_run_ids"])
    return "\n".join(
        [
            "## Answer\n",
            "No answer: the run ended with an explicit unresolved result "
            "(`finish_without_answer`), and no selected search is published "
            "in its place.\n",
            f"Reason: {_cell(result['reason'])}\n",
            f"Evidence runs: {evidence or 'none cited'}.\n",
        ]
    )


def _attributed_law(
    answer: dict[str, Any],
    result: dict[str, Any],
    *,
    catalog_fit: dict[str, Any] | None,
) -> str:
    producer = result["answer_provenance"].get("producer_ref", {"kind": "search"})
    if producer["kind"] == "search":
        return _law(answer, catalog_fit=catalog_fit, sealed=True) + (
            "\nThis law matches the cited sealed law exactly: the search produced "
            "its coefficients, so the native measurements below are its own.\n"
        )
    lines = [_law(answer, catalog_fit=catalog_fit, sealed=False)]
    if producer["kind"] == "edited":
        lines.append(
            "\nUnevaluated: no search or refinement of this run produced exactly "
            "these coefficients (`producer_ref` `edited`). The measurements below "
            "do not evaluate the submitted law.\n"
        )
        return "\n".join(lines)
    receipt = result["answer_receipt"]
    lines.extend(
        [
            f"\nProduced by refinement `{receipt['refinement_id']}` of the cited "
            f"search: method `{receipt['method']}`, support "
            f"`{receipt['support_origin']}`, measured on input "
            f"`{receipt['evaluation_input_record']['dataset_id']}`. This measurement "
            "is the submitted law's own; the cited search's native measurements "
            "below describe its sealed law, not this one.\n",
            _json(
                {
                    key: receipt[key]
                    for key in (
                        "refinement_id",
                        "parent_ref",
                        "support_origin",
                        "method",
                        "parameters",
                        "measurement",
                        "evaluation_input_record",
                        "output_input_record",
                        "law_sha256",
                        "artifact_sha256",
                    )
                }
            ),
        ]
    )
    return "\n".join(lines)


def _platform_section(
    entry: SegmentEntry,
    sealed: dict[str, Any] | None,
    *,
    catalog_fit: dict[str, Any] | None,
) -> str:
    lines = [
        "These are separately fitted coefficients on the cited support, not the native or submitted coefficients.\n",
        f"Frame: `{_cell(entry.platform_frame)}`; NMSE: {_cell(entry.platform_nmse)}.\n",
    ]
    if sealed is not None and entry.platform_coefficients is not None:
        lines.append(
            _law(
                {**sealed, "coefficients": entry.platform_coefficients},
                catalog_fit=catalog_fit,
                sealed=True,
            )
        )
    else:
        lines.append("Platform coefficients: not reported.\n")
    return "\n".join(lines)


def _trained_network(
    registry: SurrogateRegistry, surrogate_id: str | None
) -> dict[str, Any] | None:
    if surrogate_id is None:
        return None
    entry = registry.entry(surrogate_id)
    return {
        "surrogate_id": surrogate_id,
        "stopped_at_deadline": stopped_at_deadline(entry),
        "recipe": entry.recipe,
        "file": entry.file,
    }


def _segments(workspace: Path, ledger: SegmentLedger) -> str:
    registry = SurrogateRegistry(workspace)
    lines = [
        "## Workspace segments\n",
        "Includes earlier invocations when this workspace was reopened. Archive parent names the "
        "resumed checkpoint's run; decision parent names the outcome that prompted the action.\n",
        "| Run | Instrument / dataset | Status | Action | Archive parent | Decision parent | Resume iteration | Seed / reseed | Pruned | Native NMSE |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in ledger.segments():
        iteration = (
            None if entry.resume_from is None else entry.resume_from["iteration"]
        )
        lines.append(
            f"| {_cell(entry.run_id)} | {_cell(entry.instrument)} / {_cell(entry.dataset)} "
            f"| {_cell(entry.status)} | {_cell(entry.action)} | {_cell(entry.parent_run_id or 'none')} "
            f"| {_cell(entry.decision_parent or 'none')} | {_cell(iteration if iteration is not None else 'none')} | {entry.seed} / {entry.reseed} "
            f"| {ledger.is_pruned(entry.run_id)} | {_cell(entry.nmse)} |"
        )
    for entry in ledger.segments():
        lines.extend(
            [
                f"\n### Segment `{entry.run_id}`\n",
                _json(
                    {
                        "run_dir": entry.run_dir,
                        "reason": entry.reason,
                        "resume_from": entry.resume_from,
                        "params": entry.params,
                        "sketch": entry.sketch,
                        "surrogate": entry.surrogate,
                        "trained_surrogate": _trained_network(
                            registry, entry.trained_surrogate
                        ),
                        "summary": None
                        if entry.summary is None
                        else {
                            key: value
                            for key, value in entry.summary.to_dict().items()
                            if key != "signature"
                        },
                        "report_projection": entry.report_projection,
                        "cost": entry.cost,
                        "selection_forward": entry.selection_forward,
                    }
                ),
                _link(workspace / entry.run_dir, workspace),
            ]
        )
    events = ledger.to_dict()
    lines.extend(
        [
            "\n## Pruning and selection\n",
            _json(
                {
                    "pruned": events["pruned"],
                    "selected": events["selected"],
                }
            ),
            _decision_log(events["actions"]),
        ]
    )
    return "\n".join(lines) + "\n"


def _systems(ledger: SegmentLedger) -> str:
    payload = ledger.systems_dict()
    if not payload["systems"]:
        return ""
    selected = payload["selected_system"]
    lines = [
        "## ODE systems\n",
        "A system integrates one sealed law per state together along every raw "
        "trajectory, from its first to its last observation, and compares the "
        "result with those observations (kd's `forward_error_trajectories`). "
        "Its current result is its latest verification kd classified; a "
        "timeout or a failed worker stays in its history. `intervals` are the "
        "integration intervals of the trajectories; a default selection's "
        "`window` is the bound on the system error, another number.\n",
        "Selected system: "
        + ("none" if selected is None else f"`{selected['system_id']}`")
        + ".\n",
    ]
    for system in payload["systems"]:
        lines.append(_system_rows(ledger, system))
    return "\n".join(lines) + "\n"


def _system_rows(ledger: SegmentLedger, system: dict[str, Any]) -> str:
    standing = standing_verification(system)
    lines = [
        f"### System `{system['system_id']}`\n",
        f"Raw trajectories `{system['raw_fingerprint']}`; current result: "
        + ("none" if standing is None else f"`{standing['verification_id']}`")
        + ".\n",
        "| Target | Member run | Input | Sketch | Law |",
        "|---|---|---|---|---|",
    ]
    for target, run_id in system["members"].items():
        entry = ledger.entry(run_id)
        sketch = "none" if entry.sketch is None else sketch_identity(entry.sketch)[0]
        law = system["laws"][target]
        terms = " + ".join(
            f"{coefficient} * {term}"
            for term, coefficient in zip(
                law["support"], law["coefficients"], strict=True
            )
        )
        lines.append(
            f"| `{target}` | `{run_id}` | `{_cell(entry.dataset)}` | `{sketch}` "
            f"| {_cell(terms)} |"
        )
    verifications = [
        {
            key: verification[key]
            for key in (
                "verification_id",
                "input_id",
                "status",
                "forward",
                "error",
                "budget",
                "seconds",
                "reason",
                "created_at",
            )
        }
        for verification in system["verifications"]
    ]
    lines.extend(["\nVerifications:\n", _json(verifications)])
    return "\n".join(lines)


def _decision_log(actions: list[dict[str, Any]]) -> str:
    lines = [
        "\n## Decision log\n",
        "Calls that ran, other than searches (whose reasons are on their "
        "segments above). A refused call is not listed.\n",
        "| Verb | Produced | Cited run | Status | Reason |",
        "|---|---|---|---|---|",
    ]
    for action in actions:
        produced = ", ".join(
            f"{key} {json.dumps(value, ensure_ascii=False)}"
            for key, value in action["ref"].items()
        )
        lines.append(
            f"| {_cell(action['verb'])} | {_cell(produced or 'none')} "
            f"| {_cell(action['run_id'] or 'none')} | {_cell(action['status'])} "
            f"| {_cell(action['reason'] or 'none')} |"
        )
    return "\n".join(lines) + "\n"


REPORTED_TOOLS = frozenset(
    {
        "run_discovery",
        "train_surrogate",
        "prune_node",
        "select_node",
        "submit_answer",
        "refine_candidate",
        "smooth_input",
        "ablate_candidate",
        "score_term_sets",
        "verify_candidate",
        "verify_system",
        "vote_terms",
        "make_sketch",
        "set_sketch",
        "finish_without_answer",
    }
)


def _tool_outcomes(trace_path: Path) -> str:
    lines = ["## Tool outcomes from this invocation\n"]
    rows = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in sorted(rows, key=lambda row: row["start_time"]):
        if row["run_type"] != "tool" or row["name"] not in REPORTED_TOOLS:
            continue
        lines.append(f"### `{row['name']}` — `{row['id']}`\n")
        output = row["outputs"]
        if not output:


            lines.append(_code(str(row["error"])))
            continue
        message = output["output"]
        if "update" in message:
            message = message["update"]["messages"][0]
        content = message["content"]
        if message.get("status") == "error":
            lines.extend(["Framework tool error:\n", _code(content)])
        else:
            lines.append(_outcome_details(row["name"], json.loads(content)))
    return "\n".join(lines) + "\n"


def _outcome_details(name: str, outcome: dict[str, Any]) -> str:
    provenance = outcome["provenance"]
    diagnostics = outcome["diagnostics"]
    lines = [f"Status: `{outcome['status']}`.\n"]
    if provenance:
        lines.append(_json(provenance))
    for failure in outcome["failures"]:
        lines.append(f"**{_cell(failure['code'])}**: {_cell(failure['detail'])}\n")
    limits = {
        key: diagnostics[key]
        for key in (
            "time_cap_seconds",
            "stopped_by_time_cap",
            "evaluation_cap",
            "stopped_by_evaluation_cap",
            "post_seal_error",
        )
        if key in diagnostics
    }
    if limits:
        lines.append(_json(limits))
    if "surrogate" in diagnostics:
        lines.extend(
            ["Network this search trained:\n", _json(diagnostics["surrogate"])]
        )
    if "segment_report" in diagnostics:
        lines.extend(
            ["Recorded segment diagnostics:\n", _json(diagnostics["segment_report"])]
        )
    if name != "run_discovery" and (outcome["primary"] or diagnostics):
        lines.extend(
            [
                "Result:\n",
                _json({"primary": outcome["primary"], "diagnostics": diagnostics}),
            ]
        )
    return "\n".join(lines)


def write_report(
    workspace: Path,
    *,
    result: dict[str, Any],
    ledger: SegmentLedger,
    trace_path: Path,
    elapsed_seconds: float,
) -> Path:
    lines = [
        "# Discovery report\n",
        f"Endpoint: `{result['endpoint']}`; model: `{result['model']}`.\n",
        f"Stop reason: `{result['stop_reason']}`; answer source: `{result['answer_source']}`.\n",
        f"Session elapsed seconds: {elapsed_seconds}.\n",
        *(
            [
                "User Sketch (origin `user`), sha256 "
                f"`{result['sketch_sha256']}`: every search this invocation "
                "started ran under it.\n"
            ]
            if "sketch_sha256" in result
            else []
        ),
        *(
            [
                "LLM4ED was on, through its own endpoint (public configuration, "
                f"no key): `{json.dumps(result['llm4ed'], ensure_ascii=False)}`.\n"
            ]
            if "llm4ed" in result
            else []
        ),
        *([_sketch_section(result["sketches"])] if "sketches" in result else []),
        _answer_sections(workspace, result, ledger),
        _segments(workspace, ledger),
        _systems(ledger),
        _tool_outcomes(trace_path),
        "## Workspace artifacts\n",
    ]
    paths = [trace_path, workspace / "tree.jsonl", workspace / "runs/catalog.jsonl"]
    paths.extend(sorted((workspace / "inputs").glob("*.json")))
    lines.append(
        ", ".join(_link(path, workspace) for path in paths if path.exists()) + "\n"
    )
    path = workspace / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
