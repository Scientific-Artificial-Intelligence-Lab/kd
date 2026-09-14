
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from kd import LhsSpec, format_pde, render_lhs_label
from kd.search import RunDirPaths, RunRecord

from kdagent.lineage import SegmentEntry, SegmentLedger
from kdagent.segment_report import law_from_evidence


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


def _law(law: dict[str, Any], *, homogeneous: bool, sealed: bool) -> str:
    terms = law["support"]
    coefficients = law.get("coefficients")
    lhs = law.get("lhs")
    label = "0" if homogeneous else None
    if lhs is not None:
        label = render_lhs_label(LhsSpec(**lhs)) if sealed else json.dumps(lhs)
    lines = [f"LHS: `{label}`" if label is not None else "LHS: not reported."]


    if (
        sealed
        and label is not None
        and coefficients is not None
        and None not in coefficients
    ):
        equation = format_pde(terms, coefficients, lhs=label, sig_figs=0)
        lines.append(f"\n$$\n{equation.latex}\n$$\n")
    elif sealed:
        lines.append(
            "\nEquation display unavailable: LHS or coefficients not reported.\n"
        )
    else:
        lines.append("\nTerms are listed as submitted; the equation image belongs to the sealed law.\n")
    lines.extend(["| Term | Coefficient (full precision) |", "|---|---|"])
    for index, term in enumerate(terms):
        coefficient = None if coefficients is None else coefficients[index]
        lines.append(f"| `{_cell(term)}` | {_cell(coefficient)} |")
    return "\n".join(lines) + "\n"


def _sealed_section(record: RunRecord, run_dir: Path, workspace: Path) -> str:
    evidence = record.evidence
    law = law_from_evidence(evidence)
    homogeneous = (
        evidence.catalog_fit is not None
        and evidence.catalog_fit["form"] == "HOMOGENEOUS"
    )
    lines = ["## Cited sealed law\n"]
    lines.append(
        "No submittable sealed law.\n"
        if law is None
        else _law(law, homogeneous=homogeneous, sealed=True)
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
    if answer is None:
        return "## Answer\n\nNo answer was produced.\n"
    provenance = result["answer_provenance"]
    run_dir = workspace / provenance["run_dir"]
    record = RunRecord.load(RunDirPaths(root=run_dir).record)
    sealed = law_from_evidence(record.evidence)
    homogeneous = (
        record.evidence.catalog_fit is not None
        and record.evidence.catalog_fit["form"] == "HOMOGENEOUS"
    )
    title = (
        "Submitted law" if result["answer_source"] == "submitted" else "Selected law"
    )
    lines = [
        f"## {title}\n",
        _law(
            answer,
            homogeneous=homogeneous and answer.get("lhs") is None,
            sealed=answer == sealed,
        ),
    ]
    lines.append(
        "\nThis law matches the cited sealed law exactly; its native measurements are below.\n"
        if answer == sealed
        else "\nUnevaluated: this law differs from the cited sealed law. The measurements below "
        "do not evaluate the submitted law.\n"
    )
    lines.extend(
        [
            f"Cited run: `{provenance['run_id']}`\n",
            _sealed_section(record, run_dir, workspace),
        ]
    )


    entry = next(
        (entry for entry in ledger.segments() if entry.run_id == provenance["run_id"]),
        None,
    )
    lines.append("## Platform refit of cited support\n")
    if entry is None:
        lines.append(
            "No ledger entry for the cited search; platform refit and lineage cost are not reported.\n"
        )
    else:
        lines.append(_platform_section(entry, sealed, homogeneous=homogeneous))
        lines.extend(
            [
                "### Cited archive lineage cost\n",
                _json(ledger.lineage_cost(entry.run_id)),
            ]
        )
    return "\n".join(lines)


def _platform_section(
    entry: SegmentEntry,
    sealed: dict[str, Any] | None,
    *,
    homogeneous: bool,
) -> str:
    lines = [
        "These are separately fitted coefficients on the cited support, not the native or submitted coefficients.\n",
        f"Frame: `{_cell(entry.platform_frame)}`; NMSE: {_cell(entry.platform_nmse)}.\n",
    ]
    if sealed is not None and entry.platform_coefficients is not None:
        lines.append(
            _law(
                {**sealed, "coefficients": entry.platform_coefficients},
                homogeneous=homogeneous,
                sealed=True,
            )
        )
    else:
        lines.append("Platform coefficients: not reported.\n")
    return "\n".join(lines)


def _segments(workspace: Path, ledger: SegmentLedger) -> str:
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
                        "resume_from": entry.resume_from,
                        "params": entry.params,
                        "sketch": entry.sketch,
                        "surrogate": entry.surrogate,
                        "summary": None
                        if entry.summary is None
                        else {
                            key: value
                            for key, value in entry.summary.to_dict().items()
                            if key != "signature"
                        },
                        "report_projection": entry.report_projection,
                        "cost": entry.cost,
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
        ]
    )
    return "\n".join(lines) + "\n"


def _tool_outcomes(trace_path: Path) -> str:
    lines = ["## Tool outcomes from this invocation\n"]
    tools = {
        "run_discovery",
        "train_surrogate",
        "prune_node",
        "select_node",
        "submit_answer",
    }
    rows = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in sorted(rows, key=lambda row: row["start_time"]):
        if row["run_type"] != "tool" or row["name"] not in tools:
            continue
        lines.append(f"### `{row['name']}` — `{row['id']}`\n")
        output = row["outputs"]
        if output is None:
            lines.append(_code(str(row["error"])))
            continue
        message = output["output"]
        if "update" in message:
            message = message["update"]["messages"][0]
        content = message["content"]
        if message.get("status") == "error":
            lines.extend(["Framework tool error:\n", _code(content)])
        else:
            lines.append(_outcome_details(json.loads(content)))
    return "\n".join(lines) + "\n"


def _outcome_details(outcome: dict[str, Any]) -> str:
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
    if "segment_report" in diagnostics:
        lines.extend(
            ["Recorded segment diagnostics:\n", _json(diagnostics["segment_report"])]
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
        _answer_sections(workspace, result, ledger),
        _segments(workspace, ledger),
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
