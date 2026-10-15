
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kd import mode_accepts
from kd.search import mode_inputs

_ASSETS = Path(__file__).resolve().parent / "_assets"
_TASK_KINDS = frozenset({"pde-evolution", "pde-steady", "tabular-regression", "ode"})
_NOTHING_REQUIRED = "No accepted mode requires a setting for this input."


def _asset(group: str, task_kind: str) -> str:
    if task_kind not in _TASK_KINDS:
        raise ValueError(
            f"unknown task_kind {task_kind!r}; choose one of {sorted(_TASK_KINDS)}"
        )
    return (_ASSETS / group / f"{task_kind}.md").read_text(encoding="utf-8")


def _mode_table(
    report: dict[str, Any], schemas: list[dict[str, Any]]
) -> tuple[str, str, set[tuple[str, str]]]:
    rows = [
        "| Algorithm/mode | Accepts | Reason | Params | Budget |",
        "|---|---|---|---|---|",
    ]
    required = []
    accepted: set[tuple[str, str]] = set()
    inputs = mode_inputs(report)
    for schema in schemas:
        generations = next(
            p for p in schema["facade_params"] if p["name"] == "generations"
        )
        effect = generations["effect"]
        meaning = (
            f"config field {generations['config_field']}"
            if effect == "config_field"
            else effect
        )
        budget = (
            "continue with resume_from_run_id"
            if schema["segmentation"]["archive"] == "progress"
            else "rerun with other params"
        ) + f"; generations: {meaning}"
        for mode in schema["modes"]:
            if not mode["selectable_from_data"]:
                continue
            verdict = mode_accepts(mode, **inputs)
            label = f"{schema['algorithm']}/{mode['name']}"
            reason = (verdict.reason or "").replace("|", r"\|")
            params = json.dumps(verdict.params, sort_keys=True)
            rows.append(
                f"| `{label}` | {'yes' if verdict.accepts else 'no'} | {reason} | `{params}` | {budget} |"
            )
            if verdict.accepts:
                accepted.add((schema["algorithm"], mode["name"]))
                notes = _required_notes(mode, verdict.params, report["lhs_order"])
                if notes:
                    required.append(f"- `{label}`: " + "; ".join(notes) + ".")
    return "\n".join(rows), "\n".join(required) or _NOTHING_REQUIRED, accepted


def _required_notes(
    mode: dict[str, Any], params: dict[str, Any], lhs_order: int
) -> list[str]:
    parts = {
        "required_params": mode["required_params"],
        "order_params": mode["order_params"].get(str(lhs_order), {}),


        "names_from": mode["names_from"] if mode["names_from"] in params else None,
    }
    return [
        f"{key}={json.dumps(value, sort_keys=True)}"
        for key, value in parts.items()
        if value
    ]


def _cost_anchors(
    advice: dict[str, Any], task_kind: str, accepted: set[tuple[str, str]]
) -> str:
    lines = []
    for anchor in advice["cost_anchors"]:
        if anchor["task_kind"] not in (None, task_kind):
            continue
        usable = (
            any(algorithm == anchor["algorithm"] for algorithm, _ in accepted)
            if anchor["mode"] is None
            else (anchor["algorithm"], anchor["mode"]) in accepted
        )
        if usable:
            label = anchor["algorithm"] + (
                "" if anchor["mode"] is None else f"/{anchor['mode']}"
            )
            lines.append(f"- `{label}`: {anchor['text']}")
    if not lines:
        return ""
    return (
        f"\n### Cost anchors (advice {advice['advice_id']})\n\n"
        "Measured on other inputs; they do not predict this input.\n\n"
        + "\n".join(lines)
        + "\n"
    )


def render_task_section(
    summary: dict[str, Any],
    schemas: list[dict[str, Any]],
    advice: dict[str, Any] | None,
) -> str:
    report = summary["report"]
    table, required, accepted = _mode_table(report, schemas)
    fields = ", ".join(f"`{field['name']}`" for field in report["fields"])
    axes = ", ".join(f"`{axis['name']}`" for axis in (report["axes"] or [])) or "none"
    features = ", ".join(
        f"`{field['name']}`"
        for field in report["fields"]
        if field["name"] != report["lhs_field"]
    )
    section = _asset("task", summary["task_kind"]).format(
        target=json.dumps(summary["target"]),
        fields=fields,
        axes=axes,
        features=features,
        availability=table,
        required=required,
    )
    if advice is None:
        return section
    return section + _cost_anchors(advice, summary["task_kind"], accepted)


def render_pinned_glossary(task_kind: str) -> str:
    return _asset("pinned-input", task_kind)
