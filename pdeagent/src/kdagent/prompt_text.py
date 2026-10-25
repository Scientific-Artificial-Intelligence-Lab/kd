
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


_MINIMAL_VALUES: dict[tuple[str, str, str], Any] = {


    ("discover", "mode2", "pinn"): {},


    ("eqgpt", "single_wave", "sparsity_alpha"): 0.02,
    ("eqgpt", "steady", "sparsity_alpha"): 1.0,
    ("eqgpt", "steady", "steady_activation"): "sin",
}

LLM4ED_OFF = (
    "This run configures no LLM4ED endpoint (the operator enables one with "
    "`kd-agent setup --llm4ed`), so a search with it is refused before it "
    "starts."
)

_BUDGET = {
    "max_iterations": "`generations`: number of {unit}",
    "config_field": "`generations` sets `{field}`, the number of {unit}",
    "unused": "no budget parameter (one fit)",
}


def _minimal(algorithm: str, mode: dict[str, Any]) -> dict[str, Any]:
    return {
        name: _MINIMAL_VALUES[(algorithm, mode["name"], name)]
        for name in mode["required_params"]
        if (algorithm, mode["name"], name) in _MINIMAL_VALUES
    }


def _required_fields(
    schema: dict[str, Any], mode: dict[str, Any]
) -> list[dict[str, str]]:
    rows = {row["name"]: row for row in schema["fields"]}
    return [
        {
            "name": name,
            "type": rows[name]["type"],
            "description": rows[name]["description"],
        }
        for name in mode["required_params"]
    ]


def _unfilled(schema: dict[str, Any], mode: dict[str, Any]) -> list[dict[str, str]]:
    return [
        field
        for field in _required_fields(schema, mode)
        if (schema["algorithm"], mode["name"], field["name"]) not in _MINIMAL_VALUES
    ]


def _parameter_phrase(field: dict[str, str]) -> str:
    return f"`{field['name']}` (`{field['type']}`: {field['description']})"


def _selection(mode: dict[str, Any]) -> str:
    parts = [
        f"`{name}`={json.dumps(value)}"
        for name, value in sorted(mode["selected_by"].items())
    ]
    selector = mode["selector"]
    if selector is not None:
        parts.append(
            f"`{selector['field']}` " + ("passed" if selector["present"] else "omitted")
        )
    return ", ".join(parts) or "the input"


def _budget(schema: dict[str, Any]) -> str:
    generations = next(p for p in schema["facade_params"] if p["name"] == "generations")
    lever = _BUDGET[generations["effect"]].format(
        unit=schema["segmentation"]["unit"], field=generations["config_field"]
    )
    then = (
        "continue with `resume_from_run_id`"
        if schema["segmentation"]["archive"] == "progress"
        else "rerun with other params"
    )
    return f"{lever}; {then}"


def mode_calls(
    report: dict[str, Any], schemas: list[dict[str, Any]], *, llm4ed: bool = False
) -> list[dict[str, Any]]:
    rows = []
    inputs = mode_inputs(report)
    for schema in schemas:
        algorithm = schema["algorithm"]
        for mode in schema["modes"]:
            if not mode["selectable_from_data"]:
                continue
            verdict = mode_accepts(mode, **inputs)
            accepts, reason = verdict.accepts, verdict.reason
            if accepts and algorithm == "llm4ed" and not llm4ed:
                accepts, reason = False, LLM4ED_OFF
            builds = (
                mode["names_from"] if mode["names_from"] in verdict.params else None
            )
            unfilled = _unfilled(schema, mode)
            omitted = {builds, *(field["name"] for field in unfilled)}
            params = {
                name: value
                for name, value in verdict.params.items()
                if name not in omitted
            }
            params.update(_minimal(algorithm, mode))
            rows.append(
                {
                    "algorithm": algorithm,
                    "mode": mode["name"],
                    "accepts": accepts,
                    "reason": reason,
                    "selection": _selection(mode),
                    "params": params,
                    "builds": builds,
                    "required": list(mode["required_params"]),
                    "unfilled": unfilled,
                    "order_params": mode["order_params"].get(
                        str(report["lhs_order"]), {}
                    ),
                    "budget": _budget(schema),
                }
            )
    return rows


def _mode_table(
    report: dict[str, Any], schemas: list[dict[str, Any]], llm4ed: bool
) -> tuple[str, str, set[tuple[str, str]]]:
    lines = [
        "| `algorithm` | Mode | Accepts | Reason | Selected by | `params` | Budget |",
        "|---|---|---|---|---|---|---|",
    ]
    required = []
    accepted: set[tuple[str, str]] = set()
    for row in mode_calls(report, schemas, llm4ed=llm4ed):
        reason = (row["reason"] or "").replace("|", r"\|")
        params = (
            f"`{json.dumps(row['params'], sort_keys=True)}`" if row["accepts"] else ""
        )
        lines.append(
            f"| `{row['algorithm']}` | {row['mode']} | "
            f"{'yes' if row['accepts'] else 'no'} | {reason} | {row['selection']} | "
            f"{params} | {row['budget']} |"
        )
        if row["accepts"]:
            accepted.add((row["algorithm"], row["mode"]))
            notes = _required_notes(row)
            if notes:
                required.append(
                    f"- {_label(row['algorithm'], row['mode'])}: "
                    + "; ".join(notes)
                    + "."
                )
    return "\n".join(lines), "\n".join(required) or _NOTHING_REQUIRED, accepted


def _required_notes(row: dict[str, Any]) -> list[str]:
    notes = [
        f"`{name}` is required, and the row's `params` pass "
        f"{json.dumps(row['params'][name])}, a value kd accepts"
        for name in row["required"]
        if name in row["params"]
    ]
    notes += [
        f"`{field['name']}` is required and has no example value "
        f"(`{field['type']}`: {field['description']}), so add a value you "
        "choose for this input to the row's `params`"
        for field in row["unfilled"]
    ]
    notes += [
        f"`{name}`={json.dumps(value)} matches this input's target order"
        for name, value in sorted(row["order_params"].items())
    ]
    if row["builds"] is not None:
        notes.append(f"build `{row['builds']}` from this input's field and axis names")
    return notes


def combined_name_refusal(algorithm: str, schemas: list[dict[str, Any]]) -> str | None:
    if "/" not in algorithm:
        return None
    name, _, mode_name = algorithm.partition("/")
    schema = next((item for item in schemas if item["algorithm"] == name), None)
    mode = next(
        (
            item
            for item in (schema or {"modes": []})["modes"]
            if item["name"] == mode_name and item["selectable_from_data"]
        ),
        None,
    )
    if schema is None or mode is None:
        names = [item["algorithm"] for item in schemas]
        return (
            f"algorithm takes an algorithm name alone, one of {names}; the input "
            "and params select the mode, as the Availability table shows"
        )
    params = {**mode["selected_by"], **_minimal(name, mode)}
    unfilled = _unfilled(schema, mode)
    needs = (
        "; its `params` also need "
        + ", ".join(_parameter_phrase(field) for field in unfilled)
        + ", with a value you choose (there is no example value)"
        if unfilled
        else ""
    )
    return (
        f'algorithm takes the algorithm name alone: run_discovery(algorithm="{name}", '
        f"params={json.dumps(params, sort_keys=True)}) runs its {mode_name} mode "
        f"(selected by {_selection(mode)}) on an input that mode accepts{needs}"
    )


def registered_algorithms(schemas: list[dict[str, Any]]) -> str:
    names = ", ".join(f"`{schema['algorithm']}`" for schema in schemas)
    required = [
        f"`{schema['algorithm']}` mode `{mode['name']}` needs "
        + ", ".join(
            _parameter_phrase(field) for field in _required_fields(schema, mode)
        )
        for schema in schemas
        for mode in schema["modes"]
        if mode["selectable_from_data"] and mode["required_params"]
    ]
    text = f"Algorithms kd registers in this run: {names}."
    if required:
        text += (
            " A mode with a required parameter runs only when `params` "
            "carries it: " + "; ".join(required) + "."
        )
    return text


def _label(algorithm: str, mode: str | None) -> str:
    return f"`{algorithm}`" + ("" if mode is None else f", mode {mode}")


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
            label = _label(anchor["algorithm"], anchor["mode"])
            lines.append(f"- {label}: {anchor['text']}")
    if not lines:
        return ""
    return (
        f"\n### Cost anchors (advice {advice['advice_id']})\n\n"
        "Measured on other inputs; they do not predict this input.\n\n"
        + "\n".join(lines)
        + "\n"
    )


def _usage_advice(
    advice: dict[str, Any], task_kind: str, accepted: set[tuple[str, str]]
) -> str:
    lines = []
    for item in advice["usage"]:
        if item["task_kind"] not in (None, task_kind):
            continue
        if item["algorithm"] is None:
            usable, label = True, "general"
        elif item["mode"] is None:
            usable = any(algorithm == item["algorithm"] for algorithm, _ in accepted)
            label = _label(item["algorithm"], None)
        else:
            usable = (item["algorithm"], item["mode"]) in accepted
            label = _label(item["algorithm"], item["mode"])
        if usable:
            lines.append(f"- {label}: {item['text']}")
    if not lines:
        return ""
    return (
        f"\n### Usage advice (advice {advice['advice_id']})\n\n"
        "Written from runs on other inputs; check each point against this "
        "input before acting on it.\n\n" + "\n".join(lines) + "\n"
    )


def render_task_section(
    summary: dict[str, Any],
    schemas: list[dict[str, Any]],
    advice: dict[str, Any] | None,
    *,
    llm4ed: bool = False,
) -> str:
    report = summary["report"]
    table, required, accepted = _mode_table(report, schemas, llm4ed)
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
    return (
        section
        + _cost_anchors(advice, summary["task_kind"], accepted)
        + _usage_advice(advice, summary["task_kind"], accepted)
    )


def render_pinned_glossary(task_kind: str) -> str:
    return _asset("pinned-input", task_kind)
