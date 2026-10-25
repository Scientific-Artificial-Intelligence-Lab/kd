
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from kd import extension_source, instrument_schemas
from kd.search import CATALOG_FILENAME, DEFAULT_RUNS_ROOT, RunDirPaths
from kdagent import criteria
from kdagent.instrument_doc import instrument_doc_sections
from kdagent.lineage import TREE_FILENAME
from kdagent.surrogate_training import surrogate_parameter_names

DEFAULT_SEED = 0
DEFAULT_GENERATIONS = 10


_PARAM_TABLE_HEAD = (
    "| Parameter | Type | Default | Allowed | Resume | Notes |",
    "|---|---|---|---|---|---|",
)
_EFFECT_NOTES = {
    "preprocessing": "smooths data before the search; grid data only (0 = off)",
    "answer_selection": (
        "drops small-coefficient terms from the reported answer (0 = off)"
    ),
    "max_iterations": "search-loop length (the runner's max_iterations)",
}


def _cell(value: Any) -> str:
    return "" if value is None else f"`{json.dumps(value)}`"


def _text_cell(text: str) -> str:
    return text.replace("|", r"\|")


def _default_cell(schema: dict[str, Any], name: str, default: Any) -> str:
    suffixes = [
        f"({mode['name']}: {json.dumps(mode['default_overrides'][name])})"
        for mode in schema["modes"]
        if name in mode["default_overrides"]
    ]
    return " ".join([_cell(default), *suffixes])


def _override_notes(schema: dict[str, Any]) -> list[str]:
    return [
        f"The `({mode['name']}: ...)` defaults of "
        + ", ".join(f"`{name}`" for name in sorted(mode["default_overrides"]))
        + f" are one group (mode `{mode['name']}`'s `default_overrides`) that kd "
        "applies together and only when the call passes none of these keys; a "
        "call that passes any of them gets the plain default, the first value in "
        "the Default cell, for each key it leaves out."
        for mode in schema["modes"]
        if mode["default_overrides"]
    ]


def _parameter_objects_markdown(schema: dict[str, Any]) -> str:
    notes = []
    for row in schema["fields"]:
        if (
            row["object_fields"] is None
            or row["settable_from"] != "json"
            or row["name"] in schema["facade_inert_fields"]
        ):
            continue
        lines = [
            f"`{row['name']}` takes a JSON object with the keys below. Each key "
            "is optional and replaces the default listed for it, so `{}` runs "
            "the listed defaults."
        ]
        for nested in row["object_fields"]:
            allowed = (
                f", one of {_cell(nested['literal_values'])}"
                if nested["literal_values"] is not None
                else ""
            )
            lines.append(
                f"- `{nested['name']}` (`{nested['type']}`, default "
                f"`{json.dumps(nested['default'])}`{allowed}): "
                f"{nested['description']}"
            )
        notes.append("\n".join(lines))
    return "\n\n".join(notes)


def _parameters_markdown(schema: dict[str, Any]) -> str:
    lines = list(_PARAM_TABLE_HEAD)
    fields = {row["name"]: row for row in schema["fields"]}
    tiers = {name: row["resume_tier"] for name, row in fields.items()}
    tiers.update(
        {name: "identity_breaking" for name in schema["identity_breaking_fields"]}
    )
    for row in schema["facade_params"]:
        if row["name"] == "seed" or row["effect"] == "unused":
            continue
        name = row["config_field"]
        note = fields[name]["description"] if name else _EFFECT_NOTES[row["effect"]]
        default = row["default"]
        if row["name"] == "generations":
            default = DEFAULT_GENERATIONS
            note += f"; this tool applies {DEFAULT_GENERATIONS} if you omit it"
        tier = tiers[name] if name else "facade control"
        lines.append(
            f"| `{row['name']}` | `{row['kind']}` | {_default_cell(schema, name, default)} "
            f"| {_cell(row['literal_values'])} | `{tier}` | {_text_cell(note)} |"
        )
    for row in schema["fields"]:
        if (
            row["settable_from"] != "json"
            or row["name"] in schema["facade_inert_fields"]
        ):
            continue
        default_cell = (
            "**required**"
            if row["required"]
            else _default_cell(schema, row["name"], row["default"])
        )
        lines.append(
            f"| `{row['name']}` | `{_text_cell(row['type'])}` | {default_cell} "
            f"| {_cell(row['literal_values'])} | `{tiers[row['name']]}` "
            f"| {_text_cell(row['description'])} |"
        )
    return "\n\n".join([*_override_notes(schema), "\n".join(lines)])


def _sketch_markdown(schema: dict[str, Any]) -> str:
    modes = schema["modes"]
    if not criteria.sketch_supported(modes):
        return ""
    clauses = list(modes[0]["sketch"])
    lines = [
        "| Clause | " + " | ".join(f"`{mode['name']}`" for mode in modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    for clause in clauses:
        cells = " | ".join(f"`{mode['sketch'][clause]}`" for mode in modes)
        lines.append(f"| `{clause}` | {cells} |")
    derivatives = (
        ", and its derivatives are not `none` (kd subtracts the pinned terms "
        "with its own derivatives for this instrument)"
        if schema["sketch_lower_owner"] == "platform"
        else ""
    )
    selected = [
        f"`{mode['name']}` when `{mode['selector']['field']}` is "
        + ("set" if mode["selector"]["present"] else "not set")
        for mode in modes
        if mode["selector"] is not None
    ]
    selection = (
        " A mode that a parameter selects counts only in a call that selects "
        f"it: {', '.join(selected)}."
        if selected
        else ""
    )
    return (
        "Pass a `kd-sketch-v2` payload as `run_discovery(sketch=...)` to narrow "
        "the search (pinned terms, anchored terms, and holes over a vocabulary; "
        "the common kd-tools card shows the payload). Each clause is enforced at "
        "the level kd declares for the mode the search runs in; a clause the "
        "sketch actually uses must be at least `exit_checked` on every mode "
        "that can run the sketch, or kd refuses the fit naming the clause. A "
        "mode can run the sketch when its form includes `EVOLUTION` and its "
        f"LHS orders include the sketch's{derivatives}; the other modes in "
        f"the table below do not count.{selection}\n\n" + "\n".join(lines)
    )


def _surrogate_markdown(schema: dict[str, Any]) -> str:
    modes = [mode for mode in schema["modes"] if mode["provider_kind"] == "autograd"]
    if not schema["config_artifact_keys"] or not modes:
        return ""
    fields = surrogate_parameter_names(schema)
    selections = "; ".join(
        json.dumps(mode["selected_by"], sort_keys=True) for mode in modes
    )
    training = (
        "There is no early stop unless `autograd_train_patience` and "
        "`autograd_train_val_ratio > 0` are set."
        if "autograd_train_patience" in fields
        else "A resumed segment reuses the checkpointed surrogate; a fresh call "
        "retrains unless a surrogate is injected."
    )
    return (
        "Train once with `train_surrogate`, then inject the registered file with "
        "`run_discovery(surrogate=<surrogate_id>)`. Training accepts "
        + ", ".join(f"`{name}`" for name in fields)
        + ", plus the separate `seed` argument. Injection enters under "
        + ", ".join(f"`{key}`" for key in schema["config_artifact_keys"])
        + f"; select an autograd mode with params {selections}. "
        + training
        + " A lineage keeps one surrogate; a different id requires a fresh lineage."
    )


def run_record_layout() -> dict[str, str]:
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


def applicability_line(schema: dict[str, Any]) -> str:
    parts = []
    for mode in schema["modes"]:
        if not mode["selectable_from_data"]:
            parts.append(f"{mode['name']} is not selectable from input data")
            continue
        text = (
            f"{mode['name']}: {', '.join(mode['topologies'])}, "
            f"{', '.join(mode['forms'])}, LHS orders {mode['lhs_orders']}"
        )
        for key in ("field_names", "spatial_axes", "field_counts", "spatial_dims"):
            if mode[key] is not None:
                text += f", {key}={json.dumps(mode[key])}"
        if mode["names_from"] is not None:
            text += f" (other names require {mode['names_from']})"
        parts.append(text)
    return "; ".join(parts) + "."


def _capabilities_markdown(schema: dict[str, Any]) -> str:
    lines = [
        "| Mode | Form | Data layout | Derivatives | LHS orders | Description |",
        "|---|---|---|---|---|---|",
    ]
    for mode in schema["modes"]:
        cells = [
            f"`{mode['name']}`",
            ", ".join(mode["forms"]),
            ", ".join(mode["topologies"]),
            mode["provider_kind"],
            str(mode["lhs_orders"]),
            _text_cell(mode["description"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("\n### Mode notes\n")
    for mode in schema["modes"]:
        notes = [
            f"{key}={json.dumps(mode[key], sort_keys=True)}"
            for key in (
                "selected_by",
                "order_params",
                "required_params",
                "default_overrides",
                "refused_fields",
                "field_names",
                "spatial_axes",
                "field_counts",
                "spatial_dims",
                "names_from",
            )
        ]
        lines.append(
            f"- `{mode['name']}`: "
            + "; ".join(f"`{note}`" for note in notes)
            + (
                "."
                if mode["selectable_from_data"]
                else "; not selectable from input data."
            )
        )
    lines.append(
        f"\nNative score: `{schema['score_kind']}` "
        f"(direction: `{schema['score_direction']}`). Cost class: `{schema['cost_class']}`."
    )
    return "\n".join(lines)


def instrument_briefs() -> list[dict[str, str]]:
    return [
        {
            "algorithm": schema["algorithm"],
            "summary": schema["summary"],







            "cost_class": schema["cost_class"],
            "applicability": applicability_line(schema),






            "segmentation_archive": schema["segmentation"]["archive"],
            "capabilities": _capabilities_markdown(schema),
            "parameters": _parameters_markdown(schema),
            "parameter_objects": _parameter_objects_markdown(schema),



            "narrowing": _sketch_markdown(schema),
            "surrogate": _surrogate_markdown(schema),


            "source": extension_source("algorithm", schema["algorithm"]),
        }
        for schema in instrument_schemas()
    ]


PARAMETERS_HEADING = "## Parameters"
NARROWING_HEADING = "## Narrowing with a sketch"
SURROGATE_HEADING = "## Surrogate"
_PROSE_SENTENCE = (
    "Generated from kd's instrument schema when this skill was installed; the "
    "card prose above is hand-written, this table is not. "
)
_GENERATED_SENTENCE = (
    "Generated from kd's instrument schema when this skill was installed, as "
    "is the whole card: an extension registered this algorithm through "
    "`kd.register_algorithm`, and no hand-written prose describes it. "
)
_PARAMETERS_PREAMBLE = (
    "Pass these as the `params` mapping of `run_discovery`. `seed` is a "
    "separate argument of that tool, not a `params` key.\n\n"
    "`Resume` is kd's classification of changing that parameter: `resume_safe` "
    "can take effect on a search restored from a checkpoint, `init_only` "
    "cannot, and `identity_breaking` makes it a different run rather than the "
    "same one continued. Facade controls have no config-field tier; generations "
    "adds work, and preprocessing changes the fit input."
)
_RESUMABLE_ARCHIVE = "progress"
_PROGRESS_RESUME_SENTENCE = (
    " Use `resume_from_run_id` without `resume_iteration` to continue from the "
    "latest checkpoint, or add `resume_iteration` to roll back and fork; kd's "
    "tier gate names any parameter it refuses."
)
_CONCLUSION_RESUME_SENTENCE = (
    " This instrument's checkpoint archive carries a finished conclusion rather "
    "than search state, so `run_discovery` refuses `resume_from_run_id` "
    "pointing at one of its runs; run it again with different parameters "
    "instead."
)


def _parameters_preamble(segmentation_archive: str, *, builtin: bool) -> str:
    sentence = (
        _PROGRESS_RESUME_SENTENCE
        if segmentation_archive == _RESUMABLE_ARCHIVE
        else _CONCLUSION_RESUME_SENTENCE
    )
    opening = _PROSE_SENTENCE if builtin else _GENERATED_SENTENCE
    return f"{opening}{_PARAMETERS_PREAMBLE}{sentence}"




_ASSETS_DIR = Path(__file__).resolve().parent / "_assets"


def _prose_sections(algorithm: str) -> tuple[list[str], str]:
    sections = instrument_doc_sections(algorithm)
    guidance = (_ASSETS_DIR / "guidance" / f"{algorithm}.md").read_text(
        encoding="utf-8"
    )
    guidance, _, narrowing_note = guidance.partition("\n### Narrowing with a sketch\n")
    return [
        f"## Search space\n\n{sections['search_space']}",
        f"## Method\n\n{sections['method']}",
        f"## Tool notes\n\n{guidance.strip()}",
    ], narrowing_note


def algorithm_card(brief: dict[str, str]) -> str:
    algorithm = brief["algorithm"]
    builtin = brief["source"] == "builtin"
    prose, narrowing_note = _prose_sections(algorithm) if builtin else ([], "")
    description = (
        f"{brief['summary']} Cost class {brief['cost_class']}. {brief['applicability']}"
    )
    preamble = _parameters_preamble(brief["segmentation_archive"], builtin=builtin)
    parts = [
        f"---\nname: {algorithm}\ndescription: >-\n {description}\n---",
        f"## Capabilities\n\n{brief['capabilities']}",
        *prose,
        f"{PARAMETERS_HEADING}\n\n{preamble}\n\n{brief['parameters']}"
        + (f"\n\n{brief['parameter_objects']}" if brief["parameter_objects"] else ""),
    ]
    for heading, key in (
        (NARROWING_HEADING, "narrowing"),
        (SURROGATE_HEADING, "surrogate"),
    ):
        if brief[key]:
            body = brief[key]
            if key == "narrowing" and narrowing_note:
                body += "\n\n" + narrowing_note.strip()
            parts.append(f"{heading}\n\n{body}")
    return "\n\n".join(parts) + "\n"


SHELL_PASSAGE = re.compile(
    r"^<!-- shell -->\n(?P<body>.*?)^<!-- /shell -->\n(?P<gap>\n?)",
    re.DOTALL | re.MULTILINE,
)


def common_card(*, shell: bool = False) -> str:
    text = (_ASSETS_DIR / "kd-tools.md").read_text(encoding="utf-8")
    text = SHELL_PASSAGE.sub(
        (lambda match: match["body"] + match["gap"]) if shell else "", text
    )
    return text.format(
        DEFAULT_GENERATIONS=DEFAULT_GENERATIONS,
        DEFAULT_SEED=DEFAULT_SEED,
    )


def experience_card() -> str:
    return (_ASSETS_DIR / "experience" / "SKILL.md").read_text(encoding="utf-8")
