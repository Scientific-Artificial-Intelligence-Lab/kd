
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ASSET = Path(__file__).resolve().parent / "_assets" / "experience" / "render.md"
_DIAGNOSTICS = {
    "uxx_gap_bin": "second_derivative_gap",
    "noise_bin": "noise_estimate",
    "ut_gap_bin": "first_derivative_gap",
}


def _periodic_reading(reading: dict[str, Any]) -> str:
    if reading["status"] == "undefined":
        return (
            "x_periodic: undefined for this input (requires spatial axes "
            "and a time-derivative target)"
        )
    if "axes" in reading:
        return (
            "x_periodic = false (the low-pass route requires one spatial axis; "
            f"this input has spatial axes {', '.join(reading['axes'])}, "
            "irrespective of their periodic flags)"
        )
    return (
        f"x_periodic = {json.dumps(reading['value'])} (whether the input "
        f"declares its spatial axis {reading['axis']} periodic)"
    )


def _reading(
    key: str, reading: dict[str, Any], task_kind: str, bins: dict[str, list[float]]
) -> str:
    status = reading["status"]
    if key == "x_periodic":
        return _periodic_reading(reading)
    if status == "undefined":
        return (
            f"{key}: undefined for this input (bins require spatial axes "
            "and a time-derivative target field)"
        )
    measurement = (
        f"{reading['reduction']} across spatial axes {', '.join(reading['axes'])}"
        if "reduction" in reading
        else f"along {reading['axis']}"
    )
    if status == "not_computed":
        if "reduction" in reading:
            return (
                f"{key}: not computed ({_DIAGNOSTICS[key]} {reading['reduction']} "
                f"requires every spatial axis: {', '.join(reading['axes'])}; "
                "kd did not compute at least one); rows indexed on it cannot match this input"
            )
        return (
            f"{key}: not computed (kd computed no {_DIAGNOSTICS[key]} along {reading['axis']}: "
            "the axis is not uniform, has fewer than 5 points, or the field holds NaN or Inf); "
            "rows indexed on it cannot match this input"
        )
    value = reading["value"]
    if key in _DIAGNOSTICS:
        raw = "infinity" if status == "infinite" else str(reading["raw"])
        return (
            f"{key} = {value} ({_DIAGNOSTICS[key]} {measurement} = {raw}; "
            f"edges {json.dumps(bins[key])}; the top bin includes infinity)"
        )
    suffix = (
        " (columns of the table, target included)"
        if key == "n_fields" and task_kind in {"tabular-regression", "ode"}
        else ""
    )
    return f"{key} = {value}{suffix}"


def _reported(value: Any, missing: str) -> str:
    return missing if value is None else str(value)


_INPUT_STEPS = {
    "raw": "on the input as given",
    "lowpass": (
        "on a low-passed copy of the input (smooth_input, then the search on "
        "the derived id)"
    ),
    "pinn": (
        "on a PINN-refined field of the input (refine_candidate with "
        "pinn_epochs, then the search on the derived id)"
    ),
    "network": "on a sampled field network",
    "savgol": (
        "on a Savitzky-Golay-smoothed copy of the input "
        '(smooth_input(method="savgol", savgol_window=<the recorded '
        "window_length>), then the search on the derived id)"
    ),
    "composite": "on a sequence of derived inputs",
}
_SURROGATE_STEP = "with a trained derivative surrogate (train_surrogate on that input, passed as surrogate=)"
_MATCH = {"task_kind": "task kind", "topology": "topology", "n_axes": "axis count"}


def _input(pipeline: dict[str, Any]) -> str:
    kind = pipeline.get("input")
    if kind is None:
        return ""
    base, _, surrogate = kind.partition("+")
    text = _INPUT_STEPS[base]
    if surrogate:
        text += f", {_SURROGATE_STEP}"
    return f" Searches ran {text}."


def _budget(conditions: dict[str, Any]) -> str:
    seconds, tier = conditions["soft_seconds"], conditions["budget_tier"]
    if tier == "pipeline":
        return (
            f"complete pipeline cap {seconds} s (including preparation and refinement)"
        )
    if tier == "full":
        return f"problem budget {seconds} s (search cap at least half of it)"
    if tier == "short":
        return f"problem budget {seconds} s, short tier (search cap under half of it)"
    return "no search cap" if seconds is None else f"search cap {seconds} s"


def _noise(match: dict[str, Any]) -> str:
    if "noise_bin" in match:
        return (
            "Noise matching: this row matches this input's noise_bin "
            f"{match['noise_bin']}."
        )
    return (
        "Noise matching: noise_bin was not used for this row; the noise levels "
        "of its training runs are not recorded here."
    )


def _coverage(n_problems: int) -> str:
    if n_problems < 3:
        return (
            f" Limited coverage: this row represents only {n_problems} training "
            "problems."
        )
    return ""


def _ungraded(support: dict[str, Any]) -> str:
    if "n_refused" not in support:
        return ""
    return (
        "Observations in this denominator that failed during the search or were "
        f"not graded: {support['n_refused']}. "
    )


def _records_calls(pipeline: dict[str, Any]) -> bool:
    return isinstance(pipeline.get("refinement"), list)


def _context(pipeline: dict[str, Any], calls: bool) -> str:
    conditioning = pipeline.get("conditioning", {})
    context = ""
    if conditioning:
        context += (
            f" Full search conditioning: {json.dumps(conditioning, sort_keys=True)}."
        )
    if pipeline.get("input_steps") and not calls:
        context += f" Ordered input transformations: {json.dumps(pipeline['input_steps'], sort_keys=True)}."
    if pipeline.get("continuation"):
        context += f" Continuation schedule (incremental generations, conditions, checkpoint iteration and reseed per segment): {json.dumps(pipeline['continuation'], sort_keys=True)}."
    if calls:
        context += " The grade is for the result of the calls listed below, not for the search alone. The search cost below excludes input preparation and refinement."
    elif pipeline.get("refinement"):
        context += f" The grade is for this refinement of the search answer: {json.dumps(pipeline['refinement'], sort_keys=True)}. The search cost below excludes refinement expenditure, which is not reported."
    return context


def _search(pipeline: dict[str, Any], dataset: str) -> str:
    conditioning = pipeline.get("conditioning", {})
    config = {
        **pipeline["config"],
        **{key: value for key, value in conditioning.items() if key != "sketch"},
    }
    return (
        f"run_discovery(dataset_id={dataset}, algorithm={json.dumps(pipeline['algorithm'])}, "
        f"params={json.dumps(config, sort_keys=True)}, seed=<yours>"
        f"{', sketch=' + json.dumps(conditioning['sketch'], sort_keys=True) if 'sketch' in conditioning else ''}); "
        "the config is complete for this kd version."
    )


_STEP_TEXT = {"then": "Then", "choose": "Choose", "note": "Note"}
_MEASURED = "measured_"


def _step(number: int, step: dict[str, Any]) -> str:
    arguments = ", ".join(
        f"{key} {json.dumps(value, sort_keys=True)}"
        for key, value in step.items()
        if key != "tool" and key not in _STEP_TEXT and not key.startswith(_MEASURED)
    )
    text = f"{number}. {step['tool']}" + (f" with {arguments}" if arguments else "")
    text += "." + "".join(
        f" {label}: {step[key]}." for key, label in _STEP_TEXT.items() if key in step
    )
    measured = [
        f"{key.removeprefix(_MEASURED).replace('_', ' ')} "
        f"{value if isinstance(value, str) else json.dumps(value)}"
        for key, value in step.items()
        if key.startswith(_MEASURED)
    ]
    return text + (f" Measured {'; '.join(measured)}." if measured else "")


def _calls(pipeline: dict[str, Any], dataset_id: str) -> str:
    steps = pipeline.get("input_steps", [])
    dataset = (
        f"<primary.dataset_id returned by call {len(steps)}>"
        if steps
        else json.dumps(dataset_id)
    )
    lines = [
        *(_step(number, step) for number, step in enumerate(steps, 1)),
        f"{len(steps) + 1}. {_search(pipeline, dataset)}",
        *(
            _step(number, step)
            for number, step in enumerate(pipeline["refinement"], len(steps) + 2)
        ),
    ]
    return (
        "To follow this row, make these calls in order; a quoted description "
        "stands for a value of this run, such as the pinned input or an id an "
        "earlier call returned:\n" + "\n".join(lines)
    )


def _recipe(pipeline: dict[str, Any], dataset_id: str, support: dict[str, Any]) -> str:
    if _records_calls(pipeline):
        ungraded = _ungraded(support).rstrip()
        return _calls(pipeline, dataset_id) + (f"\n{ungraded}" if ungraded else "")
    return (
        f"To reproduce: {_search(pipeline, json.dumps(dataset_id))} {_ungraded(support)}"
        "Apply any listed input transformations, continuation schedule and refinement separately."
    )


def _rate(result: dict[str, Any], support: dict[str, Any]) -> str:
    if "n_completed" in support:
        completed = support["n_completed"]
        rate = (
            f"{result['structure_rate']} "
            f"({support['n_structure_recovered']}/{completed} completed runs)"
            if completed
            else "not measured (0 completed runs)"
        )
        return (
            f"Over {support['n_runs']} attempted runs on {support['n_problems']} "
            f"training problems ({completed} completed; {support['n_stopped']} "
            f"stopped early and excluded from this rate), conditional structure_rate {rate}"
        )
    return (
        f"Over {support['n_runs']} runs on {support['n_problems']} training problems, "
        f"judged structure_rate {result['structure_rate']}"
    )


def _wall_cost(cost: dict[str, Any]) -> str:
    wall = _reported(cost["median_wall_seconds"], "not reported")
    if "median_pipeline_seconds" in cost:
        pipeline = _reported(cost["median_pipeline_seconds"], "not reported")
        return (
            f"median pipeline wall {pipeline} s "
            f"(network training + search + subset scoring); median search wall {wall} s"
        )
    return f"median wall {wall} s"


def _row(
    index: int, row: dict[str, Any], vocabulary: list[str], dataset_id: str
) -> str:
    pipeline, result, cost = row["pipeline"], row["result"], row["cost"]
    support, conditions = row["support"], row["conditions"]
    used = [key for key in vocabulary if key in row["match"]]
    unused = [key for key in vocabulary if key not in row["match"]]
    return (
        f"Row {index}: {pipeline['algorithm']}, mode {pipeline['mode']}.{_input(pipeline)} "
        f"{_context(pipeline, _records_calls(pipeline))} "
        f"Indexed on {len(used)} keys: {', '.join(used)}; "
        f"keys not used for matching: {', '.join(unused) or 'none'}. "
        f"{_noise(row['match'])} "
        f"{_rate(result, support)}; median max relative coefficient error "
        f"{_reported(result['median_max_relative_error'], 'not measured')}; median held-out NRMSE "
        f"{_reported(result['median_heldout_nrmse'], 'not measured')}; {_wall_cost(cost)} and "
        f"{_reported(cost['median_evaluations'], 'not reported')} candidate evaluations, measured on "
        f"{conditions['device']} (blas_threads {conditions['blas_threads']}, "
        f"PYTHONHASHSEED {_reported(conditions['python_hash_seed'], 'not set')}, "
        f"{_budget(conditions)})."
        f"{_coverage(support['n_problems'])} {_recipe(pipeline, dataset_id, support)}"
    )


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _listing(names: list[str]) -> str:
    if len(names) <= 1:
        return ", ".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _refusal(index: int, row: dict[str, Any]) -> str:
    algorithm, params, refusal = row["algorithm"], row["params"], row["refusal"]
    name, status, correction = (
        algorithm["name"],
        algorithm["status"],
        refusal["correction"],
    )
    declared, undeclared = params["declared"], params["undeclared"]
    typed = []
    if declared:
        typed.append(
            f"parameter{'s' if len(declared) > 1 else ''} {', '.join(declared)}"
        )
    if undeclared:
        typed.append(_count(undeclared, "undeclared parameter name"))
    with_params = " and ".join(typed) or "no parameters"
    if status == "unregistered":
        subject = f"an algorithm name kd does not register, with {with_params}"
    elif status == "registered_name_with_suffix":
        subject = (
            f"{name} typed with a suffix after the algorithm name, with {with_params}"
        )
    else:
        subject = f"{name} with {with_params}"
    support = (
        f"{_count(row['support']['n_calls'], 'call')} on "
        f"{_count(row['support']['n_problems'], 'training problem')}"
    )
    if refusal["phase"] == "fit":
        cause = (
            f"kd found no {name} mode ({', '.join(correction['modes'])}) that "
            "accepted those training inputs (their field names, axes or target "
            "order can differ from this input's; the Availability table says "
            f"whether a mode accepts this input); {refusal['exception']} inside "
            f"fit, before the search started, {support}"
        )
    else:
        cause = f"{refusal['exception']} at construction, {support}"
    tail = ""
    if refusal["category"] == "unregistered_algorithm" and correction is not None:
        tail = (
            f'; kd takes algorithm="{name}" and selects its mode '
            f"({', '.join(correction['modes'])}) from the input and parameters, "
            "not from a suffix"
        )
    elif refusal["category"] == "facade_owned_field":
        tail = "; " + ", ".join(
            f"{field} is set through {facade}="
            for field, facade in sorted(correction["fields"].items())
        )
    return f"Refusal {index} ({refusal['category']}): {subject}: {cause}{tail}."


def render(
    *,
    task_kind: str,
    dataset_id: str,
    rows: list[dict[str, Any]],
    refusals: list[dict[str, Any]],
    readings: dict[str, dict[str, Any]],
    vocabulary: list[str],
    templates: list[list[str]],
    bins: dict[str, list[float]],
    min_support: int,
    rules_version: str,
    usage_scope: str | None = None,
    lessons: list[dict[str, Any]] | None = None,
) -> str:
    if not rows and not refusals and not lessons:
        return ""
    head, rest = _ASSET.read_text(encoding="utf-8").split("<!-- rows -->\n")
    rows_section, refusals_section = rest.split("<!-- refusals -->\n")
    features = "\n".join(
        _reading(key, readings[key], task_kind, bins) for key in vocabulary
    )
    text = head.format(features=features)
    if usage_scope == "re-attempt":
        text += "\nUsage scope: re-attempt; training may include this problem.\n"
    elif usage_scope == "leave-one-out":
        text += "\nUsage scope: leave-one-out; training excludes this problem.\n"
        rows_section = rows_section.replace(
            "which may include this input", "which exclude this problem"
        )
    if rows:
        text += _rows(
            rows,
            rows_section,
            dataset_id,
            vocabulary,
            templates,
            min_support,
            rules_version,
        )
    if refusals:
        text += refusals_section.format(
            match=_listing([_MATCH[key] for key in refusals[0]["match"]]),
            refusals="\n".join(
                _refusal(index, row) for index, row in enumerate(refusals, 1)
            ),
        )
    if lessons:
        text += (
            "\nCurated observations (separate from the pipeline success rates; "
            "conditions and limitations are part of each observation):\n"
            + "\n".join(
                f"- {lesson['text']} [Contributing problems: {lesson['n_problems']}]."
                for lesson in lessons
            )
            + "\n"
        )
    return text


def _rows(
    rows: list[dict[str, Any]],
    section: str,
    dataset_id: str,
    vocabulary: list[str],
    templates: list[list[str]],
    min_support: int,
    rules_version: str,
) -> str:
    search, pipeline = (
        _ASSET.with_name("denominators.md")
        .read_text(encoding="utf-8")
        .split("<!-- pipeline -->\n")
    )
    denominator = (
        pipeline.strip()
        if all(row["conditions"]["budget_tier"] == "pipeline" for row in rows)
        else search.strip()
    )
    return section.format(
        denominator=denominator,
        templates=", ".join(f"{len(keys)} keys" for keys in templates),
        min_support=min_support,
        rules_version=rules_version,
        rows="\n\n".join(
            _row(index, row, vocabulary, dataset_id)
            for index, row in enumerate(rows, 1)
        ),
    )
