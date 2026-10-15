
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


def _reading(
    key: str, reading: dict[str, Any], task_kind: str, bins: dict[str, list[float]]
) -> str:
    status = reading["status"]
    if status == "undefined":
        return (
            f"{key}: undefined for this input (bins are defined only for one spatial "
            "axis and a time-derivative target)"
        )
    if status == "not_computed":
        return (
            f"{key}: not computed (kd computed no {_DIAGNOSTICS[key]} along {reading['axis']}: "
            "the axis is not uniform, has fewer than 5 points, or the field holds NaN or Inf); "
            "rows indexed on it cannot match this input"
        )
    value = reading["value"]
    if key in _DIAGNOSTICS:
        raw = "infinity" if status == "infinite" else str(reading["raw"])
        return (
            f"{key} = {value} ({_DIAGNOSTICS[key]} along {reading['axis']} = {raw}; "
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


def _row(
    index: int, row: dict[str, Any], vocabulary: list[str], dataset_id: str
) -> str:
    pipeline, result, cost = row["pipeline"], row["result"], row["cost"]
    support, conditions = row["support"], row["conditions"]
    used = [key for key in vocabulary if key in row["match"]]
    unused = [key for key in vocabulary if key not in row["match"]]
    return (
        f"Row {index}: {pipeline['algorithm']}, mode {pipeline['mode']}. "
        f"Indexed on {len(used)} keys: {', '.join(used)}; "
        f"keys not used for matching: {', '.join(unused) or 'none'}. "
        f"Judged structure_rate {result['structure_rate']} over {support['n_runs']} runs on "
        f"{support['n_problems']} training problems; median max relative coefficient error "
        f"{_reported(result['median_max_relative_error'], 'not measured')}; median held-out NRMSE "
        f"{_reported(result['median_heldout_nrmse'], 'not measured')}; median wall "
        f"{_reported(cost['median_wall_seconds'], 'not reported')} s and "
        f"{_reported(cost['median_evaluations'], 'not reported')} candidate evaluations, measured on "
        f"{conditions['device']} (blas_threads {conditions['blas_threads']}, "
        f"PYTHONHASHSEED {conditions['python_hash_seed']}, soft budget "
        f"{_reported(conditions['soft_seconds'], 'none')}). To reproduce: "
        f"run_discovery(dataset_id={json.dumps(dataset_id)}, algorithm={json.dumps(pipeline['algorithm'])}, "
        f"params={json.dumps(pipeline['config'], sort_keys=True)}, seed=<yours>); "
        "the config is complete for this kd version."
    )


def render(
    *,
    task_kind: str,
    dataset_id: str,
    rows: list[dict[str, Any]],
    readings: dict[str, dict[str, Any]],
    vocabulary: list[str],
    templates: list[list[str]],
    bins: dict[str, list[float]],
    min_support: int,
    rules_version: str,
    advice_id: str | None,
    usage: list[dict[str, Any]],
) -> str:
    if not rows:
        return ""
    features = "\n".join(
        _reading(key, readings[key], task_kind, bins) for key in vocabulary
    )
    rendered = []
    for index, row in enumerate(rows, 1):
        text = _row(index, row, vocabulary, dataset_id)
        for item in usage:
            if (
                item["algorithm"] == row["pipeline"]["algorithm"]
                and item["mode"] in (None, row["pipeline"]["mode"])
                and item["task_kind"] in (None, task_kind)
            ):
                text += f"\nAdvice ({advice_id}): {item['text']}"
        rendered.append(text)
    return _ASSET.read_text(encoding="utf-8").format(
        features=features,
        templates=", ".join(f"{len(keys)} keys" for keys in templates),
        min_support=min_support,
        rules_version=rules_version,
        rows="\n\n".join(rendered),
    )
