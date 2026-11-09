
from __future__ import annotations

import re
from typing import Any

import kd
from kd.search import mode_inputs

from kdagent import experience


def retrieve_for_report(
    document: experience.Experience,
    task_kind: str,
    report: dict[str, Any],
    *,
    k: int = 2,
) -> list[dict[str, Any]]:
    features = experience.features_of(
        report, task_kind, document["features_vocabulary"][task_kind], document["bins"]
    )
    if task_kind != "pde-evolution" or len(report["fields"]) <= features["n_fields"]:
        return experience.retrieve(document, task_kind, features, k=k)
    companions = {
        field["name"]
        for field in report["fields"]
        if field["name"] != report["lhs_field"]
        and (
            report["lhs_axis"] in (field.get("constant_axes") or [])
            or report["lhs_axis"] in (field.get("noise_limited_axes") or [])
        )
    }
    schemas = {schema["algorithm"]: schema for schema in kd.instrument_schemas()}
    return experience.retrieve(
        document,
        task_kind,
        features,
        k=k,
        eligible=lambda row: _applicable(row, report, companions, schemas),
    )


def _uses(pipeline: dict[str, Any], method: str) -> bool:
    return pipeline.get("input", "raw").split("+")[0] == method or any(
        step.get("kind") == method
        or (
            step.get("tool") == "smooth_input"
            and step.get("method", "lowpass") == method
        )
        for step in pipeline.get("input_steps", [])
    )


def _applicable(
    row: dict[str, Any],
    report: dict[str, Any],
    companions: set[str],
    schemas: dict[str, Any],
) -> bool:
    pipeline = experience.instantiate(row, report)["pipeline"]
    if _uses(pipeline, "network"):
        return False
    if _uses(pipeline, "lowpass") and not all(
        axis["periodic"]
        for axis in report["axes"]
        if axis["name"] != report["lhs_axis"]
    ):
        return False
    schema = schemas[pipeline["algorithm"]]
    mode = next(mode for mode in schema["modes"] if mode["name"] == pipeline["mode"])
    verdict = kd.mode_accepts(mode, **mode_inputs(report))
    if not verdict.accepts or not _matches_mode(pipeline, schema, mode, verdict.params):
        return False
    conditioning = pipeline.get("conditioning", {})
    if mode["names_from"] in verdict.params and mode["names_from"] not in conditioning:
        return False
    return all(
        _covers(conditioning[name], companions, report)
        for name in ("terms", "library")
        if name in conditioning
    )


def _matches_mode(
    pipeline: dict[str, Any],
    schema: dict[str, Any],
    mode: dict[str, Any],
    required: dict[str, Any],
) -> bool:
    supplied = {**pipeline["config"], **pipeline.get("conditioning", {})}
    defaults = {
        field["name"]: field["default"]
        for field in [*schema["fields"], *schema["facade_params"]]
    }
    effective = {**defaults, **supplied}
    if any(effective.get(key) != value for key, value in mode["selected_by"].items()):
        return False
    if any(
        key in supplied and supplied[key] != value
        for key, value in mode["config_values"].items()
    ):
        return False
    selector = mode["selector"]
    if selector is not None and (
        (effective.get(selector["field"]) is not None) != selector["present"]
    ):
        return False
    return all(
        effective.get(key) is not None
        if key in mode["required_params"]
        else effective.get(key) == value
        for key, value in required.items()
        if key != mode["names_from"]
    )


def _covers(
    library: list[str] | dict[str, Any],
    companions: set[str],
    report: dict[str, Any],
) -> bool:
    if isinstance(library, dict):
        return companions <= set(library["state_vars"])
    identifiers = {
        token for term in library for token in re.findall(r"\b[A-Za-z_]\w*\b", term)
    }
    fields = {field["name"] for field in report["fields"]}
    axes = {axis["name"] for axis in report["axes"]}
    covered = {_field_of(token, fields, axes) for token in identifiers}
    return companions <= covered


def _field_of(token: str, fields: set[str], axes: set[str]) -> str | None:
    if token in fields:
        return token
    for field in sorted(fields, key=len, reverse=True):
        prefix = f"{field}_"
        if not token.startswith(prefix):
            continue
        segments = token[len(prefix):].split("_")
        if all(
            sum(
                bool(segment)
                and "_" not in axis
                and len(segment) % len(axis) == 0
                and segment == axis * (len(segment) // len(axis))
                for axis in axes
            )
            == 1
            for segment in segments
        ):
            return field
    return None
