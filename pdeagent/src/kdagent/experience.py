
from __future__ import annotations

import bisect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

from kdagent import priors

__all__ = [
    "Reading",
    "Retrieval",
    "features_of",
    "load_experience",
    "readings",
    "retrieve",
]



Experience = dict[str, Any]


class Reading(TypedDict):

    value: Any
    status: Literal["computed", "not_computed", "infinite", "undefined"]
    axis: str | None
    raw: float | None


class Retrieval(TypedDict):

    rows: list[dict[str, Any]]
    readings: dict[str, Reading]
    vocabulary: list[str]
    templates: list[list[str]]
    bins: dict[str, list[float]]
    min_support: int
    rules_version: str
    task_kind: str


FORMAT = "kd-experience-v2"
VERSION = 2




BIN_KEYS: tuple[str, ...] = ("uxx_gap_bin", "noise_bin", "ut_gap_bin")

FEATURES: dict[str, tuple[str, ...]] = {
    "pde-evolution": ("task_kind", *priors.FEATURES, *BIN_KEYS),
    "pde-steady": ("task_kind", *priors.FEATURES),
    "tabular-regression": ("task_kind", "topology", "n_fields", "points_bucket"),
    "ode": ("task_kind", "topology", "n_fields", "points_bucket"),
}

DOCUMENT_KEYS = frozenset(
    {
        "format",
        "version",
        "built_at",
        "builder",
        "source",
        "rules_version",
        "examinee_versions",
        "instrument_schema_sha256",
        "features_vocabulary",
        "bins",
        "templates",
        "min_support",
        "excluded_groups",
        "rows",
    }
)

ROW_KEYS = frozenset({"match", "pipeline", "conditions", "result", "cost", "support"})

NESTED_ROW_KEYS: dict[str, frozenset[str]] = {
    "pipeline": frozenset({"algorithm", "mode", "config"}),
    "conditions": frozenset(
        {"device", "soft_seconds", "blas_threads", "python_hash_seed"}
    ),
    "result": frozenset(
        {"structure_rate", "median_max_relative_error", "median_heldout_nrmse"}
    ),
    "cost": frozenset({"median_evaluations", "median_wall_seconds"}),
    "support": frozenset({"n_runs", "n_problems"}),
}

SOURCE_KEYS = frozenset({"kind", "digest"})

FORBIDDEN_CONFIG_KEYS = frozenset(
    {"terms", "library", "cap", "max_evaluations", "max_search_seconds"}
)


def load_experience(path: Path) -> Experience:
    document: Experience = json.loads(path.read_text(encoding="utf-8"))
    if document["format"] != FORMAT or document["version"] != VERSION:
        raise ValueError(
            f"{path}: rebuild the document as {FORMAT} version {VERSION}; "
            f"received format {document['format']!r}, version {document['version']!r}"
        )
    missing = sorted(DOCUMENT_KEYS - {"excluded_groups"} - set(document))
    if missing:
        raise ValueError(f"{path} lacks the required document keys {missing}")
    _check_index(document, path)
    vocabulary = document["features_vocabulary"]
    unknown_document_keys = sorted(set(document) - DOCUMENT_KEYS)
    if unknown_document_keys:
        raise ValueError(
            f"{path} carries document keys {unknown_document_keys} outside "
            f"{sorted(DOCUMENT_KEYS)}"
        )
    unknown_source_keys = sorted(set(document["source"]) - SOURCE_KEYS)
    if unknown_source_keys:
        raise ValueError(
            f"{path} carries source keys {unknown_source_keys} outside "
            f"{sorted(SOURCE_KEYS)}"
        )
    min_support = document["min_support"]



    if type(min_support) is not int or min_support < 2:
        raise ValueError(
            f"{path} declares min_support {min_support!r}; the support floor is "
            "an integer of at least 2"
        )
    for index, row in enumerate(document["rows"]):
        kind = row["match"].get("task_kind")
        if kind not in FEATURES:
            raise ValueError(f"row {index}: match requires a valid task_kind")
        templates = [frozenset(template) for template in document["templates"][kind]]
        _check_row(row, index, vocabulary[kind], templates, min_support)
    return document


def _check_index(document: Experience, path: Path) -> None:
    vocabulary = document["features_vocabulary"]
    if vocabulary != {kind: list(keys) for kind, keys in FEATURES.items()}:
        raise ValueError(
            f"{path}: features_vocabulary must use the {FORMAT} task vocabularies"
        )
    if set(document["templates"]) != set(FEATURES):
        raise ValueError(f"{path}: templates must be indexed by task_kind")
    unknown_bins = sorted(set(document["bins"]) - set(BIN_KEYS))
    if unknown_bins:
        raise ValueError(f"{path}: bins carries unknown keys {unknown_bins}")
    for kind, templates in document["templates"].items():
        for template in templates:
            unknown = sorted(set(template) - set(vocabulary[kind]))
            if unknown:
                raise ValueError(
                    f"{path}: {kind} templates carry unknown keys {unknown}"
                )
            if "task_kind" not in template:
                raise ValueError(f"{path}: {kind} templates require task_kind")


def _check_row(
    row: dict[str, Any],
    index: int,
    vocabulary: Sequence[str],
    templates: Sequence[frozenset[str]],
    min_support: int,
) -> None:
    unknown_row_keys = sorted(set(row) - ROW_KEYS)
    if unknown_row_keys:
        raise ValueError(
            f"row {index} carries keys {unknown_row_keys} outside {sorted(ROW_KEYS)}"
        )
    for name, allowed in NESTED_ROW_KEYS.items():
        unknown = sorted(set(row[name]) - allowed)
        if unknown:
            raise ValueError(
                f"row {index} carries {name} keys {unknown} outside {sorted(allowed)}"
            )
    forbidden = sorted(FORBIDDEN_CONFIG_KEYS & set(row["pipeline"]["config"]))
    if forbidden:
        raise ValueError(
            f"row {index} carries config keys {forbidden}: a term library is "
            "equation shape and a budget is the exam's, neither is a pipeline's"
        )
    _check_match(row["match"], index, vocabulary, templates)
    n_problems = row["support"]["n_problems"]


    if type(n_problems) is not int:
        raise ValueError(
            f"row {index} records n_problems {n_problems!r}; the number of "
            "distinct problems behind a row is an integer count"
        )
    if n_problems < min_support:
        raise ValueError(
            f"row {index} records n_problems={n_problems} below min_support="
            f"{min_support}: a row that fewer problems reach is a fingerprint "
            "of them"
        )


def _check_match(
    match: dict[str, Any],
    index: int,
    vocabulary: Sequence[str],
    templates: Sequence[frozenset[str]],
) -> None:
    unknown_match_keys = sorted(set(match) - set(vocabulary))
    if unknown_match_keys:
        raise ValueError(
            f"row {index} matches on {unknown_match_keys}, outside the "
            f"features_vocabulary: a key outside it is a problem identity in "
            "disguise or dead"
        )
    if frozenset(match) not in templates:
        raise ValueError(
            f"row {index} matches on {sorted(match)}, which is none of the "
            "document's templates: no walk of `retrieve` would read it"
        )
    null_keys = sorted(key for key, value in match.items() if value is None)
    if null_keys:
        raise ValueError(
            f"row {index} matches {null_keys} on null: a feature kd did not "
            "compute would equal it and hit a template it may not"
        )


def readings(
    report: dict[str, Any],
    task_kind: str,
    vocabulary: Sequence[str],
    bins: dict[str, Sequence[float]],
) -> dict[str, Reading]:
    coarse = {"task_kind": task_kind, **priors.features_from_report(report)}
    return {
        key: (
            {"value": coarse[key], "status": "computed", "axis": None, "raw": None}
            if key in FEATURES[task_kind] and key not in BIN_KEYS
            else _diagnostic_reading(report, task_kind, key, bins)
        )
        for key in vocabulary
    }


def _diagnostic_reading(
    report: dict[str, Any],
    task_kind: str,
    key: str,
    bins: dict[str, Sequence[float]],
) -> Reading:
    undefined: Reading = {
        "value": None,
        "status": "undefined",
        "axis": None,
        "raw": None,
    }
    if (
        key not in FEATURES[task_kind]
        or report["lhs_order"] < 1
        or report["lhs_field"] is None
    ):
        return undefined
    lhs_axis = report["lhs_axis"]
    spatial = [axis["name"] for axis in report["axes"] if axis["name"] != lhs_axis]
    if len(spatial) != 1:
        return undefined
    field = next(
        entry for entry in report["fields"] if entry["name"] == report["lhs_field"]
    )
    diagnostic, axis = {
        "uxx_gap_bin": (field["second_derivative_gap"], spatial[0]),
        "noise_bin": (field["noise_estimate"], spatial[0]),
        "ut_gap_bin": (field["first_derivative_gap"], lhs_axis),
    }[key]
    if axis not in diagnostic:
        return {"value": None, "status": "not_computed", "axis": axis, "raw": None}
    raw = diagnostic[axis]
    if raw is None:
        return {
            "value": len(bins[key]),
            "status": "infinite",
            "axis": axis,
            "raw": None,
        }
    return {
        "value": bisect.bisect_right(bins[key], raw),
        "status": "computed",
        "axis": axis,
        "raw": raw,
    }


def features_of(
    report: dict[str, Any],
    task_kind: str,
    vocabulary: Sequence[str],
    bins: dict[str, Sequence[float]],
) -> dict[str, Any]:
    return {
        key: reading["value"]
        for key, reading in readings(report, task_kind, vocabulary, bins).items()
    }


def retrieve(
    document: Experience, task_kind: str, features: dict[str, Any], k: int = 2
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for template in reversed(document["templates"][task_kind]):
        keys = set(template)
        hits = [
            row
            for row in document["rows"]
            if row["match"]["task_kind"] == task_kind
            and set(row["match"]) == keys
            and all(
                features[key] is not None and features[key] == row["match"][key]
                for key in keys
            )
        ]
        for row in sorted(hits, key=_rank):
            if len(chosen) == k:
                return chosen
            if all(row["pipeline"] != taken["pipeline"] for taken in chosen):
                chosen.append(row)
    return chosen


def _rank(row: dict[str, Any]) -> tuple[Any, ...]:
    result = row["result"]
    error = result["median_max_relative_error"]
    nrmse = result["median_heldout_nrmse"]
    evaluations = row["cost"]["median_evaluations"]
    return (
        -result["structure_rate"],
        error is None,
        0.0 if error is None else error,
        nrmse is None,
        0.0 if nrmse is None else nrmse,
        evaluations is None,
        0.0 if evaluations is None else evaluations,
    )
