
from __future__ import annotations

import bisect
import itertools
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, Literal, NotRequired, TypedDict

from kdagent import priors

__all__ = [
    "Reading",
    "Retrieval",
    "check_refusal_vocabulary",
    "features_of",
    "instantiate",
    "library_terms",
    "load_experience",
    "pipeline_identity",
    "readings",
    "refusal_identity",
    "retrieve",
    "retrieve_lessons",
    "retrieve_refusals",
]



Experience = dict[str, Any]


class Reading(TypedDict):

    value: Any
    status: Literal["computed", "not_computed", "infinite", "undefined"]
    axis: str | None
    raw: float | None
    axes: NotRequired[list[str]]
    reduction: NotRequired[Literal["max"]]


class Retrieval(TypedDict):

    rows: list[dict[str, Any]]
    readings: dict[str, Reading]
    vocabulary: list[str]
    templates: list[list[str]]
    bins: dict[str, list[float]]
    min_support: int
    rules_version: str
    task_kind: str
    refusals: list[dict[str, Any]]
    usage_scope: NotRequired[str]
    lessons: NotRequired[list[dict[str, Any]]]


FORMAT = "kd-experience-v4"
VERSION = 4




BIN_KEYS: tuple[str, ...] = ("uxx_gap_bin", "noise_bin", "ut_gap_bin")

FEATURES: dict[str, tuple[str, ...]] = {
    "pde-evolution": ("task_kind", *priors.FEATURES, *BIN_KEYS),
    "pde-steady": ("task_kind", *priors.FEATURES),
    "tabular-regression": ("task_kind", "topology", "n_fields", "points_bucket"),
    "ode": ("task_kind", "topology", "n_fields", "points_bucket"),
}

OPTIONAL_FEATURES: dict[str, tuple[str, ...]] = {"pde-evolution": ("x_periodic",)}

PERIODIC_KEY = "x_periodic"

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
        "refusals",
        "admission",
        "usage_scope",
        "lessons",
    }
)

ROW_KEYS = frozenset({"match", "pipeline", "conditions", "result", "cost", "support"})

INPUT_KINDS = frozenset(
    {
        "raw",
        "lowpass",
        "savgol",
        "pinn",
        "raw+surrogate",
        "lowpass+surrogate",
        "savgol+surrogate",
        "pinn+surrogate",
        "network",
        "network+surrogate",
        "composite",
        "composite+surrogate",
    }
)

NESTED_ROW_KEYS: dict[str, frozenset[str]] = {
    "pipeline": frozenset(
        {
            "algorithm",
            "mode",
            "config",
            "input",
            "conditioning",
            "refinement",
            "input_steps",
            "continuation",
        }
    ),
    "conditions": frozenset(
        {"device", "soft_seconds", "budget_tier", "blas_threads", "python_hash_seed"}
    ),
    "result": frozenset(
        {"structure_rate", "median_max_relative_error", "median_heldout_nrmse"}
    ),
    "cost": frozenset(
        {"median_evaluations", "median_wall_seconds", "median_pipeline_seconds"}
    ),
    "support": frozenset(
        {
            "n_runs",
            "n_problems",
            "n_refused",
            "n_completed",
            "n_stopped",
            "n_structure_recovered",
        }
    ),
}

SOURCE_KEYS = frozenset({"kind", "digest"})

FORBIDDEN_CONFIG_KEYS = frozenset(
    {"terms", "library", "cap", "max_evaluations", "max_search_seconds"}
)

CALLER_ERROR_TYPES = frozenset({"ValueError", "TypeError", "NotImplementedError"})

REFUSAL_KEYS: dict[str, tuple[str, ...]] = {
    "pde-evolution": ("task_kind", "topology", "n_axes"),
    "pde-steady": ("task_kind", "topology", "n_axes"),
    "tabular-regression": ("task_kind",),
    "ode": ("task_kind",),
}

REFUSAL_CATEGORIES = frozenset(
    {
        "unregistered_algorithm",
        "facade_owned_field",
        "undeclared_parameter",
        "argument_refused",
        "no_mode_accepts_input",
    }
)

REFUSAL_ROW_KEYS = frozenset({"match", "algorithm", "params", "refusal", "support"})

NESTED_REFUSAL_KEYS: dict[str, frozenset[str]] = {
    "algorithm": frozenset({"name", "status"}),
    "params": frozenset({"declared", "undeclared"}),
    "refusal": frozenset({"phase", "exception", "category", "correction"}),
    "support": frozenset({"n_calls", "n_problems"}),
}

ALGORITHM_STATUSES = frozenset(
    {"registered", "registered_name_with_suffix", "unregistered"}
)

REFUSAL_PHASES = frozenset({"construction", "fit"})


def load_experience(path: Path) -> Experience:
    document: Experience = json.loads(path.read_text(encoding="utf-8"))
    if document["format"] != FORMAT or document["version"] != VERSION:
        raise ValueError(
            f"{path}: rebuild the document as {FORMAT} version {VERSION}; "
            f"received format {document['format']!r}, version {document['version']!r}"
        )
    missing = sorted(
        DOCUMENT_KEYS
        - {"excluded_groups", "admission", "usage_scope", "lessons"}
        - set(document)
    )
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
    scope = document.get("usage_scope", "leave-one-out")
    if scope not in ("re-attempt", "leave-one-out"):
        raise ValueError(f"{path}: usage_scope must be re-attempt or leave-one-out")
    floor = 1
    if type(min_support) is not int or min_support < floor:
        raise ValueError(
            f"{path} declares min_support {min_support!r}; the support floor is "
            f"an integer of at least {floor}"
        )
    _check_lessons(document.get("lessons", []), vocabulary, min_support)
    for index, row in enumerate(document["rows"]):
        kind = row["match"].get("task_kind")
        if kind not in FEATURES:
            raise ValueError(f"row {index}: match requires a valid task_kind")
        templates = [frozenset(template) for template in document["templates"][kind]]
        _check_row(row, index, vocabulary[kind], templates, min_support)
    if not isinstance(document["refusals"], list):
        raise ValueError(f"{path}: refusals must be a list of refusal rows")
    identities: set[str] = set()
    for index, row in enumerate(document["refusals"]):
        _check_refusal(row, index, min_support)
        identity = refusal_identity(row)
        if identity in identities:
            raise ValueError(
                f"refusal row {index} repeats the identity of an earlier row"
            )
        identities.add(identity)
    return document


def _check_index(document: Experience, path: Path) -> None:
    vocabulary = document["features_vocabulary"]
    if set(vocabulary) != set(FEATURES) or any(
        vocabulary[kind] != [*FEATURES[kind], *OPTIONAL_FEATURES.get(kind, ())]
        and vocabulary[kind] != list(FEATURES[kind])
        for kind in FEATURES
    ):
        raise ValueError(
            f"{path}: features_vocabulary must use the {FORMAT} task vocabularies, "
            "each optionally followed by its OPTIONAL_FEATURES"
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


def _check_lessons(
    lessons: list[dict[str, Any]], vocabulary: dict[str, list[str]], min_support: int
) -> None:
    if not isinstance(lessons, list):
        raise ValueError("lessons must be a list")
    for lesson in lessons:
        if not isinstance(lesson, dict) or set(lesson) != {
            "match",
            "text",
            "n_problems",
        }:
            raise ValueError("lesson requires exactly match, text and n_problems")
        match = lesson["match"]
        if not isinstance(match, dict) or match.get("task_kind") not in vocabulary:
            raise ValueError("lesson match requires a valid task_kind")
        if set(match) - set(vocabulary[match["task_kind"]]) or any(
            value is None for value in match.values()
        ):
            raise ValueError(
                "lesson match must use computed features in the vocabulary"
            )
        if not isinstance(lesson["text"], str) or not lesson["text"].strip():
            raise ValueError("lesson text must be non-empty")
        count = lesson["n_problems"]
        if type(count) is not int or count < min_support:
            raise ValueError(
                "lesson n_problems is below min_support or is not an integer"
            )


def retrieve_lessons(
    document: Experience, task_kind: str, features: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        lesson
        for lesson in document.get("lessons", [])
        if lesson["match"]["task_kind"] == task_kind
        and all(features[key] == value for key, value in lesson["match"].items())
    ]


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
    if set(row["conditions"]) != NESTED_ROW_KEYS["conditions"]:
        raise ValueError(
            f"row {index} conditions require the exact five keys, including budget_tier"
        )
    if row["conditions"]["budget_tier"] not in ("full", "short", "pipeline", None):
        raise ValueError(
            f"row {index} conditions.budget_tier must be full, short, pipeline or null"
        )
    forbidden = sorted(FORBIDDEN_CONFIG_KEYS & set(row["pipeline"]["config"]))
    if forbidden:
        raise ValueError(
            f"row {index} carries config keys {forbidden}: a term library is "
            "equation shape and a budget is the exam's, neither is a pipeline's"
        )
    if "input" in row["pipeline"] and row["pipeline"]["input"] not in INPUT_KINDS:
        raise ValueError(
            f"row {index} names pipeline input {row['pipeline']['input']!r}, "
            f"outside {sorted(INPUT_KINDS)}"
        )
    check_conditioning(row["pipeline"].get("conditioning", {}))
    if LIBRARY_RULE_KEY in row["pipeline"].get("conditioning", {}):
        _check_rule_steps(row["pipeline"], index)
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





_TERM_ARGUMENTS = ("terms", "term_sets", "law", "support", "candidate")


def _lists_text(value: Any) -> bool:
    if isinstance(value, list):
        return any(isinstance(item, str) or _lists_text(item) for item in value)
    if isinstance(value, dict):
        return any(_lists_text(item) for item in value.values())
    return False


def _check_rule_steps(pipeline: dict[str, Any], index: int) -> None:
    steps = pipeline.get("refinement")
    if not isinstance(steps, list):
        raise ValueError(
            f"row {index} carries a library_rule without recorded calls; a rule "
            "row lists its refinement as calls"
        )
    for step in [*pipeline.get("input_steps", []), *steps]:
        named = [key for key in _TERM_ARGUMENTS if key in step]
        if (
            ("subsets_up_to" in step and "terms" in step)
            or any(not isinstance(step[key], str) for key in named)
            or any(_lists_text(value) for value in step.values())
        ):
            raise ValueError(
                f"row {index} carries a library_rule and names the terms of its "
                f"{step['tool']} call; the rule writes them for each input, and "
                "a later call refers to them in words"
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


def refusal_identity(row: dict[str, Any]) -> str:
    return json.dumps(
        [row["match"], row["algorithm"], row["params"], row["refusal"]],
        sort_keys=True,
        separators=(",", ":"),
    )


def _is_count(value: Any) -> bool:

    return type(value) is int and value >= 0


def _identifiers(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(name, str) and name.isidentifier() for name in value)
        and value == sorted(set(value))
    )


def _check_refusal(row: dict[str, Any], index: int, min_support: int) -> None:
    where = f"refusal row {index}"
    if not isinstance(row, dict) or set(row) != REFUSAL_ROW_KEYS:
        raise ValueError(f"{where} must carry exactly {sorted(REFUSAL_ROW_KEYS)}")
    for name, keys in NESTED_REFUSAL_KEYS.items():
        if not isinstance(row[name], dict) or set(row[name]) != keys:
            raise ValueError(f"{where}: {name} must carry exactly {sorted(keys)}")
    _check_refusal_match(row["match"], where)
    algorithm, params = row["algorithm"], row["params"]
    name, status = algorithm["name"], algorithm["status"]
    if status not in ALGORITHM_STATUSES:
        raise ValueError(
            f"{where}: algorithm.status outside {sorted(ALGORITHM_STATUSES)}"
        )
    if (name is None) != (status == "unregistered"):
        raise ValueError(f"{where}: algorithm.name is null exactly when unregistered")
    if name is not None and not (isinstance(name, str) and name.isidentifier()):
        raise ValueError(f"{where}: algorithm.name must be an identifier")
    if not _identifiers(params["declared"]):
        raise ValueError(
            f"{where}: params.declared must be a sorted list of distinct identifiers"
        )
    if not _is_count(params["undeclared"]):
        raise ValueError(f"{where}: params.undeclared must be a count")
    _check_refusal_cause(row["refusal"], status, where)
    support = row["support"]
    if not (_is_count(support["n_calls"]) and _is_count(support["n_problems"])):
        raise ValueError(f"{where}: support holds integer counts")
    if support["n_problems"] < min_support:
        raise ValueError(
            f"{where} records n_problems={support['n_problems']} below "
            f"min_support={min_support}: a row that fewer problems reach is a "
            "fingerprint of them"
        )
    if support["n_calls"] < support["n_problems"]:
        raise ValueError(f"{where}: n_calls is at least n_problems")


def _check_refusal_match(match: Any, where: str) -> None:
    kind = match.get("task_kind") if isinstance(match, dict) else None
    if kind not in REFUSAL_KEYS:
        raise ValueError(
            f"{where}: match requires a task_kind of {sorted(REFUSAL_KEYS)}"
        )
    if set(match) != set(REFUSAL_KEYS[kind]):
        raise ValueError(
            f"{where}: {kind} matches on exactly {list(REFUSAL_KEYS[kind])}"
        )
    if "topology" in match and not (
        isinstance(match["topology"], str) and match["topology"].isidentifier()
    ):
        raise ValueError(f"{where}: match.topology must be an identifier")
    if "n_axes" in match and not _is_count(match["n_axes"]):
        raise ValueError(f"{where}: match.n_axes must be a count")


def _check_refusal_cause(refusal: dict[str, Any], status: str, where: str) -> None:
    category, correction = refusal["category"], refusal["correction"]
    if refusal["exception"] not in CALLER_ERROR_TYPES:
        raise ValueError(f"{where}: exception outside {sorted(CALLER_ERROR_TYPES)}")
    if refusal["phase"] not in REFUSAL_PHASES:
        raise ValueError(f"{where}: phase outside {sorted(REFUSAL_PHASES)}")
    if category not in REFUSAL_CATEGORIES:
        raise ValueError(f"{where}: category outside {sorted(REFUSAL_CATEGORIES)}")
    if (refusal["phase"] == "fit") != (category == "no_mode_accepts_input"):
        raise ValueError(f"{where}: only no_mode_accepts_input is a fit refusal")
    if (status == "registered") == (category == "unregistered_algorithm"):
        raise ValueError(
            f"{where}: unregistered_algorithm is the category of every name "
            "kd does not register, and of no other"
        )
    modes = (
        category == "no_mode_accepts_input" or status == "registered_name_with_suffix"
    )
    if modes:
        valid = (
            isinstance(correction, dict)
            and set(correction) == {"modes"}
            and _identifiers(correction["modes"])
            and bool(correction["modes"])
        )
    elif category == "facade_owned_field":
        fields = correction.get("fields") if isinstance(correction, dict) else None
        valid = (
            isinstance(correction, dict)
            and set(correction) == {"fields"}
            and isinstance(fields, dict)
            and bool(fields)
            and all(
                isinstance(value, str)
                and field.isidentifier()
                and value.isidentifier()
                and field != value
                for field, value in fields.items()
            )
        )
    else:
        valid = correction is None
    if not valid:
        raise ValueError(f"{where}: correction has the wrong shape for {category}")


def check_refusal_vocabulary(
    document: Experience, vocabulary: dict[str, dict[str, Any]]
) -> None:
    unknown: dict[str, Any] = {"params": set(), "modes": set(), "facade": {}}
    for index, row in enumerate(document["refusals"]):
        name = row["algorithm"]["name"]
        if name is not None and name not in vocabulary:
            raise ValueError(f"refusal row {index}: kd registers no algorithm {name!r}")
        known = unknown if name is None else vocabulary[name]
        if not set(row["params"]["declared"]) <= set(known["params"]):
            raise ValueError(
                f"refusal row {index}: params.declared names a parameter the "
                "algorithm does not declare"
            )
        correction = row["refusal"]["correction"] or {}
        if not set(correction.get("modes", ())) <= set(known["modes"]):
            raise ValueError(
                f"refusal row {index}: correction names a mode the algorithm "
                "does not declare"
            )
        for field, facade in correction.get("fields", {}).items():
            if known["facade"].get(field) != facade:
                raise ValueError(
                    f"refusal row {index}: correction pairs a field with a facade "
                    "parameter the algorithm does not declare"
                )


def retrieve_refusals(
    document: Experience, task_kind: str, features: dict[str, Any], k: int = 8
) -> list[dict[str, Any]]:
    keys = REFUSAL_KEYS[task_kind]
    hits = [
        row
        for row in document["refusals"]
        if row["match"]["task_kind"] == task_kind
        and all(
            features[key] is not None and features[key] == row["match"][key]
            for key in keys
        )
    ]
    hits.sort(
        key=lambda row: (
            -row["support"]["n_problems"],
            -row["support"]["n_calls"],
            json.dumps(row, sort_keys=True),
        )
    )
    return hits[:k]


def readings(
    report: dict[str, Any],
    task_kind: str,
    vocabulary: Sequence[str],
    bins: dict[str, Sequence[float]],
) -> dict[str, Reading]:
    coarse = {"task_kind": task_kind, **priors.features_from_report(report)}
    if task_kind == "pde-evolution":
        coarse["n_fields"] = sum(
            field["name"] == report["lhs_field"]
            or (
                report["lhs_axis"] not in (field.get("constant_axes") or [])
                and report["lhs_axis"] not in (field.get("noise_limited_axes") or [])
            )
            for field in report["fields"]
        )
    return {
        key: (
            _periodic_reading(report)
            if key == PERIODIC_KEY
            else {"value": coarse[key], "status": "computed", "axis": None, "raw": None}
            if key in FEATURES[task_kind] and key not in BIN_KEYS
            else _diagnostic_reading(report, task_kind, key, bins)
        )
        for key in vocabulary
    }


def _spatial_axes(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [axis for axis in report["axes"] if axis["name"] != report["lhs_axis"]]


def _periodic_reading(report: dict[str, Any]) -> Reading:
    spatial = _spatial_axes(report)
    if report["lhs_order"] < 1 or not spatial:
        return {"value": None, "status": "undefined", "axis": None, "raw": None}
    if len(spatial) > 1:
        return {
            "value": False,
            "status": "computed",
            "axis": None,
            "raw": None,
            "axes": [axis["name"] for axis in spatial],
        }
    (axis,) = spatial
    return {
        "value": axis["periodic"],
        "status": "computed",
        "axis": axis["name"],
        "raw": None,
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
    if not spatial:
        return undefined
    field = next(
        entry for entry in report["fields"] if entry["name"] == report["lhs_field"]
    )
    diagnostic, axes = {
        "uxx_gap_bin": (field["second_derivative_gap"], spatial),
        "noise_bin": (field["noise_estimate"], spatial),
        "ut_gap_bin": (field["first_derivative_gap"], [lhs_axis]),
    }[key]
    return _reduce_diagnostic(diagnostic, axes, bins[key])


def _reduce_diagnostic(
    diagnostic: dict[str, float | None], axes: list[str], edges: Sequence[float]
) -> Reading:
    reading: Reading = {
        "value": None,
        "status": "not_computed",
        "axis": axes[0] if len(axes) == 1 else None,
        "raw": None,
    }
    if len(axes) > 1:
        reading.update({"axes": axes, "reduction": "max"})
    if any(axis not in diagnostic for axis in axes):
        return reading
    values = [diagnostic[axis] for axis in axes]
    if None in values:
        reading.update({"value": len(edges), "status": "infinite"})
    else:
        raw = max(value for value in values if value is not None)
        reading.update(
            {"value": bisect.bisect_right(edges, raw), "status": "computed", "raw": raw}
        )
    return reading


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


def pipeline_identity(pipeline: dict[str, Any]) -> dict[str, Any]:
    identity = {**pipeline, "input": pipeline.get("input", "raw")}
    if not identity.get("conditioning"):
        identity.pop("conditioning", None)
    return identity


def check_conditioning(conditioning: dict[str, Any]) -> None:
    unknown = set(conditioning) - {"sketch", "terms", "library", LIBRARY_RULE_KEY}
    if unknown:
        raise ValueError(f"conditioning carries unknown keys {sorted(unknown)}")
    json.dumps(conditioning, allow_nan=False)
    if "sketch" in conditioning and not isinstance(conditioning["sketch"], dict):
        raise ValueError("conditioning.sketch must contain the full Sketch payload")
    if LIBRARY_RULE_KEY not in conditioning:
        return
    rule = conditioning[LIBRARY_RULE_KEY]
    if {"terms", "library", "sketch"} & set(conditioning):
        raise ValueError(
            "conditioning names terms (a library or a sketch) beside a "
            "library_rule; a rule row carries the rule alone, written out for "
            "each input"
        )
    if not isinstance(rule, list) or not rule or not set(rule) <= set(LIBRARY_RULES):
        raise ValueError(
            f"conditioning.library_rule must be a non-empty list of "
            f"{sorted(LIBRARY_RULES)}, got {rule!r}"
        )


LIBRARY_RULE_KEY = "library_rule"

LIBRARY_RULES: dict[str, tuple[str, ...]] = {
    "poly2": ("{f}", "{f}_{a}", "{f}_{a}{a}", "n2({f})", "mul({f},{f}_{a})"),
    "poly3": (
        "{f}",
        "{f}_{a}",
        "{f}_{a}{a}",
        "n2({f})",
        "n3({f})",
        "mul({f},{f}_{a})",
    ),
    "deriv3": (
        "{f}",
        "{f}_{a}",
        "{f}_{a}{a}",
        "{f}_{a}{a}{a}",
        "n2({f})",
        "n3({f})",
        "mul({f},{f}_{a})",
    ),
    "product_derivative": ("diff_{a}(mul({f},{f}_{a}))",),
    "coordinate_ratio_if_axis_excludes_zero": ("mul({f}_{a},recip({a}))",),
}

_AXIS_RANGE_TOKEN = "coordinate_ratio_if_axis_excludes_zero"



MAX_TERM_SUBSETS: Final = 200


def library_terms(rule: Sequence[str], report: dict[str, Any]) -> list[str]:
    return [
        term.format(f=report["lhs_field"], a=axis["name"])
        for token in rule
        for index, axis in enumerate(_spatial_axes(report))
        if token != _AXIS_RANGE_TOKEN or axis["min"] > 0 or axis["max"] < 0
        for term in LIBRARY_RULES[token]
        if index == 0 or "{a}" in term
    ]


def instantiate(row: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    pipeline = row["pipeline"]
    conditioning = pipeline.get("conditioning", {})
    if LIBRARY_RULE_KEY not in conditioning:
        return row
    terms = library_terms(conditioning[LIBRARY_RULE_KEY], report)
    return {
        **row,
        "pipeline": {
            **pipeline,
            "conditioning": {
                **{
                    key: value
                    for key, value in conditioning.items()
                    if key != LIBRARY_RULE_KEY
                },
                "terms": terms,
            },
            "refinement": [
                call
                for step in pipeline["refinement"]
                for call in _subset_calls(step, terms)
            ],
        },
    }


def _with_terms(step: dict[str, Any], terms: list[str]) -> dict[str, Any]:
    if "subsets_up_to" not in step:
        return step
    written: dict[str, Any] = {}
    for key, value in step.items():
        if key == "subsets_up_to":
            written["terms"] = terms
        written[key] = value
    return written


def _subset_calls(step: dict[str, Any], terms: list[str]) -> list[dict[str, Any]]:
    if "subsets_up_to" not in step:
        return [step]
    largest = step["subsets_up_to"]
    count = sum(math.comb(len(terms), size) for size in range(1, largest + 1))
    if count <= MAX_TERM_SUBSETS:
        return [_with_terms(step, terms)]
    candidates = [
        list(subset)
        for size in range(1, largest + 1)
        for subset in itertools.combinations(terms, size)
    ]
    common = {
        key: value
        for key, value in step.items()
        if key not in {"tool", "subsets_up_to", "choose", "ratio"}
    }
    calls = [
        {
            "tool": "score_term_sets",
            **common,
            "term_sets": candidates[start: start + MAX_TERM_SUBSETS],
        }
        for start in range(0, len(candidates), MAX_TERM_SUBSETS)
    ]
    choice = step.get("choose", "")
    if step["tool"] == "select_support":
        choice = (
            f"Choose the fewest terms within {step.get('ratio', 2.0)} times the "
            "lowest finite NMSE, breaking size ties by lower NMSE. " + choice
        )
    calls[-1]["choose"] = (
        "Pool the scored entries from all batches above before choosing; use "
        "the global lowest NMSE, never a per-batch minimum. " + choice
    )
    return _batch_frame_calls(calls, common, terms)


def _batch_frame_calls(
    calls: list[dict[str, Any]], common: dict[str, Any], terms: list[str]
) -> list[dict[str, Any]]:
    if common.get("trim") != "by_order":
        return calls


    for call in calls:
        call["trim"] = (
            "diagnostics.frame.trim returned by the full-library frame call above"
        )
    calls.insert(
        0,
        {
            "tool": "score_term_sets",
            **common,
            "term_sets": [terms],
            "choose": (
                "Resolve the frame for the full candidate union. Copy this call's "
                "diagnostics.frame.trim into every following batch's trim argument. "
                "Exclude this full-library score from candidate selection; pool only "
                "the following subset batches."
            ),
        },
    )
    return calls


def retrieve(
    document: Experience,
    task_kind: str,
    features: dict[str, Any],
    k: int = 2,
    *,
    eligible: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for template in reversed(document["templates"][task_kind]):
        keys = set(template)
        hits = [
            row
            for row in document["rows"]
            if row["match"]["task_kind"] == task_kind
            and set(row["match"]) == keys
            and not (
                task_kind == "pde-evolution"
                and features["n_axes"] > 2
                and _uses_lowpass(row["pipeline"])
            )
            and all(
                features[key] is not None and features[key] == row["match"][key]
                for key in keys
            )
            and (eligible is None or eligible(row))
        ]
        for row in sorted(hits, key=_rank):
            if len(chosen) == k:
                return chosen
            if not chosen or (
                row["result"]["structure_rate"] > 0
                and all(_route(row) != _route(taken) for taken in chosen)
            ):
                chosen.append(row)
    return chosen


def _uses_lowpass(pipeline: dict[str, Any]) -> bool:
    return pipeline.get("input", "raw").split("+")[0] == "lowpass" or any(
        step.get("kind") == "lowpass"
        or (
            step.get("tool") == "smooth_input"
            and step.get("method", "lowpass") == "lowpass"
        )
        for step in pipeline.get("input_steps", [])
    )


def _route(row: dict[str, Any]) -> tuple[str, str, str]:
    pipeline = row["pipeline"]
    return (pipeline["algorithm"], pipeline["mode"], pipeline.get("input", "raw"))


def _rank(row: dict[str, Any]) -> tuple[Any, ...]:
    result = row["result"]
    n_problems = row["support"]["n_problems"]
    error = result["median_max_relative_error"]
    nrmse = result["median_heldout_nrmse"]
    evaluations = row["cost"]["median_evaluations"]
    return (
        -result["structure_rate"] * n_problems / (n_problems + 2),
        error is None,
        0.0 if error is None else error,
        nrmse is None,
        0.0 if nrmse is None else nrmse,
        evaluations is None,
        0.0 if evaluations is None else evaluations,
    )
