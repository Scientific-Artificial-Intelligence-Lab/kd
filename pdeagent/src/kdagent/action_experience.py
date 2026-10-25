
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

FORMAT = "kd-action-experience-v1"
VERSION = 1
RULES_VERSION = "kd-action-memory-rules-v1"
BIN_KEYS = ("uxx_gap_bin", "noise_bin", "ut_gap_bin")
_COARSE = (
    "task_kind",
    "topology",
    "lhs_order",
    "n_axes",
    "n_fields",
    "points_bucket",
    "has_nan",
)
_DYNAMIC = ("progress", "segments_stalled_bin", "metric_status", "last_action")
FEATURES: dict[str, tuple[str, ...]] = {
    "pde-evolution": (*_COARSE, *BIN_KEYS, *_DYNAMIC),
    "pde-steady": (*_COARSE, *_DYNAMIC),
    "tabular-regression": (
        "task_kind",
        "topology",
        "n_fields",
        "points_bucket",
        *_DYNAMIC,
    ),
    "ode": ("task_kind", "topology", "n_fields", "points_bucket", *_DYNAMIC),
}
METRIC = {
    "name": "platform_refit_nmse",
    "frame": "kd-evaluate-terms-full-grid-v1",
    "data_split": "input_full_grid",
    "coefficient_source": "ols_refit",
    "direction": "min",
    "improvement_epsilon": 1e-9,
}
_DOCUMENT_KEYS = {
    "format",
    "version",
    "built_at",
    "builder",
    "source",
    "instrument_schema_sha256",
    "implementation_identity",
    "rules_version",
    "features_vocabulary",
    "templates",
    "bins",
    "min_support",
    "excluded_groups",
    "rows",
}
_ROW_KEYS = {
    "match",
    "source_pipeline",
    "action",
    "conditions",
    "metric",
    "result",
    "cost",
    "support",
    "comparison",
}
_FORBIDDEN_CONFIG = {
    "terms",
    "library",
    "sketch",
    "cap",
    "max_evaluations",
    "max_search_seconds",
    "generations",
    "seed",
    "surrogate",
    "surrogates",
    "field_model",
}
_IDENTITY_PARTS = (
    "fingerprint",
    "run_id",
    "dataset",
    "problem",
    "checkpoint",
    "api_key",
    "endpoint",
    "base_url",
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ACTION_KINDS = {"continue", "restart", "retune", "switch"}


def load_action_experience(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    validate_action_experience(document)
    return document


def validate_action_experience(document: Mapping[str, Any]) -> None:
    _keys(document, _DOCUMENT_KEYS, "document")
    _header(document)
    _index(document)
    seen: set[str] = set()
    for index, row in enumerate(_list(document["rows"], "rows")):
        _row(row, document, f"rows[{index}]")
        canonical = _canonical(row)
        if canonical in seen:
            raise ValueError(f"rows[{index}]: duplicate row")
        seen.add(canonical)


def validate_pipeline(pipeline: Mapping[str, Any]) -> None:
    _pipeline(pipeline, "pipeline")


def validate_state_action(state: Mapping[str, Any], action: Mapping[str, Any]) -> None:
    _keys(
        state,
        {"features", "pipeline", "conditions", "remaining_budget", "metric"},
        "state",
    )
    validate_state_features(state["features"])
    _pipeline(state["pipeline"], "state.pipeline")
    _conditions(state["conditions"], "state.conditions")
    _metric(state["metric"], "state.metric")
    _remaining_budget(state["remaining_budget"])
    _action(action, state["pipeline"], "action")
    if not _affordable(action, state["remaining_budget"]):
        raise ValueError("action.budget: exceeds remaining_budget")


def validate_query_state(state: Mapping[str, Any]) -> None:
    _keys(
        state,
        {"features", "pipeline", "conditions", "remaining_budget", "metric"},
        "state",
    )
    _remaining_budget(state["remaining_budget"])


def matching_action_rows(
    document: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    legal_actions: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int, Mapping[str, Any]]]:
    features = state["features"]
    kind = features["task_kind"]
    if kind not in FEATURES:
        raise ValueError(f"state task_kind {kind!r}")
    eligible = [
        (index, row)
        for index, row in enumerate(document["rows"])
        if _eligible(row, state, legal_actions)
    ]
    return [
        (depth, index, row)
        for depth, template in enumerate(document["templates"][kind])
        for index, row in eligible
        if set(row["match"]) == set(template)
        and all(
            features.get(key) is not None and features[key] == value
            for key, value in row["match"].items()
        )
    ]


def retrieve_action_experience(
    document: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    legal_actions: Sequence[Mapping[str, Any]],
    k: int = 2,
) -> list[dict[str, Any]]:
    _integer(k, "k")
    validate_action_experience(document)
    validate_query_state(state)
    if k == 0 or state["pipeline"] is None or state["conditions"] is None:
        return []
    matches = matching_action_rows(document, state, legal_actions=legal_actions)
    selected: list[dict[str, Any]] = []
    actions: set[str] = set()
    for _, index, row in sorted(matches, key=lambda item: (-item[0], _rank(item[2]))):
        action = _canonical(row["action"])
        if action not in actions:
            selected.append({"row_index": index, "row": deepcopy(row)})
            actions.add(action)
            if len(selected) == k:
                break
    return selected


def _keys(
    value: Any, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: expected object")
    missing, unknown = expected - value.keys(), value.keys() - expected
    if missing or unknown:
        raise ValueError(
            f"{label}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}"
        )
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label}: expected list")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: expected integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label}: expected finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label}: expected number >= {minimum}")
    return float(value)


def _digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label}: expected lowercase sha256 digest")


def _identifier(value: Any, label: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label}: expected identifier")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _header(document: Mapping[str, Any]) -> None:
    for key, expected in (
        ("format", FORMAT),
        ("builder", "build_action_experience.py"),
        ("rules_version", RULES_VERSION),
    ):
        if document[key] != expected:
            raise ValueError(f"{key}: expected {expected!r}")
    if _integer(document["version"], "version") != VERSION:
        raise ValueError(f"version: expected {VERSION}")
    timestamp = document["built_at"]
    if not isinstance(timestamp, str) or "T" not in timestamp:
        raise ValueError("built_at: expected ISO timestamp")
    try:
        datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError("built_at: expected ISO timestamp") from error
    source = _keys(document["source"], {"kind", "digest"}, "source")
    if source["kind"] != "kd-action-observations-v1":
        raise ValueError("source.kind: expected kd-action-observations-v1")
    _digest(source["digest"], "source.digest")
    _digest(document["instrument_schema_sha256"], "instrument_schema_sha256")
    identity = _keys(
        document["implementation_identity"],
        {"kd_source_sha256", "platform_eval_sha256"},
        "implementation_identity",
    )
    for key, value in identity.items():
        _digest(value, f"implementation_identity.{key}")
    _integer(document["min_support"], "min_support", 2)
    groups = _list(document["excluded_groups"], "excluded_groups")
    if any(not isinstance(group, str) or not group for group in groups):
        raise ValueError("excluded_groups: expected nonempty group names")
    if groups != sorted(set(groups)):
        raise ValueError("excluded_groups: expected sorted unique groups")


def _index(document: Mapping[str, Any]) -> None:
    vocabulary = _keys(
        document["features_vocabulary"], set(FEATURES), "features_vocabulary"
    )
    templates = _keys(document["templates"], set(FEATURES), "templates")
    for kind, allowed in FEATURES.items():
        if vocabulary[kind] != list(allowed):
            raise ValueError(f"features_vocabulary.{kind}: expected closed vocabulary")
        _templates(templates[kind], allowed, kind)
    bins = _keys(document["bins"], set(BIN_KEYS), "bins")
    for key, edges in bins.items():
        edges = _list(edges, f"bins.{key}")
        for value in edges:
            _number(value, f"bins.{key}")
        if any(left >= right for left, right in zip(edges, edges[1:], strict=False)):
            raise ValueError(f"bins.{key}: edges must be strictly increasing")


def _templates(value: Any, vocabulary: tuple[str, ...], kind: str) -> None:
    seen: set[frozenset[str]] = set()
    previous_length = 0
    for template in _list(value, f"templates.{kind}"):
        keys = _list(template, f"templates.{kind}")
        if any(not isinstance(key, str) or key not in vocabulary for key in keys):
            raise ValueError(f"templates.{kind}: unknown keys")
        frozen = frozenset(keys)
        if len(frozen) != len(keys) or frozen in seen:
            raise ValueError(f"templates.{kind}: duplicate keys or template")
        if not {"task_kind", "progress"} <= frozen:
            raise ValueError(f"templates.{kind}: missing task_kind/progress")
        if len(keys) < previous_length:
            raise ValueError(f"templates.{kind}: expected coarse to fine order")
        previous_length = len(keys)
        seen.add(frozen)


def _row(value: Any, document: Mapping[str, Any], label: str) -> None:
    row = _keys(value, _ROW_KEYS, label)
    _match(row["match"], document, label)
    _pipeline(row["source_pipeline"], f"{label}.source_pipeline")
    _action(row["action"], row["source_pipeline"], f"{label}.action")
    _conditions(row["conditions"], f"{label}.conditions")
    _metric(row["metric"], f"{label}.metric")
    attempts = _result(row["result"], f"{label}.result")
    if (
        row["match"].get("metric_status") in {"unavailable", "failed"}
        and row["result"]["n_comparable"]
    ):
        raise ValueError(f"{label}.result: unavailable metric cannot be comparable")
    _cost(row["cost"], attempts, f"{label}.cost")
    support = _keys(row["support"], {"n_problems", "n_groups"}, f"{label}.support")
    problems = _integer(support["n_problems"], f"{label}.support.n_problems", 1)
    groups = _integer(
        support["n_groups"], f"{label}.support.n_groups", document["min_support"]
    )
    if not groups <= problems <= attempts:
        raise ValueError(f"{label}.support: inconsistent counts")
    if row["comparison"] is not None:
        _comparison(row["comparison"], row, document["min_support"], label)


def _match(value: Any, document: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.match: expected object")
    kind = value.get("task_kind")
    if not isinstance(kind, str) or kind not in FEATURES:
        raise ValueError(f"{label}.match.task_kind: unknown task_kind")
    allowed = set(FEATURES[kind])
    if value.keys() - allowed:
        raise ValueError(
            f"{label}.match: unknown keys {sorted(value.keys() - allowed)}"
        )
    if set(value) not in [set(template) for template in document["templates"][kind]]:
        raise ValueError(f"{label}.match: keys do not match a declared template")
    for key, feature in value.items():
        _feature(key, feature, document["bins"], f"{label}.match.{key}")


def validate_state_features(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("state.features: expected object")
    kind = value.get("task_kind")
    if not isinstance(kind, str) or kind not in FEATURES:
        raise ValueError("state.features.task_kind: unknown task_kind")
    _keys(value, set(FEATURES[kind]), "state.features")
    for key, feature in value.items():
        if feature is None and key not in {"task_kind", "progress", "metric_status"}:
            continue
        _feature(key, feature, None, f"state.features.{key}")


def _feature(key: str, value: Any, bins: Mapping[str, Any] | None, label: str) -> None:
    choices: dict[str, set[Any]] = {
        "task_kind": set(FEATURES),
        "topology": {"grid", "scattered", "tabular"},
        "progress": {"improved", "stalled", "unknown"},
        "metric_status": {"measured", "unavailable", "failed"},
        "last_action": _ACTION_KINDS,
    }
    if value is None:
        raise ValueError(f"{label}: null feature cannot match")
    if key in choices:
        if not isinstance(value, str) or value not in choices[key]:
            raise ValueError(f"{label}: unknown feature value")
    elif key == "has_nan":
        if type(value) is not bool:
            raise ValueError(f"{label}: expected bool")
    else:
        integer = _integer(value, label, 1 if key == "n_fields" else 0)
        maximum = (
            2
            if key == "segments_stalled_bin"
            else len(bins[key])
            if bins is not None and key in BIN_KEYS
            else None
        )
        if maximum is not None and integer > maximum:
            raise ValueError(f"{label}: invalid bin")


def _pipeline(value: Any, label: str) -> None:
    pipeline = _keys(value, {"algorithm", "mode", "config"}, label)
    _identifier(pipeline["algorithm"], f"{label}.algorithm")
    _identifier(pipeline["mode"], f"{label}.mode")
    _config(pipeline["config"], f"{label}.config")


def _config(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: expected config object")
    for key, setting in value.items():
        _identifier(key, f"{label}.{key}")
        if (
            key in _FORBIDDEN_CONFIG
            or key.endswith(("_seed", "_path", "_file", "_dir"))
            or any(part in key for part in _IDENTITY_PARTS)
        ):
            raise ValueError(f"{label}.{key}: forbidden config key")
        _config_value(setting, f"{label}.{key}")


def _config_value(value: Any, label: str) -> None:
    if value is None or type(value) is bool:
        return
    if type(value) in (int, float):
        _number(value, label)
    elif isinstance(value, str):


        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", value) is None:
            raise ValueError(f"{label}: config string contains path or expression")
    elif isinstance(value, Mapping):
        _config(value, label)
    elif isinstance(value, list):
        for item in value:
            if type(item) is not bool:
                _number(item, label)
    else:
        raise ValueError(f"{label}: unsupported config value")


def _action(value: Any, source: Mapping[str, Any], label: str) -> None:
    action = _keys(
        value, {"kind", "execution", "pipeline", "seed_policy", "budget"}, label
    )
    _pipeline(action["pipeline"], f"{label}.pipeline")
    kind, execution, seed = action["kind"], action["execution"], action["seed_policy"]
    if not isinstance(kind, str) or kind not in _ACTION_KINDS:
        raise ValueError(f"{label}.kind: unknown action")
    if execution not in ("resume", "fresh"):
        raise ValueError(f"{label}.execution: expected resume/fresh")
    if (execution == "resume" and seed != "inherit") or (
        execution == "fresh" and seed not in ("same", "new")
    ):
        raise ValueError(f"{label}.seed_policy: disagrees with execution")
    target = action["pipeline"]
    same = target == source
    same_algorithm = target["algorithm"] == source["algorithm"]
    if kind == "continue" and (execution != "resume" or not same):
        raise ValueError(f"{label}: continue requires same pipeline and resume")
    if kind == "restart" and (execution != "fresh" or not same):
        raise ValueError(f"{label}: restart requires same pipeline and fresh")
    if kind == "retune" and (not same_algorithm or same):
        raise ValueError(
            f"{label}: retune requires changed config/mode on same algorithm"
        )
    if kind == "retune" and execution == "resume" and target["mode"] != source["mode"]:
        raise ValueError(f"{label}: resume retune requires same mode")
    if kind == "switch" and (execution != "fresh" or same_algorithm):
        raise ValueError(f"{label}: switch requires fresh different algorithm")
    _budget(action["budget"], f"{label}.budget")


def _budget(value: Any, label: str) -> None:
    budget = _keys(value, {"generations", "seconds", "evaluations"}, label)
    for key in ("generations", "evaluations"):
        if budget[key] is not None:
            _integer(budget[key], f"{label}.{key}", 1)
    if (
        budget["seconds"] is not None
        and _number(budget["seconds"], f"{label}.seconds") <= 0
    ):
        raise ValueError(f"{label}.seconds: expected positive cap")
    if budget["seconds"] is None and budget["evaluations"] is None:
        raise ValueError(f"{label}: seconds or evaluations cap required")


def _conditions(value: Any, label: str) -> None:
    conditions = _keys(value, {"device", "blas_threads", "python_hash_seed"}, label)
    if conditions["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError(f"{label}.device: unknown execution policy")
    _integer(conditions["blas_threads"], f"{label}.blas_threads", 1)
    seed = conditions["python_hash_seed"]
    if seed is not None and type(seed) is not int:
        raise ValueError(f"{label}.python_hash_seed: expected integer or null")


def _metric(value: Any, label: str) -> None:
    metric = _keys(value, set(METRIC), label)
    _number(metric["improvement_epsilon"], f"{label}.improvement_epsilon")
    if metric != METRIC:
        raise ValueError(f"{label}: unsupported metric/frame/split/coefficient_source")


def _result(value: Any, label: str) -> int:
    counts = {
        "n_attempts",
        "n_comparable",
        "n_improved",
        "n_worsened",
        "n_failed",
        "n_timed_out",
    }
    result = _keys(value, counts | {"median_metric_improvement"}, label)
    attempts = _integer(result["n_attempts"], f"{label}.n_attempts", 1)
    for key in counts - {"n_attempts"}:
        if _integer(result[key], f"{label}.{key}") > attempts:
            raise ValueError(f"{label}.{key}: count exceeds n_attempts")
    if result["n_improved"] + result["n_worsened"] > result["n_comparable"]:
        raise ValueError(f"{label}: improved/worsened count exceeds n_comparable")
    _median(
        result["median_metric_improvement"],
        result["n_comparable"],
        f"{label}.median_metric_improvement",
    )
    return attempts


def _median(
    value: Any, count: int, label: str, *, minimum: float | None = None
) -> None:
    if (value is None) != (count == 0):
        raise ValueError(f"{label}: null iff measurement count is zero")
    if value is not None:
        _number(value, label, minimum=minimum)


def _cost(value: Any, attempts: int, label: str) -> None:
    cost = _keys(
        value,
        {
            "median_wall_seconds",
            "n_wall_seconds",
            "median_evaluations",
            "n_evaluations",
        },
        label,
    )
    for axis in ("wall_seconds", "evaluations"):
        count = _integer(cost[f"n_{axis}"], f"{label}.n_{axis}")
        if count > attempts:
            raise ValueError(f"{label}.n_{axis}: count exceeds n_attempts")
        _median(cost[f"median_{axis}"], count, f"{label}.median_{axis}", minimum=0)


def _comparison(value: Any, row: Mapping[str, Any], minimum: int, label: str) -> None:
    label = f"{label}.comparison"
    counts = {"n_pairs", "n_groups", "n_comparable", "n_wins", "n_ties", "n_losses"}
    comparison = _keys(
        value, counts | {"against_action", "median_metric_advantage"}, label
    )
    _action(
        comparison["against_action"], row["source_pipeline"], f"{label}.against_action"
    )
    if comparison["against_action"] == row["action"]:
        raise ValueError(f"{label}: against_action must be distinct")
    for axis in ("seconds", "evaluations"):
        if (
            comparison["against_action"]["budget"][axis]
            != row["action"]["budget"][axis]
        ):
            raise ValueError(f"{label}: paired budget {axis} must be equal")
    for key in counts:
        _integer(comparison[key], f"{label}.{key}")
    pairs, groups = comparison["n_pairs"], comparison["n_groups"]
    if not minimum <= groups <= pairs <= row["result"]["n_attempts"]:
        raise ValueError(f"{label}: invalid pair/support counts")
    if groups > row["support"]["n_groups"]:
        raise ValueError(f"{label}.n_groups: exceeds observation support")
    comparable = comparison["n_comparable"]
    if (
        comparable > pairs
        or sum(comparison[key] for key in ("n_wins", "n_ties", "n_losses"))
        != comparable
    ):
        raise ValueError(f"{label}: invalid comparable/outcome counts")
    _median(
        comparison["median_metric_advantage"],
        comparable,
        f"{label}.median_metric_advantage",
    )


def _remaining_budget(value: Any) -> None:
    remaining = _keys(value, {"seconds", "evaluations"}, "state.remaining_budget")
    if remaining["seconds"] is not None:
        _number(remaining["seconds"], "state.remaining_budget.seconds", minimum=0)
    if remaining["evaluations"] is not None:
        _integer(remaining["evaluations"], "state.remaining_budget.evaluations")


def _eligible(
    row: Mapping[str, Any], state: Mapping[str, Any], legal: Sequence[Mapping[str, Any]]
) -> bool:
    if (
        row["source_pipeline"] != state["pipeline"]
        or row["conditions"] != state["conditions"]
        or row["metric"] != state["metric"]
        or row["action"] not in legal
    ):
        return False
    return _affordable(row["action"], state["remaining_budget"])


def _affordable(action: Mapping[str, Any], remaining_budget: Mapping[str, Any]) -> bool:
    for axis, remaining in remaining_budget.items():
        allowance = action["budget"][axis]
        if remaining is not None and (allowance is None or allowance > remaining):
            return False
    return True


def _rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    wall = row["cost"]["median_wall_seconds"]
    return (
        -row["support"]["n_groups"],
        -row["result"]["n_improved"] / row["result"]["n_attempts"],
        wall is None,
        0.0 if wall is None else wall,
        _canonical(row["action"]),
        _canonical(row),
    )
