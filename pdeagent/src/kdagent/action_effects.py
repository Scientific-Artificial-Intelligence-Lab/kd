
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from kdagent import action_experience

FORMAT = "kd-action-experience-v2"
VERSION = 2
PROTOCOL = {
    "name": "kd-action-selected-recovery-v1",
    "selection_policy": "kdagent-default-selection-v1",
    "judge": "kdbench-j0-native-v1",
    "structure": "canonical-exact",
    "coefficient_rtol": 0.05,
    "nmse_reference_multiplier": 10.0,
    "absolute_nmse_diagnostic": 0.05,
    "derivative": "numpy.gradient",
    "edge_order": 1,
    "edge_trim": 2,
    "data_split": "input_train_grid",
    "zero_reference": "unknown",
}
TRANSITIONS = ("n_gained", "n_retained", "n_lost", "n_unrecovered", "n_unknown")


def _keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label}: missing or unknown keys")
    return value


def _count(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label}: expected nonnegative integer count")
    return value


def _digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label}: expected lowercase sha256")


def validate_effect_protocol(protocol: Mapping[str, Any]) -> None:
    _keys(protocol, {*PROTOCOL, "selector_sha256", "judge_sha256"}, "effect_protocol")
    for key, expected in PROTOCOL.items():
        value = protocol[key]
        if type(value) is bool or value != expected:
            raise ValueError(f"effect_protocol.{key}: unsupported protocol value")
        if isinstance(expected, int) and type(value) is not int:
            raise ValueError(f"effect_protocol.{key}: expected integer")
        if isinstance(expected, float) and (
            type(value) not in (int, float) or not math.isfinite(value)
        ):
            raise ValueError(f"effect_protocol.{key}: expected finite number")
    for key in ("selector_sha256", "judge_sha256"):
        _digest(protocol[key], f"effect_protocol.{key}")


def _proxy_view(document: Mapping[str, Any]) -> dict[str, Any]:
    for key, expected in (
        ("format", FORMAT),
        ("version", VERSION),
        ("builder", "build_action_effects.py"),
    ):
        if document.get(key) != expected or (
            key == "version" and type(document[key]) is not int
        ):
            raise ValueError(f"{key}: expected {expected!r}")
    for key in ("effect_source", "effect_protocol"):
        if key not in document:
            raise ValueError(f"missing {key}")
    source = _keys(document["effect_source"], {"kind", "digest"}, "effect_source")
    if source["kind"] != "kd-action-effect-labels-v1":
        raise ValueError("effect_source.kind: unsupported source")
    _digest(source["digest"], "effect_source.digest")
    validate_effect_protocol(document["effect_protocol"])
    base = deepcopy(dict(document))
    del base["effect_source"], base["effect_protocol"]
    base.update(
        format=action_experience.FORMAT,
        version=action_experience.VERSION,
        builder="build_action_experience.py",
    )
    if not isinstance(base.get("rows"), list):
        raise ValueError("rows: expected list")
    for row in base["rows"]:
        if not isinstance(row, dict) or "effects" not in row:
            raise ValueError("row missing effects")
        del row["effects"]
    return base


def _transition(value: Any, attempts: int, groups: int, label: str) -> None:
    counts = _keys(value, {*TRANSITIONS, "n_judged_groups"}, label)
    for key, count in counts.items():
        _count(count, f"{label}.{key}")
    if sum(counts[key] for key in TRANSITIONS) != attempts:
        raise ValueError(f"{label}: transition counts must sum to attempts")
    known = attempts - counts["n_unknown"]
    judged = counts["n_judged_groups"]
    if (judged == 0) != (known == 0) or judged > min(known, groups):
        raise ValueError(f"{label}.n_judged_groups: inconsistent known groups")


def _paired(value: Any, comparison: Any) -> None:
    if value is None and comparison is None:
        return
    if value is None or comparison is None:
        raise ValueError("effects.paired and comparison must both be present")
    counts = _keys(
        value,
        {"n_pairs", "n_groups", "n_wins", "n_ties", "n_losses", "n_unknown"},
        "effects.paired",
    )
    for key, count in counts.items():
        _count(count, f"effects.paired.{key}")
    for key in ("n_pairs", "n_groups"):
        if counts[key] != comparison[key]:
            raise ValueError(f"effects.paired.{key}: differs from comparison")
    if (
        sum(counts[key] for key in ("n_wins", "n_ties", "n_losses", "n_unknown"))
        != counts["n_pairs"]
    ):
        raise ValueError("effects.paired: counts must sum to pairs")


def _effects(row: Mapping[str, Any]) -> None:
    effects = _keys(
        row["effects"], {"selection", "structure", "candidates", "paired"}, "effects"
    )
    attempts, groups = row["result"]["n_attempts"], row["support"]["n_groups"]
    for key in ("selection", "structure"):
        _transition(effects[key], attempts, groups, f"effects.{key}")
    candidates = _keys(
        effects["candidates"],
        {"n_new_recovery", "n_new_structure", "n_unknown"},
        "effects.candidates",
    )
    for key, count in candidates.items():
        _count(count, f"effects.candidates.{key}")
    known = attempts - candidates["n_unknown"]
    if (
        known < 0
        or max(candidates["n_new_recovery"], candidates["n_new_structure"]) > known
    ):
        raise ValueError("effects.candidates: inconsistent known counts")
    _paired(effects["paired"], row["comparison"])


def validate_action_effects(document: Mapping[str, Any]) -> None:
    action_experience.validate_action_experience(_proxy_view(document))
    for row in document["rows"]:
        _effects(row)


def load_action_effects(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    validate_action_effects(document)
    return document


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    effects, attempts = row["effects"]["selection"], row["result"]["n_attempts"]
    wall = row["cost"]["median_wall_seconds"]
    return (
        -(attempts - effects["n_unknown"]) / attempts,
        -(effects["n_gained"] - effects["n_lost"]) / attempts,
        effects["n_lost"] / attempts,
        -row["support"]["n_groups"],
        row["result"]["n_failed"] / attempts,
        row["result"]["n_timed_out"] / attempts,
        wall is None,
        0.0 if wall is None else wall,
        _canonical(row["action"]),
        _canonical(row),
    )


def retrieve_action_effects(
    document: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    legal_actions: Sequence[Mapping[str, Any]],
    k: int = 2,
) -> list[dict[str, Any]]:
    _count(k, "k")
    validate_action_effects(document)
    action_experience.validate_query_state(state)
    action_experience.validate_state_features(state["features"])
    if k == 0 or state["pipeline"] is None or state["conditions"] is None:
        return []
    matches = action_experience.matching_action_rows(
        document, state, legal_actions=legal_actions
    )
    matches = [
        item
        for item in matches
        if item[2]["effects"]["selection"]["n_judged_groups"] >= document["min_support"]
    ]
    selected, actions = [], set()
    for _, index, row in sorted(matches, key=lambda item: (-item[0], _rank(item[2]))):
        key = _canonical(row["action"])
        if key not in actions:
            selected.append({"row_index": index, "row": deepcopy(row)})
            actions.add(key)
            if len(selected) == k:
                break
    return selected
