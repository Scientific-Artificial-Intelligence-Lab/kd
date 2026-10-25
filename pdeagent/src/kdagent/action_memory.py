
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import kd
from kd.search import mode_inputs
from kdagent import action_effects, action_experience, criteria
from kdagent.lineage import SegmentEntry, SegmentLedger
from kdagent.resume import eligible_iterations


def implementation_identity() -> dict[str, str]:
    root = Path(str(kd.__file__)).resolve().parent
    sources = sorted(root.rglob("*.py"))
    if not sources:
        raise ValueError("kd_source_sha256: installed Python sources unavailable")
    digest = hashlib.sha256(b"kd-action-memory-kd-source-v1\0")
    for path in sources:
        content = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "kd_source_sha256": digest.hexdigest(),
        "platform_eval_sha256": hashlib.sha256(
            Path(__file__).with_name("platform_eval.py").read_bytes()
        ).hexdigest(),
    }


def _schema(algorithm: str) -> dict[str, Any]:
    for schema in kd.instrument_schemas():
        if schema["algorithm"] == algorithm:
            return schema
    raise ValueError(f"unknown algorithm {algorithm!r}")


def _plain_numbers(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float)):
        return True
    return isinstance(value, (list, tuple)) and all(_plain_numbers(v) for v in value)


def _field_value(field: Mapping[str, Any], value: Any) -> tuple[bool, Any]:
    name, default = field["name"], field["default"]
    if name == "seed":
        return False, None
    if name in {"terms", "library"} or name.endswith("_seed"):
        if value != default and not (name in {"terms", "library"} and not value):
            raise ValueError(f"{name}: nondefault library or internal seed")
        return False, None
    if field["settable_from"] == "python":
        if value is not None:
            raise ValueError(f"{name}: Python-only scientific artifact")
        return False, None
    if value is not None and field["object_fields"] is not None:
        return True, _nested_fields(field["object_fields"], value, name)
    literals = field["literal_values"]
    if isinstance(value, str) and literals is not None:
        if value not in literals:
            raise ValueError(f"{name}: outside declared literal_values")
        return True, value
    if _plain_numbers(value):
        return True, deepcopy(value)
    if value == default:
        return False, None
    raise ValueError(f"{name}: nondefault freeform scientific setting")


def _nested_fields(
    fields: list[dict[str, Any]], values: Any, label: str
) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{label}: expected effective object")
    known = {field["name"] for field in fields}
    if set(values) != known:
        raise ValueError(f"{label}: incomplete or unknown effective fields")
    result = {}
    for field in fields:
        keep, value = _field_value(field, values[field["name"]])
        if keep:
            result[field["name"]] = value
    return result


def _effective_fields(
    schema: dict[str, Any], effective: Mapping[str, Any]
) -> dict[str, Any]:
    config = {}
    for field in schema["fields"]:
        name = field["name"]
        if name not in effective:
            raise ValueError(f"effective_config missing {name}")
        keep, value = _field_value(field, effective[name])
        if not keep:
            continue
        key = field["facade_param"] or name
        if key == "generations":
            continue
        if name == "use_autograd" and key == "derivatives":
            value = "autograd" if value else "finite_diff"
        config[key] = value
    return config


def _facade_controls(
    schema: dict[str, Any],
    effective: Mapping[str, Any],
    facade: Mapping[str, Any],
) -> dict[str, Any]:
    controls = {}
    for param in schema["facade_params"]:
        if param["config_field"] is not None:
            continue
        name = param["name"]
        if name not in facade:
            raise ValueError(f"facade_params missing {name}")
        if name in effective and effective[name] != facade[name]:
            raise ValueError(f"{name}: sealed and facade controls disagree")
        if name != "generations":
            controls[name] = deepcopy(facade[name])
    return controls


def _selected_modes(
    schema: dict[str, Any],
    config: Mapping[str, Any],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    modes = []
    for mode in schema["modes"]:
        verdict = kd.mode_accepts(mode, **mode_inputs(report))
        if not verdict.accepts or not mode["selectable_from_data"]:
            continue
        if any(config.get(key) != value for key, value in mode["selected_by"].items()):
            continue
        selector = mode["selector"]
        if selector is not None and (
            (config.get(selector["field"]) is not None) != selector["present"]
        ):
            continue
        if any(config.get(key) != value for key, value in verdict.params.items()):
            continue
        modes.append(mode)
    return modes


def _check_extra_settings(schema: dict[str, Any], effective: Mapping[str, Any]) -> None:
    known = {field["name"] for field in schema["fields"]}
    metadata = {"algorithm", "provider_kind", "small_term_rtol", "searched_vocabulary"}
    for key, value in effective.items():
        if key in known or key in metadata or key.endswith(("_fingerprint", "_sha256")):
            continue
        if key in {"sketch", "surrogate", "field_model"} and value is None:
            continue
        raise ValueError(f"effective_config {key}: unsupported setting or artifact")


def normalize_pipeline(
    algorithm: str,
    *,
    effective_config: Mapping[str, Any],
    facade_params: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    schema = _schema(algorithm)
    _check_extra_settings(schema, effective_config)
    config = _effective_fields(schema, effective_config)
    config.update(_facade_controls(schema, effective_config, facade_params))
    modes = _selected_modes(schema, config, report)
    if len(modes) != 1:
        raise ValueError(f"{algorithm}: effective config has no unique accepted mode")
    pipeline = {"algorithm": algorithm, "mode": modes[0]["name"], "config": config}
    action_experience.validate_pipeline(pipeline)
    kd.Model(algorithm=algorithm, **config)
    return pipeline


def project_state(
    *,
    features: Mapping[str, Any],
    entry: SegmentEntry | None,
    pipeline: Mapping[str, Any] | None,
    metric_status: str,
    last_action: str | None,
    conditions: Mapping[str, Any] | None,
    remaining_budget: Mapping[str, Any],
) -> dict[str, Any]:
    progress, stalled = "unknown", None
    if entry is not None and entry.summary is not None:
        facts = criteria.progress(entry)
        progress = "improved" if facts["improved"] else "stalled"
        stalled = min(2, facts["segments_without_improvement"])
    return {
        "features": {
            **deepcopy(dict(features)),
            "progress": progress,
            "segments_stalled_bin": stalled,
            "metric_status": metric_status,
            "last_action": last_action,
        },
        "pipeline": deepcopy(pipeline),
        "conditions": deepcopy(conditions),
        "remaining_budget": deepcopy(dict(remaining_budget)),
        "metric": deepcopy(action_experience.METRIC),
    }


def _resume_safe(
    schema: dict[str, Any], source: Mapping[str, Any], target: Mapping[str, Any]
) -> bool:
    if source["mode"] != target["mode"]:
        return False
    mapping = {
        field["facade_param"] or field["name"]: field["name"]
        for field in schema["fields"]
    }
    tiers = {field["name"]: field["resume_tier"] for field in schema["fields"]}
    changed = {
        key
        for key in source["config"].keys() | target["config"].keys()
        if source["config"].get(key) != target["config"].get(key)
    }
    return all(tiers.get(mapping.get(key, key)) == "resume_safe" for key in changed)


def validate_action_recipe(
    state: Mapping[str, Any],
    action: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    action_experience.validate_state_action(state, action)
    target = action["pipeline"]
    schema = _schema(target["algorithm"])
    modes = _selected_modes(schema, target["config"], report)
    if not any(mode["name"] == target["mode"] for mode in modes):
        raise ValueError("action pipeline mode does not accept this input")
    kd.Model(algorithm=target["algorithm"], **target["config"])
    generation = next(p for p in schema["facade_params"] if p["name"] == "generations")
    if generation["effect"] == "unused" and action["budget"]["generations"] is not None:
        raise ValueError("action budget generations is unused by the target")
    if action["execution"] == "resume":
        source_schema = _schema(state["pipeline"]["algorithm"])
        if source_schema["segmentation"]["archive"] != "progress":
            raise ValueError("action resume requires a progress archive")
        if not _resume_safe(source_schema, state["pipeline"], target):
            raise ValueError("action retune changes a non-resume-safe setting")


def _legal_action(
    action: Mapping[str, Any],
    state: Mapping[str, Any],
    moves: list[str],
    report: Mapping[str, Any],
) -> bool:
    try:
        validate_action_recipe(state, action, report)
    except (ValueError, TypeError):
        return False
    if action["execution"] == "resume":
        required = "continue" if action["kind"] == "continue" else "adjust"
        return required in moves
    return ("switch" if action["kind"] == "switch" else "fresh") in moves


def legal_actions(
    document: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    entry: SegmentEntry | None,
    ledger: SegmentLedger,
    workspace: Path,
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if entry is None or state["pipeline"] is None or state["conditions"] is None:
        return []
    schema = _schema(entry.instrument)
    iterations: tuple[int, ...] = ()
    if (
        schema["segmentation"]["archive"] == "progress"
        and entry.run_id in ledger.resumable_ids()
    ):
        iterations = eligible_iterations(workspace / entry.run_dir)
    moves = criteria.legal_moves(
        entry,
        schema,
        ledger=ledger,
        available_iterations=iterations,
        alternatives=[
            s["algorithm"]
            for s in kd.instrument_schemas()
            if s["algorithm"] != entry.instrument
        ],
    )
    actions = {}
    for row in document["rows"]:
        action = row["action"]
        if _legal_action(action, state, moves, report):
            key = json.dumps(action, sort_keys=True, separators=(",", ":"))
            actions[key] = deepcopy(action)
    return [actions[key] for key in sorted(actions)]


def selector_identity() -> str:
    return hashlib.sha256(Path(criteria.__file__).read_bytes()).hexdigest()


def validate_document(document: Mapping[str, Any]) -> None:
    format_ = document.get("format")
    if format_ == action_experience.FORMAT:
        action_experience.validate_action_experience(document)
    elif format_ == action_effects.FORMAT:
        action_effects.validate_action_effects(document)
    else:
        raise ValueError("unsupported action memory format")


def retrieve_document(
    document: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    legal_actions: Sequence[Mapping[str, Any]],
    k: int = 2,
) -> list[dict[str, Any]]:
    format_ = document.get("format")
    if format_ == action_experience.FORMAT:
        return action_experience.retrieve_action_experience(
            document, state, legal_actions=legal_actions, k=k
        )
    if format_ == action_effects.FORMAT:
        return action_effects.retrieve_action_effects(
            document, state, legal_actions=legal_actions, k=k
        )
    raise ValueError("unsupported action memory format")
