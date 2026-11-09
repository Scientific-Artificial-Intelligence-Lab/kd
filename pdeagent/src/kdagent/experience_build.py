
from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import kd
from kd.search import mode_inputs

from kdagent import experience, experience_applicability
from kdagent.experience_sources import SourceRefusedError, read_export, read_graded_runs

logger = logging.getLogger("build_experience")



BINS: dict[str, Sequence[float]] = {
    "uxx_gap_bin": [0.05, 0.5, 2.0],
    "noise_bin": [3e-4, 3e-3, 3e-2],
    "ut_gap_bin": [0.02, 0.2, 1.0],
}

TEMPLATES: dict[str, list[list[str]]] = {
    kind: [list(keys)] for kind, keys in experience.FEATURES.items()
}
TEMPLATES["pde-evolution"] = [
    list(experience.FEATURES["pde-evolution"][: 7 + count])
    for count in range(len(experience.BIN_KEYS) + 1)
]

Observation = dict[str, Any]


def emit(
    observations: Sequence[Observation],
    *,
    refusals: Sequence[dict[str, Any]],
    templates: dict[str, list[list[str]]],
    bins: dict[str, Sequence[float]],
    min_support: int,
    excluded_groups: Sequence[str],
    source: dict[str, str],
    across_versions: bool = False,
    source_rules_version: str | None = None,
) -> dict[str, Any]:
    withheld = sorted(set(excluded_groups))
    _refuse([*observations, *refusals], withheld)
    rows, _log = carve(
        observations, templates=templates, bins=bins, min_support=min_support
    )
    document: dict[str, Any] = {
        "format": experience.FORMAT,
        "version": experience.VERSION,

        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "builder": Path(__file__).name,
        "source": dict(source),
        "rules_version": _rules_version(observations)
        if observations or source_rules_version is None
        else source_rules_version,
        "examinee_versions": _examinee_versions(observations, across_versions),
        "instrument_schema_sha256": kd.instrument_schema_digest(),
        "features_vocabulary": {
            kind: _vocabulary(kind, templates[kind]) for kind in experience.FEATURES
        },
        "bins": {key: list(bins[key]) for key in experience.BIN_KEYS},
        "templates": templates,
        "min_support": min_support,
    }
    if withheld:
        document["excluded_groups"] = withheld
    document["rows"] = rows
    document["refusals"], _log = carve_refusals(
        refusals, bins=bins, min_support=min_support
    )
    return document


def _refuse(observations: Sequence[Observation], withheld: Sequence[str]) -> None:
    observed = sorted(set(withheld) & {obs["group"] for obs in observations})
    if observed:
        raise ValueError(
            f"the observations carry groups {observed}, which this build "
            "declares withheld: an evaluation document is read while those "
            "groups are measured, so it may not be built over their runs"
        )


class ObservationsRefusedError(ValueError):

    def __init__(self, refused: list[dict[str, Any]]) -> None:
        self.refused = refused
        super().__init__(
            "observations refused; rebuild from valid runs: " + json.dumps(refused)
        )


def validate_observations(observations: Sequence[Observation]) -> None:
    digest = kd.instrument_schema_digest()
    schemas = {schema["algorithm"]: schema for schema in kd.instrument_schemas()}
    refused = []
    for index, obs in enumerate(observations):
        try:
            if obs["instrument_schema_sha256"] != digest:
                raise ValueError(
                    "recorded under a different kd declaration set; rebuild from new runs"
                )
            if obs["task_kind"] not in experience.FEATURES:
                raise ValueError(f"unknown task_kind {obs['task_kind']!r}")
            _validate_pipeline(obs, schemas)
        except (ValueError, TypeError) as error:
            refused.append(
                {
                    "index": index,
                    "problem_key": obs["problem_key"],
                    "reason": str(error),
                }
            )
    if refused:
        raise ObservationsRefusedError(refused)


def _validate_pipeline(obs: Observation, schemas: dict[str, dict[str, Any]]) -> None:
    pipeline = obs["pipeline"]
    algorithm, config = pipeline["algorithm"], pipeline["config"]
    forbidden = sorted(experience.FORBIDDEN_CONFIG_KEYS & config.keys())
    if forbidden:
        raise ValueError(
            f"config carries forbidden keys {forbidden}; remove the library or budget"
        )
    conditioning = pipeline.get("conditioning", {})
    experience.check_conditioning(conditioning)
    if experience.LIBRARY_RULE_KEY in conditioning:
        conditioning = experience.instantiate({"pipeline": pipeline}, obs["report"])[
            "pipeline"
        ]["conditioning"]
    config = {
        **config,
        **{key: value for key, value in conditioning.items() if key != "sketch"},
    }
    if "sketch" in conditioning:
        kd.sketch_from_dict(conditioning["sketch"])
    if algorithm not in schemas:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; use an installed kd algorithm"
        )
    if "input" in pipeline and pipeline["input"] not in experience.INPUT_KINDS:
        raise ValueError(
            f"pipeline input {pipeline['input']!r} is outside "
            f"{sorted(experience.INPUT_KINDS)}"
        )
    schema = schemas[algorithm]
    modes = {mode["name"]: mode for mode in schema["modes"]}
    if pipeline["mode"] not in modes:
        raise ValueError(f"unknown mode {pipeline['mode']!r} for {algorithm}")
    mode = modes[pipeline["mode"]]
    verdict = kd.mode_accepts(
        mode,
        **mode_inputs(
            obs["search_report"] if "search_report" in obs else obs["report"]
        ),
    )
    if not verdict.accepts:
        raise ValueError(f"{algorithm}/{mode['name']}: {verdict.reason}")
    missing = sorted(verdict.params.keys() - config.keys())
    if missing:
        raise ValueError(f"{algorithm}/{mode['name']} requires config keys {missing}")
    if any(config[key] != value for key, value in mode["selected_by"].items()):
        raise ValueError(
            f"config disagrees with selected_by for {algorithm}/{mode['name']}"
        )
    for other in schema["modes"]:
        if (
            other["name"] != mode["name"]
            and other["selected_by"]
            and all(
                key in config and config[key] == value
                for key, value in other["selected_by"].items()
            )
        ):
            raise ValueError(
                f"config contains selected_by for another mode: {other['name']}"
            )
    _validate_config(schema, config)
    kd.Model(algorithm=algorithm, **config)


def _validate_config(schema: dict[str, Any], config: dict[str, Any]) -> None:
    algorithm = schema["algorithm"]
    allowed = {param["name"] for param in schema["facade_params"]} | {
        field["name"] for field in schema["fields"] if field["settable_from"] == "json"
    }
    if unknown := sorted(config.keys() - allowed):
        raise ValueError(
            f"config keys {unknown} are not JSON facade parameters for {algorithm}"
        )
    for field in schema["fields"]:
        if (
            field["name"].endswith("_seed")
            and field["name"] in config
            and config[field["name"]] != field["default"]
        ):
            raise ValueError(f"{field['name']} must use its default {field['default']}")


def _rules_version(observations: Sequence[Observation]) -> str:
    versions: list[str] = sorted({obs["rules_version"] for obs in observations})
    if len(versions) != 1:
        raise ValueError(
            f"the observations carry rules_version {versions}; a document is "
            "scored under exactly one, or its rows rank against each other on "
            "different rules"
        )
    return versions[0]


def _examinee_versions(
    observations: Sequence[Observation], across_versions: bool = False
) -> dict[str, str]:
    seen: dict[str, set[str]] = {}
    for obs in observations:
        seen.setdefault(obs["pipeline"]["algorithm"], set()).add(
            obs["examinee_version"]
        )
    mixed = {name: sorted(found) for name, found in seen.items() if len(found) > 1}
    if mixed and not across_versions:
        raise ValueError(
            f"algorithms {mixed} were observed at more than one examinee "
            "version: an adapter change changes the numbers, so runs across "
            "versions are not aggregated"
        )
    if mixed:
        logger.warning("pooling algorithms across examinee versions: %s", mixed)
    return {name: "+".join(sorted(found)) for name, found in sorted(seen.items())}


def _typed_projection(
    instrument: str, typed: Sequence[str], schemas: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], str | None, Any]:
    prefix, slash, _suffix = instrument.partition("/")
    if instrument in schemas:
        name, status = instrument, "registered"
    elif slash and prefix in schemas:
        name, status = prefix, "registered_name_with_suffix"
    else:
        algorithm = {"name": None, "status": "unregistered"}
        params = {"declared": [], "undeclared": len(typed)}
        return algorithm, params, "unregistered_algorithm", None
    schema = schemas[name]
    known = {field["name"] for field in schema["fields"]} | {
        param["name"] for param in schema["facade_params"]
    }
    declared = sorted(set(typed) & known)
    algorithm = {"name": name, "status": status}
    params = {"declared": declared, "undeclared": len(set(typed) - known)}
    if status == "registered_name_with_suffix":
        return algorithm, params, "unregistered_algorithm", {"modes": _modes(schema)}
    owned = {
        field["name"]: field["facade_param"]
        for field in schema["fields"]
        if field["facade_param"] not in (None, field["name"]) and field["name"] in typed
    }
    if owned:
        return algorithm, params, "facade_owned_field", {"fields": owned}
    if params["undeclared"]:
        return algorithm, params, "undeclared_parameter", None
    return algorithm, params, None, None


def _modes(schema: dict[str, Any]) -> list[str]:
    return sorted(mode["name"] for mode in schema["modes"])


def _replay(instrument: str, merged: dict[str, Any]) -> str | None:
    try:
        kd.Model(algorithm=instrument, seed=0, **merged)
    except Exception as error:
        if type(error).__name__ not in experience.CALLER_ERROR_TYPES:
            raise
        return type(error).__name__
    return None


def project_refusal(
    call: dict[str, Any],
    raised: str | None,
    schemas: dict[str, dict[str, Any]],
    *,
    bound: bool,
) -> dict[str, Any] | None:
    algorithm, params, category, correction = _typed_projection(
        call["instrument"], sorted(call["requested"]), schemas
    )
    if raised is not None:
        refusal = {
            "phase": "construction",
            "exception": raised,
            "category": category or "argument_refused",
            "correction": correction,
        }
        return {"algorithm": algorithm, "params": params, "refusal": refusal}
    if category is not None:
        raise ValueError(
            f"the building kd constructs a call its schema names {category}"
        )
    schema = schemas[algorithm["name"]]
    inputs = mode_inputs(call["report"])
    if any(kd.mode_accepts(mode, **inputs).accepts for mode in schema["modes"]):
        return None
    if not bound:
        return None
    refusal = {
        "phase": "fit",
        "exception": "NotImplementedError",
        "category": "no_mode_accepts_input",
        "correction": {"modes": _modes(schema)},
    }
    return {"algorithm": algorithm, "params": params, "refusal": refusal}


_UNVERIFIED = {
    False: "unverified_before_search",
    True: "unverified_in_search",
    None: "unverified_phase_unknown",
}


def replay_refusals(
    calls: Sequence[dict[str, Any]], *, produced_under: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    schemas = {schema["algorithm"]: schema for schema in kd.instrument_schemas()}
    same_declarations = produced_under == kd.instrument_schema_digest()
    replays: dict[str, str | None] = {}
    verified: list[dict[str, Any]] = []
    unverified: dict[str, Counter[str]] = {
        bucket: Counter() for bucket in _UNVERIFIED.values()
    }
    refused = []
    for call in calls:
        merged = {**call["requested"], **call["defaults_applied"]}
        key = json.dumps([call["instrument"], merged], sort_keys=True)
        bound = (
            same_declarations
            and call["code"] == "NotImplementedError"
            and call["search_started"] is False
        )
        try:
            if key not in replays:
                replays[key] = _replay(call["instrument"], merged)
            projection = project_refusal(call, replays[key], schemas, bound=bound)
        except Exception as error:
            _algorithm, params, _category, _correction = _typed_projection(
                call["instrument"], sorted(call["requested"]), schemas
            )
            refused.append(
                {
                    "problem_key": call["problem_key"],
                    "instrument": call["instrument"],
                    "declared": params["declared"],
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        if projection is None:
            unverified[_UNVERIFIED[call["search_started"]]][call["code"]] += 1
            continue
        verified.append(
            {
                "problem_key": call["problem_key"],
                "group": call["group"],
                "task_kind": call["task_kind"],
                "report": call["report"],
                "projection": projection,
            }
        )
    if refused:
        raise ObservationsRefusedError(refused)
    return verified, {
        bucket: dict(sorted(counts.items())) for bucket, counts in unverified.items()
    }


def carve_refusals(
    refusals: Sequence[dict[str, Any]],
    *,
    bins: dict[str, Sequence[float]],
    min_support: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    null_match = 0
    for refusal in refusals:
        kind = refusal["task_kind"]
        features = experience.features_of(
            refusal["report"], kind, experience.FEATURES[kind], bins
        )
        match = {key: features[key] for key in experience.REFUSAL_KEYS[kind]}
        if any(value is None for value in match.values()):
            null_match += 1
            continue
        identity = experience.refusal_identity(
            {"match": match, **refusal["projection"]}
        )
        groups.setdefault(identity, []).append({"match": match, **refusal})
    rows: list[dict[str, Any]] = []
    written: dict[str, dict[str, int]] = {}
    below_support = 0
    for identity in sorted(groups):
        members = groups[identity]
        problems = len({member["problem_key"] for member in members})
        if problems < min_support:
            below_support += len(members)
            continue
        first = members[0]
        rows.append(
            {
                "match": first["match"],
                **first["projection"],
                "support": {"n_calls": len(members), "n_problems": problems},
            }
        )
        counts = written.setdefault(
            first["projection"]["refusal"]["category"], {"rows": 0, "calls": 0}
        )
        counts["rows"] += 1
        counts["calls"] += len(members)
    rows.sort(
        key=lambda row: (
            -row["support"]["n_problems"],
            -row["support"]["n_calls"],
            experience.refusal_identity(row),
        )
    )
    return rows, {
        "written": dict(sorted(written.items())),
        "below_support": below_support,
        "null_match": null_match,
    }


def refusal_log(
    calls: Sequence[dict[str, Any]],
    refusals: Sequence[dict[str, Any]],
    unverified: dict[str, dict[str, int]],
    *,
    bins: dict[str, Sequence[float]],
    min_support: int,
) -> dict[str, Any]:
    _rows, carved = carve_refusals(refusals, bins=bins, min_support=min_support)
    return {
        "refusals": [
            {
                "problem_key": call["problem_key"],
                "dataset_id": call["dataset_id"],
                "instrument": call["instrument"],
                "typed": sorted(call["requested"]),
                "code": call["code"],
                "search_started": call["search_started"],
                "detail": call["detail"],
            }
            for call in calls
        ],
        "refusals_written": carved["written"],
        "refusals_dropped": {
            "below_support": carved["below_support"],
            "null_match": carved["null_match"],
            **unverified,
        },
    }


def _vocabulary(kind: str, templates: list[list[str]]) -> list[str]:
    requested = {key for template in templates for key in template}
    return [
        *experience.FEATURES[kind],
        *(
            key
            for key in experience.OPTIONAL_FEATURES.get(kind, ())
            if key in requested
        ),
    ]


def carve(
    observations: Sequence[Observation],
    *,
    templates: dict[str, list[list[str]]],
    bins: dict[str, Sequence[float]],
    min_support: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    validate_observations(observations)
    rows: list[dict[str, Any]] = []
    log: dict[str, list[dict[str, Any]]] = {"rows": [], "dropped": []}
    for kind in experience.FEATURES:
        vocabulary = _vocabulary(kind, templates[kind])
        featured = [
            {
                **obs,
                "features": experience.features_of(
                    obs["report"], kind, vocabulary, bins
                ),
            }
            for obs in observations
            if obs["task_kind"] == kind
        ]
        for keys in templates[kind]:
            for match, members, _template in grouped(featured, [{"keys": keys}]):
                if any(value is None for value in match.values()):
                    continue
                for recipe, runs in _by_pipeline(members):
                    problems = sorted({run["problem_key"] for run in runs})
                    entry = {"match": match, **recipe, "problems": problems}
                    if len(problems) < min_support:
                        log["dropped"].append({**entry, "n_runs": len(runs)})
                        continue
                    rows.append(_row(match, recipe, runs, len(problems)))
                    log["rows"].append(entry)
    return rows, log


def _by_pipeline(
    members: Iterable[Observation],
) -> list[tuple[dict[str, Any], list[Observation]]]:
    split: dict[str, list[Observation]] = {}
    for member in members:
        recipe = {
            "pipeline": experience.pipeline_identity(member["pipeline"]),
            "conditions": member["conditions"],
        }
        split.setdefault(json.dumps(recipe, sort_keys=True), []).append(member)
    return [(json.loads(key), split[key]) for key in sorted(split)]


def _row(
    match: dict[str, Any],
    recipe: dict[str, Any],
    runs: Sequence[Observation],
    n_problems: int,
) -> dict[str, Any]:
    return {
        "match": match,
        **recipe,
        "result": {
            "structure_rate": sum(bool(run["structure"]) for run in runs) / len(runs),
            "median_max_relative_error": _median_present(
                run["max_relative_error"] for run in runs
            ),
            "median_heldout_nrmse": _median_present(
                run["heldout_nrmse"] for run in runs
            ),
        },
        "cost": {
            "median_evaluations": _median_present(run["evaluations"] for run in runs),
            "median_wall_seconds": _median_present(run["wall_seconds"] for run in runs),
        },
        "support": {
            "n_runs": len(runs),
            "n_problems": n_problems,
            "n_refused": sum(bool(run.get("refused")) for run in runs),
        },
    }


def _median_present(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def grouped(
    observations: Sequence[dict[str, Any]], templates: Sequence[dict[str, Any]]
) -> list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]]:
    carved: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = []
    for template in templates:
        keys: Sequence[str] = template["keys"]
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for observation in observations:
            signature = tuple(observation["features"][key] for key in keys)
            groups.setdefault(signature, []).append(observation)
        carved.extend(
            (dict(zip(keys, signature, strict=True)), members, template)
            for signature, members in groups.items()
        )
    return carved


def build(args: argparse.Namespace) -> int:
    source: dict[str, str] = {}
    read: dict[str, Any] = {}
    try:
        reader = read_export if args.export is not None else read_graded_runs
        observations, excluded, source, read = reader(
            args.export if args.export is not None else args.graded_runs
        )
        calls = read.pop("refusals")
        refusals, unverified = replay_refusals(
            calls, produced_under=read["produced_under"]
        )
        document = emit(
            observations,
            refusals=refusals,
            templates=TEMPLATES,
            bins=BINS,
            min_support=args.min_support,
            excluded_groups=excluded,
            source=source,
            across_versions=args.across_examinee_versions,
            source_rules_version=read["rules_version"],
        )
        if "admission" in read:
            document["admission"] = read["admission"]
            logger.warning(
                "source admission: %s; produced_under: %s; filled: %s; "
                "pipeline rates omit unassigned refusals",
                read["admission"],
                read["produced_under"],
                read["filled"],
            )
            for refusal in read["refused"]:
                logger.warning("source row refused: %s", refusal)
            logger.warning("source failure census: %s", read["failures"])
    except (ObservationsRefusedError, SourceRefusedError) as error:
        if args.log is not None:
            _write(
                args.log,
                {
                    "source": source,
                    "read": error.read
                    if isinstance(error, SourceRefusedError)
                    else read,
                    "refused": error.refused,
                },
            )
        raise
    heldout = None if args.heldout is None else heldout_rows(args.heldout, excluded)
    _write(args.out, document)
    coverage = (
        {}
        if heldout is None
        else {
            "heldout_coverage": heldout_coverage(
                experience.load_experience(args.out), *heldout
            )
        }
    )
    _rows, carved = carve(
        observations, templates=TEMPLATES, bins=BINS, min_support=args.min_support
    )
    refused_calls = refusal_log(
        calls, refusals, unverified, bins=BINS, min_support=args.min_support
    )
    logger.info(
        "wrote %d rows and %d refusal rows to %s; %d rows below min_support=%d "
        "dropped; refusal calls dropped: %s",
        len(document["rows"]),
        len(document["refusals"]),
        args.out,
        len(carved["dropped"]),
        args.min_support,
        refused_calls["refusals_dropped"],
    )
    if args.log is not None:


        _write(
            args.log,
            {"source": source, "read": read, **carved, **refused_calls, **coverage},
        )
    return 0


def heldout_rows(
    path: Path, excluded: Sequence[str]
) -> tuple[list[Observation], dict[str, str]]:
    if not excluded:
        raise ValueError(
            "--heldout lists held-out problems of an evaluation build; this "
            "source withholds no group"
        )
    observations, _withheld, source, _read = read_export(path)
    rows = [obs for obs in observations if obs["group"] in excluded]
    missing = sorted(set(excluded) - {obs["group"] for obs in rows})
    if missing:
        raise ValueError(f"{path} holds no row of the withheld groups {missing}")
    return rows, source


def heldout_coverage(
    document: experience.Experience,
    rows: Sequence[Observation],
    source: dict[str, str],
) -> dict[str, Any]:
    problems: dict[str, dict[str, Any]] = {}
    for obs in rows:
        kind = obs["task_kind"]
        features = experience.features_of(
            obs["report"],
            kind,
            document["features_vocabulary"][kind],
            document["bins"],
        )
        key = json.dumps([obs["problem_key"], kind, features], sort_keys=True)
        if key in problems:
            continue
        problems[key] = {
            "problem_key": obs["problem_key"],
            "group": obs["group"],
            "task_kind": kind,
            "features": features,
            "rows_retrieved": len(
                experience_applicability.retrieve_for_report(
                    document, kind, obs["report"], k=2
                )
            ),
            "refusals_retrieved": len(
                experience.retrieve_refusals(document, kind, features)
            ),
        }
    listed = sorted(problems.values(), key=lambda p: (p["group"], p["problem_key"]))
    uncovered = sorted({p["problem_key"] for p in listed if p["rows_retrieved"] == 0})
    if uncovered:
        logger.warning(
            "held-out problems no experience row matches: %d of %d (%s)",
            len(uncovered),
            len({p["problem_key"] for p in listed}),
            ", ".join(uncovered),
        )
    return {"source": source, "problems": listed, "uncovered": uncovered}


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
