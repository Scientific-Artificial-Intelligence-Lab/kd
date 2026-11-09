
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from functools import lru_cache
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import kd

from kdagent import experience
from kdagent.experience_build import BINS, TEMPLATES, emit
from kdagent.experience_replay import evaluate_pipeline, load_menu, pipeline_recipe
from kdagent.experience_round import grade, read_round
from kdagent.experience_sources import strict_json

LOG = logging.getLogger(__name__)
FORMAT = "kd-experience-learning-v1"
RULES = "canonical-terms-v1:small_term_rtol=0:coeff_rtol=0.05"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _code_digest(root: Path) -> str:
    return _digest(
        [
            (str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(root.rglob("*.py"))
        ]
    )


@lru_cache(maxsize=1)
def _environment() -> dict[str, str]:
    from kdagent import experience_replay

    return {
        "kd_version": kd.__version__,
        "kd_source_sha256": _code_digest(Path(kd.__file__).parent),
        "controller_source_sha256": _code_digest(Path(__file__).parent),
        "instrument_schema_sha256": kd.instrument_schema_digest(),
        "replay_sha256": hashlib.sha256(
            Path(experience_replay.__file__).read_bytes()
        ).hexdigest(),
        "rules": RULES,
    }


def _state(out: Path) -> dict[str, Any]:
    path = out / "state.json"
    if path.exists():
        state: dict[str, Any] = strict_json(path.read_bytes())
        if state["format"] != FORMAT or state["environment"] != _environment():
            raise ValueError(
                "learning state uses a different implementation; use a new output directory"
            )
        return state
    if out.exists() and any(out.iterdir()):
        raise ValueError("learning output is nonempty without state.json")
    return {
        "format": FORMAT,
        "environment": _environment(),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": {},
        "attempts": {},
    }


def _check_sources(
    runs: list[dict[str, Any]], state: dict[str, Any], out: Path
) -> None:
    for run in runs:
        report = run["report"]
        if (
            run["task_kind"] != "pde-evolution"
            or len(report["fields"]) != 1
            or len(report["axes"]) != 2
        ):
            raise ValueError(
                "learning menu requires single-field, one-spatial-axis PDE evolution inputs"
            )
        old = state["runs"].get(run["run_id"])
        if old is not None and old["source_digest"] != run["source_digest"]:
            raise ValueError(f"sealed source changed for run {run['run_id']}")
        if out.is_relative_to(run["source_workspace"].resolve()):
            raise ValueError("learning output overlaps a source workspace")
        for artifact in run["artifact_paths"]:
            path = Path(artifact).resolve()
            if path.is_relative_to(out):
                raise ValueError("learning output overlaps a source artifact directory")


def _conditions(device: str, seconds: float) -> dict[str, Any]:
    threads = {
        os.environ.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    value = next(iter(threads)) if len(threads) == 1 else None
    blas = int(value) if value is not None else None
    seed = os.environ.get("PYTHONHASHSEED")
    return {
        "device": device,
        "soft_seconds": seconds,
        "budget_tier": "pipeline",
        "blas_threads": blas,
        "python_hash_seed": int(seed) if seed and seed != "random" else None,
    }


def _observation(
    run: dict[str, Any], result: dict[str, Any], conditions: dict[str, Any]
) -> dict[str, Any]:
    verdict = grade(result["law"], run["reference"])
    return {
        "problem_key": run["problem_key"],
        "group": _digest(run["problem_key"]),
        "report": run["report"],
        "task_kind": run["task_kind"],
        "pipeline": result["pipeline"],
        "conditions": conditions,
        "instrument_schema_sha256": kd.instrument_schema_digest(),
        "examinee_version": "experience-replay-v1/" + _environment()["replay_sha256"],
        "rules_version": RULES,
        "structure": verdict["structure_ok"],
        "max_relative_error": verdict["max_relative_error"],
        "heldout_nrmse": None,
        "evaluations": result["evaluations"],
        "wall_seconds": result["search_wall_seconds"],
        "refused": result["status"] != "success" or not verdict["graded"],
        "overall": verdict["overall"],
        "grade": verdict,
    }


def _experiment(
    run: dict[str, Any],
    recipe: dict[str, Any],
    conditions: dict[str, Any],
    state: dict[str, Any],
    out: Path,
) -> None:
    identity = _digest(
        {
            "data": run["data_sha256"],
            "load_options": run["load_options"],
            "problem": run["problem_key"],
            "reference": run["reference"],
            "recipe": recipe,
            "conditions": conditions,
        }
    )
    old = state["attempts"].get(identity)
    if old is not None:
        if run["run_id"] not in old["source_runs"]:
            old["source_runs"].append(run["run_id"])
        return
    workspace = out / "replays" / identity
    receipt = workspace / "replay.json"
    if receipt.exists():
        result = strict_json(receipt.read_bytes())
    else:
        result = evaluate_pipeline(
            data_path=run["data_path"],
            load_options=run["load_options"],
            task_kind=run["task_kind"],
            pipeline=recipe,
            workspace=workspace,
            device=conditions["device"],
            seconds=conditions["soft_seconds"],
        )


    result["pipeline"] = pipeline_recipe(recipe, run["report"])
    observation = _observation(run, result, conditions)
    state["attempts"][identity] = {
        "source_runs": [run["run_id"]],
        "recipe_id": recipe["id"],
        "observation": observation,
        "result": result,
    }
    _write(out / "state.json", state)
    LOG.info(
        "%s / %s: %s, recovered=%s",
        run["run_id"],
        recipe["id"],
        result["status"],
        observation["overall"],
    )


def _document(
    state: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    scope: str,
    excluded: list[str],
) -> dict[str, Any]:
    templates = {
        kind: [list(keys) for keys in groups] for kind, groups in TEMPLATES.items()
    }

    templates["pde-evolution"] = [
        keys + ["x_periodic"] for keys in templates["pde-evolution"]
    ]
    document = emit(
        observations,
        refusals=[],
        templates=templates,
        bins=BINS,
        min_support=1,
        excluded_groups=excluded,
        source={"kind": "post-graded-rounds", "digest": _digest(observations)},
        source_rules_version=RULES,
    )
    document["usage_scope"] = scope
    document["built_at"] = state["built_at"]
    return document


def _publish(state: dict[str, Any], out: Path) -> dict[str, Any]:
    observations = [item["observation"] for item in state["attempts"].values()]
    _write(
        out / "library.json",
        _document(state, observations, scope="re-attempt", excluded=[]),
    )
    experience.load_experience(out / "library.json")
    problems = sorted({run["problem_key"] for run in state["runs"].values()})
    documents = {}
    for problem in problems:
        key = _digest(problem)
        own = [obs for obs in observations if obs["problem_key"] == problem]
        heldout = [obs for obs in observations if obs["problem_key"] != problem]
        retry = out / "re-attempt" / f"{key}.json"
        loo = out / "leave-one-out" / f"{key}.json"
        _write(retry, _document(state, own, scope="re-attempt", excluded=[]))
        _write(loo, _document(state, heldout, scope="leave-one-out", excluded=[key]))
        documents[problem] = {"re_attempt": str(retry), "leave_one_out": str(loo)}
    summary = {
        "runs": state["runs"],
        "documents": documents,
        "attempts": len(observations),
        "recovered_attempts": sum(obs["overall"] for obs in observations),
        "failed_or_ungraded_attempts": sum(obs["refused"] for obs in observations),
        "model_calls_in_learning": 0,
        "next_agent_success_rate": None,
        "next_agent_model_calls": None,
        "pipeline_wall_seconds": sum(
            item["result"]["wall_seconds"] for item in state["attempts"].values()
        ),
    }
    _write(out / "summary.json", summary)
    return summary


def learn(
    round_path: Path,
    *,
    out: Path,
    menu_path: Path | None = None,
    pipeline_ids: list[str] | None = None,
    device: str = "cpu",
    seconds: float = 300.0,
) -> dict[str, Any]:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    runs = read_round(round_path)
    recipes = load_menu(menu_path)
    if pipeline_ids is not None:
        unknown = set(pipeline_ids) - {recipe["id"] for recipe in recipes}
        if unknown:
            raise ValueError(f"unknown pipeline ids: {sorted(unknown)}")
        recipes = [recipe for recipe in recipes if recipe["id"] in pipeline_ids]
    out = out.resolve()
    state = _state(out)
    _check_sources(runs, state, out)
    conditions = _conditions(device, seconds)
    for run in runs:
        state["runs"][run["run_id"]] = {
            key: run[key]
            for key in ("problem_key", "source_digest", "diagnosis", "model_calls")
        }
        _write(out / "state.json", state)
        for recipe in recipes:
            _experiment(run, recipe, conditions, state, out)
    _write(out / "state.json", state)
    return _publish(state, out)
