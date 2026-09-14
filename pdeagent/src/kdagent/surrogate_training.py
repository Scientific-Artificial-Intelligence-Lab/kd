
from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, cast

from kdagent.data_source import resolve_input
from kdagent.episode_worker import IsolatedTraining, run_isolated_training
from kdagent.surrogates import (
    SurrogateEntry,
    SurrogateRegistry,
    file_sha256,
    surrogate_id_of,
)

DEFAULT_SEED = 0


def artifact_key(schema: dict[str, Any]) -> str:
    keys = schema["config_artifact_keys"]
    if not keys:
        raise ValueError(
            f"{schema['algorithm']} declares no config_artifact_keys, so it fits "
            "no derivative surrogate and there is no key to inject one under"
        )


    key: str
    (key,) = keys
    return key


def unsupported_params(
    schema: dict[str, Any], requested: dict[str, Any] | None
) -> list[str]:
    return sorted(set(requested or {}) - set(schema["surrogate_fields"]))


def surrogate_params(
    schema: dict[str, Any], requested: dict[str, Any] | None
) -> dict[str, Any]:
    extra = unsupported_params(schema, requested)
    if extra:
        raise ValueError(
            f"params {extra} are not surrogate parameters of "
            f"{schema['algorithm']}; only {schema['surrogate_fields']} (plus the "
            "seed) shape the network"
        )
    return deepcopy(requested) if requested else {}


def training_model_kwargs(
    schema: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    model_kwargs = dict(params)
    if any(mode["provider_kind"] != "autograd" for mode in schema["modes"]) and any(
        row["name"] == "derivatives" for row in schema["facade_params"]
    ):
        model_kwargs["derivatives"] = "autograd"
    return model_kwargs


def reuse_hit(
    registry: SurrogateRegistry, recipe: dict[str, Any]
) -> tuple[SurrogateEntry | None, str | None]:
    resolve_input(registry.workspace, recipe["dataset_id"])
    hit = registry.find_by_recipe(recipe)
    if hit is None:
        return None, None
    path = registry.path(hit.surrogate_id)
    if path.is_file() and file_sha256(path) == hit.file_sha256:
        return hit, None
    return None, "surrogate_file_missing"


def run_training(
    *,
    registry: SurrogateRegistry,
    schema: dict[str, Any],
    recipe: dict[str, Any],
    key: str,
    time_cap_seconds: float | None,
) -> IsolatedTraining:
    input_source = resolve_input(registry.workspace, recipe["dataset_id"])
    return run_isolated_training(
        dataset_id=recipe["dataset_id"],
        algorithm=recipe["algorithm"],
        seed=recipe["seed"],
        model_kwargs=training_model_kwargs(schema, recipe["params"]),
        output_path=registry.file_for(surrogate_id_of(recipe)),
        key=key,
        time_cap_seconds=time_cap_seconds,
        input_source=input_source,
    )


def register_training(
    *,
    registry: SurrogateRegistry,
    recipe: dict[str, Any],
    key: str,
    training: IsolatedTraining,
) -> SurrogateEntry:
    return registry.register(
        recipe=recipe,
        key=key,
        file_sha256=cast("str", training.file_sha256),
        train_seconds=cast("float", training.train_seconds),
        epochs=cast("int", training.epochs),
        final_loss=cast("float", training.final_loss),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


__all__ = [
    "DEFAULT_SEED",
    "artifact_key",
    "register_training",
    "reuse_hit",
    "run_training",
    "surrogate_params",
    "training_model_kwargs",
    "unsupported_params",
]
