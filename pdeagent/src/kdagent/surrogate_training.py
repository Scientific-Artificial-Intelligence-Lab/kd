
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, cast

from kdagent.data_source import resolve_input
from kdagent.episode_worker import (
    IsolatedTraining,
    TrainedNetwork,
    run_isolated_training,
)
from kdagent.session import effective_training_dtype
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


def surrogate_parameter_names(schema: dict[str, Any]) -> list[str]:
    return sorted(
        set(schema["surrogate_fields"])
        | {
            name
            for mode in schema["modes"]
            for params in mode["order_params"].values()
            for name in params
        }
    )


def unsupported_params(
    schema: dict[str, Any], requested: dict[str, Any] | None
) -> list[str]:
    return sorted(set(requested or {}) - set(surrogate_parameter_names(schema)))


def surrogate_params(
    schema: dict[str, Any], requested: dict[str, Any] | None
) -> dict[str, Any]:
    extra = unsupported_params(schema, requested)
    if extra:
        raise ValueError(
            f"params {extra} are not surrogate parameters of "
            f"{schema['algorithm']}; use surrogate_fields or mode order_params "
            "and pass search parameters to run_discovery"
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


def takes_training_dtype(
    schema: dict[str, Any], model_kwargs: Mapping[str, Any], *, table: bool
) -> bool:
    if not any(field["name"] == "training_dtype" for field in schema["fields"]):
        return False
    defaults = {row["name"]: row["default"] for row in schema["facade_params"]}
    return any(
        "training_dtype" not in mode["refused_fields"]
        for mode in schema["modes"]
        if ("tabular" in mode["topologies"]) == table
        if all(
            model_kwargs.get(name, defaults[name]) == value
            for name, value in mode["selected_by"].items()
        )
        if mode["selector"] is None
        or (model_kwargs.get(mode["selector"]["field"]) is not None)
        == mode["selector"]["present"]
    )


def resolve_training_params(
    schema: dict[str, Any], params: dict[str, Any], run_choice: str | None
) -> dict[str, Any]:
    if not takes_training_dtype(
        schema, training_model_kwargs(schema, params), table=False
    ):
        return params
    value = effective_training_dtype(run_choice, params.get("training_dtype"))
    resolved = {key: item for key, item in params.items() if key != "training_dtype"}
    if value is not None:
        resolved["training_dtype"] = value
    return resolved


def search_recipe(
    schema: dict[str, Any],
    dataset_id: str,
    seed: int,
    model_kwargs: Mapping[str, Any],
    run_choice: str | None,
) -> dict[str, Any]:
    names = set(surrogate_parameter_names(schema))
    params = resolve_training_params(
        schema,
        {key: deepcopy(value) for key, value in model_kwargs.items() if key in names},
        run_choice,
    )
    window = model_kwargs.get("savgol_window", 0)
    if window > 0:
        params = {**params, "savgol_window": window}
    return {
        "dataset_id": dataset_id,
        "algorithm": schema["algorithm"],
        "seed": seed,
        "params": params,
    }


def reuse_hit(
    registry: SurrogateRegistry,
    recipe: dict[str, Any],
    *,
    written: Mapping[str, str] | None = None,
) -> tuple[SurrogateEntry | None, str | None]:
    resolve_input(registry.workspace, recipe["dataset_id"], written=written)
    trainings = registry.trainings_of(recipe)
    for hit in trainings:
        path = registry.path(hit.surrogate_id)
        if path.is_file() and file_sha256(path) == hit.file_sha256:
            return hit, None
    return None, "surrogate_file_missing" if trainings else None


def run_training(
    *,
    registry: SurrogateRegistry,
    schema: dict[str, Any],
    recipe: dict[str, Any],
    key: str,
    time_cap_seconds: float | None,
    device: str = "cpu",
    written: Mapping[str, str] | None = None,
) -> IsolatedTraining:
    input_source = resolve_input(
        registry.workspace, recipe["dataset_id"], written=written
    )
    _check_training_input(schema, recipe, input_source)
    return run_isolated_training(
        dataset_id=recipe["dataset_id"],
        algorithm=recipe["algorithm"],
        seed=recipe["seed"],
        model_kwargs=training_model_kwargs(schema, recipe["params"]),
        output_path=registry.file_for(surrogate_id_of(recipe)),
        key=key,
        time_cap_seconds=time_cap_seconds,
        input_source=input_source,
        device=device,
    )


def _check_training_input(
    schema: dict[str, Any], recipe: dict[str, Any], source: dict[str, Any] | None
) -> None:
    from kd import get_dataset

    if source is not None:
        report = source["summary"]["report"]
        if report["topology"] == "tabular":
            raise ValueError(
                "train_surrogate does not take tabular input; use run_discovery"
            )
        orders = {str(report["lhs_order"])}
    else:
        spec = get_dataset(recipe["dataset_id"])


        orders = {
            order
            for mode in schema["modes"]
            for order in mode["order_params"]
            if spec.lhs is not None
            and any(spec.lhs.endswith("_" + axis * int(order)) for axis in spec.axes)
        }
    required = {
        key: value
        for mode in schema["modes"]
        for order, params in mode["order_params"].items()
        if order in orders
        for key, value in params.items()
    }
    defaults = {field["name"]: field["default"] for field in schema["fields"]}
    missing = {
        key: value
        for key, value in required.items()
        if recipe["params"].get(key, defaults[key]) != value
    }
    if missing:
        raise ValueError(f"train_surrogate on this input requires params {required}")


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
        identity=training.identity,
    )


def register_search_network(
    *,
    registry: SurrogateRegistry,
    recipe: dict[str, Any],
    key: str,
    surrogate_id: str,
    network: TrainedNetwork,
    source_run: str | None,
) -> SurrogateEntry:
    return registry.register(
        recipe=recipe,
        key=key,
        file_sha256=network.file_sha256,
        train_seconds=network.train_seconds,
        epochs=network.epochs,
        final_loss=network.final_loss,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        identity=network.identity,
        surrogate_id=surrogate_id,
        source_run=source_run,
    )


__all__ = [
    "DEFAULT_SEED",
    "artifact_key",
    "register_search_network",
    "register_training",
    "resolve_training_params",
    "reuse_hit",
    "run_training",
    "search_recipe",
    "surrogate_parameter_names",
    "surrogate_params",
    "takes_training_dtype",
    "training_model_kwargs",
    "unsupported_params",
]
