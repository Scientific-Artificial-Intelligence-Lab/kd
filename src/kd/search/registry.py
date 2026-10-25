
from __future__ import annotations

import dataclasses
import inspect
from typing import TYPE_CHECKING, Final

from kd._registry import Registry, extension_origin
from kd.search.descriptor import tool_schema

if TYPE_CHECKING:
    from kd.search.protocol import FacadeWiringContract


EXTENSION_CONFIG_KEY: Final = "extension"


FACADE_CONSTRUCTOR_KWARGS: Final = frozenset(
    {"mode", "task", "surrogate_model", "provider"}
)

_algorithms: Registry[type[FacadeWiringContract]] = Registry("algorithm")


def register_algorithm(name: str, plugin_cls: type[FacadeWiringContract]) -> None:
    _algorithms.check_available(name)
    declared = plugin_cls.descriptor.algorithm
    if declared != name:
        raise ValueError(
            f"{plugin_cls.__name__}.descriptor names algorithm {declared!r}, not "
            f"{name!r}: its schema would be published under a name kd does not "
            "dispatch"
        )
    _check_constructor_kwargs(plugin_cls)
    config_cls = plugin_cls.config_cls
    if "seed" not in {field.name for field in dataclasses.fields(config_cls)}:
        raise TypeError(
            f"{config_cls.__name__}, the config of {name!r}, has no seed field; "
            "kd.Model threads its seed into every config"
        )
    keys = plugin_cls.descriptor.config_artifact_keys
    missing = [
        m for m in ("train_surrogate", "artifacts") if not hasattr(plugin_cls, m)
    ]
    if keys and missing:
        raise TypeError(
            f"{plugin_cls.__name__} declares config_artifact_keys {sorted(keys)} "
            f"but has no {missing}: Model.train_surrogate dispatches on that "
            "declaration (kd.search.SurrogateTrainer)"
        )
    tool_schema(plugin_cls)
    _algorithms.register(name, plugin_cls)


def _check_constructor_kwargs(plugin_cls: type[FacadeWiringContract]) -> None:
    declared = plugin_cls.constructor_kwargs
    foreign = sorted(declared - FACADE_CONSTRUCTOR_KWARGS)
    if foreign:
        raise ValueError(
            f"{plugin_cls.__name__}.constructor_kwargs names {foreign}; kd.Model "
            f"supplies only {sorted(FACADE_CONSTRUCTOR_KWARGS)}"
        )
    parameters = inspect.signature(plugin_cls).parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return
    keywords = {
        p.name
        for p in parameters
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    absent = sorted(declared - keywords)
    if absent:
        raise TypeError(
            f"{plugin_cls.__name__} takes no constructor keyword {absent}, which "
            "its constructor_kwargs declares"
        )


def plugin_class(name: str) -> type[FacadeWiringContract]:
    return _algorithms.get(name)


def algorithm_names() -> tuple[str, ...]:
    return _algorithms.names()


def execution_identity(name: str | None) -> dict[str, dict[str, str | None]]:
    if (
        name is None
        or name not in _algorithms
        or _algorithms.origin(name).source == "builtin"
    ):
        return {}
    return {EXTENSION_CONFIG_KEY: extension_origin("algorithm", name)}


__all__ = [
    "EXTENSION_CONFIG_KEY",
    "FACADE_CONSTRUCTOR_KWARGS",
    "algorithm_names",
    "execution_identity",
    "plugin_class",
    "register_algorithm",
]
