
from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping
from itertools import combinations
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from kd._registry import Registry, extension_dependencies, extension_origin
from kd.search.descriptor import execution_mode, tool_schema

if TYPE_CHECKING:
    from kd.data.schema import DataTopology
    from kd.search.protocol import FacadeWiringContract


EXTENSION_CONFIG_KEY: Final = "extension"
DERIVATIVE_PROVIDER_CONFIG_KEY: Final = "derivative_provider"
COMPONENT_DEPENDENCIES_CONFIG_KEY: Final = "component_dependencies"


@dataclasses.dataclass(frozen=True)
class DerivativeSelection:

    name: str | None
    default: str | None
    mode: str | None = None


@runtime_checkable
class DerivativeSelectionAlgorithm(Protocol):

    derivative_selection: DerivativeSelection | None


def bind_derivative_selection(
    algorithm: DerivativeSelectionAlgorithm,
    *,
    derivative_provider: str | None,
    default_derivative_provider: str | None,
    mode: str | None = None,
) -> None:
    algorithm.derivative_selection = DerivativeSelection(
        derivative_provider, default_derivative_provider, mode
    )


def algorithm_execution_identity(algorithm: Any, name: str | None) -> dict[str, Any]:
    config = algorithm.config
    if isinstance(algorithm, DerivativeSelectionAlgorithm):
        selection = algorithm.derivative_selection
        if selection is not None:
            return execution_identity(
                name,
                derivative_provider=selection.name,
                default_derivative_provider=selection.default,
                mode=selection.mode,
                params=config,
            )
    return execution_identity(name)



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
    if any(
        mode.derivative_service == "platform" or mode.component_fields
        for mode in plugin_cls.descriptor.modes
    ) and not hasattr(plugin_cls, "derivative_selection"):
        raise TypeError(
            f"{plugin_cls.__name__} must declare derivative_selection = None "
            "for the public DerivativeSelectionAlgorithm service contract"
        )
    for first, second in combinations(plugin_cls.descriptor.modes, 2):
        if (
            first.topologies & second.topologies
            and first.config_values == second.config_values
            and first.selector == second.selector
            and (not first.forms or not second.forms or first.forms & second.forms)
            and (
                first.lhs_orders is None
                or second.lhs_orders is None
                or first.lhs_orders & second.lhs_orders
            )
        ):
            raise ValueError(
                f"modes {first.name!r} and {second.name!r} overlap in topology, "
                "forms and lhs_orders with identical execution selectors; "
                "distinguish them with disjoint forms/lhs_orders, config_values "
                "or selector (selected_by is caller guidance; empty forms and "
                "undeclared lhs_orders do not distinguish inputs)"
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


def execution_mode_for_request(
    algorithm: str,
    topology: DataTopology,
    params: Mapping[str, Any],
    *,
    lhs_order: int | None = None,
) -> str:
    cls = plugin_class(algorithm)
    config: dict[str, Any] = {}
    for field in dataclasses.fields(cls.config_cls):
        if field.default is not dataclasses.MISSING:
            config[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            config[field.name] = field.default_factory()
    config.update(params)
    for mode in cls.descriptor.modes:
        if mode.selected_by and all(
            params.get(key) == value for key, value in mode.selected_by.items()
        ):
            config.update(mode.config_values)
    return execution_mode(cls.descriptor, topology, config, lhs_order=lhs_order).name


def execution_dependencies(
    algorithm: str,
    *,
    mode: str,
    params: Mapping[str, Any] | None = None,
) -> list[dict[str, str | None]]:
    cls = plugin_class(algorithm)
    modes = {entry.name: entry for entry in cls.descriptor.modes}
    if mode not in modes:
        raise ValueError(f"unknown mode {mode!r} for {algorithm!r}")
    selected = modes[mode]
    values = {} if params is None else dict(params)
    dependencies = extension_dependencies("algorithm", algorithm)
    if selected.derivative_service == "platform":
        provider = values.get("derivatives", selected.provider_kind)
        dependencies.extend(extension_dependencies("derivative_provider", provider))
    for field, kind in selected.component_fields.items():
        config_field = next(
            item for item in dataclasses.fields(cls.config_cls) if item.name == field
        )
        name = values.get(field, config_field.default)
        dependencies.extend(extension_dependencies(kind, name))
    return list({(item["kind"], item["name"]): item for item in dependencies}.values())


def execution_identity(
    name: str | None,
    *,
    derivative_provider: str | None = None,
    default_derivative_provider: str | None = None,
    mode: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    if (
        name is not None
        and name in _algorithms
        and _algorithms.origin(name).source != "builtin"
    ):
        identity[EXTENSION_CONFIG_KEY] = extension_origin("algorithm", name)
    if derivative_provider is not None:
        origin = extension_origin("derivative_provider", derivative_provider)
        if (
            origin["source"] != "builtin"
            or derivative_provider != default_derivative_provider
        ):
            identity[DERIVATIVE_PROVIDER_CONFIG_KEY] = {
                "name": derivative_provider,
                **origin,
            }
    dependencies: list[dict[str, str | None]] = []
    if name is not None and name in _algorithms:
        values = {} if params is None else dict(params)
        if derivative_provider is not None:
            values["derivatives"] = derivative_provider
        dependencies = (
            execution_dependencies(name, mode=mode, params=values)
            if mode is not None
            else extension_dependencies("algorithm", name)
        )
    elif derivative_provider is not None:
        dependencies = extension_dependencies(
            "derivative_provider", derivative_provider
        )
    components = [
        item
        for item in dependencies
        if item["source"] != "builtin"
        and (item["kind"], item["name"])
        not in {("algorithm", name), ("derivative_provider", derivative_provider)}
    ]
    if components:
        identity[COMPONENT_DEPENDENCIES_CONFIG_KEY] = components
    return identity


__all__ = [
    "DerivativeSelection",
    "DerivativeSelectionAlgorithm",
    "EXTENSION_CONFIG_KEY",
    "FACADE_CONSTRUCTOR_KWARGS",
    "algorithm_names",
    "execution_identity",
    "execution_dependencies",
    "plugin_class",
    "register_algorithm",
]
