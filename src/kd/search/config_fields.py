
from __future__ import annotations

import dataclasses
import functools
import types
import typing
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Final, Literal, Union, get_args, get_origin

from kd.search.resume_policy import resolve_field_tier

if TYPE_CHECKING:
    from kd.search.protocol import FacadeWiringContract


FACADE_MAPPED: Final[dict[str, dict[str, str]]] = {
    "sga": {
        "num": "population",
        "depth": "depth",
        "width": "width",
        "aic_ratio": "aic_ratio",
        "use_autograd": "derivatives",
        "seed": "seed",
    },
    "dlga": {"seed": "seed"},
    "discover": {"seed": "seed"},
    "pysr": {"niterations": "generations", "seed": "seed"},
    "eqgpt": {"seed": "seed"},
    "llm4ed": {"seed": "seed"},
    "pysindy": {"seed": "seed"},
}
_NONE_TYPE = type(None)
_UNION_ORIGINS = frozenset({Union, types.UnionType})
_SCALAR_KINDS: Final[dict[type[Any], str]] = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
}


def _without_none(annotation: Any) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in _UNION_ORIGINS and _NONE_TYPE in args:
        members = tuple(arg for arg in args if arg is not _NONE_TYPE)
        if len(members) == 1:
            return members[0], True
    return annotation, False


def _render_type(annotation: Any) -> str:
    inner, nullable = _without_none(annotation)
    if nullable:
        return f"{_render_type(inner)} | None"
    origin = get_origin(inner)
    args = get_args(inner)
    if origin is Literal:
        return "Literal[" + ",".join(repr(value) for value in args) + "]"
    if origin in (list, tuple, frozenset, dict):
        rendered = ", ".join(
            "..." if arg is Ellipsis else _render_type(arg) for arg in args
        )
        return f"{origin.__name__}[{rendered}]"
    if inner is Any:
        return "Any"
    if isinstance(inner, type):
        return inner.__name__
    return str(inner).replace("typing.", "")


def _literal_kind(annotation: Any) -> str | None:
    values = get_args(annotation)
    kinds = {_SCALAR_KINDS.get(type(value)) for value in values}
    if len(kinds) == 1:
        return kinds.pop()
    return None


def _json_kind(annotation: Any) -> str | None:
    inner, _ = _without_none(annotation)
    scalar = _SCALAR_KINDS.get(inner)
    if scalar is not None:
        return scalar
    if get_origin(inner) is Literal:
        return _literal_kind(inner)
    origin = get_origin(inner)
    args = get_args(inner)
    valid_sequence = (
        origin in (list, frozenset) and len(args) == 1
    ) or (origin is tuple and len(args) == 2 and args[1] is Ellipsis)
    if valid_sequence:
        element_kind = _SCALAR_KINDS.get(args[0])
        if element_kind is not None:
            return f"{element_kind}_list"
    if origin is dict and args == (str, Any):
        return "dict"
    return None


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_default(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, frozenset):
        return [_json_default(item) for item in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [_json_default(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_default(item) for key, item in value.items()}
    return value


def _default_of(field: dataclasses.Field[Any], json_kind: str | None) -> Any:
    if json_kind is None:
        return None
    if field.default is not dataclasses.MISSING:
        return _json_default(field.default)
    if field.default_factory is not dataclasses.MISSING:
        return _json_default(field.default_factory())
    return None


@functools.lru_cache(maxsize=None)
def _resolved_hints(config_cls: type[Any]) -> dict[str, Any]:
    return typing.get_type_hints(config_cls, include_extras=True)


def field_specs(plugin_cls: type[FacadeWiringContract]) -> list[dict[str, Any]]:
    algorithm = plugin_cls.descriptor.algorithm
    hints = _resolved_hints(typing.cast(Hashable, plugin_cls.config_cls))
    knobs = {knob.name: knob for knob in plugin_cls.descriptor.knobs}
    mapped = FACADE_MAPPED[algorithm]
    rows: list[dict[str, Any]] = []
    for field in dataclasses.fields(plugin_cls.config_cls):
        annotation = hints[field.name]
        raw_kind = _json_kind(annotation)
        knob = knobs.get(field.name)
        required = (
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        )
        settable_from = (
            "facade"
            if field.name in mapped
            else "json" if raw_kind is not None else "python"
        )
        rows.append(
            {
                "name": field.name,
                "type": _render_type(annotation),





                "json_kind": raw_kind,
                "literal_values": (
                    list(get_args(_without_none(annotation)[0]))
                    if get_origin(_without_none(annotation)[0]) is Literal
                    else None
                ),
                "nullable": _without_none(annotation)[1],
                "default": _default_of(field, raw_kind),
                "required": required,
                "settable_from": settable_from,
                "facade_param": mapped.get(field.name),
                "resume_tier": resolve_field_tier(plugin_cls, field.name),
                "knob": knob is not None,
                "description": "" if knob is None else knob.description,
            }
        )
    return rows


def _is_json_native(value: Any) -> bool:
    return value is None or type(value) in {str, int, float, bool, list, dict}


def _matches_scalar(value: Any, kind: str) -> bool:
    if kind == "float":
        return type(value) in (int, float)
    expected = {"bool": bool, "int": int, "str": str}[kind]
    return type(value) is expected


def _wrong_type(algorithm: str, spec: dict[str, Any], value: Any) -> TypeError:
    return TypeError(
        f"Model(algorithm={algorithm!r}) fields={[spec['name']]} require "
        f"declared type {spec['type']}; got {type(value).__name__}. "
        "Check kd.instrument_schemas() for legal JSON field types."
    )


def _snapshot_json_containers(value: Any) -> Any:
    if type(value) is dict:
        return {
            key: _snapshot_json_containers(item) for key, item in value.items()
        }
    if type(value) is list:
        return [_snapshot_json_containers(item) for item in value]
    return value


def _validate_list(
    algorithm: str, spec: dict[str, Any], value: Any, element_kind: str
) -> None:
    if type(value) is not list:
        raise _wrong_type(algorithm, spec, value)
    for index, item in enumerate(value):
        if not _matches_scalar(item, element_kind):
            raise TypeError(
                f"Model(algorithm={algorithm!r}) fields={[spec['name']]} "
                f"element {index} requires {element_kind}; got "
                f"{type(item).__name__}. Check kd.instrument_schemas() for "
                "legal JSON element types."
            )


def _validate_json_value(algorithm: str, spec: dict[str, Any], value: Any) -> Any:
    if value is None:
        if spec["nullable"]:
            return None
        raise TypeError(
            f"Model(algorithm={algorithm!r}) fields={[spec['name']]} cannot be "
            "null. Supply a non-null value of the declared type."
        )
    kind = spec["json_kind"]
    if kind == "dict":
        if type(value) is not dict:
            raise _wrong_type(algorithm, spec, value)
        return _snapshot_json_containers(value)
    if kind.endswith("_list"):
        element_kind = kind.removesuffix("_list")
        _validate_list(algorithm, spec, value, element_kind)
        return _coerce_list(spec, value)
    if not _matches_scalar(value, kind):
        raise _wrong_type(algorithm, spec, value)
    allowed = spec["literal_values"]
    if allowed is not None and value not in allowed:
        raise ValueError(
            f"Model(algorithm={algorithm!r}) fields={[spec['name']]} received "
            f"{value!r}; legal Literal values are "
            f"{sorted(allowed, key=repr)}. Choose one of those values."
        )
    return value


def _coerce_list(spec: dict[str, Any], value: list[Any]) -> Any:
    annotation_text = spec["type"].removesuffix(" | None")
    if annotation_text.startswith("tuple["):
        return tuple(value)
    if annotation_text.startswith("frozenset["):
        return frozenset(value)
    return list(value)


def _closed_field_error(
    algorithm: str, spec: dict[str, Any], value: Any
) -> TypeError:
    asset_hint = (
        " For EqGPT asset paths, set KD_EQGPT_ASSET_DIR or pass a Path via config=."
        if algorithm == "eqgpt" and "Path" in spec["type"]
        else ""
    )
    return TypeError(
        f"Model(algorithm={algorithm!r}) fields={[spec['name']]} are Python-only "
        f"({spec['type']}); got JSON {type(value).__name__}. Pass a typed "
        f"{algorithm} config through config= instead.{asset_hint}"
    )


def normalize_config_kwargs(
    plugin_cls: type[FacadeWiringContract], algorithm: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    specs = {row["name"]: row for row in field_specs(plugin_cls)}
    normalized: dict[str, Any] = {}
    for name, value in kwargs.items():
        spec = specs.get(name)
        if spec is None:
            raise TypeError(
                f"Model(algorithm={algorithm!r}) fields={[name]} are not fields "
                f"of {plugin_cls.config_cls.__name__}. Check "
                "kd.instrument_schemas() for legal names."
            )
        if spec["settable_from"] == "facade":
            raise TypeError(
                f"Model(algorithm={algorithm!r}) fields={[name]} are owned by "
                f"the facade; use {spec['facade_param']}= instead."
            )
        if not _is_json_native(value):
            normalized[name] = value
        elif spec["settable_from"] == "python":
            if value is None and spec["nullable"]:
                normalized[name] = None
            elif value is None:
                raise TypeError(
                    f"Model(algorithm={algorithm!r}) fields={[name]} are not "
                    "nullable. Pass a non-null Python object through kwargs or "
                    "a typed config through config=."
                )
            else:
                raise _closed_field_error(algorithm, spec, value)
        else:
            normalized[name] = _validate_json_value(algorithm, spec, value)
    return normalized


__all__ = [
    "FACADE_MAPPED",
    "field_specs",
    "normalize_config_kwargs",
]
