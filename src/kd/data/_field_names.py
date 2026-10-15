
from __future__ import annotations

from kd.core.expr.executor import DIFF_OPERATOR_PATTERN, UNITY_TOKEN
from kd.core.expr.registry import FunctionRegistry

_OPERATOR_NAMES = frozenset(FunctionRegistry.create_default().list_names())

_RESERVED_FIELD_NAMES = _OPERATOR_NAMES | {UNITY_TOKEN}

_RENAME_SUFFIX = "val"


def is_reserved_field_name(name: str) -> bool:
    return (
        name in _RESERVED_FIELD_NAMES
        or DIFF_OPERATOR_PATTERN.fullmatch(name) is not None
    )


def suggested_field_name(name: str) -> str:
    return f"{name.replace('_', '')}{_RENAME_SUFFIX}"


def reserved_field_name_message(name: str, *, label: str) -> str:
    if name == UNITY_TOKEN:
        meaning = "the constant column 1"
    elif name in _OPERATOR_NAMES:
        meaning = "an operator"
    else:
        meaning = "a derivative operator"
    return (
        f"{label} {name!r} is reserved: kd reads {name!r} as {meaning}, so "
        f"terms could never refer to this data. Rename it, for example to "
        f"{suggested_field_name(name)!r}."
    )
