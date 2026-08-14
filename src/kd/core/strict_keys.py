
from __future__ import annotations

from collections.abc import Mapping


def strict_keys(
    data: Mapping[str, object],
    *,
    object_name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    error_cls: type[Exception],
) -> None:
    actual = frozenset(data)
    unknown = actual - required - optional
    if unknown:
        keys = ", ".join(repr(key) for key in sorted(unknown))
        raise error_cls(f"Unknown {object_name} field(s): {keys}")
    missing = required - actual
    if missing:
        keys = ", ".join(repr(key) for key in sorted(missing))
        raise error_cls(f"Missing required {object_name} field(s): {keys}")
