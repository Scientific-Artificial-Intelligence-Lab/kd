
from __future__ import annotations


def _match_axis_token(
    token: str,
    known_axes: set[str] | None,
) -> tuple[str, int] | None:
    if not token:
        return None
    if known_axes:
        matches: list[tuple[str, int]] = []
        for axis in known_axes:
            if not axis or "_" in axis or len(token) % len(axis) != 0:
                continue
            order = len(token) // len(axis)
            if order >= 1 and token == axis * order:
                matches.append((axis, order))
        if len(matches) == 1:
            return matches[0]
        return None
    axis = token[0]
    if token != axis * len(token):
        return None
    return axis, len(token)


def _parse_segments(
    field: str,
    tokens: list[str],
    known_axes: set[str] | None,
) -> tuple[str, list[tuple[str, int]]] | None:
    if not field or not field.strip():
        return None
    if not tokens or any(not token for token in tokens):
        return None
    parsed: list[tuple[str, int]] = []
    for token in tokens:
        result = _match_axis_token(token, known_axes)
        if result is None:
            return None
        parsed.append(result)
    return field, parsed


def _parse_with_known_fields(
    name: str,
    known_fields: set[str],
    known_axes: set[str] | None,
) -> tuple[str, list[tuple[str, int]]] | None:
    for field in sorted(known_fields, key=len, reverse=True):
        prefix = f"{field}_"
        if not field or not name.startswith(prefix):
            continue
        tokens = name[len(prefix):].split("_")
        result = _parse_segments(field, tokens, known_axes)
        if result is not None:
            return result
    return None


def build_derivative_name(field: str, axis: str, order: int = 1) -> str:
    if not field:
        raise ValueError("field must be non-empty")
    if not axis or "_" in axis:
        raise ValueError("axis must be non-empty and must not contain underscores")
    if order < 1:
        raise ValueError("order must be >= 1")
    return f"{field}_{axis * order}"


def parse_derivative_name(
    name: str,
    *,
    known_fields: set[str] | None = None,
    known_axes: set[str] | None = None,
) -> tuple[str, str, int] | None:
    parsed = parse_compound_derivative(
        name,
        known_fields=known_fields,
        known_axes=known_axes,
    )
    if parsed is None:
        return None
    field, segments = parsed
    if len(segments) != 1:
        return None
    axis, order = segments[0]
    return field, axis, order


def parse_compound_derivative(
    name: str,
    *,
    known_fields: set[str] | None = None,
    known_axes: set[str] | None = None,
) -> tuple[str, list[tuple[str, int]]] | None:
    if not name or "_" not in name:
        return None
    if known_fields:
        return _parse_with_known_fields(name, known_fields, known_axes)
    field, *tokens = name.split("_")
    return _parse_segments(field, tokens, known_axes)


__all__ = [
    "build_derivative_name",
    "parse_compound_derivative",
    "parse_derivative_name",
]
