"""Naming rules and helpers for fields, coords, and derivative keys."""

from __future__ import annotations

from typing import Iterable, Dict, List, Tuple, Optional, Sequence, Set
import re


_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
DEFAULT_RESERVED: Set[str] = {
    "0",
    "d",
    "d^2",
    "+",
    "-",
    "*",
    "/",
    "^2",
    "^3",
}


def validate_name(name: str, *, reserved: Optional[Set[str]] = None) -> str:
    """Validate a field/coord name and return it."""
    if not isinstance(name, str) or not name:
        raise ValueError("invalid name: empty or non-string.")
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid name: '{name}'.")
    reserved_set = reserved or DEFAULT_RESERVED
    if name in reserved_set:
        raise ValueError(f"reserved name: '{name}'.")
    return name


def validate_axis_name(name: str, *, reserved: Optional[Set[str]] = None) -> str:
    """Axis names must be parseable in derivative keys."""
    validate_name(name, reserved=reserved)
    if "_" in name:
        raise ValueError(f"Invalid axis name (underscore not allowed): '{name}'.")
    return name


def build_derivative_key(field: str, axis: str, *, order: int = 1) -> str:
    """Return derivative key such as 'u_x' or 'u_xx'."""
    validate_name(field)
    validate_axis_name(axis)
    if order < 1:
        raise ValueError("Derivative order must be >= 1.")
    return f"{field}_{axis * order}"


def parse_derivative_key(
    key: str,
    *,
    known_axes: Optional[Sequence[str]] = None,
) -> Tuple[str, str, int]:
    """Parse derivative key into (field, axis, order)."""
    if not isinstance(key, str) or "_" not in key:
        raise ValueError(f"Invalid derivative key: '{key}'.")
    field, axis_token = key.rsplit("_", 1)
    validate_name(field)
    if not axis_token:
        raise ValueError(f"Invalid derivative key: '{key}'.")

    if known_axes:
        matches: List[Tuple[str, int]] = []
        for axis in known_axes:
            validate_axis_name(axis)
            if len(axis_token) % len(axis) != 0:
                continue
            order = len(axis_token) // len(axis)
            if order >= 1 and axis_token == axis * order:
                matches.append((axis, order))
        if len(matches) != 1:
            raise ValueError(f"Invalid or ambiguous axis token in '{key}'.")
        axis, order = matches[0]
    else:
        axis = axis_token[0]
        if axis_token != axis * len(axis_token):
            raise ValueError(f"Invalid axis token in '{key}'.")
        order = len(axis_token)

    return field, axis, order


def normalize_legacy_derivative_key(
    key: str,
    *,
    field_names: Iterable[str],
    axis_names: Iterable[str],
) -> str:
    """Convert legacy keys like 'uxx' to the canonical 'u_xx' form."""
    if "_" in key:
        return key

    axes = sorted(axis_names, key=len, reverse=True)
    fields = list(field_names)
    for axis in axes:
        validate_axis_name(axis)
    for field in fields:
        validate_name(field)

    candidates: List[str] = []
    for field in fields:
        if not key.startswith(field):
            continue
        remainder = key[len(field):]
        if not remainder:
            continue
        for axis in axes:
            if len(remainder) % len(axis) != 0:
                continue
            order = len(remainder) // len(axis)
            if order >= 1 and remainder == axis * order:
                candidates.append(build_derivative_key(field, axis, order=order))

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        return key
    raise ValueError(f"Ambiguous legacy derivative key: '{key}'.")


def resolve_field_name_conflicts(
    *,
    field_names: Iterable[str],
    coord_names: Iterable[str],
    reserved: Optional[Set[str]] = None,
    field_prefix: str = "f_",
) -> Dict[str, str]:
    """Resolve field name conflicts against coords/reserved names."""
    reserved_set = reserved or DEFAULT_RESERVED
    fields = list(field_names)
    coords = list(coord_names)

    if len(set(fields)) != len(fields):
        raise ValueError("Field names must be unique before resolving conflicts.")

    for coord in coords:
        validate_axis_name(coord, reserved=reserved_set)
    if len(set(coords)) != len(coords):
        raise ValueError("Coordinate names must be unique.")
    if any(coord in reserved_set for coord in coords):
        raise ValueError("Coordinate names cannot be reserved.")

    used = set(coords) | reserved_set
    resolved: Dict[str, str] = {}
    for field in fields:
        validate_name(field, reserved=reserved_set)
        candidate = field
        if candidate in used:
            candidate = f"{field_prefix}{field}"
        counter = 1
        while candidate in used or candidate in resolved.values():
            candidate = f"{field_prefix}{field}_{counter}"
            counter += 1
        validate_name(candidate, reserved=reserved_set)
        resolved[field] = candidate
        used.add(candidate)

    return resolved
