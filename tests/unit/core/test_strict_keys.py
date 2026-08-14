
from __future__ import annotations

import pytest

from kd.core.strict_keys import strict_keys

pytestmark = pytest.mark.unit

_REQUIRED = frozenset({"alpha", "beta"})
_OPTIONAL = frozenset({"gamma"})


class _SchemaError(ValueError):
    pass


def test_exact_required_keys_accepted() -> None:
    strict_keys(
        {"alpha": 1, "beta": 2},
        object_name="thing",
        required=_REQUIRED,
        error_cls=_SchemaError,
    )


def test_optional_key_accepted_but_not_required() -> None:
    for data in ({"alpha": 1, "beta": 2}, {"alpha": 1, "beta": 2, "gamma": 3}):
        strict_keys(
            data,
            object_name="thing",
            required=_REQUIRED,
            optional=_OPTIONAL,
            error_cls=_SchemaError,
        )


def test_unknown_key_rejected() -> None:
    with pytest.raises(_SchemaError, match="Unknown"):
        strict_keys(
            {"alpha": 1, "beta": 2, "delta": 4},
            object_name="thing",
            required=_REQUIRED,
            error_cls=_SchemaError,
        )


def test_optional_key_is_unknown_when_not_declared() -> None:
    with pytest.raises(_SchemaError, match="gamma"):
        strict_keys(
            {"alpha": 1, "beta": 2, "gamma": 3},
            object_name="thing",
            required=_REQUIRED,
            error_cls=_SchemaError,
        )


def test_missing_required_key_rejected() -> None:
    with pytest.raises(_SchemaError, match="Missing required"):
        strict_keys(
            {"alpha": 1},
            object_name="thing",
            required=_REQUIRED,
            optional=_OPTIONAL,
            error_cls=_SchemaError,
        )


def test_configured_error_cls_is_raised() -> None:
    class _OtherError(Exception):
        pass

    with pytest.raises(_OtherError):
        strict_keys(
            {},
            object_name="thing",
            required=_REQUIRED,
            error_cls=_OtherError,
        )
