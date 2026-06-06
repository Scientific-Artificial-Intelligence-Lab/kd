
from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections import defaultdict
from typing import Any

from torch import Tensor

logger = logging.getLogger(__name__)

_LARGE_TENSOR_WARNING_THRESHOLD = 10_000


def _detach_tensor(value: Tensor) -> Tensor:
    tensor = value.detach().clone()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor


def _is_dataclass_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _detach_field_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return _detach_tensor(value)
    return _detach_recursive(value)


def _detach_recursive(value: Any) -> Any:
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.detach().item()
        return _detach_tensor(value)
    if isinstance(value, dict):
        return {k: _detach_recursive(v) for k, v in value.items()}


    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)._make(_detach_recursive(v) for v in value)
    if isinstance(value, list):
        return [_detach_recursive(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_detach_recursive(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return type(value)(_detach_recursive(v) for v in value)
    if _is_dataclass_instance(value):
        detached = {
            f.name: _detach_field_value(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if f.init
        }
        return dataclasses.replace(value, **detached)
    return value


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True


def _sanitize_float(value: float) -> float | None:
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _make_json_safe(value: Any, *, key: str) -> Any:
    if isinstance(value, float):
        return _sanitize_float(value)

    if isinstance(value, Tensor):
        safe_list = _detach_tensor(value).tolist()

        return _make_json_safe(safe_list, key=key)

    if isinstance(value, dict):
        return {
            str(child_key): _make_json_safe(child_value, key=key)
            for child_key, child_value in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item, key=key) for item in value]

    if _is_json_serializable(value):
        return value

    logger.warning(
        "VizRecorder: key '%s' uses string fallback for %s",
        key,
        type(value).__name__,
    )
    return str(value)


class VizRecorder:

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._store: defaultdict[str, list[Any]] = defaultdict(list)

    def log(self, key: str, value: Any) -> None:
        if not self.enabled:
            return

        if isinstance(value, Tensor):
            if value.numel() > _LARGE_TENSOR_WARNING_THRESHOLD:
                logger.warning(
                    "VizRecorder: large tensor for key '%s' (%d elements)",
                    key,
                    value.numel(),
                )
            if value.numel() == 1:
                value = value.detach().item()
            else:
                value = _detach_tensor(value)
        elif isinstance(
            value, (list, tuple, dict, set, frozenset)
        ) or _is_dataclass_instance(value):
            value = _detach_recursive(value)

        self._store[key].append(value)

    def get(self, key: str) -> list[Any]:
        return list(self._store.get(key, []))

    def keys(self) -> set[str]:
        return set(self._store.keys())

    def to_dict(self) -> dict[str, list[Any]]:
        return {
            key: [_make_json_safe(value, key=key) for value in values]
            for key, values in self._store.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[Any]]) -> VizRecorder:
        recorder = cls()
        recorder._store = defaultdict(
            list,
            {key: list(values) for key, values in data.items()},
        )
        return recorder
