
from __future__ import annotations

import json
import logging
import math
from typing import Any, Final

from torch import Tensor

logger = logging.getLogger(__name__)




JSON_INDENT_SPACES: Final[int] = 2


def detach_tensor(value: Tensor) -> Tensor:
    tensor = value.detach().clone()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor


def is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True


def sanitize_float(value: float) -> float | None:
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None










NO_MEASUREMENT: Final[float] = float("nan")


def is_measured(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def make_json_safe(value: Any, *, key: str) -> Any:
    if isinstance(value, float):
        return sanitize_float(value)

    if isinstance(value, Tensor):
        safe_list = detach_tensor(value).tolist()

        return make_json_safe(safe_list, key=key)

    if isinstance(value, dict):
        return {
            str(child_key): make_json_safe(child_value, key=key)
            for child_key, child_value in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item, key=key) for item in value]

    if is_json_serializable(value):
        return value

    logger.warning(
        "json-safe: key '%s' uses string fallback for %s",
        key,
        type(value).__name__,
    )
    return str(value)
