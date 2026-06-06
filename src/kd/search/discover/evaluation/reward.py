
from __future__ import annotations

import math

from kd.core.evaluator import EvaluationResult

DEFAULT_ALPHA = 0.01
MIN_REWARD = 0.0
MAX_REWARD = 1.0


def compute_reward(
    result: EvaluationResult,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    if not result.is_valid:
        return MIN_REWARD

    nmse = result.nmse
    if not math.isfinite(nmse) or nmse < MIN_REWARD:
        return MIN_REWARD

    numerator = 1.0 - alpha * result.complexity
    denominator = 1.0 + math.sqrt(nmse)
    reward = numerator / denominator
    return float(min(MAX_REWARD, max(MIN_REWARD, reward)))
