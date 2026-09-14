
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

__all__ = ["RateSummary", "paired_exact_test", "rate_summary", "wilson_interval"]


def _require_count(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool | float) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    count = int(value)
    if count < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return count


def wilson_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    from scipy.stats import norm

    k = _require_count(successes, "successes")
    n = _require_count(trials, "trials")
    if k > n:
        raise ValueError(f"successes ({k}) cannot exceed trials ({n})")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if n == 0:
        return 0.0, 1.0
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)) ** 0.5) / denominator
    lower = 0.0 if k == 0 else max(0.0, centre - half)
    upper = 1.0 if k == n else min(1.0, centre + half)
    return lower, upper


@dataclass(frozen=True)
class RateSummary:

    successes: int
    trials: int
    rate: float
    lower: float
    upper: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "successes": self.successes,
            "trials": self.trials,
            "rate": self.rate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
        }


def rate_summary(
    successes: int, trials: int, *, confidence: float = 0.95
) -> RateSummary:


    k = _require_count(successes, "successes")
    n = _require_count(trials, "trials")
    lower, upper = wilson_interval(k, n, confidence=confidence)
    rate = 0.0 if n == 0 else k / n
    return RateSummary(
        successes=k,
        trials=n,
        rate=rate,
        lower=lower,
        upper=upper,
        confidence=confidence,
    )


def paired_exact_test(only_a: int, only_b: int) -> float:
    from scipy.stats import binomtest

    a = _require_count(only_a, "only_a")
    b = _require_count(only_b, "only_b")
    if a + b == 0:
        return 1.0
    return float(binomtest(a, a + b, 0.5, alternative="two-sided").pvalue)
