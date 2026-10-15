
import math
from collections.abc import Sequence


def small_term_indices(
    coefficients: Sequence[float],
    active: Sequence[int],
    rtol: float,
) -> list[int]:
    finite_magnitudes = [
        abs(coefficients[i]) for i in active if math.isfinite(coefficients[i])
    ]
    if not finite_magnitudes:
        return list(active)
    threshold = rtol * max(finite_magnitudes)
    return [
        i
        for i in active
        if not (math.isfinite(coefficients[i]) and abs(coefficients[i]) < threshold)
    ]
