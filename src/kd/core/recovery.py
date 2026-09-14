
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "RecoveryVerdict",
    "judge_recovery",
    "load_bearing_recall",
    "span_floor",
    "term_set_jaccard",
]


@dataclass(frozen=True)
class RecoveryVerdict:

    structure_ok: bool
    coefficients_ok: bool
    nmse_ok: bool
    projected_coefficients: dict[str, float]
    span_nmse: float
    drop_one_nmse: dict[str, float]
    nmse: float

    @property
    def overall(self) -> bool:
        return self.structure_ok and self.coefficients_ok and self.nmse_ok


def _span_nmse(rhs: Tensor, theta: Tensor, y_var: float) -> float:
    solution = torch.linalg.lstsq(theta, rhs.unsqueeze(1)).solution.squeeze(1)
    residual = rhs - theta @ solution
    return float(torch.mean(residual**2)) / max(y_var, 1e-30)


def judge_recovery(
    rhs: Tensor,
    u_t: Tensor,
    *,
    truth_columns: dict[str, Tensor],
    truth_coefficients: dict[str, float],
    coeff_rtol: float,
    nmse_threshold: float,
) -> RecoveryVerdict:
    labels = list(truth_columns.keys())
    if set(labels) != set(truth_coefficients.keys()):
        raise ValueError("truth_columns and truth_coefficients must share keys")

    rhs_flat = rhs.flatten().double()
    target = u_t.flatten().double()
    columns = {label: truth_columns[label].flatten().double() for label in labels}
    theta = torch.stack([columns[label] for label in labels], dim=1)

    solution = torch.linalg.lstsq(theta, rhs_flat.unsqueeze(1)).solution.squeeze(1)
    projected = {label: float(solution[i]) for i, label in enumerate(labels)}

    y_var = float(torch.var(target, correction=0))
    nmse = float(torch.mean((rhs_flat - target) ** 2)) / max(y_var, 1e-30)
    span_residual = rhs_flat - theta @ solution
    span_nmse = float(torch.mean(span_residual**2)) / max(y_var, 1e-30)




    drop_one_nmse: dict[str, float] = {}
    for label in labels:
        rest = [columns[other] for other in labels if other != label]
        if rest:
            reduced = torch.stack(rest, dim=1)
            drop_one_nmse[label] = _span_nmse(rhs_flat, reduced, y_var)
        else:
            drop_one_nmse[label] = float(torch.mean(rhs_flat**2)) / max(y_var, 1e-30)
    necessity_ok = all(value >= nmse_threshold for value in drop_one_nmse.values())

    coefficients_ok = all(
        abs(projected[label] - truth_coefficients[label])
        / abs(truth_coefficients[label])
        <= coeff_rtol
        for label in labels
    )

    return RecoveryVerdict(
        structure_ok=(
            math.isfinite(span_nmse) and span_nmse < nmse_threshold and necessity_ok
        ),
        coefficients_ok=coefficients_ok,
        nmse_ok=math.isfinite(nmse) and nmse < nmse_threshold,
        projected_coefficients=projected,
        span_nmse=span_nmse,
        drop_one_nmse=drop_one_nmse,
        nmse=nmse,
    )


def span_floor(
    u_t: Tensor,
    *,
    truth_columns: dict[str, Tensor],
) -> tuple[float, dict[str, float]]:
    labels = list(truth_columns.keys())
    target = u_t.flatten().double()
    theta = torch.stack(
        [truth_columns[label].flatten().double() for label in labels], dim=1
    )
    solution = torch.linalg.lstsq(theta, target.unsqueeze(1)).solution.squeeze(1)
    y_var = float(torch.var(target, correction=0))
    floor = float(torch.mean((target - theta @ solution) ** 2)) / max(y_var, 1e-30)
    return floor, {label: float(solution[i]) for i, label in enumerate(labels)}







def term_set_jaccard(actual: Iterable[str], truth: Iterable[str]) -> float:
    for name, value in (("actual", actual), ("truth", truth)):
        if isinstance(value, str):
            raise ValueError(
                f"{name} must be a collection of term labels, got the string "
                f"{value!r} (a bare str would be scored character by character)"
            )
    a = frozenset(actual)
    t = frozenset(truth)
    union = a | t
    if not union:
        return 1.0
    return len(a & t) / len(union)


def load_bearing_recall(verdict: RecoveryVerdict, *, nmse_threshold: float) -> float:
    if not (math.isfinite(verdict.span_nmse) and verdict.span_nmse < nmse_threshold):
        return 0.0
    margins = verdict.drop_one_nmse
    if not margins:
        return 1.0
    hits = sum(1 for value in margins.values() if value >= nmse_threshold)
    return hits / len(margins)
