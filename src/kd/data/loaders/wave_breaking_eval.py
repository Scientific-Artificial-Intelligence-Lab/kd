
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from kd.data.loaders.wave_breaking import WaveBreakingCase
from kd.models.field_model import FieldModel

if TYPE_CHECKING:
    from kd.core.evaluator import Evaluator
    from kd.data.schema import PDEDataset






X_WINDOWS: tuple[tuple[float, float], ...] = (
    (8.18, 9.34),
    (9.77, 10.93),
    (11.41, 12.57),
)
POINTS_PER_WINDOW: int = 100


X_STAR_OFFSET: float = 8.17


T_START: float = 0.15
T_END_MARGIN: float = 0.1
T_STEP: float = 0.05



RHS_TERMS: tuple[str, ...] = ("u_x", "u_xxx", "diff2_x(mul(u, u_x))")


LHS_ORDER: int = 1







@dataclass
class WaveBreakingFit:

    c1: float
    c2: float
    c3: float
    r_squared: float







def wave_breaking_star_grid(
    case: WaveBreakingCase, *, points_per_window: int = POINTS_PER_WINDOW
) -> tuple[Tensor, Tensor]:
    x_physical = torch.cat(
        [
            torch.linspace(lo, hi, points_per_window, dtype=torch.float64)
            for lo, hi in X_WINDOWS
        ]
    )
    x_star = (x_physical - X_STAR_OFFSET) / case.lamda

    t_end = float(case.t.detach().to(dtype=torch.float64).max().item()) - T_END_MARGIN
    if t_end <= T_START:
        raise ValueError(
            f"wave-breaking starred time range is empty: start={T_START}, "
            f"end={t_end}."
        )
    t_physical = torch.arange(T_START, t_end, T_STEP, dtype=torch.float64)
    if t_physical.numel() == 0:
        raise ValueError(
            f"wave-breaking starred time range is empty: start={T_START}, "
            f"end={t_end}, step={T_STEP}."
        )
    t_star = t_physical / case.tp_seconds
    return x_star, t_star







def evaluate_known_terms(
    surrogate: FieldModel,
    case: WaveBreakingCase,
) -> WaveBreakingFit:
    _validate_surrogate_layout(surrogate)
    device, dtype = _surrogate_device_dtype(surrogate)
    x_star, t_star = wave_breaking_star_grid(case)
    dataset = _build_eval_dataset(
        t_star=t_star.to(device=device, dtype=torch.float64),
        x_star=x_star.to(device=device, dtype=torch.float64),
        query_dtype=dtype,
    )
    from kd.core.platform.builder import PlatformBuilder
    from kd.core.platform.requirements import DerivativeReqs

    reqs = DerivativeReqs(
        provider_kind="autograd",
        surrogate_model=surrogate,
        max_atomic_order=3,
        lhs_order=LHS_ORDER,
        needs_surrogate=True,
    )
    components = PlatformBuilder(dataset, reqs).build()
    evaluator = components.evaluator
    if evaluator is None:
        raise ValueError("PlatformBuilder did not provide an Evaluator.")

    theta = _execute_rhs_terms(evaluator, list(RHS_TERMS))
    lhs = evaluator.lhs.to(device=theta.device, dtype=theta.dtype)
    solve = evaluator.solver.solve(theta, lhs)
    if not solve.is_valid or solve.coefficients is None:
        raise ValueError(solve.error_message or "wave-breaking least-squares failed.")
    if solve.coefficients.numel() != len(RHS_TERMS):
        raise ValueError(
            f"expected {len(RHS_TERMS)} coefficients, got "
            f"{solve.coefficients.numel()}."
        )
    coeffs = solve.coefficients.detach().flatten().cpu()
    return WaveBreakingFit(
        c1=-float(coeffs[0].item()),
        c2=-float(coeffs[1].item()),
        c3=-float(coeffs[2].item()),
        r_squared=float(solve.r2),
    )


def _validate_surrogate_layout(surrogate: FieldModel) -> None:
    if surrogate.coord_names != ["t", "x"]:
        raise ValueError(
            "wave-breaking surrogate must use coord_names=['t', 'x'], got "
            f"{surrogate.coord_names}."
        )
    if surrogate.field_names != ["u"]:
        raise ValueError(
            "wave-breaking surrogate must use field_names=['u'], got "
            f"{surrogate.field_names}."
        )


def _surrogate_device_dtype(surrogate: FieldModel) -> tuple[torch.device, torch.dtype]:
    try:
        param = next(surrogate.parameters())
    except StopIteration as exc:
        raise ValueError("wave-breaking surrogate has no parameters.") from exc
    return param.device, param.dtype


def _build_eval_dataset(
    *,
    t_star: Tensor,
    x_star: Tensor,
    query_dtype: torch.dtype,
) -> PDEDataset:
    from kd.data.schema import PDEDataset

    t_query = t_star.to(dtype=query_dtype)
    x_query = x_star.to(dtype=query_dtype)
    values = torch.zeros(
        (t_query.numel(), x_query.numel()),
        dtype=query_dtype,
        device=t_query.device,
    )
    return PDEDataset.from_arrays(
        coords={"t": t_query, "x": x_query},
        fields={"u": values},
        lhs="u_t",
        name="wave-breaking-star-grid",
        dtype=query_dtype,
    )


def _execute_rhs_terms(evaluator: Evaluator, terms: list[str]) -> Tensor:
    columns: list[Tensor] = []
    with torch.enable_grad():
        for term in terms:
            value = evaluator.executor.execute(term, evaluator.context).value.flatten()
            if not torch.isfinite(value).all():
                raise ValueError(f"wave-breaking term '{term}' contains NaN or Inf.")
            columns.append(value)
    theta = torch.stack(columns, dim=1).detach()
    if theta.numel() == 0:
        raise ValueError("wave-breaking RHS theta matrix is empty.")
    return theta


__all__ = [
    "LHS_ORDER",
    "POINTS_PER_WINDOW",
    "RHS_TERMS",
    "T_END_MARGIN",
    "T_START",
    "T_STEP",
    "X_STAR_OFFSET",
    "X_WINDOWS",
    "WaveBreakingFit",
    "evaluate_known_terms",
    "wave_breaking_star_grid",
]
