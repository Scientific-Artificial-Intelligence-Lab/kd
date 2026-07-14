
from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import kd.core.linear_solve._helpers as helpers
import kd.core.linear_solve.least_squares as least_squares_mod
import kd.core.linear_solve.stridge as stridge_mod
import kd.core.linear_solve.svd_null_space as svd_mod
from kd.core.linear_solve import (
    LeastSquaresSolver,
    STRidgeSolver,
    SVDNullSpaceSolver,
)
from kd.core.linear_solve._helpers import compute_r2, squared_residual
from kd.core.linear_solve.base import SolveResult, SparseSolver






_UPCAST_BINDINGS: tuple[str, ...] = (
    "kd.core.linear_solve._helpers.upcast_for_solve",
    "kd.core.linear_solve.least_squares.upcast_for_solve",
    "kd.core.linear_solve.stridge.upcast_for_solve",
    "kd.core.linear_solve.svd_null_space.upcast_for_solve",
)


class _UpcastRecorder:

    def __init__(self, real: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self._real = real
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(tensor.shape))
        return self._real(tensor)

    def theta_shaped_2d_calls(self, theta_shape: tuple[int, ...]) -> int:
        return sum(
            1 for s in self.shapes if len(s) == 2 and s == theta_shape
        )


def _patch_all_bindings(
    monkeypatch: pytest.MonkeyPatch, recorder: _UpcastRecorder
) -> None:
    for target in _UPCAST_BINDINGS:
        monkeypatch.setattr(target, recorder)


def _exact_system(
    n: int, d: int, dtype: torch.dtype, seed: int = 7
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    theta = torch.randn(n, d, generator=gen, dtype=dtype)
    coef = torch.randn(d, generator=gen, dtype=dtype)
    y = theta @ coef
    return theta, y







@pytest.mark.unit
@pytest.mark.parametrize(
    "solver",
    [
        LeastSquaresSolver(),
        STRidgeSolver(tol=0.0, normalize=2),
        SVDNullSpaceSolver(eps=1e-12),
    ],
    ids=["lstsq", "stridge", "svd_null_space"],
)
def test_theta_upcast_happens_exactly_once(
    solver: SparseSolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    theta, y = _exact_system(n=64, d=4, dtype=torch.float64)
    recorder = _UpcastRecorder(helpers.upcast_for_solve)
    _patch_all_bindings(monkeypatch, recorder)

    result = solver.solve(theta, y)
    assert result.is_valid, "solver must be on its normal path for this lock"

    theta_upcasts = recorder.theta_shaped_2d_calls(tuple(theta.shape))
    assert theta_upcasts == 1, (
        f"expected exactly 1 theta-shaped float64 upcast, got {theta_upcasts} "
        f"(recorded shapes: {recorder.shapes})"
    )


@pytest.mark.unit
def test_module_bindings_are_all_patchable() -> None:
    assert least_squares_mod.upcast_for_solve is helpers.upcast_for_solve
    assert stridge_mod.upcast_for_solve is helpers.upcast_for_solve
    assert svd_mod.upcast_for_solve is helpers.upcast_for_solve







def _zero_column_system(
    n: int, d: int, dtype: torch.dtype, zero_col: int, seed: int = 11
) -> tuple[torch.Tensor, torch.Tensor]:
    theta, y = _exact_system(n, d, dtype, seed=seed)
    theta = theta.clone()
    theta[:, zero_col] = 0.0


    gen = torch.Generator().manual_seed(seed + 1)
    coef = torch.randn(d, generator=gen, dtype=dtype)
    coef[zero_col] = 0.0
    y = theta @ coef
    return theta, y



def _parity_cases() -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    cases: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    theta32, y32 = _exact_system(n=80, d=5, dtype=torch.float32)
    cases.append(("f32_overdet_y1d", theta32, y32))

    theta64, y64 = _exact_system(n=80, d=5, dtype=torch.float64, seed=3)
    cases.append(("f64_overdet_ycol", theta64, y64.unsqueeze(1)))

    theta_fat, y_fat = _exact_system(n=6, d=10, dtype=torch.float64, seed=5)
    cases.append(("f64_underdet_y1d", theta_fat, y_fat))

    theta_z, y_z = _zero_column_system(n=80, d=6, dtype=torch.float64, zero_col=2)
    cases.append(("f64_zerocol_y1d", theta_z, y_z))

    return cases


def _solvers() -> list[tuple[str, SparseSolver]]:
    return [
        ("lstsq", LeastSquaresSolver()),
        ("stridge", STRidgeSolver(tol=0.0, normalize=2)),
        ("svd_null_space", SVDNullSpaceSolver(eps=1e-12)),
    ]


@pytest.mark.unit
@pytest.mark.numerical
@pytest.mark.parametrize(
    "case",
    _parity_cases(),
    ids=[c[0] for c in _parity_cases()],
)
@pytest.mark.parametrize(
    "solver_entry",
    _solvers(),
    ids=[s[0] for s in _solvers()],
)
def test_residual_and_r2_are_bitwise_equal_to_public_helpers(
    case: tuple[str, torch.Tensor, torch.Tensor],
    solver_entry: tuple[str, SparseSolver],
) -> None:
    _label, theta, y = case
    _sid, solver = solver_entry
    y_1d = y.squeeze(-1) if y.dim() == 2 else y

    result: SolveResult = solver.solve(theta, y)

    if not result.is_valid:
        pytest.skip(f"{_sid} returned invalid result for {_label}; parity N/A")

    expected_residual = squared_residual(theta, result.coefficients, y_1d)
    expected_r2 = compute_r2(theta, result.coefficients, y_1d)

    assert result.residual == expected_residual, (
        f"{_sid}/{_label}: residual {result.residual!r} != "
        f"squared_residual oracle {expected_residual!r}"
    )
    assert result.r2 == expected_r2, (
        f"{_sid}/{_label}: r2 {result.r2!r} != compute_r2 oracle {expected_r2!r}"
    )


@pytest.mark.unit
def test_canonical_overdetermined_case_is_valid_for_every_solver() -> None:
    theta, y = _exact_system(n=80, d=5, dtype=torch.float64, seed=3)
    for sid, solver in _solvers():
        result = solver.solve(theta, y)
        assert result.is_valid, f"{sid} unexpectedly invalid on canonical system"


@pytest.mark.unit
@pytest.mark.numerical
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["f32", "f64"]
)
def test_stridge_all_zero_theta_parity(dtype: torch.dtype) -> None:
    n, d = 32, 3
    theta = torch.zeros(n, d, dtype=dtype)
    gen = torch.Generator().manual_seed(17)
    y = torch.randn(n, generator=gen, dtype=dtype)

    solver = STRidgeSolver(tol=0.0, normalize=2)
    result = solver.solve(theta, y)

    assert result.is_valid
    assert torch.equal(result.coefficients, torch.zeros(d, dtype=dtype))
    assert result.selected_indices == []

    expected_residual = squared_residual(theta, result.coefficients, y)
    expected_r2 = compute_r2(theta, result.coefficients, y)
    assert result.residual == expected_residual, (
        f"all-zero residual {result.residual!r} != oracle {expected_residual!r}"
    )
    assert result.r2 == expected_r2, (
        f"all-zero r2 {result.r2!r} != oracle {expected_r2!r}"
    )
