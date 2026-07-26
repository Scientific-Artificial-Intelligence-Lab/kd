
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kd.core.metrics import make_aic_scorer
from kd.search.pysindy.assembly import (
    build_native_result,
    render_best_expression,
    support_from_coefficients,
)


def test_support_uses_exact_zero_semantics_in_ascending_order() -> None:
    coefficients = np.array([0.0, 1e-300, -0.0, -2.5, 0.0])
    assert support_from_coefficients(coefficients) == [1, 3]


@pytest.mark.parametrize(
    ("support", "expected"),
    [
        pytest.param([2], "u_xx", id="single"),
        pytest.param([0, 2, 3], "add(u, add(u_xx, mul(u,u_x)))", id="folded"),
    ],
)
def test_render_best_expression_follows_support_order(
    support: list[int], expected: str
) -> None:
    terms = ["u", "u_x", "u_xx", "mul(u,u_x)"]
    assert render_best_expression(terms, support) == expected


def test_build_native_result_preserves_catalog_alignment_and_metrics() -> None:
    theta = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    lhs = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
    xi = np.array([2.0, 0.0, -1.0], dtype=np.float64)
    terms = ["u", "u_x", "u_xx"]

    result = build_native_result(
        theta=theta,
        lhs_flat=lhs,
        xi=xi,
        terms=terms,
        expression="add(u,u_xx)",
    )

    assert result.is_valid
    assert result.expression == "add(u,u_xx)"
    assert result.terms == terms
    assert result.selected_indices == [0, 2]
    assert result.complexity == 2
    assert result.coefficients is not None
    assert result.coefficients.dtype == theta.dtype
    assert result.coefficients.device == theta.device
    torch.testing.assert_close(
        result.coefficients,
        torch.tensor([2.0, 0.0, -1.0], dtype=theta.dtype),
    )
    assert result.residuals is not None
    torch.testing.assert_close(
        result.residuals,
        torch.tensor([-1.0, -2.0, 2.0], dtype=theta.dtype),
    )
    assert result.mse == pytest.approx(3.0)
    assert result.nmse == pytest.approx(13.5)
    assert result.r2 == pytest.approx(1.0 - result.nmse)
    assert result.score == pytest.approx(make_aic_scorer(3)(3.0, 2))


def test_native_result_detaches_stored_tensors() -> None:
    theta = torch.eye(2, dtype=torch.float64, requires_grad=True)
    lhs = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    result = build_native_result(
        theta=theta,
        lhs_flat=lhs,
        xi=np.array([1.0, 2.0]),
        terms=["u", "u_x"],
        expression="add(u,u_x)",
    )

    assert result.coefficients is not None
    assert result.coefficients.requires_grad is False
    assert result.residuals is not None
    assert result.residuals.requires_grad is False


def test_nonfinite_native_mse_fails_loud() -> None:
    theta = torch.tensor([[1e308]], dtype=torch.float64)
    lhs = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="MSE"):
        build_native_result(
            theta=theta,
            lhs_flat=lhs,
            xi=np.array([1e308]),
            terms=["u"],
            expression="u",
        )


def test_nmse_and_r2_identity_is_exact_under_population_variance() -> None:
    theta = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
    lhs = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)
    result = build_native_result(
        theta=theta,
        lhs_flat=lhs,
        xi=np.array([1.5]),
        terms=["u"],
        expression="u",
    )



    assert result.r2 == 1.0 - result.nmse


def test_cast_flush_to_zero_support_change_fails_loud() -> None:
    theta = torch.eye(3, dtype=torch.float32)
    lhs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    xi = np.array([2.0, 1e-50, 0.0], dtype=np.float64)

    with pytest.raises(RuntimeError, match="support"):
        build_native_result(
            theta=theta,
            lhs_flat=lhs,
            xi=xi,
            terms=["u", "u_x", "u_xx"],
            expression="add(u,u_x)",
        )


def test_large_magnitude_float32_target_keeps_honest_finite_nmse() -> None:
    theta = torch.tensor([[1.5e19], [-1.5e19], [1.0]], dtype=torch.float32)
    lhs = torch.tensor([1.5e19, -1.5e19, 0.0], dtype=torch.float32)
    xi = np.array([1.0], dtype=np.float64)

    result = build_native_result(
        theta=theta,
        lhs_flat=lhs,
        xi=xi,
        terms=["u"],
        expression="u",
    )




    assert math.isfinite(result.nmse)
    assert result.nmse > 0.0
