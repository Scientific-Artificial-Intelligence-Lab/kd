
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from kd.data.loaders import wave_breaking_eval as wbe
from kd.data.loaders.wave_breaking import WaveBreakingCase
from kd.models.field_model import FieldModel






_SPEC_X_WINDOWS = ((8.18, 9.34), (9.77, 10.93), (11.41, 12.57))
_SPEC_POINTS_PER_WINDOW = 100
_SPEC_X_OFFSET = 8.17
_SPEC_T_START = 0.15
_SPEC_T_END_MARGIN = 0.1
_SPEC_T_STEP = 0.05
_SPEC_GRAVITY = 9.81


def _expected_lamda(tp_seconds: float) -> float:
    return _SPEC_GRAVITY * tp_seconds**2 / (2.0 * math.pi)


def _expected_x_star(lamda: float) -> np.ndarray:
    windows = [
        np.linspace(lo, hi, _SPEC_POINTS_PER_WINDOW) for lo, hi in _SPEC_X_WINDOWS
    ]
    x_phys = np.concatenate(windows)
    return (x_phys - _SPEC_X_OFFSET) / lamda


def _expected_t_star(t_max: float, tp_seconds: float) -> np.ndarray:
    t_phys = np.arange(_SPEC_T_START, t_max - _SPEC_T_END_MARGIN, _SPEC_T_STEP)
    return t_phys / tp_seconds


def _make_synthetic_case(
    *,
    t_max: float = 5.28,
    tp_seconds: float = 1.2,
    n: int = 64,
) -> WaveBreakingCase:
    t = torch.linspace(0.05, t_max, n, dtype=torch.float64)
    x = torch.linspace(8.15, 12.55, n, dtype=torch.float64)
    eta = torch.zeros(n, dtype=torch.float64)
    return WaveBreakingCase(
        name="N_G2Tp12A080_broad",
        t=t,
        x=x,
        eta=eta,
        g=2,
        tp_seconds=tp_seconds,
        a=80,
        lamda=_expected_lamda(tp_seconds),
        prefix="N",
    )









_KDV_C = 0.7
_KDV_D = 0.05
_KDV_K1 = 1.0
_KDV_K2 = 2.0


def _build_kdv_field_model() -> FieldModel:
    w1 = _KDV_C * _KDV_K1 - _KDV_D * _KDV_K1**3
    w2 = _KDV_C * _KDV_K2 - _KDV_D * _KDV_K2**3
    model = FieldModel(
        coord_names=["t", "x"],
        field_names=["u"],
        hidden_sizes=[2],
        activation="sin",
    ).double()
    lin0 = model.trunk[0]
    assert isinstance(lin0, nn.Linear)
    with torch.no_grad():

        lin0.weight.copy_(
            torch.tensor([[-w1, _KDV_K1], [-w2, _KDV_K2]], dtype=torch.float64)
        )
        lin0.bias.zero_()
        model.head.weight.copy_(torch.tensor([[1.0, 1.0]], dtype=torch.float64))
        model.head.bias.zero_()
    return model







class TestStarGrid:

    @pytest.mark.smoke
    def test_star_grid_callable(self) -> None:
        assert callable(wbe.wave_breaking_star_grid)

    @pytest.mark.unit
    def test_returns_two_1d_tensors(self) -> None:
        x_star, t_star = wbe.wave_breaking_star_grid(_make_synthetic_case())
        assert x_star.dim() == 1
        assert t_star.dim() == 1

    @pytest.mark.unit
    def test_x_star_has_300_points(self) -> None:
        x_star, _ = wbe.wave_breaking_star_grid(_make_synthetic_case())
        assert x_star.numel() == 3 * _SPEC_POINTS_PER_WINDOW

    @pytest.mark.numerical
    def test_x_star_is_float64(self) -> None:
        x_star, t_star = wbe.wave_breaking_star_grid(_make_synthetic_case())
        assert x_star.dtype == torch.float64
        assert t_star.dtype == torch.float64

    @pytest.mark.numerical
    def test_x_star_matches_frozen_formula(self) -> None:
        case = _make_synthetic_case()
        x_star, _ = wbe.wave_breaking_star_grid(case)
        expected = torch.tensor(_expected_x_star(case.lamda))
        torch.testing.assert_close(
            x_star.to(torch.float64), expected, rtol=1e-6, atol=1e-9
        )

    @pytest.mark.numerical
    def test_t_star_matches_frozen_formula(self) -> None:
        case = _make_synthetic_case(t_max=5.28, tp_seconds=1.2)
        _, t_star = wbe.wave_breaking_star_grid(case)
        t_max = float(case.t.max())
        expected = torch.tensor(_expected_t_star(t_max, case.tp_seconds))
        assert t_star.numel() == expected.numel()
        torch.testing.assert_close(
            t_star.to(torch.float64), expected, rtol=1e-6, atol=1e-9
        )

    @pytest.mark.numerical
    def test_x_star_windows_have_gaps(self) -> None:
        case = _make_synthetic_case()
        x_star, _ = wbe.wave_breaking_star_grid(case)
        diffs = (x_star[1:] - x_star[:-1]).to(torch.float64)
        intra = diffs[:99].abs().max().item()
        boundary_jump = diffs[99].item()
        assert boundary_jump > 3.0 * intra

    @pytest.mark.numerical
    def test_empty_time_range_raises(self) -> None:

        case = _make_synthetic_case(t_max=0.2, n=8)
        with pytest.raises(ValueError):
            wbe.wave_breaking_star_grid(case)







class TestEvaluateKnownTerms:

    @pytest.mark.smoke
    def test_callable(self) -> None:
        assert callable(wbe.evaluate_known_terms)

    @pytest.mark.numerical
    def test_recovers_closed_form_kdv_coefficients(self) -> None:
        model = _build_kdv_field_model()
        case = _make_synthetic_case()
        fit = wbe.evaluate_known_terms(model, case)
        assert fit.c1 == pytest.approx(_KDV_C, rel=1e-3)
        assert fit.c2 == pytest.approx(_KDV_D, rel=1e-3)
        assert abs(fit.c3) < 1e-3
        assert fit.r_squared == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.numerical
    def test_fit_fields_all_finite(self) -> None:
        model = _build_kdv_field_model()
        fit = wbe.evaluate_known_terms(model, _make_synthetic_case())
        for value in (fit.c1, fit.c2, fit.c3, fit.r_squared):
            assert math.isfinite(value)

    @pytest.mark.unit
    def test_eval_grid_cardinality(self) -> None:
        case = _make_synthetic_case()
        x_star, t_star = wbe.wave_breaking_star_grid(case)
        n_grid = x_star.numel() * t_star.numel()
        assert n_grid == 3 * _SPEC_POINTS_PER_WINDOW * t_star.numel()

    @pytest.mark.unit
    def test_coord_layout_mismatch_raises(self) -> None:
        wrong = FieldModel(
            coord_names=["a", "b"],
            field_names=["u"],
            hidden_sizes=[2],
            activation="sin",
        ).double()
        with pytest.raises(ValueError):
            wbe.evaluate_known_terms(wrong, _make_synthetic_case())

    @pytest.mark.numerical
    def test_empty_time_range_propagates(self) -> None:
        model = _build_kdv_field_model()
        case = _make_synthetic_case(t_max=0.2, n=8)
        with pytest.raises(ValueError):
            wbe.evaluate_known_terms(model, case)







class TestEvaluateKnownTermsDevice:

    @pytest.mark.numerical
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_surrogate_cpu_case(self) -> None:
        model = _build_kdv_field_model().to("cuda")
        case = _make_synthetic_case()
        fit = wbe.evaluate_known_terms(model, case)
        for value in (fit.c1, fit.c2, fit.c3, fit.r_squared):
            assert math.isfinite(value)
