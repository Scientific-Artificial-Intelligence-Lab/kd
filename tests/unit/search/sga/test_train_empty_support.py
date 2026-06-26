
from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from kd.search.sga.config import SGAConfig
from kd.search.sga.train import _count_active, train_sweep




_N_SAMPLES = 200
_DTYPE = torch.float64


def _sweep_config(**overrides: object) -> SGAConfig:
    base: dict[str, object] = {
        "normalize": 2,
        "d_tol": 0.5,
        "maxit": 10,
        "str_iters": 10,
        "lam": 0.0,
    }
    base.update(overrides)
    return SGAConfig(**base)


def _zero_norm_theta() -> tuple[Tensor, Tensor]:
    theta = torch.zeros(_N_SAMPLES, 2, dtype=_DTYPE)
    gen = torch.Generator().manual_seed(20260612)
    y = torch.randn(_N_SAMPLES, dtype=_DTYPE, generator=gen)
    return theta, y


def _giant_safe_div_theta() -> tuple[Tensor, Tensor]:
    gen = torch.Generator().manual_seed(20260612)
    giant = torch.randn(_N_SAMPLES, dtype=_DTYPE, generator=gen) * 1e12
    signal = torch.randn(_N_SAMPLES, dtype=_DTYPE, generator=gen)
    g_unit = giant / giant.norm()

    y = signal - (signal @ g_unit) * g_unit
    theta = giant.unsqueeze(1)
    return theta, y









class TestFixturesTriggerEmptySupportToday:

    @pytest.mark.unit
    def test_zero_norm_theta_empties_support(self) -> None:
        theta, y = _zero_norm_theta()
        result = train_sweep(theta, y, _sweep_config())
        assert _count_active(result.coefficients) == 0, (
            "zero-norm theta must drive _count_active to 0 (else the empty-"
            "support contract test below would be vacuous)."
        )

    @pytest.mark.unit
    def test_giant_safe_div_theta_empties_support(self) -> None:
        theta, y = _giant_safe_div_theta()
        result = train_sweep(theta, y, _sweep_config())
        assert _count_active(result.coefficients) == 0, (
            "giant safe_div theta must drive _count_active to 0 (else the "
            "accident-reconstruction test below would be vacuous)."
        )







class TestEmptySupportIsInvalid:

    @pytest.mark.unit
    def test_zero_norm_empty_support_aic_is_inf(self) -> None:
        theta, y = _zero_norm_theta()
        result = train_sweep(theta, y, _sweep_config())
        assert result.selected_indices == [], (
            f"empty support must report selected_indices == []; got "
            f"{result.selected_indices}"
        )
        assert result.aic_score == float("inf"), (
            f"a fit with empty support must be invalid (aic == +inf); got "
            f"{result.aic_score} — the AIC empty-support back door (k=0 has no "
            f"complexity penalty) is still open."
        )

    @pytest.mark.unit
    def test_multicolumn_giant_empty_support_is_inf(self) -> None:
        gen = torch.Generator().manual_seed(20260612)


        giant = torch.randn(_N_SAMPLES, 2, dtype=_DTYPE, generator=gen) * 1e12
        signal = torch.randn(_N_SAMPLES, dtype=_DTYPE, generator=gen)
        q, _ = torch.linalg.qr(giant)
        y = signal - q @ (q.T @ signal)
        config = _sweep_config()

        result = train_sweep(giant, y, config)

        assert _count_active(result.coefficients) == 0, (
            "precondition: multi-column giant theta must clear the support."
        )
        assert result.selected_indices == []
        assert result.aic_score == float("inf"), (
            f"multi-column giant empty support must be invalid; got {result.aic_score}"
        )

    @pytest.mark.unit
    def test_invalid_empty_support_coeffs_are_all_zero(self) -> None:
        theta, y = _zero_norm_theta()
        result = train_sweep(theta, y, _sweep_config())
        assert _count_active(result.coefficients) == 0
        assert not result.selected_indices



        assert torch.equal(result.coefficients, torch.zeros_like(result.coefficients))







class TestEmptySupportNeverOutranksValid:

    @pytest.mark.unit
    def test_giant_safe_div_empty_support_is_invalid(self) -> None:
        theta, y = _giant_safe_div_theta()
        result = train_sweep(theta, y, _sweep_config())
        assert result.selected_indices == []
        assert result.aic_score == float("inf"), (
            f"safe_div empty-support candidate must be invalid; got {result.aic_score}"
        )

    @pytest.mark.unit
    def test_empty_support_aic_exceeds_valid_candidate_aic(self) -> None:
        config = _sweep_config()




        gen = torch.Generator().manual_seed(20260612)
        giant = torch.randn(_N_SAMPLES, dtype=_DTYPE, generator=gen) * 1e12
        signal = torch.randn(_N_SAMPLES, dtype=_DTYPE, generator=gen)
        g_unit = giant / giant.norm()
        empty_y = 0.05 * (signal - (signal @ g_unit) * g_unit)
        empty = train_sweep(giant.unsqueeze(1), empty_y, config)



        gen_v = torch.Generator().manual_seed(3)
        u = torch.randn(_N_SAMPLES, dtype=_DTYPE, generator=gen_v)
        y_valid = 0.5 * u + 1.5 * torch.randn(_N_SAMPLES, dtype=_DTYPE, generator=gen_v)
        valid = train_sweep(u.unsqueeze(1), y_valid, config)


        assert valid.selected_indices, "valid competitor must keep a support"
        assert math.isfinite(valid.aic_score), "valid competitor must have a finite aic"


        assert empty.selected_indices == [], "empty candidate must have no support"

        assert empty.aic_score > valid.aic_score, (
            f"empty-support aic ({empty.aic_score}) must NOT out-rank the "
            f"mediocre valid candidate aic ({valid.aic_score}); AIC is "
            f"minimized, so the empty-support back door currently inverts this "
            f"ordering (the empty candidate wins today)."
        )







class TestDefaultOnlyStaysValid:

    @pytest.mark.unit
    def test_default_only_fit_keeps_nonempty_support(self) -> None:
        gen = torch.Generator().manual_seed(7)


        default_col = torch.randn(_N_SAMPLES, 1, dtype=_DTYPE, generator=gen)
        y = 1.7 * default_col.squeeze(-1)

        result = train_sweep(default_col, y, _sweep_config())
        assert result.selected_indices == [0], (
            f"default-only fit must select the default column (non-empty "
            f"support); got {result.selected_indices}"
        )
        assert math.isfinite(result.aic_score), (
            "a default-only fit is a legitimate equation (u_t = c*u) and must "
            "keep a finite aic — the empty-support rule must not touch it."
        )
        assert _count_active(result.coefficients) == 1
