"""Tests for numerical safety utilities."""

import pytest
import torch

from kd2.core.safety import safe_div, safe_exp, safe_log


class TestSafeDiv:
    """Tests for safe_div: protected division."""

    @pytest.mark.unit
    def test_normal_division(self) -> None:
        a = torch.tensor([6.0, 10.0])
        b = torch.tensor([2.0, 5.0])
        result = safe_div(a, b)
        torch.testing.assert_close(result, torch.tensor([3.0, 2.0]))

    @pytest.mark.unit
    def test_division_by_zero(self) -> None:
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([0.0, 0.0, 0.0])
        result = safe_div(a, b)
        assert torch.isfinite(result).all()

    @pytest.mark.unit
    def test_division_by_near_zero_positive(self) -> None:
        a = torch.tensor([1.0])
        b = torch.tensor([1e-15])
        result = safe_div(a, b)
        assert torch.isfinite(result).all()

    @pytest.mark.unit
    def test_division_by_near_zero_negative(self) -> None:
        a = torch.tensor([1.0])
        b = torch.tensor([-1e-15])
        result = safe_div(a, b)
        assert torch.isfinite(result).all()
        assert (result < 0).all()  # sign preserved

    @pytest.mark.unit
    def test_preserves_sign(self) -> None:
        a = torch.tensor([1.0, -1.0, 1.0, -1.0])
        b = torch.tensor([2.0, 2.0, -2.0, -2.0])
        result = safe_div(a, b)
        expected = torch.tensor([0.5, -0.5, -0.5, 0.5])
        torch.testing.assert_close(result, expected)

    @pytest.mark.numerical
    def test_no_nan_output(self) -> None:
        a = torch.randn(1000)
        b = torch.zeros(1000)
        result = safe_div(a, b)
        assert not torch.isnan(result).any()

    @pytest.mark.numerical
    def test_no_inf_for_normal_numerator(self) -> None:
        """safe_div prevents Inf when numerator is in normal range."""
        a = torch.tensor([1.0, 100.0, -50.0])
        b = torch.tensor([0.0, 0.0, 0.0])
        result = safe_div(a, b)
        assert torch.isfinite(result).all()

    @pytest.mark.unit
    def test_custom_eps(self) -> None:
        a = torch.tensor([1.0])
        b = torch.tensor([0.0])
        result_small = safe_div(a, b, eps=1e-15)
        result_large = safe_div(a, b, eps=1e-5)
        # Larger eps → smaller result magnitude
        assert result_large.abs() < result_small.abs()


class TestSafeExp:
    """Tests for safe_exp: clamped exponential."""

    @pytest.mark.unit
    def test_normal_values(self) -> None:
        x = torch.tensor([0.0, 1.0, -1.0])
        result = safe_exp(x)
        expected = torch.exp(x)
        torch.testing.assert_close(result, expected)

    @pytest.mark.unit
    def test_large_positive_clamped(self) -> None:
        x = torch.tensor([100.0, 1000.0, 1e10])
        result = safe_exp(x)
        assert torch.isfinite(result).all()

    @pytest.mark.unit
    def test_large_negative_ok(self) -> None:
        x = torch.tensor([-100.0, -1000.0])
        result = safe_exp(x)
        assert torch.isfinite(result).all()
        assert (result >= 0).all()

    @pytest.mark.unit
    def test_custom_max_val(self) -> None:
        x = torch.tensor([100.0])
        result = safe_exp(x, max_val=10.0)
        expected = torch.exp(torch.tensor([10.0]))
        torch.testing.assert_close(result, expected)

    @pytest.mark.numerical
    def test_no_nan_output(self) -> None:
        x = torch.randn(1000) * 100  # wide range
        result = safe_exp(x)
        assert not torch.isnan(result).any()

    @pytest.mark.numerical
    def test_no_inf_output(self) -> None:
        x = torch.tensor([1e10])
        result = safe_exp(x)
        assert torch.isfinite(result).all()


class TestSafeLog:
    """Tests for safe_log: clamped logarithm."""

    @pytest.mark.unit
    def test_normal_positive(self) -> None:
        x = torch.tensor([1.0, 2.718281828])
        result = safe_log(x)
        expected = torch.log(x)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    @pytest.mark.unit
    def test_zero_input(self) -> None:
        x = torch.tensor([0.0])
        result = safe_log(x)
        assert torch.isfinite(result).all()

    @pytest.mark.unit
    def test_negative_input(self) -> None:
        x = torch.tensor([-1.0, -100.0])
        result = safe_log(x)
        assert torch.isfinite(result).all()

    @pytest.mark.unit
    def test_very_small_positive(self) -> None:
        x = torch.tensor([1e-30])
        result = safe_log(x)
        assert torch.isfinite(result).all()

    @pytest.mark.unit
    def test_custom_eps(self) -> None:
        x = torch.tensor([0.0])
        result = safe_log(x, eps=1e-5)
        expected = torch.log(torch.tensor([1e-5]))
        torch.testing.assert_close(result, expected)

    @pytest.mark.numerical
    def test_no_nan_output(self) -> None:
        x = torch.randn(1000)  # includes negatives
        result = safe_log(x)
        assert not torch.isnan(result).any()

    @pytest.mark.numerical
    def test_no_inf_output(self) -> None:
        x = torch.tensor([0.0])
        result = safe_log(x)
        assert torch.isfinite(result).all()
