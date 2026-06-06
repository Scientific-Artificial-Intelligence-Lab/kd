
import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from kd.core.safety import safe_div, safe_exp, safe_log


finite_floats = st.floats(
    min_value=-1e10, max_value=1e10,
    allow_nan=False, allow_infinity=False,
)


positive_floats = st.floats(
    min_value=1e-10, max_value=1e10,
    allow_nan=False, allow_infinity=False,
)


nonzero_floats = st.floats(
    min_value=-1e10, max_value=1e10,
    allow_nan=False, allow_infinity=False,
).filter(lambda x: abs(x) > 1e-15)


@pytest.mark.numerical
class TestSafeDivProperties:

    @given(a=finite_floats, b=finite_floats)
    @settings(max_examples=500)
    def test_finite_input_finite_output(self, a: float, b: float) -> None:
        result = safe_div(torch.tensor(a), torch.tensor(b))
        assert torch.isfinite(result).all()

    @given(a=positive_floats, b=positive_floats)
    @settings(max_examples=200)
    def test_positive_div_positive_is_positive(self, a: float, b: float) -> None:
        result = safe_div(torch.tensor(a), torch.tensor(b))
        assert result.item() > 0

    @given(a=finite_floats, b=nonzero_floats)
    @settings(max_examples=200)
    def test_approximate_inverse(self, a: float, b: float) -> None:
        assume(abs(b) > 1e-3)
        result = safe_div(torch.tensor(a), torch.tensor(b))
        reconstructed = result * torch.tensor(b)
        torch.testing.assert_close(
            reconstructed, torch.tensor(a), rtol=1e-4, atol=1e-6,
        )

    @given(a=finite_floats)
    @settings(max_examples=200)
    def test_div_by_one(self, a: float) -> None:
        result = safe_div(torch.tensor(a), torch.tensor(1.0))
        torch.testing.assert_close(result, torch.tensor(a), rtol=1e-6, atol=1e-8)


@pytest.mark.numerical
class TestSafeExpProperties:

    @given(x=finite_floats)
    @settings(max_examples=500)
    def test_always_positive_finite(self, x: float) -> None:
        result = safe_exp(torch.tensor(x))
        assert result.item() > 0
        assert torch.isfinite(result).all()

    @given(x1=finite_floats, x2=finite_floats)
    @settings(max_examples=200)
    def test_monotonicity(self, x1: float, x2: float) -> None:
        assume(x1 < x2)
        r1 = safe_exp(torch.tensor(x1))
        r2 = safe_exp(torch.tensor(x2))
        assert r1.item() <= r2.item()

    @given(x=st.floats(min_value=-40.0, max_value=40.0))
    @settings(max_examples=200)
    def test_matches_torch_exp_in_safe_range(self, x: float) -> None:
        t = torch.tensor(x)
        torch.testing.assert_close(safe_exp(t), torch.exp(t))


@pytest.mark.numerical
class TestSafeLogProperties:

    @given(x=finite_floats)
    @settings(max_examples=500)
    def test_finite_input_finite_output(self, x: float) -> None:
        result = safe_log(torch.tensor(x))
        assert torch.isfinite(result).all()

    @given(x1=positive_floats, x2=positive_floats)
    @settings(max_examples=200)
    def test_monotonicity_positive(self, x1: float, x2: float) -> None:
        assume(x1 < x2)
        assume(x1 > 1e-8)
        assume(abs(x2 - x1) > x1 * 1e-6)
        r1 = safe_log(torch.tensor(x1))
        r2 = safe_log(torch.tensor(x2))
        assert r1.item() < r2.item()

    @given(x=st.floats(min_value=1e-5, max_value=1e8))
    @settings(max_examples=200)
    def test_matches_torch_log_for_positive(self, x: float) -> None:
        t = torch.tensor(x)
        torch.testing.assert_close(
            safe_log(t), torch.log(t), rtol=1e-5, atol=1e-8,
        )

    @given(x=st.floats(min_value=0.01, max_value=40.0))
    @settings(max_examples=200)
    def test_exp_log_roundtrip(self, x: float) -> None:
        t = torch.tensor(x)
        roundtrip = safe_log(safe_exp(t))
        torch.testing.assert_close(roundtrip, t, rtol=1e-4, atol=1e-6)
