
from __future__ import annotations

import importlib

import pytest
import torch




import kd.search.discover.tokens.prior
from kd.core.evaluator import EvaluationResult








REF_MIN = 5e-5
REF_MAX = 1e4


def _make_result(
    coefficients: torch.Tensor | None,
    *,
    nmse: float = 0.04,
    complexity: int = 2,
    is_valid: bool = True,
) -> EvaluationResult:
    return EvaluationResult(
        mse=nmse,
        nmse=nmse,
        r2=1.0 - nmse,
        complexity=complexity,
        coefficients=coefficients,
        is_valid=is_valid,
        terms=["a", "b"][: (0 if coefficients is None else len(coefficients))],
    )







class TestApplyMagnitudeFilterCore:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_module_exposes_filter_and_constants(self) -> None:
        mod = importlib.import_module("kd.search.discover.evaluation.magnitude")
        assert hasattr(mod, "apply_magnitude_filter")
        assert hasattr(mod, "MAGNITUDE_FILTER_MIN")
        assert hasattr(mod, "MAGNITUDE_FILTER_MAX")

    @pytest.mark.unit
    def test_thresholds_match_reference_bounds(self) -> None:
        mod = importlib.import_module("kd.search.discover.evaluation.magnitude")
        assert pytest.approx(REF_MIN) == mod.MAGNITUDE_FILTER_MIN
        assert pytest.approx(REF_MAX) == mod.MAGNITUDE_FILTER_MAX

    @pytest.mark.unit
    def test_healthy_unit_coefficients_pass_unchanged(self) -> None:
        from kd.search.discover.evaluation.magnitude import apply_magnitude_filter

        coefs = torch.tensor([0.1, 1.0], dtype=torch.float64)
        result = _make_result(coefs)
        gated = apply_magnitude_filter(result)
        assert gated.is_valid

    @pytest.mark.unit
    def test_tiny_coefficient_below_min_becomes_invalid(self) -> None:
        from kd.search.discover.evaluation.magnitude import apply_magnitude_filter

        coefs = torch.tensor([1.0, 1e-6], dtype=torch.float64)
        gated = apply_magnitude_filter(_make_result(coefs))
        assert not gated.is_valid
        assert "magnitude" in (gated.error_message or "").lower()

    @pytest.mark.unit
    def test_huge_coefficient_above_max_becomes_invalid(self) -> None:
        from kd.search.discover.evaluation.magnitude import apply_magnitude_filter

        coefs = torch.tensor([1.0, 2e4], dtype=torch.float64)
        gated = apply_magnitude_filter(_make_result(coefs))
        assert not gated.is_valid
        assert "magnitude" in (gated.error_message or "").lower()

    @pytest.mark.unit
    def test_td090_blowup_coefficients_rejected(self) -> None:
        from kd.search.discover.evaluation.magnitude import apply_magnitude_filter

        coefs = torch.tensor([16649.0, -16648.0], dtype=torch.float64)
        gated = apply_magnitude_filter(_make_result(coefs))
        assert not gated.is_valid

    @pytest.mark.unit
    def test_negative_coefficients_use_absolute_value(self) -> None:
        from kd.search.discover.evaluation.magnitude import apply_magnitude_filter

        coefs = torch.tensor([-5e4, -0.5], dtype=torch.float64)
        gated = apply_magnitude_filter(_make_result(coefs))
        assert not gated.is_valid







class TestMagnitudeBoundarySemantics:

    @pytest.mark.unit
    def test_coefficient_exactly_at_min_is_kept(self) -> None:
        from kd.search.discover.evaluation.magnitude import (
            MAGNITUDE_FILTER_MIN,
            apply_magnitude_filter,
        )

        coefs = torch.tensor([MAGNITUDE_FILTER_MIN, 1.0], dtype=torch.float64)
        gated = apply_magnitude_filter(_make_result(coefs))
        assert gated.is_valid

    @pytest.mark.unit
    def test_coefficient_exactly_at_max_is_kept(self) -> None:
        from kd.search.discover.evaluation.magnitude import (
            MAGNITUDE_FILTER_MAX,
            apply_magnitude_filter,
        )

        coefs = torch.tensor([MAGNITUDE_FILTER_MAX, 1.0], dtype=torch.float64)
        gated = apply_magnitude_filter(_make_result(coefs))
        assert gated.is_valid

    @pytest.mark.unit
    def test_just_below_min_trips_just_at_min_does_not(self) -> None:
        from kd.search.discover.evaluation.magnitude import (
            MAGNITUDE_FILTER_MIN,
            apply_magnitude_filter,
        )

        at = torch.tensor([MAGNITUDE_FILTER_MIN, 1.0], dtype=torch.float64)
        below = torch.tensor([MAGNITUDE_FILTER_MIN * 0.99, 1.0], dtype=torch.float64)
        assert apply_magnitude_filter(_make_result(at)).is_valid
        assert not apply_magnitude_filter(_make_result(below)).is_valid







class TestMagnitudeFilterRobustness:

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_already_invalid_result_passes_through_unchanged(self) -> None:
        from kd.search.discover.evaluation.magnitude import apply_magnitude_filter

        bad = EvaluationResult(
            mse=1e10,
            nmse=1e10,
            r2=-float("inf"),
            complexity=0,
            coefficients=None,
            is_valid=False,
            error_message="upstream failure",
        )
        gated = apply_magnitude_filter(bad)
        assert not gated.is_valid

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_none_coefficients_do_not_crash(self) -> None:
        from kd.search.discover.evaluation.magnitude import apply_magnitude_filter

        result = _make_result(None)
        gated = apply_magnitude_filter(result)
        assert gated is not None

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_empty_coefficients_do_not_crash(self) -> None:
        from kd.search.discover.evaluation.magnitude import apply_magnitude_filter

        result = _make_result(torch.empty(0, dtype=torch.float64))
        gated = apply_magnitude_filter(result)
        assert gated is not None

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_nonfinite_coefficient_is_rejected(self) -> None:
        from kd.search.discover.evaluation.magnitude import apply_magnitude_filter

        coefs = torch.tensor([1.0, float("inf")], dtype=torch.float64)
        gated = apply_magnitude_filter(_make_result(coefs))
        assert not gated.is_valid







class TestSingleSourceOfTruth:

    @pytest.mark.unit
    def test_sampled_evaluator_imports_shared_predicate(self) -> None:
        from kd.search.discover.evaluation.magnitude import magnitude_reject_reason
        from kd.search.discover.runners import sampled_evaluator

        assert sampled_evaluator.magnitude_reject_reason is magnitude_reject_reason

    @pytest.mark.unit
    def test_runner_gate_rejects_pure_nan_like_main_path(self) -> None:
        import torch as _torch

        from kd.search.discover.evaluation.magnitude import (
            apply_magnitude_filter,
            magnitude_reject_reason,
        )

        nan_coef = _torch.tensor([float("nan"), float("nan")], dtype=torch.float64)

        assert magnitude_reject_reason(nan_coef) is not None

        gated = apply_magnitude_filter(_make_result(nan_coef))
        assert gated.is_valid is False







class TestConfigMagnitudeFilterField:

    @pytest.mark.unit
    def test_config_has_magnitude_filter_field(self) -> None:
        from kd.search.discover.config import DiscoverConfig

        assert hasattr(DiscoverConfig(), "magnitude_filter")

    @pytest.mark.unit
    def test_magnitude_filter_defaults_off(self) -> None:
        from kd.search.discover.config import DiscoverConfig

        assert DiscoverConfig().magnitude_filter is False

    @pytest.mark.unit
    def test_magnitude_filter_accepts_true(self) -> None:
        from kd.search.discover.config import DiscoverConfig

        assert DiscoverConfig(magnitude_filter=True).magnitude_filter is True

    @pytest.mark.unit
    def test_presets_keep_magnitude_filter_off(self) -> None:
        from kd.search.discover.config import DiscoverConfig

        assert DiscoverConfig.burgers_preset().magnitude_filter is False
        assert DiscoverConfig.chafee_preset().magnitude_filter is False


















def _out_of_range_result() -> EvaluationResult:
    return _make_result(torch.tensor([1.0, 1e-6], dtype=torch.float64))


def _healthy_result() -> EvaluationResult:
    return _make_result(torch.tensor([0.5, 1.0], dtype=torch.float64))


class TestBuilderWiringContract:

    @pytest.mark.unit
    def test_default_off_no_result_filter(self) -> None:
        import torch as _torch

        from kd.search.discover.builder import build_engine
        from kd.search.discover.config import DiscoverConfig

        _torch.manual_seed(0)
        engine = build_engine(DiscoverConfig())
        assert engine._result_filter is None

        assert engine._reward_adapter(_out_of_range_result()) > 0.0

    @pytest.mark.unit
    def test_flag_on_wires_result_filter_that_invalidates_bad_fit(self) -> None:
        import torch as _torch

        from kd.search.discover.builder import build_engine
        from kd.search.discover.config import DiscoverConfig
        from kd.search.discover.evaluation.reward import compute_reward

        _torch.manual_seed(0)
        engine = build_engine(DiscoverConfig(magnitude_filter=True))
        assert engine._result_filter is not None
        gated = engine._result_filter(_out_of_range_result())
        assert gated.is_valid is False

        assert compute_reward(gated) == 0.0

    @pytest.mark.unit
    def test_flag_on_healthy_fit_matches_flag_off_reward(self) -> None:
        import torch as _torch

        from kd.search.discover.builder import build_engine
        from kd.search.discover.config import DiscoverConfig

        _torch.manual_seed(0)
        on = build_engine(DiscoverConfig(magnitude_filter=True))
        _torch.manual_seed(0)
        off = build_engine(DiscoverConfig())
        good = _healthy_result()
        assert on._reward_adapter(good) == pytest.approx(off._reward_adapter(good))

    @pytest.mark.unit
    def test_flag_on_does_not_change_reward_alpha_behaviour(self) -> None:
        import torch as _torch

        from kd.search.discover.builder import build_engine
        from kd.search.discover.config import DiscoverConfig
        from kd.search.discover.evaluation.reward import compute_reward

        _torch.manual_seed(0)
        engine = build_engine(DiscoverConfig(magnitude_filter=True, reward_alpha=0.05))
        good = _healthy_result()
        expected = compute_reward(good, alpha=0.05)
        assert engine._reward_adapter(good) == pytest.approx(expected)
