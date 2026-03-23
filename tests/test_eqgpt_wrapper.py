"""Tests for KD_EqGPT wrapper and Gradio App integration (TDD RED phase).

Validates:
  1. KD_EqGPT lazy import via kd.model.__getattr__
  2. KD_EqGPT class contract (init defaults, fit_pretrained signature)
  3. app.py integration (MODEL_KEYS, parameter panel, _run_local branch)
  4. Regression: existing model imports unaffected
  5. Edge/negative: invalid args, type errors, boundary values

All tests should FAIL until kd/model/kd_eqgpt.py is implemented.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMPORT_FAST_THRESHOLD_SEC = 2.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "app.py"


# ===========================================================================
# Smoke: Import & Registration
# ===========================================================================


@pytest.mark.smoke
class TestKDEqGPTImport:
    """KD_EqGPT should be accessible via kd.model lazy import."""

    def test_lazy_import(self) -> None:
        """from kd.model import KD_EqGPT should work via __getattr__."""
        from kd.model import KD_EqGPT

        assert KD_EqGPT is not None

    def test_import_speed(self) -> None:
        """Import should not trigger heavy side effects (< 2s).

        EqGPT depends on torch + pickle data; lazy import must defer these.
        Ref: common-mistakes.md #10 (native crash from eager import).
        """
        t0 = time.perf_counter()
        from kd.model import KD_EqGPT  # noqa: F811

        elapsed = time.perf_counter() - t0
        assert elapsed < IMPORT_FAST_THRESHOLD_SEC, (
            f"Import took {elapsed:.1f}s -- side effect not deferred?"
        )

    def test_in_all(self) -> None:
        """KD_EqGPT should be listed in kd.model.__all__."""
        import kd.model

        assert "KD_EqGPT" in kd.model.__all__

    def test_repeated_import_same_class(self) -> None:
        """Repeated imports should return the same class object."""
        from kd.model import KD_EqGPT as cls1
        from kd.model import KD_EqGPT as cls2

        assert cls1 is cls2


# ===========================================================================
# Smoke: Constructor Contract
# ===========================================================================


@pytest.mark.smoke
class TestKDEqGPTInit:
    """KD_EqGPT constructor should set defaults per spec."""

    def test_default_init(self) -> None:
        """Default constructor should use spec defaults."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT()
        assert model.pretrained == "wave_breaking"
        assert model.optimize_epochs == 5
        assert model.samples_per_epoch == 400
        assert model.case_filter == "N"
        assert model.seed == 0

    def test_custom_init(self) -> None:
        """All constructor params should be overridable."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(
            pretrained="wave_breaking",
            optimize_epochs=3,
            samples_per_epoch=200,
            case_filter="all",
            seed=42,
        )
        assert model.optimize_epochs == 3
        assert model.samples_per_epoch == 200
        assert model.case_filter == "all"
        assert model.seed == 42

    def test_pretrained_default(self) -> None:
        """Default pretrained should be 'wave_breaking'."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT()
        assert model.pretrained == "wave_breaking"


# ===========================================================================
# Smoke: Method Contract
# ===========================================================================


@pytest.mark.smoke
class TestKDEqGPTContract:
    """KD_EqGPT public API contract checks."""

    def test_has_fit_pretrained(self) -> None:
        """KD_EqGPT should expose a callable fit_pretrained method."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT()
        assert hasattr(model, "fit_pretrained")
        assert callable(model.fit_pretrained)

    def test_fit_pretrained_no_required_args(self) -> None:
        """fit_pretrained() should be callable with no arguments.

        It uses self.xxx for all parameters (spec: no positional args).
        We don't run it here (slow), just check it's callable without TypeError.
        """
        import inspect

        from kd.model import KD_EqGPT

        model = KD_EqGPT()
        sig = inspect.signature(model.fit_pretrained)
        # All parameters (excluding self) should have defaults
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            assert param.default is not inspect.Parameter.empty, (
                f"fit_pretrained param {name!r} has no default"
            )


# ===========================================================================
# Smoke: App Integration (static code analysis)
# ===========================================================================


@pytest.mark.smoke
class TestAppIntegration:
    """app.py should integrate KD_EqGPT in MODEL_KEYS, UI panel, and _run_local."""

    def test_model_keys_has_eqgpt(self) -> None:
        """MODEL_KEYS dict should include a KD_EqGPT entry."""
        content = APP_PATH.read_text()
        assert "KD_EqGPT" in content, (
            "app.py does not mention KD_EqGPT in MODEL_KEYS"
        )
        # Specifically check it's in the MODEL_KEYS dict
        # Look for the pattern: "KD_EqGPT": "eqgpt"
        assert '"KD_EqGPT"' in content or "'KD_EqGPT'" in content, (
            "KD_EqGPT not found as a string key in app.py"
        )

    def test_app_has_eqgpt_parameter_group(self) -> None:
        """app.py should have an eqgpt parameter group for Gradio UI."""
        content = APP_PATH.read_text()
        has_group = "eqgpt_group" in content
        has_epochs_param = "eqgpt_epochs" in content
        assert has_group or has_epochs_param, (
            "app.py missing eqgpt parameter group (eqgpt_group or eqgpt_epochs)"
        )

    def test_run_local_eqgpt_branch(self) -> None:
        """_run_local should have a KD_EqGPT branch calling fit_pretrained."""
        content = APP_PATH.read_text()
        # Must have both the model name check and the method call
        assert "KD_EqGPT" in content, "KD_EqGPT not in app.py"
        assert "fit_pretrained" in content, (
            "fit_pretrained not called in app.py _run_local"
        )

    def test_get_equation_text_eqgpt_branch(self) -> None:
        """_get_equation_text should handle KD_EqGPT result format."""
        content = APP_PATH.read_text()
        # Should extract best_equation from the result dict
        assert "best_equation" in content, (
            "app.py _get_equation_text doesn't handle best_equation key"
        )

    def test_on_model_change_eqgpt_visibility(self) -> None:
        """on_model_change should toggle eqgpt_group visibility."""
        content = APP_PATH.read_text()
        # The function must reference eqgpt to toggle its visibility
        assert "eqgpt" in content.lower(), (
            "on_model_change does not reference eqgpt visibility"
        )


# ===========================================================================
# Edge / Negative: Invalid constructor arguments
# ===========================================================================


@pytest.mark.smoke
class TestKDEqGPTEdgeCases:
    """Edge cases and negative tests for KD_EqGPT."""

    def test_zero_epochs(self) -> None:
        """optimize_epochs=0 should raise ValueError (no search to perform)."""
        from kd.model import KD_EqGPT

        with pytest.raises(ValueError, match="optimize_epochs"):
            KD_EqGPT(optimize_epochs=0)

    def test_negative_epochs(self) -> None:
        """Negative optimize_epochs should raise ValueError."""
        from kd.model import KD_EqGPT

        with pytest.raises(ValueError, match="optimize_epochs"):
            KD_EqGPT(optimize_epochs=-1)

    def test_zero_samples(self) -> None:
        """samples_per_epoch=0 should raise ValueError."""
        from kd.model import KD_EqGPT

        with pytest.raises(ValueError, match="samples_per_epoch"):
            KD_EqGPT(samples_per_epoch=0)

    def test_negative_samples(self) -> None:
        """Negative samples_per_epoch should raise ValueError."""
        from kd.model import KD_EqGPT

        with pytest.raises(ValueError, match="samples_per_epoch"):
            KD_EqGPT(samples_per_epoch=-100)

    def test_negative_seed(self) -> None:
        """Negative seed should raise ValueError."""
        from kd.model import KD_EqGPT

        with pytest.raises(ValueError, match="seed"):
            KD_EqGPT(seed=-1)

    def test_invalid_pretrained_name(self) -> None:
        """Unknown pretrained model name should raise ValueError."""
        from kd.model import KD_EqGPT

        with pytest.raises(ValueError, match="pretrained"):
            KD_EqGPT(pretrained="nonexistent_model")

    def test_invalid_case_filter(self) -> None:
        """Invalid case_filter should raise ValueError."""
        from kd.model import KD_EqGPT

        with pytest.raises(ValueError, match="case_filter"):
            KD_EqGPT(case_filter="invalid_filter")

    def test_epochs_type_float(self) -> None:
        """Float optimize_epochs should raise TypeError."""
        from kd.model import KD_EqGPT

        with pytest.raises(TypeError, match="optimize_epochs"):
            KD_EqGPT(optimize_epochs=2.5)

    def test_samples_type_float(self) -> None:
        """Float samples_per_epoch should raise TypeError."""
        from kd.model import KD_EqGPT

        with pytest.raises(TypeError, match="samples_per_epoch"):
            KD_EqGPT(samples_per_epoch=100.5)


# ===========================================================================
# Edge: Boundary values (valid but extreme)
# ===========================================================================


@pytest.mark.smoke
class TestKDEqGPTBoundaryValues:
    """Valid boundary values that should be accepted."""

    def test_min_epochs(self) -> None:
        """optimize_epochs=1 should be accepted (minimum valid)."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(optimize_epochs=1)
        assert model.optimize_epochs == 1

    def test_min_samples(self) -> None:
        """samples_per_epoch=1 should be accepted (minimum valid)."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(samples_per_epoch=1)
        assert model.samples_per_epoch == 1

    def test_seed_zero(self) -> None:
        """seed=0 should be accepted (boundary)."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(seed=0)
        assert model.seed == 0

    def test_large_seed(self) -> None:
        """Large seed value should be accepted."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(seed=2**31 - 1)
        assert model.seed == 2**31 - 1

    def test_case_filter_N(self) -> None:
        """case_filter='N' should be accepted (spec default)."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(case_filter="N")
        assert model.case_filter == "N"

    def test_case_filter_all(self) -> None:
        """case_filter='all' should be accepted."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(case_filter="all")
        assert model.case_filter == "all"


# ===========================================================================
# Regression: Existing models unaffected
# ===========================================================================


@pytest.mark.smoke
class TestRegression:
    """Adding KD_EqGPT must not break existing model imports."""

    def test_existing_models_importable(self) -> None:
        """All pre-existing models in __all__ should still import."""
        from kd.model import KD_DLGA, KD_Discover, KD_SGA

        assert KD_Discover is not None
        assert KD_SGA is not None
        assert KD_DLGA is not None

    def test_existing_all_preserved(self) -> None:
        """__all__ should still contain all pre-existing model names."""
        import kd.model

        expected_existing = [
            "DLGA",
            "KD_DLGA",
            "KD_Discover",
            "KD_Discover_SPR",
            "KD_Discover_Regression",
            "KD_SGA",
            "KD_PySR",
        ]
        for name in expected_existing:
            assert name in kd.model.__all__, (
                f"{name!r} missing from kd.model.__all__ after KD_EqGPT addition"
            )

    def test_getattr_still_raises_on_unknown(self) -> None:
        """Accessing a nonexistent name should still raise AttributeError."""
        import kd.model

        with pytest.raises(AttributeError):
            _ = kd.model.NonExistentModel  # type: ignore[attr-defined]


# ===========================================================================
# Happy path: Return value contract (structural, not functional)
# ===========================================================================


@pytest.mark.smoke
class TestFitPretrainedReturnContract:
    """Structural tests for fit_pretrained return value.

    These verify the dict schema WITHOUT running the full search.
    Actual return values are tested in the slow functional test below.
    """

    def test_return_annotation_is_dict(self) -> None:
        """fit_pretrained should have -> dict return annotation."""
        import inspect

        from kd.model import KD_EqGPT

        sig = inspect.signature(KD_EqGPT.fit_pretrained)
        assert sig.return_annotation is dict or sig.return_annotation == "dict", (
            f"fit_pretrained return annotation should be dict, "
            f"got {sig.return_annotation!r}"
        )


# ===========================================================================
# Slow: Functional end-to-end test (~3 min)
# ===========================================================================


@pytest.mark.slow
class TestKDEqGPTFunctional:
    """End-to-end functional tests (require pretrained weights, ~3 min)."""

    def test_fit_pretrained_returns_expected_keys(self) -> None:
        """fit_pretrained result dict should have all required keys."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(
            optimize_epochs=5,
            samples_per_epoch=400,
            case_filter="N",
            seed=0,
        )
        result = model.fit_pretrained()

        # Return type
        assert isinstance(result, dict), (
            f"fit_pretrained should return dict, got {type(result).__name__}"
        )
        # Required keys per spec
        assert "equations" in result, "Missing 'equations' key"
        assert "rewards" in result, "Missing 'rewards' key"
        assert "best_equation" in result, "Missing 'best_equation' key"
        assert "best_reward" in result, "Missing 'best_reward' key"

    def test_fit_pretrained_equations_count(self) -> None:
        """Should return top 10 equations."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(
            optimize_epochs=5,
            samples_per_epoch=400,
            case_filter="N",
            seed=0,
        )
        result = model.fit_pretrained()

        assert len(result["equations"]) == 10, (
            f"Expected 10 equations, got {len(result['equations'])}"
        )
        assert len(result["rewards"]) == 10, (
            f"Expected 10 rewards, got {len(result['rewards'])}"
        )

    def test_fit_pretrained_reward_range(self) -> None:
        """Best reward should be in [0.90, 1.0] range (spec: ~0.94-0.95)."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(
            optimize_epochs=5,
            samples_per_epoch=400,
            case_filter="N",
            seed=0,
        )
        result = model.fit_pretrained()

        best_reward = result["best_reward"]
        assert isinstance(best_reward, float), (
            f"best_reward should be float, got {type(best_reward).__name__}"
        )
        assert 0.90 <= best_reward <= 1.0, (
            f"best_reward={best_reward:.4f} outside expected [0.90, 1.0]"
        )

    def test_fit_pretrained_best_equation_nonempty(self) -> None:
        """Best equation should be a non-empty string."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(
            optimize_epochs=5,
            samples_per_epoch=400,
            case_filter="N",
            seed=0,
        )
        result = model.fit_pretrained()

        best_eq = result["best_equation"]
        assert isinstance(best_eq, str), (
            f"best_equation should be str, got {type(best_eq).__name__}"
        )
        assert len(best_eq) > 0, "best_equation should not be empty"

    def test_fit_pretrained_rewards_sorted_descending(self) -> None:
        """Rewards list should be sorted descending (best first)."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(
            optimize_epochs=5,
            samples_per_epoch=400,
            case_filter="N",
            seed=0,
        )
        result = model.fit_pretrained()

        rewards = result["rewards"]
        for i in range(len(rewards) - 1):
            assert rewards[i] >= rewards[i + 1], (
                f"Rewards not sorted descending: "
                f"rewards[{i}]={rewards[i]} < rewards[{i+1}]={rewards[i+1]}"
            )

    def test_fit_pretrained_best_matches_first(self) -> None:
        """best_equation and best_reward should match the first entry."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(
            optimize_epochs=5,
            samples_per_epoch=400,
            case_filter="N",
            seed=0,
        )
        result = model.fit_pretrained()

        assert result["best_equation"] == result["equations"][0], (
            "best_equation should be the first (top) equation"
        )
        assert result["best_reward"] == result["rewards"][0], (
            "best_reward should be the first (top) reward"
        )

    def test_fit_pretrained_all_rewards_finite(self) -> None:
        """All rewards should be finite floats (no NaN/Inf)."""
        import math

        from kd.model import KD_EqGPT

        model = KD_EqGPT(
            optimize_epochs=5,
            samples_per_epoch=400,
            case_filter="N",
            seed=0,
        )
        result = model.fit_pretrained()

        for i, r in enumerate(result["rewards"]):
            assert isinstance(r, float), (
                f"rewards[{i}] should be float, got {type(r).__name__}"
            )
            assert math.isfinite(r), (
                f"rewards[{i}]={r} is not finite"
            )

    def test_fit_pretrained_all_equations_nonempty_strings(self) -> None:
        """All equations should be non-empty strings."""
        from kd.model import KD_EqGPT

        model = KD_EqGPT(
            optimize_epochs=5,
            samples_per_epoch=400,
            case_filter="N",
            seed=0,
        )
        result = model.fit_pretrained()

        for i, eq in enumerate(result["equations"]):
            assert isinstance(eq, str), (
                f"equations[{i}] should be str, got {type(eq).__name__}"
            )
            assert len(eq) > 0, f"equations[{i}] should not be empty"
