
from __future__ import annotations



import importlib.util as _importlib_util
import json
import math
import sys as _sys
from pathlib import Path
from types import SimpleNamespace as _SimpleNamespace
from typing import Any, cast

import pytest
import torch

from kd.search.discover.runners.mode2_helpers import (
    assert_noise_statistics,
    compute_paired_nrmse,
    detect_structure_hit,
    run_mode1_on_noise,
)





_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_PATH = _PROJECT_ROOT / "data" / "allen_cahn_2d_paper.npz"
_ARTIFACT_DIR = (
    _PROJECT_ROOT / ".session" / "phase3-research" / "task4-multiseed"
)
_SUMMARY_PATH = _ARTIFACT_DIR / "summary.json"
_MIN_DATA_BYTES = 50_000


V13_NOISE_LEVEL = 0.01
V13_STD_RTOL = 0.05
V13_SKEW_ABS_MAX = 0.2
V13_KURT_ABS_MAX = 0.5
V13_AUTOCORR_ABS_MAX = 0.1


V15_PRIMARY_GATE = 0.15
V15_FLOOR_GATE = 0.2172
V16_PASS_RATE = 2 / 3
STRETCH_TARGET = 0.05

GROUND_TRUTH_TERMS = ("diff2_x(u)", "diff2_y(u)", "u", "n3(u)")


def _data_ready() -> bool:
    return _DATA_PATH.exists() and _DATA_PATH.stat().st_size > _MIN_DATA_BYTES


def _artifact_ready() -> bool:
    return _SUMMARY_PATH.exists()


_SKIP_NO_DATA = pytest.mark.skipif(
    not _data_ready(),
    reason=f"Allen-Cahn paper data absent at {_DATA_PATH}",
)
_SKIP_NO_ARTIFACT = pytest.mark.skipif(
    not _artifact_ready(),
    reason=f"Task 4 multiseed artifact absent at {_SUMMARY_PATH}",
)







@_SKIP_NO_DATA
class TestNoiseModelParityV13:

    @pytest.fixture(scope="class")
    def noise_stats(self) -> dict[str, float]:
        from kd.search.discover.data.allen_cahn_2d import load_allen_cahn_2d
        from kd.search.discover.data.loader import add_gaussian_noise

        clean = load_allen_cahn_2d(_DATA_PATH, dtype=torch.float64)

        noisy = add_gaussian_noise(
            clean, V13_NOISE_LEVEL, seed=0, scale="max",
        )
        return assert_noise_statistics(
            noisy, clean, V13_NOISE_LEVEL, rtol=V13_STD_RTOL
        )

    def test_mean_near_zero(self, noise_stats: dict[str, float]) -> None:
        assert abs(noise_stats["mean"]) < V13_STD_RTOL * noise_stats["std"]

    def test_std_matches_level_times_max(
        self, noise_stats: dict[str, float]
    ) -> None:
        from kd.search.discover.data.allen_cahn_2d import load_allen_cahn_2d

        clean = load_allen_cahn_2d(_DATA_PATH, dtype=torch.float64)
        assert clean.fields is not None
        expected = V13_NOISE_LEVEL * float(
            clean.fields["u"].values.abs().max().item()
        )
        assert math.isclose(
            noise_stats["std"], expected, rel_tol=V13_STD_RTOL
        )

    def test_skew_near_zero(self, noise_stats: dict[str, float]) -> None:
        assert abs(noise_stats["skew"]) < V13_SKEW_ABS_MAX

    def test_excess_kurtosis_near_zero(
        self, noise_stats: dict[str, float]
    ) -> None:
        assert abs(noise_stats["kurt"]) < V13_KURT_ABS_MAX

    def test_lag1_autocorr_near_zero(
        self, noise_stats: dict[str, float]
    ) -> None:
        assert noise_stats["max_abs_autocorr_lag1"] < V13_AUTOCORR_ABS_MAX







class TestPairedNRMSEV17:

    def test_identical_inputs_return_zero(self) -> None:
        u = torch.linspace(-1.0, 1.0, 100, dtype=torch.float64)
        assert compute_paired_nrmse(u, u) == pytest.approx(0.0)

    def test_constant_shift_scales_by_ref_std(self) -> None:
        u = torch.linspace(-1.0, 1.0, 100, dtype=torch.float64)
        shifted = u + 0.5
        expected = 0.5 / float(u.std(correction=0).item())
        assert compute_paired_nrmse(shifted, u) == pytest.approx(
            expected, rel=1e-6
        )

    def test_gaussian_residual_matches_sigma(self) -> None:
        torch.manual_seed(0)
        ref = torch.randn(10_000, dtype=torch.float64)
        pred = ref + 0.1 * torch.randn(10_000, dtype=torch.float64)

        assert compute_paired_nrmse(pred, ref) == pytest.approx(0.1, rel=0.05)

    def test_zero_reference_returns_finite(self) -> None:
        zeros = torch.zeros(100, dtype=torch.float64)
        assert math.isfinite(compute_paired_nrmse(zeros, zeros))

    def test_returns_python_float(self) -> None:
        u = torch.linspace(-1.0, 1.0, 100, dtype=torch.float64)
        assert isinstance(compute_paired_nrmse(u, u), float)







class TestStructureHitV16:

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [

            ("sub(sub(diff2_x(u),diff2_y(u)),add(u,n3(u)))", True),

            ("sub(add(diff2_y(u),sub(diff2_x(u),n3(u))),u)", True),

            (
                "sub(diff2_y(u),sub(sub(sub(n3(u),u),"
                "add(diff2_x(u),x)),mul(y,div(t,t))))",
                False,
            ),

            ("sub(diff2_x(u),n3(u))", False),

            ("", False),
        ],
    )
    def test_4term_ground_truth(
        self, expression: str, expected: bool
    ) -> None:
        assert (
            detect_structure_hit(expression, list(GROUND_TRUTH_TERMS))
            is expected
        )

    def test_single_term_positive(self) -> None:
        assert detect_structure_hit("diff2_x(u)", ["diff2_x(u)"]) is True

    def test_single_term_wrong_term(self) -> None:
        assert detect_structure_hit("u", ["diff2_x(u)"]) is False

    def test_duplicate_terms_collapse(self) -> None:
        assert (
            detect_structure_hit(
                "add(diff2_x(u),diff2_x(u))", ["diff2_x(u)"]
            )
            is True
        )







def _load_run_mode2_ac2d_module() -> Any:
    script_path = _PROJECT_ROOT / "scripts" / "discover" / "run_mode2_allen_cahn_2d.py"
    spec = _importlib_util.spec_from_file_location(
        "test_run_mode2_allen_cahn_2d", script_path
    )
    assert spec is not None and spec.loader is not None
    module = _importlib_util.module_from_spec(spec)
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_fake_result_for_gate_payload() -> Any:
    return _SimpleNamespace(
        pretrain_result=_SimpleNamespace(train_loss=1e-4),
        cycle_metrics=[{"physics_loss": 1e-4}],
    )


@pytest.mark.unit
class TestStructuralGateInGatesPayload:

    @pytest.fixture(scope="class")
    def gate_module(self) -> Any:
        return _load_run_mode2_ac2d_module()

    def test_gate_payload_has_structural_ok_field(
        self, gate_module: Any
    ) -> None:
        result = _make_fake_result_for_gate_payload()
        fit_summary = {"max_rel_coef_error": 0.003}
        payload = gate_module._gate_payload(
            result,
            "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))",
            fit_summary,
        )
        assert "structural_ok" in payload

    def test_correct_form_structural_ok_true(self, gate_module: Any) -> None:
        result = _make_fake_result_for_gate_payload()
        payload = gate_module._gate_payload(
            result,
            "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))",
            {"max_rel_coef_error": 0.003},
        )
        assert payload["structural_ok"] is True

        assert payload["structure_hit"] is True

    def test_v10_cheating_form_structural_ok_false(
        self, gate_module: Any
    ) -> None:
        result = _make_fake_result_for_gate_payload()
        payload = gate_module._gate_payload(
            result,
            "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",
            {"max_rel_coef_error": 0.003},
        )
        assert payload["structural_ok"] is False


        assert payload["structure_hit"] is True


@pytest.mark.unit
class TestReleasePassInGatesPayload:

    @pytest.fixture(scope="class")
    def gate_module(self) -> Any:
        return _load_run_mode2_ac2d_module()

    def test_release_pass_field_present(self, gate_module: Any) -> None:
        result = _make_fake_result_for_gate_payload()
        payload = gate_module._gate_payload(
            result,
            "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))",
            {"max_rel_coef_error": 0.003},
        )
        assert "release_pass" in payload

    def test_correct_form_release_pass_true(self, gate_module: Any) -> None:
        result = _make_fake_result_for_gate_payload()
        payload = gate_module._gate_payload(
            result,
            "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))",
            {"max_rel_coef_error": 0.003},
        )
        assert payload["release_pass"] is True

    def test_v10_cheating_form_release_pass_false(
        self, gate_module: Any
    ) -> None:
        result = _make_fake_result_for_gate_payload()
        payload = gate_module._gate_payload(
            result,
            "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",
            {"max_rel_coef_error": 0.003},
        )
        assert payload["release_pass"] is False

        assert payload["structural_ok"] is False

        assert payload["structure_hit"] is True

    def test_release_pass_requires_finite_cycle_physics(
        self, gate_module: Any
    ) -> None:
        result = _SimpleNamespace(
            pretrain_result=_SimpleNamespace(train_loss=1e-4),
            cycle_metrics=[{"physics_loss": float("nan")}],
        )
        payload = gate_module._gate_payload(
            result,
            "add(add(diff2_x(u), diff2_y(u)), sub(u, n3(u)))",
            {"max_rel_coef_error": 0.003},
        )


        assert payload["cycle_physics_finite"] is False
        assert payload["release_pass"] is False







@_SKIP_NO_DATA
class TestMode1OnNoiseBaselineV15:

    @pytest.fixture(scope="class")
    def baseline_result(self) -> dict[str, Any]:
        from kd.search.discover.data.allen_cahn_2d import load_allen_cahn_2d
        from kd.search.discover.data.loader import add_gaussian_noise

        clean = load_allen_cahn_2d(_DATA_PATH, dtype=torch.float64)

        noisy = add_gaussian_noise(
            clean, V13_NOISE_LEVEL, seed=0, scale="max",
        )
        return run_mode1_on_noise(
            noisy,
            seed=0,
            n_points=500,
            batch_size=50,
            n_iterations=3,
        )

    def test_returns_v10_compatible_schema(
        self, baseline_result: dict[str, Any]
    ) -> None:
        assert baseline_result["seed"] == 0
        assert "ground_truth_fit" in baseline_result
        assert "max_rel_coef_error" in baseline_result["ground_truth_fit"]
        assert "mode1_run" in baseline_result

    def test_per_term_error_has_4_entries(
        self, baseline_result: dict[str, Any]
    ) -> None:
        per_term = baseline_result["ground_truth_fit"]["per_term_rel_error"]
        assert set(per_term.keys()) == {"diff2_x", "diff2_y", "u", "n3_u"}

    def test_max_rel_is_finite(
        self, baseline_result: dict[str, Any]
    ) -> None:
        max_rel = baseline_result["ground_truth_fit"]["max_rel_coef_error"]
        assert math.isfinite(max_rel) and max_rel > 0.0







@_SKIP_NO_ARTIFACT
class TestAcceptanceV15V16V17:

    @pytest.fixture(scope="class")
    def summary(self) -> dict[str, Any]:
        with _SUMMARY_PATH.open("r", encoding="utf-8") as handle:
            return cast(dict[str, Any], json.load(handle))

    def _pass_rate(
        self, summary: dict[str, Any], predicate: Any
    ) -> float:
        per_seed = summary["per_seed_summary"]
        if not per_seed:
            return 0.0
        return sum(1 for r in per_seed if predicate(r)) / len(per_seed)

    def test_primary_gate_max_rel_below_15pct(
        self, summary: dict[str, Any]
    ) -> None:
        rate = self._pass_rate(
            summary, lambda r: r["max_rel"] < V15_PRIMARY_GATE
        )
        assert rate >= V16_PASS_RATE, (
            f"V15 primary: {rate:.2%} < required {V16_PASS_RATE:.2%}"
        )

    def test_floor_gate_max_rel_below_21_72pct(
        self, summary: dict[str, Any]
    ) -> None:
        rate = self._pass_rate(
            summary, lambda r: r["max_rel"] < V15_FLOOR_GATE
        )
        assert rate >= V16_PASS_RATE, (
            f"V15 floor: {rate:.2%} < required {V16_PASS_RATE:.2%}"
        )

    def test_v15_mode2_beats_mode1_on_noise(
        self, summary: dict[str, Any]
    ) -> None:
        rate = self._pass_rate(
            summary,
            lambda r: r["max_rel"] < r["mode1_on_noise_max_rel"],
        )
        assert rate >= V16_PASS_RATE, (
            f"V15 beats-baseline: {rate:.2%} < required {V16_PASS_RATE:.2%}"
        )

    def test_v16_structure_hit_rate(
        self, summary: dict[str, Any]
    ) -> None:
        rate = self._pass_rate(summary, lambda r: r["structure_hit"])
        assert rate >= V16_PASS_RATE, (
            f"V16 structure hit: {rate:.2%} < required {V16_PASS_RATE:.2%}"
        )

    def test_v17_nrmse_denoised_improves(
        self, summary: dict[str, Any]
    ) -> None:
        rate = self._pass_rate(
            summary, lambda r: r["nrmse_denoised"] < r["nrmse_noisy"]
        )
        assert rate >= V16_PASS_RATE, (
            f"V17 NRMSE improvement: {rate:.2%} < required {V16_PASS_RATE:.2%}"
        )

    def test_stretch_target_recorded_null_on_miss(
        self, summary: dict[str, Any]
    ) -> None:
        for r in summary["per_seed_summary"]:
            assert "stretch_met" in r

    def test_pretrain_loss_below_ceiling(
        self, summary: dict[str, Any]
    ) -> None:
        rate = self._pass_rate(
            summary, lambda r: r["pretrain_train_loss"] < 0.5
        )
        assert rate >= V16_PASS_RATE
