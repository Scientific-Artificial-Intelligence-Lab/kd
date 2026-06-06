
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = (
    _PROJECT_ROOT / "scripts" / "discover" / "research" / "run_phase3_2_task4_multiseed.py"
)


def _load_runner_module() -> ModuleType:
    scripts_dir = str(_SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "test_run_phase3_2_task4_multiseed_runner", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    return _load_runner_module()


def _per_seed_summary_row(
    *, max_rel: float, structural_ok: bool, expr: str = ""
) -> dict[str, Any]:
    return {
        "seed": 0,
        "max_rel": max_rel,
        "l1_ratio": 0.0,
        "structure_hit": True,
        "structural_ok": structural_ok,
        "nrmse_noisy": 0.05,
        "nrmse_denoised": 0.04,
        "mode1_on_noise_max_rel": 0.5,
        "stretch_met": True,
        "pretrain_train_loss": 1e-4,
        "best_expression": expr,
        "mode1_on_noise_best_expression": "",
    }


@pytest.mark.unit
class TestParseArgsLegacyGateOnly:
    def test_default_strict(self, runner: ModuleType) -> None:
        old_argv = sys.argv[:]
        try:
            sys.argv = ["prog"]
            parsed = runner.parse_args()
            assert parsed.legacy_gate_only is False
        finally:
            sys.argv = old_argv

    def test_legacy_flag_sets_true(self, runner: ModuleType) -> None:
        old_argv = sys.argv[:]
        try:
            sys.argv = ["prog", "--legacy-gate-only"]
            parsed = runner.parse_args()
            assert parsed.legacy_gate_only is True
        finally:
            sys.argv = old_argv


@pytest.mark.unit
class TestCombinedGateInSummary:

    def test_combined_gate_present(self, runner: ModuleType) -> None:
        per_seed = [
            _per_seed_summary_row(max_rel=0.001, structural_ok=True),
            _per_seed_summary_row(max_rel=0.001, structural_ok=False),
        ]
        block = runner._compute_combined_gate(
            per_seed, threshold=runner.PRIMARY_GATE
        )
        assert block["n_total"] == 2

        assert block["n_pass"] == 1
        assert block["pass_rate"] == 0.5

    def test_combined_gate_blocks_sign_flip(
        self, runner: ModuleType
    ) -> None:
        per_seed = [
            _per_seed_summary_row(max_rel=0.001, structural_ok=True),
            _per_seed_summary_row(max_rel=0.001, structural_ok=True),
            _per_seed_summary_row(max_rel=0.001, structural_ok=True),
            _per_seed_summary_row(max_rel=0.001, structural_ok=False),
            _per_seed_summary_row(max_rel=0.001, structural_ok=False),
        ]
        block = runner._compute_combined_gate(
            per_seed, threshold=runner.PRIMARY_GATE
        )
        assert block["n_pass"] == 3
        assert block["n_total"] == 5

    def test_combined_gate_requires_coef_gate(
        self, runner: ModuleType
    ) -> None:


        per_seed = [
            _per_seed_summary_row(max_rel=0.20, structural_ok=True),
        ]
        block = runner._compute_combined_gate(
            per_seed, threshold=runner.PRIMARY_GATE
        )
        assert block["n_pass"] == 0


@pytest.mark.unit
class TestEvaluateReleasePass:

    def test_strict_blocks_sign_flip_cheating(
        self, runner: ModuleType
    ) -> None:
        per_seed = [
            _per_seed_summary_row(max_rel=0.001, structural_ok=True),
            _per_seed_summary_row(max_rel=0.001, structural_ok=False),
            _per_seed_summary_row(max_rel=0.001, structural_ok=False),
        ]
        verdict = runner._evaluate_release(
            per_seed,
            threshold=runner.PRIMARY_GATE,
            min_pass_rate=runner.PASS_RATE_GATE,
            strict=True,
        )

        assert verdict.passed is False
        assert verdict.strict is True

    def test_strict_passes_when_majority_clean(
        self, runner: ModuleType
    ) -> None:
        per_seed = [
            _per_seed_summary_row(max_rel=0.001, structural_ok=True),
            _per_seed_summary_row(max_rel=0.001, structural_ok=True),
            _per_seed_summary_row(max_rel=0.001, structural_ok=False),
        ]
        verdict = runner._evaluate_release(
            per_seed,
            threshold=runner.PRIMARY_GATE,
            min_pass_rate=runner.PASS_RATE_GATE,
            strict=True,
        )
        assert verdict.passed is True
        assert verdict.n_pass == 2
        assert verdict.n_total == 3

    def test_legacy_admits_sign_flip_cheating(
        self, runner: ModuleType
    ) -> None:
        per_seed = [
            _per_seed_summary_row(max_rel=0.001, structural_ok=False),
            _per_seed_summary_row(max_rel=0.001, structural_ok=False),
            _per_seed_summary_row(max_rel=0.001, structural_ok=False),
        ]
        verdict = runner._evaluate_release(
            per_seed,
            threshold=runner.PRIMARY_GATE,
            min_pass_rate=runner.PASS_RATE_GATE,
            strict=False,
        )
        assert verdict.passed is True
        assert verdict.strict is False


@pytest.mark.unit
class TestExitCodeFromVerdict:
    def test_pass_returns_zero(self, runner: ModuleType) -> None:
        verdict = runner.ReleaseVerdict(
            passed=True, n_pass=2, n_total=3, strict=True
        )
        assert runner._exit_code(verdict) == 0

    def test_fail_returns_nonzero(self, runner: ModuleType) -> None:
        verdict = runner.ReleaseVerdict(
            passed=False, n_pass=1, n_total=3, strict=True
        )
        assert runner._exit_code(verdict) != 0
