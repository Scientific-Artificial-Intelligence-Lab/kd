
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = (
    _PROJECT_ROOT / "scripts" / "discover" / "research" / "run_phase3_2_task3_multiseed.py"
)


def _load_runner_module() -> ModuleType:
    scripts_dir = str(_SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "test_run_phase3_2_task3_multiseed_runner", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    return _load_runner_module()


def _seed_with_expr(expr: str, max_rel: float = 0.001) -> dict[str, Any]:
    return {
        "ground_truth_fit": {
            "max_rel_coef_error": max_rel,
            "l1_ratio_error": 0.0,
        },
        "mode1_run": {"best_expression": expr},
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
class TestEvaluateReleasePass:

    def test_strict_blocks_v10_cheating(self, runner: ModuleType) -> None:
        per_seed = [
            _seed_with_expr(
                "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",
                max_rel=0.003,
            ),
            _seed_with_expr(
                "add(diff2_x(u), diff2_y(u))", max_rel=0.001
            ),
        ]
        verdict = runner._evaluate_release(
            per_seed,
            threshold=0.05,
            min_pass_rate=2 / 3,
            strict=True,
        )
        assert verdict.passed is False
        assert verdict.n_pass == 1
        assert verdict.n_total == 2
        assert verdict.strict is True

    def test_strict_passes_clean_run(self, runner: ModuleType) -> None:
        per_seed = [
            _seed_with_expr("add(diff2_x(u), diff2_y(u))", max_rel=0.001),
            _seed_with_expr("add(diff2_x(u), diff2_y(u))", max_rel=0.002),
        ]
        verdict = runner._evaluate_release(
            per_seed,
            threshold=0.05,
            min_pass_rate=1.0,
            strict=True,
        )
        assert verdict.passed is True
        assert verdict.n_pass == 2

    def test_legacy_admits_sign_flip_cheating(
        self, runner: ModuleType
    ) -> None:
        per_seed = [
            _seed_with_expr(
                "sub(sub(diff2_x(u), diff2_y(u)), add(u, n3(u)))",
                max_rel=0.003,
            )
        ]
        verdict = runner._evaluate_release(
            per_seed,
            threshold=0.05,
            min_pass_rate=1.0,
            strict=False,
        )
        assert verdict.passed is True
        assert verdict.strict is False


@pytest.mark.unit
class TestExitCodeFromVerdict:

    def test_pass_returns_zero(self, runner: ModuleType) -> None:
        verdict = runner.ReleaseVerdict(
            passed=True, n_pass=2, n_total=2, strict=True
        )
        assert runner._exit_code(verdict) == 0

    def test_fail_returns_nonzero(self, runner: ModuleType) -> None:
        verdict = runner.ReleaseVerdict(
            passed=False, n_pass=1, n_total=2, strict=True
        )
        assert runner._exit_code(verdict) != 0
