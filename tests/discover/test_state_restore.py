
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.search.discover.engine import (
    _RESTORED_PLACEHOLDER_ERROR_MESSAGE as EXPECTED_ERROR_MESSAGE,
)
from kd.search.discover.engine import DiscoverEngine
from kd.search.discover.engine_types import EngineState
from kd.search.discover.plugin import (
    BEST_RESULT_COEFFICIENTS_KEY,
    BEST_RESULT_TERMS_KEY,
    DISCOVERPlugin,
)
from kd.search.discover.training.strategy import BaselineState
from kd.search.protocol import PlatformComponents





assert EXPECTED_ERROR_MESSAGE != ""
SEED = 42







def _make_state(
    *,
    best_result_terms: list[str] | None,
    best_result_coefficients: list[float] | None,
    best_expression: str = "",
) -> EngineState:
    return EngineState(
        controller_state_dict={},
        baseline_state=BaselineState(),
        best_reward=0.85,
        best_expression=best_expression,
        optimizer_state=None,
        extras=None,
        best_result_terms=best_result_terms,
        best_result_coefficients=best_result_coefficients,
    )


class _MockEvaluator:

    def evaluate_expression(self, expr: str) -> Any:
        raise AssertionError("evaluate_expression must not be invoked in this test.")


@pytest.fixture
def prepared_plugin() -> DISCOVERPlugin:
    torch.manual_seed(SEED)
    components = PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=_MockEvaluator(),
        context=MagicMock(),
        registry=MagicMock(),
        recorder=None,
    )
    plugin = DISCOVERPlugin()
    plugin.prepare(components)
    return plugin







@pytest.mark.unit
def test_rebuild_best_result_when_terms_present_is_valid_false() -> None:
    terms = ["u", "u_x"]
    coefficients = [0.5, -0.3]
    state = _make_state(
        best_result_terms=terms,
        best_result_coefficients=coefficients,
        best_expression="0.5*u - 0.3*u_x",
    )

    result = DiscoverEngine._rebuild_best_result(state)

    assert result is not None


    assert result.is_valid is False


    assert result.mse == math.inf
    assert result.nmse == math.inf
    assert result.r2 == -math.inf
    assert result.aic == math.inf

    assert not math.isnan(result.mse)
    assert not math.isnan(result.nmse)
    assert not math.isnan(result.r2)
    assert result.aic is not None and not math.isnan(result.aic)



    assert result.mse > 0
    assert result.nmse > 0
    assert result.r2 < 0
    assert result.aic > 0

    assert result.complexity == 0


    assert result.selected_indices is None
    assert result.lhs_name is None
    assert result.residuals is None




    assert result.error_message != ""

    assert result.error_message == EXPECTED_ERROR_MESSAGE

    assert result.expression == "0.5*u - 0.3*u_x"
    assert result.terms == terms


    assert result.terms is not state.best_result_terms
    assert result.coefficients is not None
    torch.testing.assert_close(
        result.coefficients,
        torch.tensor([0.5, -0.3], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    assert result.coefficients.dtype == torch.float32







@pytest.mark.unit
@pytest.mark.parametrize(
    ("terms", "coefficients"),
    [
        (None, [0.5, -0.3]),
        (["u", "u_x"], None),
        (None, None),
    ],
    ids=["terms_none", "coefficients_none", "both_none"],
)
def test_rebuild_best_result_when_terms_absent_returns_none(
    terms: list[str] | None,
    coefficients: list[float] | None,
) -> None:
    state = _make_state(
        best_result_terms=terms,
        best_result_coefficients=coefficients,
    )
    assert DiscoverEngine._rebuild_best_result(state) is None







@pytest.mark.unit
def test_rebuild_best_result_when_terms_empty_returns_none() -> None:
    state = _make_state(
        best_result_terms=[],
        best_result_coefficients=[],
        best_expression="",
    )
    assert DiscoverEngine._rebuild_best_result(state) is None







@pytest.mark.unit
def test_state_resave_after_restore_preserves_terms(
    prepared_plugin: DISCOVERPlugin,
) -> None:
    saved_terms = ["u", "u_x"]
    saved_coefficients = [0.5, -0.3]
    saved_state: dict[str, Any] = {
        "algorithm": "discover",
        "engine_state": {
            "controller_state_dict": prepared_plugin.state["engine_state"][
                "controller_state_dict"
            ],
            "baseline_state": prepared_plugin.state["engine_state"]["baseline_state"],
            "best_reward": 0.85,
            "best_expression": "0.5*u - 0.3*u_x",
            BEST_RESULT_TERMS_KEY: list(saved_terms),
            BEST_RESULT_COEFFICIENTS_KEY: list(saved_coefficients),
        },
    }


    prepared_plugin.state = saved_state






    restored = prepared_plugin.state
    restored_engine_state = restored["engine_state"]
    assert restored_engine_state.get(BEST_RESULT_TERMS_KEY) == saved_terms
    restored_coeffs = restored_engine_state.get(BEST_RESULT_COEFFICIENTS_KEY)
    assert restored_coeffs is not None
    assert restored_coeffs == pytest.approx(saved_coefficients, rel=1e-6, abs=1e-6)


    assert restored_engine_state["best_expression"] == "0.5*u - 0.3*u_x"







@pytest.mark.unit
def test_strip_result_preserves_aic_error_message_lhs_name() -> None:
    from kd.search.discover.engine import DiscoverEngine

    source = EvaluationResult(
        mse=0.1,
        nmse=0.05,
        r2=0.95,
        aic=12.5,
        complexity=2,
        coefficients=torch.tensor([1.0, -0.5], dtype=torch.float64),
        is_valid=True,
        error_message="diagnostic note from evaluator",
        selected_indices=[0, 2],
        residuals=torch.zeros(8),
        terms=["u", "u_x"],
        expression="u + u_x",
        lhs_name="u_t",
    )

    stripped = DiscoverEngine._strip_result(source)


    assert stripped.mse == 0.1
    assert stripped.nmse == 0.05
    assert stripped.r2 == 0.95
    assert stripped.complexity == 2
    assert stripped.is_valid is True
    assert stripped.expression == "u + u_x"
    assert stripped.terms == ["u", "u_x"]
    assert stripped.selected_indices == [0, 2]
    assert stripped.coefficients is not None
    assert stripped.coefficients.tolist() == pytest.approx([1.0, -0.5])

    assert stripped.residuals is None


    assert stripped.aic == 12.5, (
        "aic must survive _strip_result; live↔restored schema parity "
        "requires this. Stage 5 M4."
    )
    assert stripped.error_message == "diagnostic note from evaluator", (
        "error_message must survive _strip_result; live↔restored schema "
        "parity requires this. Stage 5 M4."
    )
    assert stripped.lhs_name == "u_t", (
        "lhs_name must survive _strip_result; live↔restored schema parity "
        "requires this. Stage 5 M4."
    )


    assert stripped.terms is not source.terms
    assert stripped.coefficients is not source.coefficients
    assert stripped.selected_indices is not source.selected_indices







_RESUME_BATCH_SIZE = 16


class _FixedScoreEvaluator:

    def __init__(self, nmse: float = 0.5) -> None:
        self._nmse = nmse

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        return EvaluationResult(
            mse=self._nmse,
            nmse=self._nmse,
            r2=max(0.0, 1.0 - self._nmse),
            complexity=3,
            is_valid=True,
            expression=expr,
        )


def _components_with_evaluator(evaluator: Any) -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=evaluator,
        context=MagicMock(),
        registry=MagicMock(),
        recorder=None,
    )


@pytest.mark.unit
def test_runner_checkpoint_resume_preserves_restored_state(tmp_path: Path) -> None:
    from kd.search.runner import ExperimentRunner


    torch.manual_seed(SEED)
    donor = DISCOVERPlugin()
    donor.prepare(_components_with_evaluator(_FixedScoreEvaluator()))
    candidates = donor.propose(_RESUME_BATCH_SIZE)
    results = donor.evaluate(candidates)
    donor.update(results)

    saved_engine_state = donor.state["engine_state"]
    saved_best_reward = float(saved_engine_state["best_reward"])
    saved_best_expression = str(saved_engine_state["best_expression"])





    fresh_probe = DISCOVERPlugin()
    fresh_probe.prepare(_components_with_evaluator(_FixedScoreEvaluator()))
    assert saved_best_expression != ""
    assert saved_best_expression != fresh_probe.best_expression
    assert saved_best_reward != fresh_probe.best_score
    assert saved_engine_state.get("optimizer_state") is not None, (
        "donor cycle must have produced Adam state; otherwise the "
        "optimizer-survival assertion below is vacuous"
    )


    ckpt_path = tmp_path / "discover_resume.pt"
    save_runner = ExperimentRunner(
        algorithm=donor, max_iterations=2, batch_size=_RESUME_BATCH_SIZE
    )
    save_runner.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()


    resumed_plugin = DISCOVERPlugin()
    resume_runner = ExperimentRunner(
        algorithm=resumed_plugin, max_iterations=2, batch_size=_RESUME_BATCH_SIZE
    )
    resume_runner.load_checkpoint(ckpt_path)

    resumed_plugin.prepare(_components_with_evaluator(_FixedScoreEvaluator()))

    assert resumed_plugin.best_expression == saved_best_expression, (
        "restored best_expression was lost by the prepare() that follows "
        "load_checkpoint (restore-preserve contract violation)"
    )
    assert resumed_plugin.best_score == pytest.approx(saved_best_reward), (
        "restored best_reward was lost by the prepare() that follows "
        "load_checkpoint (restore-preserve contract violation)"
    )
    resumed_engine_state = resumed_plugin.state["engine_state"]
    assert resumed_engine_state["best_expression"] == saved_best_expression
    assert float(resumed_engine_state["best_reward"]) == pytest.approx(
        saved_best_reward
    )
    assert resumed_engine_state.get("optimizer_state") is not None, (
        "optimizer state must survive the resume prepare(); a rebuilt "
        "engine silently drops the Adam moments and the training trajectory"
    )


    saved_baseline = saved_engine_state["baseline_state"]
    assert int(saved_baseline["n_updates"]) >= 1, (
        "donor cycle must have updated the reward baseline; otherwise the "
        "baseline-survival assertion below is vacuous"
    )
    resumed_baseline = resumed_engine_state["baseline_state"]
    assert int(resumed_baseline["n_updates"]) == int(saved_baseline["n_updates"])
    assert float(resumed_baseline["ewma_reward"]) == pytest.approx(
        float(saved_baseline["ewma_reward"])
    )


@pytest.mark.unit
def test_fresh_plugin_rejects_malformed_payload_at_assignment() -> None:
    plugin = DISCOVERPlugin()
    with pytest.raises(TypeError, match="engine_state"):
        plugin.state = {"algorithm": "discover"}

    with pytest.raises(RuntimeError, match="prepare"):
        _ = plugin.state


@pytest.mark.unit
def test_state_getter_serializes_pending_restore_before_prepare() -> None:
    donor = DISCOVERPlugin()
    donor.prepare(_components_with_evaluator(_FixedScoreEvaluator()))
    payload = donor.state
    payload["engine_state"]["best_reward"] = 0.85
    payload["engine_state"]["best_expression"] = "diff_x(u)"
    donor.state = payload
    saved = donor.state

    subject = DISCOVERPlugin()
    subject.state = saved

    echoed = subject.state
    saved_engine = saved["engine_state"]
    echoed_engine = echoed["engine_state"]
    assert echoed_engine["best_expression"] == saved_engine["best_expression"]
    assert float(echoed_engine["best_reward"]) == pytest.approx(
        float(saved_engine["best_reward"])
    )
    assert echoed_engine["baseline_state"] == saved_engine["baseline_state"]
    assert set(echoed_engine["controller_state_dict"]) == set(
        saved_engine["controller_state_dict"]
    )
    for name, tensor in saved_engine["controller_state_dict"].items():
        assert torch.equal(echoed_engine["controller_state_dict"][name], tensor)
