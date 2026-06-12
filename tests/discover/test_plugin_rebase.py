
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.engine import INITIAL_BEST_REWARD
from kd.search.discover.evaluation.magnitude import MAGNITUDE_FILTER_MAX
from kd.search.discover.evaluation.reward import compute_reward
from kd.search.discover.plugin import DISCOVERPlugin
from kd.search.protocol import PlatformComponents





CHECKPOINT_REWARD = 0.85

CHAMPION_IR = "diff_x(u)"

INVALID_CHAMPION_ERROR = "champion invalid on current data"







@dataclass(frozen=True)
class _StubSpec:

    nmse: float = 0.04
    complexity: int = 2
    coefficient: float = 1.5
    valid: bool = True
    error: str = ""


class _CountingEvaluator:

    def __init__(self, spec: _StubSpec | None = None) -> None:
        self._spec = spec or _StubSpec()
        self.calls: list[str] = []

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        self.calls.append(expr)
        return _result_from_spec(self._spec, expr)


def _result_from_spec(spec: _StubSpec, expr: str) -> EvaluationResult:
    if not spec.valid:
        return EvaluationResult(
            mse=0.0,
            nmse=0.0,
            r2=0.0,
            complexity=0,
            is_valid=False,
            error_message=spec.error,
            expression=expr,
        )
    return EvaluationResult(
        mse=spec.nmse,
        nmse=spec.nmse,
        r2=1.0 - spec.nmse,
        complexity=spec.complexity,
        is_valid=True,
        expression=expr,
        terms=[expr],
        coefficients=torch.tensor([spec.coefficient]),
        selected_indices=None,
    )







def _components(evaluator: _CountingEvaluator) -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=evaluator,
        context=MagicMock(),
        registry=MagicMock(),
        recorder=None,
    )


def _donor_payload(
    config: DiscoverConfig,
    *,
    champion_reward: float = CHECKPOINT_REWARD,
    champion_expr: str = CHAMPION_IR,
) -> dict[str, Any]:
    donor = DISCOVERPlugin(config)
    donor.prepare(_components(_CountingEvaluator()))
    payload: dict[str, Any] = donor.state
    payload["engine_state"]["best_reward"] = champion_reward
    payload["engine_state"]["best_expression"] = champion_expr
    return payload


def _expected_reprice(config: DiscoverConfig, spec: _StubSpec) -> float:
    result = _result_from_spec(spec, CHAMPION_IR)
    return float(np.float32(compute_reward(result, alpha=config.reward_alpha)))


def _restore_then_prepare(
    config: DiscoverConfig,
    payload: dict[str, Any],
    stub: _CountingEvaluator,
) -> DISCOVERPlugin:
    subject = DISCOVERPlugin(config)
    subject.state = payload
    subject.prepare(_components(stub))
    return subject







@pytest.mark.unit
class TestRestorePrepareRepricesValidChampion:

    def test_k_valid_champion_repriced_expression_preserved(self) -> None:
        config = DiscoverConfig()
        spec = _StubSpec()
        expected = _expected_reprice(config, spec)


        assert expected != pytest.approx(CHECKPOINT_REWARD)
        assert expected > INITIAL_BEST_REWARD

        payload = _donor_payload(config)
        stub = _CountingEvaluator(spec)
        subject = _restore_then_prepare(config, payload, stub)

        assert stub.calls == [CHAMPION_IR], (
            "restore -> prepare must re-price the champion through the "
            "current evaluator exactly once"
        )
        assert subject.best_score == expected, (
            "the gate must be the float32-quantized reward_adapter value of "
            "the fresh evaluation, not the checkpoint-era reward"
        )
        assert subject.best_expression == CHAMPION_IR, (
            "repricing must never clear the champion expression"
        )


        engine = subject._engine
        assert engine is not None
        best = engine.best_result
        assert best is not None
        assert best.is_valid is True
        assert best.nmse == pytest.approx(spec.nmse)

    def test_k_reprice_is_deterministic_across_restores(self) -> None:
        config = DiscoverConfig()
        spec = _StubSpec()
        first = _restore_then_prepare(
            config, _donor_payload(config), _CountingEvaluator(spec)
        )
        second = _restore_then_prepare(
            config, _donor_payload(config), _CountingEvaluator(spec)
        )
        assert first.best_score == second.best_score
        assert first.best_score == _expected_reprice(config, spec)







@pytest.mark.unit
class TestRestorePrepareInvalidChampion:

    def test_l_invalid_champion_zeroes_gate_keeps_expression(self) -> None:
        config = DiscoverConfig()
        spec = _StubSpec(valid=False, error=INVALID_CHAMPION_ERROR)
        payload = _donor_payload(config)
        assert payload["engine_state"]["best_reward"] != INITIAL_BEST_REWARD

        stub = _CountingEvaluator(spec)
        subject = _restore_then_prepare(config, payload, stub)

        assert stub.calls == [CHAMPION_IR]
        assert subject.best_score == INITIAL_BEST_REWARD, (
            "an invalid champion must release the gate to INITIAL_BEST_REWARD"
        )
        assert subject.best_expression == CHAMPION_IR, (
            "the expression is KEPT even when invalid (DSO never clears its champion)"
        )
        engine = subject._engine
        assert engine is not None
        best = engine.best_result
        assert best is not None
        assert best.is_valid is False
        assert best.error_message == INVALID_CHAMPION_ERROR







@pytest.mark.unit
class TestPrepareRebaseScopeGuards:

    def test_m_fresh_prepare_never_calls_evaluator(self) -> None:
        stub = _CountingEvaluator()
        plugin = DISCOVERPlugin(DiscoverConfig())
        plugin.prepare(_components(stub))
        assert stub.calls == []

    def test_n_restored_empty_champion_skips_reprice(self) -> None:
        config = DiscoverConfig()
        payload = _donor_payload(
            config, champion_reward=INITIAL_BEST_REWARD, champion_expr=""
        )
        assert payload["engine_state"]["best_expression"] == ""

        stub = _CountingEvaluator()
        subject = _restore_then_prepare(config, payload, stub)

        assert stub.calls == [], (
            "an empty restored champion must not trigger any evaluation "
            "(rebase early-return)"
        )
        assert subject.best_score == INITIAL_BEST_REWARD
        assert subject.best_expression == ""







@pytest.mark.unit
class TestPrepareRebaseMagnitudeFilter:

    def test_o_out_of_bounds_coefficient_gates_repriced_champion(self) -> None:
        gated_config = DiscoverConfig(magnitude_filter=True)
        spec = _StubSpec(coefficient=2.0 * MAGNITUDE_FILTER_MAX)

        stub = _CountingEvaluator(spec)
        subject = _restore_then_prepare(
            gated_config, _donor_payload(gated_config), stub
        )

        assert stub.calls == [CHAMPION_IR]
        assert subject.best_score == INITIAL_BEST_REWARD, (
            "a repriced result must pass through the magnitude filter "
            "before validity is judged"
        )
        assert subject.best_expression == CHAMPION_IR
        engine = subject._engine
        assert engine is not None
        best = engine.best_result
        assert best is not None
        assert best.is_valid is False




        ungated_config = DiscoverConfig()
        control = _restore_then_prepare(
            ungated_config, _donor_payload(ungated_config), _CountingEvaluator(spec)
        )
        assert control.best_score == _expected_reprice(ungated_config, spec)
        assert control.best_score > INITIAL_BEST_REWARD
