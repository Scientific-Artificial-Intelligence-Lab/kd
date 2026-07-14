
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.core.platform.builder import _resolve_derivative_requirements
from kd.core.platform.requirements import DerivativeReqs
from kd.search.discover.plugin import DISCOVERPlugin
from kd.search.protocol import (
    IterativeSearchAlgorithm,
    PlatformComponents,
    SearchAlgorithm,
)





SEED = 42
BATCH_SIZE = 16






class MockEvaluator:

    def __init__(self, default_nmse: float = 0.5) -> None:
        self.default_nmse = default_nmse
        self.evaluated: list[str] = []

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        self.evaluated.append(expr)
        return EvaluationResult(
            mse=self.default_nmse,
            nmse=self.default_nmse,
            r2=max(0.0, 1.0 - self.default_nmse),
            complexity=3,
            is_valid=True,
            expression=expr,
        )







@pytest.fixture
def mock_evaluator() -> MockEvaluator:
    return MockEvaluator(default_nmse=0.5)


@pytest.fixture
def mock_components(mock_evaluator: MockEvaluator) -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=mock_evaluator,
        context=MagicMock(),
        registry=MagicMock(),
        recorder=None,
    )


@pytest.fixture
def plugin(mock_components: PlatformComponents) -> DISCOVERPlugin:
    torch.manual_seed(SEED)
    p = DISCOVERPlugin()
    p.prepare(mock_components)
    return p







class TestPrepare:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_prepare_enables_propose(
        self, mock_components: PlatformComponents,
    ) -> None:
        torch.manual_seed(SEED)
        plugin = DISCOVERPlugin()
        plugin.prepare(mock_components)
        candidates = plugin.propose(BATCH_SIZE)
        assert isinstance(candidates, list)
        assert len(candidates) > 0

    @pytest.mark.unit
    def test_propose_before_prepare_raises(self) -> None:
        plugin = DISCOVERPlugin()
        with pytest.raises((RuntimeError, AttributeError)):
            plugin.propose(BATCH_SIZE)







class TestSearchCycle:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_full_cycle_completes(self, plugin: DISCOVERPlugin) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        plugin.update(results)

    @pytest.mark.unit
    def test_propose_returns_strings(self, plugin: DISCOVERPlugin) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        assert isinstance(candidates, list)
        for c in candidates:
            assert isinstance(c, str)
            assert len(c) > 0

    @pytest.mark.unit
    def test_propose_length_le_n(self, plugin: DISCOVERPlugin) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        assert len(candidates) <= BATCH_SIZE

    @pytest.mark.unit
    def test_evaluate_returns_results(self, plugin: DISCOVERPlugin) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        assert isinstance(results, list)
        assert len(results) == len(candidates)
        for r in results:
            assert isinstance(r, EvaluationResult)

    @pytest.mark.unit
    def test_evaluate_delegates_to_evaluator(
        self, plugin: DISCOVERPlugin, mock_evaluator: MockEvaluator,
    ) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        assert len(mock_evaluator.evaluated) == 0
        plugin.evaluate(candidates)
        assert len(mock_evaluator.evaluated) == len(candidates)

    @pytest.mark.unit
    def test_two_cycles_work(self, plugin: DISCOVERPlugin) -> None:
        for _ in range(2):
            torch.manual_seed(SEED)
            candidates = plugin.propose(BATCH_SIZE)
            results = plugin.evaluate(candidates)
            plugin.update(results)







class TestBestTracking:

    @pytest.mark.unit
    def test_best_score_positive_after_cycle(
        self, plugin: DISCOVERPlugin,
    ) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        plugin.update(results)
        assert isinstance(plugin.best_score, float)
        assert plugin.best_score > 0.0

    @pytest.mark.unit
    def test_best_expression_nonempty_after_cycle(
        self, plugin: DISCOVERPlugin,
    ) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        plugin.update(results)
        assert isinstance(plugin.best_expression, str)
        assert len(plugin.best_expression) > 0

    @pytest.mark.unit
    def test_best_score_non_negative_initially(
        self, plugin: DISCOVERPlugin,
    ) -> None:
        assert plugin.best_score >= 0.0







class TestBetweenIterations:

    @pytest.mark.unit
    def test_between_iterations_callable(
        self, plugin: DISCOVERPlugin,
    ) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        plugin.between_iterations()







class TestConfig:

    @pytest.mark.unit
    def test_config_is_json_safe(self, plugin: DISCOVERPlugin) -> None:
        config = plugin.config
        assert isinstance(config, dict)

        json.dumps(config)







class TestStateManagement:

    @pytest.mark.unit
    def test_state_is_dict(self, plugin: DISCOVERPlugin) -> None:
        state = plugin.state
        assert isinstance(state, dict)

    @pytest.mark.unit
    def test_state_roundtrip_preserves_best(
        self, plugin: DISCOVERPlugin,
    ) -> None:
        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        saved_score = plugin.best_score
        saved_expr = plugin.best_expression
        saved_state = plugin.state


        candidates2 = plugin.propose(BATCH_SIZE)
        results2 = plugin.evaluate(candidates2)
        plugin.update(results2)


        plugin.state = saved_state
        assert plugin.best_score == saved_score
        assert plugin.best_expression == saved_expr

    @pytest.mark.unit
    def test_state_is_pickle_serializable(
        self, plugin: DISCOVERPlugin,
    ) -> None:
        import pickle

        torch.manual_seed(SEED)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        state = plugin.state

        data = pickle.dumps(state)
        restored = pickle.loads(data)
        assert isinstance(restored, dict)







class TestProtocolCompliance:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_isinstance_search_algorithm(
        self, plugin: DISCOVERPlugin,
    ) -> None:
        assert isinstance(plugin, SearchAlgorithm)

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_isinstance_iterative(self, plugin: DISCOVERPlugin) -> None:
        assert isinstance(plugin, IterativeSearchAlgorithm)







class MockEvaluatorWithTerms:

    def __init__(self, default_nmse: float = 0.3) -> None:
        self.default_nmse = default_nmse

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        return EvaluationResult(
            mse=self.default_nmse,
            nmse=self.default_nmse,
            r2=max(0.0, 1.0 - self.default_nmse),
            complexity=2,
            is_valid=True,
            expression=expr,
            terms=["u_xx", "mul(u, u_x)", "u"],
            coefficients=torch.tensor([1.0, -1.0, 0.0]),
            selected_indices=[0, 1],
        )


class TestPluginBestResultCheckpoint:

    @pytest.mark.unit
    def test_state_includes_best_result_fields(
        self, mock_components: PlatformComponents,
    ) -> None:
        torch.manual_seed(SEED)
        plugin = DISCOVERPlugin()
        mock_components_with_terms = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MockEvaluatorWithTerms(),
            context=MagicMock(),
            registry=MagicMock(),
            recorder=None,
        )
        plugin.prepare(mock_components_with_terms)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        state = plugin.state
        engine_state = state["engine_state"]
        assert "best_result_terms" in engine_state
        assert "best_result_coefficients" in engine_state

    @pytest.mark.unit
    def test_state_roundtrip_preserves_best_result_fields(
        self, mock_components: PlatformComponents,
    ) -> None:
        torch.manual_seed(SEED)
        plugin = DISCOVERPlugin()
        mock_components_with_terms = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MockEvaluatorWithTerms(),
            context=MagicMock(),
            registry=MagicMock(),
            recorder=None,
        )
        plugin.prepare(mock_components_with_terms)
        candidates = plugin.propose(BATCH_SIZE)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        saved_state = plugin.state
        engine_state = saved_state["engine_state"]
        saved_terms = engine_state.get("best_result_terms")
        saved_coeffs = engine_state.get("best_result_coefficients")

        assert saved_terms is not None, "terms should be present after run"
        assert saved_coeffs is not None, "coeffs should be present after run"


        candidates2 = plugin.propose(BATCH_SIZE)
        results2 = plugin.evaluate(candidates2)
        plugin.update(results2)


        plugin.state = saved_state
        restored_state = plugin.state
        restored_engine = restored_state["engine_state"]
        assert restored_engine.get("best_result_terms") == saved_terms
        restored_coeffs = restored_engine.get("best_result_coefficients")
        assert restored_coeffs is not None
        assert restored_coeffs == pytest.approx(saved_coeffs)

    @pytest.mark.unit
    def test_old_checkpoint_without_best_result_fields(
        self, plugin: DISCOVERPlugin,
    ) -> None:
        state = plugin.state
        engine_state = state["engine_state"]

        engine_state.pop("best_result_terms", None)
        engine_state.pop("best_result_coefficients", None)


        plugin.state = state















class MockEvaluatorWithTarget:

    def __init__(self) -> None:
        self.evaluated: list[str] = []
        self._lhs = torch.arange(4, dtype=torch.float32) + 0.5

    @property
    def lhs_target(self) -> Tensor:
        return self._lhs.detach()

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        self.evaluated.append(expr)
        mse = float(len(expr))
        return EvaluationResult(
            mse=mse,
            nmse=mse,
            r2=1.0 - mse,
            complexity=2,
            is_valid=True,
            expression=expr,
            terms=["u_xx", "mul(u, u_x)"],
            coefficients=torch.tensor([1.0, -1.0]),
            selected_indices=[0, 1],
            residuals=torch.full((4,), mse),
        )


@pytest.fixture
def components_with_target() -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=MockEvaluatorWithTarget(),
        context=MagicMock(),
        registry=MagicMock(),
        recorder=None,
    )


@pytest.fixture
def prepared_with_target(
    components_with_target: PlatformComponents,
) -> tuple[DISCOVERPlugin, MockEvaluatorWithTarget]:
    torch.manual_seed(SEED)
    p = DISCOVERPlugin()
    p.prepare(components_with_target)
    candidates = p.propose(BATCH_SIZE)
    results = p.evaluate(candidates)
    p.update(results)
    evaluator = components_with_target.evaluator
    assert isinstance(evaluator, MockEvaluatorWithTarget)
    return p, evaluator


class TestBuildFinalResult:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_isinstance_merged_search_algorithm(
        self, plugin: DISCOVERPlugin
    ) -> None:
        assert isinstance(plugin, SearchAlgorithm)
        assert callable(plugin.build_final_result)

    @pytest.mark.unit
    def test_build_final_result_returns_evaluation_result(
        self, prepared_with_target: tuple[DISCOVERPlugin, MockEvaluatorWithTarget],
    ) -> None:
        plugin, _ = prepared_with_target
        result = plugin.build_final_result()
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.residuals, Tensor)

    @pytest.mark.unit
    def test_build_final_result_parity_with_fallback(
        self, prepared_with_target: tuple[DISCOVERPlugin, MockEvaluatorWithTarget],
    ) -> None:
        plugin, evaluator = prepared_with_target
        best_expr = plugin.best_expression
        before = len(evaluator.evaluated)

        built = plugin.build_final_result()

        assert isinstance(built, EvaluationResult)

        assert len(evaluator.evaluated) == before + 1
        assert evaluator.evaluated[-1] == best_expr
        assert built.expression == best_expr
        assert built.mse == float(len(best_expr))

        fallback = evaluator.evaluate_expression(best_expr)
        assert built.mse == fallback.mse
        assert built.nmse == fallback.nmse
        assert built.r2 == fallback.r2
        assert built.is_valid == fallback.is_valid
        assert built.terms == fallback.terms
        assert built.selected_indices == fallback.selected_indices
        assert built.coefficients is not None
        assert fallback.coefficients is not None
        assert torch.equal(built.coefficients, fallback.coefficients)
        assert built.residuals is not None
        assert fallback.residuals is not None
        assert torch.equal(built.residuals, fallback.residuals)

    @pytest.mark.unit
    def test_build_final_result_before_prepare_raises(self) -> None:
        plugin = DISCOVERPlugin()
        with pytest.raises(RuntimeError):
            plugin.build_final_result()


class TestBuildResultTarget:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_has_build_result_target(
        self, plugin: DISCOVERPlugin,
    ) -> None:
        assert callable(plugin.build_result_target)

    @pytest.mark.unit
    def test_build_result_target_parity_with_fallback(
        self, prepared_with_target: tuple[DISCOVERPlugin, MockEvaluatorWithTarget],
    ) -> None:
        plugin, evaluator = prepared_with_target
        target = plugin.build_result_target()
        assert isinstance(target, Tensor)
        assert torch.equal(target, evaluator.lhs_target)

    @pytest.mark.unit
    def test_build_result_target_is_detached_and_independent(
        self, prepared_with_target: tuple[DISCOVERPlugin, MockEvaluatorWithTarget],
    ) -> None:
        plugin, evaluator = prepared_with_target
        target = plugin.build_result_target()
        assert target.requires_grad is False
        target += 1.0
        assert not torch.equal(evaluator.lhs_target, target)

    @pytest.mark.unit
    def test_build_result_target_before_prepare_raises(self) -> None:
        plugin = DISCOVERPlugin()
        with pytest.raises(RuntimeError):
            plugin.build_result_target()


class TestDerivativeRequirements:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_derivative_requirements_is_a_property(
        self, plugin: DISCOVERPlugin,
    ) -> None:
        descriptor = type(plugin).__dict__["derivative_requirements"]
        assert isinstance(descriptor, property)

    @pytest.mark.unit
    def test_derivative_requirements_returns_derivative_reqs(
        self, plugin: DISCOVERPlugin,
    ) -> None:
        reqs = plugin.derivative_requirements
        assert isinstance(reqs, DerivativeReqs)
        assert reqs.provider_kind == "finite_diff"
        assert reqs.max_atomic_order == 2
        assert reqs.lhs_order == 1
        assert reqs.needs_surrogate is False

    @pytest.mark.unit
    def test_resolve_derivative_requirements_does_not_typeerror(
        self, plugin: DISCOVERPlugin,
    ) -> None:
        reqs = _resolve_derivative_requirements(plugin)
        assert isinstance(reqs, DerivativeReqs)
        assert reqs.provider_kind == "finite_diff"
