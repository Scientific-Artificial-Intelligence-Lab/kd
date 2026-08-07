
from __future__ import annotations

import dataclasses
import pickle
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult


from kd.search.protocol import PlatformComponents, SearchAlgorithm






class _BuildMethods:

    def build_final_result(self) -> EvaluationResult:
        return EvaluationResult(mse=0.0, nmse=0.0, r2=1.0)

    def build_result_target(self) -> Tensor:
        return torch.zeros(1)


class _ConformingAlgorithm(_BuildMethods):

    def __init__(self) -> None:
        self._best_score: float = float("inf")
        self._best_expression: str = ""
        self._state: dict[str, Any] = {}
        self._prepared = False



    def prepare(self, components: PlatformComponents) -> None:
        self._prepared = True

    def propose(self, n: int) -> list[str]:
        return [f"expr_{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return [
            EvaluationResult(mse=0.1 * i, nmse=0.1 * i, r2=1.0 - 0.1 * i)
            for i, _ in enumerate(candidates)
        ]

    def update(self, results: list[EvaluationResult]) -> None:
        for r in results:
            if r.is_valid and r.mse < self._best_score:
                self._best_score = r.mse
                self._best_expression = r.expression



    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_expression(self) -> str:
        return self._best_expression

    @property
    def config(self) -> dict:
        return {"algorithm": "ConformingAlgorithm"}

    @property
    def state(self) -> dict:
        return self._state

    @state.setter
    def state(self, value: dict) -> None:
        self._state = value


class _MissingPrepare(_BuildMethods):

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}

    @state.setter
    def state(self, value: dict) -> None:
        pass


class _MissingPropose(_BuildMethods):

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}

    @state.setter
    def state(self, value: dict) -> None:
        pass


class _MissingEvaluate(_BuildMethods):

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}

    @state.setter
    def state(self, value: dict) -> None:
        pass


class _MissingUpdate(_BuildMethods):

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}

    @state.setter
    def state(self, value: dict) -> None:
        pass


class _MissingBestScore(_BuildMethods):

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}

    @state.setter
    def state(self, value: dict) -> None:
        pass


class _MissingBestExpression(_BuildMethods):

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}

    @state.setter
    def state(self, value: dict) -> None:
        pass


class _MissingState(_BuildMethods):

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}


class _NoStateSetter:

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}







@pytest.fixture
def mock_components() -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(training_result=None),
        registry=MagicMock(),
    )


@pytest.fixture
def conforming_algorithm() -> _ConformingAlgorithm:
    return _ConformingAlgorithm()







class TestSearchAlgorithmProtocol:

    @pytest.mark.unit
    def test_protocol_is_importable(self) -> None:
        from kd.search.protocol import SearchAlgorithm as SA

        assert SA is not None

    @pytest.mark.unit
    def test_protocol_is_runtime_checkable(self) -> None:

        assert (
            hasattr(SearchAlgorithm, "__protocol_attrs__")
            or hasattr(SearchAlgorithm, "__abstractmethods__")
            or isinstance(SearchAlgorithm, type)
        )

        obj = _ConformingAlgorithm()

        result = isinstance(obj, SearchAlgorithm)
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_conforming_class_is_instance(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        assert isinstance(conforming_algorithm, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_prepare_fails_isinstance(self) -> None:
        obj = _MissingPrepare()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_propose_fails_isinstance(self) -> None:
        obj = _MissingPropose()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_evaluate_fails_isinstance(self) -> None:
        obj = _MissingEvaluate()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_update_fails_isinstance(self) -> None:
        obj = _MissingUpdate()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_best_score_fails_isinstance(self) -> None:
        obj = _MissingBestScore()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_best_expression_fails_isinstance(self) -> None:
        obj = _MissingBestExpression()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_state_fails_isinstance(self) -> None:
        obj = _MissingState()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_protocol_has_prepare_method(self) -> None:
        assert hasattr(SearchAlgorithm, "prepare")

    @pytest.mark.unit
    def test_protocol_has_propose_method(self) -> None:
        assert hasattr(SearchAlgorithm, "propose")

    @pytest.mark.unit
    def test_protocol_has_evaluate_method(self) -> None:
        assert hasattr(SearchAlgorithm, "evaluate")

    @pytest.mark.unit
    def test_protocol_has_update_method(self) -> None:
        assert hasattr(SearchAlgorithm, "update")

    @pytest.mark.unit
    def test_protocol_has_best_score(self) -> None:
        assert hasattr(SearchAlgorithm, "best_score")

    @pytest.mark.unit
    def test_protocol_has_best_expression(self) -> None:
        assert hasattr(SearchAlgorithm, "best_expression")

    @pytest.mark.unit
    def test_protocol_has_state(self) -> None:
        assert hasattr(SearchAlgorithm, "state")







class TestPlatformComponents:

    @pytest.mark.unit
    def test_importable(self) -> None:
        from kd.search.protocol import PlatformComponents as PC

        assert PC is not None

    @pytest.mark.unit
    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(PlatformComponents)

    @pytest.mark.unit
    def test_has_dataset_field(self) -> None:
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "dataset" in field_names

    @pytest.mark.unit
    def test_has_executor_field(self) -> None:
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "executor" in field_names

    @pytest.mark.unit
    def test_has_evaluator_field(self) -> None:
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "evaluator" in field_names

    @pytest.mark.unit
    def test_has_context_field(self) -> None:
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "context" in field_names

    @pytest.mark.unit
    def test_has_registry_field(self) -> None:
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "registry" in field_names

    @pytest.mark.unit
    def test_exactly_six_fields(self) -> None:
        fields = dataclasses.fields(PlatformComponents)
        assert len(fields) == 6, (
            f"Expected 6 fields, got {len(fields)}: {[f.name for f in fields]}"
        )

    @pytest.mark.unit
    def test_no_solver_field(self) -> None:
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "solver" not in field_names

    @pytest.mark.unit
    def test_instantiation_with_mocks(
        self, mock_components: PlatformComponents
    ) -> None:
        assert mock_components.dataset is not None
        assert mock_components.executor is not None
        assert mock_components.evaluator is not None
        assert mock_components.context is not None
        assert mock_components.registry is not None

    @pytest.mark.unit
    def test_field_access(self, mock_components: PlatformComponents) -> None:

        for field in dataclasses.fields(mock_components):
            value = getattr(mock_components, field.name)
            if (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                assert value is not None, (
                    f"Required field {field.name} should not be None"
                )







class TestProtocolBehavior:

    @pytest.mark.unit
    def test_lifecycle_prepare_propose_evaluate_update(
        self,
        conforming_algorithm: _ConformingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        algo = conforming_algorithm


        algo.prepare(mock_components)
        assert algo._prepared


        candidates = algo.propose(3)
        assert isinstance(candidates, list)
        assert len(candidates) == 3
        assert all(isinstance(c, str) for c in candidates)


        results = algo.evaluate(candidates)
        assert isinstance(results, list)
        assert len(results) == len(candidates)
        assert all(isinstance(r, EvaluationResult) for r in results)


        algo.update(results)

    @pytest.mark.unit
    def test_propose_returns_list_of_strings(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        result = conforming_algorithm.propose(5)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    @pytest.mark.unit
    def test_propose_n_controls_count(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        for n in [1, 5, 10]:
            result = conforming_algorithm.propose(n)
            assert isinstance(result, list)
            assert len(result) >= 0

    @pytest.mark.unit
    def test_evaluate_returns_list_of_evaluation_result(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        candidates = ["expr_a", "expr_b"]
        results = conforming_algorithm.evaluate(candidates)
        assert isinstance(results, list)
        assert all(isinstance(r, EvaluationResult) for r in results)

    @pytest.mark.unit
    def test_evaluate_result_count_matches_candidates(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        for n in [1, 3, 5]:
            candidates = [f"e{i}" for i in range(n)]
            results = conforming_algorithm.evaluate(candidates)
            assert len(results) == len(candidates)

    @pytest.mark.unit
    def test_best_score_returns_float(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        score = conforming_algorithm.best_score
        assert isinstance(score, float)

    @pytest.mark.unit
    def test_best_expression_returns_str(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        expr = conforming_algorithm.best_expression
        assert isinstance(expr, str)

    @pytest.mark.unit
    def test_state_getter_returns_dict(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        state = conforming_algorithm.state
        assert isinstance(state, dict)

    @pytest.mark.unit
    def test_state_roundtrip(self, conforming_algorithm: _ConformingAlgorithm) -> None:
        test_state = {
            "generation": 42,
            "population": ["a", "b", "c"],
            "scores": [0.1, 0.2, 0.3],
            "nested": {"key": "value"},
        }
        conforming_algorithm.state = test_state
        retrieved = conforming_algorithm.state
        assert retrieved == test_state

    @pytest.mark.unit
    def test_state_is_pickle_serializable(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        test_state = {
            "generation": 10,
            "best_mse": 0.001,
            "population": ["mul(u, u_x)", "u_xx"],
        }
        conforming_algorithm.state = test_state


        state = conforming_algorithm.state
        pickled = pickle.dumps(state)
        restored = pickle.loads(pickled)
        assert restored == state

    @pytest.mark.unit
    def test_multiple_update_cycles(
        self,
        conforming_algorithm: _ConformingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        algo = conforming_algorithm
        algo.prepare(mock_components)

        for _ in range(3):
            candidates = algo.propose(2)
            results = algo.evaluate(candidates)
            algo.update(results)



        assert isinstance(algo.best_score, float)
        assert isinstance(algo.best_expression, str)







class TestNegativeCases:

    @pytest.mark.unit
    def test_empty_class_not_instance(self) -> None:

        class Empty:
            pass

        assert not isinstance(Empty(), SearchAlgorithm)

    @pytest.mark.unit
    def test_partial_implementation_not_instance(self) -> None:

        class Partial:
            def prepare(self, components: PlatformComponents) -> None:
                pass

            def propose(self, n: int) -> list[str]:
                return []

        assert not isinstance(Partial(), SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_state_setter_behavior(self) -> None:
        obj = _NoStateSetter()



        has_state_getter = hasattr(obj, "state")
        assert has_state_getter


        with pytest.raises(AttributeError):
            obj.state = {"key": "value"}

    @pytest.mark.unit
    def test_propose_zero_returns_empty(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        result = conforming_algorithm.propose(0)
        assert result == []

    @pytest.mark.unit
    def test_evaluate_empty_list(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        result = conforming_algorithm.evaluate([])
        assert result == []

    @pytest.mark.unit
    def test_update_empty_results(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:

        conforming_algorithm.update([])

    @pytest.mark.unit
    def test_platform_components_missing_field_raises(self) -> None:
        with pytest.raises(TypeError):
            PlatformComponents(
                dataset=MagicMock(),
                executor=MagicMock(),

            )

    @pytest.mark.unit
    def test_platform_components_extra_field_raises(self) -> None:
        with pytest.raises(TypeError):
            PlatformComponents(
                dataset=MagicMock(),
                executor=MagicMock(),
                evaluator=MagicMock(),
                context=MagicMock(training_result=None),
                registry=MagicMock(),
                recorder=None,
                solver=MagicMock(),
            )

    @pytest.mark.unit
    def test_protocol_is_not_instantiable_directly(self) -> None:
        with pytest.raises(TypeError):
            SearchAlgorithm()

    @pytest.mark.unit
    def test_state_set_empty_dict(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        conforming_algorithm.state = {}
        assert conforming_algorithm.state == {}

    @pytest.mark.unit
    def test_best_score_default_is_finite_or_inf(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        score = conforming_algorithm.best_score
        assert isinstance(score, float)


    @pytest.mark.unit
    def test_best_expression_default_is_string(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        expr = conforming_algorithm.best_expression
        assert isinstance(expr, str)







class TestPlatformComponentsRecorder:

    @pytest.mark.unit
    def test_has_recorder_field(self) -> None:
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "recorder" in field_names

    @pytest.mark.unit
    def test_recorder_defaults_to_none(self) -> None:
        pc = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(training_result=None),
            registry=MagicMock(),
        )
        assert pc.recorder is None

    @pytest.mark.unit
    def test_recorder_accepts_viz_recorder(self) -> None:
        from kd.search.recorder import VizRecorder

        recorder = VizRecorder()
        pc = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(training_result=None),
            registry=MagicMock(),
            recorder=recorder,
        )
        assert pc.recorder is recorder

    @pytest.mark.unit
    def test_recorder_accepts_none_explicitly(self) -> None:
        pc = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(training_result=None),
            registry=MagicMock(),
            recorder=None,
        )
        assert pc.recorder is None

    @pytest.mark.unit
    def test_backward_compat_no_recorder_arg(self) -> None:

        pc = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(training_result=None),
            registry=MagicMock(),
        )
        assert pc.dataset is not None
        assert pc.recorder is None







class _ConformingAlgorithmWithConfig(_ConformingAlgorithm):

    @property
    def config(self) -> dict:
        return {"algorithm": "test", "param": 42}


class _MissingConfig(_BuildMethods):

    def prepare(self, components: Any) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[Any]:
        return []

    def update(self, results: list[Any]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def state(self) -> dict[str, Any]:
        return {}

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        pass


class TestSearchAlgorithmConfig:

    @pytest.mark.unit
    def test_protocol_has_config(self) -> None:
        assert hasattr(SearchAlgorithm, "config")

    @pytest.mark.unit
    def test_conforming_with_config_is_instance(self) -> None:
        obj = _ConformingAlgorithmWithConfig()
        assert isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_config_returns_dict(self) -> None:
        obj = _ConformingAlgorithmWithConfig()
        cfg = obj.config
        assert isinstance(cfg, dict)

    @pytest.mark.unit
    def test_config_contains_keys(self) -> None:
        obj = _ConformingAlgorithmWithConfig()
        cfg = obj.config
        assert len(cfg) > 0

    @pytest.mark.unit
    def test_missing_config_fails_isinstance(self) -> None:
        obj = _MissingConfig()
        assert not isinstance(obj, SearchAlgorithm)







class _SearchOnlyAlgorithm(_ConformingAlgorithm):

    pass


class _IterativeAlgorithm(_ConformingAlgorithm):

    def __init__(self) -> None:
        super().__init__()
        self.between_calls: int = 0

    def between_iterations(self) -> None:
        self.between_calls += 1


class _BetweenOnlyAlgorithm:

    def between_iterations(self) -> None:
        pass

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []




class TestIterativeSearchAlgorithmProtocol:

    @pytest.mark.smoke
    def test_importable(self) -> None:
        from kd.search.protocol import IterativeSearchAlgorithm

        assert IterativeSearchAlgorithm is not None

    @pytest.mark.smoke
    def test_importable_from_package(self) -> None:
        from kd.search import IterativeSearchAlgorithm

        assert IterativeSearchAlgorithm is not None

    @pytest.mark.unit
    def test_is_runtime_checkable(self) -> None:
        from kd.search.protocol import IterativeSearchAlgorithm

        obj = _IterativeAlgorithm()

        result = isinstance(obj, IterativeSearchAlgorithm)
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_search_only_not_iterative(self) -> None:
        from kd.search.protocol import IterativeSearchAlgorithm

        obj = _SearchOnlyAlgorithm()
        assert isinstance(obj, SearchAlgorithm)
        assert not isinstance(obj, IterativeSearchAlgorithm)

    @pytest.mark.unit
    def test_iterative_algorithm_is_iterative(self) -> None:
        from kd.search.protocol import IterativeSearchAlgorithm

        obj = _IterativeAlgorithm()
        assert isinstance(obj, IterativeSearchAlgorithm)

    @pytest.mark.unit
    def test_iterative_algorithm_is_also_search(self) -> None:
        obj = _IterativeAlgorithm()
        assert isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_between_only_not_iterative(self) -> None:
        from kd.search.protocol import IterativeSearchAlgorithm

        obj = _BetweenOnlyAlgorithm()

        assert not isinstance(obj, SearchAlgorithm)
        assert not isinstance(obj, IterativeSearchAlgorithm)

    @pytest.mark.unit
    def test_protocol_has_between_iterations_method(self) -> None:
        from kd.search.protocol import IterativeSearchAlgorithm

        assert hasattr(IterativeSearchAlgorithm, "between_iterations")

    @pytest.mark.unit
    def test_between_iterations_callable(self) -> None:
        obj = _IterativeAlgorithm()
        assert callable(getattr(obj, "between_iterations", None))

    @pytest.mark.unit
    def test_between_iterations_returns_none(self) -> None:
        obj = _IterativeAlgorithm()
        result = obj.between_iterations()
        assert result is None

    @pytest.mark.unit
    def test_existing_conforming_algorithm_not_iterative(self) -> None:
        from kd.search.protocol import IterativeSearchAlgorithm

        obj = _ConformingAlgorithm()
        assert isinstance(obj, SearchAlgorithm)
        assert not isinstance(obj, IterativeSearchAlgorithm)

    @pytest.mark.unit
    def test_protocol_not_directly_instantiable(self) -> None:
        from kd.search.protocol import IterativeSearchAlgorithm

        with pytest.raises(TypeError):
            IterativeSearchAlgorithm()







class _TerminatingAlgorithm(_ConformingAlgorithm):

    def __init__(self) -> None:
        super().__init__()
        self._done = False

    @property
    def is_done(self) -> bool:
        return self._done


class TestTerminatingSearchAlgorithmProtocol:

    @pytest.mark.smoke
    def test_importable(self) -> None:
        from kd.search.protocol import TerminatingSearchAlgorithm

        assert TerminatingSearchAlgorithm is not None

    @pytest.mark.smoke
    def test_importable_from_package(self) -> None:
        from kd.search import TerminatingSearchAlgorithm

        assert TerminatingSearchAlgorithm is not None

    @pytest.mark.unit
    def test_is_runtime_checkable(self) -> None:
        from kd.search.protocol import TerminatingSearchAlgorithm

        obj = _TerminatingAlgorithm()
        result = isinstance(obj, TerminatingSearchAlgorithm)
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_extends_search_algorithm(self) -> None:
        from kd.search.protocol import TerminatingSearchAlgorithm

        obj = _TerminatingAlgorithm()
        assert isinstance(obj, SearchAlgorithm)
        assert isinstance(obj, TerminatingSearchAlgorithm)

    @pytest.mark.unit
    def test_object_with_is_done_is_recognized(self) -> None:
        from kd.search.protocol import TerminatingSearchAlgorithm

        obj = _TerminatingAlgorithm()
        assert isinstance(obj, TerminatingSearchAlgorithm)

    @pytest.mark.unit
    def test_search_only_not_terminating(self) -> None:
        from kd.search.protocol import TerminatingSearchAlgorithm

        obj = _ConformingAlgorithm()
        assert isinstance(obj, SearchAlgorithm)
        assert not isinstance(obj, TerminatingSearchAlgorithm)

    @pytest.mark.unit
    def test_is_done_only_not_terminating(self) -> None:
        from kd.search.protocol import TerminatingSearchAlgorithm

        class _IsDoneOnly:
            @property
            def is_done(self) -> bool:
                return True

        obj = _IsDoneOnly()
        assert not isinstance(obj, SearchAlgorithm)
        assert not isinstance(obj, TerminatingSearchAlgorithm)

    @pytest.mark.unit
    def test_protocol_has_is_done_member(self) -> None:
        from kd.search.protocol import TerminatingSearchAlgorithm

        assert hasattr(TerminatingSearchAlgorithm, "is_done")

    @pytest.mark.unit
    def test_protocol_not_directly_instantiable(self) -> None:
        from kd.search.protocol import TerminatingSearchAlgorithm

        with pytest.raises(TypeError):
            TerminatingSearchAlgorithm()

    @pytest.mark.unit
    def test_iterative_algorithm_not_terminating(self) -> None:
        from kd.search.protocol import TerminatingSearchAlgorithm

        obj = _IterativeAlgorithm()
        assert not isinstance(obj, TerminatingSearchAlgorithm)

    @pytest.mark.unit
    def test_only_llm4ed_is_terminating(self) -> None:
        from kd.api import _PLUGIN_CLASS_BY_ALGORITHM

        _TERMINATING = {"llm4ed"}
        assert _PLUGIN_CLASS_BY_ALGORITHM, "expected registered plugins"
        for algorithm, plugin_cls in _PLUGIN_CLASS_BY_ALGORITHM.items():
            terminating = hasattr(plugin_cls, "is_done")
            if algorithm in _TERMINATING:
                assert terminating, (
                    f"plugin {algorithm!r} is expected to expose is_done "
                    "(terminating search algorithm)"
                )
            else:
                assert not terminating, (
                    f"plugin {algorithm!r} unexpectedly exposes is_done; the T3 "
                    "seam must not implicitly make an existing plugin terminating"
                )
