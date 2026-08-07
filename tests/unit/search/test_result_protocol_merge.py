
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor, nn

from kd.core.evaluator import EvaluationResult
from kd.core.platform.builder import PlatformBuilder
from kd.api import _PLUGIN_CLASS_BY_ALGORITHM
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.discover import DISCOVERPlugin
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.llm4ed.config import Llm4edConfig
from kd.search.protocol import (
    IterativeSearchAlgorithm,
    PlatformComponents,
    SearchAlgorithm,
)
from kd.search.pysr import PySRPlugin
from kd.search.runner import _SEARCH_ALGORITHM_MEMBERS, ExperimentRunner
from kd.search.sga import SGAPlugin



_MERGED_PROTOCOL_MEMBERS: tuple[str, ...] = (
    "prepare",
    "propose",
    "evaluate",
    "update",
    "best_score",
    "best_expression",
    "config",
    "state",
    "build_final_result",
    "build_result_target",
)

_REGISTERED_PLUGIN_CASES = tuple(
    pytest.param(plugin_cls, id=algorithm)
    for algorithm, plugin_cls in _PLUGIN_CLASS_BY_ALGORITHM.items()
)


def _make_registered_plugin(plugin_cls: type) -> Any:
    if plugin_cls.config_cls is EqGPTConfig:
        plugin = plugin_cls(EqGPTConfig(sparsity_alpha=0.02))
        plugin.state = {"pending": True}
        return plugin
    if plugin_cls.config_cls is Llm4edConfig:
        plugin = plugin_cls(Llm4edConfig())
        plugin.state = {"pending": True}
        return plugin
    return plugin_cls()


def _make_algorithm_class(
    *,
    exclude: tuple[str, ...] = (),
    final_eval: EvaluationResult | None = None,
    target: Tensor | None = None,
) -> type:
    built_eval = (
        final_eval
        if final_eval is not None
        else EvaluationResult(mse=0.5, nmse=0.5, r2=0.5)
    )
    built_target = target if target is not None else torch.zeros(3)

    def prepare(self: Any, components: PlatformComponents) -> None:
        self.prepared_with = components

    def propose(self: Any, n: int) -> list[str]:
        return [f"fake_expr_{i}" for i in range(n)]

    def evaluate(self: Any, candidates: list[str]) -> list[EvaluationResult]:
        return [EvaluationResult(mse=1.0, nmse=1.0, r2=0.0) for _ in candidates]

    def update(self: Any, results: list[EvaluationResult]) -> None:
        return None

    def build_final_result(self: Any) -> EvaluationResult:
        return built_eval

    def build_result_target(self: Any) -> Tensor:
        return built_target

    def _get_state(self: Any) -> dict[str, Any]:
        return {}

    def _set_state(self: Any, value: dict[str, Any]) -> None:
        return None

    members: dict[str, Any] = {
        "prepare": prepare,
        "propose": propose,
        "evaluate": evaluate,
        "update": update,
        "best_score": property(lambda self: 0.5),
        "best_expression": property(lambda self: "fake_expr_0"),
        "config": property(lambda self: {"algorithm": "protocol_merge_fake"}),
        "state": property(_get_state, _set_state),
        "build_final_result": build_final_result,
        "build_result_target": build_result_target,
    }
    for name in exclude:
        del members[name]





    return type("ProtocolMergeFake", (), members)


def _assert_member_listed(message: str, member: str) -> None:
    assert re.search(rf"\b{re.escape(member)}\b", message), (
        f"run() entry gate must name the missing member {member!r}, "
        f"got: {message}"
    )


def _assert_member_not_listed(message: str, member: str) -> None:
    assert not re.search(rf"\b{re.escape(member)}\b", message), (
        f"gate message must list only MISSING members, but names the "
        f"present member {member!r}: {message}"
    )


def _keyword_components(evaluator: Any) -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=evaluator,
        context=MagicMock(training_result=None),
        registry=MagicMock(),
    )







@pytest.mark.unit
class TestRunnerEntryGate:

    @pytest.mark.parametrize("member", _MERGED_PROTOCOL_MEMBERS)
    def test_missing_member_raises_typeerror_naming_it(
        self,
        member: str,
        mock_components: PlatformComponents,
    ) -> None:
        algorithm = _make_algorithm_class(exclude=(member,))()
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        with pytest.raises(TypeError) as exc_info:
            runner.run(mock_components)
        message = str(exc_info.value)
        _assert_member_listed(message, member)
        present_probe = (
            "build_result_target"
            if member != "build_result_target"
            else "build_final_result"
        )
        _assert_member_not_listed(message, present_probe)
        assert not hasattr(algorithm, "prepared_with"), (
            "gate must fire at run() entry, BEFORE algorithm.prepare() "
            "is invoked (-2 chokepoint)"
        )

    def test_missing_both_build_methods_lists_both_names(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        algorithm = _make_algorithm_class(
            exclude=("build_final_result", "build_result_target")
        )()
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        with pytest.raises(TypeError) as exc_info:
            runner.run(mock_components)
        message = str(exc_info.value)
        _assert_member_listed(message, "build_final_result")
        _assert_member_listed(message, "build_result_target")

        _assert_member_not_listed(message, "propose")
        _assert_member_not_listed(message, "best_expression")
        assert not hasattr(algorithm, "prepared_with"), (
            "gate must fire at run() entry, BEFORE algorithm.prepare() "
            "is invoked (-2 chokepoint)"
        )

    def test_gate_accepts_pre_prepare_fail_loud_state_getter(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        base = _make_algorithm_class()

        class _PrePrepareGuardedState(base):
            def prepare(self, components: PlatformComponents) -> None:
                self.prepared_with = components
                self._prepared = True

            @property
            def state(self) -> dict[str, Any]:
                if not getattr(self, "_prepared", False):
                    raise RuntimeError(
                        "prepare() must be called before using the plugin."
                    )
                return {}

            @state.setter
            def state(self, value: dict[str, Any]) -> None:
                return None

        runner = ExperimentRunner(
            algorithm=_PrePrepareGuardedState(), max_iterations=1
        )

        result = runner.run(mock_components)

        assert result.final_eval is not None

    def test_gate_accepts_instance_attribute_data_members(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        final_eval = EvaluationResult(mse=0.5, nmse=0.5, r2=0.5)
        target = torch.zeros(3)

        class _DataOnFields:
            def __init__(self) -> None:
                self.best_score = 0.5
                self.best_expression = "fake_expr_0"
                self.config: dict[str, Any] = {"algorithm": "inst_attr_fake"}
                self.state: dict[str, Any] = {}

            def prepare(self, components: PlatformComponents) -> None:
                return None

            def propose(self, n: int) -> list[str]:
                return [f"fake_expr_{i}" for i in range(n)]

            def evaluate(
                self, candidates: list[str]
            ) -> list[EvaluationResult]:
                return [
                    EvaluationResult(mse=1.0, nmse=1.0, r2=0.0)
                    for _ in candidates
                ]

            def update(self, results: list[EvaluationResult]) -> None:
                return None

            def build_final_result(self) -> EvaluationResult:
                return final_eval

            def build_result_target(self) -> Tensor:
                return target

        algorithm = _DataOnFields()

        assert isinstance(algorithm, SearchAlgorithm)
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        result = runner.run(mock_components)

        assert result.final_eval is final_eval

    def test_gate_rejects_inherited_protocol_stubs(
        self,
        mock_components: PlatformComponents,
    ) -> None:

        class _NominalLegacy(IterativeSearchAlgorithm):


            def __init__(self) -> None:
                self.prepared = False

            def prepare(self, components: PlatformComponents) -> None:
                self.prepared = True

            def propose(self, n: int) -> list[str]:
                return [f"fake_expr_{i}" for i in range(n)]

            def evaluate(
                self, candidates: list[str]
            ) -> list[EvaluationResult]:
                return [
                    EvaluationResult(mse=1.0, nmse=1.0, r2=0.0)
                    for _ in candidates
                ]

            def update(self, results: list[EvaluationResult]) -> None:
                return None

            def between_iterations(self) -> None:
                return None

            @property
            def best_score(self) -> float:
                return 0.5

            @property
            def best_expression(self) -> str:
                return "fake_expr_0"

            @property
            def config(self) -> dict[str, Any]:
                return {"algorithm": "nominal_legacy_fake"}

            @property
            def state(self) -> dict[str, Any]:
                return {}

            @state.setter
            def state(self, value: dict[str, Any]) -> None:
                return None

        algorithm = _NominalLegacy()
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        with pytest.raises(TypeError) as exc_info:
            runner.run(mock_components)
        message = str(exc_info.value)
        _assert_member_listed(message, "build_final_result")
        _assert_member_listed(message, "build_result_target")
        _assert_member_not_listed(message, "propose")
        assert not algorithm.prepared, (
            "gate must fire at run() entry, BEFORE algorithm.prepare() "
            "is invoked (-2 chokepoint)"
        )

    def test_gate_rejects_single_inherited_stub_naming_only_it(
        self,
        mock_components: PlatformComponents,
    ) -> None:

        class _HalfMigrated(IterativeSearchAlgorithm):
            def prepare(self, components: PlatformComponents) -> None:
                return None

            def propose(self, n: int) -> list[str]:
                return [f"fake_expr_{i}" for i in range(n)]

            def evaluate(
                self, candidates: list[str]
            ) -> list[EvaluationResult]:
                return [
                    EvaluationResult(mse=1.0, nmse=1.0, r2=0.0)
                    for _ in candidates
                ]

            def update(self, results: list[EvaluationResult]) -> None:
                return None

            def between_iterations(self) -> None:
                return None

            def build_result_target(self) -> Tensor:
                return torch.zeros(3)

            @property
            def best_score(self) -> float:
                return 0.5

            @property
            def best_expression(self) -> str:
                return "fake_expr_0"

            @property
            def config(self) -> dict[str, Any]:
                return {"algorithm": "half_migrated_fake"}

            @property
            def state(self) -> dict[str, Any]:
                return {}

            @state.setter
            def state(self, value: dict[str, Any]) -> None:
                return None

        runner = ExperimentRunner(algorithm=_HalfMigrated(), max_iterations=1)

        with pytest.raises(TypeError) as exc_info:
            runner.run(mock_components)
        message = str(exc_info.value)
        _assert_member_listed(message, "build_final_result")
        _assert_member_not_listed(message, "build_result_target")


@pytest.mark.unit
class TestProtocolMemberListSync:

    def test_gate_and_test_tuples_match_protocol_surface(self) -> None:
        from typing_extensions import get_protocol_members

        protocol_surface = set(get_protocol_members(SearchAlgorithm))
        assert set(_SEARCH_ALGORITHM_MEMBERS) == protocol_surface, (
            "runner._SEARCH_ALGORITHM_MEMBERS drifted from the "
            "SearchAlgorithm protocol surface -- update the gate tuple"
        )
        assert set(_MERGED_PROTOCOL_MEMBERS) == protocol_surface, (
            "_MERGED_PROTOCOL_MEMBERS drifted from the SearchAlgorithm "
            "protocol surface -- update this file's tuple"
        )







@pytest.mark.unit
class TestMergedProtocolSurface:

    @pytest.mark.parametrize("plugin_cls", _REGISTERED_PLUGIN_CASES)
    def test_builtin_plugin_satisfies_merged_protocol(
        self, plugin_cls: type
    ) -> None:
        plugin = _make_registered_plugin(plugin_cls)
        if plugin_cls is DISCOVERPlugin:
            plugin.prepare(_keyword_components(MagicMock()))
        assert isinstance(plugin, SearchAlgorithm)
        assert callable(plugin.build_final_result)
        assert callable(plugin.build_result_target)

    def test_legacy_surface_no_longer_satisfies_protocol(self) -> None:
        legacy = _make_algorithm_class(
            exclude=("build_final_result", "build_result_target")
        )()
        assert not isinstance(legacy, SearchAlgorithm)

    @pytest.mark.parametrize("member", _MERGED_PROTOCOL_MEMBERS)
    def test_missing_any_single_member_fails_isinstance(
        self, member: str
    ) -> None:
        incomplete = _make_algorithm_class(exclude=(member,))()
        assert not isinstance(incomplete, SearchAlgorithm), (
            f"an algorithm missing {member!r} must not satisfy the merged "
            "SearchAlgorithm protocol"
        )







class _RecordingEvaluatorStub:

    def __init__(self, result: EvaluationResult) -> None:
        self.calls: list[str] = []
        self._result = result

    def evaluate_expression(self, expression: str) -> EvaluationResult:
        self.calls.append(expression)
        return self._result


@pytest.mark.unit
class TestDefaultFinalResult:

    def test_delegates_to_evaluator_evaluate_expression(self) -> None:


        from kd.search.result import default_final_result

        sentinel = EvaluationResult(mse=0.25, nmse=0.5, r2=0.75)
        evaluator = _RecordingEvaluatorStub(sentinel)

        returned = default_final_result("u*diff_x(u)", evaluator)

        assert returned is sentinel
        assert evaluator.calls == ["u*diff_x(u)"]







@pytest.mark.unit
class TestPlatformComponentsKwOnly:

    def test_legacy_five_positional_args_rejected(self) -> None:
        with pytest.raises(TypeError):
            PlatformComponents(
                MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )

    def test_six_positional_args_rejected(self) -> None:
        with pytest.raises(TypeError):
            PlatformComponents(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                None,
            )

    def test_any_single_positional_arg_rejected(self) -> None:
        with pytest.raises(TypeError):
            PlatformComponents(
                MagicMock(),
                executor=MagicMock(),
                evaluator=MagicMock(),
                context=MagicMock(training_result=None),
                registry=MagicMock(),
            )

    def test_evaluator_omitted_defaults_to_none(self) -> None:
        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            context=MagicMock(training_result=None),
            registry=MagicMock(),
        )
        assert components.evaluator is None

    def test_evaluator_keyword_value_preserved(self) -> None:
        evaluator = MagicMock()
        components = _keyword_components(evaluator)
        assert components.evaluator is evaluator







class _PoisonEvaluator:

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(
            f"components.evaluator.{name} was accessed -- the runner must "
            "never consult the evaluator (acceptance criterion: "
            "'components.evaluator occurrences in runner.py = 0')"
        )


@pytest.mark.unit
class TestRunnerZeroEvaluatorContract:

    def test_run_completes_and_result_comes_from_build_methods(self) -> None:
        residuals = torch.tensor([0.1, -0.2, 0.3])
        target = torch.tensor([1.0, 2.0, 3.0])
        final_eval = EvaluationResult(
            mse=0.01, nmse=0.02, r2=0.99, residuals=residuals
        )
        algorithm = _make_algorithm_class(
            final_eval=final_eval, target=target
        )()
        components = _keyword_components(None)
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        result = runner.run(components)

        assert result.final_eval is final_eval
        torch.testing.assert_close(result.actual, target)
        torch.testing.assert_close(result.predicted, target + residuals)

    def test_present_evaluator_is_never_consulted(self) -> None:
        residuals = torch.tensor([0.1, -0.2, 0.3])
        target = torch.tensor([1.0, 2.0, 3.0])
        final_eval = EvaluationResult(
            mse=0.01, nmse=0.02, r2=0.99, residuals=residuals
        )
        algorithm = _make_algorithm_class(
            final_eval=final_eval, target=target
        )()
        components = _keyword_components(_PoisonEvaluator())
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        result = runner.run(components)

        assert result.final_eval is final_eval
        torch.testing.assert_close(result.actual, target)







@pytest.mark.unit
class TestPluginPrepareRequiresEvaluator:

    def test_pysr_prepare_rejects_none_evaluator_naming_algorithm(self) -> None:
        plugin = PySRPlugin()
        with pytest.raises(TypeError, match="(?i)pysr"):
            plugin.prepare(_keyword_components(None))

    def test_discover_prepare_rejects_none_evaluator_naming_algorithm(
        self,
    ) -> None:
        plugin = DISCOVERPlugin()
        with pytest.raises(TypeError, match="(?i)discover"):
            plugin.prepare(_keyword_components(None))







class _QuadraticSurrogate(nn.Module):

    def forward(
        self, *, x: torch.Tensor, t: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {"u": 1.0 + x * x + t * t}


def _quadratic_2d_dataset() -> PDEDataset:
    x = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = 1.0 + xg * xg + tg * tg
    return PDEDataset(
        name="protocol-merge-dlga",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.mark.unit
class TestPluginPrepareAcceptsNoneEvaluator:

    def test_sga_prepare_accepts_none_evaluator(
        self, simple_2d_dataset: PDEDataset
    ) -> None:
        plugin = SGAPlugin()
        components = replace(
            PlatformBuilder(
                simple_2d_dataset, plugin.derivative_requirements
            ).build(),
            evaluator=None,
        )

        plugin.prepare(components)

        assert plugin.propose(1), "prepared SGA must be able to propose"

    def test_dlga_prepare_accepts_none_evaluator(self) -> None:
        plugin = DLGAPlugin(
            DLGAConfig(pop_size=4, seed=7),
            surrogate_model=_QuadraticSurrogate(),
        )
        components = replace(
            PlatformBuilder(
                _quadratic_2d_dataset(), plugin.derivative_requirements
            ).build(),
            evaluator=None,
        )

        plugin.prepare(components)

        assert plugin.propose(1), "prepared DLGA must be able to propose"
