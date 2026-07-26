
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.search.protocol import PlatformComponents





_SAVED_EXPRESSION = "RESTORED_BEST_FROM_CHECKPOINT"

_SGA_SAVED_SCORE = -1234.5

_SGA_SAVED_DEDUP = ("u*u_x", "u_x")

_DLGA_SAVED_SCORE = -777.25

_DLGA_SAVED_LHS = "u_tt"

_DISCOVER_SAVED_REWARD = 0.85

_PYSR_SAVED_NMSE = 0.125

_PYSR_SAVED_HOF = ("u", "add(u, u_x)")

_PYSINDY_SAVED_NMSE = 0.0625







@dataclass(frozen=True)
class _RestoreScenario:

    components: PlatformComponents
    make_plugin: Callable[[], Any]
    saved_payload: dict[str, Any]
    expected: dict[str, Any]
    probe: Callable[[Any], dict[str, Any]]






_SGA_GRID_SIZE = 10
_SGA_TIME_SIZE = 5


def _sga_dataset() -> PDEDataset:
    generator = torch.Generator().manual_seed(0)
    x_vals = torch.linspace(0.0, 1.0, _SGA_GRID_SIZE)
    t_vals = torch.linspace(0.0, 1.0, _SGA_TIME_SIZE)
    u_data = torch.randn(_SGA_GRID_SIZE, _SGA_TIME_SIZE, generator=generator)
    return PDEDataset(
        name="restore-contract-sga",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x_vals),
            "t": AxisInfo(name="t", values=t_vals),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_data)},
        lhs_field="u",
        lhs_axis="t",
    )


def _sga_mock_context(dataset: PDEDataset) -> MagicMock:
    context = MagicMock()
    context.dataset = dataset
    provider = MagicMock()

    def get_derivative(field_name: str, axis: str, order: int) -> torch.Tensor:


        seed = hash((field_name, axis, order)) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(_SGA_GRID_SIZE, _SGA_TIME_SIZE, generator=gen)

    def get_variable(name: str) -> torch.Tensor:
        if dataset.fields is not None and name in dataset.fields:
            return dataset.fields[name].values
        if dataset.axes is not None and name in dataset.axes:
            return dataset.axes[name].values
        raise KeyError(f"Variable '{name}' not found")

    provider.get_derivative = get_derivative
    context.derivative_provider = provider
    context.get_variable = get_variable
    context.get_derivative = get_derivative
    return context


def _build_sga_scenario() -> _RestoreScenario:
    from kd.search.sga.config import SGAConfig
    from kd.search.sga.plugin import SGAPlugin

    config = SGAConfig(
        num=5,
        depth=3,
        width=3,
        p_var=0.6,
        p_mute=0.3,
        p_cro=0.5,
        p_rep=1.0,
        seed=42,
        maxit=3,
        str_iters=3,
        d_tol=0.5,
    )
    dataset = _sga_dataset()
    components = PlatformComponents(
        dataset=dataset,
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=_sga_mock_context(dataset),
        registry=MagicMock(),
    )

    def make_plugin() -> Any:
        return SGAPlugin(config=config)

    def probe(plugin: Any) -> dict[str, Any]:
        state = plugin.state
        return {
            "best_score": state["best_score"],
            "best_expression": state["best_expression"],
            "dedup_history": sorted(state["pde_lib"]),
        }

    donor = make_plugin()
    donor.prepare(components)
    payload = donor.state
    payload["best_score"] = _SGA_SAVED_SCORE
    payload["best_expression"] = _SAVED_EXPRESSION
    payload["pde_lib"] = list(_SGA_SAVED_DEDUP)
    donor.state = payload
    saved = donor.state
    return _RestoreScenario(
        components=components,
        make_plugin=make_plugin,
        saved_payload=saved,
        expected=probe(donor),
        probe=probe,
    )







class _ExactQuadraticModel(nn.Module):

    def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"u": 1.0 + x * x + t * t}


def _dlga_dataset() -> PDEDataset:
    x = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = 1.0 + xg * xg + tg * tg
    return PDEDataset(
        name="restore-contract-dlga",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _build_dlga_scenario() -> _RestoreScenario:
    from kd.core.platform.builder import PlatformBuilder
    from kd.search.dlga import DLGAConfig, DLGAPlugin

    config = DLGAConfig(pop_size=5, seed=13)

    def make_plugin() -> Any:
        return DLGAPlugin(config, surrogate_model=_ExactQuadraticModel())

    components = PlatformBuilder(
        _dlga_dataset(), make_plugin().derivative_requirements
    ).build()

    def probe(plugin: Any) -> dict[str, Any]:
        state = plugin.state
        return {
            "best_score": state["best_score"],
            "best_expression": state["best_expression"],
            "best_lhs_name": state["best_lhs_name"],
        }

    donor = make_plugin()
    donor.prepare(components)
    payload = donor.state
    payload["best_score"] = _DLGA_SAVED_SCORE
    payload["best_expression"] = _SAVED_EXPRESSION
    payload["best_lhs_name"] = _DLGA_SAVED_LHS
    donor.state = payload
    saved = donor.state
    return _RestoreScenario(
        components=components,
        make_plugin=make_plugin,
        saved_payload=saved,
        expected=probe(donor),
        probe=probe,
    )







class _StubExpressionEvaluator:

    def __init__(self) -> None:
        self.calls: list[str] = []

    def evaluate_expression(self, expr: str) -> Any:
        from kd.core.evaluator import EvaluationResult

        self.calls.append(expr)
        return EvaluationResult(
            mse=0.04,
            nmse=0.04,
            r2=0.96,
            complexity=2,
            is_valid=True,
            expression=expr,
            terms=[expr],
        )


def _build_discover_scenario() -> _RestoreScenario:
    from kd.search.discover.plugin import DISCOVERPlugin

    components = PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=_StubExpressionEvaluator(),
        context=MagicMock(),
        registry=MagicMock(),
        recorder=None,
    )

    def make_plugin() -> Any:
        return DISCOVERPlugin()

    def probe(plugin: Any) -> dict[str, Any]:









        engine_state = plugin.state["engine_state"]
        return {
            "best_expression": plugin.best_expression,
            "stored_best_expression": engine_state["best_expression"],
        }

    donor = make_plugin()
    donor.prepare(components)
    payload = donor.state
    payload["engine_state"]["best_reward"] = _DISCOVER_SAVED_REWARD
    payload["engine_state"]["best_expression"] = _SAVED_EXPRESSION
    donor.state = payload
    saved = donor.state
    return _RestoreScenario(
        components=components,
        make_plugin=make_plugin,
        saved_payload=saved,
        expected=probe(donor),
        probe=probe,
    )







def _build_pysr_scenario() -> _RestoreScenario:
    from kd.search.pysr import PySRPlugin



    components = PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(),
        registry=MagicMock(),
        recorder=None,
    )

    def make_plugin() -> Any:
        return PySRPlugin()

    def probe(plugin: Any) -> dict[str, Any]:
        state = plugin.state
        return {
            "best_score": state["best_score"],
            "best_expression": state["best_expression"],
            "fitted": state["fitted"],
            "hof_candidates": state["hof_candidates"],
        }

    donor = make_plugin()
    donor.prepare(components)
    payload = donor.state
    payload["best_score"] = _PYSR_SAVED_NMSE
    payload["best_expression"] = _SAVED_EXPRESSION
    payload["fitted"] = True
    payload["hof_candidates"] = list(_PYSR_SAVED_HOF)
    payload["hof_meta"] = [[1, 0.5], [3, 0.05]]
    payload["terms"] = ["u", "u_x"]
    donor.state = payload
    saved = donor.state
    return _RestoreScenario(
        components=components,
        make_plugin=make_plugin,
        saved_payload=saved,
        expected=probe(donor),
        probe=probe,
    )







def _build_pysindy_scenario() -> _RestoreScenario:
    from kd.search.pysindy.config import PySINDyConfig
    from kd.search.pysindy.plugin import PySINDyPlugin



    components = PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(),
        registry=MagicMock(),
        recorder=None,
    )

    def make_plugin() -> Any:
        return PySINDyPlugin(PySINDyConfig(terms=("u", "u_x", "u_xx"), seed=0))

    def probe(plugin: Any) -> dict[str, Any]:




        state = plugin.state
        return {
            "best_expression": state["best_expression"],
            "fitted": state["fitted"],
        }

    donor = make_plugin()
    donor.prepare(components)
    payload = donor.state
    payload["best_expression"] = _SAVED_EXPRESSION
    payload["best_score"] = _PYSINDY_SAVED_NMSE
    payload["fitted"] = True
    payload["terms"] = ["u", "u_x", "u_xx"]
    payload["support"] = [0, 2]
    payload["coefficient_values"] = [1.0, 1.0]
    donor.state = payload
    saved = donor.state
    return _RestoreScenario(
        components=components,
        make_plugin=make_plugin,
        saved_payload=saved,
        expected=probe(donor),
        probe=probe,
    )















def _build_eqgpt_scenario() -> _RestoreScenario:
    from tests.unit.search import _resume_conformance_helpers as rc

    components = rc.eqgpt_components()

    def make_plugin() -> Any:
        return rc.make_eqgpt_plugin()

    def probe(plugin: Any) -> dict[str, Any]:
        pending = getattr(plugin, "_pending_state", None)
        if pending:
            rewards = list(pending.get("top_k", {}).get("rewards", []))
        else:
            try:
                rewards = list(plugin.state["top_k"]["rewards"])
            except (RuntimeError, KeyError, TypeError):
                rewards = []
        return {"pool_rewards": rewards}

    donor = make_plugin()
    donor.prepare(components)
    rc.run_eqgpt_epochs(donor, 2)
    saved = donor.state
    return _RestoreScenario(
        components=components,
        make_plugin=make_plugin,
        saved_payload=saved,
        expected=probe(donor),
        probe=probe,
    )


def _build_llm4ed_scenario() -> _RestoreScenario:
    from kd.search.llm4ed.plugin import Llm4edPlugin
    from tests.unit.search.llm4ed._plugin_helpers import (
        GOOD,
        FakeProvider,
        components_for,
        make_config,
        run,
    )

    components = components_for()

    def make_plugin() -> Any:
        return Llm4edPlugin(make_config(), provider=FakeProvider(GOOD))

    def probe(plugin: Any) -> dict[str, Any]:
        pending = getattr(plugin, "_pending_state", None)
        if pending:
            best = pending.get("best", {})
            return {
                "best_reward": float(best.get("reward", 0.0)),
                "best_expression": str(best.get("expression", "")),
            }
        return {
            "best_reward": float(plugin.best_score),
            "best_expression": str(plugin.best_expression),
        }

    donor = make_plugin()
    donor.prepare(components)
    run(donor, rounds=2)
    saved = donor.state
    return _RestoreScenario(
        components=components,
        make_plugin=make_plugin,
        saved_payload=saved,
        expected=probe(donor),
        probe=probe,
    )






_SCENARIO_BUILDERS: dict[str, Callable[[], _RestoreScenario]] = {
    "sga": _build_sga_scenario,
    "dlga": _build_dlga_scenario,
    "discover": _build_discover_scenario,
    "pysr": _build_pysr_scenario,
    "pysindy": _build_pysindy_scenario,
    "eqgpt": _build_eqgpt_scenario,
    "llm4ed": _build_llm4ed_scenario,
}


@pytest.fixture(params=sorted(_SCENARIO_BUILDERS))
def scenario(request: pytest.FixtureRequest) -> _RestoreScenario:
    builder = _SCENARIO_BUILDERS[request.param]
    return builder()


def _assert_payload_is_discriminative(
    scenario: _RestoreScenario,
    fresh_defaults: dict[str, Any],
) -> None:
    assert set(fresh_defaults) == set(scenario.expected), (
        "probe/expected field sets diverged -- scenario builder bug"
    )
    for field, saved_value in scenario.expected.items():
        assert saved_value != fresh_defaults[field], (
            f"probe field {field!r} is not discriminative: the donor-injected "
            f"value {saved_value!r} equals the fresh-prepare default, so the "
            "preservation assertion below would pass even if the plugin "
            "silently dropped the restore"
        )







class TestRestorePrepareContract:

    @pytest.mark.unit
    def test_saved_payload_differs_from_fresh_prepare_defaults(
        self, scenario: _RestoreScenario
    ) -> None:
        fresh = scenario.make_plugin()
        fresh.prepare(scenario.components)
        fresh_defaults = scenario.probe(fresh)

        _assert_payload_is_discriminative(scenario, fresh_defaults)

    @pytest.mark.unit
    def test_restore_before_first_prepare_is_preserved(
        self, scenario: _RestoreScenario
    ) -> None:


        fresh = scenario.make_plugin()
        fresh.prepare(scenario.components)
        _assert_payload_is_discriminative(scenario, scenario.probe(fresh))

        subject = scenario.make_plugin()


        subject.state = scenario.saved_payload

        subject.prepare(scenario.components)

        after = scenario.probe(subject)
        for field, expected_value in scenario.expected.items():
            assert after[field] == expected_value, (
                f"{field!r} was not preserved across restore->prepare: "
                f"expected {expected_value!r}, got {after[field]!r}"
            )

    @pytest.mark.unit
    def test_empty_state_resets_to_fresh_defaults(
        self, scenario: _RestoreScenario
    ) -> None:
        control = scenario.make_plugin()
        control.prepare(scenario.components)
        control.state = {}
        control_state = scenario.probe(control)



        for field, saved_value in scenario.expected.items():
            assert saved_value != control_state[field], (
                f"probe field {field!r} cannot discriminate restored-vs-reset"
            )

        subject = scenario.make_plugin()
        subject.prepare(scenario.components)
        subject.state = scenario.saved_payload


        assert scenario.probe(subject) == scenario.expected

        subject.state = {}

        assert scenario.probe(subject) == control_state, (
            "empty-payload reset left stale per-run state exposed "
            "(protocol: empty value == full reset at assignment time)"
        )

    @pytest.mark.unit
    def test_second_prepare_without_new_restore_starts_fresh(
        self, scenario: _RestoreScenario
    ) -> None:
        fresh = scenario.make_plugin()
        fresh.prepare(scenario.components)
        fresh_defaults = scenario.probe(fresh)

        subject = scenario.make_plugin()
        subject.state = scenario.saved_payload
        subject.prepare(scenario.components)


        assert scenario.probe(subject) == scenario.expected

        subject.prepare(scenario.components)

        assert scenario.probe(subject) == fresh_defaults, (
            "second prepare() without a new restore must reset to fresh "
            "defaults (one-shot restore flag; reuse-reset semantics)"
        )
