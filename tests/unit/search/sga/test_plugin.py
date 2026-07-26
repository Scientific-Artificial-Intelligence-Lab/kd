
from __future__ import annotations

import json
import math
import pickle
from unittest.mock import MagicMock

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.search.protocol import PlatformComponents, SearchAlgorithm
from kd.search.sga.config import SGAConfig





_SMALL_GRID_SIZE = 10

_SMALL_TIME_SIZE = 5







def _make_synthetic_dataset() -> PDEDataset:
    x_vals = torch.linspace(0.0, 1.0, _SMALL_GRID_SIZE)
    t_vals = torch.linspace(0.0, 1.0, _SMALL_TIME_SIZE)
    data_gen = torch.Generator().manual_seed(20260611)
    u_data = torch.randn(_SMALL_GRID_SIZE, _SMALL_TIME_SIZE, generator=data_gen)

    return PDEDataset(
        name="test_synthetic",
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


def _make_mock_derivative_provider() -> MagicMock:
    provider = MagicMock()
    n_total = _SMALL_GRID_SIZE * _SMALL_TIME_SIZE

    def get_derivative(field_name: str, axis: str, order: int) -> torch.Tensor:



        key = f"{field_name}|{axis}|{order}".encode()
        seed = int.from_bytes(key, "little") % (2**31)
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(_SMALL_GRID_SIZE, _SMALL_TIME_SIZE, generator=gen)

    provider.get_derivative = get_derivative
    return provider


def _make_mock_context(
    dataset: PDEDataset,
) -> MagicMock:
    context = MagicMock()
    context.dataset = dataset
    provider = _make_mock_derivative_provider()
    context.derivative_provider = provider

    def get_variable(name: str) -> torch.Tensor:
        if dataset.fields is not None and name in dataset.fields:
            return dataset.fields[name].values
        if dataset.axes is not None and name in dataset.axes:
            return dataset.axes[name].values
        raise KeyError(f"Variable '{name}' not found")

    def get_derivative(field_name: str, axis: str, order: int) -> torch.Tensor:
        return provider.get_derivative(field_name, axis, order)

    context.get_variable = get_variable
    context.get_derivative = get_derivative
    return context


@pytest.fixture
def sga_config() -> SGAConfig:
    return SGAConfig(
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


@pytest.fixture
def mock_components() -> PlatformComponents:
    dataset = _make_synthetic_dataset()
    context = _make_mock_context(dataset)
    return PlatformComponents(
        dataset=dataset,
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=context,
        registry=MagicMock(),
    )







class TestSGAPluginProtocol:

    @pytest.mark.smoke
    def test_importable(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        assert SGAPlugin is not None

    @pytest.mark.smoke
    def test_instantiate_default_config(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert plugin is not None

    @pytest.mark.smoke
    def test_instantiate_custom_config(self, sga_config: SGAConfig) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        assert plugin is not None

    @pytest.mark.unit
    def test_isinstance_search_algorithm(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert isinstance(plugin, SearchAlgorithm)

    @pytest.mark.unit
    def test_has_all_protocol_methods(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert callable(getattr(plugin, "prepare", None))
        assert callable(getattr(plugin, "propose", None))
        assert callable(getattr(plugin, "evaluate", None))
        assert callable(getattr(plugin, "update", None))

        assert hasattr(plugin, "best_score")
        assert hasattr(plugin, "best_expression")
        assert hasattr(plugin, "state")







class TestSGAPluginPrepare:

    @pytest.mark.unit
    def test_delta_map_accepts_float32_linspace_grid(
        self,
        sga_config: SGAConfig,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        x_vals = torch.linspace(0.0, 1.0, 1000, dtype=torch.float32)
        t_vals = torch.linspace(0.0, 1.0, 20, dtype=torch.float32)
        x_grid, t_grid = torch.meshgrid(x_vals, t_vals, indexing="ij")
        u_vals = torch.sin(x_grid) * torch.exp(-t_grid)
        dataset = PDEDataset(
            name="float32_grid",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals),
                "t": AxisInfo(name="t", values=t_vals),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u_vals)},
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=sga_config)
        delta = plugin._build_delta_map(dataset, ["x", "t"])

        assert delta["x"] == pytest.approx(float(x_vals[1] - x_vals[0]))
        assert delta["t"] == pytest.approx(float(t_vals[1] - t_vals[0]))

    @pytest.mark.unit
    def test_prepare_initializes_population(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)


        state = plugin.state
        assert "population" in state
        assert len(state["population"]) == sga_config.num

    @pytest.mark.unit
    def test_prepare_builds_vars(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        state = plugin.state

        assert "vars" in state
        vars_list = state["vars"]
        assert isinstance(vars_list, list)
        assert len(vars_list) > 0

        assert "u" in vars_list

    @pytest.mark.unit
    def test_prepare_excludes_lhs_axis_from_vars(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        vars_list = plugin.state["vars"]

        assert "t" not in vars_list

        assert "x" in vars_list

    @pytest.mark.unit
    def test_prepare_excludes_lhs_axis_derivatives_from_vars(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        vars_list = plugin.state["vars"]

        assert "u_t" not in vars_list
        assert "u_tt" not in vars_list

        assert "u_x" in vars_list


        assert "u_xx" not in vars_list

    @pytest.mark.unit
    def test_prepare_skips_population_init_if_already_populated(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)


        plugin.prepare(mock_components)
        state_before = plugin.state
        pop_before = state_before["population"]


        plugin.state = state_before


        plugin.prepare(mock_components)
        state_after = plugin.state
        pop_after = state_after["population"]


        assert len(pop_after) == len(pop_before)

    @pytest.mark.unit
    def test_prepare_twice_without_checkpoint_reinitializes(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        n_expected = len(plugin.state["population"])
        assert n_expected > 0



        plugin._population = []
        plugin._scores = []


        plugin.prepare(mock_components)
        assert len(plugin.state["population"]) == n_expected


class TestSGAReusePrepareResetsState:

    @pytest.mark.unit
    def test_fresh_reprepare_clears_stale_best(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)




        plugin._best_score = -1.0e9
        plugin._best_expression = "STALE_FROM_PRIOR_DATASET"
        plugin._best_formatted_cache = "STALE_FORMATTED"


        plugin.prepare(mock_components)

        assert plugin.state["best_expression"] != "STALE_FROM_PRIOR_DATASET"
        assert plugin.state["best_score"] != -1.0e9
        assert plugin._best_formatted_cache != "STALE_FORMATTED"

    @pytest.mark.unit
    def test_checkpoint_restore_preserves_best(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        saved = plugin.state
        saved["best_score"] = -42.0
        saved["best_expression"] = "RESTORED_EXPR"
        n_pop = len(saved["population"])

        restored = SGAPlugin(config=sga_config)
        restored.state = saved
        restored.prepare(mock_components)

        assert restored.state["best_score"] == -42.0
        assert restored.state["best_expression"] == "RESTORED_EXPR"


        assert len(restored.state["population"]) == n_pop

    @pytest.mark.unit
    def test_checkpoint_restore_preserves_dedup_history(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        saved = plugin.state
        saved["pde_lib"] = ["u_x", "u*u_x"]

        restored = SGAPlugin(config=sga_config)
        restored.state = saved
        restored.prepare(mock_components)

        assert set(restored.state["pde_lib"]) == {"u_x", "u*u_x"}







class TestSGAPluginPropose:

    @pytest.mark.unit
    def test_propose_returns_list_of_strings(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        assert isinstance(candidates, list)
        assert all(isinstance(c, str) for c in candidates)

    @pytest.mark.unit
    def test_propose_returns_full_frontier(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        n = sga_config.num
        candidates = plugin.propose(n)
        assert len(candidates) >= 1

    @pytest.mark.unit
    def test_propose_returns_kd_format(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        for expr in candidates:


            tokens = expr.split()
            assert len(tokens) >= 1, f"Expression is empty: {expr!r}"

    @pytest.mark.unit
    def test_propose_applies_genetic_operators(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        c1 = plugin.propose(sga_config.num)

        dummy_results = [
            EvaluationResult(mse=float(i), nmse=float(i), r2=0.0)
            for i in range(len(c1))
        ]
        plugin.update(dummy_results)
        c2 = plugin.propose(sga_config.num)


        assert c1 != c2, (
            "Genetic operators did not change candidates between generations"
        )







class TestSGAPluginEvaluate:

    @pytest.mark.unit
    def test_evaluate_returns_list_of_evaluation_result(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)

        assert isinstance(results, list)
        assert all(isinstance(r, EvaluationResult) for r in results)

    @pytest.mark.unit
    def test_evaluate_result_count_matches_candidates(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)

        assert len(results) == len(candidates)

    @pytest.mark.unit
    def test_evaluate_results_have_finite_or_penalty_scores(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)

        for r in results:

            assert isinstance(r.mse, float)

            if r.is_valid:
                assert math.isfinite(r.mse)
                assert r.mse >= 0.0

    @pytest.mark.unit
    def test_evaluate_results_have_aic_scores(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)

        for r in results:
            assert r.score is not None
            assert isinstance(r.score, float)







class TestSGAPluginUpdate:

    @pytest.mark.unit
    def test_update_truncates_population_to_num(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        state = plugin.state
        assert len(state["population"]) == sga_config.num

    @pytest.mark.unit
    def test_update_tracks_best_score(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)


        initial_score = plugin.best_score
        assert isinstance(initial_score, float)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)


        updated_score = plugin.best_score
        assert updated_score <= initial_score

    @pytest.mark.unit
    def test_update_monotonically_improves_best_score(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        prev_score = plugin.best_score
        for _ in range(3):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)
            current_score = plugin.best_score
            assert current_score <= prev_score, (
                f"best_score increased from {prev_score} to {current_score}"
            )
            prev_score = current_score







class TestSGAPluginLifecycle:

    @pytest.mark.unit
    def test_full_cycle_runs_without_error(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)


        assert isinstance(plugin.best_score, float)
        assert isinstance(plugin.best_expression, str)

    @pytest.mark.unit
    def test_multi_iteration_no_crash(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        for _ in range(5):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)

        assert isinstance(plugin.best_score, float)
        assert isinstance(plugin.best_expression, str)

    @pytest.mark.unit
    def test_best_expression_is_nonempty_after_valid_cycle(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        for _ in range(3):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)




        if plugin.best_score >= float("inf"):
            pytest.fail(
                "Premise not met: no valid evaluation in 3 seeded "
                f"iterations (best_score={plugin.best_score}); "
                "best_expression contract cannot be checked."
            )
        assert len(plugin.best_expression) > 0







class TestSGAPluginCheckpoint:

    @pytest.mark.unit
    def test_state_getter_returns_dict(self, sga_config: SGAConfig) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        state = plugin.state
        assert isinstance(state, dict)

    @pytest.mark.unit
    def test_state_is_pickle_serializable(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        state = plugin.state
        pickled = pickle.dumps(state)
        restored = pickle.loads(pickled)
        assert restored == state

    @pytest.mark.unit
    def test_state_roundtrip_preserves_best(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)


        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        score_before = plugin.best_score
        expr_before = plugin.best_expression
        state = plugin.state


        plugin2 = SGAPlugin(config=sga_config)
        plugin2.state = state
        plugin2.prepare(mock_components)

        assert plugin2.best_score == score_before
        assert plugin2.best_expression == expr_before

    @pytest.mark.unit
    def test_state_roundtrip_population_preserved(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        state = plugin.state
        pop_before = state["population"]


        plugin2 = SGAPlugin(config=sga_config)
        plugin2.state = state
        plugin2.prepare(mock_components)

        state2 = plugin2.state
        pop_after = state2["population"]

        assert len(pop_after) == len(pop_before)

    @pytest.mark.unit
    def test_state_contains_expected_keys(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        state = plugin.state
        assert "population" in state
        assert "best_score" in state
        assert "best_expression" in state

    @pytest.mark.unit
    def test_checkpoint_restore_preserves_rng_state(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)


        for _ in range(3):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)

        state = plugin.state


        plugin2 = SGAPlugin(config=sga_config)
        plugin2.state = state
        plugin2.prepare(mock_components)


        plugin3 = SGAPlugin(config=sga_config)
        plugin3.prepare(mock_components)



        c2 = plugin2.propose(sga_config.num)
        c3 = plugin3.propose(sga_config.num)
        assert c2 != c3, (
            "Restored plugin produces same candidates as fresh — "
            "RNG state was reset instead of preserved"
        )







class TestSGAPluginBestProperties:

    @pytest.mark.unit
    def test_best_score_initial_is_inf(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert plugin.best_score == float("inf")

    @pytest.mark.unit
    def test_best_expression_initial_is_empty(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert plugin.best_expression == ""

    @pytest.mark.unit
    def test_best_score_returns_float(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        assert isinstance(plugin.best_score, float)

        plugin.prepare(mock_components)
        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)
        assert isinstance(plugin.best_score, float)

    @pytest.mark.unit
    def test_best_expression_returns_str(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        assert isinstance(plugin.best_expression, str)

        plugin.prepare(mock_components)
        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)
        assert isinstance(plugin.best_expression, str)



    @pytest.mark.unit
    def test_best_expression_falls_back_when_genotype_empty(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        plugin._best_expression = ""
        plugin._population = None
        plugin._best_formatted_cache = None

        formatted = plugin.best_expression
        assert formatted != "", "best_expression must not be empty"

        assert plugin._default_term_name is not None
        assert formatted == plugin._default_term_name

    @pytest.mark.unit
    def test_format_rhs_with_coefficients_skips_zero(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        coefs = torch.tensor([0.0, 1.5, -2.25, 1e-12])
        terms = ["u", "u_x", "u_xx", "u_xxx"]
        rendered = SGAPlugin._format_rhs_with_coefficients(coefs, terms)
        assert "u_x" in rendered
        assert "u_xx" in rendered

        assert "u_xxx" not in rendered

        assert rendered.startswith("1.5*u_x")

        assert "- 2.25*u_xx" in rendered

    @pytest.mark.unit
    def test_format_rhs_returns_empty_when_all_zero(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        coefs = torch.tensor([0.0, 0.0])
        terms = ["u", "u_x"]
        assert SGAPlugin._format_rhs_with_coefficients(coefs, terms) == ""

    @pytest.mark.unit
    def test_best_expression_cache_invalidated_on_state_restore(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        plugin._best_formatted_cache = "STALE"

        plugin.state = {"best_expression": "new_value"}

        assert plugin._best_formatted_cache is None

        out = plugin.best_expression
        assert out != "STALE"







class TestSGAPluginNegative:

    @pytest.mark.unit
    def test_propose_before_prepare_raises(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        with pytest.raises((RuntimeError, ValueError)):
            plugin.propose(5)

    @pytest.mark.unit
    def test_evaluate_before_prepare_raises(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        with pytest.raises((RuntimeError, ValueError)):
            plugin.evaluate(["some_expr"])

    @pytest.mark.unit
    def test_update_with_empty_results(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        plugin.update([])

    @pytest.mark.unit
    def test_evaluate_empty_candidates(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        results = plugin.evaluate([])
        assert results == []

    @pytest.mark.unit
    def test_state_setter_with_empty_dict(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()

        plugin.state = {}

    @pytest.mark.unit
    def test_update_does_not_increase_population_beyond_num(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        for _ in range(5):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)

            pop_size = len(plugin.state["population"])
            assert pop_size <= sga_config.num, (
                f"Population {pop_size} exceeds config.num={sga_config.num}"
            )

    @pytest.mark.unit
    def test_propose_with_invalid_n_raises(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        with pytest.raises(ValueError):
            plugin.propose(0)

        with pytest.raises(ValueError):
            plugin.propose(-1)

    @pytest.mark.unit
    def test_config_none_uses_default(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=None)

        state = plugin.state
        assert isinstance(state, dict)

    @pytest.mark.numerical
    def test_evaluate_handles_nan_in_data(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin


        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)

        results = plugin.evaluate(candidates)
        assert len(results) == len(candidates)

        for r in results:
            assert isinstance(r.mse, float)

    @pytest.mark.numerical
    def test_extreme_data_init_population_raises(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=2, width=2, seed=99, maxit=1)


        x_vals = torch.linspace(0.0, 1.0, 5)
        t_vals = torch.linspace(0.0, 1.0, 3)
        u_data = torch.full((5, 3), 1e15)

        dataset = PDEDataset(
            name="extreme",
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
        context = _make_mock_context(dataset)
        components = PlatformComponents(
            dataset=dataset,
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=context,
            registry=MagicMock(),
        )

        plugin = SGAPlugin(config=config)
        with pytest.raises(RuntimeError, match="resample"):
            plugin.prepare(components)

    @pytest.mark.unit
    def test_update_with_all_invalid_results(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        invalid_results = [
            EvaluationResult(
                mse=float("inf"),
                nmse=float("inf"),
                r2=-float("inf"),
                score=float("inf"),
                is_valid=False,
                error_message="test failure",
            )
            for _ in range(sga_config.num)
        ]

        plugin.update(invalid_results)


        assert len(plugin.state["population"]) > 0

    @pytest.mark.unit
    def test_seed_reproducibility(self, mock_components: PlatformComponents) -> None:
        from kd.search.sga.plugin import SGAPlugin

        config1 = SGAConfig(num=5, depth=3, width=3, seed=12345)
        config2 = SGAConfig(num=5, depth=3, width=3, seed=12345)

        plugin1 = SGAPlugin(config=config1)
        plugin1.prepare(mock_components)
        c1 = plugin1.propose(config1.num)

        plugin2 = SGAPlugin(config=config2)
        plugin2.prepare(mock_components)
        c2 = plugin2.propose(config2.num)

        assert c1 == c2, "Same seed should produce identical candidates"







class TestOffspringLifecycleAndCrossover:

    @pytest.mark.unit
    def test_crossover_is_evaluated_before_mutation_stage(self) -> None:
        from unittest.mock import patch

        from kd.search.sga.pde import PDE
        from kd.search.sga.plugin import SGAPlugin
        from kd.search.sga.train import CandidateResult, TrainResult
        from kd.search.sga.tree import Node, Tree

        def pde(name: str) -> PDE:
            return PDE([Tree(Node(name, 0))])

        elite = pde("elite")
        weak = pde("weak")
        xo_good = pde("xo_good")
        xo_bad = pde("xo_bad")
        events: list[tuple[str, str]] = []

        config = SGAConfig(
            num=2,
            depth=1,
            width=1,
            p_cro=0.5,
            p_mute=0.5,
            p_rep=0.0,
            seed=7,
        )
        plugin = SGAPlugin(config=config)
        plugin._prepared = True
        plugin._population = [elite, weak]
        plugin._scores = [0.0, 10.0]
        plugin._vars = ["elite", "weak", "xo_good", "xo_bad"]
        plugin._den = (("x", 0),)

        def fake_crossover(*args: object, **kwargs: object) -> tuple[PDE, PDE]:
            return xo_good.copy(), xo_bad.copy()

        def fake_mutate(pde_arg: PDE, *args: object, **kwargs: object) -> PDE:
            events.append(("mutate", str(pde_arg)))
            return pde_arg.copy()

        def fake_evaluate_candidate(
            pde_arg: PDE, *args: object, **kwargs: object
        ) -> CandidateResult:
            events.append(("eval", str(pde_arg)))
            aic = 0.5 if str(pde_arg) == str(xo_good) else 99.0
            train = TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=aic,
                mse=1.0,
                best_tol=0.0,
            )
            return CandidateResult(
                train_result=train,
                pruned_pde=pde_arg.copy(),
                valid_term_indices=[0],
            )

        with (
            patch("kd.search.sga.genetic.crossover", side_effect=fake_crossover),
            patch("kd.search.sga.genetic.mutate", side_effect=fake_mutate),
            patch(
                "kd.search.sga.plugin.evaluate_candidate",
                side_effect=fake_evaluate_candidate,
            ),
        ):
            plugin.propose(config.num)

        first_xo_eval = events.index(("eval", str(xo_good)))
        first_mutate = next(i for i, event in enumerate(events) if event[0] == "mutate")
        assert first_xo_eval < first_mutate, (
            f"Expected crossover evaluation before mutation, got events={events}"
        )
        assert ("mutate", str(xo_good)) in events, (
            "Expected mutation stage to operate on selected crossover child, "
            f"got events={events}"
        )
        assert ("mutate", str(xo_bad)) not in events, (
            "Weak crossover child must be truncated before mutation, "
            f"got events={events}"
        )

    @pytest.mark.unit
    def test_state_restore_clears_pending_generation(self) -> None:
        from kd.search.sga.pde import PDE
        from kd.search.sga.plugin import SGAPlugin
        from kd.search.sga.tree import Node, Tree

        def pde(name: str) -> PDE:
            return PDE([Tree(Node(name, 0))])

        plugin = SGAPlugin(config=SGAConfig(num=1, seed=7))
        stale = pde("stale_pending")
        restored = pde("restored")
        plugin._pending_population = [stale]
        plugin._pending_scores = [-999.0]
        plugin._offspring = [stale]
        plugin._offspring_results = [
            EvaluationResult(mse=0.0, nmse=0.0, r2=1.0, score=-999.0)
        ]

        plugin.state = {
            "population": [restored],
            "scores": [5.0],
            "best_score": 5.0,
            "best_expression": str(restored),
        }
        plugin.update([])

        state = plugin.state
        assert [str(p) for p in state["population"]] == [str(restored)]
        assert state["scores"] == [5.0]

    @pytest.mark.unit
    def test_prepare_clears_pending_generation(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.pde import PDE
        from kd.search.sga.plugin import SGAPlugin
        from kd.search.sga.tree import Node, Tree

        stale = PDE([Tree(Node("stale_pending", 0))])
        plugin = SGAPlugin(config=SGAConfig(num=2, seed=42))
        plugin._pending_population = [stale]
        plugin._pending_scores = [-999.0]
        plugin._offspring = [stale]
        plugin._offspring_results = [
            EvaluationResult(mse=0.0, nmse=0.0, r2=1.0, score=-999.0)
        ]

        plugin.prepare(mock_components)
        prepared_population = [str(p) for p in plugin.state["population"]]
        plugin.update([])

        assert [str(p) for p in plugin.state["population"]] == prepared_population
        assert plugin._pending_population is None
        assert plugin._pending_scores is None
        assert plugin._offspring is None
        assert plugin._offspring_results is None

    @pytest.mark.unit
    def test_invalid_evaluation_keeps_pruned_genotype(self) -> None:
        from unittest.mock import patch

        from kd.search.sga.pde import PDE
        from kd.search.sga.plugin import SGAPlugin
        from kd.search.sga.train import CandidateResult, TrainResult
        from kd.search.sga.tree import Node, Tree

        original = PDE([Tree(Node("original", 0))])
        pruned = PDE([Tree(Node("pruned", 0))])
        plugin = SGAPlugin(config=SGAConfig(num=1, seed=7))
        plugin._prepared = True

        train = TrainResult(
            coefficients=torch.tensor([1.0]),
            selected_indices=[0],
            aic_score=float("inf"),
            mse=1.0,
            best_tol=0.0,
        )

        def fake_evaluate_candidate(*args: object, **kwargs: object) -> CandidateResult:
            return CandidateResult(
                train_result=train,
                pruned_pde=pruned,
                valid_term_indices=[0],
            )

        with patch(
            "kd.search.sga.plugin.evaluate_candidate",
            side_effect=fake_evaluate_candidate,
        ):
            scored = plugin._score_offspring(original)

        assert scored.pde == pruned
        assert scored.score == float("inf")

    @pytest.mark.unit
    def test_p_cro_zero_no_crossover(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        from unittest.mock import patch

        from kd.search.sga.plugin import SGAPlugin

        config = SGAConfig(
            num=5,
            depth=3,
            width=3,
            p_var=0.6,
            p_mute=0.3,
            p_cro=0.0,
            p_rep=0.3,
            seed=42,
            maxit=3,
            str_iters=3,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        plugin.prepare(mock_components)


        import kd.search.sga.genetic as gen_mod

        original_crossover = gen_mod.crossover
        crossover_calls: list[tuple] = []

        def tracking_crossover(pde1, pde2, rng):
            crossover_calls.append((pde1, pde2))
            return original_crossover(pde1, pde2, rng)

        with patch.object(gen_mod, "crossover", side_effect=tracking_crossover):
            plugin.propose(config.num)

        assert len(crossover_calls) == 0, (
            f"p_cro=0 should produce zero crossover calls, got {len(crossover_calls)}"
        )

    @pytest.mark.unit
    def test_propose_does_not_truncate_before_evaluate(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)





        offspring_via_ops = plugin._apply_genetic_ops()
        total_offspring = len(offspring_via_ops)


        plugin2 = SGAPlugin(config=sga_config)
        plugin2.prepare(mock_components)


        small_n = 3
        candidates = plugin2.propose(small_n)


        assert len(candidates) == total_offspring, (
            f"propose({small_n}) returned {len(candidates)} candidates but "
            f"genetic ops produced {total_offspring}. "
            f"Offspring must not be truncated before evaluation."
        )

    @pytest.mark.unit
    def test_evaluate_receives_all_offspring(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)


        candidates = plugin.propose(n=1000)
        results = plugin.evaluate(candidates)


        assert len(results) == len(candidates), (
            f"evaluate returned {len(results)} results for {len(candidates)} candidates"
        )

        for i, r in enumerate(results):
            assert r.error_message != "No corresponding PDE for evaluation", (
                f"Result {i} fell through to fallback — offspring/candidate mismatch"
            )

    @pytest.mark.unit
    def test_offspring_candidate_mapping_consistency(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)


        offspring = plugin._offspring
        assert offspring is not None, "_offspring should be set after propose()"
        assert len(offspring) == len(candidates), (
            f"_offspring has {len(offspring)} PDEs but propose returned "
            f"{len(candidates)} candidates — mapping broken"
        )

    @pytest.mark.unit
    def test_p_cro_zero_only_mutation_offspring(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        config = SGAConfig(
            num=5,
            depth=3,
            width=3,
            p_var=0.6,
            p_mute=0.3,
            p_cro=0.0,
            p_rep=0.3,
            seed=42,
            maxit=3,
            str_iters=3,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(config.num)



        expected = config.num - 1
        assert len(candidates) == expected, (
            f"With p_cro=0, expected {expected} offspring (mutation only), "
            f"got {len(candidates)}"
        )

    @pytest.mark.unit
    def test_update_truncates_after_evaluation_not_before(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        pop_before = len(plugin.state["population"])
        assert pop_before == sga_config.num

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)


        assert len(plugin.state["population"]) == sga_config.num

        plugin.update(results)


        assert len(plugin.state["population"]) == sga_config.num







class TestInitPopulationResample:

    @pytest.mark.unit
    def test_pathological_individuals_rejected(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        from unittest.mock import patch

        from kd.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)

        call_count = 0
        finite_aic = 100.0

        def mock_safe_evaluate_aic(
            *args: object, **kwargs: object
        ) -> tuple[float, object]:
            nonlocal call_count
            call_count += 1
            pde = args[0] if args else None

            if call_count <= 3:
                return float("inf"), pde
            return finite_aic, pde

        with patch(
            "kd.search.sga.plugin._safe_evaluate_aic",
            side_effect=mock_safe_evaluate_aic,
        ):
            plugin = SGAPlugin(config=config)
            plugin.prepare(mock_components)


        scores = plugin.state["scores"]
        assert scores is not None
        for s in scores:
            assert math.isfinite(s), f"Population contains pathological AIC: {s}"

    @pytest.mark.unit
    def test_resample_has_finite_retry_limit(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        from unittest.mock import patch

        from kd.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)

        def always_inf(*args: object, **kwargs: object) -> tuple[float, object]:
            pde = args[0] if args else None
            return float("inf"), pde

        with patch(
            "kd.search.sga.plugin._safe_evaluate_aic",
            side_effect=always_inf,
        ):
            plugin = SGAPlugin(config=config)
            with pytest.raises(
                RuntimeError, match="[Rr]esample|[Rr]etry|[Pp]athological|[Ii]nit"
            ):
                plugin.prepare(mock_components)

    @pytest.mark.unit
    def test_resample_exhaustion_raises_clear_error(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        from unittest.mock import patch

        from kd.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)

        def always_inf(*args: object, **kwargs: object) -> tuple[float, object]:
            pde = args[0] if args else None
            return float("inf"), pde

        with patch(
            "kd.search.sga.plugin._safe_evaluate_aic",
            side_effect=always_inf,
        ):
            plugin = SGAPlugin(config=config)
            with pytest.raises(RuntimeError) as exc_info:
                plugin.prepare(mock_components)

            msg = str(exc_info.value).lower()

            assert (
                "inf" in msg
                or "pathological" in msg
                or "retry" in msg
                or "resample" in msg
            ), f"Error message not informative enough: {exc_info.value}"

    @pytest.mark.unit
    def test_healthy_individuals_not_resampled(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        from unittest.mock import patch

        from kd.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)
        call_count = 0

        def always_finite(*args: object, **kwargs: object) -> tuple[float, object]:
            nonlocal call_count
            call_count += 1
            pde = args[0] if args else None
            return 50.0 + call_count, pde

        with patch(
            "kd.search.sga.plugin._safe_evaluate_aic",
            side_effect=always_finite,
        ):
            plugin = SGAPlugin(config=config)
            plugin.prepare(mock_components)


        assert call_count == config.num, (
            f"Expected exactly {config.num} evaluations (no retries), got {call_count}"
        )

    @pytest.mark.unit
    def test_resample_count_bounded_per_individual(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        from unittest.mock import patch

        from kd.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)
        call_count = 0

        def always_inf(*args: object, **kwargs: object) -> tuple[float, object]:
            nonlocal call_count
            call_count += 1
            pde = args[0] if args else None
            return float("inf"), pde

        with patch(
            "kd.search.sga.plugin._safe_evaluate_aic",
            side_effect=always_inf,
        ):
            plugin = SGAPlugin(config=config)
            with pytest.raises(RuntimeError):
                plugin.prepare(mock_components)




        max_reasonable = config.num * 101
        assert call_count <= max_reasonable, (
            f"Too many evaluations: {call_count} > {max_reasonable}. "
            f"Resample loop may not be bounded."
        )

    @pytest.mark.unit
    def test_moderate_negative_aic_accepted(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        from unittest.mock import patch

        from kd.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)

        def negative_aic(*args: object, **kwargs: object) -> tuple[float, object]:
            pde = args[0] if args else None
            return -50.0, pde

        with patch(
            "kd.search.sga.plugin._safe_evaluate_aic",
            side_effect=negative_aic,
        ):
            plugin = SGAPlugin(config=config)
            plugin.prepare(mock_components)


        scores = plugin.state["scores"]
        assert all(s == -50.0 for s in scores)

    @pytest.mark.unit
    def test_mixed_finite_and_inf_partial_resample(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        from unittest.mock import patch

        from kd.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)
        call_count = 0

        def mixed_aic(*args: object, **kwargs: object) -> tuple[float, object]:
            nonlocal call_count
            call_count += 1
            pde = args[0] if args else None




            if call_count == 1:
                return float("inf"), pde
            return 42.0, pde

        with patch(
            "kd.search.sga.plugin._safe_evaluate_aic",
            side_effect=mixed_aic,
        ):
            plugin = SGAPlugin(config=config)
            plugin.prepare(mock_components)


        assert call_count == 4, f"Expected 4 evaluations, got {call_count}"

        scores = plugin.state["scores"]
        for s in scores:
            assert math.isfinite(s)







def _make_components_with_fields_axes(
    fields: dict[str, torch.Tensor],
    axes: dict[str, torch.Tensor],
    axis_order: list[str],
    lhs_field: str,
    lhs_axis: str,
    *,
    lhs_deriv_available: bool = True,
) -> PlatformComponents:
    field_shape = next(iter(fields.values())).shape

    field_data = {
        name: FieldData(name=name, values=val) for name, val in fields.items()
    }
    axis_info = {name: AxisInfo(name=name, values=val) for name, val in axes.items()}

    dataset = PDEDataset(
        name="guardrail_test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes=axis_info,
        axis_order=axis_order,
        fields=field_data,
        lhs_field=lhs_field,
        lhs_axis=lhs_axis,
    )

    context = MagicMock()
    n_total = 1
    for dim in field_shape:
        n_total *= dim

    def get_variable(name: str) -> torch.Tensor:
        if name in fields:
            return fields[name]
        if name in axes:
            return axes[name]
        raise KeyError(f"Variable '{name}' not found")

    def get_derivative(field_name: str, axis: str, order: int) -> torch.Tensor:
        if not lhs_deriv_available and axis == lhs_axis and field_name == lhs_field:
            raise KeyError(f"Derivative {field_name}_{axis * order} not available")
        torch.manual_seed(hash((field_name, axis, order)) % (2**31))
        return torch.randn(field_shape)

    context.get_variable = get_variable
    context.get_derivative = get_derivative

    return PlatformComponents(
        dataset=dataset,
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=context,
        registry=MagicMock(),
    )


class TestPrepareGuardrails:



    @pytest.mark.unit
    def test_lhs_derivative_missing_raises_before_population_init(self) -> None:
        from unittest.mock import patch

        from kd.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
            lhs_deriv_available=False,
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))



        init_called = False
        original_init = plugin._init_population

        def tracking_init() -> None:
            nonlocal init_called
            init_called = True
            original_init()

        with (
            patch.object(plugin, "_init_population", tracking_init),
            pytest.raises(
                (ValueError, RuntimeError),
                match="[Ll][Hh][Ss]|[Dd]erivative",
            ),
        ):
            plugin.prepare(components)

        assert not init_called, (
            "LHS derivative missing should raise BEFORE _init_population, "
            "not as a side effect of population init failure"
        )

    @pytest.mark.unit
    def test_lhs_derivative_missing_error_message_mentions_derivative(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
            lhs_deriv_available=False,
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            plugin.prepare(components)

        msg = str(exc_info.value).lower()

        assert "lhs" in msg or "derivative" in msg or "u_t" in msg, (
            f"Error message should mention LHS derivative, got: {exc_info.value}"
        )

    @pytest.mark.unit
    def test_normal_dataset_still_works(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
            lhs_deriv_available=True,
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))

        plugin.prepare(components)
        assert plugin._prepared



    @pytest.mark.unit
    def test_lhs_axis_not_in_vars_after_prepare(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        plugin.prepare(components)

        vars_list = plugin.state["vars"]
        assert "t" not in vars_list, "lhs_axis 't' leaked into VARS"

    @pytest.mark.unit
    def test_lhs_axis_derivatives_not_in_vars(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        plugin.prepare(components)

        vars_list = plugin.state["vars"]
        lhs_derivs = [v for v in vars_list if v.endswith("_t") or v.endswith("_tt")]
        assert len(lhs_derivs) == 0, (
            f"LHS-axis derivatives leaked into VARS: {lhs_derivs}"
        )



    @pytest.mark.unit
    def test_field_name_conflicts_with_axis_name_raises(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)

        x_field = torch.randn(10, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"x": x_field, "u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        with pytest.raises(
            ValueError, match="[Cc]onflict|[Aa]mbig|[Oo]verlap|[Dd]uplicate"
        ):
            plugin.prepare(components)

    @pytest.mark.unit
    def test_field_name_conflicts_with_derivative_key_raises(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)

        u_x_field = torch.randn(10, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u_x": u_x_field, "u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        with pytest.raises(ValueError, match="[Cc]onflict|[Dd]erivative"):
            plugin.prepare(components)

    @pytest.mark.unit
    def test_lhs_derivative_alias_in_fields_raises(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)

        ut_field = torch.randn(10, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"ut": ut_field, "u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        with pytest.raises(ValueError, match="[Cc]onflict|[Aa]lias|[Ll]hs"):
            plugin.prepare(components)



    @pytest.mark.unit
    def test_multi_field_no_conflict_passes(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)
        v = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u, "v": v},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        plugin.prepare(components)

        vars_list = plugin.state["vars"]

        assert "u" in vars_list
        assert "v" in vars_list

        assert "u_x" in vars_list
        assert "v_x" in vars_list

        assert "u_t" not in vars_list
        assert "v_t" not in vars_list



    @pytest.mark.unit
    def test_lhs_axis_not_in_den(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        plugin.prepare(components)


        den = plugin._den
        den_axes = [entry[0] for entry in den]
        assert "t" not in den_axes, f"lhs_axis 't' leaked into den: {den}"







class TestAICLowerBound:



    @pytest.mark.unit
    def test_valid_aic_accepts_normal_value(self) -> None:
        from kd.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(50.0) is True

    @pytest.mark.unit
    def test_valid_aic_accepts_zero(self) -> None:
        from kd.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(0.0) is True

    @pytest.mark.unit
    def test_valid_aic_accepts_negative_above_bound(self) -> None:
        from kd.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(-50.0) is True

    @pytest.mark.unit
    def test_valid_aic_accepts_at_bound(self) -> None:
        from kd.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(-100.0) is True

    @pytest.mark.unit
    def test_valid_aic_rejects_below_bound(self) -> None:
        from kd.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(-200.0) is False

    @pytest.mark.unit
    def test_valid_aic_rejects_inf(self) -> None:
        from kd.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(float("inf")) is False

    @pytest.mark.unit
    def test_valid_aic_rejects_neg_inf(self) -> None:
        from kd.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(float("-inf")) is False

    @pytest.mark.unit
    def test_valid_aic_rejects_nan(self) -> None:
        from kd.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(float("nan")) is False

    @pytest.mark.unit
    def test_valid_aic_rejects_barely_below_bound(self) -> None:
        from kd.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(-100.01) is False



    @pytest.mark.unit
    def test_aic_lower_bound_constant_exists(self) -> None:
        from kd.search.sga.plugin import _AIC_LOWER_BOUND

        assert _AIC_LOWER_BOUND == -100.0



    @pytest.mark.unit
    def test_evaluate_marks_extreme_negative_aic_invalid(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from unittest.mock import patch

        from kd.search.sga.plugin import SGAPlugin
        from kd.search.sga.train import CandidateResult, TrainResult


        mock_tr = TrainResult(
            coefficients=torch.tensor([1.0]),
            selected_indices=[0],
            aic_score=-200.0,
            mse=0.001,
            best_tol=0.1,
        )

        def fake_evaluate_candidate(*args: object, **kwargs: object) -> CandidateResult:
            pde = args[0] if args else MagicMock()
            return CandidateResult(
                train_result=mock_tr,
                pruned_pde=pde,
                valid_term_indices=[0],
            )

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        with patch(
            "kd.search.sga.plugin.evaluate_candidate",
            side_effect=fake_evaluate_candidate,
        ):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)


        for r in results:
            assert r.is_valid is False, (
                f"Expected is_valid=False for AIC=-200, got is_valid={r.is_valid}"
            )
            assert r.invalid_reason == "structural_reject"


            assert r.score == pytest.approx(-200.0), (
                f"Expected AIC=-200.0 preserved, got {r.score} "
                f"(exception handler sets inf — wrong code path)"
            )

    @pytest.mark.unit
    def test_evaluate_marks_normal_aic_valid(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from unittest.mock import patch

        from kd.search.sga.plugin import SGAPlugin
        from kd.search.sga.train import CandidateResult, TrainResult

        mock_tr = TrainResult(
            coefficients=torch.tensor([1.0]),
            selected_indices=[0],
            aic_score=-50.0,
            mse=0.01,
            best_tol=0.1,
        )

        def fake_evaluate_candidate(*args: object, **kwargs: object) -> CandidateResult:
            pde = args[0] if args else MagicMock()
            return CandidateResult(
                train_result=mock_tr,
                pruned_pde=pde,
                valid_term_indices=[0],
            )

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        with patch(
            "kd.search.sga.plugin.evaluate_candidate",
            side_effect=fake_evaluate_candidate,
        ):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)


        for r in results:
            assert r.is_valid is True, (
                f"Expected is_valid=True for AIC=-50, got is_valid={r.is_valid}"
            )



    @pytest.mark.unit
    def test_init_resamples_extreme_negative_aic(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        from unittest.mock import patch

        from kd.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)
        call_count = 0

        def aic_with_resample(*args: object, **kwargs: object) -> tuple[float, object]:
            nonlocal call_count
            call_count += 1
            pde = args[0] if args else None


            if call_count % 2 == 1:
                return -200.0, pde
            return 10.0, pde

        with patch(
            "kd.search.sga.plugin._safe_evaluate_aic",
            side_effect=aic_with_resample,
        ):
            plugin = SGAPlugin(config=config)
            plugin.prepare(mock_components)


        scores = plugin.state["scores"]
        assert scores is not None
        assert len(scores) == config.num
        for s in scores:
            assert s >= -100.0, f"Population contains AIC below lower bound: {s}"
            assert math.isfinite(s), f"Population contains non-finite AIC: {s}"







class TestSGAPluginNmseSemantics:

    @pytest.mark.unit
    def test_to_eval_result_nmse_is_normalized(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin
        from kd.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        plugin._y = y
        target_var = float(torch.var(y, correction=0).item())
        mse = 0.5
        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=-10.0,
                mse=mse,
                best_tol=0.1,
            ),
            expression="u",
        )

        assert result.nmse == pytest.approx(mse / target_var, rel=1e-9)
        assert result.nmse != pytest.approx(mse), (
            "nmse must not equal raw mse when target variance > 0"
        )

    @pytest.mark.unit
    def test_to_eval_result_nmse_falls_back_when_target_constant(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin
        from kd.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        plugin._y = torch.full((10,), 1.7)
        mse = 0.123
        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=-10.0,
                mse=mse,
                best_tol=0.1,
            ),
            expression="u",
        )
        assert result.nmse == pytest.approx(mse, rel=1e-9)

    @pytest.mark.unit
    def test_target_variance_helper(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)

        plugin._y = None
        assert plugin._target_variance() == 0.0


        plugin._y = torch.tensor([0.0, 2.0, 4.0])

        assert plugin._target_variance() == pytest.approx(8.0 / 3.0, rel=1e-6)

    @pytest.mark.unit
    def test_target_variance_single_element_returns_zero(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin._y = torch.tensor([3.14])
        assert plugin._target_variance() == 0.0

        plugin._y = torch.tensor([])
        assert plugin._target_variance() == 0.0

    @pytest.mark.unit
    def test_to_eval_result_r2_uses_target_variance(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin
        from kd.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        plugin._y = y
        target_var = float(torch.var(y, correction=0).item())
        mse = 0.5
        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=-10.0,
                mse=mse,
                best_tol=0.1,
            ),
            expression="u",
        )

        assert result.r2 == pytest.approx(1.0 - mse / target_var, rel=1e-9)

    @pytest.mark.unit
    def test_to_eval_result_r2_falls_back_when_target_constant(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin
        from kd.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        plugin._y = torch.full((10,), 1.7)
        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=-10.0,
                mse=0.123,
                best_tol=0.1,
            ),
            expression="u",
        )
        assert result.r2 == 0.0

    @pytest.mark.unit
    def test_to_eval_result_r2_minus_inf_when_mse_not_finite(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin
        from kd.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        plugin._y = torch.tensor([0.0, 1.0, 2.0])
        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=float("inf"),
                mse=float("inf"),
                best_tol=0.1,
            ),
            expression="u",
        )
        assert result.r2 == -math.inf

    @pytest.mark.unit
    def test_to_eval_result_r2_baseline_matches_compute_r2(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin
        from kd.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)



        y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        plugin._y = y
        mse_baseline = float(((y - y.mean()) ** 2).mean().item())
        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=-10.0,
                mse=mse_baseline,
                best_tol=0.1,
            ),
            expression="u",
        )
        assert result.r2 == pytest.approx(0.0, abs=1e-9)


        mean_prediction = torch.full_like(y, y.mean().item())
        assert plugin._compute_r2(mean_prediction) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.unit
    def test_r2_from_mse_can_go_below_minus_one(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        plugin._y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])

        assert plugin._r2_from_mse(20.0) == pytest.approx(-9.0, rel=1e-9)

    @pytest.mark.unit
    def test_compute_r2_can_go_below_minus_one(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        plugin._y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        bad_prediction = torch.full_like(plugin._y, 10.0)
        assert plugin._compute_r2(bad_prediction) == pytest.approx(-32.0, rel=1e-9)

    @pytest.mark.unit
    def test_perfect_fit_on_constant_target_returns_one(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin
        from kd.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        plugin._y = torch.full((10,), 1.7)

        perfect_prediction = torch.full_like(plugin._y, 1.7)
        assert plugin._compute_r2(perfect_prediction) == pytest.approx(1.0, abs=1e-9)

        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=-10.0,
                mse=0.0,
                best_tol=0.1,
            ),
            expression="u",
        )
        assert result.r2 == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.unit
    def test_compute_r2_float32_constant_target_matches_platform(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.core.linear_solve import compute_r2
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)


        y = torch.full((50,), 1.7, dtype=torch.float32)
        plugin._y = y
        predicted = y + 2e-7


        theta = predicted.unsqueeze(-1)
        coef = torch.ones(1, dtype=predicted.dtype)
        platform_r2 = compute_r2(theta, coef, y)

        assert platform_r2 == pytest.approx(1.0), (
            "sanity: platform treats float32-rounding residual as perfect fit"
        )
        assert plugin._compute_r2(predicted) == pytest.approx(platform_r2), (
            "SGA _compute_r2 must agree with platform compute_r2 on the "
            "float32 + constant-target boundary case (J-1 drift)"
        )

    @pytest.mark.unit
    def test_r2_from_mse_perfect_fit_eps_aligned_with_platform(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        plugin._y = torch.full((50,), 1.7, dtype=torch.float64)

        assert plugin._r2_from_mse(1e-12) == pytest.approx(1.0), (
            "_r2_from_mse must treat rounding-scale mse on a constant "
            "target as a perfect fit (platform R2_EPS_RES)"
        )







class TestSGAPluginConfig:

    @pytest.mark.smoke
    def test_has_config_property(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert hasattr(plugin, "config")

    @pytest.mark.unit
    def test_config_returns_dict(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        cfg = plugin.config
        assert isinstance(cfg, dict)

    @pytest.mark.unit
    def test_config_contains_algorithm_key(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        cfg = plugin.config

        has_identifier = any(
            isinstance(v, str) and "sga" in v.lower() for v in cfg.values()
        )
        assert has_identifier, f"config should contain an SGA identifier. Got: {cfg}"

    @pytest.mark.unit
    def test_config_reflects_sga_config_params(self, sga_config: SGAConfig) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        cfg = plugin.config


        assert len(cfg) >= 3, (
            f"Expected at least 3 config entries, got {len(cfg)}: {cfg}"
        )


        param_values = set()
        for v in cfg.values():
            if isinstance(v, (int, float)):
                param_values.add(v)


        sga_values = {
            sga_config.num,
            sga_config.depth,
            sga_config.width,
            sga_config.seed,
        }
        overlap = param_values & sga_values
        assert len(overlap) >= 1, (
            f"Expected some SGAConfig values in config. "
            f"Config values: {param_values}, SGAConfig: {sga_values}"
        )

    @pytest.mark.unit
    def test_config_is_pickle_serializable(self, sga_config: SGAConfig) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        cfg = plugin.config

        pickled = pickle.dumps(cfg)
        restored = pickle.loads(pickled)
        assert restored == cfg

    @pytest.mark.unit
    def test_injected_field_model_is_bound_as_artifact(self) -> None:
        from kd.models.field_model import FieldModel
        from kd.search.sga.plugin import SGAPlugin

        model = FieldModel(["x", "t"], ["u"], hidden_sizes=[2])
        plugin = SGAPlugin(SGAConfig(use_autograd=True, field_model=model))

        assert plugin.config["field_model"] == {
            "artifact": "field_model",
            "format": "kd-field-model-v1",
        }
        json.dumps(plugin.config, allow_nan=False)
        artifacts_before = plugin.artifacts
        assert artifacts_before is not None
        assert len(artifacts_before["field_model"]["sha256"]) == 64

        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        artifacts_after = plugin.artifacts
        assert artifacts_after is not None
        assert (
            artifacts_after["field_model"]["sha256"]
            != artifacts_before["field_model"]["sha256"]
        )

    @pytest.mark.unit
    def test_config_before_and_after_prepare_same_params(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        cfg_before = plugin.config

        plugin.prepare(mock_components)
        cfg_after = plugin.config


        for key in cfg_before:
            assert key in cfg_after, f"Key '{key}' lost after prepare()"







class TestSGAPluginRecorder:

    @pytest.mark.unit
    def test_prepare_stores_recorder_ref(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.recorder import VizRecorder
        from kd.search.sga.plugin import SGAPlugin

        recorder = VizRecorder()
        mock_components.recorder = recorder

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)



        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)


        assert len(recorder.keys()) > 0

    @pytest.mark.unit
    def test_update_logs_to_recorder(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.recorder import VizRecorder
        from kd.search.sga.plugin import SGAPlugin

        recorder = VizRecorder()
        mock_components.recorder = recorder

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)


        all_keys = recorder.keys()
        assert len(all_keys) > 0


        has_single_entry = any(len(recorder.get(k)) == 1 for k in all_keys)
        assert has_single_entry

    @pytest.mark.unit
    def test_no_recorder_no_crash(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin


        assert mock_components.recorder is None

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)

        plugin.update(results)

    @pytest.mark.unit
    def test_disabled_recorder_no_crash(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.recorder import VizRecorder
        from kd.search.sga.plugin import SGAPlugin

        recorder = VizRecorder(enabled=False)
        mock_components.recorder = recorder

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)


        assert len(recorder.keys()) == 0

    @pytest.mark.unit
    def test_multiple_updates_accumulate_in_recorder(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.recorder import VizRecorder
        from kd.search.sga.plugin import SGAPlugin

        recorder = VizRecorder()
        mock_components.recorder = recorder

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        n_cycles = 3
        for _ in range(n_cycles):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)


        has_n_entries = any(len(recorder.get(k)) == n_cycles for k in recorder.keys())
        assert has_n_entries, (
            f"Expected a series with {n_cycles} entries. "
            f"Keys: {recorder.keys()}, "
            f"lengths: {[(k, len(recorder.get(k))) for k in recorder.keys()]}"
        )







class TestSGAPluginBuildMethods:

    @pytest.mark.smoke
    def test_default_final_result_importable(self) -> None:
        from kd.search.result import default_final_result

        assert callable(default_final_result)

    @pytest.mark.unit
    def test_sga_plugin_isinstance_merged_search_algorithm(self) -> None:
        from kd.search.protocol import SearchAlgorithm
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert isinstance(plugin, SearchAlgorithm)
        assert callable(plugin.build_final_result)

    @pytest.mark.unit
    def test_has_build_final_result_method(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert callable(getattr(plugin, "build_final_result", None))

    @pytest.mark.unit
    def test_build_final_result_returns_evaluation_result(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)


        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert isinstance(final, EvaluationResult)

    @pytest.mark.unit
    def test_build_final_result_is_valid(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert final.is_valid is True

    @pytest.mark.unit
    def test_build_final_result_has_residuals(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert final.residuals is not None
        assert isinstance(final.residuals, torch.Tensor)

    @pytest.mark.unit
    def test_build_final_result_has_coefficients(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert final.coefficients is not None
        assert isinstance(final.coefficients, torch.Tensor)

    @pytest.mark.unit
    def test_build_final_result_has_finite_mse(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert math.isfinite(final.mse)

    @pytest.mark.unit
    def test_build_final_result_has_terms(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()

        assert final.terms is not None
        assert isinstance(final.terms, list)

    @pytest.mark.unit
    def test_build_final_result_has_expression(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert isinstance(final.expression, str)
        assert len(final.expression) > 0

    @pytest.mark.unit
    def test_build_final_result_residuals_shape_matches_data(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        expected_n = _SMALL_GRID_SIZE * _SMALL_TIME_SIZE

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert final.residuals is not None
        assert final.residuals.numel() == expected_n

    @pytest.mark.numerical
    def test_build_final_result_residuals_finite(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert final.residuals is not None
        assert torch.isfinite(final.residuals).all(), "residuals contain NaN or Inf"

    @pytest.mark.unit
    def test_build_final_result_r2_in_range(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert math.isfinite(final.r2)

    @pytest.mark.unit
    def test_build_final_result_before_any_update_raises_or_is_valid(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)


        final = plugin.build_final_result()
        assert isinstance(final, EvaluationResult)

        assert final.is_valid is True







class TestSGAPluginPDELib:

    @pytest.mark.unit
    def test_pde_lib_initialized_empty(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert plugin._pde_lib == set()
        assert plugin._repeat_cross == 0
        assert plugin._repeat_change == 0

    @pytest.mark.unit
    def test_pde_lib_grows_after_propose(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)


        assert len(plugin._pde_lib) == 0

        plugin.propose(sga_config.num)
        assert len(plugin._pde_lib) > 0, (
            "pde_lib must contain offspring expressions after first propose()"
        )

    @pytest.mark.unit
    def test_repeat_cross_counted_when_crossover_produces_duplicate(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.sga import plugin as plugin_module
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        sentinel = "DUP_SENTINEL"


        monkeypatch.setattr(
            plugin_module,
            "pde_to_kd_expr",
            lambda pde, coefficients=None: sentinel,
        )

        plugin._pde_lib.add(sentinel)


        assert sga_config.p_cro > 0
        assert sga_config.num >= 2

        plugin.propose(sga_config.num)

        assert plugin._repeat_cross > 0, (
            "Expected duplicate crossover offspring to increment _repeat_cross"
        )

    @pytest.mark.unit
    def test_repeat_counters_reset_each_generation(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)


        plugin.propose(sga_config.num)
        first_cross = plugin._repeat_cross
        first_change = plugin._repeat_change




        sentinel_cross = 9999
        sentinel_change = 8888
        plugin._repeat_cross = sentinel_cross
        plugin._repeat_change = sentinel_change

        plugin._apply_genetic_ops()



        assert plugin._repeat_cross < sentinel_cross, (
            "_repeat_cross was not reset at the start of _apply_genetic_ops"
        )
        assert plugin._repeat_change < sentinel_change, (
            "_repeat_change was not reset at the start of _apply_genetic_ops"
        )


        assert plugin._repeat_cross <= 4 * sga_config.num
        assert plugin._repeat_change <= 4 * sga_config.num


        del first_cross, first_change

    @pytest.mark.unit
    def test_state_round_trip_preserves_pde_lib(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        plugin.propose(sga_config.num)

        lib_before = set(plugin._pde_lib)
        assert len(lib_before) > 0, "Precondition: lib should be populated"

        state = plugin.state
        assert "pde_lib" in state, "state must serialize pde_lib"

        serialized = state["pde_lib"]
        assert isinstance(serialized, (list, set, tuple))
        assert set(serialized) == lib_before


        plugin2 = SGAPlugin(config=sga_config)
        plugin2.state = state

        assert plugin2._pde_lib == lib_before

    @pytest.mark.unit
    def test_state_setter_with_no_pde_lib_key_resets_to_empty(
        self,
        sga_config: SGAConfig,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)

        plugin._pde_lib.add("STALE_ENTRY")
        plugin._repeat_cross = 42
        plugin._repeat_change = 17


        legacy_state = {
            "population": None,
            "scores": None,
            "best_score": float("inf"),
            "best_expression": "",
            "vars": [],
        }
        plugin.state = legacy_state

        assert plugin._pde_lib == set()
        assert plugin._repeat_cross == 0
        assert plugin._repeat_change == 0

    @pytest.mark.unit
    def test_prepare_resets_pde_lib_from_prior_run(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)

        plugin._pde_lib.add("STALE_FROM_PRIOR_RUN")
        plugin._repeat_cross = 42
        plugin._repeat_change = 17

        plugin.prepare(mock_components)

        assert plugin._pde_lib == set()
        assert plugin._repeat_cross == 0
        assert plugin._repeat_change == 0







@pytest.fixture
def dedup_config_factory(sga_config: SGAConfig):
    from dataclasses import replace

    def _make(mode: str) -> SGAConfig:
        return replace(sga_config, dedup_mode=mode)

    return _make


class TestSGAPluginDedupMode:



    @pytest.mark.unit
    def test_none_mode_does_not_populate_pde_lib(
        self,
        dedup_config_factory,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        config = dedup_config_factory("none")
        plugin = SGAPlugin(config=config)
        plugin.prepare(mock_components)

        assert plugin._pde_lib == set(), "Precondition: prepare() resets _pde_lib"

        plugin.propose(config.num)

        assert plugin._pde_lib == set(), (
            'dedup_mode="none" must not populate _pde_lib (no dedup => no bookkeeping)'
        )

    @pytest.mark.unit
    def test_none_mode_never_increments_repeat_cross(
        self,
        dedup_config_factory,
        mock_components: PlatformComponents,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.sga import plugin as plugin_module
        from kd.search.sga.plugin import SGAPlugin

        config = dedup_config_factory("none")
        plugin = SGAPlugin(config=config)
        plugin.prepare(mock_components)

        sentinel = "DUP_SENTINEL_NONE_MODE"
        monkeypatch.setattr(
            plugin_module,
            "pde_to_kd_expr",
            lambda pde, coefficients=None: sentinel,
        )


        plugin._pde_lib.add(sentinel)


        assert config.p_cro > 0
        assert config.num >= 2

        plugin.propose(config.num)

        assert plugin._repeat_cross == 0, (
            'dedup_mode="none" must not increment _repeat_cross '
            "(dedup is disabled — every offspring is accepted)"
        )
        assert plugin._repeat_change == 0, (
            'dedup_mode="none" must not increment _repeat_change'
        )



    @pytest.mark.unit
    def test_pre_prune_mode_increments_repeat_cross_on_duplicate(
        self,
        dedup_config_factory,
        mock_components: PlatformComponents,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.sga import plugin as plugin_module
        from kd.search.sga.plugin import SGAPlugin

        config = dedup_config_factory("pre_prune")
        plugin = SGAPlugin(config=config)
        plugin.prepare(mock_components)

        sentinel = "DUP_SENTINEL_PRE_PRUNE"
        monkeypatch.setattr(
            plugin_module,
            "pde_to_kd_expr",
            lambda pde, coefficients=None: sentinel,
        )
        plugin._pde_lib.add(sentinel)

        assert config.p_cro > 0
        assert config.num >= 2

        plugin.propose(config.num)

        assert plugin._repeat_cross > 0, (
            'dedup_mode="pre_prune" must reject crossover offspring whose '
            "raw-genotype key already appears in _pde_lib"
        )



    @pytest.mark.unit
    def test_post_prune_mode_increments_repeat_cross_on_duplicate(
        self,
        dedup_config_factory,
        mock_components: PlatformComponents,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.sga import plugin as plugin_module
        from kd.search.sga.plugin import SGAPlugin

        config = dedup_config_factory("post_prune")
        plugin = SGAPlugin(config=config)
        plugin.prepare(mock_components)

        sentinel = "DUP_SENTINEL_POST_PRUNE"
        monkeypatch.setattr(
            plugin_module,
            "pde_to_kd_expr",
            lambda pde, coefficients=None: sentinel,
        )
        plugin._pde_lib.add(sentinel)

        assert config.p_cro > 0
        assert config.num >= 2

        plugin.propose(config.num)

        assert plugin._repeat_cross > 0, (
            'dedup_mode="post_prune" must reject offspring whose '
            "pruned-genotype key already appears in _pde_lib"
        )

    @pytest.mark.unit
    def test_post_prune_distinguishes_from_pre_prune_when_keys_differ(
        self,
        dedup_config_factory,
        mock_components: PlatformComponents,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.sga import plugin as plugin_module
        from kd.search.sga.plugin import SGAPlugin

        config_post = dedup_config_factory("post_prune")
        plugin_post = SGAPlugin(config=config_post)
        plugin_post.prepare(mock_components)







        call_log: list[str] = []

        def fake_expr(pde, coefficients=None):


            n = len(call_log)
            call_log.append("call")

            return f"RAW_{n}" if n % 2 == 0 else "PRUNED"

        monkeypatch.setattr(plugin_post, "_pde_lib", {"PRUNED"})
        monkeypatch.setattr(
            plugin_module,
            "pde_to_kd_expr",
            fake_expr,
        )

        assert config_post.p_cro > 0
        plugin_post.propose(config_post.num)
        post_prune_hits = plugin_post._repeat_cross + plugin_post._repeat_change


        call_log.clear()
        config_pre = dedup_config_factory("pre_prune")
        plugin_pre = SGAPlugin(config=config_pre)
        plugin_pre.prepare(mock_components)
        monkeypatch.setattr(plugin_pre, "_pde_lib", {"PRUNED"})

        plugin_pre.propose(config_pre.num)
        pre_prune_hits = plugin_pre._repeat_cross + plugin_pre._repeat_change




        assert post_prune_hits > pre_prune_hits, (
            f'dedup_mode="post_prune" must catch pruned-key collisions that '
            f'"pre_prune" misses. Got post={post_prune_hits}, pre={pre_prune_hits}. '
            "If equal, post_prune is not actually using the pruned key."
        )



    @pytest.mark.unit
    def test_dual_mode_increments_counters_on_pre_prune_duplicate(
        self,
        dedup_config_factory,
        mock_components: PlatformComponents,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.sga import plugin as plugin_module
        from kd.search.sga.plugin import SGAPlugin

        config = dedup_config_factory("dual")
        plugin = SGAPlugin(config=config)
        plugin.prepare(mock_components)

        sentinel = "DUP_SENTINEL_DUAL"
        monkeypatch.setattr(
            plugin_module,
            "pde_to_kd_expr",
            lambda pde, coefficients=None: sentinel,
        )
        plugin._pde_lib.add(sentinel)

        assert config.p_cro > 0
        plugin.propose(config.num)

        assert plugin._repeat_cross > 0, (
            'dedup_mode="dual" must include the pre-prune cheap filter; '
            "every offspring with a duplicate raw-genotype key should "
            "increment _repeat_cross"
        )

    @pytest.mark.unit
    def test_dual_mode_strictly_stricter_than_pre_prune(
        self,
        dedup_config_factory,
        mock_components: PlatformComponents,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.core.evaluator import EvaluationResult
        from kd.search.sga import plugin as plugin_module
        from kd.search.sga.plugin import SGAPlugin, _ScoredPDE

        def fake_score(self_inner, candidate_pde):

            return _ScoredPDE(
                candidate_pde,
                10.0,
                EvaluationResult(
                    mse=1.0,
                    nmse=1.0,
                    r2=0.5,
                    score=10.0,
                    complexity=1,
                    coefficients=None,
                    is_valid=True,
                    error_message="",
                    selected_indices=[0],
                    residuals=None,
                    terms=["u"],
                    expression="PRUNED",
                ),
            )

        counter = [0]

        def fake_expr(pde, coefficients=None):
            counter[0] += 1
            return f"UNIQUE_RAW_{counter[0]}"


        config_pre = dedup_config_factory("pre_prune")
        plugin_pre = SGAPlugin(config=config_pre)
        plugin_pre.prepare(mock_components)
        monkeypatch.setattr(SGAPlugin, "_score_offspring", fake_score)
        monkeypatch.setattr(plugin_module, "pde_to_kd_expr", fake_expr)
        monkeypatch.setattr(plugin_pre, "_pde_lib", {"PRUNED"})
        plugin_pre.propose(config_pre.num)
        pre_hits = plugin_pre._repeat_cross + plugin_pre._repeat_change


        counter[0] = 0
        config_dual = dedup_config_factory("dual")
        plugin_dual = SGAPlugin(config=config_dual)
        plugin_dual.prepare(mock_components)
        monkeypatch.setattr(plugin_dual, "_pde_lib", {"PRUNED"})
        plugin_dual.propose(config_dual.num)
        dual_hits = plugin_dual._repeat_cross + plugin_dual._repeat_change

        assert dual_hits > pre_hits, (
            f'dedup_mode="dual" must catch pruned-key collisions that '
            f'"pre_prune" misses. Got dual={dual_hits}, pre={pre_hits}. '
            "If equal, dual's post-prune branch is not actually firing."
        )
        assert dual_hits > 0, (
            "Dual must fire post-prune branch in this constructed scenario "
            "(non-vacuous regression for the previous '>=' assertion)."
        )

    @pytest.mark.unit
    def test_post_prune_does_not_pollute_lib_on_score_failure(
        self,
        dedup_config_factory,
        mock_components: PlatformComponents,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.sga.plugin import (
            _FAILED_EVAL_ERROR_MESSAGE,
            SGAPlugin,
            _ScoredPDE,
        )

        failed_marker = "RAW_FROM_FAILED_SCORING_DO_NOT_LEAK"

        def fake_score(self_inner, candidate_pde):



            result = self_inner._invalid_result(failed_marker)
            result.error_message = _FAILED_EVAL_ERROR_MESSAGE
            return _ScoredPDE(
                candidate_pde,
                float("inf"),
                result,
            )


        config_post = dedup_config_factory("post_prune")
        plugin_post = SGAPlugin(config=config_post)
        plugin_post.prepare(mock_components)
        monkeypatch.setattr(SGAPlugin, "_score_offspring", fake_score)
        plugin_post.propose(config_post.num)
        assert failed_marker not in plugin_post._pde_lib, (
            'dedup_mode="post_prune" leaked raw key from a failed scoring '
            "into _pde_lib (H1 regression)."
        )







        from kd.search.sga import plugin as plugin_module

        dual_pre_marker = "DUAL_PRE_KEY_DO_NOT_LEAK"
        monkeypatch.setattr(
            plugin_module,
            "pde_to_kd_expr",
            lambda pde, coefficients=None: dual_pre_marker,
        )

        config_dual = dedup_config_factory("dual")
        plugin_dual = SGAPlugin(config=config_dual)
        plugin_dual.prepare(mock_components)
        plugin_dual.propose(config_dual.num)
        assert failed_marker not in plugin_dual._pde_lib, (
            'dedup_mode="dual" leaked raw key from a failed scoring '
            "into _pde_lib (H1 regression — post_key path)."
        )
        assert dual_pre_marker not in plugin_dual._pde_lib, (
            'dedup_mode="dual" wrote pre_key BEFORE the failure-sentinel '
            "check, polluting _pde_lib (H1 regression — pre_key path)."
        )



    @pytest.mark.unit
    def test_invalid_dedup_mode_raises_value_error(
        self,
        dedup_config_factory,
        mock_components: PlatformComponents,
    ) -> None:
        from kd.search.sga.plugin import SGAPlugin

        config = dedup_config_factory("BOGUS_NOT_A_MODE")
        plugin = SGAPlugin(config=config)
        plugin.prepare(mock_components)

        with pytest.raises(ValueError, match=r"(?i)dedup_mode"):
            plugin.propose(config.num)



    @pytest.mark.unit
    def test_default_mode_matches_pre_prune_behavior(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kd.search.sga import plugin as plugin_module
        from kd.search.sga.plugin import SGAPlugin


        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        sentinel = "DUP_SENTINEL_DEFAULT"
        monkeypatch.setattr(
            plugin_module,
            "pde_to_kd_expr",
            lambda pde, coefficients=None: sentinel,
        )
        plugin._pde_lib.add(sentinel)

        plugin.propose(sga_config.num)



        assert plugin._repeat_cross > 0, (
            "Default dedup_mode must preserve current pre_prune behavior. "
            "If this fails, dev changed the default — see step 5 "
            "(default change is gated on ablation results)."
        )







class TestSGALhsOrderFailLoud:

    def test_extract_lhs_target_rejects_second_order_lhs(self) -> None:
        from kd.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(SGAConfig(num=4, depth=3, width=3, seed=0))
        nx, nt = 6, 5
        x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
        t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
        u = torch.randn(nx, nt, dtype=torch.float64)
        ds = PDEDataset.from_arrays(
            coords={"x": x, "t": t}, fields={"u": u}, lhs="u_tt"
        )
        assert ds.lhs_order == 2
        with pytest.raises(ValueError, match="lhs_order|first-order"):
            plugin._extract_lhs_target(ds, MagicMock())
