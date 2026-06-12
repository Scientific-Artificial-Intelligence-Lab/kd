
from __future__ import annotations

import math

import pytest

from kd.core.evaluator import Evaluator
from kd.core.executor.context import ExecutionContext
from kd.core.expr import FunctionRegistry, PythonExecutor
from kd.core.linear_solve.least_squares import LeastSquaresSolver
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.synthetic import generate_burgers_data
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig, SGAPlugin





_NX = 128

_NT = 51

_NU = 0.1

_SEED = 42

_SMOKE_GENERATIONS = 5

_SCIENCE_GENERATIONS = 20

_POPULATION = 10







@pytest.fixture(scope="module")
def burgers_components() -> PlatformComponents:
    dataset = generate_burgers_data(nx=_NX, nt=_NT, nu=_NU, seed=_SEED)
    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(
        dataset=dataset,
        derivative_provider=provider,
    )
    registry = FunctionRegistry.create_default()
    executor = PythonExecutor(registry)
    solver = LeastSquaresSolver()

    u_t = provider.get_derivative("u", "t", order=1).flatten()
    evaluator = Evaluator(
        executor=executor,
        solver=solver,
        context=context,
        lhs=u_t,
    )

    return PlatformComponents(
        dataset=dataset,
        executor=executor,
        evaluator=evaluator,
        context=context,
        registry=registry,
    )


def _make_sga_config(
    generations: int = _SMOKE_GENERATIONS,
    population: int = _POPULATION,
) -> SGAConfig:
    return SGAConfig(
        num=population,
        depth=3,
        width=4,
        p_var=0.5,
        p_mute=0.3,
        p_cro=0.5,
        p_rep=1.0,
        seed=_SEED,
        maxit=5,
        str_iters=5,
        d_tol=0.5,
    )







class TestSGABurgersSmoke:

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_plugin_direct_cycle(self, burgers_components: PlatformComponents) -> None:
        config = _make_sga_config(generations=1)
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        candidates = plugin.propose(config.num)
        assert len(candidates) > 0
        assert all(isinstance(c, str) for c in candidates)

        results = plugin.evaluate(candidates)
        assert len(results) == len(candidates)

        plugin.update(results)
        assert isinstance(plugin.best_score, float)
        assert isinstance(plugin.best_expression, str)

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_runner_completes(self, burgers_components: PlatformComponents) -> None:
        config = _make_sga_config(generations=_SMOKE_GENERATIONS)
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_SMOKE_GENERATIONS,
            batch_size=config.num,
        )
        result = runner.run(burgers_components)

        assert result.iterations == _SMOKE_GENERATIONS
        assert isinstance(result.best_expression, str)
        assert len(result.best_expression) > 0
        assert isinstance(result.best_score, float)
        assert not result.early_stopped

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_no_nan_in_best_score(self, burgers_components: PlatformComponents) -> None:
        config = _make_sga_config(generations=_SMOKE_GENERATIONS)
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_SMOKE_GENERATIONS,
            batch_size=config.num,
        )
        result = runner.run(burgers_components)

        assert not math.isnan(result.best_score)

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_vars_exclude_lhs_axis(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = _make_sga_config()
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        vars_list = plugin.state["vars"]

        assert "t" not in vars_list

        assert "u_t" not in vars_list
        assert "u_tt" not in vars_list

        assert "u" in vars_list
        assert "x" in vars_list
        assert "u_x" in vars_list

        assert "u_xx" not in vars_list

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_seed_reproducibility(self, burgers_components: PlatformComponents) -> None:
        config = _make_sga_config(generations=3)

        plugin1 = SGAPlugin(config=config)
        runner1 = ExperimentRunner(
            algorithm=plugin1, max_iterations=3, batch_size=config.num
        )
        result1 = runner1.run(burgers_components)

        plugin2 = SGAPlugin(config=config)
        runner2 = ExperimentRunner(
            algorithm=plugin2, max_iterations=3, batch_size=config.num
        )
        result2 = runner2.run(burgers_components)

        assert result1.best_expression == result2.best_expression
        assert result1.best_score == result2.best_score







class TestSGABurgersScience:

    @pytest.mark.integration
    def test_best_score_improves_over_generations(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = _make_sga_config(generations=_SCIENCE_GENERATIONS)
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        initial_score = plugin.best_score

        for _ in range(_SCIENCE_GENERATIONS):
            candidates = plugin.propose(config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)

        final_score = plugin.best_score

        assert final_score <= initial_score, (
            f"Final score {final_score} > initial {initial_score}"
        )

    @pytest.mark.integration
    def test_best_expression_contains_spatial_variables(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = _make_sga_config(generations=_SCIENCE_GENERATIONS)
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_SCIENCE_GENERATIONS,
            batch_size=config.num,
        )
        result = runner.run(burgers_components)

        expr = result.best_expression


        spatial_vars = ("u_x", "u_xx", "x", "u")
        assert any(v in expr for v in spatial_vars), (
            f"Best expression '{expr}' does not reference any spatial "
            f"variables from {spatial_vars}"
        )

    @pytest.mark.integration
    def test_best_score_is_finite(self, burgers_components: PlatformComponents) -> None:
        config = _make_sga_config(generations=_SCIENCE_GENERATIONS)
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_SCIENCE_GENERATIONS,
            batch_size=config.num,
        )
        result = runner.run(burgers_components)

        assert math.isfinite(result.best_score), (
            f"best_score is {result.best_score} after {_SCIENCE_GENERATIONS} "
            f"generations — no valid candidate was found"
        )

    @pytest.mark.integration
    def test_population_converges_to_finite_scores(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = _make_sga_config(generations=_SCIENCE_GENERATIONS)
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_SCIENCE_GENERATIONS,
            batch_size=config.num,
        )
        runner.run(burgers_components)

        state = plugin.state
        scores = state["scores"]
        finite_count = sum(1 for s in scores if math.isfinite(s))

        assert finite_count >= len(scores) // 2, (
            f"Only {finite_count}/{len(scores)} population members have "
            f"finite scores — search may have failed"
        )
