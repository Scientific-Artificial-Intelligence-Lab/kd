
from __future__ import annotations

import math

import pytest
import torch

from kd.core.evaluator import Evaluator
from kd.core.executor.context import ExecutionContext
from kd.core.expr import FunctionRegistry, PythonExecutor
from kd.core.linear_solve.least_squares import LeastSquaresSolver
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.synthetic import generate_burgers_data
from kd.search.protocol import PlatformComponents
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig, SGAPlugin
from kd.search.sga.config import OP1, OP2, OPS, ROOT
from kd.search.sga.evaluate import prune_invalid_terms
from kd.search.sga.genetic import crossover, mutate, replace
from kd.search.sga.pde import PDE
from kd.search.sga.train import evaluate_candidate





_NX = 128
_NT = 51
_NU = 0.1
_SEED = 42
_POPULATION = 10
_GENERATIONS = 5







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


@pytest.fixture(scope="module")
def prepared_plugin(burgers_components: PlatformComponents) -> SGAPlugin:
    config = SGAConfig(
        num=_POPULATION,
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
    plugin = SGAPlugin(config=config)
    plugin.prepare(burgers_components)
    return plugin







class TestDerivativeTreeGeneticOpsFiniteAIC:

    @pytest.mark.integration
    def test_derivative_pde_through_mutation_produces_finite_aic(
        self, prepared_plugin: SGAPlugin
    ) -> None:
        from kd.search.sga.tree import Node, Tree

        state = prepared_plugin.state
        data_dict = prepared_plugin._data_dict
        diff_ctx = prepared_plugin._diff_ctx
        default_terms = prepared_plugin._default_terms
        y = prepared_plugin._y
        config = prepared_plugin._config
        vars_list = state["vars"]
        den = prepared_plugin._den


        d_node = Node(
            name="d",
            arity=2,
            children=[
                Node(name="u", arity=0),
                Node(name="x", arity=0),
            ],
        )
        d_tree = Tree(root=d_node)

        leaf_tree = Tree(root=Node(name="u_x", arity=0))
        pde = PDE(terms=[d_tree, leaf_tree])


        rng = torch.Generator().manual_seed(123)
        mutated = mutate(pde, vars_list, OP1, OP2, den, 0.5, rng)


        cr = evaluate_candidate(
            mutated,
            data_dict,
            default_terms,
            y if y is not None else torch.zeros(1),
            config,
            diff_ctx=diff_ctx,
        )


        assert math.isfinite(cr.aic_score), (
            f"Derivative PDE after mutation has non-finite AIC: {cr.aic_score}"
        )

        assert len(cr.pruned_pde.terms) >= 1

        n_default = default_terms.shape[1] if default_terms is not None else 0
        n_valid = len(cr.pruned_pde.terms)
        assert cr.coefficients.shape[0] == n_default + n_valid, (
            f"Coeff size {cr.coefficients.shape[0]} != theta cols {n_default + n_valid}"
        )

    @pytest.mark.integration
    def test_derivative_pde_through_replace_produces_valid_result(
        self, prepared_plugin: SGAPlugin
    ) -> None:
        from kd.search.sga.tree import Node, Tree

        state = prepared_plugin.state
        data_dict = prepared_plugin._data_dict
        diff_ctx = prepared_plugin._diff_ctx
        default_terms = prepared_plugin._default_terms
        y = prepared_plugin._y
        config = prepared_plugin._config
        vars_list = state["vars"]
        den = prepared_plugin._den


        d2_node = Node(
            name="d^2",
            arity=2,
            children=[
                Node(name="u", arity=0),
                Node(name="x", arity=0),
            ],
        )
        pde = PDE(terms=[Tree(root=d2_node)])

        rng = torch.Generator().manual_seed(456)
        replaced = replace(pde, vars_list, OPS, ROOT, den, 3, 0.5, rng)

        cr = evaluate_candidate(
            replaced,
            data_dict,
            default_terms,
            y if y is not None else torch.zeros(1),
            config,
            diff_ctx=diff_ctx,
        )


        assert not math.isnan(cr.aic_score)







class TestDerivativeCrossoverGenotypeAlignment:

    @pytest.mark.integration
    def test_crossover_derivative_pdes_genotype_sync(
        self, prepared_plugin: SGAPlugin
    ) -> None:
        from kd.search.sga.tree import Node, Tree

        data_dict = prepared_plugin._data_dict
        diff_ctx = prepared_plugin._diff_ctx
        default_terms = prepared_plugin._default_terms
        y = prepared_plugin._y
        config = prepared_plugin._config


        pde1 = PDE(
            terms=[
                Tree(
                    root=Node(
                        name="d",
                        arity=2,
                        children=[
                            Node(name="u", arity=0),
                            Node(name="x", arity=0),
                        ],
                    )
                ),
                Tree(root=Node(name="u_x", arity=0)),
            ]
        )



        pde2 = PDE(
            terms=[
                Tree(
                    root=Node(
                        name="*",
                        arity=2,
                        children=[
                            Node(name="u", arity=0),
                            Node(name="x", arity=0),
                        ],
                    )
                ),
                Tree(
                    root=Node(
                        name="d",
                        arity=2,
                        children=[
                            Node(name="u_x", arity=0),
                            Node(name="x", arity=0),
                        ],
                    )
                ),
            ]
        )

        rng = torch.Generator().manual_seed(789)
        c1, c2 = crossover(pde1, pde2, rng)


        for offspring in (c1, c2):
            cr = evaluate_candidate(
                offspring,
                data_dict,
                default_terms,
                y if y is not None else torch.zeros(1),
                config,
                diff_ctx=diff_ctx,
            )



            n_default = default_terms.shape[1] if default_terms is not None else 0
            n_pde_terms = len(cr.pruned_pde.terms)



            if cr.selected_indices is not None:
                for idx in cr.selected_indices:
                    assert idx < n_default + n_pde_terms, (
                        f"selected_index {idx} out of range "
                        f"(n_default={n_default}, n_pde_terms={n_pde_terms})"
                    )


            if cr.coefficients is not None and cr.coefficients.numel() > 0:
                assert cr.coefficients.shape[0] == n_default + n_pde_terms, (
                    f"Coefficient size {cr.coefficients.shape[0]} != "
                    f"theta cols {n_default + n_pde_terms}"
                )

    @pytest.mark.integration
    def test_crossover_then_mutation_derivative_alignment(
        self, prepared_plugin: SGAPlugin
    ) -> None:
        from kd.search.sga.tree import Node, Tree

        state = prepared_plugin.state
        data_dict = prepared_plugin._data_dict
        diff_ctx = prepared_plugin._diff_ctx
        default_terms = prepared_plugin._default_terms
        y = prepared_plugin._y
        config = prepared_plugin._config
        vars_list = state["vars"]
        den = prepared_plugin._den

        pde1 = PDE(
            terms=[
                Tree(
                    root=Node(
                        name="d^2",
                        arity=2,
                        children=[
                            Node(name="u", arity=0),
                            Node(name="x", arity=0),
                        ],
                    )
                ),
                Tree(root=Node(name="u", arity=0)),
            ]
        )

        pde2 = PDE(
            terms=[
                Tree(root=Node(name="u_x", arity=0)),
                Tree(root=Node(name="x", arity=0)),
            ]
        )

        rng = torch.Generator().manual_seed(101)


        c1, _ = crossover(pde1, pde2, rng)


        m1 = mutate(c1, vars_list, OP1, OP2, den, 0.5, rng)


        cr = evaluate_candidate(
            m1,
            data_dict,
            default_terms,
            y if y is not None else torch.zeros(1),
            config,
            diff_ctx=diff_ctx,
        )


        pruned, valid_terms, valid_indices = prune_invalid_terms(
            m1,
            data_dict,
            diff_ctx=diff_ctx,
        )
        assert len(pruned.terms) == len(valid_indices)
        assert len(cr.pruned_pde.terms) == len(cr.valid_term_indices)







class TestInitPopulationGuardrailsBurgers:

    @pytest.mark.integration
    def test_init_population_all_finite_aic(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = SGAConfig(
            num=_POPULATION,
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
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        scores = plugin.state["scores"]
        assert len(scores) == _POPULATION
        for i, score in enumerate(scores):
            assert math.isfinite(score), (
                f"Population member {i} has non-finite AIC: {score}"
            )

    @pytest.mark.integration
    def test_prepare_guardrails_pass_on_burgers(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = SGAConfig(
            num=5,
            depth=3,
            width=3,
            p_var=0.5,
            seed=_SEED,
        )
        plugin = SGAPlugin(config=config)

        plugin.prepare(burgers_components)


        assert plugin._y is not None
        assert plugin._y.numel() > 0
        assert torch.isfinite(plugin._y).all()

        assert plugin._y.abs().sum().item() > 0, (
            "LHS target (_y) is all zeros — derivative extraction may have failed"
        )

    @pytest.mark.integration
    def test_init_population_finite_then_search_recovers_burgers(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = SGAConfig(
            num=_POPULATION,
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
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_GENERATIONS,
            batch_size=config.num,
        )
        result = runner.run(burgers_components)



        assert result.iterations == _GENERATIONS

        assert math.isfinite(result.best_score), (
            f"Best score after {_GENERATIONS} gens is not finite: {result.best_score}"
        )

        assert len(result.best_expression) > 0

    @pytest.mark.integration
    def test_vars_exclude_lhs_derivatives_after_prepare(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = SGAConfig(num=5, depth=3, width=3, seed=_SEED)
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







class TestPCroZeroFullSearch:

    @pytest.mark.integration
    def test_p_cro_zero_search_completes(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = SGAConfig(
            num=_POPULATION,
            depth=3,
            width=4,
            p_var=0.5,
            p_mute=0.3,
            p_cro=0.0,
            p_rep=1.0,
            seed=_SEED,
            maxit=5,
            str_iters=5,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_GENERATIONS,
            batch_size=config.num,
        )
        result = runner.run(burgers_components)


        assert result.iterations == _GENERATIONS

        assert not math.isnan(result.best_score)

        assert len(result.best_expression) > 0

    @pytest.mark.integration
    def test_p_cro_zero_offspring_count_matches_mutation_only(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = SGAConfig(
            num=_POPULATION,
            depth=3,
            width=4,
            p_var=0.5,
            p_mute=0.3,
            p_cro=0.0,
            p_rep=1.0,
            seed=_SEED,
            maxit=5,
            str_iters=5,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        candidates = plugin.propose(config.num)



        expected_count = config.num - 1
        assert len(candidates) == expected_count, (
            f"With p_cro=0, expected {expected_count} offspring "
            f"(mutation only), got {len(candidates)}"
        )

    @pytest.mark.integration
    def test_p_cro_zero_best_score_improves(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = SGAConfig(
            num=_POPULATION,
            depth=3,
            width=4,
            p_var=0.5,
            p_mute=0.3,
            p_cro=0.0,
            p_rep=1.0,
            seed=_SEED,
            maxit=5,
            str_iters=5,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        initial_score = plugin.best_score

        for _ in range(_GENERATIONS):
            candidates = plugin.propose(config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)

        final_score = plugin.best_score
        assert final_score <= initial_score, (
            f"Score worsened: {initial_score} -> {final_score}"
        )







class TestDelta007NoPreEvalTruncation:

    @pytest.mark.integration
    def test_propose_returns_full_offspring_frontier(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = SGAConfig(
            num=_POPULATION,
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
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        candidates = plugin.propose(config.num)




        assert len(candidates) > config.num, (
            f"Expected offspring count > {config.num} (no pre-eval truncation), "
            f"got {len(candidates)}. may have regressed."
        )

    @pytest.mark.integration
    def test_all_offspring_evaluated_before_selection(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = SGAConfig(
            num=_POPULATION,
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
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        candidates = plugin.propose(config.num)
        results = plugin.evaluate(candidates)


        assert len(results) == len(candidates), (
            f"evaluate() returned {len(results)} results for "
            f"{len(candidates)} candidates"
        )


        plugin.update(results)
        pop = plugin.state["population"]
        assert len(pop) == config.num, (
            f"After update, population size should be {config.num}, got {len(pop)}"
        )







class TestPluginGenotypeSync:

    @pytest.mark.integration
    def test_evaluate_syncs_offspring_genotype(
        self, burgers_components: PlatformComponents
    ) -> None:
        config = SGAConfig(
            num=_POPULATION,
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
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        candidates = plugin.propose(config.num)


        offspring_before = [pde.copy() for pde in (plugin._offspring or [])]
        assert len(offspring_before) > 0


        results = plugin.evaluate(candidates)


        offspring_after = plugin._offspring or []
        assert len(offspring_after) == len(offspring_before)



        pruned_count = sum(
            1
            for before, after in zip(offspring_before, offspring_after, strict=True)
            if after.width < before.width
        )


        assert pruned_count >= 0



        data_dict = plugin._data_dict
        diff_ctx = plugin._diff_ctx
        for i, pde in enumerate(offspring_after):
            _, valid_terms, valid_indices = prune_invalid_terms(
                pde,
                data_dict,
                diff_ctx=diff_ctx,
            )

            assert len(valid_indices) == pde.width, (
                f"Offspring {i}: synced PDE has {pde.width} terms but only "
                f"{len(valid_indices)} are valid — genotype sync failed"
            )
