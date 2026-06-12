
from __future__ import annotations

import json
import math

import pytest
import sympy
import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.core.expr.sympy_bridge import are_equivalent, to_sympy
from kd.core.platform.builder import _resolve_derivative_requirements
from kd.core.platform.requirements import DerivativeReqs
from kd.search.protocol import (
    IterativeSearchAlgorithm,
    PlatformComponents,
    SearchAlgorithm,
)
from kd.search.pysr.config import PySRConfig


from kd.search.pysr.plugin import PySRPlugin
from kd.search.result import ResultBuilder, ResultTargetProvider
from kd.viz.extension import PlotInfo, VizExtension
from tests.unit.search.pysr.conftest import (
    FakePySRBackend,
    _best_not_in_hof_recipe_factory,
    _const_in_hof_recipe,
    _const_tail_best_recipe,
    _duplicate_complexity_recipe_factory,
    _pure_constant_best_recipe,
    _sqrt_best_recipe,
    make_backend_factory,
    make_invalid_result,
)

pytestmark = pytest.mark.unit



_TERMS = ("u", "u_x", "u_xx")


_PARETO_COMPLEXITY_KEY = "pareto_complexity"
_PARETO_LOSS_KEY = "pareto_loss"
_PARETO_NMSE_KEY = "pareto_nmse"
_SELECTED_COMPLEXITY_KEY = "selected_complexity"


_SELECTED_LOSS_KEY = "selected_loss"
_SELECTED_NMSE_KEY = "selected_nmse"
_PARETO_KEYS = (
    _PARETO_COMPLEXITY_KEY,
    _PARETO_LOSS_KEY,
    _PARETO_NMSE_KEY,
    _SELECTED_COMPLEXITY_KEY,
    _SELECTED_LOSS_KEY,
    _SELECTED_NMSE_KEY,
)







def _contains_unit_term(kd_ir: str) -> bool:
    expr = to_sympy(kd_ir)
    for additive in expr.as_ordered_terms():
        _coeff, rest = additive.as_coeff_Mul()
        if rest == sympy.Integer(1):
            return True
    return False


def _make_plugin(
    backend: FakePySRBackend,
    *,
    terms: tuple[str, ...] = _TERMS,
) -> PySRPlugin:
    config = PySRConfig(terms=terms, seed=0)
    return PySRPlugin(config, backend_factory=make_backend_factory(backend))


def _prepared_plugin(
    components: PlatformComponents,
    backend: FakePySRBackend,
    *,
    terms: tuple[str, ...] = _TERMS,
) -> PySRPlugin:
    plugin = _make_plugin(backend, terms=terms)
    plugin.prepare(components)
    return plugin







class TestProtocolSmoke:

    @pytest.mark.smoke
    def test_construct_default_factory(self) -> None:
        plugin = PySRPlugin()
        assert plugin is not None

    @pytest.mark.smoke
    def test_construct_with_config_and_backend(self) -> None:
        backend = FakePySRBackend()
        plugin = _make_plugin(backend)
        assert plugin is not None

    @pytest.mark.smoke
    def test_is_search_algorithm(self) -> None:
        assert isinstance(_make_plugin(FakePySRBackend()), SearchAlgorithm)

    def test_is_not_iterative_algorithm(self) -> None:
        assert not isinstance(_make_plugin(FakePySRBackend()), IterativeSearchAlgorithm)

    def test_is_viz_extension(self) -> None:
        assert isinstance(_make_plugin(FakePySRBackend()), VizExtension)

    def test_is_result_builder(self) -> None:
        assert isinstance(_make_plugin(FakePySRBackend()), ResultBuilder)

    def test_is_result_target_provider(self) -> None:
        assert isinstance(_make_plugin(FakePySRBackend()), ResultTargetProvider)

    def test_has_all_protocol_members(self) -> None:
        plugin = _make_plugin(FakePySRBackend())
        for method in (
            "prepare",
            "propose",
            "evaluate",
            "update",
            "build_final_result",
            "build_result_target",
            "list_plots",
            "render_plot",
            "get_plot_data",
        ):
            assert callable(getattr(plugin, method)), f"missing method {method}"

        assert isinstance(plugin.best_score, float)
        assert isinstance(plugin.best_expression, str)
        assert isinstance(plugin.config, dict)
        assert isinstance(plugin.state, dict)

        assert isinstance(plugin.derivative_requirements, DerivativeReqs)
        assert not callable(plugin.derivative_requirements)







class TestFitMapping:

    @pytest.mark.smoke
    def test_first_propose_triggers_fit(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        assert backend.fit_calls == 1

    def test_first_propose_returns_kd_ir_candidates(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)
        assert isinstance(candidates, list)
        assert len(candidates) >= 1

        for cand in candidates:
            assert isinstance(cand, str) and cand
            to_sympy(cand)

    def test_second_propose_is_idempotent(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        first = plugin.propose(1)
        second = plugin.propose(1)
        assert len(first) >= 1
        assert second == []
        assert backend.fit_calls == 1

    def test_backend_receives_plugin_feature_names(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        assert backend.captured_names is not None
        assert len(backend.captured_names) == len(_TERMS)

        assert set(backend.captured_names).isdisjoint(set(_TERMS))
        for name in backend.captured_names:
            assert name.isidentifier()

    def test_backend_receives_numpy_arrays(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        assert backend.captured_X is not None
        assert backend.captured_y is not None

        assert backend.captured_X.ndim == 2
        assert backend.captured_X.shape[1] == len(_TERMS)
        assert backend.captured_y.ndim == 1
        assert backend.captured_X.shape[0] == backend.captured_y.shape[0]

    def test_evaluate_returns_one_result_per_candidate(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)
        results = plugin.evaluate(candidates)
        assert len(results) == len(candidates)
        for res in results:
            assert isinstance(res, EvaluationResult)

    def test_evaluate_empty_returns_empty(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        assert plugin.evaluate([]) == []

    def test_update_writes_pareto_whitelist_keys(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)
        results = plugin.evaluate(candidates)
        plugin.update(results)
        for key in _PARETO_KEYS:
            assert key in recorder.keys(), f"missing recorder key {key}"

    def test_update_writes_pareto_once(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        results = plugin.evaluate(plugin.propose(1))
        plugin.update(results)

        for key in _PARETO_KEYS:
            assert len(recorder.get(key)) == 1

    def test_pareto_nmse_aligns_with_candidates(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)
        results = plugin.evaluate(candidates)
        plugin.update(results)
        logged_nmse = recorder.get("pareto_nmse")[-1]
        assert len(logged_nmse) == len(results)

    def test_pareto_complexity_aligns_with_loss(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.update(plugin.evaluate(plugin.propose(1)))
        comp = recorder.get("pareto_complexity")[-1]
        loss = recorder.get("pareto_loss")[-1]
        assert len(comp) == len(loss)
        assert len(comp) >= 1

    def test_update_empty_is_noop(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        before = {k: len(recorder.get(k)) for k in _PARETO_KEYS}
        plugin.update([])
        after = {k: len(recorder.get(k)) for k in _PARETO_KEYS}
        assert before == after

    def test_partial_hof_conversion_skips_failures(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        import sympy

        from kd.search.pysr.backend import HOFEntry

        def mixed_hof(names: list[str]) -> list[HOFEntry]:
            good = sympy.Symbol(names[0])
            bad = sympy.sqrt(sympy.Symbol(names[0]))
            return [
                HOFEntry(complexity=1, loss=0.5, sympy_expr=good),
                HOFEntry(complexity=3, loss=0.1, sympy_expr=bad),
            ]

        backend = FakePySRBackend(hof_recipe=mixed_hof)
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)

        assert len(candidates) == 1
        for cand in candidates:
            to_sympy(cand)

    def test_partial_hof_keeps_meta_aligned(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        import sympy

        from kd.search.pysr.backend import HOFEntry

        def mixed_hof(names: list[str]) -> list[HOFEntry]:
            good = sympy.Symbol(names[0])
            bad = sympy.sqrt(sympy.Symbol(names[0]))
            return [
                HOFEntry(complexity=1, loss=0.5, sympy_expr=good),
                HOFEntry(complexity=3, loss=0.1, sympy_expr=bad),
            ]

        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend(hof_recipe=mixed_hof)
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)
        results = plugin.evaluate(candidates)
        plugin.update(results)
        assert len(recorder.get("pareto_complexity")[-1]) == len(candidates)
        assert len(recorder.get("pareto_loss")[-1]) == len(candidates)
        assert len(recorder.get("pareto_nmse")[-1]) == len(candidates)







class TestBestAndScore:

    def test_best_expression_empty_before_fit(self) -> None:
        plugin = _make_plugin(FakePySRBackend())
        assert plugin.best_expression == ""

    def test_best_score_inf_before_fit(self) -> None:
        plugin = _make_plugin(FakePySRBackend())
        assert plugin.best_score == float("inf")

    def test_best_expression_after_fit_is_semantically_correct(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        assert plugin.best_expression
        assert are_equivalent(plugin.best_expression, "add(u, u_x)")

    def test_best_score_is_finite_nmse_after_fit(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        assert math.isfinite(plugin.best_score)
        assert plugin.best_score >= 0.0

    def test_best_score_matches_final_result_nmse(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        final = plugin.build_final_result()
        assert math.isclose(plugin.best_score, final.nmse, rel_tol=1e-9, abs_tol=1e-12)







class TestBestHardFailure:

    def test_unconvertible_best_raises_runtime_error(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend(best_recipe=_sqrt_best_recipe)
        plugin = _prepared_plugin(real_pysr_components, backend)
        with pytest.raises(RuntimeError):
            plugin.propose(1)

    def test_unconvertible_best_does_not_swallow_into_empty(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend(best_recipe=_sqrt_best_recipe)
        plugin = _prepared_plugin(real_pysr_components, backend)
        with pytest.raises(RuntimeError):


            plugin.propose(1)

    def test_pure_constant_best_raises_runtime_error(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend(best_recipe=_pure_constant_best_recipe)
        plugin = _prepared_plugin(real_pysr_components, backend)
        with pytest.raises(RuntimeError):
            plugin.propose(1)







class TestConstantNotPoisoning:

    def test_best_with_trailing_constant_is_valid(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend(best_recipe=_const_tail_best_recipe)
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        best = plugin.best_expression
        assert best

        assert not _contains_unit_term(best)

        assert are_equivalent(best, "add(u, u_x)")

        evaluated = real_pysr_components.evaluator.evaluate_expression(best)
        assert evaluated.is_valid

    def test_best_with_trailing_constant_has_finite_score(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend(best_recipe=_const_tail_best_recipe)
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        assert math.isfinite(plugin.best_score)
        assert plugin.best_score >= 0.0

    def test_pure_constant_hof_entry_is_skipped(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend(hof_recipe=_const_in_hof_recipe)
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)

        assert len(candidates) == 2
        for cand in candidates:
            assert not _contains_unit_term(cand)
            to_sympy(cand)
        recovered = [are_equivalent(c, target) for c in candidates for target in ("u",)]
        assert any(recovered)

    def test_pure_constant_hof_entry_skipped_in_recorder(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend(hof_recipe=_const_in_hof_recipe)
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)
        plugin.update(plugin.evaluate(candidates))

        assert len(recorder.get(_PARETO_COMPLEXITY_KEY)[-1]) == 2
        assert len(recorder.get(_PARETO_LOSS_KEY)[-1]) == len(candidates)
        assert len(recorder.get(_PARETO_NMSE_KEY)[-1]) == len(candidates)








class TestInvalidNmseLoggedAsNone:

    def test_invalid_result_logs_none_in_pareto_nmse(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)
        results = plugin.evaluate(candidates)
        assert len(results) >= 3, "default HOF should yield >=3 parallel candidates"
        invalid_index = 1
        results[invalid_index] = make_invalid_result()

        plugin.update(results)
        logged = recorder.get(_PARETO_NMSE_KEY)[-1]
        assert len(logged) == len(results)
        assert logged[invalid_index] is None, "invalid nmse must log as None"

        for position, value in enumerate(logged):
            if position == invalid_index:
                continue
            assert value is not None
            assert math.isfinite(value)

    def test_no_penalty_sentinel_in_logged_pareto_nmse(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)
        results = plugin.evaluate(candidates)
        results[0] = make_invalid_result()
        plugin.update(results)
        logged = recorder.get(_PARETO_NMSE_KEY)[-1]
        for value in logged:
            assert value != 1e10, "penalty sentinel must not reach the recorder"

    def test_all_valid_results_log_finite_nmse(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        results = plugin.evaluate(plugin.propose(1))
        assert all(r.is_valid for r in results), "default recipe re-scores valid"
        plugin.update(results)
        logged = recorder.get(_PARETO_NMSE_KEY)[-1]
        assert all(value is not None and math.isfinite(value) for value in logged)







class TestSelectedComplexityScale:

    def test_selected_matches_hof_pysr_complexity(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)
        plugin.update(plugin.evaluate(candidates))




        complexities = recorder.get(_PARETO_COMPLEXITY_KEY)[-1]
        best = plugin.best_expression
        matched = [
            comp
            for comp, cand in zip(complexities, candidates, strict=True)
            if are_equivalent(cand, best)
        ]
        assert matched, "best must be equivalent to some HOF candidate"
        expected_pysr_complexity = matched[0]

        selected = recorder.get(_SELECTED_COMPLEXITY_KEY)[-1]
        assert selected == expected_pysr_complexity


        assert selected in complexities

    def test_selected_differs_from_kd_term_count(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.update(plugin.evaluate(plugin.propose(1)))
        final = plugin.build_final_result()
        kd_term_count = final.complexity
        selected = recorder.get(_SELECTED_COMPLEXITY_KEY)[-1]
        assert selected != kd_term_count

    def test_selected_none_when_best_not_in_hof(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        best_recipe, hof_recipe = _best_not_in_hof_recipe_factory()
        backend = FakePySRBackend(best_recipe=best_recipe, hof_recipe=hof_recipe)
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)

        best = plugin.best_expression
        assert not any(are_equivalent(c, best) for c in candidates)
        plugin.update(plugin.evaluate(candidates))
        assert recorder.get(_SELECTED_COMPLEXITY_KEY)[-1] is None

    def test_selected_none_viz_omits_marker(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        best_recipe, hof_recipe = _best_not_in_hof_recipe_factory()
        backend = FakePySRBackend(best_recipe=best_recipe, hof_recipe=hof_recipe)
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.update(plugin.evaluate(plugin.propose(1)))
        data = plugin.get_plot_data("pareto_front")

        assert data.get("selected_x") is None
        assert data.get("selected_y") is None








class TestSelectedPointCoordinates:

    def test_selected_loss_and_nmse_keys_logged(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.update(plugin.evaluate(plugin.propose(1)))
        logged_keys = recorder.keys()
        assert _SELECTED_LOSS_KEY in logged_keys
        assert _SELECTED_NMSE_KEY in logged_keys

    def test_selected_loss_is_matched_entry_loss(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)
        plugin.update(plugin.evaluate(candidates))

        losses = recorder.get(_PARETO_LOSS_KEY)[-1]
        best = plugin.best_expression
        matched_loss = [
            loss
            for loss, cand in zip(losses, candidates, strict=True)
            if are_equivalent(cand, best)
        ]
        assert matched_loss, "best must match some HOF candidate"
        selected_loss = recorder.get(_SELECTED_LOSS_KEY)[-1]
        assert math.isclose(selected_loss, matched_loss[0], rel_tol=1e-9, abs_tol=1e-12)

    def test_selected_nmse_is_best_kd_nmse(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.update(plugin.evaluate(plugin.propose(1)))
        selected_nmse = recorder.get(_SELECTED_NMSE_KEY)[-1]
        assert selected_nmse is not None
        assert math.isclose(
            selected_nmse, plugin.best_score, rel_tol=1e-9, abs_tol=1e-12
        )

    def test_selected_loss_picks_second_of_two_same_complexity(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        best_recipe, hof_recipe = _duplicate_complexity_recipe_factory()
        backend = FakePySRBackend(best_recipe=best_recipe, hof_recipe=hof_recipe)
        plugin = _prepared_plugin(real_pysr_components, backend)
        candidates = plugin.propose(1)
        plugin.update(plugin.evaluate(candidates))

        losses = recorder.get(_PARETO_LOSS_KEY)[-1]
        best = plugin.best_expression
        matched = [
            (loss, cand)
            for loss, cand in zip(losses, candidates, strict=True)
            if are_equivalent(cand, best)
        ]
        assert matched, "best must match one of the two same-complexity rows"
        expected_loss = matched[0][0]


        other = [loss for loss in losses if loss != expected_loss]
        assert other, "the two rows must differ in loss for this to discriminate"
        selected_loss = recorder.get(_SELECTED_LOSS_KEY)[-1]
        assert math.isclose(selected_loss, expected_loss, rel_tol=1e-9, abs_tol=1e-12)

    def test_selected_coordinates_none_when_best_not_in_hof(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        best_recipe, hof_recipe = _best_not_in_hof_recipe_factory()
        backend = FakePySRBackend(best_recipe=best_recipe, hof_recipe=hof_recipe)
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.update(plugin.evaluate(plugin.propose(1)))
        assert recorder.get(_SELECTED_COMPLEXITY_KEY)[-1] is None
        assert recorder.get(_SELECTED_LOSS_KEY)[-1] is None
        assert recorder.get(_SELECTED_NMSE_KEY)[-1] is None







class TestUpdateIdempotency:

    def test_double_update_logs_pareto_once(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        results = plugin.evaluate(plugin.propose(1))
        plugin.update(results)
        plugin.update(results)
        for key in _PARETO_KEYS:
            assert len(recorder.get(key)) == 1, f"{key} logged more than once"

    def test_triple_update_still_logs_once(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        results = plugin.evaluate(plugin.propose(1))
        for _ in range(3):
            plugin.update(results)
        for key in _PARETO_KEYS:
            assert len(recorder.get(key)) == 1

    def test_guard_resets_on_fresh_prepare(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        recorder = real_pysr_components.recorder
        assert recorder is not None
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        results = plugin.evaluate(plugin.propose(1))
        plugin.update(results)
        plugin.update(results)
        assert len(recorder.get(_PARETO_COMPLEXITY_KEY)) == 1



        plugin.prepare(real_pysr_components)
        results2 = plugin.evaluate(plugin.propose(1))
        plugin.update(results2)
        plugin.update(results2)
        assert len(recorder.get(_PARETO_COMPLEXITY_KEY)) == 2







class TestTermsProperty:

    def test_terms_none_before_fit(self) -> None:
        plugin = _make_plugin(FakePySRBackend())
        assert plugin.terms is None

    def test_terms_after_fit_returns_valid_terms(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        assert plugin.terms == list(_TERMS)

    def test_terms_returns_copy(self, real_pysr_components: PlatformComponents) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        terms = plugin.terms
        assert terms is not None
        terms.append("tampered")
        assert plugin.terms == list(_TERMS)







class TestDerivativeRequirements:

    def test_order_two_from_uxx(self) -> None:
        plugin = _make_plugin(FakePySRBackend(), terms=("u", "u_x", "u_xx"))
        reqs = plugin.derivative_requirements
        assert reqs.max_atomic_order == 2

    def test_order_three_from_uxxx(self) -> None:
        plugin = _make_plugin(FakePySRBackend(), terms=("u", "u_xxx"))
        reqs = plugin.derivative_requirements
        assert reqs.max_atomic_order == 3

    def test_order_floor_one_for_no_derivatives(self) -> None:
        plugin = _make_plugin(FakePySRBackend(), terms=("u",))
        reqs = plugin.derivative_requirements
        assert reqs.max_atomic_order == 1

    def test_requirements_shape(self) -> None:
        plugin = _make_plugin(FakePySRBackend(), terms=_TERMS)
        reqs = plugin.derivative_requirements
        assert reqs.provider_kind == "finite_diff"
        assert reqs.lhs_order == 1
        assert reqs.needs_surrogate is False

    def test_requirements_resolves_via_builder_helper(self) -> None:
        plugin = _make_plugin(FakePySRBackend(), terms=_TERMS)
        resolved = _resolve_derivative_requirements(plugin)
        assert isinstance(resolved, DerivativeReqs)
        assert resolved.max_atomic_order == 2







class TestResultBuilderTarget:

    def test_build_final_result_returns_evaluation_result(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        result = plugin.build_final_result()
        assert isinstance(result, EvaluationResult)
        assert result.is_valid

    def test_build_final_result_matches_best_expression(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        result = plugin.build_final_result()
        assert are_equivalent(result.expression, plugin.best_expression)

    def test_build_result_target_returns_tensor(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        target = plugin.build_result_target()
        assert isinstance(target, Tensor)

    def test_build_result_target_value_parity(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        target = plugin.build_result_target()
        expected = real_pysr_components.evaluator.lhs_target
        torch.testing.assert_close(target, expected, rtol=1e-7, atol=1e-9)

    def test_build_result_target_is_independent_storage(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        target = plugin.build_result_target()
        baseline = real_pysr_components.evaluator.lhs_target.clone()
        target += 1234.0
        after = real_pysr_components.evaluator.lhs_target
        torch.testing.assert_close(after, baseline, rtol=1e-7, atol=1e-9)

    def test_build_result_target_no_grad(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        target = plugin.build_result_target()
        assert target.requires_grad is False
        assert target.grad_fn is None







class TestStateRoundTrip:

    def test_state_is_pickle_safe(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        import pickle

        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        blob = pickle.dumps(plugin.state)
        restored = pickle.loads(blob)
        assert restored["algorithm"] == "pysr"

    def test_state_roundtrip_preserves_best(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _prepared_plugin(real_pysr_components, backend)
        plugin.propose(1)
        saved = plugin.state

        fresh = _make_plugin(FakePySRBackend())
        fresh.state = saved
        assert are_equivalent(fresh.best_expression, plugin.best_expression)
        assert math.isclose(
            fresh.best_score, plugin.best_score, rel_tol=1e-9, abs_tol=1e-12
        )

    def test_state_reports_fitted_flag(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _make_plugin(backend)
        plugin.prepare(real_pysr_components)
        assert plugin.state["fitted"] is False
        plugin.propose(1)
        assert plugin.state["fitted"] is True

    def test_restore_then_prepare_does_not_refit(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        donor = _prepared_plugin(real_pysr_components, backend)
        donor.propose(1)
        saved = donor.state

        fresh_backend = FakePySRBackend()
        fresh = _make_plugin(fresh_backend)
        fresh.state = saved
        fresh.prepare(real_pysr_components)

        assert fresh.propose(1) == []
        assert fresh_backend.fit_calls == 0

    def test_fresh_prepare_resets_fit_state(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _make_plugin(backend)
        plugin.prepare(real_pysr_components)
        plugin.propose(1)
        assert backend.fit_calls == 1

        plugin.prepare(real_pysr_components)
        assert plugin.propose(1) != []
        assert backend.fit_calls == 2







class TestConfig:

    def test_config_is_json_safe(self) -> None:
        plugin = _make_plugin(FakePySRBackend())
        json.dumps(plugin.config)

    def test_config_tags_algorithm(self) -> None:
        plugin = _make_plugin(FakePySRBackend())
        assert plugin.config["algorithm"] == "pysr"

    def test_config_includes_terms(self) -> None:
        plugin = _make_plugin(FakePySRBackend(), terms=_TERMS)
        assert "terms" in plugin.config







class TestLifecycleGuards:

    def test_propose_before_prepare_raises(self) -> None:
        plugin = _make_plugin(FakePySRBackend())
        with pytest.raises(RuntimeError):
            plugin.propose(1)

    def test_evaluate_before_prepare_raises(self) -> None:
        plugin = _make_plugin(FakePySRBackend())
        with pytest.raises(RuntimeError):
            plugin.evaluate(["u"])

    def test_build_final_result_before_fit_is_safe_or_raises(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        backend = FakePySRBackend()
        plugin = _make_plugin(backend)
        plugin.prepare(real_pysr_components)
        try:
            result = plugin.build_final_result()
        except (RuntimeError, ValueError):
            return
        assert isinstance(result, EvaluationResult)
        assert result.is_valid is False
