
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn

from kd.core.evaluator import EvaluationResult
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder

if TYPE_CHECKING:
    from collections.abc import Sequence






EXPECTED_METRICS: frozenset[str] = frozenset(
    {
        "gen_best_fitness",
        "gen_mean_fitness",
        "gen_best_nmse",
        "n_valid",
        "n_unique",
        "gen_mean_complexity",
        "lhs_ut",
        "lhs_utt",
    }
)




_PLATFORM_KEYS: frozenset[str] = frozenset(
    {"_best_score", "_best_expr", "_n_candidates"}
)
_SGA_KEYS: frozenset[str] = frozenset({"best_aic"})








def _valid_result(
    *,
    expression: str,
    score: float,
    nmse: float,
    complexity: int,
    lhs_name: str,
) -> EvaluationResult:
    return EvaluationResult(
        mse=nmse,
        nmse=nmse,
        r2=max(0.0, 1.0 - nmse),
        score=score,
        complexity=complexity,
        coefficients=torch.ones(complexity, dtype=torch.float64),
        is_valid=True,
        selected_indices=list(range(complexity)),
        residuals=None,
        terms=[f"t{i}" for i in range(complexity)],
        expression=expression,
        lhs_name=lhs_name,
    )


def _invalid_result(*, expression: str, lhs_name: str = "u_t") -> EvaluationResult:
    return EvaluationResult(
        mse=float("inf"),
        nmse=float("inf"),
        r2=-float("inf"),
        score=float("inf"),
        complexity=0,
        coefficients=None,
        is_valid=False,
        error_message="forced-invalid",
        selected_indices=[],
        residuals=None,
        terms=[],
        expression=expression,
        lhs_name=lhs_name,
    )


def _distinct_batch() -> list[EvaluationResult]:
    return [
        _valid_result(
            expression="A", score=12.0, nmse=0.30, complexity=5, lhs_name="u_t"
        ),
        _valid_result(
            expression="B", score=15.0, nmse=0.40, complexity=7, lhs_name="u_t"
        ),
        _valid_result(
            expression="A", score=18.0, nmse=0.55, complexity=9, lhs_name="u_tt"
        ),
        _invalid_result(expression="C", lhs_name="u_t"),
        _invalid_result(expression="D", lhs_name="u_tt"),
    ]




DISTINCT_EXPECTED: dict[str, float] = {
    "gen_best_fitness": 12.0,
    "gen_mean_fitness": 15.0,
    "gen_best_nmse": 0.30,
    "n_valid": 3,
    "n_unique": 4,
    "gen_mean_complexity": 7.0,
    "lhs_ut": 2,
    "lhs_utt": 1,
}


def _all_invalid_batch() -> list[EvaluationResult]:
    return [
        _invalid_result(expression="X", lhs_name="u_t"),
        _invalid_result(expression="Y", lhs_name="u_tt"),
        _invalid_result(expression="Z", lhs_name="u_t"),
    ]













def _plugin_with_recorder(recorder: VizRecorder | None) -> DLGAPlugin:
    plugin = DLGAPlugin(DLGAConfig(pop_size=4))
    plugin._recorder = recorder
    return plugin


@pytest.fixture
def recorder() -> VizRecorder:
    return VizRecorder(enabled=True)


@pytest.fixture
def plugin(recorder: VizRecorder) -> DLGAPlugin:
    return _plugin_with_recorder(recorder)


def _logged_series_lengths(
    recorder: VizRecorder, names: Sequence[str]
) -> dict[str, int]:
    return {name: len(recorder.get(name)) for name in names}


def _non_underscore_keys(recorder: VizRecorder) -> set[str]:
    keys = recorder.keys()
    return {name for name in keys if not name.startswith("_")}







@pytest.mark.unit
def test_logged_metrics_constant_equals_whitelist() -> None:
    from kd.search.dlga import plugin as dlga_plugin

    logged = getattr(dlga_plugin, "_LOGGED_METRICS", None)
    assert logged is not None, "dlga/plugin.py must define _LOGGED_METRICS"
    assert isinstance(logged, tuple), (
        f"_LOGGED_METRICS must be a tuple[str, ...], got {type(logged).__name__}"
    )
    assert len(logged) == len(set(logged)), (
        f"_LOGGED_METRICS has duplicate entries: {logged}"
    )
    assert set(logged) == EXPECTED_METRICS, (
        f"_LOGGED_METRICS must equal the 8-field whitelist.\n"
        f" missing: {sorted(EXPECTED_METRICS - set(logged))}\n"
        f" extra: {sorted(set(logged) - EXPECTED_METRICS)}"
    )


@pytest.mark.unit
def test_update_logs_exactly_the_whitelist(
    plugin: DLGAPlugin,
    recorder: VizRecorder,
) -> None:
    plugin.update(_distinct_batch())

    logged = _non_underscore_keys(recorder)
    assert logged == EXPECTED_METRICS, (
        f"update() must log exactly the 8-field whitelist (no more, no less).\n"
        f" missing: {sorted(EXPECTED_METRICS - logged)}\n"
        f" extra: {sorted(logged - EXPECTED_METRICS)}\n"
        f" got: {sorted(logged)}"
    )


@pytest.mark.unit
def test_metric_names_no_underscore_prefix(
    plugin: DLGAPlugin,
    recorder: VizRecorder,
) -> None:
    for name in EXPECTED_METRICS:
        assert not name.startswith("_"), (
            f"Whitelist entry {name!r} must not start with '_' (-D14)."
        )

    plugin.update(_distinct_batch())
    live_keys = recorder.keys()
    assert live_keys, (
        "update() logged nothing; expected the 8 plugin-written metric keys "
        "(the '_' prefix check is vacuous on an empty recorder)."
    )
    for name in live_keys:
        assert not name.startswith("_"), (
            f"Live recorder key {name!r} has a reserved '_' prefix (-D14)."
        )


@pytest.mark.unit
def test_whitelist_namespace_disjoint() -> None:
    overlap_platform = EXPECTED_METRICS & _PLATFORM_KEYS
    assert not overlap_platform, (
        f"DLGA whitelist collides with platform keys: {sorted(overlap_platform)}."
    )
    overlap_sga = EXPECTED_METRICS & _SGA_KEYS
    assert not overlap_sga, (
        f"DLGA whitelist collides with SGA keys: {sorted(overlap_sga)}."
    )







@pytest.mark.unit
@pytest.mark.parametrize("field", sorted(DISTINCT_EXPECTED))
def test_each_metric_value_is_correct(
    plugin: DLGAPlugin,
    recorder: VizRecorder,
    field: str,
) -> None:
    plugin.update(_distinct_batch())

    series = recorder.get(field)
    assert len(series) == 1, (
        f"recorder.get({field!r}) has {len(series)} entries after one "
        f"update(); expected exactly 1."
    )
    recorded = series[-1]
    expected = DISTINCT_EXPECTED[field]

    if field in {"n_valid", "n_unique", "lhs_ut", "lhs_utt"}:





        assert type(recorded) is int, (
            f"{field}: recorded value must be a built-in int, got "
            f"{type(recorded).__name__} ({recorded!r}). A numpy/tensor scalar "
            f"breaks VizRecorder JSON serialization (string fallback)."
        )
        assert recorded == expected, (
            f"{field}: recorded {recorded!r}, expected exactly {expected!r}. "
            f"(A count landing in another count's value indicates a "
            f"swap-by-key bug.)"
        )
    else:
        assert recorded == pytest.approx(expected, rel=1e-9, abs=1e-12), (
            f"{field}: recorded {recorded!r}, expected {expected!r}. "
            f"Bands are disjoint (fitness [10,20) / nmse [0,1) / "
            f"complexity [5,10)); a value in the wrong band means the plugin "
            f"logged the wrong source under this key."
        )


@pytest.mark.unit
def test_best_nmse_tracks_best_fitness_individual(
    plugin: DLGAPlugin,
    recorder: VizRecorder,
) -> None:
    batch = [
        _valid_result(
            expression="M0", score=10.0, nmse=0.50, complexity=3, lhs_name="u_t"
        ),
        _valid_result(
            expression="M1", score=20.0, nmse=0.10, complexity=3, lhs_name="u_tt"
        ),
        _valid_result(
            expression="M2", score=15.0, nmse=0.30, complexity=3, lhs_name="u_t"
        ),
        _invalid_result(expression="M3", lhs_name="u_tt"),
        _invalid_result(expression="M4", lhs_name="u_t"),
    ]
    plugin.update(batch)

    best_nmse = recorder.get("gen_best_nmse")[-1]
    assert best_nmse == pytest.approx(0.50, rel=1e-9), (
        f"gen_best_nmse must be the nmse of the argmin-fitness VALID "
        f"individual (m0, nmse=0.50), not the global-min nmse (m1, 0.10) "
        f"and not min over all/valid. Got {best_nmse!r}."
    )
    best_fitness = recorder.get("gen_best_fitness")[-1]
    assert best_fitness == pytest.approx(10.0, rel=1e-9), (
        f"gen_best_fitness must be 10.0 (argmin over valid aic). Got {best_fitness!r}."
    )


@pytest.mark.unit
def test_mean_fitness_passes_inf_through_from_valid_guarded_individual() -> None:
    recorder = VizRecorder(enabled=True)
    plugin = _plugin_with_recorder(recorder)
    batch = [
        _valid_result(
            expression="A", score=12.0, nmse=0.30, complexity=4, lhs_name="u_t"
        ),
        _valid_result(
            expression="B", score=15.0, nmse=0.40, complexity=6, lhs_name="u_t"
        ),

        _valid_result(
            expression="C", score=float("inf"), nmse=0.55, complexity=8, lhs_name="u_tt"
        ),
        _invalid_result(expression="D", lhs_name="u_t"),
    ]
    plugin.update(batch)

    mean_fitness = recorder.get("gen_mean_fitness")[-1]
    assert mean_fitness == float("inf"), (
        f"gen_mean_fitness must propagate the +inf of a valid-but-guarded "
        f"individual (no secondary finite-only filter); expected +inf, got "
        f"{mean_fitness!r}. A finite value here means the impl wrongly "
        f"dropped the cross-LHS-guard inf from the mean."
    )

    assert recorder.get("n_valid")[-1] == 3, (
        f"n_valid must count is_valid==True (the guarded inf individual is "
        f"still valid) -> 3; got {recorder.get('n_valid')[-1]!r}."
    )

    best_fitness = recorder.get("gen_best_fitness")[-1]
    assert best_fitness == pytest.approx(12.0, rel=1e-9), (
        f"gen_best_fitness must be 12.0 (min ignores the +inf). Got {best_fitness!r}."
    )
    best_nmse = recorder.get("gen_best_nmse")[-1]
    assert best_nmse == pytest.approx(0.30, rel=1e-9), (
        f"gen_best_nmse must be 0.30 (nmse of the score=12 best individual), "
        f"not the guarded individual's 0.55. Got {best_nmse!r}."
    )


@pytest.mark.unit
def test_all_valid_guarded_logs_inf_nmse_distinct_from_no_valid() -> None:
    recorder = VizRecorder(enabled=True)
    plugin = _plugin_with_recorder(recorder)
    batch = [
        _valid_result(
            expression="A",
            score=float("inf"),
            nmse=float("inf"),
            complexity=3,
            lhs_name="u_t",
        ),
        _valid_result(
            expression="B",
            score=float("inf"),
            nmse=float("inf"),
            complexity=5,
            lhs_name="u_tt",
        ),
    ]
    plugin.update(batch)

    assert recorder.get("gen_best_nmse")[-1] == float("inf"), (
        "All-guarded generation: the best individual's nmse is +inf, so "
        f"gen_best_nmse must be +inf. Got {recorder.get('gen_best_nmse')[-1]!r}."
    )
    assert recorder.get("gen_best_fitness")[-1] == float("inf")
    assert recorder.get("gen_mean_fitness")[-1] == float("inf")
    assert recorder.get("n_valid")[-1] == 2, (
        "n_valid must stay > 0 (both individuals are is_valid=True despite "
        "the guard) — the ONLY signal distinguishing an all-guarded "
        f"generation from a no-valid one. Got {recorder.get('n_valid')[-1]!r}."
    )


@pytest.mark.unit
def test_logging_metrics_does_not_regress_best_tracking(
    plugin: DLGAPlugin,
    recorder: VizRecorder,
) -> None:
    plugin.update(_distinct_batch())

    assert plugin.best_score == pytest.approx(12.0, rel=1e-9), (
        f"best_score must remain the min valid _fitness (12.0) after update(); "
        f"logging metrics must not perturb best-tracking. Got {plugin.best_score!r}."
    )
    assert plugin.best_expression == "A", (
        f"best_expression must be 'A' (the score=12.0 individual). Got "
        f"{plugin.best_expression!r}."
    )







@pytest.mark.unit
def test_all_invalid_generation_still_logs_all_series(
    plugin: DLGAPlugin,
    recorder: VizRecorder,
) -> None:
    plugin.update(_all_invalid_batch())

    lengths = _logged_series_lengths(recorder, sorted(EXPECTED_METRICS))
    for name, length in lengths.items():
        assert length == 1, (
            f"series {name!r} length {length} after a zero-valid update(); "
            f"expected 1. The plugin must NOT skip logging when n_valid==0."
        )

    assert recorder.get("n_valid")[-1] == 0
    assert recorder.get("lhs_ut")[-1] == 0
    assert recorder.get("lhs_utt")[-1] == 0



    assert recorder.get("n_unique")[-1] == 3, (
        f"n_unique must count distinct expressions over ALL results "
        f"({{X,Y,Z}} = 3) even when n_valid==0; got "
        f"{recorder.get('n_unique')[-1]!r}."
    )

    best_fitness = recorder.get("gen_best_fitness")[-1]
    mean_fitness = recorder.get("gen_mean_fitness")[-1]
    best_nmse = recorder.get("gen_best_nmse")[-1]


    assert best_fitness == float("inf"), (
        f"gen_best_fitness must be +inf when no valid individual (and NOT "
        f"-inf — that would invert the fitness plot). Got {best_fitness!r}."
    )
    assert mean_fitness == float("inf"), (
        f"gen_mean_fitness must be +inf when no valid individual (and NOT "
        f"-inf). Got {mean_fitness!r}."
    )
    assert best_nmse == float("inf"), (
        f"gen_best_nmse must be +inf when no valid individual — there is no "
        f"best individual to read nmse from, so the fitness-matching +inf "
        f"sentinel is required (NOT NaN / 0.0 / None / -inf). Got {best_nmse!r}."
    )

    mean_complexity = recorder.get("gen_mean_complexity")[-1]
    assert isinstance(mean_complexity, float) and math.isnan(mean_complexity), (
        f"gen_mean_complexity is LOCKED to NaN (not-measured, audit m4) for "
        f"the no-valid case; got {mean_complexity!r}."
    )


@pytest.mark.unit
def test_series_length_equals_update_count_across_generations(
    plugin: DLGAPlugin,
    recorder: VizRecorder,
) -> None:
    generations = [_distinct_batch(), _all_invalid_batch(), _distinct_batch()]
    for batch in generations:
        plugin.update(batch)

    n = len(generations)
    for name in sorted(EXPECTED_METRICS):
        series = recorder.get(name)
        assert isinstance(series, list), (
            f"recorder.get({name!r}) must be a list, got {type(series).__name__}"
        )
        assert len(series) == n, (
            f"After {n} update() calls, series {name!r} has length "
            f"{len(series)}; expected {n} (one append per generation, "
            f"including the zero-valid one)."
        )







@pytest.mark.unit
def test_n_unique_counts_distinct_expressions_over_all_results(
    plugin: DLGAPlugin,
    recorder: VizRecorder,
) -> None:
    batch = [
        _valid_result(
            expression="A", score=1.0, nmse=0.1, complexity=2, lhs_name="u_t"
        ),
        _valid_result(
            expression="A", score=2.0, nmse=0.2, complexity=2, lhs_name="u_t"
        ),
        _valid_result(
            expression="B", score=3.0, nmse=0.3, complexity=2, lhs_name="u_t"
        ),
        _invalid_result(expression="B"),
        _invalid_result(expression="C"),
        _invalid_result(expression="D"),
    ]
    plugin.update(batch)

    n_unique = recorder.get("n_unique")[-1]
    assert n_unique == 4, (
        f"n_unique must count distinct expressions across ALL results "
        f"({{A,B,C,D}} = 4), not total results (6) nor valid count (3). "
        f"Got {n_unique!r}."
    )
    n_valid = recorder.get("n_valid")[-1]
    assert n_valid == 3, (
        f"n_valid must be 3 (A,A,B valid); got {n_valid!r}. (Sanity: confirms "
        f"n_unique != n_valid so they cannot be silently swapped.)"
    )


@pytest.mark.unit
def test_n_unique_folds_repeated_empty_string_expressions(
    plugin: DLGAPlugin,
    recorder: VizRecorder,
) -> None:
    batch = [
        _valid_result(
            expression="A", score=1.0, nmse=0.1, complexity=2, lhs_name="u_t"
        ),
        _invalid_result(expression=""),
        _invalid_result(expression=""),
        _invalid_result(expression=""),
    ]
    plugin.update(batch)

    n_unique = recorder.get("n_unique")[-1]
    assert n_unique == 2, (
        f"n_unique must fold repeated empty-string expressions into one "
        f'distinct value ({{"A", ""}} = 2), not count each blank separately '
        f"(would be 4). Got {n_unique!r}."
    )







@pytest.mark.unit
def test_lhs_counts_mixed_branches(
    plugin: DLGAPlugin,
    recorder: VizRecorder,
) -> None:
    batch = [
        _valid_result(
            expression="A", score=1.0, nmse=0.1, complexity=2, lhs_name="u_t"
        ),
        _valid_result(
            expression="B", score=2.0, nmse=0.2, complexity=2, lhs_name="u_t"
        ),
        _valid_result(
            expression="C", score=3.0, nmse=0.3, complexity=2, lhs_name="u_tt"
        ),
        _invalid_result(expression="D", lhs_name="u_t"),
        _invalid_result(expression="E", lhs_name="u_tt"),
    ]
    plugin.update(batch)

    assert recorder.get("lhs_ut")[-1] == 2, (
        "lhs_ut must count only VALID u_t individuals (2); invalid u_t "
        "must be excluded."
    )
    assert recorder.get("lhs_utt")[-1] == 1, (
        "lhs_utt must count only VALID u_tt individuals (1); invalid u_tt "
        "must be excluded."
    )


@pytest.mark.unit
def test_lhs_counts_single_branch_utt_zero(
    plugin: DLGAPlugin,
    recorder: VizRecorder,
) -> None:
    batch = [
        _valid_result(
            expression="A", score=1.0, nmse=0.1, complexity=2, lhs_name="u_t"
        ),
        _valid_result(
            expression="B", score=2.0, nmse=0.2, complexity=2, lhs_name="u_t"
        ),
    ]
    plugin.update(batch)

    assert recorder.get("lhs_ut")[-1] == 2
    assert recorder.get("lhs_utt")[-1] == 0, (
        "lhs_utt must be logged as 0 (a real append), not absent, when no "
        "u_tt individual is present."
    )

    assert "lhs_utt" in recorder.keys(), (
        "lhs_utt series must be logged even when 0."
    )







@pytest.mark.unit
def test_update_with_recorder_none_does_not_crash() -> None:
    plugin = _plugin_with_recorder(None)

    plugin.update(_distinct_batch())
    plugin.update(_all_invalid_batch())


@pytest.mark.unit
def test_update_with_disabled_recorder_logs_nothing() -> None:
    disabled = VizRecorder(enabled=False)
    plugin = _plugin_with_recorder(disabled)

    plugin.update(_distinct_batch())

    assert disabled.keys() == set(), (
        f"A disabled recorder must stay empty after update(); got "
        f"{sorted(disabled.keys())}."
    )
    for name in EXPECTED_METRICS:
        assert disabled.get(name) == [], (
            f"Disabled recorder series {name!r} must be empty, got "
            f"{disabled.get(name)!r}."
        )


@pytest.mark.unit
def test_empty_results_logs_full_whitelist_with_sentinels(
    plugin: DLGAPlugin, recorder: VizRecorder
) -> None:
    plugin.update([])

    keys = _non_underscore_keys(recorder)
    assert keys == EXPECTED_METRICS, (
        f"Empty-results update() must log the FULL 8-field whitelist (an "
        f"empty batch is still a generation; dropping it would contradict "
        f"the all-invalid logging contract).\n"
        f" missing: {sorted(EXPECTED_METRICS - keys)}\n"
        f" extra: {sorted(keys - EXPECTED_METRICS)}"
    )


    for name in ("n_valid", "n_unique", "lhs_ut", "lhs_utt"):
        assert recorder.get(name)[-1] == 0, (
            f"{name} must be 0 for an empty batch; got {recorder.get(name)[-1]!r}."
        )


    for name in ("gen_best_fitness", "gen_mean_fitness", "gen_best_nmse"):
        value = recorder.get(name)[-1]
        assert value == float("inf"), (
            f"{name} must be +inf for an empty batch (no valid individual); "
            f"got {value!r}."
        )

    mean_complexity = recorder.get("gen_mean_complexity")[-1]
    assert isinstance(mean_complexity, float) and math.isnan(mean_complexity), (
        f"gen_mean_complexity must be NaN (not-measured) for an empty batch; "
        f"got {mean_complexity!r}."
    )







@pytest.mark.unit
def test_recorder_is_none_before_prepare() -> None:
    plugin = DLGAPlugin()
    captured = getattr(plugin, "_recorder", "MISSING_ATTR")
    assert captured is None, (
        f"DLGAPlugin().__init__ must set self._recorder = None (attribute "
        f"must exist with value None before prepare()). Got: {captured!r}"
    )


class _ExactQuadraticModel(nn.Module):

    def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"u": 1.0 + x * x + t * t}


def _real_components_with_recorder(
    plugin: DLGAPlugin, recorder: VizRecorder | None
) -> PlatformComponents:
    from kd.core.platform.builder import PlatformBuilder

    x = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = 1.0 + xg * xg + tg * tg
    dataset = PDEDataset(
        name="dlga-recorder-test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="t",
    )
    components = PlatformBuilder(dataset, plugin.derivative_requirements).build()
    components.recorder = recorder
    return components


@pytest.mark.unit
def test_real_prepare_binds_recorder() -> None:
    recorder = VizRecorder(enabled=True)
    plugin = DLGAPlugin(
        DLGAConfig(pop_size=4, seed=7),
        surrogate_model=_ExactQuadraticModel(),
    )
    components = _real_components_with_recorder(plugin, recorder)

    plugin.prepare(components)

    captured = getattr(plugin, "_recorder", "MISSING_ATTR")
    assert captured is recorder, (
        "DLGAPlugin.prepare() must capture components.recorder into "
        "self._recorder (mirror SGAPlugin.prepare line 233)."
    )


@pytest.mark.unit
def test_real_prepare_then_update_logs_into_bound_recorder() -> None:
    recorder = VizRecorder(enabled=True)
    plugin = DLGAPlugin(
        DLGAConfig(pop_size=4, seed=7),
        surrogate_model=_ExactQuadraticModel(),
    )
    components = _real_components_with_recorder(plugin, recorder)
    plugin.prepare(components)

    plugin.update(_distinct_batch())

    logged = _non_underscore_keys(recorder)
    assert logged == EXPECTED_METRICS, (
        f"After real prepare()+update(), recorder must hold the 8-field "
        f"whitelist.\n missing: {sorted(EXPECTED_METRICS - logged)}\n"
        f" extra: {sorted(logged - EXPECTED_METRICS)}"
    )
