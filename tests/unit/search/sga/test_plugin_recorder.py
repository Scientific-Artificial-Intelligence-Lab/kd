
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder
from kd.search.sga.config import SGAConfig
from kd.search.sga.plugin import SGAPlugin

if TYPE_CHECKING:
    from collections.abc import Sequence







EXPECTED_METRICS_ORDERED: tuple[str, ...] = (
    "gen_best_aic",
    "gen_mean_aic",
    "gen_best_nmse",
    "n_valid",
    "n_unique",
    "gen_mean_complexity",
)
EXPECTED_METRICS: frozenset[str] = frozenset(EXPECTED_METRICS_ORDERED)




_PLATFORM_KEYS: frozenset[str] = frozenset(
    {"_best_score", "_best_expr", "_n_candidates"}
)
_LEGACY_SGA_KEY = "best_aic"


_SMALL_GRID_SIZE = 10
_SMALL_TIME_SIZE = 5









def _valid_result(
    *,
    expression: str,
    aic: float,
    nmse: float,
    complexity: int,
) -> EvaluationResult:
    return EvaluationResult(
        mse=nmse,
        nmse=nmse,
        r2=max(0.0, 1.0 - nmse),
        aic=aic,
        complexity=complexity,
        coefficients=torch.ones(complexity, dtype=torch.float64),
        is_valid=True,
        selected_indices=list(range(complexity)),
        residuals=None,
        terms=[f"t{i}" for i in range(complexity)],
        expression=expression,
    )


def _invalid_result(*, expression: str) -> EvaluationResult:
    return EvaluationResult(
        mse=float("inf"),
        nmse=float("inf"),
        r2=-float("inf"),
        aic=float("inf"),
        complexity=0,
        coefficients=None,
        is_valid=False,
        error_message="forced-invalid",




        selected_indices=None,
        residuals=None,
        terms=[],
        expression=expression,
    )


def _distinct_batch() -> list[EvaluationResult]:
    return [
        _valid_result(expression="A", aic=12.0, nmse=0.30, complexity=5),
        _valid_result(expression="B", aic=15.0, nmse=0.10, complexity=7),
        _valid_result(expression="A", aic=18.0, nmse=0.55, complexity=9),
        _invalid_result(expression="C"),
        _invalid_result(expression="D"),
    ]




DISTINCT_EXPECTED: dict[str, float] = {
    "gen_best_aic": 12.0,
    "gen_mean_aic": 15.0,
    "gen_best_nmse": 0.30,
    "n_valid": 3,
    "n_unique": 4,
    "gen_mean_complexity": 7.0,
}

_COUNT_FIELDS: frozenset[str] = frozenset({"n_valid", "n_unique"})


def _all_invalid_batch() -> list[EvaluationResult]:
    return [
        _invalid_result(expression="X"),
        _invalid_result(expression="Y"),
        _invalid_result(expression="Z"),
    ]















def _seed_pending(plugin: SGAPlugin, batch: Sequence[EvaluationResult]) -> None:
    plugin._offspring_results = list(batch)
    plugin._pending_population = []
    plugin._pending_scores = []


def _plugin_with_recorder(recorder: VizRecorder | None) -> SGAPlugin:
    plugin = SGAPlugin(SGAConfig(num=4))
    plugin._recorder = recorder
    return plugin


def _commit_generation(plugin: SGAPlugin, batch: Sequence[EvaluationResult]) -> None:
    _seed_pending(plugin, batch)
    plugin._commit_pending_generation()


@pytest.fixture
def recorder() -> VizRecorder:
    return VizRecorder(enabled=True)


@pytest.fixture
def plugin(recorder: VizRecorder) -> SGAPlugin:
    return _plugin_with_recorder(recorder)


def _logged_series_lengths(
    recorder: VizRecorder, names: Sequence[str]
) -> dict[str, int]:
    return {name: len(recorder.get(name)) for name in names}


def _last(recorder: VizRecorder, key: str) -> object:
    series = recorder.get(key)
    assert series, (
        f"recorder series {key!r} is empty — the metric was never logged "
        f"(unimplemented in src/). Expected exactly one entry after a commit."
    )
    return series[-1]


def _non_underscore_keys(recorder: VizRecorder) -> set[str]:
    keys = recorder.keys()
    return {name for name in keys if not name.startswith("_")}







@pytest.mark.unit
def test_logged_metrics_constant_strict_equals_whitelist() -> None:
    from kd.search.sga import plugin as sga_plugin

    logged = getattr(sga_plugin, "_LOGGED_METRICS", None)
    assert logged is not None, "sga/plugin.py must define _LOGGED_METRICS"
    assert isinstance(logged, tuple), (
        f"_LOGGED_METRICS must be a tuple[str, ...], got {type(logged).__name__}"
    )
    assert len(logged) == len(set(logged)), (
        f"_LOGGED_METRICS has duplicate entries: {logged}"
    )


    assert logged == EXPECTED_METRICS_ORDERED, (
        f"_LOGGED_METRICS must equal the 6-field whitelist exactly (order "
        f"matters).\n expected: {EXPECTED_METRICS_ORDERED}\n got: {logged}"
    )


@pytest.mark.unit
def test_commit_logs_exactly_the_whitelist_plus_best_aic(
    plugin: SGAPlugin,
    recorder: VizRecorder,
) -> None:
    from kd.search.sga.plugin import _SURROGATE_METRICS

    _commit_generation(plugin, _distinct_batch())

    logged = _non_underscore_keys(recorder)
    per_gen_expected = EXPECTED_METRICS | {_LEGACY_SGA_KEY}







    assert logged == per_gen_expected, (
        f"a commit with no surrogate training must log exactly the 6-field "
        f"whitelist + the legacy {_LEGACY_SGA_KEY!r} series (surrogate keys "
        f"are logged in prepare, not per-generation).\n"
        f" missing: {sorted(per_gen_expected - logged)}\n"
        f" extra: {sorted(logged - per_gen_expected)}\n"
        f" got: {sorted(logged)}"
    )
    assert logged.isdisjoint(_SURROGATE_METRICS), (
        f"the per-generation commit must NOT log any surrogate key; leaked "
        f"{sorted(logged & set(_SURROGATE_METRICS))}."
    )


@pytest.mark.unit
def test_metric_names_no_underscore_prefix(
    plugin: SGAPlugin,
    recorder: VizRecorder,
) -> None:
    for name in EXPECTED_METRICS:
        assert not name.startswith("_"), (
            f"Whitelist entry {name!r} must not start with '_'."
        )

    _commit_generation(plugin, _distinct_batch())
    live_keys = recorder.keys()




    missing = EXPECTED_METRICS - _non_underscore_keys(recorder)
    assert not missing, (
        f"commit must log all 6 plugin-written metrics; missing "
        f"{sorted(missing)}. (The '_' prefix check would otherwise pass "
        f"vacuously on the legacy {_LEGACY_SGA_KEY!r} series.)"
    )
    for name in live_keys:
        assert not name.startswith("_"), (
            f"Live recorder key {name!r} has a reserved '_' prefix."
        )


@pytest.mark.unit
def test_whitelist_namespace_disjoint() -> None:
    overlap_platform = EXPECTED_METRICS & _PLATFORM_KEYS
    assert not overlap_platform, (
        f"SGA whitelist collides with platform keys: {sorted(overlap_platform)}."
    )
    assert _LEGACY_SGA_KEY not in EXPECTED_METRICS, (
        f"SGA whitelist must not re-use the legacy {_LEGACY_SGA_KEY!r} key "
        f"(it stays a separate series)."
    )







@pytest.mark.unit
@pytest.mark.parametrize("field", sorted(DISTINCT_EXPECTED))
def test_each_metric_value_is_correct(
    plugin: SGAPlugin,
    recorder: VizRecorder,
    field: str,
) -> None:
    _commit_generation(plugin, _distinct_batch())

    series = recorder.get(field)
    assert len(series) == 1, (
        f"recorder.get({field!r}) has {len(series)} entries after one commit; "
        f"expected exactly 1."
    )
    recorded = series[-1]
    expected = DISTINCT_EXPECTED[field]

    if field in _COUNT_FIELDS:



        assert type(recorded) is int, (
            f"{field}: recorded value must be a built-in int, got "
            f"{type(recorded).__name__} ({recorded!r}). A numpy/tensor scalar "
            f"breaks VizRecorder JSON serialization (string fallback)."
        )
        assert recorded == expected, (
            f"{field}: recorded {recorded!r}, expected exactly {expected!r} "
            f"(a count under another count's value indicates a swap-by-key bug)."
        )
    else:
        assert recorded == pytest.approx(expected, rel=1e-9, abs=1e-12), (
            f"{field}: recorded {recorded!r}, expected {expected!r}. Bands are "
            f"disjoint (aic [10,20) / nmse [0,1) / complexity [5,10)); a value "
            f"in the wrong band means the plugin logged the wrong source."
        )


@pytest.mark.unit
def test_mean_aic_is_mean_not_min(
    plugin: SGAPlugin,
    recorder: VizRecorder,
) -> None:
    _commit_generation(plugin, _distinct_batch())

    mean_aic = _last(recorder, "gen_mean_aic")
    best_aic = _last(recorder, "gen_best_aic")
    assert mean_aic == pytest.approx(15.0, rel=1e-9), (
        f"gen_mean_aic must be the arithmetic mean of valid aics "
        f"((12+15+18)/3 = 15.0), NOT the min (12.0). Got {mean_aic!r}."
    )
    assert best_aic == pytest.approx(12.0, rel=1e-9), (
        f"gen_best_aic must be the min valid aic (12.0). Got {best_aic!r}."
    )
    assert mean_aic != pytest.approx(best_aic), (
        "gen_mean_aic must differ from gen_best_aic on this batch — if they "
        "are equal the impl logged min twice (mean collapsed to best)."
    )


@pytest.mark.unit
def test_best_nmse_tracks_best_aic_individual(
    plugin: SGAPlugin,
    recorder: VizRecorder,
) -> None:
    batch = [
        _valid_result(expression="M0", aic=10.0, nmse=0.50, complexity=3),
        _valid_result(expression="M1", aic=20.0, nmse=0.10, complexity=3),
        _valid_result(expression="M2", aic=15.0, nmse=0.30, complexity=3),
        _invalid_result(expression="M3"),
        _invalid_result(expression="M4"),
    ]
    _commit_generation(plugin, batch)

    best_nmse = _last(recorder, "gen_best_nmse")
    assert best_nmse == pytest.approx(0.50, rel=1e-9), (
        f"gen_best_nmse must be the nmse of the argmin-aic VALID individual "
        f"(m0, nmse=0.50), not the global-min nmse (m1, 0.10). Got {best_nmse!r}."
    )
    best_aic = _last(recorder, "gen_best_aic")
    assert best_aic == pytest.approx(10.0, rel=1e-9), (
        f"gen_best_aic must be 10.0 (argmin over valid aic). Got {best_aic!r}."
    )


@pytest.mark.unit
def test_mean_complexity_uses_selected_indices_count(
    plugin: SGAPlugin,
    recorder: VizRecorder,
) -> None:
    batch = [
        _valid_result(expression="A", aic=1.0, nmse=0.1, complexity=2),
        _valid_result(expression="B", aic=2.0, nmse=0.2, complexity=4),
        _valid_result(expression="C", aic=3.0, nmse=0.3, complexity=6),
        _invalid_result(expression="D"),
        _invalid_result(expression="E"),
    ]
    _commit_generation(plugin, batch)

    mean_cplx = _last(recorder, "gen_mean_complexity")
    assert mean_cplx == pytest.approx(4.0, rel=1e-9), (
        f"gen_mean_complexity must average over VALID individuals only "
        f"((2+4+6)/3 = 4.0), not include the zero-complexity invalids "
        f"((2+4+6+0+0)/5 = 2.4). Got {mean_cplx!r}."
    )


@pytest.mark.unit
def test_logging_metrics_does_not_regress_best_aic_series(
    plugin: SGAPlugin,
    recorder: VizRecorder,
) -> None:
    _commit_generation(plugin, _distinct_batch())

    best_aic_series = recorder.get(_LEGACY_SGA_KEY)
    assert len(best_aic_series) == 1, (
        f"legacy {_LEGACY_SGA_KEY!r} series must still log once per commit; "
        f"got length {len(best_aic_series)}."
    )
    assert best_aic_series[-1] == float("inf"), (
        f"best_aic must log self._best_score (still +inf with empty pending "
        f"scores), NOT the generation's gen_best_aic (12.0). Got "
        f"{best_aic_series[-1]!r} — a finite 12.0 means the impl logged "
        f"gen_best_aic under the legacy key."
    )







@pytest.mark.unit
def test_all_invalid_generation_still_logs_all_series(
    plugin: SGAPlugin,
    recorder: VizRecorder,
) -> None:
    _commit_generation(plugin, _all_invalid_batch())

    lengths = _logged_series_lengths(recorder, sorted(EXPECTED_METRICS))
    for name, length in lengths.items():
        assert length == 1, (
            f"series {name!r} length {length} after a zero-valid commit; "
            f"expected 1. The plugin must NOT skip logging when n_valid==0."
        )

    assert recorder.get("n_valid")[-1] == 0

    assert recorder.get("n_unique")[-1] == 3, (
        f"n_unique must count distinct expressions over ALL results "
        f"({{X,Y,Z}} = 3) even when n_valid==0; got "
        f"{recorder.get('n_unique')[-1]!r}."
    )

    best_aic = recorder.get("gen_best_aic")[-1]
    mean_aic = recorder.get("gen_mean_aic")[-1]
    best_nmse = recorder.get("gen_best_nmse")[-1]

    assert best_aic == float("inf"), (
        f"gen_best_aic must be +inf when no valid individual (and NOT -inf — "
        f"that would invert the fitness plot). Got {best_aic!r}."
    )
    assert mean_aic == float("inf"), (
        f"gen_mean_aic must be +inf when no valid individual (and NOT -inf). "
        f"Got {mean_aic!r}."
    )
    assert best_nmse == float("inf"), (
        f"gen_best_nmse must be +inf when no valid individual — there is no "
        f"best individual to read nmse from. Got {best_nmse!r}."
    )

    mean_complexity = recorder.get("gen_mean_complexity")[-1]
    assert mean_complexity == pytest.approx(0.0, abs=1e-12), (
        f"gen_mean_complexity is LOCKED to 0.0 for the no-valid case; got "
        f"{mean_complexity!r}."
    )


@pytest.mark.unit
def test_series_length_equals_commit_count_across_generations(
    plugin: SGAPlugin,
    recorder: VizRecorder,
) -> None:
    generations = [_distinct_batch(), _all_invalid_batch(), _distinct_batch()]
    for batch in generations:
        _commit_generation(plugin, batch)

    n = len(generations)
    for name in sorted(EXPECTED_METRICS):
        series = recorder.get(name)
        assert isinstance(series, list), (
            f"recorder.get({name!r}) must be a list, got {type(series).__name__}"
        )
        assert len(series) == n, (
            f"After {n} commits, series {name!r} has length {len(series)}; "
            f"expected {n} (one append per generation, incl. the zero-valid one)."
        )







@pytest.mark.unit
def test_n_unique_counts_distinct_expressions_over_all_results(
    plugin: SGAPlugin,
    recorder: VizRecorder,
) -> None:
    batch = [
        _valid_result(expression="A", aic=1.0, nmse=0.1, complexity=2),
        _valid_result(expression="A", aic=2.0, nmse=0.2, complexity=2),
        _valid_result(expression="B", aic=3.0, nmse=0.3, complexity=2),
        _invalid_result(expression="B"),
        _invalid_result(expression="C"),
        _invalid_result(expression="D"),
    ]
    _commit_generation(plugin, batch)

    n_unique = _last(recorder, "n_unique")
    assert n_unique == 4, (
        f"n_unique must count distinct expressions across ALL results "
        f"({{A,B,C,D}} = 4), not total results (6) nor valid count (3). "
        f"Got {n_unique!r}."
    )
    n_valid = _last(recorder, "n_valid")
    assert n_valid == 3, (
        f"n_valid must be 3 (A,A,B valid); got {n_valid!r}. (Sanity: confirms "
        f"n_unique != n_valid so they cannot be silently swapped.)"
    )


@pytest.mark.unit
def test_n_unique_folds_repeated_empty_string_expressions(
    plugin: SGAPlugin,
    recorder: VizRecorder,
) -> None:
    batch = [
        _valid_result(expression="A", aic=1.0, nmse=0.1, complexity=2),
        _invalid_result(expression=""),
        _invalid_result(expression=""),
        _invalid_result(expression=""),
    ]
    _commit_generation(plugin, batch)

    n_unique = _last(recorder, "n_unique")
    assert n_unique == 2, (
        f"n_unique must fold repeated empty-string expressions into one "
        f'distinct value ({{"A", ""}} = 2), not count each blank separately '
        f"(would be 4). Got {n_unique!r}."
    )







@pytest.mark.unit
def test_commit_with_recorder_none_does_not_crash() -> None:
    plugin = _plugin_with_recorder(None)

    _commit_generation(plugin, _distinct_batch())
    _commit_generation(plugin, _all_invalid_batch())


@pytest.mark.unit
def test_commit_with_disabled_recorder_logs_nothing() -> None:
    disabled = VizRecorder(enabled=False)
    plugin = _plugin_with_recorder(disabled)

    _commit_generation(plugin, _distinct_batch())

    assert disabled.keys() == set(), (
        f"A disabled recorder must stay empty after commit; got "
        f"{sorted(disabled.keys())}."
    )
    for name in EXPECTED_METRICS:
        assert disabled.get(name) == [], (
            f"Disabled recorder series {name!r} must be empty, got "
            f"{disabled.get(name)!r}."
        )


@pytest.mark.unit
def test_empty_offspring_logs_full_whitelist_with_sentinels(
    plugin: SGAPlugin, recorder: VizRecorder
) -> None:
    _commit_generation(plugin, [])

    keys = _non_underscore_keys(recorder)


    metric_keys = keys - {_LEGACY_SGA_KEY}
    assert metric_keys == EXPECTED_METRICS, (
        f"Empty generation must log the FULL 6-field whitelist (an empty "
        f"batch is still a generation; dropping it would contradict the "
        f"all-invalid logging contract).\n"
        f" missing: {sorted(EXPECTED_METRICS - metric_keys)}\n"
        f" extra: {sorted(metric_keys - EXPECTED_METRICS)}"
    )
    assert _LEGACY_SGA_KEY in keys, (
        f"legacy {_LEGACY_SGA_KEY!r} series must still log on an empty generation."
    )

    for name in ("n_valid", "n_unique"):
        assert recorder.get(name)[-1] == 0, (
            f"{name} must be 0 for an empty batch; got {recorder.get(name)[-1]!r}."
        )
    for name in ("gen_best_aic", "gen_mean_aic", "gen_best_nmse"):
        value = recorder.get(name)[-1]
        assert value == float("inf"), (
            f"{name} must be +inf for an empty batch (no valid individual); "
            f"got {value!r}."
        )
    mean_complexity = recorder.get("gen_mean_complexity")[-1]
    assert mean_complexity == pytest.approx(0.0, abs=1e-12), (
        f"gen_mean_complexity must be 0.0 for an empty batch; got {mean_complexity!r}."
    )







def _make_synthetic_dataset() -> PDEDataset:
    x_vals = torch.linspace(0.0, 1.0, _SMALL_GRID_SIZE)
    t_vals = torch.linspace(0.0, 1.0, _SMALL_TIME_SIZE)

    xg, tg = torch.meshgrid(x_vals, t_vals, indexing="ij")
    u_data = torch.sin(xg) + tg
    return PDEDataset(
        name="sga-recorder-test",
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


def _real_components_with_recorder(
    plugin: SGAPlugin, recorder: VizRecorder | None
) -> PlatformComponents:
    from kd.core.platform.builder import PlatformBuilder

    dataset = _make_synthetic_dataset()
    components = PlatformBuilder(dataset, plugin.derivative_requirements).build()
    components.recorder = recorder
    return components


@pytest.fixture
def real_sga_config() -> SGAConfig:
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


@pytest.mark.unit
def test_real_prepare_binds_recorder(real_sga_config: SGAConfig) -> None:
    recorder = VizRecorder(enabled=True)
    plugin = SGAPlugin(real_sga_config)
    components = _real_components_with_recorder(plugin, recorder)

    plugin.prepare(components)

    captured = getattr(plugin, "_recorder", "MISSING_ATTR")
    assert captured is recorder, (
        "SGAPlugin.prepare() must capture components.recorder into "
        "self._recorder (plugin.py:233)."
    )


@pytest.mark.unit
def test_real_generation_logs_whitelist(real_sga_config: SGAConfig) -> None:
    recorder = VizRecorder(enabled=True)
    plugin = SGAPlugin(real_sga_config)
    components = _real_components_with_recorder(plugin, recorder)
    plugin.prepare(components)

    candidates = plugin.propose(real_sga_config.num)
    results = plugin.evaluate(candidates)
    plugin.update(results)

    logged = _non_underscore_keys(recorder)
    expected = EXPECTED_METRICS | {_LEGACY_SGA_KEY}
    assert expected <= logged, (
        f"After a real generation, recorder must hold the 6-field whitelist + "
        f"{_LEGACY_SGA_KEY!r}.\n missing: {sorted(expected - logged)}\n"
        f" got: {sorted(logged)}"
    )

    for name in EXPECTED_METRICS:
        series = recorder.get(name)
        assert len(series) == 1, (
            f"series {name!r} must have exactly 1 entry after one real "
            f"generation; got {len(series)}."
        )
