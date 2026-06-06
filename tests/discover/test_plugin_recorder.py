
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.search.discover.builder import build_engine
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.plugin import DISCOVERPlugin
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder

if TYPE_CHECKING:
    from kd.search.discover.engine import DiscoverEngine





SEED = 42
BATCH_SIZE = 16


_LOSS_INFO_FIELDS: frozenset[str] = frozenset(
    {
        "pg_loss",
        "entropy_loss",
        "total_loss",
        "baseline",
        "reward",
        "grad_norm",
    },
)


_ENGINE_METRIC_FIELDS: frozenset[str] = frozenset(
    {
        "reward_max",
        "best_reward",
        "n_valid",
        "n_eval_valid",
        "n_invalid_in_topk",
        "n_unique",
    },
)


EXPECTED_METRICS: frozenset[str] = _LOSS_INFO_FIELDS | _ENGINE_METRIC_FIELDS






class _MockEvaluator:

    def __init__(self, default_nmse: float = 0.5) -> None:
        self.default_nmse = default_nmse

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        return EvaluationResult(
            mse=self.default_nmse,
            nmse=self.default_nmse,
            r2=max(0.0, 1.0 - self.default_nmse),
            complexity=3,
            is_valid=True,
            expression=expr,
        )







def _make_components(recorder: VizRecorder | None) -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=_MockEvaluator(),
        context=MagicMock(),
        registry=MagicMock(),
        recorder=recorder,
    )


@pytest.fixture
def recorder() -> VizRecorder:
    return VizRecorder(enabled=True)


@pytest.fixture
def components(recorder: VizRecorder) -> PlatformComponents:
    return _make_components(recorder)


@pytest.fixture
def components_no_recorder() -> PlatformComponents:
    return _make_components(None)


@pytest.fixture
def plugin(components: PlatformComponents) -> DISCOVERPlugin:
    torch.manual_seed(SEED)
    p = DISCOVERPlugin()
    p.prepare(components)
    return p


def _run_one_cycle(p: DISCOVERPlugin) -> None:
    torch.manual_seed(SEED)
    candidates = p.propose(BATCH_SIZE)
    results = p.evaluate(candidates)
    p.update(results)







@pytest.mark.unit
def test_recorder_is_none_before_prepare() -> None:
    plugin = DISCOVERPlugin()
    captured = getattr(plugin, "_recorder", "MISSING_ATTR")
    assert captured is None, (
        f"DISCOVERPlugin().__init__ must set self._recorder = None "
        f"(attribute must exist with value None before prepare()). "
        f"Got: {captured!r}"
    )


@pytest.mark.unit
def test_prepare_captures_recorder(
    components: PlatformComponents,
    recorder: VizRecorder,
) -> None:
    plugin = DISCOVERPlugin()
    plugin.prepare(components)

    captured = getattr(plugin, "_recorder", None)
    assert captured is recorder, (
        "DISCOVERPlugin.prepare() must capture components.recorder into "
        "self._recorder (mirror SGAPlugin.prepare line 233)."
    )







@pytest.mark.unit
def test_update_logs_engine_metrics(
    plugin: DISCOVERPlugin,
    recorder: VizRecorder,
) -> None:
    _run_one_cycle(plugin)

    logged = recorder.keys()

    assert logged == EXPECTED_METRICS, (
        f"recorder.keys() must equal the explicit 12-tuple whitelist.\n"
        f" missing: {sorted(EXPECTED_METRICS - logged)}\n"
        f" extra: {sorted(logged - EXPECTED_METRICS)}\n"
        f" got: {sorted(logged)}"
    )


@pytest.mark.unit
def test_logged_values_come_from_engine_last_metrics(
    plugin: DISCOVERPlugin,
    recorder: VizRecorder,
) -> None:
    _run_one_cycle(plugin)


    engine = getattr(plugin, "_engine", None)
    assert engine is not None, (
        "Test invariant broken: plugin._engine must exist after prepare()."
    )
    engine_metrics = engine.last_metrics
    assert engine_metrics, (
        "engine.last_metrics is empty after one cycle — engine.update() "
        "did not populate _last_metrics."
    )

    for name in EXPECTED_METRICS:
        assert name in engine_metrics, (
            f"engine.last_metrics missing {name!r} after one cycle — "
            f"engine contract drift, not plugin bug. "
            f"engine.last_metrics keys: {sorted(engine_metrics.keys())}"
        )
        series = recorder.get(name)
        assert len(series) == 1, (
            f"recorder.get({name!r}) has {len(series)} entries; "
            f"expected exactly 1 after one cycle."
        )
        recorded = series[-1]
        expected = engine_metrics[name]
        assert recorded == pytest.approx(expected), (
            f"recorder['{name}'][-1] = {recorded!r} does not match "
            f"engine.last_metrics['{name}'] = {expected!r}. "
            f"Plugin must forward engine values literally, not "
            f"placeholders/zeros/constants."
        )


@pytest.mark.unit
def test_update_appends_one_entry_per_cycle(
    plugin: DISCOVERPlugin,
    recorder: VizRecorder,
) -> None:
    n_cycles = 2
    for _ in range(n_cycles):
        _run_one_cycle(plugin)

    for name in EXPECTED_METRICS:
        series = recorder.get(name)
        assert isinstance(series, list), (
            f"recorder.get({name!r}) must return list, got {type(series).__name__}"
        )
        assert len(series) == n_cycles, (
            f"After {n_cycles} cycles, recorder.get({name!r}) length = "
            f"{len(series)}; expected {n_cycles} (one append per update)."
        )







@pytest.mark.unit
def test_recorder_log_uses_get_api(
    plugin: DISCOVERPlugin,
    recorder: VizRecorder,
) -> None:
    _run_one_cycle(plugin)


    sample_series = recorder.get("pg_loss")
    assert isinstance(sample_series, list), (
        f"recorder.get('pg_loss') must be list, got {type(sample_series).__name__}"
    )
    assert len(sample_series) >= 1, (
        "recorder.get('pg_loss') must have >= 1 entry after one update() cycle."
    )

    keys = recorder.keys()
    assert isinstance(keys, set), (
        f"recorder.keys() must be set, got {type(keys).__name__}"
    )



    assert not hasattr(recorder, "history"), (
        "VizRecorder must not expose a .history attribute — plugins MUST "
        "route writes through .log(key, value)."
    )







@pytest.mark.unit
def test_metric_names_no_underscore_prefix(
    plugin: DISCOVERPlugin,
    recorder: VizRecorder,
) -> None:

    for name in EXPECTED_METRICS:
        assert not name.startswith("_"), (
            f"Whitelist entry {name!r} must not start with '_' — that "
            "namespace is reserved for platform-written keys (-D14)."
        )


    _run_one_cycle(plugin)
    for name in EXPECTED_METRICS:
        assert name in recorder.keys(), (
            f"Expected plugin-written key {name!r} missing from recorder."
        )
        assert not name.startswith("_"), (
            f"Live recorder key {name!r} has reserved '_' prefix."
        )


@pytest.mark.unit
def test_metrics_namespace_disjoint() -> None:
    PLATFORM_KEYS = {"_best_score", "_best_expr", "_n_candidates"}
    SGA_KEYS = {"best_aic"}

    overlap_platform = EXPECTED_METRICS & PLATFORM_KEYS
    assert not overlap_platform, (
        f"DISCOVER whitelist collides with platform-written keys: "
        f"{sorted(overlap_platform)}. Platform owns the '_' namespace "
        f"(-D14); DISCOVER metrics must stay outside it."
    )

    overlap_sga = EXPECTED_METRICS & SGA_KEYS
    assert not overlap_sga, (
        f"DISCOVER whitelist collides with SGA plugin keys: "
        f"{sorted(overlap_sga)}. Pick names that disambiguate the source "
        f"plugin so multi-algorithm runs do not silently overwrite."
    )


@pytest.mark.unit
def test_n_valid_semantic_documented() -> None:
    import inspect

    from kd.search.discover import plugin as discover_plugin

    source = inspect.getsource(discover_plugin)

    assert "_LOGGED_METRICS" in source, "module must define _LOGGED_METRICS tuple"


    assert "_n_candidates" in source and "n_valid" in source, (
        "plugin.py must document the n_valid vs _n_candidates semantic "
        "distinction next to _LOGGED_METRICS (Stage 5 M6)."
    )
    assert "syntactic-valid" in source, (
        "plugin.py docstring must explain that n_valid is the "
        "syntactic-valid IR count, distinct from the platform's "
        "total-proposed-batch counter (Stage 5 M6)."
    )







@pytest.mark.unit
def test_recorder_none_does_not_crash(
    components_no_recorder: PlatformComponents,
) -> None:
    torch.manual_seed(SEED)
    plugin = DISCOVERPlugin()
    plugin.prepare(components_no_recorder)


    _run_one_cycle(plugin)



    captured = getattr(plugin, "_recorder", "MISSING_ATTR")
    assert captured is None, (
        "With components.recorder=None, plugin._recorder must be None "
        f"(no synthesized fallback). Got: {captured!r}"
    )







@pytest.mark.unit
def test_engine_last_metrics_public_property() -> None:
    config = DiscoverConfig()
    torch.manual_seed(SEED)
    engine: DiscoverEngine = build_engine(config)


    metrics_after_run = engine.run_iteration(_MockEvaluator())
    assert metrics_after_run, "Test setup error: engine returned empty metrics."




    descriptor = type(engine).__dict__.get("last_metrics", None)
    assert isinstance(descriptor, property), (
        "DiscoverEngine.last_metrics must be an @property; got "
        f"{type(descriptor).__name__ if descriptor is not None else 'MISSING'}"
    )


    m1 = engine.last_metrics
    assert isinstance(m1, dict), (
        f"engine.last_metrics must return dict, got {type(m1).__name__}"
    )
    assert "pg_loss" in m1, (
        "engine.last_metrics missing 'pg_loss' after run_iteration() — "
        "did the property forget to copy from self._last_metrics?"
    )


    m1["pg_loss"] = 999.0
    m2 = engine.last_metrics
    assert m2["pg_loss"] != 999.0, (
        "engine.last_metrics returned an aliased dict; caller mutation "
        "of m1 leaked into the engine's _last_metrics. Property must "
        "return a defensive copy (dict(self._last_metrics))."
    )
