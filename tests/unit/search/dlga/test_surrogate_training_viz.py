
from __future__ import annotations

import json
import math
from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn
from matplotlib.axes import Axes

from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.models.trainer import TrainingResult
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.dlga import viz as dlga_viz
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder




from tests.unit.search.dlga.test_viz_extension import (
    GA_PLOT_NAMES,
    SURROGATE_PLOT_NAME,
)








SURROGATE_KEYS_ALWAYS: frozenset[str] = frozenset(
    {
        "surrogate_epoch",
        "surrogate_train_loss",
        "surrogate_epochs_run",
        "surrogate_early_stopped",
    }
)
SURROGATE_KEYS_CONDITIONAL: frozenset[str] = frozenset(
    {"surrogate_val_loss", "surrogate_best_epoch"}
)
SURROGATE_KEYS_ALL: frozenset[str] = SURROGATE_KEYS_ALWAYS | SURROGATE_KEYS_CONDITIONAL


NEW_PLOT_NAME = SURROGATE_PLOT_NAME

EXISTING_PLOT_NAMES = ("fitness_spread", "population_diversity", "complexity_evolution")

assert set(EXISTING_PLOT_NAMES) == GA_PLOT_NAMES

_MAX_LOGGED_EPOCHS = 1024


def _lazy_surrogate_metrics() -> tuple[str, ...]:
    from kd.search.dlga import plugin as dlga_plugin

    metrics = getattr(dlga_plugin, "_SURROGATE_METRICS", None)
    assert metrics is not None, "dlga/plugin.py must define _SURROGATE_METRICS"
    assert isinstance(metrics, tuple)
    return tuple(str(name) for name in metrics)







def _make_training_result(
    *,
    n_epochs: int,
    with_val: bool,
    early_stopped: bool = False,
    best_epoch: int | None = None,
) -> TrainingResult:
    train = [100.0 / (k + 1) for k in range(n_epochs)]
    val = [10.0 / (k + 1) for k in range(n_epochs)] if with_val else None
    return TrainingResult(
        final_loss=train[-1],
        epochs_run=n_epochs,
        early_stopped=early_stopped,
        val_loss=(val[-1] if val is not None else None),
        best_val_loss=(min(val) if val is not None else None),
        best_epoch=best_epoch,
        best_restored=best_epoch is not None,
        loss_history=train,
        val_loss_history=val,
    )


class _ExactQuadraticModel(nn.Module):

    def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"u": 1.0 + x * x + t * t}


def _real_components(
    plugin: DLGAPlugin, recorder: VizRecorder | None
) -> PlatformComponents:
    from kd.core.platform.builder import PlatformBuilder

    x = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = 1.0 + xg * xg + tg * tg
    dataset = PDEDataset(
        name="dlga-surrogate-viz-test",
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


def _attach_training_result(
    components: PlatformComponents, training_result: TrainingResult | None
) -> None:
    components.context.training_result = training_result


def _prepared_plugin_with_curve(
    training_result: TrainingResult | None,
    recorder: VizRecorder | None,
) -> tuple[DLGAPlugin, VizRecorder | None]:
    plugin = DLGAPlugin(
        DLGAConfig(pop_size=4, seed=7),
        surrogate_model=_ExactQuadraticModel(),
    )
    components = _real_components(plugin, recorder)
    _attach_training_result(components, training_result)
    plugin.prepare(components)
    return plugin, recorder







@pytest.mark.unit
def test_surrogate_metrics_constant_equals_six_key_whitelist() -> None:
    metrics = _lazy_surrogate_metrics()
    assert isinstance(metrics, tuple), (
        f"_SURROGATE_METRICS must be a tuple[str, ...], got {type(metrics).__name__}"
    )
    assert len(metrics) == len(set(metrics)), (
        f"_SURROGATE_METRICS has duplicate entries: {metrics}"
    )
    assert set(metrics) == SURROGATE_KEYS_ALL, (
        f"_SURROGATE_METRICS must equal the 6-key surrogate whitelist.\n"
        f" missing: {sorted(SURROGATE_KEYS_ALL - set(metrics))}\n"
        f" extra: {sorted(set(metrics) - SURROGATE_KEYS_ALL)}"
    )


@pytest.mark.unit
def test_surrogate_metrics_disjoint_from_generation_whitelist() -> None:
    from kd.search.dlga import plugin as dlga_plugin

    gen_metrics = set(dlga_plugin._LOGGED_METRICS)
    surrogate = set(_lazy_surrogate_metrics())
    overlap = gen_metrics & surrogate
    assert not overlap, (
        f"_SURROGATE_METRICS must be disjoint from _LOGGED_METRICS (a shared "
        f"recorder would silently overwrite); overlap={sorted(overlap)}."
    )


@pytest.mark.unit
def test_surrogate_metric_names_no_underscore_prefix() -> None:
    for name in _lazy_surrogate_metrics():
        assert not name.startswith("_"), (
            f"surrogate metric {name!r} must not start with '_' (reserved for "
            "platform-written series)."
        )







@pytest.mark.unit
def test_prepare_logs_exactly_six_surrogate_series_with_val() -> None:
    recorder = VizRecorder(enabled=True)
    tr = _make_training_result(n_epochs=40, with_val=True, best_epoch=12)
    _prepared_plugin_with_curve(tr, recorder)

    keys = recorder.keys()
    assert keys == SURROGATE_KEYS_ALL, (
        f"prepare-only (val+best) must log EXACTLY the 6 surrogate series.\n"
        f" missing: {sorted(SURROGATE_KEYS_ALL - keys)}\n"
        f" extra: {sorted(keys - SURROGATE_KEYS_ALL)}"
    )


@pytest.mark.unit
def test_prepare_epoch_series_is_one_based_increasing_bounded() -> None:
    recorder = VizRecorder(enabled=True)
    n_epochs = 40
    tr = _make_training_result(n_epochs=n_epochs, with_val=True, best_epoch=5)
    _prepared_plugin_with_curve(tr, recorder)

    epoch_series = recorder.get("surrogate_epoch")

    assert len(epoch_series) == 1, (
        f"surrogate_epoch must be logged exactly once (one-shot); got "
        f"{len(epoch_series)} log entries."
    )
    epochs = epoch_series[-1]
    assert isinstance(epochs, list) and epochs, "surrogate_epoch payload must be a list"
    assert len(epochs) <= _MAX_LOGGED_EPOCHS
    assert epochs[0] == 1, f"first epoch must be 1 (1-based); got {epochs[0]!r}"
    assert epochs[-1] == n_epochs, (
        f"last epoch must be epochs_run ({n_epochs}); got {epochs[-1]!r}"
    )
    assert all(b > a for a, b in zip(epochs, epochs[1:], strict=False)), (
        f"surrogate_epoch must be strictly increasing; got {epochs!r}"
    )


@pytest.mark.unit
def test_prepare_train_series_aligned_with_epoch_and_monotone() -> None:
    recorder = VizRecorder(enabled=True)
    tr = _make_training_result(n_epochs=40, with_val=True, best_epoch=3)
    _prepared_plugin_with_curve(tr, recorder)

    epochs = recorder.get("surrogate_epoch")[-1]
    train = recorder.get("surrogate_train_loss")[-1]
    assert len(train) == len(epochs), (
        f"surrogate_train_loss ({len(train)}) must align 1:1 with "
        f"surrogate_epoch ({len(epochs)})."
    )

    assert all(b < a for a, b in zip(train, train[1:], strict=False)), (
        f"aligned train series must stay strictly decreasing; got {train!r}"
    )

    for e, v in zip(epochs, train, strict=True):
        assert v == pytest.approx(100.0 / e, rel=1e-9), (
            f"train_loss at epoch {e} must be {100.0 / e!r} (index alignment "
            f"with the source history); got {v!r}."
        )


@pytest.mark.unit
def test_prepare_val_series_aligned_with_epoch() -> None:
    recorder = VizRecorder(enabled=True)
    tr = _make_training_result(n_epochs=40, with_val=True, best_epoch=3)
    _prepared_plugin_with_curve(tr, recorder)

    epochs = recorder.get("surrogate_epoch")[-1]
    val = recorder.get("surrogate_val_loss")[-1]
    assert len(val) == len(epochs), "val series must align 1:1 with epoch series"
    for e, v in zip(epochs, val, strict=True):
        assert v == pytest.approx(10.0 / e, rel=1e-9), (
            f"val_loss at epoch {e} must be {10.0 / e!r} (10x below train — the "
            f"swap guard); got {v!r}."
        )


@pytest.mark.unit
def test_prepare_scalar_summaries_are_single_sample_series() -> None:
    recorder = VizRecorder(enabled=True)
    n_epochs = 37
    best = 9
    tr = _make_training_result(
        n_epochs=n_epochs, with_val=True, early_stopped=True, best_epoch=best
    )
    _prepared_plugin_with_curve(tr, recorder)

    epochs_run = recorder.get("surrogate_epochs_run")
    assert epochs_run == [float(n_epochs)], (
        f"surrogate_epochs_run must be a single-sample series [float(epochs_run)]; "
        f"got {epochs_run!r}."
    )



    assert type(epochs_run[0]) is float, (
        f"surrogate_epochs_run sample must be a built-in float; got "
        f"{type(epochs_run[0]).__name__}."
    )
    best_epoch = recorder.get("surrogate_best_epoch")
    assert best_epoch == [float(best)], (
        f"surrogate_best_epoch must be [float(best_epoch)]; got {best_epoch!r}."
    )
    assert type(best_epoch[0]) is float, (
        f"surrogate_best_epoch sample must be a built-in float; got "
        f"{type(best_epoch[0]).__name__}."
    )
    early = recorder.get("surrogate_early_stopped")
    assert early == [1.0], (
        f"surrogate_early_stopped must be [1.0] when early_stopped; got {early!r}."
    )

    assert type(early[0]) is float, (
        f"surrogate_early_stopped sample must be a float (not bool); got "
        f"{type(early[0]).__name__}."
    )


@pytest.mark.unit
def test_prepare_early_stopped_false_logs_zero() -> None:
    recorder = VizRecorder(enabled=True)
    tr = _make_training_result(
        n_epochs=20, with_val=True, early_stopped=False, best_epoch=4
    )
    _prepared_plugin_with_curve(tr, recorder)
    early = recorder.get("surrogate_early_stopped")
    assert early == [0.0], f"not-early-stopped must log [0.0]; got {early!r}."
    assert type(early[0]) is float


@pytest.mark.unit
def test_prepare_no_val_omits_val_series_entirely() -> None:
    recorder = VizRecorder(enabled=True)
    tr = _make_training_result(n_epochs=30, with_val=False, best_epoch=None)
    _prepared_plugin_with_curve(tr, recorder)

    keys = recorder.keys()
    assert "surrogate_val_loss" not in keys, (
        "surrogate_val_loss must be absent entirely when val_loss_history is "
        f"None (not logged as an empty series); got keys {sorted(keys)}."
    )

    assert keys == SURROGATE_KEYS_ALWAYS, (
        f"no-val run must log EXACTLY the 4 always-present surrogate series.\n"
        f" missing: {sorted(SURROGATE_KEYS_ALWAYS - keys)}\n"
        f" extra: {sorted(keys - SURROGATE_KEYS_ALWAYS)}"
    )


@pytest.mark.unit
def test_prepare_no_best_epoch_omits_best_epoch_series() -> None:
    recorder = VizRecorder(enabled=True)

    tr = _make_training_result(n_epochs=30, with_val=False, best_epoch=None)
    _prepared_plugin_with_curve(tr, recorder)
    keys = recorder.keys()
    assert "surrogate_best_epoch" not in keys, (
        "surrogate_best_epoch must be absent when best_epoch is None; got "
        f"{sorted(keys)}."
    )


@pytest.mark.unit
def test_prepare_downsamples_long_history_within_bound() -> None:
    recorder = VizRecorder(enabled=True)
    n_epochs = 3000
    tr = _make_training_result(n_epochs=n_epochs, with_val=True, best_epoch=1500)
    _prepared_plugin_with_curve(tr, recorder)

    epochs = recorder.get("surrogate_epoch")[-1]
    train = recorder.get("surrogate_train_loss")[-1]
    val = recorder.get("surrogate_val_loss")[-1]

    assert len(epochs) <= _MAX_LOGGED_EPOCHS, (
        f"downsampled epoch series must be <= {_MAX_LOGGED_EPOCHS}; got {len(epochs)}."
    )
    assert epochs[0] == 1, "must keep the first epoch (1)"
    assert epochs[-1] == n_epochs, f"must keep the last epoch ({n_epochs})"
    assert all(b > a for a, b in zip(epochs, epochs[1:], strict=False)), (
        "downsampled epochs must stay strictly increasing"
    )
    assert len(train) == len(epochs) == len(val), (
        f"train/val/epoch must stay index-aligned after downsampling; got "
        f"train={len(train)} val={len(val)} epoch={len(epochs)}."
    )

    for e, tv, vv in zip(epochs, train, val, strict=True):
        assert tv == pytest.approx(100.0 / e, rel=1e-9)
        assert vv == pytest.approx(10.0 / e, rel=1e-9)


@pytest.mark.unit
def test_prepare_short_history_is_not_downsampled() -> None:
    recorder = VizRecorder(enabled=True)
    n_epochs = 16
    tr = _make_training_result(n_epochs=n_epochs, with_val=True, best_epoch=4)
    _prepared_plugin_with_curve(tr, recorder)

    epochs = recorder.get("surrogate_epoch")[-1]
    assert epochs == list(range(1, n_epochs + 1)), (
        f"a short history must keep every 1-based epoch; got {epochs!r}."
    )







@pytest.mark.unit
def test_prepare_twice_does_not_duplicate_surrogate_series() -> None:
    recorder = VizRecorder(enabled=True)
    plugin = DLGAPlugin(
        DLGAConfig(pop_size=4, seed=7),
        surrogate_model=_ExactQuadraticModel(),
    )
    components = _real_components(plugin, recorder)
    tr = _make_training_result(n_epochs=40, with_val=True, best_epoch=8)
    _attach_training_result(components, tr)

    plugin.prepare(components)
    lengths_after_first = {k: len(recorder.get(k)) for k in SURROGATE_KEYS_ALL}

    plugin.prepare(components)
    lengths_after_second = {k: len(recorder.get(k)) for k in SURROGATE_KEYS_ALL}

    assert lengths_after_second == lengths_after_first, (
        "prepare() must be idempotent for the surrogate curve: a second "
        "prepare() with a non-empty surrogate_train_loss series must not "
        f"re-append. first={lengths_after_first} second={lengths_after_second}."
    )

    assert len(recorder.get("surrogate_train_loss")) == 1


@pytest.mark.unit
def test_prepare_reused_plugin_fresh_recorder_does_log() -> None:
    plugin = DLGAPlugin(
        DLGAConfig(pop_size=4, seed=7),
        surrogate_model=_ExactQuadraticModel(),
    )

    rec1 = VizRecorder(enabled=True)
    comp1 = _real_components(plugin, rec1)
    _attach_training_result(
        comp1, _make_training_result(n_epochs=20, with_val=True, best_epoch=4)
    )
    plugin.prepare(comp1)
    assert rec1.get("surrogate_train_loss"), "first run must log the curve"


    rec2 = VizRecorder(enabled=True)
    comp2 = _real_components(plugin, rec2)
    _attach_training_result(
        comp2, _make_training_result(n_epochs=20, with_val=True, best_epoch=4)
    )
    plugin.prepare(comp2)

    assert rec2.get("surrogate_train_loss"), (
        "a reused plugin must log the surrogate curve into a FRESH recorder on "
        "the next prepare(); the idempotence guard must depend on recorder "
        "state, not a never-reset instance flag (which would drop the curve)."
    )
    assert len(rec2.get("surrogate_train_loss")) == 1


@pytest.mark.unit
def test_prepare_skips_when_recorder_prefilled_via_roundtrip() -> None:


    seed_rec = VizRecorder(enabled=True)
    seed_plugin = DLGAPlugin(
        DLGAConfig(pop_size=4, seed=7),
        surrogate_model=_ExactQuadraticModel(),
    )
    seed_comp = _real_components(seed_plugin, seed_rec)
    _attach_training_result(
        seed_comp, _make_training_result(n_epochs=30, with_val=True, best_epoch=6)
    )
    seed_plugin.prepare(seed_comp)
    restored = VizRecorder.from_dict(seed_rec.to_dict())
    pre_lengths = {k: len(restored.get(k)) for k in SURROGATE_KEYS_ALL}
    assert restored.get("surrogate_train_loss"), "round-trip must preserve the curve"


    fresh_plugin = DLGAPlugin(
        DLGAConfig(pop_size=4, seed=7),
        surrogate_model=_ExactQuadraticModel(),
    )
    fresh_comp = _real_components(fresh_plugin, restored)
    _attach_training_result(
        fresh_comp, _make_training_result(n_epochs=30, with_val=True, best_epoch=6)
    )
    fresh_plugin.prepare(fresh_comp)

    post_lengths = {k: len(restored.get(k)) for k in SURROGATE_KEYS_ALL}
    assert post_lengths == pre_lengths, (
        "prepare() on a fresh plugin with a recorder already holding the "
        "surrogate curve (checkpoint replay) must NOT re-append; the guard must "
        f"key off recorder state, not the plugin instance. pre={pre_lengths} "
        f"post={post_lengths}."
    )
    assert len(restored.get("surrogate_train_loss")) == 1


@pytest.mark.unit
def test_prepare_with_none_recorder_does_not_crash() -> None:
    tr = _make_training_result(n_epochs=20, with_val=True, best_epoch=4)

    plugin, _ = _prepared_plugin_with_curve(tr, recorder=None)
    captured = getattr(plugin, "_recorder", "MISSING")
    assert captured is None, (
        "with a None recorder, prepare() must leave _recorder None and skip "
        f"surrogate logging; got {captured!r}."
    )


@pytest.mark.unit
def test_prepare_without_training_result_logs_no_surrogate_keys() -> None:
    recorder = VizRecorder(enabled=True)

    plugin = DLGAPlugin(
        DLGAConfig(pop_size=4, seed=7),
        surrogate_model=_ExactQuadraticModel(),
    )
    components = _real_components(plugin, recorder)
    _attach_training_result(components, None)
    plugin.prepare(components)

    logged_surrogate = recorder.keys() & SURROGATE_KEYS_ALL
    assert not logged_surrogate, (
        "prepare() with no training_result must log NONE of the surrogate "
        f"series; unexpectedly logged {sorted(logged_surrogate)}."
    )


@pytest.mark.unit
def test_prepare_surrogate_logging_does_not_disturb_generation_logging() -> None:
    from kd.search.dlga import plugin as dlga_plugin

    recorder = VizRecorder(enabled=True)
    tr = _make_training_result(n_epochs=20, with_val=True, best_epoch=4)
    plugin, _ = _prepared_plugin_with_curve(tr, recorder)


    from kd.core.evaluator import EvaluationResult

    invalid = EvaluationResult(
        mse=float("inf"),
        nmse=float("inf"),
        r2=-float("inf"),
        aic=float("inf"),
        complexity=0,
        coefficients=None,
        is_valid=False,
        error_message="forced",
        selected_indices=[],
        residuals=None,
        terms=[],
        expression="X",
        lhs_name="u_t",
    )
    plugin.update([invalid])

    gen_keys = set(dlga_plugin._LOGGED_METRICS)
    assert gen_keys <= recorder.keys(), (
        "per-generation metric logging must still populate after the surrogate "
        f"one-shot log; missing {sorted(gen_keys - recorder.keys())}."
    )







@pytest.fixture
def ax() -> Iterator[Axes]:
    _fig, _ax = plt.subplots()
    yield _ax
    plt.close(_fig)


def _recorder_with_surrogate_curve(
    *,
    with_val: bool = True,
    with_best: bool = True,
    train: list[float] | None = None,
    val: list[float] | None = None,
    epochs: list[int] | None = None,
    best_epoch: float | None = 3.0,
) -> VizRecorder:
    recorder = VizRecorder(enabled=True)
    epochs = epochs if epochs is not None else [1, 2, 3, 4, 5]
    train = train if train is not None else [100.0 / e for e in epochs]
    recorder.log("surrogate_epoch", list(epochs))
    recorder.log("surrogate_train_loss", list(train))
    if with_val:
        val = val if val is not None else [10.0 / e for e in epochs]
        recorder.log("surrogate_val_loss", list(val))
    recorder.log("surrogate_epochs_run", float(epochs[-1]))
    recorder.log("surrogate_early_stopped", 0.0)
    if with_best:
        recorder.log("surrogate_best_epoch", best_epoch)
    return recorder


def _plugin_with_recorder(recorder: VizRecorder | None) -> DLGAPlugin:
    plugin = DLGAPlugin(DLGAConfig())
    plugin._recorder = recorder
    return plugin





@pytest.mark.unit
def test_list_plot_infos_returns_four_with_surrogate_last() -> None:
    infos = dlga_viz.list_plot_infos()
    assert len(infos) == 4, (
        f"list_plot_infos() must return 4 PlotInfo (3 existing + surrogate); "
        f"got {len(infos)}: {[getattr(p, 'name', '?') for p in infos]}."
    )
    names = [p.name for p in infos]
    assert names[:3] == list(EXISTING_PLOT_NAMES), (
        f"the 3 existing plots must come first in order; got {names!r}."
    )
    assert names[-1] == NEW_PLOT_NAME, (
        f"the new plot must be LAST and named {NEW_PLOT_NAME!r}; got {names!r}."
    )


@pytest.mark.unit
def test_plugin_list_plots_returns_four() -> None:
    plugin = DLGAPlugin(DLGAConfig())
    plots = plugin.list_plots()
    assert len(plots) == 4, (
        f"DLGAPlugin.list_plots() must return 4 plots once surrogate_training "
        f"is added; got {len(plots)}: {[getattr(p, 'name', '?') for p in plots]}."
    )
    assert NEW_PLOT_NAME in {p.name for p in plots}





@pytest.mark.unit
def test_render_surrogate_training_returns_none(ax: Axes) -> None:
    recorder = _recorder_with_surrogate_curve()
    result = dlga_viz.render(NEW_PLOT_NAME, ax, recorder)
    assert result is None, f"render must return None; got {type(result).__name__}."


@pytest.mark.unit
def test_render_surrogate_training_is_log_scale(ax: Axes) -> None:
    recorder = _recorder_with_surrogate_curve()
    dlga_viz.render(NEW_PLOT_NAME, ax, recorder)
    assert ax.get_yscale() == "log", (
        f"surrogate_training y-axis must be log scale; got {ax.get_yscale()!r}."
    )


def _line_ydata(line: object) -> np.ndarray:
    return np.asarray(line.get_ydata(), dtype=float)


def _line_xdata(line: object) -> np.ndarray:
    return np.asarray(line.get_xdata(), dtype=float)


def _find_line_in_band(ax: Axes, low: float, high: float) -> object | None:
    for line in ax.lines:
        y = _line_ydata(line)
        finite = y[np.isfinite(y)]
        if finite.size and np.all((finite > low) & (finite < high)):
            return line
    return None


@pytest.mark.unit
def test_render_surrogate_training_two_lines_bound_to_recorder(ax: Axes) -> None:
    epochs = [1, 2, 4, 8, 16]
    train = [200.0 / e for e in epochs]
    val = [2.0 / e for e in epochs]
    recorder = _recorder_with_surrogate_curve(
        with_val=True, with_best=False, epochs=epochs, train=train, val=val
    )
    dlga_viz.render(NEW_PLOT_NAME, ax, recorder)


    assert len(ax.lines) == 2, (
        f"train+val must draw exactly 2 data lines (no marker requested); got "
        f"{len(ax.lines)}."
    )
    train_line = _find_line_in_band(ax, 10.0, 1000.0)
    val_line = _find_line_in_band(ax, 0.0, 10.0)
    assert train_line is not None, "no line found in the train band (10, 1000)"
    assert val_line is not None, "no line found in the val band (0, 10)"
    assert train_line is not val_line, "train and val must be distinct lines"

    np.testing.assert_allclose(_line_ydata(train_line), np.asarray(train))
    np.testing.assert_allclose(_line_ydata(val_line), np.asarray(val))

    np.testing.assert_allclose(_line_xdata(train_line), np.asarray(epochs, dtype=float))
    np.testing.assert_allclose(_line_xdata(val_line), np.asarray(epochs, dtype=float))


@pytest.mark.unit
def test_render_surrogate_training_one_line_bound_to_recorder(ax: Axes) -> None:
    epochs = [1, 2, 4, 8, 16]
    train = [100.0 / e for e in epochs]
    recorder = _recorder_with_surrogate_curve(
        with_val=False, with_best=False, epochs=epochs, train=train
    )
    dlga_viz.render(NEW_PLOT_NAME, ax, recorder)
    assert len(ax.lines) == 1, (
        f"no-val curve must draw exactly one line; got {len(ax.lines)}."
    )
    line = ax.lines[0]
    np.testing.assert_allclose(_line_ydata(line), np.asarray(train))
    np.testing.assert_allclose(_line_xdata(line), np.asarray(epochs, dtype=float))


@pytest.mark.unit
def test_render_surrogate_training_best_epoch_marker_x_equals_best(ax: Axes) -> None:
    epochs = [1, 2, 4, 8, 16]
    train = [100.0 / e for e in epochs]
    best = 8.0

    no_marker = _recorder_with_surrogate_curve(
        with_val=False, with_best=False, epochs=epochs, train=train
    )
    dlga_viz.render(NEW_PLOT_NAME, ax, no_marker)
    base_lines = len(ax.lines)
    ax.clear()

    with_marker = _recorder_with_surrogate_curve(
        with_val=False, with_best=True, epochs=epochs, train=train, best_epoch=best
    )
    dlga_viz.render(NEW_PLOT_NAME, ax, with_marker)
    assert len(ax.lines) == base_lines + 1, (
        f"a present surrogate_best_epoch must add exactly one vertical marker; "
        f"base={base_lines}, with-marker={len(ax.lines)}."
    )

    marker_lines = [
        line
        for line in ax.lines
        if np.allclose(_line_xdata(line), best) and _line_xdata(line).size >= 2
    ]
    assert marker_lines, (
        f"a vertical marker with constant xdata == best_epoch ({best}) must be "
        f"present; line xdatas were "
        f"{[_line_xdata(line).tolist() for line in ax.lines]}."
    )

    assert ax.get_yscale() == "log"





@pytest.mark.unit
def test_render_surrogate_training_masks_nonfinite_and_nonpositive(ax: Axes) -> None:
    epochs = [1, 2, 3, 4, 5, 6, 7]
    bad = [None, float("nan"), float("inf"), 0.0, -5.0]
    train = [200.0, *bad, 50.0]
    val = [9.0, *bad, 2.0]
    masked_idx = {1, 2, 3, 4, 5}
    good_idx = {0, 6}
    recorder = _recorder_with_surrogate_curve(
        with_val=True,
        with_best=False,
        epochs=epochs,
        train=train,
        val=val,
    )
    try:
        dlga_viz.render(NEW_PLOT_NAME, ax, recorder)
    except Exception as exc:
        pytest.fail(
            f"render must mask non-finite / non-positive loss on a log axis, "
            f"not raise; got {type(exc).__name__}: {exc!r}."
        )
    assert ax.get_yscale() == "log"

    train_line = _find_line_in_band(ax, 10.0, 1000.0)
    val_line = _find_line_in_band(ax, 0.0, 10.0)
    assert train_line is not None and val_line is not None, (
        "both train and val curves must still be drawn (with masked gaps)."
    )
    for label, line, good_value in (
        ("train", train_line, train),
        ("val", val_line, val),
    ):
        y = _line_ydata(line)
        assert len(y) == len(epochs), (
            f"{label}: rendered ydata length must match the epoch axis "
            f"({len(epochs)}); got {len(y)}."
        )

        assert not np.isinf(y).any(), (
            f"{label}: no raw +inf may reach matplotlib; got {y.tolist()!r}."
        )
        for i in masked_idx:
            assert math.isnan(y[i]), (
                f"{label} index {i} ({good_value[i]!r}) must be masked to nan on "
                f"the log axis; got {y[i]!r}."
            )
        for i in good_idx:
            assert y[i] == pytest.approx(good_value[i]), (
                f"{label} good index {i} must survive masking verbatim; expected "
                f"{good_value[i]!r}, got {y[i]!r}."
            )


@pytest.mark.unit
def test_render_surrogate_training_all_nonfinite_series(ax: Axes) -> None:
    epochs = [1, 2, 3]
    train = [float("inf"), 0.0, float("nan")]
    recorder = _recorder_with_surrogate_curve(
        with_val=False,
        with_best=False,
        epochs=epochs,
        train=train,
    )
    try:
        dlga_viz.render(NEW_PLOT_NAME, ax, recorder)
    except Exception as exc:
        pytest.fail(
            f"render of an all-non-finite series must not raise; got "
            f"{type(exc).__name__}: {exc!r}."
        )

    for line in ax.lines:
        y = _line_ydata(line)
        assert not np.isinf(y).any(), (
            f"all-non-finite series must not leak raw inf; got {y.tolist()!r}."
        )


@pytest.mark.unit
def test_render_surrogate_training_survives_recorder_roundtrip(ax: Axes) -> None:
    epochs = [1, 2, 3, 4]
    train = [100.0, float("inf"), 25.0, 12.5]
    recorder = _recorder_with_surrogate_curve(
        with_val=False, with_best=True, epochs=epochs, train=train, best_epoch=3.0
    )
    restored = VizRecorder.from_dict(recorder.to_dict())
    try:
        dlga_viz.render(NEW_PLOT_NAME, ax, restored)
    except Exception as exc:
        pytest.fail(
            f"render after to_dict/from_dict must not crash; got "
            f"{type(exc).__name__}: {exc!r}."
        )
    assert ax.get_yscale() == "log"





@pytest.mark.unit
@pytest.mark.parametrize(
    "recorder_state", ["empty", "none", "disabled"], ids=["empty", "none", "disabled"]
)
def test_render_surrogate_training_empty_recorder_no_data_panel(
    recorder_state: str, ax: Axes
) -> None:
    recorder: VizRecorder | None = {
        "empty": VizRecorder(enabled=True),
        "disabled": VizRecorder(enabled=False),
        "none": None,
    }[recorder_state]
    try:
        dlga_viz.render(NEW_PLOT_NAME, ax, recorder)
    except Exception as exc:
        pytest.fail(
            f"render({NEW_PLOT_NAME!r}) with {recorder_state} recorder must not "
            f"raise; got {type(exc).__name__}: {exc!r}."
        )
    text_strs = [t.get_text().lower() for t in ax.texts]
    title = ax.get_title().lower()
    signals = ("no data", "empty", "unavailable", "no recorder")
    has_warning = any(any(sig in s for sig in signals) for s in [*text_strs, title])
    assert has_warning, (
        f"{recorder_state} recorder must surface a 'No data' panel; "
        f"texts={text_strs!r}, title={title!r}."
    )
    assert len(ax.lines) == 0, (
        f"no-data panel must not draw a phantom line; got {len(ax.lines)}."
    )


@pytest.mark.unit
def test_render_unknown_name_lists_four_available(ax: Axes) -> None:
    recorder = _recorder_with_surrogate_curve()
    with pytest.raises(ValueError) as exc_info:
        dlga_viz.render("nonexistent", ax, recorder)
    msg = str(exc_info.value)
    for name in (*EXISTING_PLOT_NAMES, NEW_PLOT_NAME):
        assert name in msg, f"unknown-name ValueError must list {name!r}; got {msg!r}."


@pytest.mark.unit
def test_get_data_unknown_name_lists_four_available() -> None:
    recorder = _recorder_with_surrogate_curve()
    with pytest.raises(ValueError) as exc_info:
        dlga_viz.get_data("nonexistent", recorder)
    msg = str(exc_info.value)
    for name in (*EXISTING_PLOT_NAMES, NEW_PLOT_NAME):
        assert name in msg, (
            f"get_data unknown-name ValueError must list {name!r}; got {msg!r}."
        )





@pytest.mark.unit
def test_get_data_surrogate_training_schema_with_val() -> None:
    epochs = [1, 7, 23, 99, 250]
    train = [100.0 / e for e in epochs]
    val = [10.0 / e for e in epochs]
    recorder = _recorder_with_surrogate_curve(
        with_val=True,
        with_best=True,
        epochs=epochs,
        train=train,
        val=val,
        best_epoch=23.0,
    )
    data = dlga_viz.get_data(NEW_PLOT_NAME, recorder)


    assert set(data.keys()) == {
        "x",
        "y_train",
        "y_val",
        "best_epoch",
        "xlabel",
        "ylabel",
        "title",
    }, f"get_data({NEW_PLOT_NAME!r}) key set wrong; got {sorted(data.keys())}."
    json.dumps(data)

    x = data["x"]
    y_train = data["y_train"]
    y_val = data["y_val"]

    assert list(x) == epochs, (
        f"get_data['x'] must equal the logged surrogate_epoch values {epochs!r} "
        f"(not range(len(y))); got {list(x)!r}."
    )
    assert list(x) != list(range(len(y_train))), (
        "get_data['x'] must NOT be a positional range — the explicit epoch axis "
        "is the whole point of logging surrogate_epoch."
    )
    np.testing.assert_allclose(np.asarray(y_train), np.asarray(train))
    assert y_val is not None and len(y_val) == len(x), (
        "y_val must be a list aligned with x when the val key is present"
    )
    np.testing.assert_allclose(np.asarray(y_val), np.asarray(val))

    for t, v in zip(y_train, y_val, strict=True):
        assert t > v, (
            f"y_train must stay above y_val (swap guard); got train={t!r} val={v!r}."
        )
    assert data["best_epoch"] == pytest.approx(23.0)


@pytest.mark.unit
def test_get_data_surrogate_training_no_val_is_none() -> None:
    recorder = _recorder_with_surrogate_curve(with_val=False, with_best=False)
    data = dlga_viz.get_data(NEW_PLOT_NAME, recorder)
    assert data["y_val"] is None, (
        f"y_val must be None when surrogate_val_loss is absent; got {data['y_val']!r}."
    )
    assert data["best_epoch"] is None, (
        f"best_epoch must be None when surrogate_best_epoch is absent; got "
        f"{data['best_epoch']!r}."
    )
    json.dumps(data)


@pytest.mark.unit
def test_get_data_surrogate_training_nonfinite_becomes_none() -> None:
    epochs = [1, 2, 3, 4]

    train = [100.0, float("inf"), float("nan"), 12.5]
    val = [10.0, 5.0, float("inf"), float("nan")]
    recorder = _recorder_with_surrogate_curve(
        with_val=True, with_best=False, epochs=epochs, train=train, val=val
    )
    data = dlga_viz.get_data(NEW_PLOT_NAME, recorder)


    assert data["y_train"][1] is None, (
        f"+inf train sample must serialize to None; got {data['y_train'][1]!r}."
    )
    assert data["y_train"][2] is None, (
        f"NaN train sample must serialize to None; got {data['y_train'][2]!r}."
    )
    assert data["y_train"][0] == pytest.approx(100.0)
    assert data["y_train"][3] == pytest.approx(12.5)


    assert data["y_val"] is not None
    assert data["y_val"][2] is None, (
        f"+inf val sample must serialize to None; got {data['y_val'][2]!r}."
    )
    assert data["y_val"][3] is None, (
        f"NaN val sample must serialize to None; got {data['y_val'][3]!r}."
    )
    assert data["y_val"][0] == pytest.approx(10.0)
    assert data["y_val"][1] == pytest.approx(5.0)

    json.dumps(data, allow_nan=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    "recorder_state", ["empty", "none", "disabled"], ids=["empty", "none", "disabled"]
)
def test_get_data_surrogate_training_empty_recorder_jsonable(
    recorder_state: str,
) -> None:
    recorder: VizRecorder | None = {
        "empty": VizRecorder(enabled=True),
        "disabled": VizRecorder(enabled=False),
        "none": None,
    }[recorder_state]
    data = dlga_viz.get_data(NEW_PLOT_NAME, recorder)
    assert isinstance(data, dict)
    assert list(data["x"]) == [], (
        f"{recorder_state}: x must be empty; got {data['x']!r}"
    )
    assert list(data["y_train"]) == [], (
        f"{recorder_state}: y_train must be empty; got {data['y_train']!r}"
    )
    assert data["y_val"] is None
    assert data["best_epoch"] is None
    json.dumps(data)







@pytest.mark.unit
@pytest.mark.parametrize("plot_name", EXISTING_PLOT_NAMES)
def test_existing_plots_keep_xy_schema(plot_name: str) -> None:
    recorder = VizRecorder(enabled=True)
    for i in range(4):
        recorder.log("gen_mean_fitness", float(100.0 - 10.0 * i))
        recorder.log("n_unique", 20 - i)
        recorder.log("gen_mean_complexity", 1.0 + 0.5 * i)
    data = dlga_viz.get_data(plot_name, recorder)
    assert set(data.keys()) == {"x", "y", "xlabel", "ylabel", "title"}, (
        f"existing plot {plot_name!r} must keep its {{x, y, xlabel, ylabel, "
        f"title}} schema; got {sorted(data.keys())}."
    )
