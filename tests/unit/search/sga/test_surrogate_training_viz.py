
from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.axes import Axes

from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.models.field_model import FieldModel




from kd.search.surrogate_log import (
    SURROGATE_BEST_EPOCH_KEY,
    SURROGATE_EARLY_STOPPED_KEY,
    SURROGATE_EPOCH_KEY,
    SURROGATE_EPOCHS_RUN_KEY,
    SURROGATE_TRAIN_LOSS_KEY,
    SURROGATE_VAL_LOSS_KEY,
)
from kd.search.surrogate_log import (
    SURROGATE_METRICS as _DLGA_SURROGATE_METRICS,
)
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder
from kd.search.sga import viz as sga_viz
from kd.search.sga.config import SGAConfig
from kd.search.sga.plugin import SGAPlugin

_DTYPE = torch.float64
_NX = 8
_NT = 5


_EXISTING_PLOT_NAMES = (
    "population_diversity",
    "complexity_evolution",
    "fitness_spread",
)
_NEW_PLOT_NAME = "surrogate_training"



_SURROGATE_KEYS_ALWAYS: frozenset[str] = frozenset(
    {
        SURROGATE_EPOCH_KEY,
        SURROGATE_TRAIN_LOSS_KEY,
        SURROGATE_EPOCHS_RUN_KEY,
        SURROGATE_EARLY_STOPPED_KEY,
    }
)
_SURROGATE_KEYS_CONDITIONAL: frozenset[str] = frozenset(
    {SURROGATE_VAL_LOSS_KEY, SURROGATE_BEST_EPOCH_KEY}
)
_SURROGATE_KEYS_ALL: frozenset[str] = (
    _SURROGATE_KEYS_ALWAYS | _SURROGATE_KEYS_CONDITIONAL
)


@pytest.fixture
def ax() -> Iterator[Axes]:
    fig, axes = plt.subplots()
    try:
        yield axes
    finally:
        plt.close(fig)


def _lazy_surrogate_metrics() -> tuple[str, ...]:
    from kd.search.sga import plugin as sga_plugin

    metrics = getattr(sga_plugin, "_SURROGATE_METRICS", None)
    assert metrics is not None, "sga/plugin.py must define _SURROGATE_METRICS"
    assert isinstance(metrics, tuple), (
        f"_SURROGATE_METRICS must be a tuple[str, ...], got {type(metrics).__name__}"
    )
    return tuple(str(name) for name in metrics)


def _recorder_with_surrogate_curve(
    *,
    with_val: bool = True,
    with_best: bool = True,
    epochs: list[int] | None = None,
    train: list[float] | None = None,
    val: list[float] | None = None,
    best_epoch: float | None = None,
) -> VizRecorder:
    recorder = VizRecorder(enabled=True)
    epochs = epochs if epochs is not None else [1, 2, 4, 8, 16]
    train = train if train is not None else [200.0 / e for e in epochs]
    recorder.log(SURROGATE_EPOCH_KEY, list(epochs))
    recorder.log(SURROGATE_TRAIN_LOSS_KEY, list(train))
    if with_val:
        val = val if val is not None else [2.0 / e for e in epochs]
        recorder.log(SURROGATE_VAL_LOSS_KEY, list(val))
    if with_best:
        marker = (
            best_epoch if best_epoch is not None else float(epochs[len(epochs) // 2])
        )
        recorder.log(SURROGATE_BEST_EPOCH_KEY, marker)
    recorder.log(SURROGATE_EPOCHS_RUN_KEY, float(epochs[-1]))
    recorder.log(SURROGATE_EARLY_STOPPED_KEY, 0.0)
    return recorder







def _tiny_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 6.0, _NX, dtype=_DTYPE)
    t = torch.linspace(0.0, 1.0, _NT, dtype=_DTYPE)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(gx) * torch.exp(-gt)
    return PDEDataset(
        name="sga-surrogate-viz",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _real_components(
    dataset: PDEDataset, recorder: VizRecorder | None
) -> PlatformComponents:
    from kd.core.evaluator import Evaluator
    from kd.core.executor.context import ExecutionContext
    from kd.core.expr import FunctionRegistry, PythonExecutor
    from kd.core.linear_solve.least_squares import LeastSquaresSolver
    from kd.data.derivatives.finite_diff import FiniteDiffProvider

    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    registry = FunctionRegistry.create_default()
    executor = PythonExecutor(registry)
    solver = LeastSquaresSolver()
    u_t = provider.get_derivative("u", "t", order=1).flatten()
    evaluator = Evaluator(executor=executor, solver=solver, context=context, lhs=u_t)
    return PlatformComponents(
        dataset=dataset,
        executor=executor,
        evaluator=evaluator,
        context=context,
        registry=registry,
        recorder=recorder,
    )


def _autograd_config(**overrides: Any) -> SGAConfig:
    base: dict[str, Any] = {
        "num": 4,
        "depth": 2,
        "width": 2,
        "seed": 42,
        "maxit": 2,
        "use_autograd": True,
        "autograd_train_epochs": 8,
    }
    base.update(overrides)
    return SGAConfig(**base)


def _pretrained_model(dataset: PDEDataset) -> FieldModel:



    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = FieldModel(
            coord_names=list(dataset.axes.keys()),
            field_names=list(dataset.fields.keys()),
            hidden_sizes=[8, 8],
        ).to(dtype=_DTYPE)
    model.eval()
    return model







@pytest.mark.unit
def test_surrogate_metrics_constant_equals_dlga_six_key_whitelist() -> None:
    metrics = _lazy_surrogate_metrics()
    assert len(metrics) == len(set(metrics)), (
        f"_SURROGATE_METRICS has duplicate entries: {metrics}"
    )
    assert set(metrics) == set(_DLGA_SURROGATE_METRICS), (
        f"_SURROGATE_METRICS must equal the DLGA surrogate whitelist "
        f"(逐字相同).\n missing: {sorted(set(_DLGA_SURROGATE_METRICS) - set(metrics))}"
        f"\n extra: {sorted(set(metrics) - set(_DLGA_SURROGATE_METRICS))}"
    )
    assert set(metrics) == _SURROGATE_KEYS_ALL, (
        "the local key constants must match _SURROGATE_METRICS exactly."
    )


@pytest.mark.unit
def test_surrogate_metrics_disjoint_from_logged_metrics() -> None:
    from kd.search.sga import plugin as sga_plugin

    gen_metrics = set(sga_plugin._LOGGED_METRICS)
    surrogate = set(_lazy_surrogate_metrics())
    overlap = gen_metrics & surrogate
    assert not overlap, (
        f"_SURROGATE_METRICS must be disjoint from _LOGGED_METRICS; "
        f"overlap={sorted(overlap)}."
    )


@pytest.mark.unit
def test_surrogate_metric_names_no_underscore_prefix() -> None:
    for name in _lazy_surrogate_metrics():
        assert not name.startswith("_"), (
            f"surrogate metric {name!r} must not start with '_' (reserved for "
            "platform-written series)."
        )







@pytest.mark.unit
def test_list_plot_infos_returns_four_with_surrogate_last() -> None:
    infos = sga_viz.list_plot_infos()
    assert len(infos) == 4, (
        f"list_plot_infos() must return 4 PlotInfo (3 existing + surrogate); "
        f"got {len(infos)}: {[getattr(p, 'name', '?') for p in infos]}."
    )
    names = [p.name for p in infos]
    assert names[:3] == list(_EXISTING_PLOT_NAMES), (
        f"the 3 existing plots must come first in order; got {names!r}."
    )
    assert names[-1] == _NEW_PLOT_NAME, (
        f"the new plot must be LAST and named {_NEW_PLOT_NAME!r}; got {names!r}."
    )


@pytest.mark.unit
def test_plugin_list_plots_returns_five() -> None:
    plugin = SGAPlugin(SGAConfig())
    plots = plugin.list_plots()
    names = [getattr(p, "name", "?") for p in plots]
    assert len(plots) == 5, (
        f"SGAPlugin.list_plots() must return 5 plots (viz's 4 + genome_tree); "
        f"got {len(plots)}: {names}."
    )
    assert _NEW_PLOT_NAME in set(names)
    assert names[-1] == "genome_tree", (
        f"genome_tree must be appended last (after viz's 4); got {names!r}."
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
def test_render_surrogate_training_returns_empty_warnings(ax: Axes) -> None:
    recorder = _recorder_with_surrogate_curve()
    result = sga_viz.render(_NEW_PLOT_NAME, ax, recorder)
    assert result == [], f"clean render must return no warnings; got {result!r}."


@pytest.mark.unit
def test_render_surrogate_training_is_log_scale(ax: Axes) -> None:
    recorder = _recorder_with_surrogate_curve()
    sga_viz.render(_NEW_PLOT_NAME, ax, recorder)
    assert ax.get_yscale() == "log", (
        f"surrogate_training y-axis must be log scale; got {ax.get_yscale()!r}."
    )


@pytest.mark.unit
def test_render_surrogate_training_two_lines_bound_to_recorder(ax: Axes) -> None:
    epochs = [1, 2, 4, 8, 16]
    train = [200.0 / e for e in epochs]
    val = [2.0 / e for e in epochs]
    recorder = _recorder_with_surrogate_curve(
        with_val=True, with_best=False, epochs=epochs, train=train, val=val
    )
    sga_viz.render(_NEW_PLOT_NAME, ax, recorder)

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
def test_render_surrogate_training_one_line_when_val_absent(ax: Axes) -> None:
    epochs = [1, 2, 4, 8, 16]
    train = [100.0 / e for e in epochs]
    recorder = _recorder_with_surrogate_curve(
        with_val=False, with_best=False, epochs=epochs, train=train
    )
    sga_viz.render(_NEW_PLOT_NAME, ax, recorder)
    assert len(ax.lines) == 1, (
        f"val absent must draw exactly one line; got {len(ax.lines)}."
    )
    np.testing.assert_allclose(_line_ydata(ax.lines[0]), np.asarray(train))
    np.testing.assert_allclose(
        _line_xdata(ax.lines[0]), np.asarray(epochs, dtype=float)
    )


@pytest.mark.unit
def test_render_surrogate_training_best_epoch_marker_x_equals_best(ax: Axes) -> None:
    epochs = [1, 2, 4, 8, 16]
    best = 4.0
    recorder = _recorder_with_surrogate_curve(
        with_val=False, with_best=False, epochs=epochs
    )
    recorder.log(SURROGATE_BEST_EPOCH_KEY, best)
    sga_viz.render(_NEW_PLOT_NAME, ax, recorder)

    marker_xs = {
        float(_line_xdata(line)[0])
        for line in ax.lines
        if _line_xdata(line).size and np.all(_line_xdata(line) == _line_xdata(line)[0])
    }
    assert best in marker_xs, (
        f"a best-epoch vertical marker at x={best} must be drawn; constant-x "
        f"lines were at {sorted(marker_xs)}."
    )


@pytest.mark.numerical
def test_render_surrogate_training_masks_nonfinite_and_nonpositive(ax: Axes) -> None:
    epochs = [1, 2, 3, 4]
    train = [100.0, float("inf"), -5.0, 25.0]
    recorder = _recorder_with_surrogate_curve(
        with_val=False, with_best=False, epochs=epochs, train=train
    )
    sga_viz.render(_NEW_PLOT_NAME, ax, recorder)
    assert ax.lines, "a train line must still be drawn"
    ydata = _line_ydata(ax.lines[0])

    assert math.isnan(ydata[1]), "the +inf sample must be masked to nan"
    assert math.isnan(ydata[2]), "the non-positive sample must be masked to nan"
    np.testing.assert_allclose([ydata[0], ydata[3]], [100.0, 25.0])



    assert [bool(np.isnan(v)) for v in ydata] == [False, True, True, False], (
        f"only the +inf and non-positive samples may be masked; got nan-mask "
        f"{[bool(np.isnan(v)) for v in ydata]}"
    )


    np.testing.assert_allclose(
        _line_xdata(ax.lines[0]), np.asarray(epochs, dtype=float)
    )







@pytest.mark.unit
@pytest.mark.parametrize("recorder_state", ["empty", "disabled", "none"])
def test_render_surrogate_training_absent_curve_no_data_panel(
    recorder_state: str, ax: Axes
) -> None:
    recorder: VizRecorder | None = {
        "empty": VizRecorder(enabled=True),
        "disabled": VizRecorder(enabled=False),
        "none": None,
    }[recorder_state]
    try:
        sga_viz.render(_NEW_PLOT_NAME, ax, recorder)
    except Exception as exc:
        pytest.fail(
            f"render({_NEW_PLOT_NAME!r}) with {recorder_state} recorder must not "
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
def test_render_surrogate_training_length_mismatch_does_not_raise(ax: Axes) -> None:
    recorder = VizRecorder(enabled=True)

    recorder.log(SURROGATE_EPOCH_KEY, [1, 2, 3, 4, 5])
    recorder.log(SURROGATE_TRAIN_LOSS_KEY, [100.0, 50.0, 25.0])
    recorder.log(SURROGATE_EPOCHS_RUN_KEY, 5.0)
    recorder.log(SURROGATE_EARLY_STOPPED_KEY, 0.0)
    try:
        result = sga_viz.render(_NEW_PLOT_NAME, ax, recorder)
    except Exception as exc:
        pytest.fail(
            f"render must not raise on a length-drifted surrogate recorder; got "
            f"{type(exc).__name__}: {exc!r}."
        )


    assert any("drifted recorder" in note for note in result), result


@pytest.mark.unit
def test_render_unknown_name_lists_four_available(ax: Axes) -> None:
    recorder = _recorder_with_surrogate_curve()
    with pytest.raises(ValueError) as exc_info:
        sga_viz.render("nonexistent", ax, recorder)
    msg = str(exc_info.value)
    for name in (*_EXISTING_PLOT_NAMES, _NEW_PLOT_NAME):
        assert name in msg, f"unknown-name ValueError must list {name!r}; got {msg!r}."


@pytest.mark.unit
def test_get_data_surrogate_training_schema_with_val() -> None:
    import json

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
    data = sga_viz.get_data(_NEW_PLOT_NAME, recorder)


    for key in ("x", "y_train", "y_val", "best_epoch"):
        assert key in data, (
            f"get_data({_NEW_PLOT_NAME!r}) must expose {key!r} (DLGA surrogate "
            f"schema); got {sorted(data.keys())}."
        )


    json.dumps(data, allow_nan=False)


    assert list(data["x"]) == epochs, (
        f"get_data['x'] must equal the logged surrogate_epoch values {epochs!r} "
        f"(not range(len(y))); got {list(data['x'])!r}."
    )
    assert list(data["x"]) != list(range(len(train))), (
        "get_data['x'] must NOT be a positional range — the explicit epoch axis "
        "is the whole point of logging surrogate_epoch."
    )

    np.testing.assert_allclose(np.asarray(data["y_train"]), np.asarray(train))
    assert data["y_val"] is not None and len(data["y_val"]) == len(data["x"]), (
        "y_val must be a list aligned with x when the val key is present"
    )
    np.testing.assert_allclose(np.asarray(data["y_val"]), np.asarray(val))
    for t, v in zip(data["y_train"], data["y_val"], strict=True):
        assert t > v, f"y_train must stay above y_val (swap guard); train={t} val={v}"
    assert data["best_epoch"] == pytest.approx(23.0)


@pytest.mark.unit
def test_get_data_surrogate_training_no_val_is_none() -> None:
    import json

    recorder = _recorder_with_surrogate_curve(with_val=False, with_best=False)
    data = sga_viz.get_data(_NEW_PLOT_NAME, recorder)
    assert data["y_val"] is None, (
        f"y_val must be None when surrogate_val_loss is absent; got {data['y_val']!r}."
    )
    assert data["best_epoch"] is None, (
        f"best_epoch must be None when surrogate_best_epoch is absent; got "
        f"{data['best_epoch']!r}."
    )
    json.dumps(data, allow_nan=False)







def _last_logged_list(recorder: VizRecorder, key: str) -> list[Any]:
    series = recorder.get(key)
    assert series, f"recorder series {key!r} is empty (the metric was not logged)"
    inner = series[-1]
    assert isinstance(inner, list), f"{key!r} payload must be a list; got {inner!r}"
    return inner


def _last_logged_scalar(recorder: VizRecorder, key: str) -> float:
    series = recorder.get(key)
    assert series, f"recorder series {key!r} is empty (the scalar was not logged)"
    return float(series[-1])


@pytest.mark.unit
def test_autograd_prepare_logs_surrogate_curve_with_epochs_run_length() -> None:
    n_epochs = 8
    dataset = _tiny_dataset()
    recorder = VizRecorder(enabled=True)
    plugin = SGAPlugin(_autograd_config(autograd_train_epochs=n_epochs))
    plugin.prepare(_real_components(dataset, recorder))

    keys = set(recorder.keys())
    missing = _SURROGATE_KEYS_ALWAYS - keys
    assert not missing, (
        f"a trained autograd prepare() must log the always-present surrogate "
        f"series; missing {sorted(missing)} (the plugin still discards the "
        f"TrainingResult)."
    )

    curve = _last_logged_list(recorder, SURROGATE_TRAIN_LOSS_KEY)
    assert len(curve) == n_epochs, (
        f"the train curve length must equal epochs_run ({n_epochs}); got "
        f"{len(curve)} (no early stop / downsample expected for {n_epochs} "
        f"epochs)."
    )


    epochs = _last_logged_list(recorder, SURROGATE_EPOCH_KEY)
    assert epochs == list(range(1, n_epochs + 1)), (
        f"surrogate_epoch must be 1..epochs_run ({list(range(1, n_epochs + 1))}); "
        f"got {epochs!r}."
    )
    assert len(epochs) == len(curve), "epoch axis must align 1:1 with the train series"

    assert _last_logged_scalar(recorder, SURROGATE_EPOCHS_RUN_KEY) == float(n_epochs)


@pytest.mark.unit
def test_autograd_prepare_logs_val_curve_when_val_ratio_set() -> None:
    n_epochs = 8
    dataset = _tiny_dataset()
    recorder = VizRecorder(enabled=True)
    plugin = SGAPlugin(
        _autograd_config(
            autograd_train_epochs=n_epochs,
            autograd_train_val_ratio=0.25,
            autograd_train_patience=None,
        )
    )
    plugin.prepare(_real_components(dataset, recorder))

    keys = set(recorder.keys())
    assert SURROGATE_VAL_LOSS_KEY in keys, (
        "a val split (val_ratio=0.25) must log surrogate_val_loss; missing it "
        f"(keys={sorted(k for k in keys if k.startswith('surrogate'))})."
    )
    assert SURROGATE_BEST_EPOCH_KEY in keys, (
        "a val split must log surrogate_best_epoch (the best-validate epoch)."
    )
    train = _last_logged_list(recorder, SURROGATE_TRAIN_LOSS_KEY)
    val = _last_logged_list(recorder, SURROGATE_VAL_LOSS_KEY)
    assert len(val) == len(train), (
        f"surrogate_val_loss length ({len(val)}) must match surrogate_train_loss "
        f"length ({len(train)})."
    )
    best = _last_logged_scalar(recorder, SURROGATE_BEST_EPOCH_KEY)
    assert 1 <= best <= n_epochs, (
        f"surrogate_best_epoch must be a 1..epochs_run epoch number; got {best}."
    )


@pytest.mark.unit
def test_autograd_prepare_logs_surrogate_curve_once() -> None:
    dataset = _tiny_dataset()
    recorder = VizRecorder(enabled=True)
    plugin = SGAPlugin(_autograd_config(autograd_train_epochs=6))
    plugin.prepare(_real_components(dataset, recorder))


    candidates = plugin.propose(plugin._config.num)
    results = plugin.evaluate(candidates)
    plugin.update(results)

    train_log = recorder.get(SURROGATE_TRAIN_LOSS_KEY)
    assert len(train_log) == 1, (
        f"surrogate_train_loss must be logged exactly once (one-shot in "
        f"prepare); got {len(train_log)} log entries."
    )


@pytest.mark.unit
def test_surrogate_curve_is_complete_after_prepare_and_stable_across_updates() -> None:
    n_epochs = 6
    dataset = _tiny_dataset()
    recorder = VizRecorder(enabled=True)
    plugin = SGAPlugin(_autograd_config(autograd_train_epochs=n_epochs))
    plugin.prepare(_real_components(dataset, recorder))


    after_prepare_train = list(_last_logged_list(recorder, SURROGATE_TRAIN_LOSS_KEY))
    after_prepare_epoch = list(_last_logged_list(recorder, SURROGATE_EPOCH_KEY))
    assert len(after_prepare_train) == n_epochs, (
        "the curve must be COMPLETE after prepare(), before any update()."
    )

    candidates = plugin.propose(plugin._config.num)
    results = plugin.evaluate(candidates)
    plugin.update(results)


    assert (
        _last_logged_list(recorder, SURROGATE_TRAIN_LOSS_KEY) == after_prepare_train
    ), (
        "the surrogate train curve must NOT change across update() (logging "
        "belongs to prepare, not update)."
    )
    assert _last_logged_list(recorder, SURROGATE_EPOCH_KEY) == after_prepare_epoch, (
        "the surrogate epoch axis must NOT change across update()."
    )


@pytest.mark.unit
def test_full_autograd_flow_logs_exactly_the_union_whitelist() -> None:
    from kd.search.sga.plugin import _LOGGED_METRICS

    dataset = _tiny_dataset()
    recorder = VizRecorder(enabled=True)
    plugin = SGAPlugin(_autograd_config(autograd_train_epochs=6))
    plugin.prepare(_real_components(dataset, recorder))
    candidates = plugin.propose(plugin._config.num)
    results = plugin.evaluate(candidates)
    plugin.update(results)

    expected = set(_LOGGED_METRICS) | {"best_aic"} | _SURROGATE_KEYS_ALWAYS
    actual = set(recorder.keys())
    assert actual == expected, (
        f"a full autograd flow (val_ratio=0) must log EXACTLY the per-gen "
        f"whitelist + best_aic + the 4 always-surrogate keys.\n"
        f" missing: {sorted(expected - actual)}\n"
        f" extra: {sorted(actual - expected)}"
    )


@pytest.mark.unit
def test_fd_prepare_logs_no_surrogate_keys() -> None:
    dataset = _tiny_dataset()
    recorder = VizRecorder(enabled=True)
    plugin = SGAPlugin(SGAConfig(num=4, depth=2, width=2, seed=42, maxit=2))
    plugin.prepare(_real_components(dataset, recorder))

    present = _SURROGATE_KEYS_ALL & set(recorder.keys())
    assert not present, (
        f"FD mode trains no surrogate, so no surrogate key may appear; found "
        f"{sorted(present)}."
    )


@pytest.mark.unit
def test_pretrained_field_model_prepare_logs_no_surrogate_keys() -> None:
    dataset = _tiny_dataset()
    recorder = VizRecorder(enabled=True)
    model = _pretrained_model(dataset)
    plugin = SGAPlugin(_autograd_config(field_model=model))
    plugin.prepare(_real_components(dataset, recorder))

    present = _SURROGATE_KEYS_ALL & set(recorder.keys())
    assert not present, (
        f"a pre-trained field_model has no training history; no surrogate key "
        f"may appear, found {sorted(present)}."
    )









@pytest.mark.unit
def test_autograd_prepare_exposes_surrogate_train_seconds() -> None:
    dataset = _tiny_dataset()
    components = _real_components(dataset, VizRecorder(enabled=True))
    plugin = SGAPlugin(_autograd_config(autograd_train_epochs=4))
    plugin.prepare(components)

    seconds = plugin.surrogate_train_seconds
    assert isinstance(seconds, float), (
        "a trained autograd prepare() must expose numeric surrogate_train_seconds "
        f"(RunCost source); got {seconds!r}."
    )
    assert seconds >= 0.0


@pytest.mark.unit
def test_pretrained_field_model_prepare_surrogate_train_seconds_none() -> None:
    dataset = _tiny_dataset()
    components = _real_components(dataset, VizRecorder(enabled=True))
    model = _pretrained_model(dataset)
    plugin = SGAPlugin(_autograd_config(field_model=model))
    plugin.prepare(components)

    assert plugin.surrogate_train_seconds is None, (
        "a pre-trained field_model trains nothing this run; "
        "surrogate_train_seconds must stay None (honest invocation-local cost)."
    )


@pytest.mark.unit
def test_surrogate_train_seconds_is_plugin_local_not_shared_via_context() -> None:
    dataset = _tiny_dataset()
    components = _real_components(dataset, VizRecorder(enabled=True))


    trained = SGAPlugin(_autograd_config(autograd_train_epochs=4))
    trained.prepare(components)
    assert trained.surrogate_train_seconds is not None



    fd = SGAPlugin(SGAConfig(num=4, depth=2, width=2, seed=42, maxit=2))
    fd.prepare(components)
    assert fd.surrogate_train_seconds is None
