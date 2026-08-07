
from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.axes import Axes

from kd.core.evaluator import EvaluationResult
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.plugin import DISCOVERPlugin
from kd.search.protocol import PlatformComponents
from kd.search.recorder import VizRecorder
from kd.viz.extension import PlotInfo, VizExtension





SEED = 42



EXPECTED_PLOT_NAMES: frozenset[str] = frozenset(
    {"reward_convergence", "reward_full_mean", "entropy_loss_decay", "baseline_ewma"},
)




PLOT_TO_METRIC: dict[str, str] = {
    "reward_convergence": "reward",
    "reward_full_mean": "reward_full",
    "entropy_loss_decay": "entropy_loss",
    "baseline_ewma": "baseline",
}


N_SYNTHETIC_ITERS = 5







class _MockEvaluator:

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        return EvaluationResult(
            mse=0.5,
            nmse=0.5,
            r2=0.5,
            complexity=3,
            is_valid=True,
            expression=expr,
        )







def _make_components(recorder: VizRecorder | None) -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=_MockEvaluator(),
        context=MagicMock(training_result=None),
        registry=MagicMock(),
        recorder=recorder,
    )


def _populate_recorder(recorder: VizRecorder, n_iters: int = N_SYNTHETIC_ITERS) -> None:
    for i in range(n_iters):

        recorder.log("reward", float(100.0 + 10.0 * i))
        recorder.log("reward_full", float(30.0 + 2.0 * i))
        recorder.log("entropy_loss", float(-2.0 + 0.5 * i))
        recorder.log("baseline", float(0.001 * (i + 1)))


        recorder.log("pg_loss", float(1.0 - 0.1 * i))
        recorder.log("total_loss", float(1.8 - 0.15 * i))
        recorder.log("grad_norm", float(2.0 - 0.1 * i))
        recorder.log("reward_max", float(0.6 + 0.1 * i))
        recorder.log("best_reward", float(0.5 + 0.05 * i))
        recorder.log("n_valid", 10 + i)
        recorder.log("n_eval_valid", 8 + i)
        recorder.log("n_invalid_in_topk", 2)
        recorder.log("n_unique", 6 + i)


def _make_plugin_with_recorder() -> tuple[DISCOVERPlugin, VizRecorder]:
    torch.manual_seed(SEED)
    recorder = VizRecorder(enabled=True)
    plugin = DISCOVERPlugin(DiscoverConfig())
    plugin.prepare(_make_components(recorder))
    _populate_recorder(recorder, n_iters=N_SYNTHETIC_ITERS)
    return plugin, recorder


def _make_plugin_empty_recorder() -> DISCOVERPlugin:
    torch.manual_seed(SEED)
    recorder = VizRecorder(enabled=True)
    plugin = DISCOVERPlugin(DiscoverConfig())
    plugin.prepare(_make_components(recorder))
    return plugin


def _make_plugin_disabled_recorder() -> DISCOVERPlugin:
    torch.manual_seed(SEED)
    recorder = VizRecorder(enabled=False)
    plugin = DISCOVERPlugin(DiscoverConfig())
    plugin.prepare(_make_components(recorder))
    return plugin


def _make_plugin_none_recorder() -> DISCOVERPlugin:
    torch.manual_seed(SEED)
    plugin = DISCOVERPlugin(DiscoverConfig())
    plugin.prepare(_make_components(None))
    return plugin







@pytest.fixture
def ax() -> Iterator[Axes]:
    _fig, _ax = plt.subplots()
    yield _ax
    plt.close(_fig)







@pytest.mark.unit
def test_plugin_is_runtime_checkable_viz_extension() -> None:
    plugin = DISCOVERPlugin(DiscoverConfig())
    assert isinstance(plugin, VizExtension), (
        "DISCOVERPlugin must satisfy the VizExtension Protocol structurally. "
        "Required methods: list_plots(), render_plot(name, ax), "
        "get_plot_data(name). See src/kd/viz/extension.py."
    )





    assert callable(plugin.list_plots), (
        "plugin.list_plots must be callable, not a plain attribute."
    )
    assert callable(plugin.render_plot), (
        "plugin.render_plot must be callable, not a plain attribute."
    )
    assert callable(plugin.get_plot_data), (
        "plugin.get_plot_data must be callable, not a plain attribute."
    )

    list_plots_params = list(inspect.signature(plugin.list_plots).parameters.keys())
    render_plot_params = list(inspect.signature(plugin.render_plot).parameters.keys())
    get_plot_data_params = list(
        inspect.signature(plugin.get_plot_data).parameters.keys()
    )

    assert list_plots_params == [], (
        f"list_plots() must take no parameters beyond self; "
        f"got parameters: {list_plots_params!r}"
    )
    assert render_plot_params == ["name", "ax"], (
        f"render_plot signature must be (name, ax) in that order; "
        f"got parameters: {render_plot_params!r}"
    )
    assert get_plot_data_params == ["name"], (
        f"get_plot_data signature must be (name,); "
        f"got parameters: {get_plot_data_params!r}"
    )







@pytest.mark.unit
def test_list_plots_returns_expected_plotinfo() -> None:
    plugin = DISCOVERPlugin(DiscoverConfig())
    plots = plugin.list_plots()

    assert isinstance(plots, list), (
        f"list_plots() must return list, got {type(plots).__name__}"
    )
    assert len(plots) == 4, (
        f"list_plots() must return exactly 4 PlotInfo descriptors; got {len(plots)}: "
        f"{[getattr(p, 'name', '?') for p in plots]}"
    )

    for p in plots:
        assert isinstance(p, PlotInfo), (
            f"Every entry must be a PlotInfo, got {type(p).__name__}"
        )
        assert isinstance(p.title, str) and p.title.strip(), (
            f"PlotInfo({p.name!r}).title must be a non-empty string, got {p.title!r}"
        )

        assert isinstance(p.description, str), (
            f"PlotInfo({p.name!r}).description must be str, "
            f"got {type(p.description).__name__}"
        )

    names = {p.name for p in plots}
    assert names == EXPECTED_PLOT_NAMES, (
        f"list_plots() names must equal the expected plot set.\n"
        f" missing: {sorted(EXPECTED_PLOT_NAMES - names)}\n"
        f" extra: {sorted(names - EXPECTED_PLOT_NAMES)}"
    )







@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(EXPECTED_PLOT_NAMES))
def test_render_plot_returns_empty_warnings(plot_name: str, ax: Axes) -> None:
    plugin, _ = _make_plugin_with_recorder()
    result = plugin.render_plot(plot_name, ax)
    assert result == [], (
        f"render_plot({plot_name!r}, ax) must return no warnings on a clean "
        f"render; got {type(result).__name__}: {result!r}"
    )







@pytest.mark.unit
def test_render_reward_convergence_writes_to_ax(ax: Axes) -> None:
    plugin, recorder = _make_plugin_with_recorder()
    plugin.render_plot("reward_convergence", ax)

    assert len(ax.lines) >= 1, (
        "reward_convergence must draw at least one line; got 0 Line2D artists."
    )




    y_data = np.asarray(ax.lines[0].get_ydata()).tolist()
    expected = recorder.get(PLOT_TO_METRIC["reward_convergence"])
    assert y_data == pytest.approx(expected), (
        f"reward_convergence must draw from recorder['reward']. "
        f"Got ydata={y_data!r}, expected={expected!r}."
    )
    assert len(y_data) == N_SYNTHETIC_ITERS, (
        f"reward_convergence must plot all {N_SYNTHETIC_ITERS} recorder "
        f"iterations; got {len(y_data)} points."
    )

    assert ax.get_ylabel() == "reward", (
        f"reward_convergence ylabel must be exactly 'reward', got {ax.get_ylabel()!r}"
    )
    xlabel = ax.get_xlabel().lower()
    assert xlabel.startswith("iter"), (
        f"reward_convergence xlabel must start with 'iter' (case-insensitive); "
        f"got {ax.get_xlabel()!r}"
    )


@pytest.mark.unit
def test_render_entropy_loss_decay_writes_to_ax(ax: Axes) -> None:
    plugin, recorder = _make_plugin_with_recorder()
    plugin.render_plot("entropy_loss_decay", ax)

    assert len(ax.lines) >= 1, (
        "entropy_loss_decay must draw at least one line; got 0 Line2D artists."
    )




    y_data = np.asarray(ax.lines[0].get_ydata()).tolist()
    expected = recorder.get(PLOT_TO_METRIC["entropy_loss_decay"])
    assert y_data == pytest.approx(expected), (
        f"entropy_loss_decay must draw from recorder['entropy_loss']. "
        f"Got ydata={y_data!r}, expected={expected!r}."
    )
    assert len(y_data) == N_SYNTHETIC_ITERS, (
        f"entropy_loss_decay must plot all {N_SYNTHETIC_ITERS} recorder "
        f"iterations; got {len(y_data)} points."
    )

    assert ax.get_ylabel() == "entropy_loss", (
        f"entropy_loss_decay ylabel must be exactly 'entropy_loss', "
        f"got {ax.get_ylabel()!r}"
    )


@pytest.mark.unit
def test_render_baseline_ewma_writes_to_ax(ax: Axes) -> None:
    plugin, recorder = _make_plugin_with_recorder()
    plugin.render_plot("baseline_ewma", ax)

    assert len(ax.lines) >= 1, (
        "baseline_ewma must draw at least one line; got 0 Line2D artists."
    )




    y_data = np.asarray(ax.lines[0].get_ydata()).tolist()
    expected = recorder.get(PLOT_TO_METRIC["baseline_ewma"])
    assert y_data == pytest.approx(expected), (
        f"baseline_ewma must draw from recorder['baseline']. "
        f"Got ydata={y_data!r}, expected={expected!r}."
    )
    assert len(y_data) == N_SYNTHETIC_ITERS, (
        f"baseline_ewma must plot all {N_SYNTHETIC_ITERS} recorder "
        f"iterations; got {len(y_data)} points."
    )

    assert ax.get_ylabel() == "baseline", (
        f"baseline_ewma ylabel must be exactly 'baseline', got {ax.get_ylabel()!r}"
    )







@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(EXPECTED_PLOT_NAMES))
def test_get_plot_data_returns_jsonable_dict(plot_name: str) -> None:
    plugin, recorder = _make_plugin_with_recorder()
    data = plugin.get_plot_data(plot_name)

    assert isinstance(data, dict), (
        f"get_plot_data({plot_name!r}) must return dict, got {type(data).__name__}"
    )

    required = {"x", "y", "xlabel", "ylabel"}
    missing = required - data.keys()
    assert not missing, (
        f"get_plot_data({plot_name!r}) dict missing required keys: "
        f"{sorted(missing)}. Got keys: {sorted(data.keys())}"
    )


    try:
        json.dumps(data)
    except (TypeError, ValueError) as exc:
        pytest.fail(
            f"get_plot_data({plot_name!r}) returned a non-JSON-safe dict: "
            f"{exc!r}. Values must be primitives / lists / dicts only "
            f"(no Tensor, ndarray, etc.)."
        )

    expected_metric = PLOT_TO_METRIC[plot_name]
    expected_y = recorder.get(expected_metric)
    assert list(data["y"]) == pytest.approx(expected_y), (
        f"get_plot_data({plot_name!r})['y'] must equal "
        f"recorder.get({expected_metric!r}). "
        f"Got y={list(data['y'])!r}, expected={expected_y!r}."
    )

    assert len(data["x"]) == N_SYNTHETIC_ITERS, (
        f"get_plot_data({plot_name!r})['x'] must have "
        f"{N_SYNTHETIC_ITERS} entries (one per recorded iteration); "
        f"got len={len(data['x'])}."
    )
    assert len(data["x"]) == len(data["y"]), (
        f"get_plot_data({plot_name!r}) x and y lengths must match; "
        f"got len(x)={len(data['x'])}, len(y)={len(data['y'])}"
    )

    assert data["ylabel"] == expected_metric, (
        f"get_plot_data({plot_name!r})['ylabel'] must be {expected_metric!r}, "
        f"got {data['ylabel']!r}"
    )







@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(EXPECTED_PLOT_NAMES))
@pytest.mark.parametrize(
    "recorder_state",
    ["empty", "none", "disabled"],
    ids=["empty_recorder", "none_recorder", "disabled_recorder"],
)
def test_get_plot_data_empty_recorder_returns_jsonable_dict(
    plot_name: str,
    recorder_state: str,
) -> None:
    if recorder_state == "empty":
        plugin = _make_plugin_empty_recorder()
    elif recorder_state == "disabled":
        plugin = _make_plugin_disabled_recorder()
    else:
        plugin = _make_plugin_none_recorder()

    data = plugin.get_plot_data(plot_name)

    assert isinstance(data, dict), (
        f"get_plot_data({plot_name!r}) with {recorder_state} recorder must "
        f"return dict, got {type(data).__name__}"
    )
    required = {"x", "y", "xlabel", "ylabel"}
    missing = required - data.keys()
    assert not missing, (
        f"get_plot_data({plot_name!r}) with {recorder_state} recorder is "
        f"missing required keys: {sorted(missing)}. "
        f"Got keys: {sorted(data.keys())}"
    )

    assert list(data["y"]) == [], (
        f"get_plot_data({plot_name!r}) with {recorder_state} recorder must "
        f"yield an empty y series; got {data['y']!r}"
    )
    assert list(data["x"]) == [], (
        f"get_plot_data({plot_name!r}) with {recorder_state} recorder must "
        f"yield an empty x series; got {data['x']!r}"
    )

    expected_metric = PLOT_TO_METRIC[plot_name]
    assert data["ylabel"] == expected_metric, (
        f"get_plot_data({plot_name!r}) with {recorder_state} recorder must "
        f"still report ylabel={expected_metric!r}; got {data['ylabel']!r}"
    )


    try:
        json.dumps(data)
    except (TypeError, ValueError) as exc:
        pytest.fail(
            f"get_plot_data({plot_name!r}) with {recorder_state} recorder "
            f"must return a JSON-safe dict; got {exc!r}"
        )







@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(EXPECTED_PLOT_NAMES))
@pytest.mark.parametrize(
    "recorder_state",
    ["empty", "none", "disabled"],
    ids=["empty_recorder", "none_recorder", "disabled_recorder"],
)
def test_render_plot_empty_recorder_does_not_crash(
    plot_name: str,
    recorder_state: str,
    ax: Axes,
) -> None:
    if recorder_state == "empty":
        plugin = _make_plugin_empty_recorder()
    elif recorder_state == "disabled":
        plugin = _make_plugin_disabled_recorder()
    else:
        plugin = _make_plugin_none_recorder()


    try:
        channel = plugin.render_plot(plot_name, ax)
    except Exception as exc:
        pytest.fail(
            f"render_plot({plot_name!r}) with {recorder_state} recorder must "
            f"not raise; got {type(exc).__name__}: {exc!r}"
        )

    text_strs = [t.get_text().lower() for t in ax.texts]
    title_lower = ax.get_title().lower()
    all_strs = text_strs + [title_lower]
    signals = ("no data", "empty", "unavailable", "no recorder")
    has_warning = any(any(sig in s for sig in signals) for s in all_strs)
    assert has_warning, (
        f"render_plot({plot_name!r}) with {recorder_state} recorder must "
        f"surface a warning panel containing one of {signals}. "
        f"ax.texts={text_strs!r}, title={title_lower!r}"
    )


    assert any("no data" in note.lower() for note in channel), channel

    assert len(ax.lines) == 0, (
        f"With {recorder_state} recorder, render_plot({plot_name!r})'s "
        f"warning panel must not draw a phantom data line; got "
        f"{len(ax.lines)} Line2D artists. Downstream report consumers "
        f"would mistake an empty run for a real one."
    )







@pytest.mark.unit
def test_render_plot_unknown_name_raises(ax: Axes) -> None:
    plugin, _ = _make_plugin_with_recorder()

    with pytest.raises(ValueError) as exc_info:
        plugin.render_plot("nonexistent_plot", ax)

    msg = str(exc_info.value)
    for name in EXPECTED_PLOT_NAMES:
        assert name in msg, (
            f"ValueError message must list the available plot name {name!r} "
            f"so users can self-correct. Got message: {msg!r}"
        )







@pytest.mark.unit
def test_render_plot_unknown_name_raises_with_none_recorder(ax: Axes) -> None:
    plugin = _make_plugin_none_recorder()

    with pytest.raises(ValueError) as exc_info:
        plugin.render_plot("nonexistent_plot", ax)


    msg = str(exc_info.value)
    for name in EXPECTED_PLOT_NAMES:
        assert name in msg, (
            f"With recorder=None, unknown-name ValueError must still list "
            f"available plot name {name!r}. Got message: {msg!r}"
        )

















@pytest.mark.unit
def test_sanitize_y_warns_on_unsupported_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from kd.search.discover.viz import _sanitize_y

    caplog.set_level(logging.WARNING, logger="kd.search.discover.viz")


    assert _sanitize_y(1.5) == 1.5
    assert _sanitize_y(2) == 2.0
    assert _sanitize_y(True) is None
    assert _sanitize_y(float("nan")) is None
    assert _sanitize_y(float("inf")) is None

    assert len(caplog.records) == 0, (
        f"Valid numeric payloads (float / int / bool / NaN / Inf) must NOT "
        f"trigger warnings — got {len(caplog.records)}: "
        f"{[r.message for r in caplog.records]}"
    )


    result = _sanitize_y("not-a-number")
    assert result is None, "exotic payload must degrade to None"
    assert len(caplog.records) == 1, (
        f"exotic payload must emit exactly one warning; got "
        f"{len(caplog.records)}: {[r.message for r in caplog.records]}"
    )
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert "str" in record.getMessage(), (
        f"warning must include the offending type name; got: {record.getMessage()!r}"
    )
