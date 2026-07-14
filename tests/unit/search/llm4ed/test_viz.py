
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from kd.search.llm4ed import viz as llm4ed_viz
from kd.search.llm4ed.plugin import _LOGGED_METRICS, Llm4edPlugin
from kd.search.recorder import VizRecorder
from kd.viz.extension import PlotInfo, VizExtension

from ._plugin_helpers import GOOD, FakeProvider, make_config, prepared, run

pytestmark = pytest.mark.unit

_PLOT_NAMES = ("pool_reward_spread", "invalid_count", "llm_calls")


def _recorder_with(**series: list[object]) -> VizRecorder:
    recorder = VizRecorder()
    for key, values in series.items():
        for value in values:
            recorder.log(key, value)
    return recorder







def test_plugin_satisfies_vizextension() -> None:
    plugin = Llm4edPlugin(make_config())
    assert isinstance(plugin, VizExtension)


def test_list_plots_returns_three_in_order() -> None:
    plugin = Llm4edPlugin(make_config())
    infos = plugin.list_plots()
    assert [info.name for info in infos] == list(_PLOT_NAMES)
    assert all(isinstance(info, PlotInfo) for info in infos)


def test_list_plots_returns_fresh_copies() -> None:
    plugin = Llm4edPlugin(make_config())
    plugin.list_plots()[0].title = "MUTATED"
    assert plugin.list_plots()[0].title == "Pool Reward Spread"


def test_whitelist_is_the_six_render_inputs() -> None:
    assert set(_LOGGED_METRICS) == {
        "pool_best",
        "pool_median",
        "pool_worst",
        "n_invalid",
        "n_llm_calls",
        "n_valid",
    }


def test_no_fifth_sanitize_y_copy() -> None:
    from pathlib import Path

    source = Path(llm4ed_viz.__file__).read_text(encoding="utf-8")


    assert "def _sanitize_y" not in source
    assert "sanitize_float" in source







def test_invalid_count_is_raw_per_round_count() -> None:
    recorder = _recorder_with(n_invalid=[1, 3, 0])
    data = llm4ed_viz.get_data("invalid_count", recorder)
    assert data["y"] == [1.0, 3.0, 0.0]
    assert data["x"] == [0, 1, 2]


def test_llm_calls_is_cumulative() -> None:
    recorder = _recorder_with(n_llm_calls=[4, 2, 3])
    data = llm4ed_viz.get_data("llm_calls", recorder)
    assert data["y"] == [4.0, 6.0, 9.0]


def test_pool_reward_spread_binds_each_series() -> None:
    recorder = _recorder_with(
        pool_best=[0.9, 0.95],
        pool_median=[0.5, 0.6],
        pool_worst=[0.1, 0.2],
    )
    data = llm4ed_viz.get_data("pool_reward_spread", recorder)
    assert data["y"]["pool_best"] == [0.9, 0.95]
    assert data["y"]["pool_median"] == [0.5, 0.6]
    assert data["y"]["pool_worst"] == [0.1, 0.2]







def test_get_data_sanitizes_nan_inf_to_none() -> None:
    recorder = _recorder_with(
        pool_best=[float("nan")],
        pool_median=[float("inf")],
        pool_worst=[0.3],
    )
    data = llm4ed_viz.get_data("pool_reward_spread", recorder)
    assert data["y"]["pool_best"] == [None]
    assert data["y"]["pool_median"] == [None]
    assert data["y"]["pool_worst"] == [0.3]


def test_get_data_is_json_serializable() -> None:
    import json

    recorder = _recorder_with(
        pool_best=[float("nan"), 0.9],
        pool_median=[0.4, 0.5],
        pool_worst=[0.1, 0.2],
    )
    for name in _PLOT_NAMES:
        json.dumps(llm4ed_viz.get_data(name, recorder))


def test_render_tolerates_nan(tmp_path) -> None:
    recorder = _recorder_with(n_invalid=[1, float("nan")])
    fig, ax = plt.subplots()
    try:
        llm4ed_viz.render("invalid_count", ax, recorder)
    finally:
        plt.close(fig)







@pytest.mark.parametrize("name", _PLOT_NAMES)
def test_empty_recorder_degrades(name: str) -> None:
    fig, ax = plt.subplots()
    try:
        llm4ed_viz.render(name, ax, VizRecorder())
        llm4ed_viz.render(name, ax, None)
    finally:
        plt.close(fig)
    y = llm4ed_viz.get_data(name, VizRecorder())["y"]
    series = list(y.values()) if isinstance(y, dict) else [y]
    assert all(s == [] for s in series)


@pytest.mark.parametrize("name", _PLOT_NAMES)
def test_disabled_recorder_is_empty(name: str) -> None:
    data = llm4ed_viz.get_data(name, VizRecorder(enabled=False))
    assert data["x"] == []


def test_unknown_name_raises_before_recorder_access() -> None:
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="Unknown plot name"):
            llm4ed_viz.render("nope", ax, None)
    finally:
        plt.close(fig)
    with pytest.raises(ValueError, match="Unknown plot name"):
        llm4ed_viz.get_data("nope", None)







def test_plots_render_after_a_real_run() -> None:
    plugin, components = prepared(config=make_config(), provider=FakeProvider(GOOD))
    run(plugin, rounds=2)
    for name in _PLOT_NAMES:
        data = plugin.get_plot_data(name)
        assert data["title"]
        fig, ax = plt.subplots()
        try:
            plugin.render_plot(name, ax)
        finally:
            plt.close(fig)

    calls = plugin.get_plot_data("llm_calls")["y"]
    assert calls and all(
        b >= a for a, b in zip(calls, calls[1:], strict=False)
    )
