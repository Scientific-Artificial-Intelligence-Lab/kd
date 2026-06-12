
from __future__ import annotations

import inspect
import json
import math
from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.recorder import VizRecorder
from kd.viz.extension import PlotInfo, VizExtension









GA_PLOT_NAMES: frozenset[str] = frozenset(
    {"fitness_spread", "population_diversity", "complexity_evolution"},
)


SURROGATE_PLOT_NAME = "surrogate_training"




EXPECTED_PLOT_NAMES: frozenset[str] = GA_PLOT_NAMES | {SURROGATE_PLOT_NAME}







PLOT_TO_METRIC: dict[str, str] = {
    "fitness_spread": "gen_mean_fitness",
    "population_diversity": "n_unique",
    "complexity_evolution": "gen_mean_complexity",
}

N_SYNTHETIC_GENS = 5







def _populate_recorder(recorder: VizRecorder, n_gens: int = N_SYNTHETIC_GENS) -> None:
    for i in range(n_gens):

        recorder.log("gen_mean_fitness", float(1000.0 - 50.0 * i))
        recorder.log("n_unique", 90 - 5 * i)
        recorder.log("gen_mean_complexity", float(1.0 + 0.5 * i))

        recorder.log("gen_best_fitness", float(1500.0 - 30.0 * i))
        recorder.log("gen_best_nmse", float(0.50 - 0.05 * i))
        recorder.log("n_valid", 20 + i)
        recorder.log("lhs_ut", 15)
        recorder.log("lhs_utt", 5)


def _populate_recorder_with_inf(recorder: VizRecorder) -> None:
    fitness_vals = [float("inf"), float("inf"), 900.0, 850.0, 800.0]
    for i, fitness in enumerate(fitness_vals):
        recorder.log("gen_mean_fitness", fitness)
        recorder.log("n_unique", 90 - 5 * i)
        recorder.log("gen_mean_complexity", float(1.0 + 0.5 * i))
        recorder.log("gen_best_fitness", float("inf") if i < 2 else 1000.0 - 30.0 * i)
        recorder.log("gen_best_nmse", float("inf") if i < 2 else 0.40 - 0.05 * i)
        recorder.log("n_valid", 0 if i < 2 else 20 + i)
        recorder.log("lhs_ut", 0 if i < 2 else 15)
        recorder.log("lhs_utt", 0 if i < 2 else 5)


def _plugin_with_recorder(
    recorder: VizRecorder,
) -> DLGAPlugin:
    plugin = DLGAPlugin(DLGAConfig())
    plugin._recorder = recorder
    return plugin


def _plugin_populated() -> tuple[DLGAPlugin, VizRecorder]:
    recorder = VizRecorder(enabled=True)
    _populate_recorder(recorder)
    return _plugin_with_recorder(recorder), recorder


def _plugin_empty_recorder() -> DLGAPlugin:
    return _plugin_with_recorder(VizRecorder(enabled=True))


def _plugin_disabled_recorder() -> DLGAPlugin:
    return _plugin_with_recorder(VizRecorder(enabled=False))


def _plugin_none_recorder() -> DLGAPlugin:
    plugin = DLGAPlugin(DLGAConfig())
    plugin._recorder = None
    return plugin







@pytest.fixture
def ax() -> Iterator[Axes]:
    _fig, _ax = plt.subplots()
    yield _ax
    plt.close(_fig)







@pytest.mark.unit
def test_plugin_is_runtime_checkable_viz_extension() -> None:
    plugin = DLGAPlugin(DLGAConfig())
    assert isinstance(plugin, VizExtension), (
        "DLGAPlugin must satisfy the VizExtension Protocol structurally. "
        "Required: list_plots(), render_plot(name, ax), get_plot_data(name)."
    )


    assert VizExtension not in DLGAPlugin.__mro__, (
        "DLGAPlugin must NOT inherit VizExtension; "
        f"got MRO {[c.__name__ for c in DLGAPlugin.__mro__]}"
    )

    assert callable(plugin.list_plots)
    assert callable(plugin.render_plot)
    assert callable(plugin.get_plot_data)

    list_params = list(inspect.signature(plugin.list_plots).parameters.keys())
    render_params = list(inspect.signature(plugin.render_plot).parameters.keys())
    get_params = list(inspect.signature(plugin.get_plot_data).parameters.keys())
    assert list_params == [], (
        f"list_plots() takes no params beyond self; got {list_params!r}"
    )
    assert render_params == ["name", "ax"], (
        f"render_plot signature must be (name, ax); got {render_params!r}"
    )
    assert get_params == ["name"], (
        f"get_plot_data signature must be (name,); got {get_params!r}"
    )








@pytest.mark.unit
def test_list_plots_returns_four_plotinfo() -> None:
    plugin = DLGAPlugin(DLGAConfig())
    plots = plugin.list_plots()

    assert isinstance(plots, list), (
        f"list_plots() must return list, got {type(plots).__name__}"
    )
    assert len(plots) == 4, (
        f"list_plots() must return exactly 4 PlotInfo (3 GA + surrogate); got "
        f"{len(plots)}: {[getattr(p, 'name', '?') for p in plots]}"
    )
    for p in plots:
        assert isinstance(p, PlotInfo), (
            f"Every entry must be PlotInfo, got {type(p).__name__}"
        )
        assert isinstance(p.title, str) and p.title.strip(), (
            f"PlotInfo({p.name!r}).title must be a non-empty str, got {p.title!r}"
        )
        assert isinstance(p.description, str)

    names = {p.name for p in plots}
    assert names == EXPECTED_PLOT_NAMES, (
        f"list_plots() names must equal spec set.\n"
        f" missing: {sorted(EXPECTED_PLOT_NAMES - names)}\n"
        f" extra: {sorted(names - EXPECTED_PLOT_NAMES)}"
    )


@pytest.mark.unit
def test_list_plots_returns_fresh_copies() -> None:
    plugin = DLGAPlugin(DLGAConfig())
    first = plugin.list_plots()
    first[0].title = "MUTATED"
    second = plugin.list_plots()
    assert all(p.title != "MUTATED" for p in second), (
        "list_plots() must rebuild descriptors each call so a caller mutating "
        "the returned objects cannot poison the module-level table."
    )







@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(GA_PLOT_NAMES))
def test_render_plot_returns_none(plot_name: str, ax: Axes) -> None:
    plugin, _ = _plugin_populated()
    result = plugin.render_plot(plot_name, ax)
    assert result is None, (
        f"render_plot({plot_name!r}, ax) must return None; got {type(result).__name__}"
    )







@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(GA_PLOT_NAMES))
def test_render_binds_recorder_series(plot_name: str, ax: Axes) -> None:
    plugin, recorder = _plugin_populated()
    plugin.render_plot(plot_name, ax)

    assert len(ax.lines) >= 1, f"{plot_name} must draw >=1 line; got 0 Line2D artists."

    y_data = np.asarray(ax.lines[0].get_ydata()).tolist()
    metric = PLOT_TO_METRIC[plot_name]
    expected = recorder.get(metric)
    assert y_data == pytest.approx(expected), (
        f"{plot_name} must draw from recorder[{metric!r}]. "
        f"Got ydata={y_data!r}, expected={expected!r}."
    )
    assert len(y_data) == N_SYNTHETIC_GENS, (
        f"{plot_name} must plot all {N_SYNTHETIC_GENS} generations; got {len(y_data)}."
    )
    assert ax.get_ylabel() == metric, (
        f"{plot_name} ylabel must be exactly {metric!r}, got {ax.get_ylabel()!r}"
    )
    xlabel = ax.get_xlabel().lower()
    assert xlabel.startswith("gen"), (
        f"{plot_name} xlabel must start with 'gen'; got {ax.get_xlabel()!r}"
    )







@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(GA_PLOT_NAMES))
def test_get_plot_data_returns_jsonable_dict(plot_name: str) -> None:
    plugin, recorder = _plugin_populated()
    data = plugin.get_plot_data(plot_name)

    assert isinstance(data, dict), (
        f"get_plot_data({plot_name!r}) must return dict, got {type(data).__name__}"
    )
    required = {"x", "y", "xlabel", "ylabel"}
    missing = required - data.keys()
    assert not missing, f"get_plot_data({plot_name!r}) missing keys: {sorted(missing)}"

    json.dumps(data)

    metric = PLOT_TO_METRIC[plot_name]
    expected_y = recorder.get(metric)
    assert list(data["y"]) == pytest.approx(expected_y), (
        f"get_plot_data({plot_name!r})['y'] must equal recorder[{metric!r}]."
    )
    assert len(data["x"]) == N_SYNTHETIC_GENS
    assert len(data["x"]) == len(data["y"])
    assert data["ylabel"] == metric, (
        f"{plot_name} ylabel must be {metric!r}, got {data['ylabel']!r}"
    )







@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(GA_PLOT_NAMES))
@pytest.mark.parametrize(
    "recorder_state",
    ["empty", "none", "disabled"],
    ids=["empty", "none", "disabled"],
)
def test_render_empty_recorder_warns_not_crash(
    plot_name: str, recorder_state: str, ax: Axes
) -> None:
    plugin = {
        "empty": _plugin_empty_recorder,
        "disabled": _plugin_disabled_recorder,
        "none": _plugin_none_recorder,
    }[recorder_state]()

    try:
        plugin.render_plot(plot_name, ax)
    except Exception as exc:
        pytest.fail(
            f"render_plot({plot_name!r}) with {recorder_state} recorder must not "
            f"raise; got {type(exc).__name__}: {exc!r}"
        )

    text_strs = [t.get_text().lower() for t in ax.texts]
    title = ax.get_title().lower()
    signals = ("no data", "empty", "unavailable", "no recorder")
    has_warning = any(any(sig in s for sig in signals) for s in [*text_strs, title])
    assert has_warning, (
        f"render_plot({plot_name!r}) with {recorder_state} recorder must surface a "
        f"warning panel with one of {signals}. texts={text_strs!r}, title={title!r}"
    )
    assert len(ax.lines) == 0, (
        f"{recorder_state} recorder warning panel must not draw a phantom line; "
        f"got {len(ax.lines)} lines."
    )


@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(GA_PLOT_NAMES))
@pytest.mark.parametrize(
    "recorder_state",
    ["empty", "none", "disabled"],
    ids=["empty", "none", "disabled"],
)
def test_get_plot_data_empty_recorder_jsonable(
    plot_name: str, recorder_state: str
) -> None:
    plugin = {
        "empty": _plugin_empty_recorder,
        "disabled": _plugin_disabled_recorder,
        "none": _plugin_none_recorder,
    }[recorder_state]()

    data = plugin.get_plot_data(plot_name)
    assert isinstance(data, dict)
    assert list(data["y"]) == [], (
        f"{recorder_state}: y must be empty; got {data['y']!r}"
    )
    assert list(data["x"]) == [], (
        f"{recorder_state}: x must be empty; got {data['x']!r}"
    )
    assert data["ylabel"] == PLOT_TO_METRIC[plot_name]
    json.dumps(data)







@pytest.mark.unit
def test_render_unknown_name_raises(ax: Axes) -> None:
    plugin, _ = _plugin_populated()
    with pytest.raises(ValueError) as exc_info:
        plugin.render_plot("nonexistent", ax)
    msg = str(exc_info.value)
    for name in EXPECTED_PLOT_NAMES:
        assert name in msg, f"ValueError must list available name {name!r}; got {msg!r}"


@pytest.mark.unit
def test_render_unknown_name_raises_with_none_recorder(ax: Axes) -> None:
    plugin = _plugin_none_recorder()
    with pytest.raises(ValueError) as exc_info:
        plugin.render_plot("nonexistent", ax)
    msg = str(exc_info.value)
    for name in EXPECTED_PLOT_NAMES:
        assert name in msg, (
            f"recorder=None unknown-name ValueError must list {name!r}; got {msg!r}"
        )


@pytest.mark.unit
def test_get_plot_data_unknown_name_raises() -> None:
    plugin, _ = _plugin_populated()
    with pytest.raises(ValueError):
        plugin.get_plot_data("nonexistent")







@pytest.mark.unit
def test_render_fitness_masks_inf_no_autoscale_blowup(ax: Axes) -> None:
    recorder = VizRecorder(enabled=True)
    _populate_recorder_with_inf(recorder)
    plugin = _plugin_with_recorder(recorder)

    plugin.render_plot("fitness_spread", ax)

    y_data = np.asarray(ax.lines[0].get_ydata(), dtype=float)

    assert not np.isinf(y_data).any(), (
        f"render must mask +inf to nan before plotting; raw inf in ydata={y_data!r}"
    )

    finite_vals = y_data[np.isfinite(y_data)]
    assert finite_vals.tolist() == pytest.approx([900.0, 850.0, 800.0]), (
        f"finite fitness must be preserved in order; got {finite_vals.tolist()!r}"
    )

    ylim = ax.get_ylim()
    assert all(math.isfinite(v) for v in ylim), (
        f"y-axis limits must be finite; got {ylim!r}"
    )
    assert ylim[1] < 1e6, (
        f"y-axis upper bound must not blow up from +inf sentinels; got {ylim[1]!r}"
    )


@pytest.mark.unit
def test_get_plot_data_fitness_inf_becomes_none() -> None:
    recorder = VizRecorder(enabled=True)
    _populate_recorder_with_inf(recorder)
    plugin = _plugin_with_recorder(recorder)

    data = plugin.get_plot_data("fitness_spread")

    assert data["y"][0] is None, (
        f"+inf gen0 must serialize to None; got {data['y'][0]!r}"
    )
    assert data["y"][1] is None, (
        f"+inf gen1 must serialize to None; got {data['y'][1]!r}"
    )
    assert data["y"][2] == pytest.approx(900.0)
    assert data["y"][3] == pytest.approx(850.0)
    assert data["y"][4] == pytest.approx(800.0)

    json.dumps(data, allow_nan=False)


@pytest.mark.unit
def test_render_fitness_after_recorder_roundtrip(ax: Axes) -> None:
    recorder = VizRecorder(enabled=True)
    _populate_recorder_with_inf(recorder)
    restored = VizRecorder.from_dict(recorder.to_dict())
    plugin = _plugin_with_recorder(restored)

    plugin.render_plot("fitness_spread", ax)

    y_data = np.asarray(ax.lines[0].get_ydata(), dtype=float)
    assert not np.isinf(y_data).any(), "no raw inf may reach matplotlib after roundtrip"
    finite_vals = y_data[np.isfinite(y_data)]
    assert finite_vals.tolist() == pytest.approx([900.0, 850.0, 800.0]), (
        f"finite fitness must survive to_dict/from_dict; got {finite_vals.tolist()!r}"
    )


@pytest.mark.unit
def test_get_plot_data_fitness_roundtrip_none_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    recorder = VizRecorder(enabled=True)
    _populate_recorder_with_inf(recorder)
    restored = VizRecorder.from_dict(recorder.to_dict())
    plugin = _plugin_with_recorder(restored)

    caplog.set_level(logging.WARNING, logger="kd.search.dlga.viz")
    data = plugin.get_plot_data("fitness_spread")

    assert data["y"][0] is None
    assert data["y"][1] is None
    assert data["y"][2] == pytest.approx(900.0)
    json.dumps(data, allow_nan=False)
    assert len(caplog.records) == 0, (
        f"None gaps from deserialization must not warn; got "
        f"{[r.message for r in caplog.records]}"
    )







@pytest.mark.unit
def test_fitness_spread_plots_population_mean_not_best(ax: Axes) -> None:
    recorder = VizRecorder(enabled=True)
    _populate_recorder(recorder)
    plugin = _plugin_with_recorder(recorder)

    plugin.render_plot("fitness_spread", ax)

    y_data = np.asarray(ax.lines[0].get_ydata()).tolist()
    assert y_data == pytest.approx(recorder.get("gen_mean_fitness")), (
        "fitness_spread must draw the population MEAN (gen_mean_fitness)"
    )

    assert y_data != pytest.approx(recorder.get("gen_best_fitness")), (
        "fitness_spread must NOT draw the per-generation best (gen_best_fitness) "
        "— that duplicates-and-disagrees with the platform convergence plot."
    )
    assert ax.get_ylabel() == "gen_mean_fitness"
