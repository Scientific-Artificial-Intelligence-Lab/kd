
from __future__ import annotations

import inspect
import json
import logging
import math
from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from kd.search.recorder import VizRecorder
from kd.search.sga import plugin as sga_plugin
from kd.search.sga import viz as sga_viz
from kd.search.sga.config import SGAConfig
from kd.search.sga.pde import PDE
from kd.search.sga.plugin import SGAPlugin
from kd.search.sga.tree import Node, Tree
from kd.viz.extension import PlotInfo, VizExtension









EXPECTED_PLOT_NAMES: frozenset[str] = frozenset(
    {"population_diversity", "complexity_evolution", "fitness_spread"},
)


EXPECTED_RECORDER_PLOT_NAMES: frozenset[str] = EXPECTED_PLOT_NAMES | {
    "surrogate_training"
}





EXPECTED_ALL_PLOT_NAMES: frozenset[str] = EXPECTED_RECORDER_PLOT_NAMES | {"genome_tree"}





PLOT_TO_METRIC: dict[str, str] = {
    "population_diversity": "n_unique",
    "complexity_evolution": "gen_mean_complexity",
    "fitness_spread": "pop_mean_aic",
}

N_SYNTHETIC_GENS = 5


_X_LABEL_PREFIX = "gen"







def _populate_recorder(recorder: VizRecorder, n_gens: int = N_SYNTHETIC_GENS) -> None:
    for i in range(n_gens):

        recorder.log("pop_mean_aic", float(1000.0 - 50.0 * i))
        recorder.log("n_unique", 90 - 5 * i)
        recorder.log("gen_mean_complexity", float(1.0 + 0.5 * i))





        recorder.log("gen_best_aic", float(900.0 - 50.0 * i))
        recorder.log("gen_mean_aic", float(1300.0 - 50.0 * i))
        recorder.log("gen_best_nmse", float(0.50 - 0.05 * i))
        recorder.log("n_valid", 20 + i)


def _populate_recorder_with_inf(recorder: VizRecorder) -> None:
    mean_aic_vals = [float("inf"), float("inf"), 900.0, 850.0, 800.0]
    for i, mean_aic in enumerate(mean_aic_vals):
        recorder.log("pop_mean_aic", mean_aic)
        recorder.log("gen_mean_aic", mean_aic + (0.0 if i < 2 else 300.0))
        recorder.log("n_unique", 90 - 5 * i)
        recorder.log("gen_mean_complexity", float(1.0 + 0.5 * i))
        recorder.log("gen_best_aic", float("inf") if i < 2 else 800.0 - 50.0 * i)
        recorder.log("gen_best_nmse", float("inf") if i < 2 else 0.40 - 0.05 * i)
        recorder.log("n_valid", 0 if i < 2 else 20 + i)


def _plugin_with_recorder(recorder: VizRecorder) -> SGAPlugin:
    plugin = SGAPlugin(SGAConfig())
    plugin._recorder = recorder
    return plugin


def _plugin_populated() -> tuple[SGAPlugin, VizRecorder]:
    recorder = VizRecorder(enabled=True)
    _populate_recorder(recorder)
    return _plugin_with_recorder(recorder), recorder


def _plugin_empty_recorder() -> SGAPlugin:
    return _plugin_with_recorder(VizRecorder(enabled=True))


def _plugin_disabled_recorder() -> SGAPlugin:
    return _plugin_with_recorder(VizRecorder(enabled=False))


def _plugin_none_recorder() -> SGAPlugin:
    plugin = SGAPlugin(SGAConfig())
    plugin._recorder = None
    return plugin







@pytest.fixture
def ax() -> Iterator[Axes]:
    _fig, _ax = plt.subplots()
    yield _ax
    plt.close(_fig)







@pytest.mark.unit
def test_plugin_is_runtime_checkable_viz_extension() -> None:
    plugin = SGAPlugin(SGAConfig())
    assert isinstance(plugin, VizExtension), (
        "SGAPlugin must satisfy the VizExtension Protocol structurally. "
        "Required: list_plots(), render_plot(name, ax), get_plot_data(name)."
    )


    assert VizExtension not in SGAPlugin.__mro__, (
        "SGAPlugin must NOT inherit VizExtension; "
        f"got MRO {[c.__name__ for c in SGAPlugin.__mro__]}"
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
def test_list_plots_returns_five_plotinfo() -> None:
    plugin = SGAPlugin(SGAConfig())
    plots = plugin.list_plots()

    assert isinstance(plots, list), (
        f"list_plots() must return list, got {type(plots).__name__}"
    )
    assert len(plots) == 5, (
        f"list_plots() must return exactly 5 PlotInfo; got {len(plots)}: "
        f"{[getattr(p, 'name', '?') for p in plots]}"
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
    assert names == EXPECTED_ALL_PLOT_NAMES, (
        f"list_plots() names must equal spec set.\n"
        f" missing: {sorted(EXPECTED_ALL_PLOT_NAMES - names)}\n"
        f" extra: {sorted(names - EXPECTED_ALL_PLOT_NAMES)}"
    )


@pytest.mark.unit
def test_list_plots_returns_fresh_copies() -> None:
    plugin = SGAPlugin(SGAConfig())
    first = plugin.list_plots()
    first[0].title = "MUTATED"
    first[-1].title = "MUTATED"
    second = plugin.list_plots()
    assert all(p.title != "MUTATED" for p in second), (
        "list_plots() must rebuild descriptors each call so a caller mutating "
        "the returned objects cannot poison the module-level table — including "
        "the genome_tree descriptor appended by the plugin."
    )







def _plugin_with_population() -> SGAPlugin:
    plugin = SGAPlugin(SGAConfig())
    plugin._population = [
        PDE(
            [
                Tree(
                    Node(
                        "*",
                        2,
                        [Node("u", 0), Node("d", 2, [Node("u", 0), Node("x", 0)])],
                    )
                )
            ]
        )
    ]
    return plugin


@pytest.mark.unit
def test_genome_tree_render_uses_live_population(ax: Axes) -> None:
    plugin = _plugin_with_population()
    plugin.render_plot("genome_tree", ax)
    labels = {t.get_text() for t in ax.texts}
    assert {"*", "u", "d", "x"} <= labels
    assert "unavailable" not in " ".join(labels).lower()


@pytest.mark.unit
def test_genome_tree_get_plot_data_live_population() -> None:
    plugin = _plugin_with_population()
    data = plugin.get_plot_data("genome_tree")
    assert data["available"] is True
    assert data["tree"]["label"] == "*"


@pytest.mark.unit
def test_genome_tree_degrades_without_population(ax: Axes) -> None:
    plugin = SGAPlugin(SGAConfig())
    channel = plugin.render_plot("genome_tree", ax)
    text = " ".join(t.get_text() for t in ax.texts).lower()
    assert "unavailable" in text




    assert any(
        note.startswith("plugin plot 'genome_tree':") and "unavailable" in note
        for note in channel
    ), channel
    assert plugin.get_plot_data("genome_tree")["available"] is False







@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(EXPECTED_PLOT_NAMES))
def test_render_plot_returns_empty_warnings(plot_name: str, ax: Axes) -> None:
    plugin, _ = _plugin_populated()
    result = plugin.render_plot(plot_name, ax)
    assert result == [], (
        f"render_plot({plot_name!r}, ax) must return no warnings; got {result!r}"
    )







@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(EXPECTED_PLOT_NAMES))
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
    assert xlabel.startswith(_X_LABEL_PREFIX), (
        f"{plot_name} xlabel must start with {_X_LABEL_PREFIX!r}; got "
        f"{ax.get_xlabel()!r}"
    )


@pytest.mark.unit
def test_fitness_spread_renders_population_mean(ax: Axes) -> None:
    plugin, recorder = _plugin_populated()
    plugin.render_plot("fitness_spread", ax)

    y_data = np.asarray(ax.lines[0].get_ydata()).tolist()
    pop_series = recorder.get("pop_mean_aic")
    offspring_series = recorder.get("gen_mean_aic")
    best_series = recorder.get("gen_best_aic")


    assert len({tuple(pop_series), tuple(offspring_series), tuple(best_series)}) == 3, (
        "fixture bug: pop_mean_aic / gen_mean_aic / gen_best_aic must all "
        "differ for this test to discriminate them."
    )
    assert y_data == pytest.approx(pop_series), (
        f"fitness_spread must plot pop_mean_aic (population mean), got "
        f"ydata={y_data!r}, population={pop_series!r}."
    )
    assert y_data[-1] != pytest.approx(best_series[-1]), (
        "fitness_spread must NOT plot gen_best_aic."
    )
    assert y_data[-1] != pytest.approx(offspring_series[-1]), (
        "fitness_spread must NOT plot the offspring mean gen_mean_aic."
    )







@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(EXPECTED_PLOT_NAMES))
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
def test_get_plot_data_fitness_spread_is_population_series() -> None:
    plugin, recorder = _plugin_populated()
    data = plugin.get_plot_data("fitness_spread")

    pop_series = recorder.get("pop_mean_aic")
    best_series = recorder.get("gen_best_aic")
    assert list(data["y"]) == pytest.approx(pop_series), (
        f"fitness_spread y must equal pop_mean_aic; got {data['y']!r}, "
        f"population={pop_series!r}."
    )
    assert data["y"][-1] != pytest.approx(best_series[-1]), (
        "fitness_spread y[-1] must not equal gen_best_aic[-1]."
    )







@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(EXPECTED_PLOT_NAMES))
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
        channel = plugin.render_plot(plot_name, ax)
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
        f"render_plot({plot_name!r}) with {recorder_state} recorder must surface "
        f"a warning panel with one of {signals}. texts={text_strs!r}, "
        f"title={title!r}"
    )


    assert any("no data" in note.lower() for note in channel), channel
    assert len(ax.lines) == 0, (
        f"{recorder_state} recorder warning panel must not draw a phantom line; "
        f"got {len(ax.lines)} lines."
    )


@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(EXPECTED_PLOT_NAMES))
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


def _plugin_only_best_aic() -> SGAPlugin:
    recorder = VizRecorder(enabled=True)
    recorder.log("best_aic", 42.0)
    recorder.log("best_aic", 41.0)
    return _plugin_with_recorder(recorder)


@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(EXPECTED_PLOT_NAMES))
def test_render_missing_metric_series_warns_not_crash(plot_name: str, ax: Axes) -> None:
    plugin = _plugin_only_best_aic()

    try:
        plugin.render_plot(plot_name, ax)
    except Exception as exc:
        pytest.fail(
            f"render_plot({plot_name!r}) with a missing metric series must not "
            f"raise; got {type(exc).__name__}: {exc!r}"
        )

    text_strs = [t.get_text().lower() for t in ax.texts]
    title = ax.get_title().lower()
    signals = ("no data", "empty", "unavailable", "no recorder")
    has_warning = any(any(sig in s for sig in signals) for s in [*text_strs, title])
    assert has_warning, (
        f"render_plot({plot_name!r}) with a missing metric series must surface a "
        f"warning panel with one of {signals}. texts={text_strs!r}, title={title!r}"
    )
    assert len(ax.lines) == 0, (
        f"{plot_name}: a missing-metric warning panel must not draw a phantom "
        f"line from the unrelated best_aic series; got {len(ax.lines)} lines."
    )


@pytest.mark.unit
@pytest.mark.parametrize("plot_name", sorted(EXPECTED_PLOT_NAMES))
def test_get_plot_data_missing_metric_series_empty(plot_name: str) -> None:
    plugin = _plugin_only_best_aic()

    data = plugin.get_plot_data(plot_name)
    assert isinstance(data, dict)
    assert list(data["y"]) == [], (
        f"{plot_name}: y must be empty when the metric series is missing; got "
        f"{data['y']!r} (must not leak the unrelated best_aic series)."
    )
    assert list(data["x"]) == []
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
def test_unknown_name_hint_lists_genome_tree(ax: Axes) -> None:
    plugin, _ = _plugin_populated()
    with pytest.raises(ValueError, match="genome_tree"):
        plugin.render_plot("nonexistent", ax)
    with pytest.raises(ValueError, match="genome_tree"):
        plugin.get_plot_data("nonexistent")


@pytest.mark.unit
def test_get_plot_data_unknown_name_raises() -> None:
    plugin, _ = _plugin_populated()
    with pytest.raises(ValueError):
        plugin.get_plot_data("nonexistent")


@pytest.mark.unit
def test_get_plot_data_unknown_name_raises_with_none_recorder() -> None:
    plugin = _plugin_none_recorder()
    with pytest.raises(ValueError) as exc_info:
        plugin.get_plot_data("nonexistent")
    msg = str(exc_info.value)
    for name in EXPECTED_PLOT_NAMES:
        assert name in msg, (
            f"recorder=None unknown-name ValueError must list {name!r}; got {msg!r}"
        )







@pytest.mark.unit
def test_render_fitness_spread_masks_inf_no_autoscale_blowup(ax: Axes) -> None:
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
        f"finite aic must be preserved in order; got {finite_vals.tolist()!r}"
    )
    ylim = ax.get_ylim()
    assert all(math.isfinite(v) for v in ylim), (
        f"y-axis limits must be finite; got {ylim!r}"
    )
    assert ylim[1] < 1e6, (
        f"y-axis upper bound must not blow up from +inf sentinels; got {ylim[1]!r}"
    )


@pytest.mark.unit
def test_render_fitness_spread_all_inf_no_crash(ax: Axes) -> None:
    recorder = VizRecorder(enabled=True)
    n_gens = 4
    for _ in range(n_gens):
        recorder.log("pop_mean_aic", float("inf"))
    plugin = _plugin_with_recorder(recorder)

    plugin.render_plot("fitness_spread", ax)

    y_data = np.asarray(ax.lines[0].get_ydata(), dtype=float)
    assert len(y_data) == n_gens
    assert not np.isfinite(y_data).any(), (
        f"all-inf series must mask to all-nan; got {y_data!r}"
    )
    ylim = ax.get_ylim()
    assert all(math.isfinite(v) for v in ylim), (
        f"y-axis limits must stay finite with all-nan data; got {ylim!r}"
    )


@pytest.mark.unit
def test_get_plot_data_fitness_spread_inf_becomes_none() -> None:
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
def test_render_fitness_spread_after_recorder_roundtrip(ax: Axes) -> None:
    recorder = VizRecorder(enabled=True)
    _populate_recorder_with_inf(recorder)
    restored = VizRecorder.from_dict(recorder.to_dict())
    plugin = _plugin_with_recorder(restored)

    plugin.render_plot("fitness_spread", ax)

    y_data = np.asarray(ax.lines[0].get_ydata(), dtype=float)
    assert not np.isinf(y_data).any(), "no raw inf may reach matplotlib after roundtrip"
    finite_vals = y_data[np.isfinite(y_data)]







    pop_finite = [v for v in restored.get("pop_mean_aic") if v is not None]
    best_finite = [v for v in restored.get("gen_best_aic") if v is not None]
    assert pop_finite != best_finite, (
        "fixture bug: finite pop_mean_aic and gen_best_aic must differ for the "
        "round-trip discriminator to bite."
    )
    assert finite_vals.tolist() == pytest.approx(pop_finite), (
        f"after round-trip, fitness_spread must still plot pop_mean_aic's "
        f"finite values {pop_finite!r} (NOT gen_best_aic's {best_finite!r}); "
        f"got {finite_vals.tolist()!r}."
    )


_SGA_VIZ_LOGGER = "kd.search.sga.viz"


@pytest.mark.unit
def test_get_plot_data_fitness_spread_roundtrip_none_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = VizRecorder(enabled=True)
    _populate_recorder_with_inf(recorder)
    restored = VizRecorder.from_dict(recorder.to_dict())
    plugin = _plugin_with_recorder(restored)

    caplog.set_level(logging.WARNING, logger=_SGA_VIZ_LOGGER)
    data = plugin.get_plot_data("fitness_spread")

    assert data["y"][0] is None
    assert data["y"][1] is None
    assert data["y"][2] == pytest.approx(900.0)
    json.dumps(data, allow_nan=False)
    assert len(caplog.records) == 0, (
        f"None gaps from deserialization must not warn; got "
        f"{[r.message for r in caplog.records]}"
    )


    finite_y = [v for v in data["y"] if v is not None]
    pop_finite = [v for v in restored.get("pop_mean_aic") if v is not None]
    best_finite = [v for v in restored.get("gen_best_aic") if v is not None]
    assert pop_finite != best_finite, (
        "fixture bug: finite pop_mean_aic/gen_best_aic must differ for the "
        "round-trip data discriminator to bite."
    )
    assert finite_y == pytest.approx(pop_finite), (
        f"get_plot_data after round-trip must keep pop_mean_aic's finite values "
        f"{pop_finite!r} (NOT gen_best_aic's {best_finite!r}); got {finite_y!r}."
    )


@pytest.mark.unit
def test_sga_viz_logger_warns_on_exotic_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = VizRecorder(enabled=True)

    recorder.log("pop_mean_aic", 100.0)
    recorder.log("pop_mean_aic", "not-a-number")
    recorder.log("pop_mean_aic", 80.0)
    plugin = _plugin_with_recorder(recorder)

    caplog.set_level(logging.WARNING, logger=_SGA_VIZ_LOGGER)
    data = plugin.get_plot_data("fitness_spread")

    scoped = [r for r in caplog.records if r.name == _SGA_VIZ_LOGGER]
    assert scoped, (
        f"an exotic str payload in a plotted series must emit >=1 warning on "
        f"{_SGA_VIZ_LOGGER!r} (proves the no-warning test's logger scope is "
        f"live, not vacuous). Records: {[(r.name, r.message) for r in caplog.records]}"
    )

    assert data["y"][1] is None, (
        f"exotic str sample must be dropped to None; got {data['y'][1]!r}"
    )
    assert data["y"][0] == pytest.approx(100.0)
    assert data["y"][2] == pytest.approx(80.0)
    json.dumps(data, allow_nan=False)


@pytest.mark.unit
def test_render_all_plots_after_roundtrip_no_crash(ax: Axes) -> None:
    recorder = VizRecorder(enabled=True)
    _populate_recorder_with_inf(recorder)
    restored = VizRecorder.from_dict(recorder.to_dict())
    plugin = _plugin_with_recorder(restored)

    for name in sorted(EXPECTED_PLOT_NAMES):
        ax.clear()
        try:
            plugin.render_plot(name, ax)
        except Exception as exc:
            pytest.fail(
                f"render_plot({name!r}) after recorder round-trip must not "
                f"raise; got {type(exc).__name__}: {exc!r}"
            )







@pytest.mark.unit
def test_plot_metric_keys_are_within_the_logged_whitelist() -> None:
    used = set(sga_viz._PLOT_METRIC.values())
    assert used <= set(sga_plugin._LOGGED_METRICS)
