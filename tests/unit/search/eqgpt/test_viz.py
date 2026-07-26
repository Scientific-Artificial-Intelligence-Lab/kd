
from __future__ import annotations

import dataclasses
import json
import math

import matplotlib

matplotlib.use("Agg")
import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.eqgpt import _scoring
from kd.search.eqgpt import viz as eqgpt_viz
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import _LOGGED_METRICS, EqGPTPlugin
from kd.search.recorder import VizRecorder
from kd.viz.extension import PlotInfo, VizExtension

_PLOT_NAMES = ("reward_convergence", "pool_reward_spread", "finetune_loss")
_SINGLE_PLOTS = ("reward_convergence", "finetune_loss")
_SINGLE_METRIC = {"reward_convergence": "pool_best", "finetune_loss": "finetune_loss"}
_SPREAD_KEYS = ("pool_best", "pool_median", "pool_worst")
_N_EPOCHS = 4
_BATCH = 8







@pytest.fixture
def ax():
    fig, axis = plt.subplots()
    try:
        yield axis
    finally:
        plt.close(fig)


def _components(recorder: VizRecorder | None = None):
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 6)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    dataset = PDEDataset(
        name="viz_tiny",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(gx) * torch.cos(gt))},
        lhs_field="u",
        lhs_axis="t",
    )
    built = PlatformBuilder(dataset, DerivativeReqs()).build()
    if recorder is not None:
        built = dataclasses.replace(built, recorder=recorder)
    return built


def _plugin() -> EqGPTPlugin:
    config = EqGPTConfig(
        sparsity_alpha=0.02,
        seed=0,
        samples_per_epoch=_BATCH,
        top_k=4,
        max_length=12,
        variables=("t", "x"),
    )
    return EqGPTPlugin(config, backend=FakeGPTBackend(57, seed=0))


def _disjoint_recorder() -> VizRecorder:
    recorder = VizRecorder(enabled=True)
    for epoch in range(_N_EPOCHS):
        recorder.log("pool_best", 100.0 + epoch)
        recorder.log("pool_median", 50.0 + epoch)
        recorder.log("pool_worst", 10.0 + epoch)
        recorder.log("finetune_loss", 5.0 + epoch)
    return recorder


def _ydata(line) -> list[float]:
    return np.asarray(line.get_ydata()).tolist()







def test_plugin_is_runtime_checkable_viz_extension() -> None:
    assert isinstance(_plugin(), VizExtension)


def test_dispatch_methods_have_the_protocol_signatures() -> None:
    plugin = _plugin()
    assert callable(plugin.list_plots)
    assert callable(plugin.render_plot)
    assert callable(plugin.get_plot_data)
    assert list(inspect.signature(plugin.list_plots).parameters) == []
    assert list(inspect.signature(plugin.render_plot).parameters) == ["name", "ax"]
    assert list(inspect.signature(plugin.get_plot_data).parameters) == ["name"]


def test_list_plots_returns_three_named_descriptors() -> None:
    infos = _plugin().list_plots()
    assert [i.name for i in infos] == list(_PLOT_NAMES)
    assert all(isinstance(i, PlotInfo) and i.title for i in infos)


def test_list_plots_returns_fresh_copies() -> None:
    first = _plugin().list_plots()
    first[0].title = "MUTATED"
    assert _plugin().list_plots()[0].title != "MUTATED"


def test_logged_metrics_whitelist_is_locked() -> None:
    assert _LOGGED_METRICS == (
        "pool_best",
        "pool_median",
        "pool_worst",
        "finetune_loss",
    )


def test_plot_metric_keys_are_within_the_whitelist() -> None:
    used = set(eqgpt_viz._PLOT_METRIC.values()) | set(eqgpt_viz._SPREAD_METRICS)
    assert used <= set(_LOGGED_METRICS)







@pytest.mark.parametrize("bad", ["", "bogus", "reward", "Reward_Convergence"])
def test_render_unknown_name_raises_valueerror(ax, bad) -> None:
    with pytest.raises(ValueError, match="Unknown plot name"):
        _plugin().render_plot(bad, ax)


def test_get_plot_data_unknown_name_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="Unknown plot name"):
        _plugin().get_plot_data("bogus")


def test_unknown_name_message_lists_all_three_available(ax) -> None:
    with pytest.raises(ValueError) as excinfo:
        eqgpt_viz.render("bogus", ax, None)
    msg = str(excinfo.value)
    for name in _PLOT_NAMES:
        assert name in msg, name







@pytest.mark.parametrize("name", _PLOT_NAMES)
def test_render_plot_returns_none(name, ax) -> None:
    plugin = _plugin()
    plugin.prepare(_components(_disjoint_recorder()))
    assert plugin.render_plot(name, ax) is None







@pytest.mark.parametrize("name", _SINGLE_PLOTS)
def test_render_single_series_binds_to_its_metric(name, ax) -> None:
    recorder = _disjoint_recorder()
    eqgpt_viz.render(name, ax, recorder)
    assert len(ax.lines) == 1
    assert _ydata(ax.lines[0]) == pytest.approx(recorder.get(_SINGLE_METRIC[name]))
    assert ax.get_ylabel() == _SINGLE_METRIC[name]


def test_render_pool_reward_spread_draws_three_bound_lines(ax) -> None:
    recorder = _disjoint_recorder()
    eqgpt_viz.render("pool_reward_spread", ax, recorder)
    assert len(ax.lines) == 3
    for line, key in zip(ax.lines, _SPREAD_KEYS, strict=True):
        assert _ydata(line) == pytest.approx(recorder.get(key)), key


def test_reward_convergence_flat_curve_gets_subtitle(ax) -> None:
    recorder = VizRecorder(enabled=True)
    for _ in range(_N_EPOCHS):
        recorder.log("pool_best", 0.9486)
    eqgpt_viz.render("reward_convergence", ax, recorder)
    assert len(ax.lines) == 1
    assert "constant" in ax.get_title().lower()
    assert not any("constant" in t.get_text().lower() for t in ax.texts)


def test_reward_convergence_rising_curve_has_plain_title(ax) -> None:
    eqgpt_viz.render("reward_convergence", ax, _disjoint_recorder())
    assert "constant" not in ax.get_title().lower()













_GAP_EPOCHS = 5


def _empty_pool_recorder() -> VizRecorder:
    recorder = VizRecorder(enabled=True)
    plugin = _plugin()
    plugin.prepare(_components(recorder))
    for _ in range(_GAP_EPOCHS):
        plugin.update([])
    return recorder


def _gapped_recorder(pool_best: list[float]) -> VizRecorder:
    recorder = VizRecorder(enabled=True)
    for value in pool_best:
        recorder.log("pool_best", value)
    return recorder


def test_empty_pool_epochs_record_gaps_not_measured_zeros() -> None:
    recorder = _empty_pool_recorder()
    for metric in _LOGGED_METRICS:
        series = recorder.get(metric)
        assert len(series) == _GAP_EPOCHS, metric
        assert all(math.isnan(value) for value in series), metric
        assert 0.0 not in series, metric


def test_empty_pool_gaps_export_as_json_null() -> None:
    recorder = _empty_pool_recorder()
    assert recorder.to_dict()["pool_best"] == [None] * _GAP_EPOCHS

    convergence = eqgpt_viz.get_data("reward_convergence", recorder)
    assert convergence["y"] == [None] * _GAP_EPOCHS
    spread = eqgpt_viz.get_data("pool_reward_spread", recorder)
    for key in _SPREAD_KEYS:
        assert spread["y"][key] == [None] * _GAP_EPOCHS, key
    json.dumps([convergence, spread])


def test_measured_zero_reward_stays_a_measured_zero() -> None:
    recorder = VizRecorder(enabled=True)
    plugin = _plugin()
    plugin.prepare(_components(recorder))
    scored = plugin.evaluate(plugin.propose(_BATCH))
    zeroed = [dataclasses.replace(r, score=0.0) for r in scored if r.is_valid]
    assert zeroed

    plugin.update(zeroed)

    for key in _SPREAD_KEYS:
        assert recorder.get(key) == [0.0], key
    assert eqgpt_viz.get_data("reward_convergence", recorder)["y"] == [0.0]

    assert all(math.isfinite(v) for v in recorder.get("finetune_loss"))


def test_reward_convergence_all_gap_series_is_not_called_constant(ax) -> None:
    eqgpt_viz.render(
        "reward_convergence", ax, _gapped_recorder([float("nan")] * _GAP_EPOCHS)
    )
    title = ax.get_title().lower()
    assert "constant" not in title
    assert "no pool" in title
    assert all(math.isnan(value) for value in _ydata(ax.lines[0]))


def test_reward_convergence_partial_gap_does_not_claim_the_first_epoch(ax) -> None:
    nan = float("nan")
    eqgpt_viz.render("reward_convergence", ax, _gapped_recorder([nan, nan, 0.9, 0.9]))
    title = ax.get_title().lower()
    assert "0.9" in title
    assert "first epoch" not in title
    assert "no pool" in title


def test_reward_convergence_moving_series_with_gaps_reports_the_gaps(ax) -> None:
    eqgpt_viz.render(
        "reward_convergence", ax, _gapped_recorder([float("nan"), 0.1, 0.5, 0.9])
    )
    title = ax.get_title().lower()
    assert "constant" not in title
    assert "no pool" in title


def test_spread_band_draws_gaps_not_zeros(ax) -> None:
    recorder = VizRecorder(enabled=True)
    for key, measured in zip(_SPREAD_KEYS, (0.9, 0.5, 0.1), strict=True):
        recorder.log(key, float("nan"))
        recorder.log(key, measured)

    eqgpt_viz.render("pool_reward_spread", ax, recorder)
    assert len(ax.lines) == 3
    for line, measured in zip(ax.lines, (0.9, 0.5, 0.1), strict=True):
        ydata = _ydata(line)
        assert math.isnan(ydata[0])
        assert ydata[1] == pytest.approx(measured)

    data = eqgpt_viz.get_data("pool_reward_spread", recorder)
    for key, measured in zip(_SPREAD_KEYS, (0.9, 0.5, 0.1), strict=True):
        assert data["y"][key] == [None, pytest.approx(measured)], key
    json.dumps(data)


def test_spread_band_pads_a_short_series_with_gaps(ax) -> None:
    recorder = VizRecorder(enabled=True)
    recorder.log("pool_best", float("nan"))
    recorder.log("pool_best", 0.9)
    recorder.log("pool_median", 0.5)
    recorder.log("pool_worst", 0.1)

    data = eqgpt_viz.get_data("pool_reward_spread", recorder)
    assert data["x"] == [0, 1]
    assert data["y"]["pool_best"] == [None, pytest.approx(0.9)]
    assert data["y"]["pool_median"] == [pytest.approx(0.5), None]
    assert data["y"]["pool_worst"] == [pytest.approx(0.1), None]
    eqgpt_viz.render("pool_reward_spread", ax, recorder)
    assert len(ax.lines) == 3











def _band_recorder(**series: list[float]) -> VizRecorder:
    recorder = VizRecorder(enabled=True)
    for key, values in series.items():
        for value in values:
            recorder.log(key, value)
    return recorder


def _subtitle(ax) -> str:
    lines = ax.get_title().split("\n")
    return lines[1] if len(lines) > 1 else ""


def test_spread_band_all_gap_epochs_are_disclosed(ax) -> None:
    eqgpt_viz.render("pool_reward_spread", ax, _empty_pool_recorder())
    assert len(ax.lines) == 3
    assert all(math.isnan(v) for v in _ydata(ax.lines[0]))
    subtitle = _subtitle(ax)
    assert "no pool" in subtitle
    assert str(_GAP_EPOCHS) in subtitle


def test_spread_band_partial_gaps_are_counted(ax) -> None:
    nan = float("nan")
    eqgpt_viz.render(
        "pool_reward_spread",
        ax,
        _band_recorder(
            pool_best=[nan, 0.9, 0.9],
            pool_median=[nan, 0.5, 0.5],
            pool_worst=[nan, 0.1, 0.1],
        ),
    )
    assert "1 of 3 epochs had no pool" in _subtitle(ax)


def test_spread_band_padding_is_not_reported_as_an_empty_pool(ax) -> None:
    eqgpt_viz.render(
        "pool_reward_spread",
        ax,
        _band_recorder(pool_best=[0.9, 0.8], pool_median=[0.5], pool_worst=[0.1]),
    )
    assert _subtitle(ax) == ""


def test_finetune_loss_all_gap_names_the_fine_tune_not_the_pool(ax) -> None:
    eqgpt_viz.render("finetune_loss", ax, _empty_pool_recorder())
    subtitle = _subtitle(ax)
    assert "fine-tune" in subtitle
    assert "no pool" not in subtitle
    assert str(_GAP_EPOCHS) in subtitle


def test_finetune_loss_partial_gap_counts_the_skipped_epochs(ax) -> None:
    eqgpt_viz.render(
        "finetune_loss", ax, _band_recorder(finetune_loss=[float("nan"), 2.0, 1.0])
    )
    assert "1 of 3 epochs" in _subtitle(ax)


@pytest.mark.parametrize("name", _PLOT_NAMES)
def test_gapless_panel_keeps_its_plain_title(name, ax) -> None:
    eqgpt_viz.render(name, ax, _disjoint_recorder())
    titles = {info.name: info.title for info in eqgpt_viz.list_plot_infos()}
    assert ax.get_title() == titles[name]







def _degraded_recorders() -> list[VizRecorder | None]:
    return [VizRecorder(enabled=True), None, VizRecorder(enabled=False)]


@pytest.mark.parametrize("name", _PLOT_NAMES)
@pytest.mark.parametrize("recorder", _degraded_recorders())
def test_render_degrades_with_no_data_panel_and_no_lines(name, recorder, ax) -> None:
    eqgpt_viz.render(name, ax, recorder)
    assert len(ax.lines) == 0
    assert any("no data" in txt.get_text().lower() for txt in ax.texts)







@pytest.mark.parametrize("name", _SINGLE_PLOTS)
def test_get_data_single_series_binds_flat_list(name) -> None:
    recorder = _disjoint_recorder()
    data = eqgpt_viz.get_data(name, recorder)
    assert set(data) >= {"x", "y", "xlabel", "ylabel", "title"}
    assert isinstance(data["y"], list)
    assert data["y"] == pytest.approx(recorder.get(_SINGLE_METRIC[name]))
    assert len(data["x"]) == len(data["y"])
    assert data["ylabel"] == _SINGLE_METRIC[name]
    json.dumps(data)


def test_get_data_pool_reward_spread_is_dict_of_three_series() -> None:
    recorder = _disjoint_recorder()
    data = eqgpt_viz.get_data("pool_reward_spread", recorder)
    assert isinstance(data["y"], dict)
    assert set(data["y"]) == set(_SPREAD_KEYS)
    for key in _SPREAD_KEYS:
        assert data["y"][key] == pytest.approx(recorder.get(key)), key
        assert len(data["y"][key]) == len(data["x"])
    json.dumps(data)







@pytest.mark.parametrize("name", _SINGLE_PLOTS)
@pytest.mark.parametrize("recorder", _degraded_recorders())
def test_get_data_single_series_degrades_to_empty(name, recorder) -> None:
    data = eqgpt_viz.get_data(name, recorder)
    assert data["x"] == [] and data["y"] == []


@pytest.mark.parametrize("recorder", _degraded_recorders())
def test_get_data_spread_degrades_to_three_empty_series(recorder) -> None:
    data = eqgpt_viz.get_data("pool_reward_spread", recorder)
    assert data["x"] == []
    assert set(data["y"]) == set(_SPREAD_KEYS)
    assert all(series == [] for series in data["y"].values())







def test_get_data_sanitizes_non_finite_to_none() -> None:
    recorder = VizRecorder(enabled=True)
    for value in (5.0, float("nan"), float("inf"), float("-inf"), 6.0):
        recorder.log("finetune_loss", value)
    data = eqgpt_viz.get_data("finetune_loss", recorder)
    assert None in data["y"]
    assert all(v is None or isinstance(v, float) for v in data["y"])
    json.dumps(data)







def test_prepare_captures_the_components_recorder() -> None:
    recorder = VizRecorder(enabled=True)
    plugin = _plugin()
    plugin.prepare(_components(recorder))
    assert plugin._recorder is recorder


def test_update_logs_the_whitelisted_metrics_per_epoch() -> None:
    recorder = VizRecorder(enabled=True)
    plugin = _plugin()
    plugin.prepare(_components(recorder))
    for _ in range(2):
        plugin.update(plugin.evaluate(plugin.propose(_BATCH)))
    for metric in _LOGGED_METRICS:
        assert len(recorder.get(metric)) == 2, metric







def test_finetune_returns_mean_ce_over_steps() -> None:
    pool = [[7, 2, 8], [7, 2, 20]]

    def _fresh():
        backend = FakeGPTBackend(57, seed=0)
        return backend, torch.optim.Adam(backend.parameters(), lr=1e-3)

    backend, optimizer = _fresh()
    l1 = _scoring.finetune(backend, optimizer, pool, 1)
    assert isinstance(l1, float)
    l2 = _scoring.finetune(backend, optimizer, pool, 1)
    assert l1 != pytest.approx(l2)

    fresh_backend, fresh_optimizer = _fresh()
    mean_two = _scoring.finetune(fresh_backend, fresh_optimizer, pool, 2)
    assert mean_two == pytest.approx((l1 + l2) / 2, rel=1e-5)


def test_finetune_empty_pool_returns_zero_or_none() -> None:
    backend = FakeGPTBackend(57, seed=0)
    optimizer = torch.optim.Adam(backend.parameters(), lr=1e-5)
    result = _scoring.finetune(backend, optimizer, [], 3)
    assert result is None or result == 0.0







def test_invalid_final_result_has_none_residuals() -> None:


    result = _scoring.invalid_final_result("no candidates", best_reward=0.0)
    assert result.is_valid is False
    assert result.residuals is None


def test_empty_pool_build_final_result_is_invalid_with_none_residuals() -> None:
    plugin = _plugin()
    plugin.prepare(_components())
    final = plugin.build_final_result()
    assert final.is_valid is False
    assert final.residuals is None








_PER_CASE_NAME = "per_case_reward"


class _FakeMulticase:

    primary_case = "N_case0"

    def __init__(self, per_case: dict[str, float]) -> None:
        self._per_case = per_case
        self.terms_seen: list[str] | None = None

    def per_case_rewards(self, terms: list[str]) -> dict[str, float]:
        self.terms_seen = list(terms)
        return dict(self._per_case)


def _wave_plugin(per_case, monkeypatch, *, terms=("u_x",)):
    plugin = _plugin()
    plugin._multicase = _FakeMulticase(per_case)
    monkeypatch.setattr(plugin, "_best_terms", lambda: list(terms) if terms else None)
    return plugin


def test_list_plots_wave_mode_appends_per_case_panel(monkeypatch) -> None:
    names = [i.name for i in _wave_plugin({"N_a": 0.9}, monkeypatch).list_plots()]
    assert names == [*_PLOT_NAMES, _PER_CASE_NAME]


def test_list_plots_wave_panel_is_a_fresh_copy(monkeypatch) -> None:
    plugin = _wave_plugin({"N_a": 0.9}, monkeypatch)
    plugin.list_plots()[-1].title = "MUTATED"
    assert plugin.list_plots()[-1].title != "MUTATED"


def test_single_case_list_plots_has_no_per_case_panel() -> None:

    assert [i.name for i in _plugin().list_plots()] == list(_PLOT_NAMES)


def test_render_plot_per_case_draws_a_bar_per_case(monkeypatch, ax) -> None:
    plugin = _wave_plugin({"N_a": 0.94, "N_b": 0.92, "N_c": 0.95}, monkeypatch)
    assert plugin.render_plot(_PER_CASE_NAME, ax) is None
    assert len(ax.patches) == 3
    assert [round(p.get_height(), 2) for p in ax.patches] == [0.94, 0.92, 0.95]


def test_render_plot_per_case_forwards_winning_terms(monkeypatch, ax) -> None:
    plugin = _wave_plugin({"N_a": 0.9}, monkeypatch, terms=("u_x", "u_xxx"))
    plugin.render_plot(_PER_CASE_NAME, ax)
    assert plugin._multicase.terms_seen == ["u_x", "u_xxx"]


def test_render_plot_per_case_no_candidate_degrades(monkeypatch, ax) -> None:
    plugin = _wave_plugin({"N_a": 0.9}, monkeypatch, terms=())
    plugin.render_plot(_PER_CASE_NAME, ax)
    assert len(ax.patches) == 0


    texts = [t.get_text() for t in ax.texts]
    assert any("No data" in t and "candidate" in t for t in texts), texts


def test_render_plot_per_case_unknown_without_multicase(ax) -> None:

    with pytest.raises(ValueError, match="Unknown plot name"):
        _plugin().render_plot(_PER_CASE_NAME, ax)


def test_get_plot_data_per_case_shape_and_nan(monkeypatch) -> None:
    plugin = _wave_plugin({"N_a": 0.9, "N_b": float("nan")}, monkeypatch)
    data = plugin.get_plot_data(_PER_CASE_NAME)
    assert data["x"] == ["N_a", "N_b"]
    assert data["y"] == [0.9, None]
    assert data["xlabel"] == "case"
    assert data["ylabel"] == "reward"


def test_render_per_case_reward_marks_nan_case(ax) -> None:
    eqgpt_viz.render_per_case_reward(ax, {"N_a": 0.9, "N_b": float("nan"), "N_c": 0.8})
    assert math.isnan(ax.patches[1].get_height())
    assert any(t.get_text() == "n/a" for t in ax.texts)


def test_render_per_case_reward_na_mark_visible_when_all_rewards_negative(ax) -> None:
    eqgpt_viz.render_per_case_reward(
        ax, {"N_a": -5.0, "N_b": float("nan"), "N_c": -6.0}
    )
    ax.figure.canvas.draw()
    marks = [t for t in ax.texts if t.get_text() == "n/a"]
    assert len(marks) == 1
    text_box = marks[0].get_window_extent()
    axes_box = ax.get_window_extent()
    assert axes_box.contains(text_box.x0, text_box.y0)
    assert axes_box.contains(text_box.x1, text_box.y1)


@pytest.mark.parametrize("per_case", [{}, {"N_a": float("nan")}])
def test_render_per_case_reward_degrades_to_no_data_panel(per_case, ax) -> None:
    eqgpt_viz.render_per_case_reward(ax, per_case)
    assert len(ax.patches) == 0
    assert any("No data" in t.get_text() for t in ax.texts)


def test_per_case_data_sanitizes_nan_to_none() -> None:
    assert eqgpt_viz.per_case_data({"N_a": 0.9, "N_b": float("nan")}) == {
        "x": ["N_a", "N_b"],
        "y": [0.9, None],
        "xlabel": "case",
        "ylabel": "reward",
        "title": "Per-case Reward",
    }
