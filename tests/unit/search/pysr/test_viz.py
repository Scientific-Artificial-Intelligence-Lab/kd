
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from kd.search.protocol import PlatformComponents


from kd.search.pysr import plugin as pysr_plugin
from kd.search.pysr import viz as pysr_viz
from kd.search.pysr.config import PySRConfig
from kd.search.pysr.plugin import PySRPlugin
from kd.search.recorder import VizRecorder
from kd.viz.extension import PlotInfo
from tests.unit.search.pysr.conftest import FakePySRBackend, make_backend_factory

pytestmark = pytest.mark.unit

_PLOT_NAMES = ("pareto_front", "kd_audit_path", "score_agreement")
_TERMS = ("u", "u_x", "u_xx")


_PARETO_COMPLEXITY_KEY = "pareto_complexity"
_PARETO_LOSS_KEY = "pareto_loss"
_PARETO_NMSE_KEY = "pareto_nmse"
_SELECTED_COMPLEXITY_KEY = "selected_complexity"
_SELECTED_LOSS_KEY = "selected_loss"
_SELECTED_NMSE_KEY = "selected_nmse"







def _fitted_plugin_with_recorder(
    components: PlatformComponents,
) -> tuple[PySRPlugin, VizRecorder]:
    recorder = components.recorder
    assert isinstance(recorder, VizRecorder)
    backend = FakePySRBackend()
    config = PySRConfig(terms=_TERMS, seed=0)
    plugin = PySRPlugin(config, backend_factory=make_backend_factory(backend))
    plugin.prepare(components)
    results = plugin.evaluate(plugin.propose(1))
    plugin.update(results)
    return plugin, recorder


def _populated_recorder() -> VizRecorder:
    recorder = VizRecorder()
    recorder.log(_PARETO_COMPLEXITY_KEY, [1, 3, 5])
    recorder.log(_PARETO_LOSS_KEY, [0.5, 0.05, 0.001])
    recorder.log(_PARETO_NMSE_KEY, [0.6, 0.04, 0.002])


    recorder.log(_SELECTED_COMPLEXITY_KEY, 3)
    recorder.log(_SELECTED_LOSS_KEY, 0.05)
    recorder.log(_SELECTED_NMSE_KEY, 0.04)
    return recorder







class TestListPlots:

    @pytest.mark.smoke
    def test_returns_three_plots(self) -> None:
        infos = pysr_viz.list_plot_infos()
        assert len(infos) == 3

    def test_all_are_plot_infos(self) -> None:
        for info in pysr_viz.list_plot_infos():
            assert isinstance(info, PlotInfo)

    def test_names_and_order(self) -> None:
        names = tuple(info.name for info in pysr_viz.list_plot_infos())
        assert names == _PLOT_NAMES

    def test_returns_fresh_copies(self) -> None:
        first = pysr_viz.list_plot_infos()
        first[0].title = "tampered"
        second = pysr_viz.list_plot_infos()
        assert second[0].title != "tampered"







class TestGetPlotDataRealPlugin:

    def test_pareto_front_data_shape(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        _plugin, recorder = _fitted_plugin_with_recorder(real_pysr_components)
        data = pysr_viz.get_data("pareto_front", recorder)
        assert set(data) >= {"x", "y", "xlabel", "ylabel", "title"}
        assert len(data["x"]) >= 1
        assert len(data["x"]) == len(data["y"])

    def test_kd_audit_path_uses_nmse(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        _plugin, recorder = _fitted_plugin_with_recorder(real_pysr_components)
        data = pysr_viz.get_data("kd_audit_path", recorder)
        assert len(data["x"]) == len(data["y"])
        assert len(data["y"]) >= 1

    def test_score_agreement_pairs_loss_and_nmse(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        _plugin, recorder = _fitted_plugin_with_recorder(real_pysr_components)
        data = pysr_viz.get_data("score_agreement", recorder)
        assert len(data["x"]) == len(data["y"])
        assert len(data["x"]) >= 1

    def test_data_lengths_match_pareto_front_size(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        _plugin, recorder = _fitted_plugin_with_recorder(real_pysr_components)
        front_size = len(recorder.get("pareto_complexity")[-1])
        for name in _PLOT_NAMES:
            data = pysr_viz.get_data(name, recorder)
            assert len(data["x"]) == front_size
            assert len(data["y"]) == front_size

    def test_data_is_json_safe(self, real_pysr_components: PlatformComponents) -> None:
        _plugin, recorder = _fitted_plugin_with_recorder(real_pysr_components)
        for name in _PLOT_NAMES:
            json.dumps(pysr_viz.get_data(name, recorder))







class TestGetPlotDataAppendSemantics:

    def test_takes_last_logged_list(self) -> None:
        recorder = _populated_recorder()
        data = pysr_viz.get_data("pareto_front", recorder)

        assert data["x"] == [1, 3, 5]
        assert data["y"] == [0.5, 0.05, 0.001]

    def test_reads_most_recent_when_logged_twice(self) -> None:
        recorder = _populated_recorder()

        recorder.log("pareto_complexity", [2, 4])
        recorder.log("pareto_loss", [0.3, 0.02])
        recorder.log("pareto_nmse", [0.25, 0.01])
        recorder.log("selected_complexity", 4)
        data = pysr_viz.get_data("pareto_front", recorder)
        assert data["x"] == [2, 4]
        assert data["y"] == [0.3, 0.02]

    def test_pareto_front_marks_selected(self) -> None:
        recorder = _populated_recorder()
        data = pysr_viz.get_data("pareto_front", recorder)
        assert data["selected_x"] == 3
        assert data["selected_y"] == 0.05

    def test_kd_audit_path_marks_selected(self) -> None:
        recorder = _populated_recorder()
        data = pysr_viz.get_data("kd_audit_path", recorder)
        assert data["selected_x"] == 3
        assert data["selected_y"] == 0.04

    def test_selected_none_omits_marker(self) -> None:
        recorder = VizRecorder()
        recorder.log(_PARETO_COMPLEXITY_KEY, [1, 3, 5])
        recorder.log(_PARETO_LOSS_KEY, [0.5, 0.05, 0.001])
        recorder.log(_PARETO_NMSE_KEY, [0.6, 0.04, 0.002])
        recorder.log(_SELECTED_COMPLEXITY_KEY, None)
        recorder.log(_SELECTED_LOSS_KEY, None)
        recorder.log(_SELECTED_NMSE_KEY, None)
        data = pysr_viz.get_data("pareto_front", recorder)

        assert data["x"] == [1, 3, 5]
        assert data.get("selected_x") is None
        assert data.get("selected_y") is None

    def test_render_selected_none_no_raise(self) -> None:
        recorder = VizRecorder()
        recorder.log(_PARETO_COMPLEXITY_KEY, [1, 3, 5])
        recorder.log(_PARETO_LOSS_KEY, [0.5, 0.05, 0.001])
        recorder.log(_PARETO_NMSE_KEY, [0.6, 0.04, 0.002])
        recorder.log(_SELECTED_COMPLEXITY_KEY, None)
        recorder.log(_SELECTED_LOSS_KEY, None)
        recorder.log(_SELECTED_NMSE_KEY, None)
        fig, ax = plt.subplots()
        try:
            assert pysr_viz.render("pareto_front", ax, recorder) == []
        finally:
            plt.close(fig)







def _recorder_with_nmse(nmse: list[object]) -> VizRecorder:
    recorder = VizRecorder()
    recorder.log(_PARETO_COMPLEXITY_KEY, [1, 3, 5])
    recorder.log(_PARETO_LOSS_KEY, [0.5, 0.05, 0.001])
    recorder.log(_PARETO_NMSE_KEY, list(nmse))
    return recorder


class TestNonFinitePairsDropped:

    def test_kd_audit_path_drops_none_nmse_pair(self) -> None:
        recorder = _recorder_with_nmse([0.6, None, 0.002])
        data = pysr_viz.get_data("kd_audit_path", recorder)
        assert len(data["x"]) == 2
        assert len(data["x"]) == len(data["y"])
        assert None not in data["y"]

        assert data["x"] == [1, 5]
        assert data["y"] == [0.6, 0.002]

    def test_score_agreement_drops_none_nmse_pair(self) -> None:
        recorder = _recorder_with_nmse([0.6, None, 0.002])
        data = pysr_viz.get_data("score_agreement", recorder)
        assert len(data["x"]) == 2
        assert len(data["x"]) == len(data["y"])
        assert None not in data["y"]

        assert data["x"] == [0.5, 0.001]
        assert data["y"] == [0.6, 0.002]

    def test_inf_nmse_pair_dropped(self) -> None:
        recorder = _recorder_with_nmse([0.6, float("inf"), 0.002])
        data = pysr_viz.get_data("kd_audit_path", recorder)
        assert len(data["x"]) == 2
        assert len(data["x"]) == len(data["y"])
        assert all(value is not None for value in data["y"])

    def test_finite_penalty_not_special_cased(self) -> None:
        recorder = _recorder_with_nmse([0.6, 0.04, 0.002])
        data = pysr_viz.get_data("kd_audit_path", recorder)
        assert len(data["x"]) == 3
        assert len(data["y"]) == 3

    def test_render_with_none_nmse_no_raise(self) -> None:
        recorder = _recorder_with_nmse([0.6, None, 0.002])
        for name in ("kd_audit_path", "score_agreement"):
            fig, ax = plt.subplots()
            try:
                warnings = pysr_viz.render(name, ax, recorder)
                assert len(warnings) == 1


                assert "1 of 3" in warnings[0] and "dropped" in warnings[0]
            finally:
                plt.close(fig)

    def test_render_with_inf_nmse_no_raise(self) -> None:
        recorder = _recorder_with_nmse([0.6, float("inf"), 0.002])
        fig, ax = plt.subplots()
        try:
            warnings = pysr_viz.render("score_agreement", ax, recorder)
            assert len(warnings) == 1

            assert "1 of 3" in warnings[0] and "dropped" in warnings[0]
        finally:
            plt.close(fig)

    def test_all_nmse_none_yields_empty_pairs(self) -> None:
        recorder = _recorder_with_nmse([None, None, None])
        data = pysr_viz.get_data("kd_audit_path", recorder)
        assert data["x"] == []
        assert data["y"] == []

    def test_pareto_front_drops_non_finite_loss_pair(self) -> None:
        recorder = VizRecorder()
        recorder.log(_PARETO_COMPLEXITY_KEY, [1, 3, 5])
        recorder.log(_PARETO_LOSS_KEY, [0.5, float("nan"), 0.001])
        recorder.log(_PARETO_NMSE_KEY, [0.6, 0.04, 0.002])
        data = pysr_viz.get_data("pareto_front", recorder)
        assert len(data["x"]) == 2
        assert len(data["x"]) == len(data["y"])
        assert data["x"] == [1, 5]
        assert data["y"] == [0.5, 0.001]







class TestSerializationRoundTrip:

    def test_roundtrip_inf_becomes_none_then_dropped(self) -> None:
        source = _recorder_with_nmse([0.6, float("inf"), 0.002])
        restored = VizRecorder.from_dict(source.to_dict())

        assert restored.get(_PARETO_NMSE_KEY)[-1] == [0.6, None, 0.002]
        data = pysr_viz.get_data("kd_audit_path", restored)
        assert len(data["x"]) == 2
        assert len(data["x"]) == len(data["y"])
        assert None not in data["y"]

    def test_roundtrip_get_data_is_json_safe(self) -> None:
        source = _recorder_with_nmse([0.6, float("inf"), 0.002])
        restored = VizRecorder.from_dict(source.to_dict())
        for name in _PLOT_NAMES:
            json.dumps(pysr_viz.get_data(name, restored))

    def test_roundtrip_render_no_raise(self) -> None:
        source = _recorder_with_nmse([0.6, float("inf"), 0.002])
        restored = VizRecorder.from_dict(source.to_dict())
        for name in _PLOT_NAMES:
            fig, ax = plt.subplots()
            try:
                warnings = pysr_viz.render(name, ax, restored)
                if name == "pareto_front":
                    assert warnings == []
                else:

                    assert len(warnings) == 1 and "1 of 3" in warnings[0]
            finally:
                plt.close(fig)

    def test_roundtrip_with_selected_inf_drops_marker(self) -> None:
        recorder = VizRecorder()
        recorder.log(_PARETO_COMPLEXITY_KEY, [1, 3, 5])
        recorder.log(_PARETO_LOSS_KEY, [0.5, 0.05, 0.001])
        recorder.log(_PARETO_NMSE_KEY, [0.6, 0.04, 0.002])
        recorder.log(_SELECTED_COMPLEXITY_KEY, 3)
        recorder.log(_SELECTED_LOSS_KEY, 0.05)
        recorder.log(_SELECTED_NMSE_KEY, float("inf"))
        restored = VizRecorder.from_dict(recorder.to_dict())
        data = pysr_viz.get_data("kd_audit_path", restored)

        assert data.get("selected_y") is None








class TestSelectedCoordinatesDirect:

    def test_pareto_front_uses_logged_loss(self) -> None:
        recorder = _populated_recorder()
        data = pysr_viz.get_data("pareto_front", recorder)
        assert data["selected_x"] == 3
        assert data["selected_y"] == 0.05

    def test_kd_audit_path_uses_logged_nmse(self) -> None:
        recorder = _populated_recorder()
        data = pysr_viz.get_data("kd_audit_path", recorder)
        assert data["selected_x"] == 3
        assert data["selected_y"] == 0.04

    def test_score_agreement_selected_pairs_loss_and_nmse(self) -> None:
        recorder = _populated_recorder()
        data = pysr_viz.get_data("score_agreement", recorder)
        assert data.get("selected_x") == 0.05
        assert data.get("selected_y") == 0.04

    def test_duplicate_complexity_picks_logged_not_first_match(self) -> None:
        recorder = VizRecorder()
        recorder.log(_PARETO_COMPLEXITY_KEY, [4, 4])
        recorder.log(_PARETO_LOSS_KEY, [0.40, 0.10])
        recorder.log(_PARETO_NMSE_KEY, [0.30, 0.02])
        recorder.log(_SELECTED_COMPLEXITY_KEY, 4)
        recorder.log(_SELECTED_LOSS_KEY, 0.10)
        recorder.log(_SELECTED_NMSE_KEY, 0.02)
        data = pysr_viz.get_data("pareto_front", recorder)


        assert data["selected_y"] == 0.10
        assert data["selected_y"] != 0.40

    def test_audit_path_duplicate_complexity_uses_logged_nmse(self) -> None:
        recorder = VizRecorder()
        recorder.log(_PARETO_COMPLEXITY_KEY, [4, 4])
        recorder.log(_PARETO_LOSS_KEY, [0.40, 0.10])
        recorder.log(_PARETO_NMSE_KEY, [0.30, 0.02])
        recorder.log(_SELECTED_COMPLEXITY_KEY, 4)
        recorder.log(_SELECTED_LOSS_KEY, 0.10)
        recorder.log(_SELECTED_NMSE_KEY, 0.02)
        data = pysr_viz.get_data("kd_audit_path", recorder)
        assert data["selected_y"] == 0.02
        assert data["selected_y"] != 0.30

    def test_missing_selected_loss_omits_marker(self) -> None:
        recorder = VizRecorder()
        recorder.log(_PARETO_COMPLEXITY_KEY, [1, 3, 5])
        recorder.log(_PARETO_LOSS_KEY, [0.5, 0.05, 0.001])
        recorder.log(_PARETO_NMSE_KEY, [0.6, 0.04, 0.002])
        recorder.log(_SELECTED_COMPLEXITY_KEY, 3)
        recorder.log(_SELECTED_LOSS_KEY, None)
        recorder.log(_SELECTED_NMSE_KEY, 0.04)
        data = pysr_viz.get_data("pareto_front", recorder)
        assert data.get("selected_x") is None
        assert data.get("selected_y") is None

    def test_score_agreement_marker_omitted_when_nmse_none(self) -> None:
        recorder = VizRecorder()
        recorder.log(_PARETO_COMPLEXITY_KEY, [1, 3, 5])
        recorder.log(_PARETO_LOSS_KEY, [0.5, 0.05, 0.001])
        recorder.log(_PARETO_NMSE_KEY, [0.6, 0.04, 0.002])
        recorder.log(_SELECTED_COMPLEXITY_KEY, 3)
        recorder.log(_SELECTED_LOSS_KEY, 0.05)
        recorder.log(_SELECTED_NMSE_KEY, None)
        data = pysr_viz.get_data("score_agreement", recorder)
        assert data.get("selected_x") is None
        assert data.get("selected_y") is None







class TestGetPlotDataDegradation:

    def test_none_recorder_returns_empty(self) -> None:
        for name in _PLOT_NAMES:
            data = pysr_viz.get_data(name, None)
            assert data["x"] == []
            assert data["y"] == []

    def test_none_recorder_still_has_labels(self) -> None:
        data = pysr_viz.get_data("pareto_front", None)
        assert data["xlabel"]
        assert data["ylabel"]
        assert data["title"]

    def test_empty_recorder_returns_empty(self) -> None:
        for name in _PLOT_NAMES:
            data = pysr_viz.get_data(name, VizRecorder())
            assert data["x"] == []
            assert data["y"] == []

    def test_disabled_recorder_returns_empty(self) -> None:
        recorder = VizRecorder(enabled=False)
        recorder.log("pareto_complexity", [1, 3, 5])
        for name in _PLOT_NAMES:
            data = pysr_viz.get_data(name, recorder)
            assert data["x"] == []
            assert data["y"] == []

    def test_unknown_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            pysr_viz.get_data("does_not_exist", _populated_recorder())

    def test_unknown_name_raises_before_recorder_access(self) -> None:
        with pytest.raises(ValueError):
            pysr_viz.get_data("nope", None)







class TestRenderPlot:

    def test_render_returns_empty_warnings(self) -> None:
        recorder = _populated_recorder()
        fig, ax = plt.subplots()
        try:
            assert pysr_viz.render("pareto_front", ax, recorder) == []
        finally:
            plt.close(fig)

    def test_render_all_plots_no_raise(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        _plugin, recorder = _fitted_plugin_with_recorder(real_pysr_components)
        for name in _PLOT_NAMES:
            fig, ax = plt.subplots()
            try:
                pysr_viz.render(name, ax, recorder)
            finally:
                plt.close(fig)

    def test_render_sets_axis_labels(self) -> None:
        recorder = _populated_recorder()
        fig, ax = plt.subplots()
        try:
            pysr_viz.render("pareto_front", ax, recorder)
            assert ax.get_xlabel()
            assert ax.get_ylabel()
            assert ax.get_title()
        finally:
            plt.close(fig)

    def test_render_none_recorder_no_raise(self) -> None:
        fig, ax = plt.subplots()
        try:
            warnings = pysr_viz.render("pareto_front", ax, None)
            assert warnings and "No data" in warnings[0]
        finally:
            plt.close(fig)

    def test_render_unknown_name_raises(self) -> None:
        fig, ax = plt.subplots()
        try:
            with pytest.raises(ValueError):
                pysr_viz.render("bogus", ax, _populated_recorder())
        finally:
            plt.close(fig)







class TestPluginVizDelegation:

    def test_plugin_list_plots_matches_helper(self) -> None:
        backend = FakePySRBackend()
        plugin = PySRPlugin(
            PySRConfig(terms=_TERMS), backend_factory=make_backend_factory(backend)
        )
        names = tuple(info.name for info in plugin.list_plots())
        assert names == _PLOT_NAMES

    def test_plugin_get_plot_data_uses_its_recorder(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        plugin, _recorder = _fitted_plugin_with_recorder(real_pysr_components)
        data = plugin.get_plot_data("pareto_front")
        assert len(data["x"]) >= 1
        assert len(data["x"]) == len(data["y"])

    def test_plugin_render_plot_no_raise(
        self, real_pysr_components: PlatformComponents
    ) -> None:
        plugin, _recorder = _fitted_plugin_with_recorder(real_pysr_components)
        fig, ax = plt.subplots()
        try:
            plugin.render_plot("kd_audit_path", ax)
        finally:
            plt.close(fig)

    def test_plugin_get_plot_data_unknown_raises(self) -> None:
        backend = FakePySRBackend()
        plugin = PySRPlugin(
            PySRConfig(terms=_TERMS), backend_factory=make_backend_factory(backend)
        )
        with pytest.raises(ValueError):
            plugin.get_plot_data("not_a_plot")







def test_plot_spec_keys_are_within_the_logged_whitelist() -> None:
    used = {
        key
        for spec in pysr_viz._PLOT_SPECS
        for key in (spec.x_key, spec.y_key, spec.selected_x_key, spec.selected_y_key)
    }
    assert used <= set(pysr_plugin._LOGGED_METRICS)
