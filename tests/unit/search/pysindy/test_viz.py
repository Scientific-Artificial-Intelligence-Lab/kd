
from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from kd.search.pysindy import viz as pysindy_viz
from kd.search.pysindy.plugin import PySINDyPlugin
from kd.search.recorder import VizRecorder, log_whitelisted_metrics
from kd.viz.extension import PlotInfo, VizExtension

pytestmark = pytest.mark.unit

_PLOT_NAME = "native_refit_agreement"


def _populated_recorder(
    *,
    native: float | None = 1e-3,
    refit: float | None = 2e-3,
    support: int = 2,
) -> VizRecorder:
    recorder = VizRecorder()
    log_whitelisted_metrics(
        recorder,
        pysindy_viz.LOGGED_METRICS,
        {
            pysindy_viz.NATIVE_NMSE_KEY: native,
            pysindy_viz.REFIT_NMSE_KEY: refit,
            pysindy_viz.SUPPORT_SIZE_KEY: support,
        },
    )
    return recorder


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


class TestPlotCatalog:

    def test_single_agreement_plot_info(self) -> None:
        infos = pysindy_viz.list_plot_infos()
        assert len(infos) == 1
        assert isinstance(infos[0], PlotInfo)
        assert infos[0].name == _PLOT_NAME
        assert infos[0].title
        assert infos[0].description

    def test_plugin_satisfies_viz_extension(self) -> None:
        plugin = PySINDyPlugin()
        assert isinstance(plugin, VizExtension)
        assert [info.name for info in plugin.list_plots()] == [_PLOT_NAME]

    def test_render_unknown_name_raises(self, ax) -> None:
        with pytest.raises(ValueError, match="bogus"):
            pysindy_viz.render("bogus", ax, None)

    def test_get_data_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="bogus"):
            pysindy_viz.get_data("bogus", None)


class TestRender:

    def test_no_recorder_draws_no_data_panel(self, ax) -> None:
        pysindy_viz.render(_PLOT_NAME, ax, None)
        texts = [t.get_text() for t in ax.texts]
        assert any("No data" in t for t in texts)
        assert len(ax.patches) == 0

    def test_unlogged_recorder_draws_no_data_panel(self, ax) -> None:
        pysindy_viz.render(_PLOT_NAME, ax, VizRecorder())
        texts = [t.get_text() for t in ax.texts]
        assert any("No data" in t for t in texts)

    def test_two_bars_log_scale_and_support_note(self, ax) -> None:
        pysindy_viz.render(_PLOT_NAME, ax, _populated_recorder(support=3))
        assert len(ax.patches) == 2
        assert ax.get_yscale() == "log"
        texts = " ".join(t.get_text() for t in ax.texts)
        assert "support size: 3" in texts

    def test_invalid_refit_drops_bar_and_says_so(self, ax) -> None:
        pysindy_viz.render(_PLOT_NAME, ax, _populated_recorder(refit=None))
        assert len(ax.patches) == 1
        texts = " ".join(t.get_text() for t in ax.texts)
        assert "refit invalid" in texts

    def test_zero_nmse_falls_back_to_linear_scale(self, ax) -> None:
        pysindy_viz.render(_PLOT_NAME, ax, _populated_recorder(native=0.0))
        assert len(ax.patches) == 2
        assert ax.get_yscale() == "linear"


class TestGetData:

    def test_populated_data_round_trips_json(self) -> None:
        data = pysindy_viz.get_data(_PLOT_NAME, _populated_recorder(support=2))
        assert data[pysindy_viz.NATIVE_NMSE_KEY] == pytest.approx(1e-3)
        assert data[pysindy_viz.REFIT_NMSE_KEY] == pytest.approx(2e-3)
        assert data[pysindy_viz.SUPPORT_SIZE_KEY] == 2
        assert data["title"]
        json.dumps(data, allow_nan=False)

    def test_non_finite_scores_become_none(self) -> None:
        data = pysindy_viz.get_data(
            _PLOT_NAME, _populated_recorder(native=math.nan, refit=math.inf)
        )
        assert data[pysindy_viz.NATIVE_NMSE_KEY] is None
        assert data[pysindy_viz.REFIT_NMSE_KEY] is None
        json.dumps(data, allow_nan=False)

    def test_none_recorder_degrades_to_all_none(self) -> None:
        data = pysindy_viz.get_data(_PLOT_NAME, None)
        assert data[pysindy_viz.NATIVE_NMSE_KEY] is None
        assert data[pysindy_viz.REFIT_NMSE_KEY] is None
        assert data[pysindy_viz.SUPPORT_SIZE_KEY] is None
        json.dumps(data, allow_nan=False)
