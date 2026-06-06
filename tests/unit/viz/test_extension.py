
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from kd.viz.extension import PlotInfo, VizExtension




class TestPlotInfo:

    def test_minimal_creation(self) -> None:
        info = PlotInfo(name="test_plot", title="Test Plot")
        assert info.name == "test_plot"
        assert info.title == "Test Plot"
        assert info.description == ""

    def test_with_description(self) -> None:
        info = PlotInfo(
            name="aic", title="AIC Landscape", description="AIC vs complexity"
        )
        assert info.description == "AIC vs complexity"





class _MockExtension:

    def list_plots(self) -> list[PlotInfo]:
        return [
            PlotInfo(name="custom_a", title="Custom A"),
            PlotInfo(name="custom_b", title="Custom B", description="second plot"),
        ]

    def render_plot(self, name: str, ax: Any) -> None:
        ax.plot([1, 2, 3], [1, 2, 3], label=name)

    def get_plot_data(self, name: str) -> Any:
        return {"name": name, "values": [1, 2, 3]}


class TestVizExtensionProtocol:

    def test_mock_satisfies_protocol(self) -> None:
        ext = _MockExtension()
        assert isinstance(ext, VizExtension)

    def test_list_plots_returns_plot_info(self) -> None:
        ext = _MockExtension()
        plots = ext.list_plots()
        assert len(plots) == 2
        assert all(isinstance(p, PlotInfo) for p in plots)

    def test_render_plot_draws_on_axes(self) -> None:
        ext = _MockExtension()
        fig, ax = plt.subplots()
        ext.render_plot("custom_a", ax)
        assert len(ax.get_lines()) >= 1
        plt.close(fig)

    def test_get_plot_data_returns_data(self) -> None:
        ext = _MockExtension()
        data = ext.get_plot_data("custom_a")
        assert isinstance(data, dict)
        assert data["name"] == "custom_a"

    def test_non_implementing_class_fails_check(self) -> None:
        class NotAnExtension:
            pass

        assert not isinstance(NotAnExtension(), VizExtension)
