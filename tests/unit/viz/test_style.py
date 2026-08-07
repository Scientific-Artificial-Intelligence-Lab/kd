
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt

from kd.search.result import ExperimentResult
from kd.viz.engine import VizEngine
from kd.viz.plots.convergence import plot_convergence
from kd.viz.style import DEFAULT_STYLE, style_context


class TestStyleContext:

    def test_restores_original_params(self) -> None:
        original = mpl.rcParams.copy()
        with style_context():
            pass
        for key in DEFAULT_STYLE:
            assert mpl.rcParams[key] == original[key]

    def test_applies_default_style(self) -> None:
        with style_context():
            for key, value in DEFAULT_STYLE.items():
                assert mpl.rcParams[key] == value

    def test_applies_extra_style(self) -> None:
        extra = {"font.size": 99}
        with style_context(extra):
            assert mpl.rcParams["font.size"] == 99

    def test_restores_after_extra(self) -> None:
        original_size = mpl.rcParams["font.size"]
        with style_context({"font.size": 99}):
            pass
        assert mpl.rcParams["font.size"] == original_size

    def test_restores_on_exception(self) -> None:
        original_size = mpl.rcParams["font.size"]
        try:
            with style_context({"font.size": 99}):
                raise ValueError("test error")
        except ValueError:
            pass
        assert mpl.rcParams["font.size"] == original_size


class TestStyleOwnership:

    def test_direct_call_honors_style_on_created_artists(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        fig, ax = plt.subplots()
        try:
            plot_convergence(
                mock_experiment_result, ax, style={"axes.titlesize": 33.0}
            )
            assert ax.title.get_fontsize() == 33.0
        finally:
            plt.close(fig)

    def test_engine_axes_born_inside_style_context(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        captured: list[float] = []

        def _probe(
            result: ExperimentResult, ax: object, *, style: object = None
        ) -> list[str]:
            captured.append(ax.xaxis.label.get_fontsize())
            return []

        engine = VizEngine(output_dir=tmp_path, style={"axes.labelsize": 21.0})
        with patch("kd.viz.engine.plot_convergence", _probe):
            engine.render_universal(mock_experiment_result)
        assert captured and captured[0] == 21.0
