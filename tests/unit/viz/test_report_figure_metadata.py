
from __future__ import annotations

import html as html_lib
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.axes import Axes

from kd.core.equation import Scalar, make_homogeneous
from kd.search.result import ExperimentResult
from kd.viz.engine import VizEngine
from kd.viz.extension import PlotInfo
from kd.viz.report import FigureSpec, generate_report

pytestmark = pytest.mark.unit

_DECLARED_TITLE = "Raw GP Genome (pre-STRidge)"
_DECLARED_DESCRIPTION = (
    "Shows all evolved terms before STRidge pruning, so its term count may "
    "exceed the discovered equation's."
)
_DERIVED_TITLE = "Plugin Genome Tree"

_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
    '<rect width="10" height="10" fill="blue"/>'
    "</svg>"
)


def _svg_file(tmp_path: Path, stem: str) -> Path:
    path = tmp_path / f"{stem}.svg"
    path.write_text(_SVG, encoding="utf-8")
    return path


def _visible_text(content: str) -> str:
    return html_lib.unescape(content)


class _DescriptorPlugin:

    def list_plots(self) -> list[PlotInfo]:
        return [
            PlotInfo(
                name="genome_tree",
                title=_DECLARED_TITLE,
                description=_DECLARED_DESCRIPTION,
            )
        ]

    def render_plot(self, name: str, ax: Axes) -> None:
        ax.plot([0.0, 1.0], [0.0, 1.0])

    def get_plot_data(self, name: str) -> Any:
        return {}


class _HomogeneousDescriptorPlugin:

    def list_homogeneous_plots(self) -> list[PlotInfo]:
        return [
            PlotInfo(
                name="term_balance",
                title=_DECLARED_TITLE,
                description=_DECLARED_DESCRIPTION,
            )
        ]

    def render_homogeneous_plot(
        self, name: str, ax: Axes, result: ExperimentResult
    ) -> None:
        ax.plot([0.0, 1.0], [0.0, 1.0])


def _homogeneous(result: ExperimentResult) -> ExperimentResult:
    equation = make_homogeneous((("u_xx", Scalar(1.0)), ("u_yy", Scalar(1.0))))
    return replace(result, equation=equation)





def test_declared_title_replaces_the_filename_derived_one(
    tmp_path: Path,
    mock_experiment_result: ExperimentResult,
) -> None:
    output = tmp_path / "report.html"
    spec = FigureSpec(
        path=_svg_file(tmp_path, "plugin_genome_tree"),
        title=_DECLARED_TITLE,
        description=_DECLARED_DESCRIPTION,
    )

    generate_report(mock_experiment_result, [], output, plugin_figures=[spec])

    content = output.read_text(encoding="utf-8")
    assert _DECLARED_TITLE in content
    assert _DERIVED_TITLE not in content


def test_declared_description_is_rendered_under_the_figure(
    tmp_path: Path,
    mock_experiment_result: ExperimentResult,
) -> None:
    output = tmp_path / "report.html"
    spec = FigureSpec(
        path=_svg_file(tmp_path, "plugin_genome_tree"),
        title=_DECLARED_TITLE,
        description=_DECLARED_DESCRIPTION,
    )

    generate_report(mock_experiment_result, [], output, plugin_figures=[spec])

    content = output.read_text(encoding="utf-8")
    assert _DECLARED_DESCRIPTION in _visible_text(content)

    assert 'class="figure-description"' in content


def test_universal_figure_without_descriptor_keeps_filename_title(
    tmp_path: Path,
    mock_experiment_result: ExperimentResult,
) -> None:
    output = tmp_path / "report.html"

    generate_report(
        mock_experiment_result, [_svg_file(tmp_path, "convergence")], output
    )

    content = output.read_text(encoding="utf-8")
    assert "Convergence" in content
    assert 'class="figure-description"' not in content


def test_figure_spec_without_title_falls_back_to_the_filename(
    tmp_path: Path,
    mock_experiment_result: ExperimentResult,
) -> None:
    output = tmp_path / "report.html"
    spec = FigureSpec(path=_svg_file(tmp_path, "plugin_genome_tree"))

    generate_report(mock_experiment_result, [], output, plugin_figures=[spec])

    assert _DERIVED_TITLE in output.read_text(encoding="utf-8")


def test_description_is_escaped_not_trusted_as_markup(
    tmp_path: Path,
    mock_experiment_result: ExperimentResult,
) -> None:
    output = tmp_path / "report.html"
    spec = FigureSpec(
        path=_svg_file(tmp_path, "plugin_genome_tree"),
        title="Title <b>x</b>",
        description="Caption <script>INJECTED</script>",
    )

    generate_report(mock_experiment_result, [], output, plugin_figures=[spec])

    content = output.read_text(encoding="utf-8")
    assert "<script>INJECTED</script>" not in content
    assert "<b>x</b>" not in content





def test_engine_forwards_plugin_descriptor_to_the_report(
    tmp_path: Path,
    mock_experiment_result: ExperimentResult,
) -> None:
    report = VizEngine(tmp_path).render_all(
        mock_experiment_result, algorithm=_DescriptorPlugin()
    )

    assert report.report is not None
    content = report.report.read_text(encoding="utf-8")
    assert _DECLARED_TITLE in content
    assert _DERIVED_TITLE not in content
    assert _DECLARED_DESCRIPTION in _visible_text(content)


def test_engine_forwards_homogeneous_descriptor_to_the_report(
    tmp_path: Path,
    mock_experiment_result: ExperimentResult,
) -> None:
    result = _homogeneous(mock_experiment_result)

    report = VizEngine(tmp_path).render_all(
        result, algorithm=_HomogeneousDescriptorPlugin()
    )

    assert report.report is not None
    content = report.report.read_text(encoding="utf-8")
    assert _DECLARED_TITLE in content
    assert "Homogeneous Term Balance" not in content
    assert _DECLARED_DESCRIPTION in _visible_text(content)


def test_engine_keeps_filename_titles_for_universal_plots(
    tmp_path: Path,
    mock_experiment_result: ExperimentResult,
) -> None:
    report = VizEngine(tmp_path).render_all(
        mock_experiment_result, algorithm=_DescriptorPlugin()
    )

    assert report.report is not None
    content = report.report.read_text(encoding="utf-8")
    for derived in ("Convergence", "Parity", "Equation", "Residual"):
        assert derived in content
