
from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import pytest
import torch

from kd.core.equation import LhsSpec, Scalar, make_evolution, make_homogeneous
from kd.data.schema import PDEDataset
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.result import ExperimentResult
from kd.viz import extension as viz_extension
from kd.viz.engine import VizEngine
from kd.viz.extension import PlotInfo

pytestmark = pytest.mark.unit

_HOMOGENEOUS_NAMES = ("residual_domain", "term_balance", "surrogate_fit")
_HOMOGENEOUS_STEMS = {f"homogeneous_{name}" for name in _HOMOGENEOUS_NAMES}


class _HomogeneousOnlyViz:

    def __init__(self) -> None:
        self.rendered: list[tuple[str, ExperimentResult]] = []

    def list_homogeneous_plots(self) -> list[PlotInfo]:
        return [
            PlotInfo(
                name=name,
                title=f"Reproduces EqGPT steady: {name}",
                projection="3d" if name == "residual_domain" else None,
            )
            for name in _HOMOGENEOUS_NAMES
        ]

    def render_homogeneous_plot(
        self, name: str, ax, result: ExperimentResult
    ) -> None:
        self.rendered.append((name, result))
        text_2d = getattr(ax, "text2D", None)
        if text_2d is not None:
            text_2d(0.5, 0.5, name)
        else:
            ax.text(0.5, 0.5, name)


def _scatter_dataset() -> PDEDataset:
    x = torch.linspace(-1.0, 1.0, 12)
    y = torch.linspace(0.0, 2.0, 12)
    return PDEDataset.from_scatter(
        coords={"x": x, "y": y},
        fields={"u": x**2 - y**2},
        lhs="",
        name="homogeneous-viz",
    )


def _homogeneous_result(result: ExperimentResult) -> ExperimentResult:
    equation = make_homogeneous(
        (("u_xx", Scalar(1.0)), ("u_yy", Scalar(1.0)))
    )
    final = replace(
        result.final_eval,
        form=equation.form,
        terms=["u_xx", "u_yy"],
        coefficients=torch.tensor([1.0, 1.0]),
        selected_indices=None,
    )
    return replace(result, equation=equation, final_eval=final)


def _evolution_same_shape(result: ExperimentResult) -> ExperimentResult:
    equation = make_evolution(
        LhsSpec(field="u", axis="t", order=1),
        (("u_xx", Scalar(1.0)), ("u_yy", Scalar(1.0))),
    )


    return replace(result, equation=equation)


def test_homogeneous_viz_extension_is_runtime_checkable() -> None:
    protocol = viz_extension.HomogeneousVizExtension
    assert isinstance(_HomogeneousOnlyViz(), protocol)


def test_steady_eqgpt_exposes_the_homogeneous_viz_extension() -> None:
    protocol = viz_extension.HomogeneousVizExtension
    config = EqGPTConfig.steady_preset("smile", steady_train_iters=1)
    plugin = EqGPTPlugin(config, backend=FakeGPTBackend(57, seed=0))

    assert isinstance(plugin, protocol)
    infos = plugin.list_homogeneous_plots()
    assert [info.name for info in infos] == list(_HOMOGENEOUS_NAMES)
    for info in infos:
        title = info.title.lower()
        assert "reproduces eqgpt" in title
        assert "kd discovers" not in title


def test_engine_renders_homogeneous_triple_by_explicit_form(
    tmp_path, mock_experiment_result: ExperimentResult
) -> None:
    result = _homogeneous_result(mock_experiment_result)
    algorithm = _HomogeneousOnlyViz()

    report = VizEngine(tmp_path).render_all(
        result, algorithm=algorithm, dataset=_scatter_dataset()
    )

    stems = {path.stem for path in report.figures}
    assert stems >= _HOMOGENEOUS_STEMS
    assert [name for name, _result in algorithm.rendered] == list(
        _HOMOGENEOUS_NAMES
    )
    assert all(seen is result for _name, seen in algorithm.rendered)


def test_homogeneous_render_warnings_reach_report(
    tmp_path, mock_experiment_result: ExperimentResult
) -> None:

    class _WarningHomogeneousViz(_HomogeneousOnlyViz):
        def render_homogeneous_plot(
            self, name: str, ax, result: ExperimentResult
        ) -> list[str]:
            super().render_homogeneous_plot(name, ax, result)
            return [f"homogeneous plot '{name}': No data (probe)"]

    result = _homogeneous_result(mock_experiment_result)
    report = VizEngine(tmp_path).render_all(
        result, algorithm=_WarningHomogeneousViz(), dataset=_scatter_dataset()
    )

    for name in _HOMOGENEOUS_NAMES:
        assert any(f"homogeneous plot '{name}'" in w for w in report.warnings)


def test_engine_does_not_shape_guess_homogeneous_plots_for_evolution(
    tmp_path, mock_experiment_result: ExperimentResult
) -> None:
    result = _evolution_same_shape(mock_experiment_result)
    algorithm = _HomogeneousOnlyViz()

    report = VizEngine(tmp_path).render_all(
        result, algorithm=algorithm, dataset=_scatter_dataset()
    )

    stems = {path.stem for path in report.figures}
    assert not (_HOMOGENEOUS_STEMS & stems)
    assert algorithm.rendered == []


def test_skipped_homogeneous_group_is_named_in_warnings(
    tmp_path, mock_experiment_result: ExperimentResult
) -> None:
    result = replace(mock_experiment_result, equation=None)
    algorithm = _HomogeneousOnlyViz()

    report = VizEngine(tmp_path).render_all(
        result, algorithm=algorithm, dataset=_scatter_dataset()
    )

    assert algorithm.rendered == []
    assert any(
        "Steady-state plots (3) skipped" in w and "no equation" in w
        for w in report.warnings
    ), report.warnings


def test_no_skip_note_when_producer_declares_no_homogeneous_plots(
    tmp_path, mock_experiment_result: ExperimentResult
) -> None:

    class _NoneDeclaredViz(_HomogeneousOnlyViz):
        def list_homogeneous_plots(self) -> list[PlotInfo]:
            return []

    result = replace(mock_experiment_result, equation=None)
    report = VizEngine(tmp_path).render_all(
        result, algorithm=_NoneDeclaredViz(), dataset=_scatter_dataset()
    )

    assert not any("Steady-state plots" in w for w in report.warnings), (
        report.warnings
    )
