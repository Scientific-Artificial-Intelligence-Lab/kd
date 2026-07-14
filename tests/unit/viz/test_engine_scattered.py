
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest
import torch

from kd.data.schema import PDEDataset
from kd.search.result import ExperimentResult
from kd.viz.engine import VizEngine
from kd.viz.report import ReportResult

pytestmark = pytest.mark.unit



_GRID_FIELD_PLOTS = frozenset(
    {
        "field_comparison",
        "pde_residual_field",
        "time_slices",
        "error_heatmap",
        "field_animation",
    }
)


def _scatter_dataset() -> PDEDataset:
    n = 16
    torch.manual_seed(0)
    return PDEDataset.from_scatter(
        coords={"t": torch.rand(n), "x": torch.rand(n)},
        fields={"u": torch.rand(n)},
        lhs="u_t",
        name="scatter-viz",
    )


def _render(engine: VizEngine, result: ExperimentResult) -> ReportResult:
    report = ReportResult()
    engine._render_field_comparison(result, _scatter_dataset(), report)
    return report


class TestScatteredFieldComparison:
    def test_does_not_raise(
        self, tmp_path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)

        _render(engine, mock_experiment_result)

    def test_renders_coefficient_bar(
        self, tmp_path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = _render(engine, mock_experiment_result)
        stems = {p.stem for p in report.figures}
        assert "coefficient_bar" in stems

    def test_emits_scatter_degrade_note(
        self, tmp_path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = _render(engine, mock_experiment_result)

        def _is_degrade_note(w: str) -> bool:
            low = w.lower()
            return "field-grid" in low or ("scatter" in low and "skip" in low)

        assert any(_is_degrade_note(w) for w in report.warnings), (
            f"expected an honest 'scattered: field-grid plots skipped' degrade "
            f"note, got warnings={report.warnings!r}"
        )

    def test_skips_grid_field_plots(
        self, tmp_path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = _render(engine, mock_experiment_result)
        rendered = {p.stem for p in report.figures}
        leaked = rendered & _GRID_FIELD_PLOTS
        assert not leaked, f"grid-field plots must be skipped on scatter: {leaked}"

    def test_no_grid_field_failure_warnings(
        self, tmp_path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = _render(engine, mock_experiment_result)
        offenders = [
            w for w in report.warnings if any(name in w for name in _GRID_FIELD_PLOTS)
        ]
        assert not offenders, (
            f"grid-field plots must be cleanly skipped, not crash-warned: {offenders!r}"
        )


class TestScatteredRenderAllResidual:

    def test_no_field_shape_mismatch_warning(
        self, tmp_path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result, dataset=_scatter_dataset())
        offenders = [w for w in report.warnings if "does not match data size" in w]
        assert not offenders, (
            f"SCATTERED must not trigger a spatial-residual shape-mismatch "
            f"warning; got {offenders!r}"
        )

    def test_report_is_still_written(
        self, tmp_path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result, dataset=_scatter_dataset())
        assert report.report is not None and report.report.exists()
