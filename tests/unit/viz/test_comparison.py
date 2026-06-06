
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz import VizEngine
from kd.viz.report import ReportResult


def _make_result(
    name: str,
    r2: float = 0.9,
    nmse: float = 0.05,
    *,
    n_iterations: int = 5,
    has_recorder: bool = True,
    algorithm: str | None = None,
) -> ExperimentResult:
    n_samples = 20
    recorder = VizRecorder()
    if has_recorder:
        for i in range(n_iterations):
            recorder.log("_best_score", 1.0 / (i + 1))
            recorder.log("_best_expr", f"expr_{i}")

    config: dict[str, object] = {"max_iter": n_iterations}
    if algorithm is not None:
        config["algorithm"] = algorithm

    return ExperimentResult(
        best_expression=f"expr_{name}",
        best_score=nmse,
        iterations=n_iterations,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=nmse * 2,
            nmse=nmse,
            r2=r2,
            aic=-50.0,
            complexity=2,
            coefficients=torch.tensor([1.0, 0.5]),
            is_valid=True,
            error_message="",
            selected_indices=[0, 1],
            residuals=torch.randn(n_samples) * 0.1,
            terms=["u", "u_x"],
            expression=f"expr_{name}",
        ),
        actual=torch.randn(n_samples),
        predicted=torch.randn(n_samples),
        dataset_name="test",
        algorithm_name=name,
        config=config,
        recorder=recorder,
    )


class TestRenderComparison:

    def test_returns_report_result(self, tmp_path: Path) -> None:
        results = [_make_result("A"), _make_result("B")]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results)
        assert isinstance(report, ReportResult)

    def test_creates_comparison_files(self, tmp_path: Path) -> None:
        results = [_make_result("A"), _make_result("B")]
        engine = VizEngine(output_dir=tmp_path)
        engine.render_comparison(results)
        svg_files = list(tmp_path.glob("*.svg"))
        assert len(svg_files) >= 2

    def test_custom_labels(self, tmp_path: Path) -> None:
        results = [_make_result("A"), _make_result("B")]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results, labels=["Run 1", "Run 2"])
        assert isinstance(report, ReportResult)

    def test_all_figures_closed(self, tmp_path: Path) -> None:
        results = [_make_result("A"), _make_result("B")]
        figs_before = plt.get_fignums()
        engine = VizEngine(output_dir=tmp_path)
        engine.render_comparison(results)
        figs_after = plt.get_fignums()
        assert len(figs_after) <= len(figs_before)

    def test_missing_recorder_data_warns(self, tmp_path: Path) -> None:
        results = [
            _make_result("A", has_recorder=False),
            _make_result("B", has_recorder=False),
        ]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results)
        assert isinstance(report, ReportResult)

        assert any(
            "convergence" in w.lower() or "skip" in w.lower() for w in report.warnings
        )

    def test_single_result(self, tmp_path: Path) -> None:
        results = [_make_result("A")]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results)
        assert isinstance(report, ReportResult)

    def test_figures_exist(self, tmp_path: Path) -> None:
        results = [_make_result("A"), _make_result("B")]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results)
        for fig_path in report.figures:
            assert fig_path.exists(), f"Missing: {fig_path}"


def _has_mean_band(ax) -> bool:
    return "mean" in [line.get_label() for line in ax.lines]


class TestOverlaidConvergenceBandGating:

    @pytest.mark.unit
    def test_band_drawn_for_same_algorithm(self) -> None:
        from kd.viz.plots.comparison import render_overlaid_convergence

        results = [
            _make_result("sga1", algorithm="sga"),
            _make_result("sga2", algorithm="sga"),
        ]
        fig, ax = plt.subplots()
        try:
            warnings = render_overlaid_convergence(results, ax)
            assert _has_mean_band(ax), (
                "two same-algorithm runs must still get the mean±std band"
            )
            assert not any("mixed" in w.lower() for w in warnings)
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_band_suppressed_for_mixed_algorithms(self) -> None:
        from kd.viz.plots.comparison import render_overlaid_convergence

        results = [
            _make_result("s", algorithm="sga"),
            _make_result("d", algorithm="dlga"),
        ]
        fig, ax = plt.subplots()
        try:
            warnings = render_overlaid_convergence(results, ax)
            assert not _has_mean_band(ax), (
                "mixed-algorithm overlay must NOT average incommensurable "
                "series into one mean±std band (AUDIT-01)"
            )
            assert any(
                "mixed" in w.lower() or "incommensurable" in w.lower() for w in warnings
            ), f"mixed-algorithm overlay must warn; got {warnings!r}"
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_individual_curves_still_drawn_when_band_suppressed(self) -> None:
        from kd.viz.plots.comparison import render_overlaid_convergence

        results = [
            _make_result("s", algorithm="sga"),
            _make_result("d", algorithm="dlga"),
        ]
        fig, ax = plt.subplots()
        try:
            render_overlaid_convergence(results, ax)

            run_curves = [ln for ln in ax.lines if ln.get_label() != "mean"]
            assert len(run_curves) >= 2, (
                f"both run curves must still be plotted; got {len(run_curves)}"
            )
            assert not _has_mean_band(ax)
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_band_drawn_when_algorithm_unspecified(self) -> None:
        from kd.viz.plots.comparison import render_overlaid_convergence

        results = [_make_result("A"), _make_result("B")]
        fig, ax = plt.subplots()
        try:
            render_overlaid_convergence(results, ax)
            assert _has_mean_band(ax), (
                "algorithm-free results must keep the legacy pooled band"
            )
        finally:
            plt.close(fig)
