
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
    score_kind: str = "Score",
    score_direction: str = "min",
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
            score=-50.0,
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
        score_kind=score_kind,
        score_direction=score_direction,
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
            _make_result("s", algorithm="sga", score_kind="AIC", score_direction="min"),
            _make_result(
                "d",
                algorithm="dlga",
                score_kind="DLGA fitness",
                score_direction="min",
            ),
        ]
        fig, ax = plt.subplots()
        try:
            warnings = render_overlaid_convergence(results, ax)
            assert not _has_mean_band(ax), (
                "mixed-algorithm overlay must NOT average incommensurable "
                "series into one mean±std band"
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
            _make_result("s", algorithm="sga", score_kind="AIC", score_direction="min"),
            _make_result(
                "d",
                algorithm="dlga",
                score_kind="DLGA fitness",
                score_direction="min",
            ),
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








def _result_with_equation(
    name: str, term_irs: tuple[str, ...] | None
) -> ExperimentResult:
    import dataclasses

    from kd.core.equation import LhsSpec, Scalar, make_evolution

    equation = None
    if term_irs is not None:
        equation = make_evolution(
            LhsSpec(field="u", axis="t", order=1),
            tuple((term_ir, Scalar(1.0)) for term_ir in term_irs),
        )
    return dataclasses.replace(_make_result(name), equation=equation)


def _table_cell_texts(ax) -> list[str]:
    tables = ax.tables
    assert tables, "plot_summary_table must render a table"
    return [
        cell.get_text().get_text() for cell in tables[0].get_celld().values()
    ]


class TestSummaryTableTermDiff:

    @pytest.mark.unit
    def test_diff_column_present_with_baseline_marker(self) -> None:
        from kd.viz.plots.comparison import plot_summary_table

        results = [
            _result_with_equation("A", ("u", "div(u, x)")),
            _result_with_equation("B", ("u", "div(u, x)")),
        ]
        fig, ax = plt.subplots()
        try:
            plot_summary_table(results, ax)
            texts = _table_cell_texts(ax)
            assert any("Δ Terms vs Run 0" in t for t in texts), texts
            assert any("baseline" in t for t in texts), texts
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_added_and_removed_terms_named_canonically(self) -> None:
        from kd.viz.plots.comparison import plot_summary_table

        results = [
            _result_with_equation("A", ("u", "div(u, x)")),
            _result_with_equation("B", ("u", "diff2_x(u)")),
        ]
        fig, ax = plt.subplots()
        try:
            plot_summary_table(results, ax)
            texts = _table_cell_texts(ax)
            assert any("+diff2_x(u)" in t for t in texts), texts
            assert any("-div(u,x)" in t for t in texts), texts
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_identical_structure_renders_equals_sign(self) -> None:
        from kd.viz.plots.comparison import plot_summary_table

        results = [
            _result_with_equation("A", ("mul(u, diff_x(u))",)),
            _result_with_equation("B", ("mul(diff_x(u),u)",)),
        ]
        fig, ax = plt.subplots()
        try:
            plot_summary_table(results, ax)
            texts = _table_cell_texts(ax)
            assert "=" in texts, texts
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_missing_equation_degrades_to_na_with_warning(self) -> None:
        from kd.viz.plots.comparison import plot_summary_table

        results = [
            _result_with_equation("A", ("u",)),
            _result_with_equation("B", None),
        ]
        fig, ax = plt.subplots()
        try:
            warnings_list = plot_summary_table(results, ax)
            texts = _table_cell_texts(ax)
            assert any("n/a" in t for t in texts), texts
            assert any("equation" in w for w in warnings_list), warnings_list
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_baseline_missing_equation_degrades_for_all_rows(self) -> None:
        from kd.viz.plots.comparison import plot_summary_table

        results = [
            _result_with_equation("A", None),
            _result_with_equation("B", ("u",)),
        ]
        fig, ax = plt.subplots()
        try:
            warnings_list = plot_summary_table(results, ax)
            texts = _table_cell_texts(ax)
            assert any("n/a" in t for t in texts), texts
            assert any("equation" in w for w in warnings_list), warnings_list
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_lhs_change_is_flagged(self) -> None:
        import dataclasses

        from kd.core.equation import LhsSpec, Scalar, make_evolution
        from kd.viz.plots.comparison import plot_summary_table

        eq_utt = make_evolution(
            LhsSpec(field="u", axis="t", order=2), (("u", Scalar(1.0)),)
        )
        results = [
            _result_with_equation("A", ("u",)),
            dataclasses.replace(_make_result("B"), equation=eq_utt),
        ]
        fig, ax = plt.subplots()
        try:
            plot_summary_table(results, ax)
            texts = _table_cell_texts(ax)
            assert any("lhs!" in t for t in texts), texts
            assert "=" not in texts, texts
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_long_diff_degrades_to_counts_not_truncation(self) -> None:
        from kd.viz.plots.comparison import plot_summary_table

        results = [
            _result_with_equation("A", ("u", "div(u, x)", "diff2_x(u)")),
            _result_with_equation(
                "B", ("u", "mul(u, diff_x(u))", "diff_x(diff_x(diff_x(u)))")
            ),
        ]
        fig, ax = plt.subplots()
        try:
            warnings_list = plot_summary_table(results, ax)
            texts = _table_cell_texts(ax)
            assert any("+2 -2 terms" in t for t in texts), texts
            assert not any("..." in t and "+" in t for t in texts), texts
            assert any("full diff:" in w for w in warnings_list), warnings_list
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_invalid_baseline_ir_warned_once_with_cause(self) -> None:
        from kd.viz.plots.comparison import plot_summary_table

        results = [
            _result_with_equation("A", ("mul(0.5, u)",)),
            _result_with_equation("B", ("u",)),
            _result_with_equation("C", ("diff2_x(u)",)),
        ]
        fig, ax = plt.subplots()
        try:
            warnings_list = plot_summary_table(results, ax)
            texts = _table_cell_texts(ax)
            assert any("n/a" in t for t in texts), texts
            canon_warnings = [w for w in warnings_list if "canonicalizable" in w]
            assert len(canon_warnings) == 1, warnings_list
            assert "Baseline" in canon_warnings[0], warnings_list
            assert "mul(0.5, u)" in canon_warnings[0], warnings_list
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_missing_baseline_equation_warned_once(self) -> None:
        from kd.viz.plots.comparison import plot_summary_table

        results = [
            _result_with_equation("A", None),
            _result_with_equation("B", ("u",)),
            _result_with_equation("C", ("u",)),
            _result_with_equation("D", ("u",)),
        ]
        fig, ax = plt.subplots()
        try:
            warnings_list = plot_summary_table(results, ax)
            equation_warnings = [w for w in warnings_list if "equation" in w]
            assert len(equation_warnings) == 1, warnings_list
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_mixed_algorithms_get_vocabulary_caveat(self) -> None:
        import dataclasses

        from kd.viz.plots.comparison import plot_summary_table

        results = [
            dataclasses.replace(
                _make_result("A", algorithm="sga"),
                equation=_result_with_equation("A", ("diff2_x(u)",)).equation,
            ),
            dataclasses.replace(
                _make_result("B", algorithm="dlga"),
                equation=_result_with_equation("B", ("u_xx",)).equation,
            ),
        ]
        fig, ax = plt.subplots()
        try:
            warnings_list = plot_summary_table(results, ax)
            assert any("vocabular" in w for w in warnings_list), warnings_list
        finally:
            plt.close(fig)

    @pytest.mark.unit
    def test_same_algorithm_no_vocabulary_caveat(self) -> None:
        import dataclasses

        from kd.viz.plots.comparison import plot_summary_table

        results = [
            dataclasses.replace(
                _make_result("A", algorithm="sga"),
                equation=_result_with_equation("A", ("u",)).equation,
            ),
            dataclasses.replace(
                _make_result("B", algorithm="sga"),
                equation=_result_with_equation("B", ("u",)).equation,
            ),
        ]
        fig, ax = plt.subplots()
        try:
            warnings_list = plot_summary_table(results, ax)
            assert not any("vocabular" in w for w in warnings_list), warnings_list
        finally:
            plt.close(fig)


class TestMixedMetricDisclosure:

    def test_mixed_metric_overlay_labels_and_subtitle(self) -> None:
        from kd.viz.plots.comparison import render_overlaid_convergence

        results = [
            _make_result("SGA", score_kind="AIC", score_direction="min"),
            _make_result("DISCOVER", score_kind="reward", score_direction="max"),
        ]
        fig, ax = plt.subplots()
        try:
            render_overlaid_convergence(results, ax)
            legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
            assert "SGA (AIC, min)" in legend_texts
            assert "DISCOVER (reward, max)" in legend_texts
            assert "Mixed score metrics" in ax.get_title()
        finally:
            plt.close(fig)

    def test_uniform_metric_overlay_keeps_plain_labels(self) -> None:
        from kd.viz.plots.comparison import render_overlaid_convergence

        results = [
            _make_result("A", score_kind="AIC", score_direction="min"),
            _make_result("B", score_kind="AIC", score_direction="min"),
        ]
        fig, ax = plt.subplots()
        try:
            render_overlaid_convergence(results, ax)
            legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
            assert "A" in legend_texts
            assert "B" in legend_texts
            assert "Mixed score metrics" not in ax.get_title()
        finally:
            plt.close(fig)

    def test_summary_table_non_finite_metrics_shown_as_na(self) -> None:
        from kd.viz.plots.comparison import plot_summary_table

        results = [
            _make_result("A", r2=float("nan"), nmse=float("inf")),
            _make_result("B"),
        ]
        fig, ax = plt.subplots()
        try:
            warnings = plot_summary_table(results, ax)
            cells = ax.tables[0].get_celld()
            assert cells[(1, 2)].get_text().get_text() == "N/A"
            assert cells[(1, 3)].get_text().get_text() == "N/A"
            assert cells[(2, 2)].get_text().get_text() == "0.05"
            assert any("NMSE" in w and "N/A" in w for w in warnings)
            assert any("R2" in w and "N/A" in w for w in warnings)
        finally:
            plt.close(fig)
