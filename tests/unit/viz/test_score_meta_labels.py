
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes

from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder
from kd.search.result import (
    DEFAULT_SCORE_DIRECTION,
    DEFAULT_SCORE_KIND,
    ExperimentResult,
)
from kd.viz.plots.comparison import render_overlaid_convergence
from kd.viz.plots.convergence import plot_convergence
from kd.viz.report import generate_report

_N_SAMPLES = 12
_SCORE_SERIES = [1.0, 0.5, 0.25]


def _make_result(
    name: str,
    *,
    score_kind: str,
    score_direction: str,
    algorithm: str | None = None,
) -> ExperimentResult:
    recorder = VizRecorder()
    for score in _SCORE_SERIES:
        recorder.log("_best_score", score)
        recorder.log("_best_expr", f"expr_{name}")
    config: dict[str, Any] = {"max_iter": len(_SCORE_SERIES)}
    if algorithm is not None:
        config["algorithm"] = algorithm
    return ExperimentResult(
        best_expression=f"expr_{name}",
        best_score=_SCORE_SERIES[-1],
        iterations=len(_SCORE_SERIES),
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.01,
            nmse=0.005,
            r2=0.99,
            score=-50.0,
            complexity=2,
            coefficients=torch.tensor([1.0, 0.5]),
            is_valid=True,
            error_message="",
            selected_indices=[0, 1],
            residuals=torch.zeros(_N_SAMPLES),
            terms=["u", "u_x"],
            expression=f"expr_{name}",
        ),
        actual=torch.linspace(0.0, 1.0, _N_SAMPLES),
        predicted=torch.linspace(0.0, 1.0, _N_SAMPLES),
        dataset_name="meta_viz",
        algorithm_name=name,
        config=config,
        recorder=recorder,
        score_kind=score_kind,
        score_direction=score_direction,
    )


def _has_mean_band(ax: Axes) -> bool:
    return "mean" in [line.get_label() for line in ax.lines]







class TestConvergenceYlabelFromScoreKind:

    def test_ylabel_reads_field_not_lookup_table(self) -> None:
        result = _make_result(
            "r1", score_kind="CUSTOM", score_direction="min", algorithm="sga"
        )
        fig, ax = plt.subplots()
        try:
            plot_convergence(result, ax)
            assert ax.get_ylabel() == "Best CUSTOM"
        finally:
            plt.close(fig)

    def test_ylabel_for_declared_reward_kind(self) -> None:
        result = _make_result(
            "r1", score_kind="reward", score_direction="max", algorithm="discover"
        )
        fig, ax = plt.subplots()
        try:
            plot_convergence(result, ax)
            assert ax.get_ylabel() == "Best reward"
        finally:
            plt.close(fig)

    def test_empty_recorder_still_labels_from_field(self) -> None:
        result = _make_result(
            "r1", score_kind="CUSTOM", score_direction="min", algorithm="sga"
        )
        result.recorder = VizRecorder()
        fig, ax = plt.subplots()
        try:
            warnings = plot_convergence(result, ax)
            assert warnings
            assert ax.get_ylabel() == "Best CUSTOM"
        finally:
            plt.close(fig)







class TestReportLabelFromScoreKind:

    def test_report_row_reads_field_not_lookup_table(self, tmp_path: Path) -> None:
        result = _make_result(
            "SGAPlugin", score_kind="CUSTOM", score_direction="min", algorithm="sga"
        )
        output = tmp_path / "report.html"
        generate_report(result, [], output)

        content = output.read_text(encoding="utf-8")
        assert "<th>Best CUSTOM</th>" in content

        assert "<th>Best AIC</th>" not in content

    def test_report_row_default_kind_renders_generic_label(
        self, tmp_path: Path
    ) -> None:
        result = _make_result(
            "thirdparty", score_kind="Score", score_direction="min", algorithm=None
        )
        output = tmp_path / "report.html"
        generate_report(result, [], output)

        assert "<th>Best Score</th>" in output.read_text(encoding="utf-8")







class TestSharedYlabelFromMeta:

    def test_common_kind_shared_despite_different_algorithm_names(self) -> None:
        results = [
            _make_result(
                "a", score_kind="CUSTOM", score_direction="min", algorithm="algo_a"
            ),
            _make_result(
                "b", score_kind="CUSTOM", score_direction="min", algorithm="algo_b"
            ),
        ]
        fig, ax = plt.subplots()
        try:
            render_overlaid_convergence(results, ax)
            assert ax.get_ylabel() == "Best CUSTOM"
        finally:
            plt.close(fig)

    def test_mixed_kinds_fall_back_to_generic_ylabel(self) -> None:
        results = [
            _make_result("s", score_kind="AIC", score_direction="min", algorithm="sga"),
            _make_result(
                "d", score_kind="reward", score_direction="max", algorithm="discover"
            ),
        ]
        fig, ax = plt.subplots()
        try:
            render_overlaid_convergence(results, ax)
            assert ax.get_ylabel() == "Best Score"
        finally:
            plt.close(fig)







class TestBandPoolingByScoreIdentity:

    def test_band_pooled_for_same_meta_despite_different_names(self) -> None:
        results = [
            _make_result(
                "a", score_kind="NMSE", score_direction="min", algorithm="algo_a"
            ),
            _make_result(
                "b", score_kind="NMSE", score_direction="min", algorithm="algo_b"
            ),
        ]
        fig, ax = plt.subplots()
        try:
            warnings = render_overlaid_convergence(results, ax)
            assert _has_mean_band(ax), (
                "runs sharing one (score_kind, score_direction) identity must "
                "pool into a mean±std band even across algorithm names"
            )
            assert not warnings, f"pooled overlay must not warn; got {warnings!r}"
        finally:
            plt.close(fig)

    def test_band_suppressed_for_mixed_kinds(self) -> None:
        results = [
            _make_result("s", score_kind="AIC", score_direction="min", algorithm="sga"),
            _make_result(
                "d", score_kind="reward", score_direction="max", algorithm="discover"
            ),
        ]
        fig, ax = plt.subplots()
        try:
            warnings = render_overlaid_convergence(results, ax)
            assert not _has_mean_band(ax)
            assert warnings, "mixed-identity overlay must warn"
        finally:
            plt.close(fig)

    def test_band_suppressed_for_same_kind_different_direction(self) -> None:
        results = [
            _make_result(
                "x1", score_kind="reward", score_direction="max", algorithm="weird"
            ),
            _make_result(
                "x2", score_kind="reward", score_direction="min", algorithm="weird"
            ),
        ]
        fig, ax = plt.subplots()
        try:
            warnings = render_overlaid_convergence(results, ax)
            assert not _has_mean_band(ax), (
                "same kind but opposite directions must not pool"
            )
            assert warnings, "incommensurable directions must warn"
        finally:
            plt.close(fig)

    def test_band_pooled_for_same_algorithm_multiseed(self) -> None:
        results = [
            _make_result(
                "sga1", score_kind="AIC", score_direction="min", algorithm="sga"
            ),
            _make_result(
                "sga2", score_kind="AIC", score_direction="min", algorithm="sga"
            ),
        ]
        fig, ax = plt.subplots()
        try:
            render_overlaid_convergence(results, ax)
            assert _has_mean_band(ax)
        finally:
            plt.close(fig)

    def test_individual_curves_remain_when_band_suppressed(self) -> None:
        results = [
            _make_result("s", score_kind="AIC", score_direction="min", algorithm="sga"),
            _make_result(
                "d", score_kind="reward", score_direction="max", algorithm="discover"
            ),
        ]
        fig, ax = plt.subplots()
        try:
            render_overlaid_convergence(results, ax)
            run_curves = [ln for ln in ax.lines if ln.get_label() != "mean"]
            assert len(run_curves) >= 2
            assert not _has_mean_band(ax)
        finally:
            plt.close(fig)

    def test_band_suppressed_for_undeclared_runs_of_different_algorithms(
        self,
    ) -> None:
        results = [
            _make_result(
                "ext_a",
                score_kind=DEFAULT_SCORE_KIND,
                score_direction=DEFAULT_SCORE_DIRECTION,
                algorithm="external_a",
            ),
            _make_result(
                "ext_b",
                score_kind=DEFAULT_SCORE_KIND,
                score_direction=DEFAULT_SCORE_DIRECTION,
                algorithm="external_b",
            ),
        ]
        fig, ax = plt.subplots()
        try:
            warnings = render_overlaid_convergence(results, ax)
            assert not _has_mean_band(ax), (
                "undeclared results of different algorithms must not pool: "
                "a shared fallback kind is absence of information, not a "
                "shared metric"
            )
            assert warnings, "suppressed undeclared overlay must warn"
        finally:
            plt.close(fig)

    def test_band_pooled_for_undeclared_runs_of_same_algorithm(self) -> None:
        results = [
            _make_result(
                "ext1",
                score_kind=DEFAULT_SCORE_KIND,
                score_direction=DEFAULT_SCORE_DIRECTION,
                algorithm="external_a",
            ),
            _make_result(
                "ext2",
                score_kind=DEFAULT_SCORE_KIND,
                score_direction=DEFAULT_SCORE_DIRECTION,
                algorithm="external_a",
            ),
        ]
        fig, ax = plt.subplots()
        try:
            warnings = render_overlaid_convergence(results, ax)
            assert _has_mean_band(ax), (
                "undeclared multi-seed runs of one algorithm must keep "
                "pooling (legacy behavior)"
            )
            assert not warnings, f"pooled overlay must not warn; got {warnings!r}"
        finally:
            plt.close(fig)
