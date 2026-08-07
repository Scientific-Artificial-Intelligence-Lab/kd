
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use("Agg")

from matplotlib.figure import Figure

from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz.plots.residual import plot_residual






def _make_result(n_samples: int = 50) -> ExperimentResult:
    actual = torch.randn(n_samples)
    predicted = actual + torch.randn(n_samples) * 0.1
    residuals = predicted - actual
    recorder = VizRecorder()
    recorder.log("_best_score", 1.0)
    recorder.log("_best_expr", "u")
    recorder.log("_n_candidates", 10)
    return ExperimentResult(
        best_expression="u",
        best_score=1.0,
        iterations=1,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.01,
            nmse=0.005,
            r2=0.95,
            score=-100.0,
            complexity=1,
            coefficients=torch.tensor([1.0]),
            is_valid=True,
            error_message="",
            selected_indices=[0],
            residuals=residuals,
            terms=["u"],
            expression="u",
        ),
        actual=actual,
        predicted=predicted,
        dataset_name="test",
        algorithm_name="SGA",
        config={},
        recorder=recorder,
    )







class TestResidualTier2Smoke:

    def test_new_signature_returns_figure_tuple(self) -> None:
        result = _make_result()

        out = plot_residual(result)
        try:
            assert isinstance(out, tuple), f"Expected tuple, got {type(out)}"
            assert len(out) == 2
            fig, warnings = out
            assert isinstance(fig, Figure)
            assert isinstance(warnings, list)
        finally:
            if isinstance(out, tuple) and isinstance(out[0], Figure):
                plt.close(out[0])

    def test_accepts_style_kwarg(self) -> None:
        result = _make_result()
        fig, _ = plot_residual(result, style={"font.size": 12})
        plt.close(fig)

    def test_accepts_field_shape_kwarg(self) -> None:
        result = _make_result(n_samples=100)
        fig, _ = plot_residual(result, field_shape=(10, 10))
        plt.close(fig)







class TestResidualTier2HappyPath:

    def test_figure_has_histogram(self) -> None:
        result = _make_result()
        fig, _ = plot_residual(result)
        try:

            has_patches = any(len(ax.patches) > 0 for ax in fig.get_axes())
            assert has_patches, "Expected histogram patches in figure"
        finally:
            plt.close(fig)

    def test_figure_has_stats_annotation(self) -> None:
        result = _make_result()
        fig, _ = plot_residual(result)
        try:
            all_texts = []
            for ax in fig.get_axes():
                all_texts.extend(t.get_text().lower() for t in ax.texts)
            text_joined = " ".join(all_texts)
            assert "mean" in text_joined or "std" in text_joined
        finally:
            plt.close(fig)

    def test_figure_has_labels(self) -> None:
        result = _make_result()
        fig, _ = plot_residual(result)
        try:
            has_labels = any(
                ax.get_xlabel() != "" and ax.get_ylabel() != "" for ax in fig.get_axes()
            )
            assert has_labels
        finally:
            plt.close(fig)







class TestResidualTier2EdgeCases:

    def test_none_residuals_warns(self) -> None:
        result = _make_result()
        result.final_eval.residuals = None
        fig, warnings = plot_residual(result)
        try:
            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_all_nan_residuals(self) -> None:
        result = _make_result()
        result.final_eval.residuals = torch.full((10,), float("nan"))
        fig, warnings = plot_residual(result)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_inf_residuals(self) -> None:
        result = _make_result(n_samples=20)
        result.final_eval.residuals[0] = float("inf")
        fig, warnings = plot_residual(result)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_no_figure_leak(self) -> None:
        result = _make_result()
        figs_before = len(plt.get_fignums())
        fig, _ = plot_residual(result)
        plt.close(fig)
        figs_after = len(plt.get_fignums())
        assert figs_after <= figs_before


class TestResidualInferGrid:

    def _spatial_texts(self, fig: Figure) -> list[str]:

        return [t.get_text() for t in fig.axes[1].texts]

    def test_infer_grid_false_suppresses_square_guess(self) -> None:
        result = _make_result(n_samples=16)
        fig, warnings = plot_residual(result, field_shape=None, infer_grid=False)
        try:
            assert any("No spatial data" in t for t in self._spatial_texts(fig))
            assert not any("does not match" in w for w in warnings)
        finally:
            plt.close(fig)

    def test_infer_grid_true_still_guesses_square(self) -> None:


        result = _make_result(n_samples=16)
        fig, _ = plot_residual(result, field_shape=None, infer_grid=True)
        try:
            assert not any("No spatial data" in t for t in self._spatial_texts(fig))
        finally:
            plt.close(fig)

    def test_infer_grid_true_discloses_the_guess(self) -> None:




        result = _make_result(n_samples=16)
        fig, warnings = plot_residual(result, field_shape=None, infer_grid=True)
        try:
            assert not any("No spatial data" in t for t in self._spatial_texts(fig))
            assert any("(4, 4)" in w for w in warnings), (
                f"guessed shape not disclosed: {warnings}"
            )
            assert any("field_shape" in w for w in warnings), (
                f"remedy not disclosed: {warnings}"
            )
        finally:
            plt.close(fig)


class TestResidualSpatialSkipWarnings:

    def test_non_square_default_warns(self) -> None:
        result = _make_result(n_samples=48)
        fig, warnings = plot_residual(result)
        try:
            assert any("not a perfect square" in w for w in warnings)
            assert any("field_shape" in w for w in warnings)
        finally:
            plt.close(fig)

    def test_one_dimensional_field_shape_warns(self) -> None:
        result = _make_result(n_samples=48)
        fig, warnings = plot_residual(result, field_shape=(48,))
        try:
            assert any("fewer than 2 dimensions" in w for w in warnings)
        finally:
            plt.close(fig)

    def test_infer_grid_false_stays_silent(self) -> None:


        result = _make_result(n_samples=16)
        fig, warnings = plot_residual(result, field_shape=None, infer_grid=False)
        try:
            assert warnings == []
        finally:
            plt.close(fig)


class TestResidualHeatmapClipDisclosure:

    def test_heatmap_title_discloses_clipped_range(self) -> None:
        result = _make_result(n_samples=100)
        residuals = result.final_eval.residuals
        assert residuals is not None
        residuals[0] = 1e12
        fig, _ = plot_residual(result, field_shape=(10, 10))
        try:

            assert "clipped, actual" in fig.axes[1].get_title()


            image = fig.axes[1].images[0]
            assert image.colorbar is not None
            assert image.colorbar.extend == "max"
        finally:
            plt.close(fig)

    def test_unclipped_heatmap_carries_no_clip_note(self) -> None:


        result = _make_result(n_samples=100)
        residuals = result.final_eval.residuals
        assert residuals is not None
        residuals[:] = 1.0
        fig, _ = plot_residual(result, field_shape=(10, 10))
        try:


            assert fig.axes[1].images, "spatial heatmap did not render"
            assert "clipped" not in fig.axes[1].get_title()
        finally:
            plt.close(fig)
