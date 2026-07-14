
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from kd.core.integrator import IntegrationResult
from kd.data.schema import PDEDataset
from kd.search.result import ExperimentResult
from kd.viz import VizEngine
from kd.viz.report import ReportResult


class TestVizEngineInit:

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "viz_output"
        engine = VizEngine(output_dir=out)
        assert out.exists()

    def test_accepts_custom_style(self, tmp_path: Path) -> None:
        engine = VizEngine(output_dir=tmp_path, style={"font.size": 20})
        assert engine._style["font.size"] == 20


class TestRenderUniversal:

    def test_returns_report_result(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_universal(mock_experiment_result)
        assert isinstance(report, ReportResult)

    def test_creates_svg_files(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_universal(mock_experiment_result)

        svg_files = list(tmp_path.glob("*.svg"))
        assert len(svg_files) >= 4

    def test_all_figures_closed(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        figs_before = plt.get_fignums()
        engine = VizEngine(output_dir=tmp_path)
        engine.render_universal(mock_experiment_result)
        figs_after = plt.get_fignums()

        assert len(figs_after) <= len(figs_before)

    def test_figures_in_report(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_universal(mock_experiment_result)

        for fig_path in report.figures:
            assert fig_path.exists(), f"Missing: {fig_path}"

    def test_report_has_no_unexpected_warnings(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_universal(mock_experiment_result)

        assert len(report.warnings) == 0, f"Unexpected warnings: {report.warnings}"


class TestRenderAll:

    def test_without_dataset(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result)
        assert isinstance(report, ReportResult)

        field_files = [f for f in report.figures if "field" in str(f)]
        assert len(field_files) == 0

    def test_with_dataset_renders_field(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        import torch

        from kd.data.schema import (
            AxisInfo,
            DataTopology,
            FieldData,
            PDEDataset,
            TaskType,
        )

        nx, nt = 10, 5
        x_vals = torch.linspace(0, 1, nx)
        t_vals = torch.linspace(0, 1, nt)
        u_field = torch.randn(nx, nt, dtype=torch.float64)

        ds = PDEDataset(
            name="test_1d",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals, is_periodic=True),
                "t": AxisInfo(name="t", values=t_vals),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u_field)},
            lhs_field="u",
            lhs_axis="t",
        )

        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result, dataset=ds)
        field_files = [f for f in report.figures if f.name == "field_comparison.svg"]
        assert len(field_files) == 1

    def test_dataset_without_proper_api_warns(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:

        class _EmptyDataset:
            pass

        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result, dataset=_EmptyDataset())

        assert any("failed" in w.lower() for w in report.warnings)

    def test_all_figures_closed(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        figs_before = plt.get_fignums()
        engine = VizEngine(output_dir=tmp_path)
        engine.render_all(mock_experiment_result)
        figs_after = plt.get_fignums()
        assert len(figs_after) <= len(figs_before)

    def test_universal_plot_error_isolation(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        from unittest.mock import patch

        def _raise_on_equation(result: ExperimentResult, ax: object) -> list[str]:
            raise RuntimeError("Simulated plot failure")

        engine = VizEngine(output_dir=tmp_path)
        with patch("kd.viz.engine.plot_equation", _raise_on_equation):
            report = engine.render_all(mock_experiment_result)

        assert len(report.figures) >= 2

        assert any("failed" in w.lower() for w in report.warnings)







def _make_pde_dataset_for_engine() -> PDEDataset:
    import torch

    from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType

    nx, nt = 10, 5
    x = torch.linspace(0, 1, nx)
    t = torch.linspace(0, 1, nt)
    u_field = torch.randn(nx, nt, dtype=torch.float64)
    return PDEDataset(
        name="test_1d",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_2d_pde_dataset_for_engine() -> PDEDataset:
    import torch

    from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType

    nx, ny, nt = 5, 4, 3
    x = torch.linspace(-2.0, 3.0, nx)
    y = torch.linspace(10.0, 14.0, ny)
    t = torch.linspace(0.0, 1.0, nt)
    u_field = (
        torch.sin(x).reshape(nx, 1, 1)
        * torch.cos(y).reshape(1, ny, 1)
        * torch.exp(-t).reshape(1, 1, nt)
    ).to(torch.float64)
    return PDEDataset(
        name="test_2d",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x),
            "y": AxisInfo(name="y", values=y),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "y", "t"],
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


class TestRenderAllTier2Plots:

    def test_coefficient_bar_in_render_all(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result, dataset=ds)
        coeff_files = [f for f in report.figures if "coefficient" in f.name.lower()]
        assert len(coeff_files) >= 1, (
            f"Expected coefficient plot in output, got: "
            f"{[f.name for f in report.figures]}"
        )

    def test_time_slices_in_render_all(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result, dataset=ds)
        slice_files = [f for f in report.figures if "time_slice" in f.name.lower()]
        assert len(slice_files) >= 1, (
            f"Expected time_slices plot in output, got: "
            f"{[f.name for f in report.figures]}"
        )

    def test_error_heatmap_in_render_all(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result, dataset=ds)
        heatmap_files = [f for f in report.figures if "error_heatmap" in f.name.lower()]
        assert len(heatmap_files) >= 1, (
            f"Expected error_heatmap plot in output, got: "
            f"{[f.name for f in report.figures]}"
        )

    def test_pde_residual_passes_dataset(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        from unittest.mock import patch

        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)

        calls: list[dict] = []
        original = engine._render_pde_residual

        def _spy(result, dataset, report):
            calls.append({"dataset": dataset})
            return original(result, dataset, report)

        with patch.object(engine, "_render_pde_residual", _spy):
            engine.render_all(mock_experiment_result, dataset=ds)

        assert len(calls) >= 1, "Expected _render_pde_residual to be called"
        assert calls[0]["dataset"] is ds

    def test_new_plots_do_not_break_without_dataset(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result)
        assert isinstance(report, ReportResult)

        names = [f.name for f in report.figures]
        assert not any("coefficient" in n.lower() for n in names)
        assert not any("time_slice" in n.lower() for n in names)
        assert not any("error_heatmap" in n.lower() for n in names)

    def test_tier2_error_isolation(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        from unittest.mock import patch

        import matplotlib.pyplot as mpl_plt
        from matplotlib.figure import Figure as MplFigure

        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)

        def _failing_plot(**kwargs):
            raise RuntimeError("Simulated Tier 2 failure")

        with patch("kd.viz.engine.plot_field_comparison", _failing_plot):
            report = engine.render_all(mock_experiment_result, dataset=ds)

        assert len(report.figures) >= 3

        assert any("failed" in w.lower() for w in report.warnings)

    def test_all_figures_closed_with_m3(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        ds = _make_pde_dataset_for_engine()
        figs_before = plt.get_fignums()
        engine = VizEngine(output_dir=tmp_path)
        engine.render_all(mock_experiment_result, dataset=ds)
        figs_after = plt.get_fignums()
        assert len(figs_after) <= len(figs_before)

    def test_animation_default_false_does_not_render_gif(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        from unittest.mock import patch

        ds = _make_2d_pde_dataset_for_engine()
        ir = IntegrationResult(success=True, predicted_field=ds.get_field("u"))
        engine = VizEngine(output_dir=tmp_path)

        with (
            patch.object(engine, "_get_integration_result", return_value=(ir, [])),
            patch(
                "kd.viz.engine.plot_field_animation",
                side_effect=AssertionError("animation should be gated off"),
            ),
        ):
            report = engine.render_all(mock_experiment_result, dataset=ds)

        assert not (tmp_path / "field_animation.gif").exists()
        assert not any(path.name == "field_animation.gif" for path in report.figures)

    def test_animation_true_saves_gif_outside_html_figures(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        from unittest.mock import patch

        ds = _make_2d_pde_dataset_for_engine()
        ir = IntegrationResult(success=True, predicted_field=ds.get_field("u"))
        engine = VizEngine(output_dir=tmp_path)

        class _FakeAnimation:
            def __init__(self) -> None:
                self._fig = plt.figure()

            def save(self, path: Path, *, writer: object) -> None:
                Path(path).write_bytes(b"GIF89a")

        class _FakePillowWriter:
            @classmethod
            def isAvailable(cls) -> bool:
                return True

            def __init__(self, *, fps: int) -> None:
                self.fps = fps

        def _fake_plot(
            *args: object,
            **kwargs: object,
        ) -> tuple[_FakeAnimation, list[str]]:
            return _FakeAnimation(), []

        with (
            patch.object(engine, "_get_integration_result", return_value=(ir, [])),
            patch("kd.viz.engine.plot_field_animation", _fake_plot),
            patch("kd.viz.engine.PillowWriter", _FakePillowWriter),
        ):
            report = engine.render_all(
                mock_experiment_result,
                dataset=ds,
                animate=True,
            )

        gif_path = tmp_path / "field_animation.gif"
        assert gif_path.read_bytes() == b"GIF89a"
        assert gif_path in report.data_files
        assert not any(path.name == "field_animation.gif" for path in report.figures)

    def test_animation_missing_writer_warns_and_skips(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        from unittest.mock import patch

        ds = _make_2d_pde_dataset_for_engine()
        ir = IntegrationResult(success=True, predicted_field=ds.get_field("u"))
        engine = VizEngine(output_dir=tmp_path)

        class _UnavailablePillowWriter:
            @classmethod
            def isAvailable(cls) -> bool:
                return False

        with (
            patch.object(engine, "_get_integration_result", return_value=(ir, [])),
            patch(
                "kd.viz.engine.plot_field_animation",
                side_effect=AssertionError("writer check should skip before plotting"),
            ),
            patch("kd.viz.engine.PillowWriter", _UnavailablePillowWriter),
        ):
            report = engine.render_all(
                mock_experiment_result,
                dataset=ds,
                animate=True,
            )

        assert not (tmp_path / "field_animation.gif").exists()
        assert any("PillowWriter" in warning for warning in report.warnings)







class TestGetIntegrationResultTryExcept:

    def test_integrate_pde_error_caught(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        from unittest.mock import patch

        from kd.core.integrator import IntegrationResult

        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)



        with patch(
            "kd.core.integrator.integrate_pde",
            side_effect=RuntimeError("Solver diverged"),
        ):
            result, _notes = engine._get_integration_result(mock_experiment_result, ds)

        assert isinstance(result, IntegrationResult)
        assert not result.success

        assert result.warning is not None

    def test_format_pde_out_of_integration_path(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        from unittest.mock import patch

        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)

        with patch(
            "kd.core.expr.sympy_bridge.format_pde",
            side_effect=TypeError("BUG: format_pde must not be called"),
        ):
            result, _notes = engine._get_integration_result(
                mock_experiment_result, ds
            )


        assert "BUG" not in (result.warning or "")

    def test_attribute_access_bug_not_swallowed(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        from unittest.mock import PropertyMock, patch

        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)


        bad_result = mock_experiment_result
        original_final_eval = bad_result.final_eval

        class _BrokenEval:

            @property
            def terms(self):
                raise AttributeError("BUG: terms property broken")

            @property
            def coefficients(self):
                return original_final_eval.coefficients

            @property
            def selected_indices(self):
                return original_final_eval.selected_indices

        bad_result.final_eval = _BrokenEval()
        try:
            with pytest.raises(AttributeError, match="BUG"):
                engine._get_integration_result(bad_result, ds)
        finally:
            bad_result.final_eval = original_final_eval

    def test_missing_terms_still_handled(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        from kd.core.integrator import IntegrationResult

        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)


        original_terms = mock_experiment_result.final_eval.terms
        mock_experiment_result.final_eval.terms = None
        try:
            result, notes = engine._get_integration_result(mock_experiment_result, ds)
            assert isinstance(result, IntegrationResult)
            assert not result.success
            assert notes == []
        finally:
            mock_experiment_result.final_eval.terms = original_terms







def _make_smooth_dataset() -> PDEDataset:
    import torch

    from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType

    nx, nt = 16, 8
    x = torch.linspace(0.0, 1.0, nx)
    t = torch.linspace(0.0, 0.2, nt)
    u = (torch.sin(2 * torch.pi * x)[:, None] * torch.exp(-t)[None,:]).to(
        torch.float64
    )
    return PDEDataset(
        name="smooth_1d",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _with_terms(
    base: ExperimentResult,
    terms: list[str],
    coefficients: list[float],
    selected_indices: list[int] | None,
) -> ExperimentResult:
    from dataclasses import replace

    import torch

    final_eval = replace(
        base.final_eval,
        terms=terms,
        coefficients=torch.tensor(coefficients, dtype=torch.float64),
        selected_indices=selected_indices,
    )
    return replace(base, final_eval=final_eval)


class TestNearZeroTermPruning:

    def test_near_zero_term_pruned_and_disclosed(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        ds = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(
            mock_experiment_result, ["u_xx", "t"], [0.1, -9.5e-16], None
        )

        integration, notes = engine._get_integration_result(result, ds)

        assert integration.success is True, integration.warning
        assert len(notes) == 1
        assert "'t'" in notes[0]
        assert "9.5e-16" in notes[0]

    def test_genuine_small_coefficient_not_pruned(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        ds = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(mock_experiment_result, ["u_xx", "t"], [1.0, 1e-4], None)

        integration, notes = engine._get_integration_result(result, ds)

        assert notes == []
        assert integration.success is False
        assert "t" in integration.warning

    def test_prune_threshold_ignores_inactive_terms(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        ds = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(
            mock_experiment_result,
            ["u_xx", "t", "v"],
            [0.1, 1e-16, 1e12],
            [0, 1],
        )

        integration, notes = engine._get_integration_result(result, ds)

        assert integration.success is True, integration.warning
        assert len(notes) == 1
        assert "'t'" in notes[0]
        assert "u_xx" not in notes[0]

    def test_all_zero_coefficients_left_alone(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        ds = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(mock_experiment_result, ["u_xx", "t"], [0.0, 0.0], None)

        integration, notes = engine._get_integration_result(result, ds)

        assert notes == []
        assert integration.success is True, integration.warning


class TestIntegrationWarningReportedOnce:

    def test_failure_warning_appears_exactly_once(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        ds = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)

        result = _with_terms(mock_experiment_result, ["u", "t"], [1.0, 0.5], None)

        report = engine.render_all(result, dataset=ds)

        integration_warnings = [
            w for w in report.warnings if "unrecognised symbols" in w
        ]
        assert len(integration_warnings) == 1, report.warnings

    def test_engine_owns_consequence_framing(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        ds = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(mock_experiment_result, ["u", "t"], [1.0, 0.5], None)

        report = engine.render_all(result, dataset=ds)

        consequence = [w for w in report.warnings if "degraded" in w]
        assert len(consequence) == 1, report.warnings
        assert "other plots and metrics" in consequence[0]

    def test_pruned_run_reports_note_not_warning(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        ds = _make_smooth_dataset()
        engine = VizEngine(output_dir=tmp_path)
        result = _with_terms(
            mock_experiment_result, ["u_xx", "t"], [0.1, -9.5e-16], None
        )

        report = engine.render_all(result, dataset=ds)

        assert not any("unrecognised symbols" in w for w in report.warnings)
        prune_notes = [
            w for w in report.warnings if "excluded from time integration" in w
        ]
        assert len(prune_notes) == 1, report.warnings


class TestAutogradDomainNote:

    @pytest.mark.parametrize(
        ("config", "expect_note"),
        [
            ({"algorithm": "sga", "provider_kind": "finite_diff"}, False),
            (
                {
                    "algorithm": "sga",
                    "provider_kind": "finite_diff",
                    "use_autograd": True,
                },
                True,
            ),
            ({"algorithm": "dlga", "provider_kind": "autograd"}, True),
            ({"algorithm": "discover", "provider_kind": "finite_diff"}, False),
            ({"algorithm": "pysr", "provider_kind": "finite_diff"}, False),
            ({"algorithm": "eqgpt", "provider_kind": "finite_diff"}, False),
        ],
        ids=["sga-fd", "sga-autograd", "dlga-autograd", "discover", "pysr", "eqgpt"],
    )
    def test_note_uses_autograd_domain_metadata(
        self,
        mock_experiment_result: ExperimentResult,
        config: dict[str, object],
        expect_note: bool,
    ) -> None:
        result = replace(mock_experiment_result, config=config)

        note = VizEngine._maybe_autograd_domain_note(result)

        assert (note is not None) is expect_note

    def test_note_wording_is_algorithm_neutral(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        result = replace(
            mock_experiment_result,
            config={"algorithm": "dlga", "provider_kind": "autograd"},
        )

        note = VizEngine._maybe_autograd_domain_note(result)

        assert note is not None
        assert "SGA" not in note
        assert "autograd" in note
        assert "finite-difference" in note
