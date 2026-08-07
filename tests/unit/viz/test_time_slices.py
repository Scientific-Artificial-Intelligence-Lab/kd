
from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

matplotlib.use("Agg")

from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from kd.core.integrator import IntegrationResult
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.viz.plots.time_slices import plot_time_slices





_TWO_PI = 2.0 * math.pi


def _make_1d_dataset(nx: int = 20, nt: int = 10) -> PDEDataset:
    x = torch.linspace(0, _TWO_PI, nx)
    t = torch.linspace(0, 1, nt)
    u_field = torch.sin(x).unsqueeze(1) * torch.exp(-t).unsqueeze(0)
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


def _make_2d_dataset(nx: int = 10, ny: int = 10, nt: int = 5) -> PDEDataset:
    x = torch.linspace(0, _TWO_PI, nx)
    y = torch.linspace(0, _TWO_PI, ny)
    t = torch.linspace(0, 1, nt)
    u_field = (
        torch.sin(x).reshape(nx, 1, 1)
        * torch.cos(y).reshape(1, ny, 1)
        * torch.exp(-t).reshape(1, 1, nt)
    )
    return PDEDataset(
        name="test_2d",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "y": AxisInfo(name="y", values=y, is_periodic=True),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "y", "t"],
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_integration_result(
    dataset: PDEDataset,
    *,
    success: bool = True,
    noise_std: float = 0.01,
) -> IntegrationResult:
    if success:
        true_field = dataset.get_field("u")
        return IntegrationResult(
            success=True,
            predicted_field=true_field + torch.randn_like(true_field) * noise_std,
        )
    return IntegrationResult(success=False, warning="Integration failed")


_AMPLIFY = 100.0


def _panels(fig: Figure, prefix: str) -> list[Axes]:
    return [ax for ax in fig.get_axes() if ax.get_title().startswith(prefix)]


def _mappable(ax: Axes) -> ScalarMappable:
    if ax.images:
        return ax.images[0]
    return ax.collections[0]







class TestTimeSlicesSmoke:

    def test_callable_returns_tuple(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        fig, warnings = plot_time_slices(ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_accepts_style_kwarg(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        fig, _ = plot_time_slices(ds, ir, style={"font.size": 12})
        plt.close(fig)

    def test_accepts_n_slices_kwarg(self) -> None:
        ds = _make_1d_dataset(nt=10)
        ir = _make_integration_result(ds)
        fig, _ = plot_time_slices(ds, ir, n_slices=5)
        plt.close(fig)







class TestTimeSlicesHappyPath:

    def test_1d_custom_axis_titles_use_lhs_axis_name(
        self,
        custom_axis_dataset: PDEDataset,
    ) -> None:
        ir = _make_integration_result(custom_axis_dataset)
        fig, _ = plot_time_slices(
            custom_axis_dataset,
            ir,
            n_slices=2,
        )
        try:
            plot_axes = fig.get_axes()
            title_text = " ".join(ax.get_title() for ax in plot_axes)
            assert "tau =" in title_text
            assert "t =" not in title_text
            assert {ax.get_xlabel() for ax in plot_axes} == {"xi"}
        finally:
            plt.close(fig)

    def test_1d_produces_panels(self) -> None:
        ds = _make_1d_dataset(nt=10)
        ir = _make_integration_result(ds)
        fig, _ = plot_time_slices(ds, ir, n_slices=3)
        try:

            assert len(fig.get_axes()) >= 3
        finally:
            plt.close(fig)

    def test_2d_produces_panels(self) -> None:
        ds = _make_2d_dataset(nt=5)
        ir = _make_integration_result(ds)
        fig, _ = plot_time_slices(ds, ir, n_slices=3)
        try:
            assert len(fig.get_axes()) >= 3
        finally:
            plt.close(fig)

    def test_2d_uses_physical_extent_and_axis_labels(
        self,
        rectangular_2d_dataset: PDEDataset,
    ) -> None:
        ir = _make_integration_result(rectangular_2d_dataset)
        fig, _ = plot_time_slices(rectangular_2d_dataset, ir, n_slices=3)
        try:
            data_axes = [ax for ax in fig.get_axes() if ax.images]
            assert data_axes
            image = data_axes[0].images[0]


            assert tuple(image.get_extent()) == pytest.approx((10.0, 14.0, -2.0, 3.0))
            assert tuple(image.get_extent()) != pytest.approx((-0.5, 3.5, -0.5, 4.5))
            assert {ax.get_xlabel() for ax in data_axes} == {"eta"}
            assert {ax.get_ylabel() for ax in data_axes} == {"xi"}
        finally:
            plt.close(fig)

    def test_n_slices_affects_panel_count(self) -> None:
        ds = _make_1d_dataset(nt=20)
        ir = _make_integration_result(ds)

        fig2, _ = plot_time_slices(ds, ir, n_slices=2)
        fig5, _ = plot_time_slices(ds, ir, n_slices=5)
        try:
            assert len(fig5.get_axes()) > len(fig2.get_axes())
        finally:
            plt.close(fig2)
            plt.close(fig5)

    def test_panels_have_titles(self) -> None:
        ds = _make_1d_dataset(nt=10)
        ir = _make_integration_result(ds)
        fig, _ = plot_time_slices(ds, ir, n_slices=3)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            non_empty = [t for t in titles if t]
            assert len(non_empty) >= 3
        finally:
            plt.close(fig)







class TestTimeSlicesEdgeCases:

    def test_failed_integration_no_crash(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds, success=False)
        fig, warnings = plot_time_slices(ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_single_time_step(self) -> None:
        ds = _make_1d_dataset(nx=10, nt=1)
        ir = _make_integration_result(ds)
        fig, _ = plot_time_slices(ds, ir, n_slices=3)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_n_slices_larger_than_nt(self) -> None:
        ds = _make_1d_dataset(nt=2)
        ir = _make_integration_result(ds)
        fig, _ = plot_time_slices(ds, ir, n_slices=10)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_diverged_integration(self) -> None:
        ds = _make_1d_dataset()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(
            success=False,
            predicted_field=pred,
            warning="Diverged at t=0.5",
            diverged_at_t=0.5,
        )
        fig, warnings = plot_time_slices(ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_no_figure_leak(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        figs_before = len(plt.get_fignums())
        fig, _ = plot_time_slices(ds, ir)
        plt.close(fig)
        figs_after = len(plt.get_fignums())
        assert figs_after <= figs_before







def _make_ode_dataset_no_axis_order() -> PDEDataset:
    nt = 10
    t = torch.linspace(0, 1, nt)
    u_field = torch.sin(t)
    return PDEDataset(
        name="test_ode",
        task_type=TaskType.ODE,
        axes=None,
        axis_order=None,
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_ode_dataset_n_spatial_0() -> PDEDataset:
    nt = 10
    t = torch.linspace(0, 1, nt)
    u_field = torch.sin(t)
    return PDEDataset(
        name="test_ode_t_only",
        task_type=TaskType.ODE,
        axes={"t": AxisInfo(name="t", values=t)},
        axis_order=["t"],
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


class TestTimeSlicesAxisOrderNone:

    def test_axis_order_none_no_crash(self) -> None:
        ds = _make_ode_dataset_no_axis_order()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        fig, warnings = plot_time_slices(ds, ir)
        try:
            assert isinstance(fig, Figure)

            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_axis_order_none_returns_valid_figure(self) -> None:
        ds = _make_ode_dataset_no_axis_order()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        fig, _ = plot_time_slices(ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_axis_order_none_failed_integration(self) -> None:
        ds = _make_ode_dataset_no_axis_order()
        ir = IntegrationResult(success=False, warning="Integration failed")
        fig, warnings = plot_time_slices(ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


class TestTimeSlicesNSpatialZero:

    def test_n_spatial_zero_no_crash(self) -> None:
        ds = _make_ode_dataset_n_spatial_0()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        fig, warnings = plot_time_slices(ds, ir)
        try:
            assert isinstance(fig, Figure)

            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_n_spatial_zero_returns_valid_figure(self) -> None:
        ds = _make_ode_dataset_n_spatial_0()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        fig, _ = plot_time_slices(ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_n_spatial_zero_failed_integration(self) -> None:
        ds = _make_ode_dataset_n_spatial_0()
        ir = IntegrationResult(success=False, warning="Integration failed")
        fig, warnings = plot_time_slices(ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)







class TestTimeSlicesDivergedTitle:

    def test_diverged_1d_title_contains_diverged(self) -> None:
        ds = _make_1d_dataset()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(
            success=False,
            predicted_field=pred,
            warning="Diverged at t=0.5",
            diverged_at_t=0.5,
        )
        fig, _ = plot_time_slices(ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            all_titles = " ".join(titles)
            assert "DIVERGED" in all_titles, (
                f"Expected 'DIVERGED' in panel titles, got: {titles}"
            )
        finally:
            plt.close(fig)

    def test_diverged_2d_title_contains_diverged(self) -> None:
        ds = _make_2d_dataset()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(
            success=False,
            predicted_field=pred,
            warning="Diverged at t=0.3",
            diverged_at_t=0.3,
        )
        fig, _ = plot_time_slices(ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            all_titles = " ".join(titles)
            assert "DIVERGED" in all_titles, (
                f"Expected 'DIVERGED' in panel titles, got: {titles}"
            )
        finally:
            plt.close(fig)

    def test_diverged_without_at_t_still_has_tag(self) -> None:
        ds = _make_1d_dataset()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(
            success=False,
            predicted_field=pred,
            warning="Integration did not succeed",
            diverged_at_t=None,
        )
        fig, _ = plot_time_slices(ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            all_titles = " ".join(titles)
            assert "DIVERGED" in all_titles, (
                f"Expected 'DIVERGED' in panel titles, got: {titles}"
            )
        finally:
            plt.close(fig)

    def test_non_diverged_no_diverged_tag(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds, success=True)
        fig, _ = plot_time_slices(ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            all_titles = " ".join(titles)
            assert "DIVERGED" not in all_titles, (
                f"Non-diverged should not have DIVERGED tag, got: {titles}"
            )
        finally:
            plt.close(fig)







class TestTimeSlicesNormalRegression:

    def test_1d_normal_dataset_unchanged(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        fig, warnings = plot_time_slices(ds, ir, n_slices=3)
        try:
            assert isinstance(fig, Figure)
            assert len(fig.get_axes()) >= 3

            guard_warnings = [
                w for w in warnings if "spatial" in w.lower() or "axis" in w.lower()
            ]
            assert len(guard_warnings) == 0, (
                f"Normal dataset should not trigger guards: {guard_warnings}"
            )
        finally:
            plt.close(fig)

    def test_2d_normal_dataset_unchanged(self) -> None:
        ds = _make_2d_dataset()
        ir = _make_integration_result(ds)
        fig, warnings = plot_time_slices(ds, ir, n_slices=3)
        try:
            assert isinstance(fig, Figure)
            assert len(fig.get_axes()) >= 3
        finally:
            plt.close(fig)







class TestTimeSlicesColorScale:

    def test_true_and_predicted_share_clim_with_colorbars(self) -> None:
        ds = _make_2d_dataset(nt=5)
        ir = IntegrationResult(
            success=True, predicted_field=ds.get_field("u") * _AMPLIFY
        )
        fig, _ = plot_time_slices(ds, ir, n_slices=3)
        try:
            field_axes = _panels(fig, "True") + _panels(fig, "Predicted")
            assert field_axes
            clims = {_mappable(ax).get_clim() for ax in field_axes}
            assert len(clims) == 1, f"panels disagree on scale: {clims}"
            assert all(_mappable(ax).colorbar is not None for ax in field_axes)
        finally:
            plt.close(fig)

    def test_snapshots_share_one_scale_so_amplitude_decay_stays_visible(self) -> None:
        ds = _make_2d_dataset(nt=5)
        ir = _make_integration_result(ds)
        fig, _ = plot_time_slices(ds, ir, n_slices=3)
        try:
            true_axes = _panels(fig, "True")
            assert len(true_axes) > 1
            clims = {_mappable(ax).get_clim() for ax in true_axes}
            assert len(clims) == 1, f"snapshots renormalised per column: {clims}"


            _, vmax = clims.pop()
            assert vmax == pytest.approx(float(ds.get_field("u").max()))
        finally:
            plt.close(fig)

    def test_clipped_prediction_reports_its_real_range(self) -> None:
        ds = _make_2d_dataset(nt=5)
        ir = IntegrationResult(
            success=True, predicted_field=ds.get_field("u") * _AMPLIFY
        )
        fig, _ = plot_time_slices(ds, ir, n_slices=3)
        try:
            pred_axes = _panels(fig, "Predicted")
            assert pred_axes
            for ax in pred_axes:
                assert "clipped" in ax.get_title()
                assert _mappable(ax).colorbar.extend == "both"
        finally:
            plt.close(fig)

    def test_failed_integration_still_leaves_true_panels_scaled(self) -> None:
        ds = _make_2d_dataset(nt=5)
        ir = _make_integration_result(ds, success=False)
        fig, _ = plot_time_slices(ds, ir, n_slices=3)
        try:
            true_axes = _panels(fig, "True")
            assert true_axes
            clims = {_mappable(ax).get_clim() for ax in true_axes}
            assert len(clims) == 1
            assert all(_mappable(ax).colorbar is not None for ax in true_axes)
        finally:
            plt.close(fig)


class TestTimeSlices1DYScale:

    def test_columns_share_one_ylim_so_amplitude_decay_stays_visible(self) -> None:
        ds = _make_1d_dataset(nt=10)
        ir = _make_integration_result(ds)
        fig, _ = plot_time_slices(ds, ir, n_slices=3)
        try:
            panels = [ax for ax in fig.get_axes() if ax.lines]
            assert len(panels) > 1
            ylims = {ax.get_ylim() for ax in panels}
            assert len(ylims) == 1, f"columns renormalised independently: {ylims}"


            _, top = ylims.pop()
            assert top >= float(ds.get_field("u").max())
        finally:
            plt.close(fig)

    def test_constant_true_field_does_not_shrink_the_prediction_away(self) -> None:
        ds = _make_1d_dataset(nt=10)
        ds.fields["u"] = FieldData(name="u", values=torch.zeros_like(ds.get_field("u")))
        x = ds.get_coords("x")
        pred = (0.05 * torch.sin(x)).unsqueeze(1).expand_as(ds.get_field("u"))
        ir = IntegrationResult(success=True, predicted_field=pred.contiguous())
        fig, _ = plot_time_slices(ds, ir, n_slices=3)
        try:
            panels = [ax for ax in fig.get_axes() if ax.lines]
            assert panels
            lo, hi = panels[0].get_ylim()
            pred_span = float(pred.max() - pred.min())


            assert pred_span / (hi - lo) > 0.5
        finally:
            plt.close(fig)

    def test_prediction_inside_the_margin_is_not_called_clipped(self) -> None:
        ds = _make_1d_dataset(nt=10)
        ir = IntegrationResult(success=True, predicted_field=ds.get_field("u") * 1.02)
        fig, _ = plot_time_slices(ds, ir, n_slices=3)
        try:
            panels = [ax for ax in fig.get_axes() if ax.lines]
            assert panels
            assert not any("clipped" in ax.get_title() for ax in panels)
        finally:
            plt.close(fig)

    def test_prediction_leaving_the_shared_range_is_disclosed(self) -> None:
        ds = _make_1d_dataset(nt=10)
        ir = IntegrationResult(
            success=True, predicted_field=ds.get_field("u") * _AMPLIFY
        )
        fig, _ = plot_time_slices(ds, ir, n_slices=3)
        try:
            panels = [ax for ax in fig.get_axes() if ax.lines]
            assert panels
            for ax in panels:
                assert "Predicted" in ax.get_title()
                assert "clipped" in ax.get_title()
        finally:
            plt.close(fig)


class TestNoPredictionReason:

    def test_2d_missing_prediction_panel_carries_reason_and_title(
        self, custom_axis_2d_dataset: PDEDataset
    ) -> None:
        reason = "Unsupported function calls in RHS expression: ['diff_x']"
        ir = IntegrationResult(success=False, warning=reason)
        fig, warnings = plot_time_slices(custom_axis_2d_dataset, ir)
        try:
            texts = [t.get_text() for ax in fig.axes for t in ax.texts]
            assert any("diff_x" in t for t in texts)
            assert any(ax.get_title() == "Predicted" for ax in fig.axes)
            assert any("diff_x" in w for w in warnings)
        finally:
            plt.close(fig)

    def test_1d_missing_prediction_carries_reason(
        self, custom_axis_dataset: PDEDataset
    ) -> None:
        reason = "Unsupported function calls in RHS expression: ['diff_x']"
        ir = IntegrationResult(success=False, warning=reason)
        fig, _warnings = plot_time_slices(custom_axis_dataset, ir)
        try:
            texts = [t.get_text() for ax in fig.axes for t in ax.texts]
            assert any("diff_x" in t for t in texts)
        finally:
            plt.close(fig)

    def test_shape_mismatch_is_not_reported_as_integration_failure(
        self, custom_axis_dataset: PDEDataset
    ) -> None:
        wrong_shape = torch.zeros(3, 3)
        ir = IntegrationResult(success=True, predicted_field=wrong_shape)
        fig, warnings = plot_time_slices(custom_axis_dataset, ir)
        try:
            texts = [t.get_text() for ax in fig.axes for t in ax.texts]
            assert any("Shape mismatch" in t for t in texts)
            assert not any("Integration failed" in t for t in texts)
            assert any("Shape mismatch" in w for w in warnings)
        finally:
            plt.close(fig)
