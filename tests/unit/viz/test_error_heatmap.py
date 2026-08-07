
from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

matplotlib.use("Agg")

from matplotlib.figure import Figure

from kd.core.integrator import IntegrationResult
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.viz.plots.error_heatmap import plot_error_heatmap





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
    noise_std: float = 0.05,
) -> IntegrationResult:
    if success:
        true_field = dataset.get_field("u")
        return IntegrationResult(
            success=True,
            predicted_field=true_field + torch.randn_like(true_field) * noise_std,
        )
    return IntegrationResult(success=False, warning="Integration failed")







class TestErrorHeatmapSmoke:

    def test_callable_returns_tuple(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        fig, warnings = plot_error_heatmap(ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_accepts_style_kwarg(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        fig, _ = plot_error_heatmap(ds, ir, style={"font.size": 12})
        plt.close(fig)







class TestErrorHeatmapHappyPath:

    def test_2d_custom_time_title_uses_lhs_axis_name(
        self,
        custom_axis_2d_dataset: PDEDataset,
    ) -> None:
        ir = _make_integration_result(custom_axis_2d_dataset)
        fig, _ = plot_error_heatmap(custom_axis_2d_dataset, ir)
        try:
            title_text = " ".join(ax.get_title() for ax in fig.get_axes())
            assert "tau=" in title_text
            assert "t=" not in title_text
        finally:
            plt.close(fig)

    def test_1d_produces_axes(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        fig, _ = plot_error_heatmap(ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_2d_produces_axes(self) -> None:
        ds = _make_2d_dataset()
        ir = _make_integration_result(ds)
        fig, _ = plot_error_heatmap(ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_2d_renders_multiple_physical_panels_with_shared_colorbar(
        self,
        rectangular_2d_dataset: PDEDataset,
    ) -> None:
        ir = _make_integration_result(rectangular_2d_dataset)
        fig, _ = plot_error_heatmap(rectangular_2d_dataset, ir)
        try:
            data_axes = [ax for ax in fig.get_axes() if ax.images]
            assert len(data_axes) == 3
            assert len(fig.get_axes()) == 4
            clims = {ax.images[0].get_clim() for ax in data_axes}
            assert len(clims) == 1
            for ax in data_axes:
                assert tuple(ax.images[0].get_extent()) == pytest.approx(
                    (10.0, 14.0, -2.0, 3.0)
                )
                assert ax.get_xlabel() == "eta"
                assert ax.get_ylabel() == "xi"
        finally:
            plt.close(fig)

    def test_has_colorbar_or_colormap(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        fig, _ = plot_error_heatmap(ds, ir)
        try:

            has_visual = any(
                len(ax.images) > 0 or len(ax.collections) > 0 for ax in fig.get_axes()
            )
            assert has_visual, "Expected heatmap/pcolormesh in the figure"
        finally:
            plt.close(fig)

    def test_title_present(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        fig, _ = plot_error_heatmap(ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            suptitle = fig._suptitle.get_text() if fig._suptitle else ""
            all_text = " ".join(titles) + " " + suptitle
            assert len(all_text.strip()) > 0, "Expected at least one title"
        finally:
            plt.close(fig)







class TestErrorHeatmapEdgeCases:

    def test_failed_integration_no_crash(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds, success=False)
        fig, warnings = plot_error_heatmap(ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_none_predicted_field(self) -> None:
        ds = _make_1d_dataset()
        ir = IntegrationResult(
            success=False,
            predicted_field=None,
            warning="Total failure",
        )
        fig, warnings = plot_error_heatmap(ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert len(warnings) > 0
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
        fig, _ = plot_error_heatmap(ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_minimal_dataset(self) -> None:
        ds = _make_1d_dataset(nx=2, nt=2)
        ir = _make_integration_result(ds)
        fig, _ = plot_error_heatmap(ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_no_figure_leak(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        figs_before = len(plt.get_fignums())
        fig, _ = plot_error_heatmap(ds, ir)
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


class TestErrorHeatmapAxisOrderNone:

    def test_axis_order_none_no_crash(self) -> None:
        ds = _make_ode_dataset_no_axis_order()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        fig, warnings = plot_error_heatmap(ds, ir)
        try:
            assert isinstance(fig, Figure)

            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_axis_order_none_returns_valid_figure(self) -> None:
        ds = _make_ode_dataset_no_axis_order()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        fig, _ = plot_error_heatmap(ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_axis_order_none_failed_integration(self) -> None:
        ds = _make_ode_dataset_no_axis_order()
        ir = IntegrationResult(success=False, warning="Integration failed")
        fig, warnings = plot_error_heatmap(ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


class TestErrorHeatmapNSpatialZero:

    def test_n_spatial_zero_no_crash(self) -> None:
        ds = _make_ode_dataset_n_spatial_0()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        fig, warnings = plot_error_heatmap(ds, ir)
        try:
            assert isinstance(fig, Figure)

            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_n_spatial_zero_returns_valid_figure(self) -> None:
        ds = _make_ode_dataset_n_spatial_0()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        fig, _ = plot_error_heatmap(ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_n_spatial_zero_failed_integration(self) -> None:
        ds = _make_ode_dataset_n_spatial_0()
        ir = IntegrationResult(success=False, warning="Integration failed")
        fig, warnings = plot_error_heatmap(ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)







class TestErrorHeatmapNormalRegression:

    def test_1d_normal_dataset_unchanged(self) -> None:
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        fig, warnings = plot_error_heatmap(ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert len(fig.get_axes()) >= 1

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
        fig, warnings = plot_error_heatmap(ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)
