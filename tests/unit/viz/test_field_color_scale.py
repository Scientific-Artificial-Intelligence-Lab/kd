
from __future__ import annotations

import math
import re

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

matplotlib.use("Agg")

from kd.core.integrator import IntegrationResult
from kd.data.schema import FieldData, PDEDataset
from kd.viz.plots.field import plot_field_comparison
from tests.unit.viz.test_field import (
    _make_1d_pde_dataset,
    _make_2d_pde_dataset,
)





_AMPLIFY = 100.0
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _dataset_with_field(values: torch.Tensor) -> PDEDataset:
    base = _make_1d_pde_dataset(nx=values.shape[0], nt=values.shape[1])
    return PDEDataset(
        name=base.name,
        task_type=base.task_type,
        axes=base.axes,
        axis_order=base.axis_order,
        fields={"u": FieldData(name="u", values=values, allow_nan=True)},
        lhs_field="u",
        lhs_axis="t",
    )


def _panels(fig: Figure, prefix: str) -> list[Axes]:
    return [ax for ax in fig.get_axes() if ax.get_title().startswith(prefix)]


def _one_panel(fig: Figure, prefix: str) -> Axes:
    matches = _panels(fig, prefix)
    assert len(matches) == 1, f"expected one {prefix!r} panel, got {len(matches)}"
    return matches[0]


def _mappable(ax: Axes) -> ScalarMappable:
    if ax.images:
        return ax.images[0]
    return ax.collections[0]


def _floats_in(text: str) -> list[float]:
    return [float(tok) for tok in _FLOAT_RE.findall(text)]


def _amplified_result(dataset: PDEDataset) -> IntegrationResult:
    return IntegrationResult(
        success=True,
        predicted_field=dataset.get_field("u") * _AMPLIFY,
    )







class TestSharedColorScale1d:

    def test_true_and_predicted_share_clim_with_colorbars(self) -> None:
        ds = _make_1d_pde_dataset()
        fig, _ = plot_field_comparison(ds, _amplified_result(ds))
        try:
            true_m = _mappable(_one_panel(fig, "True"))
            pred_m = _mappable(_one_panel(fig, "Predicted"))
            assert true_m.get_clim() == pred_m.get_clim()
            assert true_m.colorbar is not None
            assert pred_m.colorbar is not None
        finally:
            plt.close(fig)

    def test_shared_limits_track_the_true_field_not_the_union(self) -> None:
        ds = _make_1d_pde_dataset()
        true_max = float(ds.get_field("u").abs().max())
        fig, _ = plot_field_comparison(ds, _amplified_result(ds))
        try:
            vmin, vmax = _mappable(_one_panel(fig, "True")).get_clim()


            assert max(abs(vmin), abs(vmax)) <= true_max
        finally:
            plt.close(fig)

    def test_saturated_predicted_colorbar_is_marked_extended(self) -> None:
        ds = _make_1d_pde_dataset()
        fig, _ = plot_field_comparison(ds, _amplified_result(ds))
        try:
            colorbar = _mappable(_one_panel(fig, "Predicted")).colorbar
            assert colorbar.extend == "both"
        finally:
            plt.close(fig)

    def test_predicted_title_carries_actual_out_of_range_values(self) -> None:
        ds = _make_1d_pde_dataset()
        pred = ds.get_field("u") * _AMPLIFY
        fig, _ = plot_field_comparison(ds, _amplified_result(ds))
        try:
            numbers = _floats_in(_one_panel(fig, "Predicted").get_title())
            for expected in (float(pred.min()), float(pred.max())):
                assert any(n == pytest.approx(expected, rel=0.02) for n in numbers), (
                    f"{expected} missing from title numbers {numbers}"
                )
        finally:
            plt.close(fig)

    def test_in_range_prediction_gets_no_clipping_note(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = IntegrationResult(success=True, predicted_field=ds.get_field("u") * 0.5)
        fig, _ = plot_field_comparison(ds, ir)
        try:
            assert "clipped" not in _one_panel(fig, "Predicted").get_title()
        finally:
            plt.close(fig)

    def test_residual_keeps_independent_symmetric_scale(self) -> None:
        ds = _make_1d_pde_dataset()
        fig, _ = plot_field_comparison(ds, _amplified_result(ds))
        try:
            res_vmin, res_vmax = _mappable(_one_panel(fig, "Residual")).get_clim()
            shared = _mappable(_one_panel(fig, "True")).get_clim()
            assert res_vmin == pytest.approx(-res_vmax)

            assert res_vmax > max(abs(v) for v in shared)
        finally:
            plt.close(fig)


class TestTrueFieldReferenceScale:

    def test_ordinary_true_field_is_not_clipped(self) -> None:
        ds = _make_1d_pde_dataset()
        ir = IntegrationResult(success=True, predicted_field=ds.get_field("u"))
        fig, _ = plot_field_comparison(ds, ir)
        try:
            true_ax = _one_panel(fig, "True")
            vmin, vmax = _mappable(true_ax).get_clim()
            field = ds.get_field("u")
            assert vmin == pytest.approx(float(field.min()))
            assert vmax == pytest.approx(float(field.max()))
            assert "clipped" not in true_ax.get_title()
        finally:
            plt.close(fig)

    def test_lone_spike_stays_inside_the_true_scale(self) -> None:
        values = _make_1d_pde_dataset().get_field("u").clone()
        values[0, 0] = 1.0e6
        ds = _dataset_with_field(values)
        ir = IntegrationResult(success=True, predicted_field=values.clone())
        fig, _ = plot_field_comparison(ds, ir)
        try:
            true_ax = _one_panel(fig, "True")
            _, vmax = _mappable(true_ax).get_clim()
            assert vmax == pytest.approx(1.0e6)

            assert "clipped" not in true_ax.get_title()
        finally:
            plt.close(fig)

    def test_sparse_signal_over_a_noise_floor_is_not_trimmed_away(self) -> None:
        generator = torch.Generator().manual_seed(20260725)
        values = torch.randn(100, 100, generator=generator) * 1.0e-3

        values[10:15, 10:20] = 1.0
        ds = _dataset_with_field(values)
        ir = IntegrationResult(success=True, predicted_field=values.clone())
        fig, _ = plot_field_comparison(ds, ir)
        try:
            true_ax = _one_panel(fig, "True")
            vmin, vmax = _mappable(true_ax).get_clim()
            assert vmax == pytest.approx(float(values.max()))
            assert vmin == pytest.approx(float(values.min()))
            assert "clipped" not in true_ax.get_title()
        finally:
            plt.close(fig)


class TestDegenerateFieldLimits:

    def test_constant_field_limits_are_finite_and_non_degenerate(self) -> None:
        values = torch.full((6, 5), 3.0)
        ds = _dataset_with_field(values)
        ir = IntegrationResult(success=True, predicted_field=values.clone())
        fig, _ = plot_field_comparison(ds, ir)
        try:
            vmin, vmax = _mappable(_one_panel(fig, "True")).get_clim()
            assert math.isfinite(vmin)
            assert math.isfinite(vmax)
            assert vmin < vmax
        finally:
            plt.close(fig)

    def test_all_nan_field_limits_are_finite_and_non_degenerate(self) -> None:
        values = torch.full((6, 5), float("nan"))
        ds = _dataset_with_field(values)
        ir = IntegrationResult(success=True, predicted_field=values.clone())
        fig, _ = plot_field_comparison(ds, ir)
        try:
            vmin, vmax = _mappable(_one_panel(fig, "True")).get_clim()
            assert math.isfinite(vmin)
            assert math.isfinite(vmax)
            assert vmin < vmax
        finally:
            plt.close(fig)

    def test_zero_field_limits_are_finite_and_non_degenerate(self) -> None:
        values = torch.zeros(6, 5)
        ds = _dataset_with_field(values)
        ir = IntegrationResult(success=True, predicted_field=values.clone())
        fig, _ = plot_field_comparison(ds, ir)
        try:
            vmin, vmax = _mappable(_one_panel(fig, "True")).get_clim()
            assert math.isfinite(vmin)
            assert math.isfinite(vmax)
            assert vmin < vmax
        finally:
            plt.close(fig)







class TestSharedColorScale2d:

    def test_all_field_panels_share_one_clim_and_have_colorbars(self) -> None:
        ds = _make_2d_pde_dataset(nt=5)
        fig, _ = plot_field_comparison(ds, _amplified_result(ds))
        try:
            field_axes = _panels(fig, "True") + _panels(fig, "Predicted")
            assert len(field_axes) == 6
            clims = {_mappable(ax).get_clim() for ax in field_axes}
            assert len(clims) == 1, f"panels disagree on scale: {clims}"
            assert all(_mappable(ax).colorbar is not None for ax in field_axes)
        finally:
            plt.close(fig)

    def test_predicted_titles_carry_actual_out_of_range_values(self) -> None:
        ds = _make_2d_pde_dataset(nt=5)
        fig, _ = plot_field_comparison(ds, _amplified_result(ds))
        try:
            pred_axes = _panels(fig, "Predicted")
            assert pred_axes
            for ax in pred_axes:
                image_data = _mappable(ax).get_array()
                numbers = _floats_in(ax.get_title())
                for expected in (float(image_data.min()), float(image_data.max())):
                    assert any(
                        n == pytest.approx(expected, rel=0.02) for n in numbers
                    ), f"{expected} missing from title numbers {numbers}"
        finally:
            plt.close(fig)

    def test_residual_row_keeps_independent_symmetric_scale(self) -> None:
        ds = _make_2d_pde_dataset(nt=5)
        fig, _ = plot_field_comparison(ds, _amplified_result(ds))
        try:
            shared = _mappable(_panels(fig, "True")[0]).get_clim()
            for ax in _panels(fig, "Residual"):
                vmin, vmax = _mappable(ax).get_clim()
                assert vmin == pytest.approx(-vmax)
                assert vmax > max(abs(v) for v in shared)
        finally:
            plt.close(fig)
