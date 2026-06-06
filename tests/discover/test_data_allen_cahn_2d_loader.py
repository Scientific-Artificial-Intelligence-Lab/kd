
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.discover.data.allen_cahn_2d import load_allen_cahn_2d
from kd.search.discover.data.loader import add_gaussian_noise


def _field_from_axes(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    t: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    xx, yy, tt = np.meshgrid(x, y, t, indexing="ij")
    field = np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy) * np.exp(-tt)
    return field.astype(np.float32)


def _write_npz(
    path: Path,
    *,
    x: np.ndarray[Any, Any] | None = None,
    y: np.ndarray[Any, Any] | None = None,
    t: np.ndarray[Any, Any] | None = None,
    u: np.ndarray[Any, Any] | None = None,
    axis_order: np.ndarray[Any, Any] | None = None,
) -> Path:
    x_arr = x if x is not None else np.linspace(0.0, 1.0, 8, endpoint=False)
    y_arr = y if y is not None else np.linspace(0.0, 1.0, 8, endpoint=False)
    t_arr = t if t is not None else np.linspace(0.0, 0.5, 6, endpoint=False)
    u_arr = u if u is not None else _field_from_axes(x_arr, y_arr, t_arr)
    order = axis_order if axis_order is not None else np.array(["x", "y", "t"])
    np.savez(
        path,
        u=u_arr.astype(np.float32),
        x=x_arr.astype(np.float32),
        y=y_arr.astype(np.float32),
        t=t_arr.astype(np.float32),
        axis_order=order,
    )
    return path


@pytest.fixture()
def allen_cahn_npz(tmp_path: Path) -> Path:
    return _write_npz(tmp_path / "allen_cahn_2d_smoke.npz")


@pytest.mark.unit
def test_load_allen_cahn_2d_metadata_and_dtype(allen_cahn_npz: Path) -> None:
    dataset = load_allen_cahn_2d(allen_cahn_npz)

    assert dataset.name == "allen_cahn_2d_smoke"
    assert dataset.task_type == TaskType.PDE
    assert dataset.axis_order == ["x", "y", "t"]
    assert dataset.lhs_field == "u"
    assert dataset.lhs_axis == "t"
    assert dataset.ground_truth is None
    assert dataset.fields is not None
    assert dataset.fields["u"].values.dtype == torch.float32
    assert dataset.fields["u"].values.shape == (8, 8, 6)


@pytest.mark.unit
def test_load_allen_cahn_2d_explicit_dtype_flows_to_field(
    allen_cahn_npz: Path,
) -> None:
    dataset = load_allen_cahn_2d(allen_cahn_npz, dtype=torch.float64)

    assert dataset.fields is not None
    assert dataset.fields["u"].values.dtype == torch.float64


@pytest.mark.unit
def test_load_allen_cahn_2d_float64_retains_grid_invariants(
    allen_cahn_npz: Path,
) -> None:
    dataset = load_allen_cahn_2d(allen_cahn_npz, dtype=torch.float64)

    assert dataset.axis_order == ["x", "y", "t"]
    assert dataset.axes is not None
    assert dataset.fields is not None
    assert dataset.axes["x"].is_periodic is True
    assert dataset.axes["y"].is_periodic is True
    assert dataset.axes["t"].is_periodic is False
    assert dataset.fields["u"].values.dtype == torch.float64
    assert dataset.axes["x"].values.dtype == torch.float64
    assert dataset.axes["y"].values.dtype == torch.float64
    assert dataset.axes["t"].values.dtype == torch.float64
    for axis in dataset.axis_order:
        diffs = torch.diff(dataset.axes[axis].values)
        assert torch.all(diffs > 0.0)
    FiniteDiffProvider(dataset, max_order=2)


@pytest.mark.unit
def test_load_allen_cahn_2d_periodic_axes(allen_cahn_npz: Path) -> None:
    dataset = load_allen_cahn_2d(allen_cahn_npz)

    assert dataset.axes is not None
    assert dataset.axes["x"].is_periodic is True
    assert dataset.axes["y"].is_periodic is True
    assert dataset.axes["t"].is_periodic is False


@pytest.mark.unit
def test_load_allen_cahn_2d_decodes_bytes_axis_order(tmp_path: Path) -> None:
    path = _write_npz(
        tmp_path / "bytes_axis_order.npz",
        axis_order=np.array([b"x", b"y", b"t"]),
    )

    dataset = load_allen_cahn_2d(path)

    assert dataset.axis_order == ["x", "y", "t"]


@pytest.mark.unit
def test_load_allen_cahn_2d_rejects_wrong_axis_order(tmp_path: Path) -> None:
    x = np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float32)
    t = np.linspace(0.0, 0.5, 6, endpoint=False, dtype=np.float32)
    u = np.transpose(_field_from_axes(x, y, t), (0, 2, 1))
    path = _write_npz(
        tmp_path / "wrong_order.npz",
        u=u,
        axis_order=np.array(["x", "t", "y"]),
    )

    with pytest.raises(ValueError, match="axis_order"):
        load_allen_cahn_2d(path)


@pytest.mark.unit
def test_load_allen_cahn_2d_regularizes_float32_time_axis(tmp_path: Path) -> None:
    x = np.linspace(0.0, 1.0, 5, endpoint=False, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 5, endpoint=False, dtype=np.float32)
    t = np.linspace(0.0, 5.0, 100, endpoint=False, dtype=np.float32)
    path = _write_npz(tmp_path / "float32_time.npz", x=x, y=y, t=t)

    dataset = load_allen_cahn_2d(path)

    assert dataset.axes is not None
    assert dataset.axes["t"].values.dtype == torch.float64
    FiniteDiffProvider(dataset, max_order=2)


@pytest.mark.integration
def test_load_allen_cahn_2d_finite_diff_provider_smoke(
    allen_cahn_npz: Path,
) -> None:
    dataset = load_allen_cahn_2d(allen_cahn_npz)
    assert dataset.fields is not None
    values = dataset.fields["u"].values
    provider = FiniteDiffProvider(dataset, max_order=2)

    diff2_x = _evaluate_diff2(provider, "x", values)
    diff2_y = _evaluate_diff2(provider, "y", values)

    assert torch.isfinite(diff2_x).all()
    assert torch.isfinite(diff2_y).all()
    assert not torch.isnan(diff2_x).any()
    assert not torch.isnan(diff2_y).any()


def _evaluate_diff2(
    provider: FiniteDiffProvider,
    axis: str,
    values: torch.Tensor,
) -> torch.Tensor:
    evaluate = getattr(provider, "evaluate", None)
    if callable(evaluate):
        result = evaluate(f"diff2_{axis}", {"u": values})
        if isinstance(result, torch.Tensor):
            return result
    return provider.diff(values, axis, 2)


@pytest.mark.unit
def test_load_allen_cahn_2d_rejects_missing_required_key(tmp_path: Path) -> None:
    path = tmp_path / "missing.npz"
    np.savez(path, u=np.zeros((2, 2, 2)), x=np.arange(2), y=np.arange(2))

    with pytest.raises(ValueError, match="missing required keys"):
        load_allen_cahn_2d(path)


@pytest.mark.unit
def test_load_allen_cahn_2d_rejects_shape_mismatch(tmp_path: Path) -> None:
    path = _write_npz(tmp_path / "bad_shape.npz", u=np.zeros((8, 8, 5)))

    with pytest.raises(ValueError, match="shape"):
        load_allen_cahn_2d(path)


@pytest.mark.unit
def test_load_allen_cahn_2d_rejects_nonfinite_values(tmp_path: Path) -> None:
    u = np.zeros((8, 8, 6), dtype=np.float32)
    u[0, 0, 0] = np.nan
    path = _write_npz(tmp_path / "nonfinite.npz", u=u)

    with pytest.raises(ValueError, match="finite"):
        load_allen_cahn_2d(path)


@pytest.mark.unit
def test_load_allen_cahn_2d_rejects_nonmonotonic_axis(tmp_path: Path) -> None:
    x = np.array([0.0, 0.25, 0.5, 0.5, 0.75], dtype=np.float32)
    y = np.linspace(0.0, 1.0, 5, endpoint=False, dtype=np.float32)
    t = np.linspace(0.0, 0.5, 6, endpoint=False, dtype=np.float32)
    u = _field_from_axes(x, y, t)
    path = _write_npz(tmp_path / "nonmonotonic.npz", x=x, y=y, t=t, u=u)

    with pytest.raises(ValueError, match="strictly increasing"):
        load_allen_cahn_2d(path)


@pytest.mark.unit
def test_load_allen_cahn_2d_rejects_bad_uniform_endpoint(tmp_path: Path) -> None:
    x = np.array([0.0, 0.2, 0.4, 0.6, 0.95], dtype=np.float32)
    y = np.linspace(0.0, 1.0, 5, endpoint=False, dtype=np.float32)
    t = np.linspace(0.0, 0.5, 6, endpoint=False, dtype=np.float32)
    u = _field_from_axes(x, y, t)
    path = _write_npz(tmp_path / "bad_uniform.npz", x=x, y=y, t=t, u=u)

    with pytest.raises(ValueError, match="not uniformly spaced"):
        load_allen_cahn_2d(path)


def _toy_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 5, dtype=torch.float32)
    t = torch.linspace(0.0, 1.0, 5, dtype=torch.float32)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    u = xx + tt
    v = 2.0 * xx - tt
    return PDEDataset(
        name="toy",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "t"],
        fields={
            "u": FieldData(name="u", values=u),
            "v": FieldData(name="v", values=v),
        },
        lhs_field="u",
        lhs_axis="t",
        ground_truth="u_t = u_xx",
    )


@pytest.mark.unit
def test_add_gaussian_noise_preserves_metadata_and_does_not_mutate() -> None:
    dataset = _toy_dataset()
    assert dataset.fields is not None
    original_u = dataset.fields["u"].values.clone()

    noisy = add_gaussian_noise(dataset, level=0.1, seed=123)
    assert noisy.fields is not None

    assert noisy is not dataset
    assert noisy.name == "toy_noisy"
    assert noisy.noise_level == pytest.approx(0.1)
    assert noisy.axes is dataset.axes
    assert noisy.axis_order == dataset.axis_order
    assert noisy.topology == dataset.topology
    assert noisy.lhs_field == dataset.lhs_field
    assert noisy.lhs_axis == dataset.lhs_axis
    assert noisy.ground_truth == dataset.ground_truth
    assert dataset.fields["u"].values.equal(original_u)
    assert not noisy.fields["u"].values.equal(original_u)


@pytest.mark.unit
def test_add_gaussian_noise_uses_absmax_scale_and_seed() -> None:
    dataset = _toy_dataset()
    assert dataset.fields is not None
    level = 0.2
    seed = 7
    generator = torch.Generator().manual_seed(seed)
    values = dataset.fields["u"].values
    expected = values + level * values.abs().max() * torch.randn(
        values.shape,
        generator=generator,
        dtype=values.dtype,
        device=values.device,
    )

    noisy = add_gaussian_noise(dataset, level=level, seed=seed, scale="max")
    assert noisy.fields is not None

    assert torch.allclose(noisy.fields["u"].values, expected)
    assert not torch.equal(
        noisy.fields["u"].values - values,
        noisy.fields["v"].values - dataset.fields["v"].values,
    )


@pytest.mark.unit
def test_add_gaussian_noise_default_uses_std_scale() -> None:
    dataset = _toy_dataset()
    assert dataset.fields is not None
    level = 0.2
    seed = 7
    generator = torch.Generator().manual_seed(seed)
    values = dataset.fields["u"].values
    expected = values + level * torch.std(
        values, unbiased=True,
    ) * torch.randn(
        values.shape,
        generator=generator,
        dtype=values.dtype,
        device=values.device,
    )

    noisy = add_gaussian_noise(dataset, level=level, seed=seed)
    assert noisy.fields is not None

    assert torch.allclose(noisy.fields["u"].values, expected)


@pytest.mark.unit
def test_add_gaussian_noise_rejects_invalid_scale() -> None:
    dataset = _toy_dataset()
    with pytest.raises(ValueError, match="scale"):
        add_gaussian_noise(dataset, level=0.1, seed=0, scale="bogus")
