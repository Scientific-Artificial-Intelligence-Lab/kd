
from __future__ import annotations

import io

import pytest
import torch

from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.data.synthetic import generate_burgers_data
from kd.inspect import preview






def _make_dataset(
    *,
    nx: int = 32,
    nt: int = 16,
    periodic_x: bool = False,
    nonuniform: bool = False,
    name: str = "synthetic",
) -> PDEDataset:
    if nonuniform:

        x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64) ** 2
    else:
        x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.randn(nx, nt, dtype=torch.float64)
    return PDEDataset(
        name=name,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=periodic_x),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _capture_preview(ds: PDEDataset) -> str:
    buf = io.StringIO()
    preview(ds, file=buf)
    return buf.getvalue()







@pytest.mark.smoke
def test_preview_basic() -> None:
    ds = generate_burgers_data(nx=32, nt=16, nu=0.1, seed=0)
    text = _capture_preview(ds)

    assert "Dataset:" in text
    assert "Axes:" in text
    assert "Fields:" in text
    assert "LHS:" in text

    assert ds.name in text

    assert "x " in text or "x|" in text or "x |" in text
    assert "t " in text or "t|" in text or "t |" in text







def test_preview_uniform_spacing() -> None:
    ds = _make_dataset(nx=32, nt=20)
    text = _capture_preview(ds)
    assert "uniform" in text
    assert "NON-UNIFORM" not in text
    assert "WARNING:" not in text







def test_preview_non_uniform_warning() -> None:
    ds = _make_dataset(nx=32, nt=20, nonuniform=True)
    text = _capture_preview(ds)
    assert "NON-UNIFORM" in text
    assert "WARNING:" in text
    assert "not uniformly spaced" in text or "uniform" in text







def test_preview_nan_warning() -> None:
    ds = _make_dataset(nx=20, nt=12)
    assert ds.fields is not None

    f = ds.fields["u"].values
    f[0, 0] = float("nan")
    f[1, 1] = float("inf")

    text = _capture_preview(ds)
    assert "NaN=1" in text or "NaN=" in text

    assert "NaN" in text

    assert "warning" in text.lower()







def test_preview_small_grid_warning() -> None:
    ds = _make_dataset(nx=8, nt=20)
    text = _capture_preview(ds)
    assert "WARNING:" in text

    assert "small grid" in text or "8 points" in text







def test_preview_no_warnings_clean_data() -> None:
    ds = generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
    text = _capture_preview(ds)
    assert "Status: ready to fit" in text
    assert "WARNING:" not in text







def test_preview_default_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    ds = generate_burgers_data(nx=32, nt=16, nu=0.1, seed=0)
    preview(ds)
    out = capsys.readouterr().out
    assert "Dataset:" in out
    assert ds.name in out







def test_preview_top_level_export() -> None:
    import kd
    import kd.inspect as kd_inspect

    assert kd.preview is kd_inspect.preview











def _make_dataset_with_x(x: torch.Tensor, name: str = "f3") -> PDEDataset:
    nt = 8
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.zeros(x.shape[0], nt, dtype=torch.float64)
    return PDEDataset(
        name=name,
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def test_preview_float32_linspace_does_not_warn() -> None:
    x = torch.linspace(0.0, 1.0, 2000, dtype=torch.float32)
    ds = _make_dataset_with_x(x, name="float32_linspace")
    text = _capture_preview(ds)

    assert "NON-UNIFORM" not in text, (
        "preview must use the same uniform-grid predicate as "
        "FiniteDiffProvider (UNIFORM_GRID_RTOL=1e-4); the local rtol=1e-6 "
        "was rejecting float32 linspace drift the FD path accepts."
    )
    assert "uniform" in text


def test_preview_descending_axis_warns_explicitly() -> None:
    x = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0], dtype=torch.float64)
    ds = _make_dataset_with_x(x, name="descending")
    text = _capture_preview(ds)
    assert "WARNING:" in text, (
        "preview must warn on descending coords — FD stencils reject them "
        "(dx<0 silently flips derivative signs), so the verdict cannot be "
        "'uniform'."
    )


    assert "decreasing" in text.lower() or "NON-UNIFORM" in text


def test_preview_pathological_inf_dx_warns() -> None:
    x = torch.tensor([0.0, 0.1, 0.2, 1.0e308, 0.4], dtype=torch.float64)
    ds = _make_dataset_with_x(x, name="inf_drift")
    text = _capture_preview(ds)
    assert "WARNING:" in text, (
        "preview must reject grids with inf or extreme drift the FD "
        "predicate rejects, not display them as uniform."
    )
