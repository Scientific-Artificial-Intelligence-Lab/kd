
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "discover" / "generate_allen_cahn_2d.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_generate_allen_cahn_2d",
        _SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator_module() -> ModuleType:
    return _load_script_module()


def _valid_snapshots() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    base = (0.4 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)).astype(
        np.float32,
    )
    scales = np.linspace(1.0, 0.8, 6, dtype=np.float32)
    return np.stack([scale * base for scale in scales], axis=-1)


@pytest.mark.unit
def test_initial_condition_is_deterministic(generator_module: ModuleType) -> None:
    first = generator_module._initial_condition(123, 64)
    second = generator_module._initial_condition(123, 64)

    assert first.shape == (64, 64)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)


@pytest.mark.unit
def test_write_dataset_uses_canonical_keys_and_axis_order(
    generator_module: ModuleType,
    tmp_path: Path,
) -> None:
    x = np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float32)
    t = np.linspace(0.0, 0.3, 6, endpoint=False, dtype=np.float32)
    u = _valid_snapshots()
    out = tmp_path / "allen_cahn_2d_smoke.npz"

    generator_module._save_dataset(out, u, x, y, t)

    with np.load(out) as data:
        assert set(data.files) == {"u", "t", "x", "y", "axis_order"}
        assert data["u"].shape == (8, 8, 6)
        assert data["u"].dtype == np.float32
        assert data["x"].dtype == np.float32
        assert data["y"].dtype == np.float32
        assert data["t"].dtype == np.float32
        assert data["axis_order"].tolist() == ["x", "y", "t"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("flag", "value", "attribute", "expected"),
    [
        ("--epsilon", "0.005", "epsilon", 0.005),
        ("--t-max", "2.5", "t_max", 2.5),
        ("--n-snapshots", "12", "n_snapshots", 12),
        ("--dtype", "float64", "dtype", "float64"),
    ],
)
def test_parse_args_accepts_new_generation_flags(
    generator_module: ModuleType,
    tmp_path: Path,
    flag: str,
    value: str,
    attribute: str,
    expected: object,
) -> None:
    args = generator_module.parse_args([
        "--seed",
        "0",
        "--out",
        str(tmp_path / "out.npz"),
        flag,
        value,
    ])

    assert getattr(args, attribute) == expected


@pytest.mark.unit
def test_interface_width_warning_and_metadata(
    generator_module: ModuleType,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    x = np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float32)
    t = np.linspace(0.0, 0.2, 2, endpoint=False, dtype=np.float32)
    u = np.zeros((8, 8, 2), dtype=np.float32)
    u[:4, :,:] = -1.0
    u[4:, :,:] = 1.0

    interface_cells = generator_module._compute_interface_cells(
        u,
        spacing=float(x[1] - x[0]),
    )
    with caplog.at_level(logging.WARNING, logger="generate_allen_cahn_2d"):
        generator_module._warn_if_underresolved(interface_cells)

    assert interface_cells < generator_module.MIN_INTERFACE_CELLS
    assert any("V8 interface-width diagnostic" in r.message for r in caplog.records)

    out = tmp_path / "allen_cahn_2d_paper.npz"
    generator_module._save_dataset(out, u, x, y, t, interface_cells=interface_cells)

    with np.load(out) as data:
        assert "interface_cells" in data.files
        assert float(data["interface_cells"]) == pytest.approx(interface_cells)


@pytest.mark.unit
def test_interface_width_warning_skips_resolved_fields(
    generator_module: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    u = np.zeros((8, 8, 2), dtype=np.float32)
    interface_cells = generator_module.MIN_INTERFACE_CELLS

    with caplog.at_level(logging.WARNING, logger="generate_allen_cahn_2d"):
        generator_module._warn_if_underresolved(interface_cells)

    assert not any("V8 interface-width diagnostic" in r.message for r in caplog.records)
    assert generator_module._compute_interface_cells(u, spacing=1.0 / 8.0) == float(
        "inf"
    )


@pytest.mark.unit
def test_validate_solution_rejects_saturated_final_snapshot(
    generator_module: ModuleType,
) -> None:
    u = _valid_snapshots()
    u[:, :, -1] = 0.25

    with pytest.raises(ValueError, match="final spatial variance"):
        generator_module._validate_solution(u, spacing=1.0 / 8.0)


@pytest.mark.unit
def test_validate_solution_rejects_energy_increase(
    generator_module: ModuleType,
) -> None:
    u = _valid_snapshots()
    u[:, :, -1] = 1.2 * u[:, :, 0]

    with pytest.raises(ValueError, match="free energy"):
        generator_module._validate_solution(u, spacing=1.0 / 8.0)


@pytest.mark.unit
def test_validate_solution_rejects_nonfinite_values(
    generator_module: ModuleType,
) -> None:
    u = _valid_snapshots()
    u[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        generator_module._validate_solution(u, spacing=1.0 / 8.0)
