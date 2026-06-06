
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest





import kd.search.discover.tokens.prior

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "discover" / "generate_allen_cahn_2d_spectral.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_generate_allen_cahn_2d_spectral",
        _SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen() -> ModuleType:
    return _load_script_module()







@pytest.mark.unit
def test_paper_ic_default_matches_sin4pix_cos4piy(gen: ModuleType) -> None:
    x = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=np.float64)
    u = gen._initial_condition(x, y)
    expected = np.outer(
        np.sin(4.0 * np.pi * x), np.cos(4.0 * np.pi * y)
    ).astype(np.float64)
    np.testing.assert_allclose(u, expected, atol=1e-15)


@pytest.mark.unit
def test_paper_ic_xy_derivatives_are_numerically_equivalent(
    gen: ModuleType,
) -> None:
    x = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=np.float64)
    u = gen._initial_condition(x, y, mode="paper")
    spacing = float(x[1] - x[0])

    u_xx = (
        np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)
    ) / spacing**2
    u_yy = (
        np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)
    ) / spacing**2
    rel_diff = abs(np.abs(u_xx).max() - np.abs(u_yy).max()) / np.abs(
        u_xx
    ).max()



    assert rel_diff < 1e-10, (
        f"Paper IC xx/yy derivatives differ by {rel_diff:.3e} — "
        f"degeneracy assumption broken; check IC formula"
    )







@pytest.mark.unit
def test_asymmetric_ic_breaks_xy_symmetry(gen: ModuleType) -> None:
    x = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=np.float64)
    u = gen._initial_condition(x, y, mode="asymmetric")
    spacing = float(x[1] - x[0])
    u_xx = (
        np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)
    ) / spacing**2
    u_yy = (
        np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)
    ) / spacing**2
    max_xx = float(np.abs(u_xx).max())
    max_yy = float(np.abs(u_yy).max())
    rel_diff = abs(max_xx - max_yy) / max(max_xx, max_yy)
    assert rel_diff > 0.05, (
        f"Asymmetric IC xx/yy still nearly equal "
        f"(rel_diff={rel_diff:.3e}); IC formula not asymmetric enough"
    )


@pytest.mark.unit
def test_asymmetric_ic_differs_from_paper(gen: ModuleType) -> None:
    x = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=np.float64)
    u_paper = gen._initial_condition(x, y, mode="paper")
    u_asym = gen._initial_condition(x, y, mode="asymmetric")
    assert not np.allclose(u_paper, u_asym, atol=1e-3)


@pytest.mark.unit
def test_asymmetric_ic_in_range(gen: ModuleType) -> None:
    x = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=np.float64)
    u = gen._initial_condition(x, y, mode="asymmetric")

    assert float(np.abs(u).max()) <= 1.31


@pytest.mark.unit
def test_unknown_ic_mode_raises(gen: ModuleType) -> None:
    x = np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float64)
    with pytest.raises(ValueError, match="ic"):
        gen._initial_condition(x, y, mode="not-a-mode")


@pytest.mark.unit
def test_parse_args_accepts_ic_mode(gen: ModuleType, tmp_path: Path) -> None:
    args = gen.parse_args(
        [
            "--seed",
            "0",
            "--out",
            str(tmp_path / "out.npz"),
            "--ic-mode",
            "asymmetric",
        ]
    )
    assert args.ic_mode == "asymmetric"


@pytest.mark.unit
def test_parse_args_default_ic_mode_is_paper(
    gen: ModuleType, tmp_path: Path
) -> None:
    args = gen.parse_args(
        ["--seed", "0", "--out", str(tmp_path / "out.npz")]
    )
    assert args.ic_mode == "paper"


@pytest.mark.unit
def test_parse_args_rejects_unknown_ic_mode(
    gen: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        gen.parse_args(
            [
                "--seed",
                "0",
                "--out",
                str(tmp_path / "out.npz"),
                "--ic-mode",
                "bogus",
            ]
        )







@pytest.mark.unit
def test_fft_laplacian_sanity_works_for_asymmetric_ic(gen: ModuleType) -> None:
    x = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=np.float64)
    u = gen._initial_condition(x, y, mode="asymmetric")
    spacing = float(x[1] - x[0])
    kx = gen._wave_numbers(64, spacing)
    ky = gen._wave_numbers(64, spacing)
    rel_err = gen._sanity_check_fft_laplacian(u, kx, ky, mode="asymmetric")
    assert rel_err < 1e-10
