
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from kd.data.schema import DataTopology, PDEDataset, TaskType
from kd.data.synthetic import (
    generate_advection_data,
    generate_diffusion_data,
    load_chafee_infante,
    load_kdv,
)






_DOMAIN_LO = 0.0
_DOMAIN_HI = 2.0 * math.pi


_T_MIN = 0.0
_T_MAX = 1.0


_ANALYTIC_RTOL = 1e-10
_ANALYTIC_ATOL = 1e-10


_VALIDATION_DIR = Path(__file__).resolve().parents[3] / "validation" / "data"
_CI_DATA_FILES = [
    _VALIDATION_DIR / "chafee_infante_CI.npy",
    _VALIDATION_DIR / "chafee_infante_x.npy",
    _VALIDATION_DIR / "chafee_infante_t.npy",
]
_KDV_DATA_FILE = _VALIDATION_DIR / "KdV_equation.mat"


_CI_EXPECTED_SHAPE = (301, 200)
_KDV_EXPECTED_SHAPE = (256, 201)


_ci_data_available = all(f.exists() for f in _CI_DATA_FILES)
_kdv_data_available = _KDV_DATA_FILE.exists()

skip_no_ci_data = pytest.mark.skipif(
    not _ci_data_available,
    reason="Chafee-Infante data files not found in data/",
)
skip_no_kdv_data = pytest.mark.skipif(
    not _kdv_data_available,
    reason="KdV data file not found in data/",
)


_HYPOTHESIS_SETTINGS = settings(max_examples=30, deadline=5000)







def _advection_analytic_1d(
    x: torch.Tensor,
    t: torch.Tensor,
    speed: float,
    wave: float,
) -> torch.Tensor:
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    return torch.sin(wave * (xg - speed * tg))


def _advection_analytic_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    speeds: tuple[float, float],
    waves: tuple[float, float],
) -> torch.Tensor:
    xg, yg, tg = torch.meshgrid(x, y, t, indexing="ij")
    term_x = torch.sin(waves[0] * (xg - speeds[0] * tg))
    term_y = torch.sin(waves[1] * (yg - speeds[1] * tg))
    return term_x * term_y


def _diffusion_analytic_1d(
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
    wave: float,
) -> torch.Tensor:
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    return torch.exp(-alpha * wave**2 * tg) * torch.sin(wave * xg)


def _diffusion_analytic_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
    waves: tuple[float, float],
) -> torch.Tensor:
    xg, yg, tg = torch.meshgrid(x, y, t, indexing="ij")
    decay = torch.exp(-alpha * (waves[0] ** 2 + waves[1] ** 2) * tg)
    return decay * torch.sin(waves[0] * xg) * torch.sin(waves[1] * yg)







class TestAdvectionSmoke:

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        assert callable(generate_advection_data)

    @pytest.mark.smoke
    def test_returns_pde_dataset_1d(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(16,), nt=11
        )
        assert isinstance(ds, PDEDataset)

    @pytest.mark.smoke
    def test_returns_pde_dataset_2d(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0, 0.5), waves=(1.0, 1.0), grid_sizes=(16, 16), nt=11
        )
        assert isinstance(ds, PDEDataset)







class TestAdvectionShape:

    @pytest.mark.unit
    def test_1d_shape(self) -> None:
        nx, nt = 32, 21
        ds = generate_advection_data(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(nx,), nt=nt
        )
        assert ds.get_shape() == (nx, nt)

    @pytest.mark.unit
    def test_2d_shape(self) -> None:
        nx, ny, nt = 20, 24, 15
        ds = generate_advection_data(
            speeds=(1.0, 0.5),
            waves=(1.0, 2.0),
            grid_sizes=(nx, ny),
            nt=nt,
        )
        assert ds.get_shape() == (nx, ny, nt)

    @pytest.mark.unit
    def test_1d_field_shape(self) -> None:
        nx, nt = 64, 31
        ds = generate_advection_data(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(nx,), nt=nt
        )
        u = ds.get_field("u")
        assert u.shape == (nx, nt)

    @pytest.mark.unit
    def test_2d_field_shape(self) -> None:
        nx, ny, nt = 20, 24, 15
        ds = generate_advection_data(
            speeds=(1.0, 0.5),
            waves=(1.0, 2.0),
            grid_sizes=(nx, ny),
            nt=nt,
        )
        u = ds.get_field("u")
        assert u.shape == (nx, ny, nt)

    @pytest.mark.unit
    def test_1d_coords_range(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(32,), nt=21
        )
        x = ds.get_coords("x")
        t = ds.get_coords("t")
        torch.testing.assert_close(
            x[0], torch.tensor(_DOMAIN_LO, dtype=x.dtype), rtol=1e-6, atol=1e-6
        )

        assert x[-1].item() < _DOMAIN_HI
        assert len(x) == 32
        torch.testing.assert_close(
            t[0], torch.tensor(_T_MIN, dtype=t.dtype), rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            t[-1], torch.tensor(_T_MAX, dtype=t.dtype), rtol=1e-6, atol=1e-6
        )

    @pytest.mark.unit
    def test_2d_coords_range(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0, 0.5),
            waves=(1.0, 2.0),
            grid_sizes=(16, 20),
            nt=11,
        )
        for axis_name in ("x", "y"):
            coord = ds.get_coords(axis_name)
            torch.testing.assert_close(
                coord[0],
                torch.tensor(_DOMAIN_LO, dtype=coord.dtype),
                rtol=1e-6,
                atol=1e-6,
            )

            assert coord[-1].item() < _DOMAIN_HI
        t = ds.get_coords("t")
        torch.testing.assert_close(
            t[-1], torch.tensor(_T_MAX, dtype=t.dtype), rtol=1e-6, atol=1e-6
        )







class TestAdvectionMetadata:

    @pytest.mark.unit
    def test_task_type(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(16,), nt=11
        )
        assert ds.task_type == TaskType.PDE

    @pytest.mark.unit
    def test_topology(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(16,), nt=11
        )
        assert ds.topology == DataTopology.GRID

    @pytest.mark.unit
    def test_lhs_field(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(16,), nt=11
        )
        assert ds.lhs_field == "u"

    @pytest.mark.unit
    def test_lhs_axis(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(16,), nt=11
        )
        assert ds.lhs_axis == "t"

    @pytest.mark.unit
    def test_axis_order_1d(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(16,), nt=11
        )
        assert ds.axis_order == ["x", "t"]

    @pytest.mark.unit
    def test_axis_order_2d(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0, 0.5),
            waves=(1.0, 2.0),
            grid_sizes=(16, 16),
            nt=11,
        )
        assert ds.axis_order == ["x", "y", "t"]

    @pytest.mark.unit
    def test_ground_truth_1d(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(16,), nt=11
        )
        assert ds.ground_truth is not None
        assert "u_t" in ds.ground_truth
        assert "u_x" in ds.ground_truth

    @pytest.mark.unit
    def test_ground_truth_2d(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0, 0.5),
            waves=(1.0, 2.0),
            grid_sizes=(16, 16),
            nt=11,
        )
        assert ds.ground_truth is not None
        assert "u_x" in ds.ground_truth
        assert "u_y" in ds.ground_truth

    @pytest.mark.unit
    def test_noise_level_stored(self) -> None:
        noise = 0.05
        ds = generate_advection_data(
            speeds=(1.0,),
            waves=(1.0,),
            grid_sizes=(16,),
            nt=11,
            noise_level=noise,
        )
        assert ds.noise_level == noise

    @pytest.mark.unit
    def test_name_contains_advection(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(16,), nt=11
        )
        assert "advection" in ds.name.lower()

    @pytest.mark.unit
    def test_spatial_axes_periodic(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0, 0.5),
            waves=(1.0, 2.0),
            grid_sizes=(16, 16),
            nt=11,
        )
        assert ds.axes is not None
        assert ds.axes["x"].is_periodic is True
        assert ds.axes["y"].is_periodic is True
        assert ds.axes["t"].is_periodic is False







class TestAdvectionPhysics:

    @pytest.mark.unit
    def test_1d_analytic_exact_no_noise(self) -> None:
        speed, wave = 1.5, 2.0
        nx, nt = 64, 31
        ds = generate_advection_data(
            speeds=(speed,),
            waves=(wave,),
            grid_sizes=(nx,),
            nt=nt,
            noise_level=0.0,
        )
        x = ds.get_coords("x")
        t = ds.get_coords("t")
        u_actual = ds.get_field("u")
        u_expected = _advection_analytic_1d(x, t, speed, wave)
        torch.testing.assert_close(
            u_actual, u_expected, rtol=_ANALYTIC_RTOL, atol=_ANALYTIC_ATOL
        )

    @pytest.mark.unit
    def test_2d_analytic_exact_no_noise(self) -> None:
        speeds = (1.0, 0.5)
        waves = (2.0, 1.0)
        nx, ny, nt = 20, 24, 15
        ds = generate_advection_data(
            speeds=speeds,
            waves=waves,
            grid_sizes=(nx, ny),
            nt=nt,
            noise_level=0.0,
        )
        x = ds.get_coords("x")
        y = ds.get_coords("y")
        t = ds.get_coords("t")
        u_actual = ds.get_field("u")
        u_expected = _advection_analytic_2d(x, y, t, speeds, waves)
        torch.testing.assert_close(
            u_actual, u_expected, rtol=_ANALYTIC_RTOL, atol=_ANALYTIC_ATOL
        )

    @pytest.mark.unit
    def test_1d_initial_condition(self) -> None:
        wave = 3.0
        ds = generate_advection_data(
            speeds=(1.0,),
            waves=(wave,),
            grid_sizes=(64,),
            nt=21,
            noise_level=0.0,
        )
        x = ds.get_coords("x")
        u = ds.get_field("u")
        u_t0 = u[:, 0]
        expected = torch.sin(wave * x)
        torch.testing.assert_close(u_t0, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.unit
    def test_data_is_finite(self) -> None:
        ds = generate_advection_data(
            speeds=(2.0,), waves=(1.0,), grid_sizes=(64,), nt=51
        )
        u = ds.get_field("u")
        assert torch.isfinite(u).all(), "Advection data contains NaN or Inf"

    @pytest.mark.unit
    def test_data_bounded_by_one(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0,),
            waves=(1.0,),
            grid_sizes=(64,),
            nt=51,
            noise_level=0.0,
        )
        u = ds.get_field("u")
        assert u.abs().max() <= 1.0 + 1e-10, "Advection solution exceeds [-1, 1]"

    @pytest.mark.unit
    def test_2d_data_bounded(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0, 0.5),
            waves=(1.0, 1.0),
            grid_sizes=(16, 16),
            nt=11,
            noise_level=0.0,
        )
        u = ds.get_field("u")
        assert u.abs().max() <= 1.0 + 1e-10

    @pytest.mark.unit
    def test_advection_speed_effect(self) -> None:
        wave = 1.0
        ds_slow = generate_advection_data(
            speeds=(0.5,),
            waves=(wave,),
            grid_sizes=(128,),
            nt=21,
            noise_level=0.0,
        )
        ds_fast = generate_advection_data(
            speeds=(2.0,),
            waves=(wave,),
            grid_sizes=(128,),
            nt=21,
            noise_level=0.0,
        )

        u_slow_final = ds_slow.get_field("u")[:, -1]
        u_fast_final = ds_fast.get_field("u")[:, -1]
        assert not torch.allclose(u_slow_final, u_fast_final, atol=1e-3)







class TestAdvectionNoise:

    @pytest.mark.unit
    def test_zero_noise_deterministic(self) -> None:
        kwargs = dict(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(32,), nt=21, noise_level=0.0
        )
        ds1 = generate_advection_data(**kwargs, seed=42)
        ds2 = generate_advection_data(**kwargs, seed=42)
        torch.testing.assert_close(ds1.get_field("u"), ds2.get_field("u"))

    @pytest.mark.unit
    def test_noise_adds_variation(self) -> None:
        kwargs = dict(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(32,), nt=21, noise_level=0.1
        )
        ds1 = generate_advection_data(**kwargs, seed=42)
        ds2 = generate_advection_data(**kwargs, seed=123)
        assert not torch.allclose(ds1.get_field("u"), ds2.get_field("u"))

    @pytest.mark.unit
    def test_seed_reproducibility(self) -> None:
        kwargs = dict(
            speeds=(1.0,),
            waves=(1.0,),
            grid_sizes=(32,),
            nt=21,
            noise_level=0.05,
            seed=999,
        )
        ds1 = generate_advection_data(**kwargs)
        ds2 = generate_advection_data(**kwargs)
        torch.testing.assert_close(ds1.get_field("u"), ds2.get_field("u"))

    @pytest.mark.unit
    def test_noise_magnitude(self) -> None:
        noise_level = 0.1
        ds_clean = generate_advection_data(
            speeds=(1.0,),
            waves=(1.0,),
            grid_sizes=(128,),
            nt=51,
            noise_level=0.0,
            seed=42,
        )
        ds_noisy = generate_advection_data(
            speeds=(1.0,),
            waves=(1.0,),
            grid_sizes=(128,),
            nt=51,
            noise_level=noise_level,
            seed=42,
        )
        noise = ds_noisy.get_field("u") - ds_clean.get_field("u")
        actual_std = noise.std().item()
        assert actual_std == pytest.approx(noise_level, rel=0.3)







class TestAdvectionEdgeCases:

    @pytest.mark.unit
    def test_minimal_grid_1d(self) -> None:
        ds = generate_advection_data(speeds=(1.0,), waves=(1.0,), grid_sizes=(4,), nt=3)
        assert ds.get_shape() == (4, 3)
        assert torch.isfinite(ds.get_field("u")).all()

    @pytest.mark.unit
    def test_minimal_grid_2d(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0, 0.5),
            waves=(1.0, 1.0),
            grid_sizes=(4, 4),
            nt=3,
        )
        assert ds.get_shape() == (4, 4, 3)
        assert torch.isfinite(ds.get_field("u")).all()

    @pytest.mark.unit
    def test_zero_speed(self) -> None:
        ds = generate_advection_data(
            speeds=(0.0,),
            waves=(1.0,),
            grid_sizes=(32,),
            nt=11,
            noise_level=0.0,
        )
        u = ds.get_field("u")

        torch.testing.assert_close(
            u[:, 0], u[:, -1], rtol=_ANALYTIC_RTOL, atol=_ANALYTIC_ATOL
        )

    @pytest.mark.unit
    def test_high_wave_number(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0,),
            waves=(10.0,),
            grid_sizes=(128,),
            nt=21,
            noise_level=0.0,
        )
        u = ds.get_field("u")
        assert torch.isfinite(u).all()
        assert u.abs().max() <= 1.0 + 1e-10







class TestAdvectionValidation:

    @pytest.mark.unit
    def test_mismatched_speeds_waves_raises(self) -> None:
        with pytest.raises(ValueError, match="speeds.*waves"):
            generate_advection_data(
                speeds=(1.0, 0.5), waves=(1.0,), grid_sizes=(16,), nt=11
            )

    @pytest.mark.unit
    def test_mismatched_speeds_grid_sizes_raises(self) -> None:
        with pytest.raises(ValueError, match="speeds.*grid_sizes"):
            generate_advection_data(
                speeds=(1.0, 0.5), waves=(1.0, 2.0), grid_sizes=(16,), nt=11
            )

    @pytest.mark.unit
    def test_empty_speeds_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_advection_data(speeds=(), waves=(), grid_sizes=(), nt=11)

    @pytest.mark.unit
    def test_negative_noise_raises(self) -> None:
        with pytest.raises(ValueError, match="noise_level"):
            generate_advection_data(
                speeds=(1.0,),
                waves=(1.0,),
                grid_sizes=(16,),
                nt=11,
                noise_level=-0.1,
            )

    @pytest.mark.unit
    def test_nt_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="nt"):
            generate_advection_data(speeds=(1.0,), waves=(1.0,), grid_sizes=(16,), nt=0)

    @pytest.mark.unit
    def test_grid_size_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_advection_data(speeds=(1.0,), waves=(1.0,), grid_sizes=(0,), nt=11)

    @pytest.mark.unit
    def test_negative_grid_size_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_advection_data(
                speeds=(1.0,), waves=(1.0,), grid_sizes=(-5,), nt=11
            )







class TestAdvectionProperties:

    @given(
        speed=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        wave=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
    @_HYPOTHESIS_SETTINGS
    @pytest.mark.unit
    def test_1d_output_always_finite(self, speed: float, wave: float) -> None:
        ds = generate_advection_data(
            speeds=(speed,),
            waves=(wave,),
            grid_sizes=(16,),
            nt=5,
            noise_level=0.0,
        )
        u = ds.get_field("u")
        assert torch.isfinite(u).all()

    @given(
        speed=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        wave=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
    @_HYPOTHESIS_SETTINGS
    @pytest.mark.unit
    def test_1d_bounded_by_one(self, speed: float, wave: float) -> None:
        ds = generate_advection_data(
            speeds=(speed,),
            waves=(wave,),
            grid_sizes=(16,),
            nt=5,
            noise_level=0.0,
        )
        u = ds.get_field("u")
        assert u.abs().max() <= 1.0 + 1e-10







class TestDiffusionSmoke:

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        assert callable(generate_diffusion_data)

    @pytest.mark.smoke
    def test_returns_pde_dataset_1d(self) -> None:
        ds = generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(16,), nt=11)
        assert isinstance(ds, PDEDataset)

    @pytest.mark.smoke
    def test_returns_pde_dataset_2d(self) -> None:
        ds = generate_diffusion_data(
            alpha=0.1, waves=(1.0, 1.0), grid_sizes=(16, 16), nt=11
        )
        assert isinstance(ds, PDEDataset)







class TestDiffusionShape:

    @pytest.mark.unit
    def test_1d_shape(self) -> None:
        nx, nt = 32, 21
        ds = generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(nx,), nt=nt)
        assert ds.get_shape() == (nx, nt)

    @pytest.mark.unit
    def test_2d_shape(self) -> None:
        nx, ny, nt = 20, 24, 15
        ds = generate_diffusion_data(
            alpha=0.1, waves=(1.0, 2.0), grid_sizes=(nx, ny), nt=nt
        )
        assert ds.get_shape() == (nx, ny, nt)

    @pytest.mark.unit
    def test_1d_field_shape(self) -> None:
        nx, nt = 64, 31
        ds = generate_diffusion_data(alpha=0.1, waves=(2.0,), grid_sizes=(nx,), nt=nt)
        u = ds.get_field("u")
        assert u.shape == (nx, nt)

    @pytest.mark.unit
    def test_1d_coords_range(self) -> None:
        ds = generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(32,), nt=21)
        x = ds.get_coords("x")
        t = ds.get_coords("t")
        torch.testing.assert_close(
            x[0], torch.tensor(_DOMAIN_LO, dtype=x.dtype), rtol=1e-6, atol=1e-6
        )

        assert x[-1].item() < _DOMAIN_HI
        assert len(x) == 32
        torch.testing.assert_close(
            t[-1], torch.tensor(_T_MAX, dtype=t.dtype), rtol=1e-6, atol=1e-6
        )







class TestDiffusionMetadata:

    @pytest.mark.unit
    def test_task_type(self) -> None:
        ds = generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(16,), nt=11)
        assert ds.task_type == TaskType.PDE

    @pytest.mark.unit
    def test_topology(self) -> None:
        ds = generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(16,), nt=11)
        assert ds.topology == DataTopology.GRID

    @pytest.mark.unit
    def test_lhs_field_and_axis(self) -> None:
        ds = generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(16,), nt=11)
        assert ds.lhs_field == "u"
        assert ds.lhs_axis == "t"

    @pytest.mark.unit
    def test_axis_order_1d(self) -> None:
        ds = generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(16,), nt=11)
        assert ds.axis_order == ["x", "t"]

    @pytest.mark.unit
    def test_axis_order_2d(self) -> None:
        ds = generate_diffusion_data(
            alpha=0.1, waves=(1.0, 2.0), grid_sizes=(16, 16), nt=11
        )
        assert ds.axis_order == ["x", "y", "t"]

    @pytest.mark.unit
    def test_ground_truth_contains_laplacian_terms(self) -> None:
        ds = generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(16,), nt=11)
        assert ds.ground_truth is not None
        assert "u_t" in ds.ground_truth

        assert "u_xx" in ds.ground_truth or "Laplacian" in ds.ground_truth

    @pytest.mark.unit
    def test_name_contains_diffusion(self) -> None:
        ds = generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(16,), nt=11)
        assert "diffusion" in ds.name.lower()

    @pytest.mark.unit
    def test_spatial_axes_periodic(self) -> None:
        ds = generate_diffusion_data(
            alpha=0.1, waves=(1.0, 2.0), grid_sizes=(16, 16), nt=11
        )
        assert ds.axes is not None
        assert ds.axes["x"].is_periodic is True
        assert ds.axes["y"].is_periodic is True







class TestDiffusionPhysics:

    @pytest.mark.unit
    def test_1d_analytic_exact_no_noise(self) -> None:
        alpha, wave = 0.1, 2.0
        nx, nt = 64, 31
        ds = generate_diffusion_data(
            alpha=alpha,
            waves=(wave,),
            grid_sizes=(nx,),
            nt=nt,
            noise_level=0.0,
        )
        x = ds.get_coords("x")
        t = ds.get_coords("t")
        u_actual = ds.get_field("u")
        u_expected = _diffusion_analytic_1d(x, t, alpha, wave)
        torch.testing.assert_close(
            u_actual, u_expected, rtol=_ANALYTIC_RTOL, atol=_ANALYTIC_ATOL
        )

    @pytest.mark.unit
    def test_2d_analytic_exact_no_noise(self) -> None:
        alpha = 0.05
        waves = (1.0, 2.0)
        nx, ny, nt = 20, 24, 15
        ds = generate_diffusion_data(
            alpha=alpha,
            waves=waves,
            grid_sizes=(nx, ny),
            nt=nt,
            noise_level=0.0,
        )
        x = ds.get_coords("x")
        y = ds.get_coords("y")
        t = ds.get_coords("t")
        u_actual = ds.get_field("u")
        u_expected = _diffusion_analytic_2d(x, y, t, alpha, waves)
        torch.testing.assert_close(
            u_actual, u_expected, rtol=_ANALYTIC_RTOL, atol=_ANALYTIC_ATOL
        )

    @pytest.mark.unit
    def test_1d_initial_condition(self) -> None:
        wave = 3.0
        ds = generate_diffusion_data(
            alpha=0.1,
            waves=(wave,),
            grid_sizes=(64,),
            nt=21,
            noise_level=0.0,
        )
        x = ds.get_coords("x")
        u = ds.get_field("u")
        u_t0 = u[:, 0]
        expected = torch.sin(wave * x)
        torch.testing.assert_close(u_t0, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.unit
    def test_data_is_finite(self) -> None:
        ds = generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(64,), nt=51)
        assert torch.isfinite(ds.get_field("u")).all()

    @pytest.mark.unit
    def test_solution_decays_over_time(self) -> None:
        ds = generate_diffusion_data(
            alpha=0.5,
            waves=(1.0,),
            grid_sizes=(64,),
            nt=51,
            noise_level=0.0,
        )
        u = ds.get_field("u")
        energy_t0 = (u[:, 0] ** 2).sum()
        energy_final = (u[:, -1] ** 2).sum()
        assert energy_final < energy_t0, "Diffusion should decay the signal"

    @pytest.mark.unit
    def test_higher_alpha_decays_faster(self) -> None:
        kwargs = dict(waves=(1.0,), grid_sizes=(64,), nt=51, noise_level=0.0)
        ds_slow = generate_diffusion_data(alpha=0.01, **kwargs)
        ds_fast = generate_diffusion_data(alpha=1.0, **kwargs)
        energy_slow = (ds_slow.get_field("u")[:, -1] ** 2).sum()
        energy_fast = (ds_fast.get_field("u")[:, -1] ** 2).sum()
        assert energy_fast < energy_slow

    @pytest.mark.unit
    def test_higher_wave_decays_faster(self) -> None:
        kwargs = dict(alpha=0.1, grid_sizes=(64,), nt=51, noise_level=0.0)
        ds_low_k = generate_diffusion_data(waves=(1.0,), **kwargs)
        ds_high_k = generate_diffusion_data(waves=(3.0,), **kwargs)
        energy_low_k = (ds_low_k.get_field("u")[:, -1] ** 2).sum()
        energy_high_k = (ds_high_k.get_field("u")[:, -1] ** 2).sum()
        assert energy_high_k < energy_low_k

    @pytest.mark.unit
    def test_data_bounded_by_initial(self) -> None:
        ds = generate_diffusion_data(
            alpha=0.1,
            waves=(1.0,),
            grid_sizes=(64,),
            nt=51,
            noise_level=0.0,
        )
        u = ds.get_field("u")
        max_initial = u[:, 0].abs().max()
        max_all = u.abs().max()

        assert max_all <= max_initial + 1e-10







class TestDiffusionNoise:

    @pytest.mark.unit
    def test_zero_noise_deterministic(self) -> None:
        kwargs = dict(alpha=0.1, waves=(1.0,), grid_sizes=(32,), nt=21, noise_level=0.0)
        ds1 = generate_diffusion_data(**kwargs, seed=42)
        ds2 = generate_diffusion_data(**kwargs, seed=42)
        torch.testing.assert_close(ds1.get_field("u"), ds2.get_field("u"))

    @pytest.mark.unit
    def test_seed_reproducibility(self) -> None:
        kwargs = dict(
            alpha=0.1, waves=(1.0,), grid_sizes=(32,), nt=21, noise_level=0.05, seed=999
        )
        ds1 = generate_diffusion_data(**kwargs)
        ds2 = generate_diffusion_data(**kwargs)
        torch.testing.assert_close(ds1.get_field("u"), ds2.get_field("u"))

    @pytest.mark.unit
    def test_noise_adds_variation(self) -> None:
        kwargs = dict(alpha=0.1, waves=(1.0,), grid_sizes=(32,), nt=21, noise_level=0.1)
        ds1 = generate_diffusion_data(**kwargs, seed=42)
        ds2 = generate_diffusion_data(**kwargs, seed=123)
        assert not torch.allclose(ds1.get_field("u"), ds2.get_field("u"))







class TestDiffusionEdgeCases:

    @pytest.mark.unit
    def test_minimal_grid(self) -> None:
        ds = generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(4,), nt=3)
        assert ds.get_shape() == (4, 3)

    @pytest.mark.unit
    def test_large_alpha(self) -> None:
        ds = generate_diffusion_data(
            alpha=10.0,
            waves=(1.0,),
            grid_sizes=(32,),
            nt=21,
            noise_level=0.0,
        )
        u = ds.get_field("u")
        assert torch.isfinite(u).all()

        energy_final = (u[:, -1] ** 2).sum()
        assert energy_final < 1e-5

    @pytest.mark.unit
    def test_very_small_alpha(self) -> None:
        ds = generate_diffusion_data(
            alpha=1e-6,
            waves=(1.0,),
            grid_sizes=(32,),
            nt=21,
            noise_level=0.0,
        )
        u = ds.get_field("u")

        torch.testing.assert_close(u[:, 0], u[:, -1], rtol=1e-4, atol=1e-4)







class TestDiffusionValidation:

    @pytest.mark.unit
    def test_negative_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            generate_diffusion_data(alpha=-0.1, waves=(1.0,), grid_sizes=(16,), nt=11)

    @pytest.mark.unit
    def test_zero_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            generate_diffusion_data(alpha=0.0, waves=(1.0,), grid_sizes=(16,), nt=11)

    @pytest.mark.unit
    def test_mismatched_waves_grid_sizes_raises(self) -> None:
        with pytest.raises(ValueError, match="waves.*grid_sizes"):
            generate_diffusion_data(
                alpha=0.1, waves=(1.0, 2.0), grid_sizes=(16,), nt=11
            )

    @pytest.mark.unit
    def test_empty_waves_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_diffusion_data(alpha=0.1, waves=(), grid_sizes=(), nt=11)

    @pytest.mark.unit
    def test_negative_noise_raises(self) -> None:
        with pytest.raises(ValueError, match="noise_level"):
            generate_diffusion_data(
                alpha=0.1,
                waves=(1.0,),
                grid_sizes=(16,),
                nt=11,
                noise_level=-0.01,
            )

    @pytest.mark.unit
    def test_nt_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="nt"):
            generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(16,), nt=0)







class TestDiffusionProperties:

    @given(
        alpha=st.floats(min_value=0.01, max_value=5.0, allow_nan=False),
        wave=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
    @_HYPOTHESIS_SETTINGS
    @pytest.mark.unit
    def test_1d_output_always_finite(self, alpha: float, wave: float) -> None:
        ds = generate_diffusion_data(
            alpha=alpha,
            waves=(wave,),
            grid_sizes=(16,),
            nt=5,
            noise_level=0.0,
        )
        assert torch.isfinite(ds.get_field("u")).all()

    @given(
        alpha=st.floats(min_value=0.01, max_value=5.0, allow_nan=False),
        wave=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
    @_HYPOTHESIS_SETTINGS
    @pytest.mark.unit
    def test_1d_monotone_energy_decay(self, alpha: float, wave: float) -> None:
        ds = generate_diffusion_data(
            alpha=alpha,
            waves=(wave,),
            grid_sizes=(32,),
            nt=11,
            noise_level=0.0,
        )
        u = ds.get_field("u")
        for ti in range(1, u.shape[-1]):
            e_prev = (u[..., ti - 1] ** 2).sum()
            e_curr = (u[..., ti] ** 2).sum()
            assert e_curr <= e_prev + 1e-10, (
                f"Energy increased at t-step {ti}: {e_prev} -> {e_curr}"
            )







class TestChafeeInfanteSmoke:

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        assert callable(load_chafee_infante)

    @skip_no_ci_data
    @pytest.mark.smoke
    def test_returns_pde_dataset(self) -> None:
        ds = load_chafee_infante()
        assert isinstance(ds, PDEDataset)


class TestChafeeInfanteSkip:

    @pytest.mark.unit
    def test_missing_data_raises_file_not_found(self) -> None:

        with pytest.raises(FileNotFoundError):
            load_chafee_infante(data_dir=Path("/nonexistent/path"))


@skip_no_ci_data
class TestChafeeInfanteData:

    @pytest.mark.unit
    def test_shape(self) -> None:
        ds = load_chafee_infante()
        assert ds.get_shape() == _CI_EXPECTED_SHAPE

    @pytest.mark.unit
    def test_metadata(self) -> None:
        ds = load_chafee_infante()
        assert ds.task_type == TaskType.PDE
        assert ds.topology == DataTopology.GRID
        assert ds.lhs_field == "u"
        assert ds.lhs_axis == "t"
        assert ds.axis_order == ["x", "t"]

    @pytest.mark.unit
    def test_ground_truth(self) -> None:
        ds = load_chafee_infante()
        assert ds.ground_truth is not None

        assert "u_xx" in ds.ground_truth

    @pytest.mark.unit
    def test_name_contains_chafee(self) -> None:
        ds = load_chafee_infante()
        assert "chafee" in ds.name.lower()

    @pytest.mark.numerical
    def test_data_is_finite(self) -> None:
        ds = load_chafee_infante()
        u = ds.get_field("u")
        assert torch.isfinite(u).all()

    @pytest.mark.numerical
    def test_data_bounded(self) -> None:
        ds = load_chafee_infante()
        u = ds.get_field("u")
        assert u.abs().max() < 100.0

    @pytest.mark.unit
    def test_coords_monotonic(self) -> None:
        ds = load_chafee_infante()
        x = ds.get_coords("x")
        t = ds.get_coords("t")
        assert (x[1:] > x[:-1]).all(), "x not strictly increasing"
        assert (t[1:] > t[:-1]).all(), "t not strictly increasing"







class TestKdVSmoke:

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        assert callable(load_kdv)

    @skip_no_kdv_data
    @pytest.mark.smoke
    def test_returns_pde_dataset(self) -> None:
        ds = load_kdv()
        assert isinstance(ds, PDEDataset)


class TestKdVSkip:

    @pytest.mark.unit
    def test_missing_data_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_kdv(data_dir=Path("/nonexistent/path"))


@skip_no_kdv_data
class TestKdVData:

    @pytest.mark.unit
    def test_shape(self) -> None:
        ds = load_kdv()
        assert ds.get_shape() == _KDV_EXPECTED_SHAPE

    @pytest.mark.unit
    def test_metadata(self) -> None:
        ds = load_kdv()
        assert ds.task_type == TaskType.PDE
        assert ds.topology == DataTopology.GRID
        assert ds.lhs_field == "u"
        assert ds.lhs_axis == "t"
        assert ds.axis_order == ["x", "t"]

    @pytest.mark.unit
    def test_ground_truth(self) -> None:
        ds = load_kdv()
        assert ds.ground_truth is not None

        assert "u_x" in ds.ground_truth
        assert "u_xxx" in ds.ground_truth

    @pytest.mark.unit
    def test_name_contains_kdv(self) -> None:
        ds = load_kdv()
        assert "kdv" in ds.name.lower()

    @pytest.mark.numerical
    def test_data_is_finite(self) -> None:
        ds = load_kdv()
        u = ds.get_field("u")
        assert torch.isfinite(u).all()

    @pytest.mark.numerical
    def test_data_bounded(self) -> None:
        ds = load_kdv()
        u = ds.get_field("u")
        assert u.abs().max() < 1000.0

    @pytest.mark.unit
    def test_coords_monotonic(self) -> None:
        ds = load_kdv()
        x = ds.get_coords("x")
        t = ds.get_coords("t")
        assert (x[1:] > x[:-1]).all(), "x not strictly increasing"
        assert (t[1:] > t[:-1]).all(), "t not strictly increasing"







class TestGeneratorsDevice:

    @pytest.mark.numerical
    def test_advection_default_cpu(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0,), waves=(1.0,), grid_sizes=(16,), nt=11
        )
        assert ds.get_field("u").device == torch.device("cpu")

    @pytest.mark.numerical
    def test_diffusion_default_cpu(self) -> None:
        ds = generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(16,), nt=11)
        assert ds.get_field("u").device == torch.device("cpu")

    @pytest.mark.numerical
    def test_advection_on_specified_device(self, device: torch.device) -> None:
        ds = generate_advection_data(
            speeds=(1.0,),
            waves=(1.0,),
            grid_sizes=(16,),
            nt=11,
            device=device,
        )
        u = ds.get_field("u")
        assert u.device.type == device.type

        assert ds.get_coords("x").device.type == device.type
        assert ds.get_coords("t").device.type == device.type

    @pytest.mark.numerical
    def test_diffusion_on_specified_device(self, device: torch.device) -> None:
        ds = generate_diffusion_data(
            alpha=0.1,
            waves=(1.0,),
            grid_sizes=(16,),
            nt=11,
            device=device,
        )
        u = ds.get_field("u")
        assert u.device.type == device.type

    @pytest.mark.numerical
    def test_advection_dtype_float64_on_cpu(self) -> None:
        ds = generate_advection_data(
            speeds=(1.0,),
            waves=(1.0,),
            grid_sizes=(16,),
            nt=11,
            device=torch.device("cpu"),
        )
        assert ds.get_field("u").dtype == torch.float64

    @pytest.mark.numerical
    def test_diffusion_dtype_float64_on_cpu(self) -> None:
        ds = generate_diffusion_data(
            alpha=0.1,
            waves=(1.0,),
            grid_sizes=(16,),
            nt=11,
            device=torch.device("cpu"),
        )
        assert ds.get_field("u").dtype == torch.float64







class TestAdvectionNegative:

    @pytest.mark.numerical
    def test_nan_speed_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            generate_advection_data(
                speeds=(float("nan"),), waves=(1.0,), grid_sizes=(16,), nt=11
            )

    @pytest.mark.numerical
    def test_inf_speed_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            generate_advection_data(
                speeds=(float("inf"),), waves=(1.0,), grid_sizes=(16,), nt=11
            )

    @pytest.mark.numerical
    def test_nan_wave_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            generate_advection_data(
                speeds=(1.0,), waves=(float("nan"),), grid_sizes=(16,), nt=11
            )

    @pytest.mark.numerical
    def test_inf_wave_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            generate_advection_data(
                speeds=(1.0,), waves=(float("inf"),), grid_sizes=(16,), nt=11
            )

    @pytest.mark.numerical
    def test_nan_noise_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            generate_advection_data(
                speeds=(1.0,),
                waves=(1.0,),
                grid_sizes=(16,),
                nt=11,
                noise_level=float("nan"),
            )

    @pytest.mark.unit
    def test_nt_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="nt"):
            generate_advection_data(
                speeds=(1.0,), waves=(1.0,), grid_sizes=(16,), nt=-5
            )

    @pytest.mark.unit
    def test_3d_not_supported(self) -> None:


        try:
            ds = generate_advection_data(
                speeds=(1.0, 0.5, 0.3),
                waves=(1.0, 1.0, 1.0),
                grid_sizes=(8, 8, 8),
                nt=5,
            )

            assert isinstance(ds, PDEDataset)
            assert torch.isfinite(ds.get_field("u")).all()
        except (ValueError, NotImplementedError):
            pass

    @pytest.mark.unit
    def test_excessive_noise_produces_finite_output(self) -> None:
        try:
            ds = generate_advection_data(
                speeds=(1.0,),
                waves=(1.0,),
                grid_sizes=(16,),
                nt=5,
                noise_level=100.0,
                seed=42,
            )
            u = ds.get_field("u")
            assert torch.isfinite(u).all(), "Large noise produced NaN/Inf"
        except ValueError:
            pass


class TestDiffusionNegative:

    @pytest.mark.numerical
    def test_nan_alpha_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            generate_diffusion_data(
                alpha=float("nan"), waves=(1.0,), grid_sizes=(16,), nt=11
            )

    @pytest.mark.numerical
    def test_inf_alpha_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            generate_diffusion_data(
                alpha=float("inf"), waves=(1.0,), grid_sizes=(16,), nt=11
            )

    @pytest.mark.numerical
    def test_nan_wave_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            generate_diffusion_data(
                alpha=0.1, waves=(float("nan"),), grid_sizes=(16,), nt=11
            )

    @pytest.mark.numerical
    def test_inf_wave_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            generate_diffusion_data(
                alpha=0.1, waves=(float("inf"),), grid_sizes=(16,), nt=11
            )

    @pytest.mark.numerical
    def test_nan_noise_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            generate_diffusion_data(
                alpha=0.1,
                waves=(1.0,),
                grid_sizes=(16,),
                nt=11,
                noise_level=float("nan"),
            )

    @pytest.mark.unit
    def test_nt_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="nt"):
            generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(16,), nt=-3)

    @pytest.mark.unit
    def test_grid_size_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(0,), nt=11)

    @pytest.mark.unit
    def test_negative_grid_size_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_diffusion_data(alpha=0.1, waves=(1.0,), grid_sizes=(-5,), nt=11)


class TestLoaderNegative:

    @pytest.mark.unit
    def test_ci_wrong_dir_type_raises(self) -> None:
        with pytest.raises((TypeError, AttributeError)):
            load_chafee_infante(data_dir=12345)

    @pytest.mark.unit
    def test_kdv_wrong_dir_type_raises(self) -> None:
        with pytest.raises((TypeError, AttributeError)):
            load_kdv(data_dir=12345)

    @pytest.mark.unit
    def test_ci_empty_dir_raises(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(FileNotFoundError):
            load_chafee_infante(data_dir=Path(tmpdir))

    @pytest.mark.unit
    def test_kdv_empty_dir_raises(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(FileNotFoundError):
            load_kdv(data_dir=Path(tmpdir))
