
import pytest
import torch

from kd.data.schema import DataTopology, PDEDataset, TaskType
from kd.data.synthetic import generate_burgers_data






class TestSyntheticSmoke:

    @pytest.mark.smoke
    def test_generate_burgers_exists(self) -> None:
        assert callable(generate_burgers_data)

    @pytest.mark.smoke
    def test_generate_burgers_returns_pde_dataset(self) -> None:
        dataset = generate_burgers_data(nx=16, nt=11)
        assert isinstance(dataset, PDEDataset)

    @pytest.mark.smoke
    def test_generate_burgers_default_params(self) -> None:
        dataset = generate_burgers_data()
        assert dataset is not None
        assert dataset.name is not None







class TestBurgersShape:

    @pytest.mark.unit
    def test_shape_matches_nx_nt(self) -> None:
        nx, nt = 128, 51
        dataset = generate_burgers_data(nx=nx, nt=nt)
        assert dataset.get_shape() == (nx, nt)

    @pytest.mark.unit
    def test_field_shape_matches(self) -> None:
        nx, nt = 64, 31
        dataset = generate_burgers_data(nx=nx, nt=nt)
        u = dataset.get_field("u")
        assert u.shape == (nx, nt)

    @pytest.mark.unit
    def test_x_coords_length(self) -> None:
        nx = 100
        dataset = generate_burgers_data(nx=nx, nt=21)
        x = dataset.get_coords("x")
        assert x.shape == (nx,)

    @pytest.mark.unit
    def test_t_coords_length(self) -> None:
        nt = 50
        dataset = generate_burgers_data(nx=32, nt=nt)
        t = dataset.get_coords("t")
        assert t.shape == (nt,)

    @pytest.mark.unit
    def test_x_coords_range(self) -> None:
        dataset = generate_burgers_data(nx=64, nt=21)
        x = dataset.get_coords("x")
        assert x.min().item() == pytest.approx(-1.0, abs=1e-6)

        assert x.max().item() < 1.0
        assert len(x) == 64

    @pytest.mark.unit
    def test_t_coords_range(self) -> None:
        dataset = generate_burgers_data(nx=32, nt=51)
        t = dataset.get_coords("t")
        assert t.min().item() == pytest.approx(0.0, abs=1e-6)
        assert t.max().item() == pytest.approx(1.0, abs=1e-6)







class TestBurgersMetadata:

    @pytest.mark.unit
    def test_task_type_is_pde(self) -> None:
        dataset = generate_burgers_data(nx=16, nt=11)
        assert dataset.task_type == TaskType.PDE

    @pytest.mark.unit
    def test_topology_is_grid(self) -> None:
        dataset = generate_burgers_data(nx=16, nt=11)
        assert dataset.topology == DataTopology.GRID

    @pytest.mark.unit
    def test_lhs_field_is_u(self) -> None:
        dataset = generate_burgers_data(nx=16, nt=11)
        assert dataset.lhs_field == "u"

    @pytest.mark.unit
    def test_lhs_axis_is_t(self) -> None:
        dataset = generate_burgers_data(nx=16, nt=11)
        assert dataset.lhs_axis == "t"

    @pytest.mark.unit
    def test_axis_order_is_x_t(self) -> None:
        dataset = generate_burgers_data(nx=16, nt=11)
        assert dataset.axis_order == ["x", "t"]

    @pytest.mark.unit
    def test_ground_truth_string(self) -> None:
        dataset = generate_burgers_data(nx=16, nt=11, nu=0.1)
        assert dataset.ground_truth is not None

        assert "u" in dataset.ground_truth

    @pytest.mark.unit
    def test_noise_level_stored(self) -> None:
        noise = 0.05
        dataset = generate_burgers_data(nx=16, nt=11, noise_level=noise)
        assert dataset.noise_level == noise

    @pytest.mark.unit
    def test_x_axis_is_periodic(self) -> None:
        dataset = generate_burgers_data(nx=32, nt=21)
        assert dataset.axes is not None
        assert dataset.axes["x"].is_periodic is True

    @pytest.mark.unit
    def test_t_axis_is_not_periodic(self) -> None:
        dataset = generate_burgers_data(nx=32, nt=21)
        assert dataset.axes is not None
        assert dataset.axes["t"].is_periodic is False







class TestBurgersPhysics:

    @pytest.mark.unit
    def test_initial_condition_shape(self) -> None:
        dataset = generate_burgers_data(nx=64, nt=51, nu=0.1, noise_level=0.0)
        x = dataset.get_coords("x")
        u = dataset.get_field("u")
        u_initial = u[:, 0]
        expected = -torch.sin(torch.pi * x)
        torch.testing.assert_close(u_initial, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.unit
    def test_data_is_finite(self) -> None:
        dataset = generate_burgers_data(nx=128, nt=101, nu=0.1)
        u = dataset.get_field("u")
        assert torch.isfinite(u).all(), "Data contains NaN or Inf"

    @pytest.mark.unit
    def test_data_bounded(self) -> None:
        dataset = generate_burgers_data(nx=128, nt=101, nu=0.1)
        u = dataset.get_field("u")

        assert u.abs().max() < 10.0, "Data appears to have blown up"

    @pytest.mark.unit
    def test_viscosity_effect(self) -> None:

        ds_low_nu = generate_burgers_data(nx=128, nt=51, nu=0.01, noise_level=0.0)
        ds_high_nu = generate_burgers_data(nx=128, nt=51, nu=0.5, noise_level=0.0)


        u_low = ds_low_nu.get_field("u")[:, -1]
        u_high = ds_high_nu.get_field("u")[:, -1]


        tv_low = (u_low[1:] - u_low[:-1]).abs().sum()
        tv_high = (u_high[1:] - u_high[:-1]).abs().sum()

        assert tv_high < tv_low, "Higher viscosity should produce smoother solution"

    @pytest.mark.integration
    def test_burgers_equation_residual(self) -> None:
        nu = 0.1
        nx, nt = 128, 101
        dataset = generate_burgers_data(nx=nx, nt=nt, nu=nu, noise_level=0.0)

        x = dataset.get_coords("x")
        t = dataset.get_coords("t")
        u = dataset.get_field("u")

        dx = (x[1] - x[0]).item()
        dt = (t[1] - t[0]).item()


        i_slice = slice(2, nx - 2)
        t_slice = slice(1, nt - 1)

        u_interior = u[i_slice, t_slice]



        u_t = (u[i_slice, 2:nt] - u[i_slice, 0: nt - 2]) / (2 * dt)


        u_x = (u[3: nx - 1, t_slice] - u[1: nx - 3, t_slice]) / (2 * dx)


        u_xx = (
            u[3: nx - 1, t_slice] - 2 * u[i_slice, t_slice] + u[1: nx - 3, t_slice]
        ) / (dx**2)


        residual = u_t + u_interior * u_x - nu * u_xx


        mean_residual = residual.abs().mean().item()
        assert mean_residual < 0.1, f"Mean residual {mean_residual} too large"







class TestBurgersNoise:

    @pytest.mark.unit
    def test_zero_noise_deterministic(self) -> None:
        ds1 = generate_burgers_data(nx=32, nt=21, noise_level=0.0, seed=42)
        ds2 = generate_burgers_data(nx=32, nt=21, noise_level=0.0, seed=42)
        torch.testing.assert_close(ds1.get_field("u"), ds2.get_field("u"))

    @pytest.mark.unit
    def test_noise_adds_variation(self) -> None:
        ds1 = generate_burgers_data(nx=32, nt=21, noise_level=0.1, seed=42)
        ds2 = generate_burgers_data(nx=32, nt=21, noise_level=0.1, seed=123)

        assert not torch.allclose(ds1.get_field("u"), ds2.get_field("u"))

    @pytest.mark.unit
    def test_noise_magnitude(self) -> None:
        noise_level = 0.1

        ds_clean = generate_burgers_data(nx=128, nt=101, noise_level=0.0, seed=42)
        ds_noisy = generate_burgers_data(
            nx=128, nt=101, noise_level=noise_level, seed=42
        )

        u_clean = ds_clean.get_field("u")
        u_noisy = ds_noisy.get_field("u")
        noise = u_noisy - u_clean


        actual_std = noise.std().item()
        assert actual_std == pytest.approx(noise_level, rel=0.3), (
            f"Noise std {actual_std} differs from expected {noise_level}"
        )

    @pytest.mark.unit
    def test_seed_reproducibility(self) -> None:
        ds1 = generate_burgers_data(nx=32, nt=21, noise_level=0.05, seed=999)
        ds2 = generate_burgers_data(nx=32, nt=21, noise_level=0.05, seed=999)
        torch.testing.assert_close(ds1.get_field("u"), ds2.get_field("u"))







class TestBurgersDevice:

    @pytest.mark.numerical
    def test_default_device_is_cpu(self) -> None:
        dataset = generate_burgers_data(nx=16, nt=11)
        u = dataset.get_field("u")
        assert u.device == torch.device("cpu")

    @pytest.mark.numerical
    def test_explicit_cpu_device(self) -> None:
        device = torch.device("cpu")
        dataset = generate_burgers_data(nx=16, nt=11, device=device)
        u = dataset.get_field("u")
        assert u.device == device

    @pytest.mark.numerical
    def test_field_on_specified_device(self, device: torch.device) -> None:
        dataset = generate_burgers_data(nx=32, nt=21, device=device)
        u = dataset.get_field("u")
        assert u.device.type == device.type

    @pytest.mark.numerical
    def test_coords_on_specified_device(self, device: torch.device) -> None:
        dataset = generate_burgers_data(nx=32, nt=21, device=device)
        x = dataset.get_coords("x")
        t = dataset.get_coords("t")
        assert x.device.type == device.type
        assert t.device.type == device.type

    @pytest.mark.numerical
    def test_all_tensors_same_device(self, device: torch.device) -> None:
        dataset = generate_burgers_data(nx=32, nt=21, device=device)


        assert dataset.axes is not None
        for axis_info in dataset.axes.values():
            assert axis_info.values.device.type == device.type


        assert dataset.fields is not None
        for field_data in dataset.fields.values():
            assert field_data.values.device.type == device.type







class TestBurgersEdgeCases:

    @pytest.mark.unit
    def test_minimum_grid_size(self) -> None:
        dataset = generate_burgers_data(nx=8, nt=5)
        assert dataset.get_shape() == (8, 5)

    @pytest.mark.unit
    def test_large_grid_size(self) -> None:
        dataset = generate_burgers_data(nx=512, nt=201)
        assert dataset.get_shape() == (512, 201)
        assert torch.isfinite(dataset.get_field("u")).all()

    @pytest.mark.unit
    def test_high_viscosity(self) -> None:
        dataset = generate_burgers_data(nx=64, nt=51, nu=1.0)
        u = dataset.get_field("u")
        assert torch.isfinite(u).all()

    @pytest.mark.unit
    def test_low_viscosity(self) -> None:
        dataset = generate_burgers_data(nx=256, nt=101, nu=0.01)
        u = dataset.get_field("u")

        assert torch.isfinite(u).all()

    @pytest.mark.unit
    def test_different_nu_values(self) -> None:
        for nu in [0.001, 0.01, 0.1, 0.5, 1.0]:
            dataset = generate_burgers_data(nx=64, nt=51, nu=nu)
            assert torch.isfinite(dataset.get_field("u")).all()







class TestBurgersParameterValidation:

    @pytest.mark.unit
    def test_nx_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="nx.*positive"):
            generate_burgers_data(nx=0, nt=10)

    @pytest.mark.unit
    def test_nx_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="nx.*positive"):
            generate_burgers_data(nx=-10, nt=10)

    @pytest.mark.unit
    def test_nt_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="nt.*positive"):
            generate_burgers_data(nx=10, nt=0)

    @pytest.mark.unit
    def test_nt_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="nt.*positive"):
            generate_burgers_data(nx=10, nt=-5)

    @pytest.mark.unit
    def test_nu_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="nu.*positive"):
            generate_burgers_data(nx=64, nt=51, nu=0.0)

    @pytest.mark.unit
    def test_nu_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="nu.*positive"):
            generate_burgers_data(nx=64, nt=51, nu=-0.1)

    @pytest.mark.unit
    def test_noise_level_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="noise_level.*non-negative"):
            generate_burgers_data(nx=32, nt=21, noise_level=-0.1)

    @pytest.mark.unit
    def test_nx_exceeds_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="nx"):
            generate_burgers_data(
                nx=10001, nt=5
            )

    @pytest.mark.unit
    def test_nt_exceeds_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="nt"):
            generate_burgers_data(
                nx=5, nt=10001
            )

    @pytest.mark.slow
    @pytest.mark.unit
    def test_nx_at_limit_is_valid(self) -> None:


        dataset = generate_burgers_data(nx=10000, nt=5)
        assert dataset.get_shape()[0] == 10000

    @pytest.mark.slow
    @pytest.mark.unit
    def test_nt_at_limit_is_valid(self) -> None:

        dataset = generate_burgers_data(nx=5, nt=10000)
        assert dataset.get_shape()[1] == 10000

    @pytest.mark.unit
    def test_noise_level_exceeds_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="noise_level"):
            generate_burgers_data(nx=32, nt=21, noise_level=10.1)

    @pytest.mark.unit
    def test_noise_level_at_limit_is_valid(self) -> None:
        dataset = generate_burgers_data(nx=32, nt=21, noise_level=10.0)

        u = dataset.get_field("u")
        assert torch.isfinite(u).all()







class TestBurgersRK4IFAccuracy:

    @pytest.mark.numerical
    def test_rk4if_step_matches_linear_decay_single_mode(self) -> None:
        from kd.data.synthetic._burgers import _rk4if_step

        nx = 64
        L = 2.0
        dx = L / nx
        x = torch.linspace(-1.0, 1.0 - dx, nx, dtype=torch.float64)

        amp = 1e-3
        k_idx = 4
        u0 = amp * torch.cos(k_idx * torch.pi * x)

        nu = 0.1
        dt = 0.005
        nsteps = 10
        t_final = nsteps * dt

        k = torch.fft.fftfreq(nx, d=dx) * 2 * torch.pi
        nu_k2 = nu * k * k
        dealias_mask = torch.ones(nx, dtype=torch.float64)

        u = u0.clone()
        for _ in range(nsteps):
            u = _rk4if_step(u, k, nu_k2, dealias_mask, dt)


        u_exact = u0 * torch.exp(
            torch.tensor(-nu * (k_idx * torch.pi) ** 2 * t_final, dtype=torch.float64)
        )
        max_err = (u - u_exact).abs().max().item()
        rel_err = max_err / amp


        assert rel_err < 1e-3, (
            f"RK4-IF linear decay error {rel_err:.4e} exceeds tolerance"
        )

    @pytest.mark.numerical
    def test_rk4if_step_stage3_no_extra_exp_half(self) -> None:
        from kd.data.synthetic._burgers import _nonlinear_term

        nx = 32
        L = 2.0
        dx = L / nx
        x = torch.linspace(-1.0, 1.0 - dx, nx, dtype=torch.float64)
        nu = 0.5
        dt = 0.01
        amp = 1.0
        k_idx = 6
        u0 = amp * torch.cos(k_idx * torch.pi * x)
        k = torch.fft.fftfreq(nx, d=dx) * 2 * torch.pi
        nu_k2 = nu * k * k
        dealias_mask = torch.ones(nx, dtype=torch.float64)


        exp_half = torch.exp(-nu_k2 * dt / 2)
        u_hat = torch.fft.fft(u0)
        n1 = _nonlinear_term(u0, k, dealias_mask)
        k1_hat = torch.fft.fft(n1)
        u2_hat = exp_half * u_hat + (dt / 2) * exp_half * k1_hat
        u2 = torch.fft.ifft(u2_hat).real
        n2 = _nonlinear_term(u2, k, dealias_mask)
        k2_hat = torch.fft.fft(n2)



        u3_hat_correct = exp_half * u_hat + (dt / 2) * k2_hat


        u3_hat_buggy = exp_half * u_hat + (dt / 2) * exp_half * k2_hat



        diff = (u3_hat_correct - u3_hat_buggy).abs()


        active_idx = k_idx
        assert diff[active_idx].item() > 0.0, (
            "Correct vs buggy Stage 3 must differ at active mode"
        )




        from kd.data.synthetic._burgers import _rk4if_step



        n3 = _nonlinear_term(torch.fft.ifft(u3_hat_correct).real, k, dealias_mask)
        k3_hat = torch.fft.fft(n3)
        exp_full = torch.exp(-nu_k2 * dt)
        u4_hat = exp_full * u_hat + dt * exp_half * k3_hat
        u4 = torch.fft.ifft(u4_hat).real
        n4 = _nonlinear_term(u4, k, dealias_mask)
        k4_hat = torch.fft.fft(n4)
        u_new_hat_expected = exp_full * u_hat + (dt / 6) * (
            exp_full * k1_hat + 2 * exp_half * k2_hat + 2 * exp_half * k3_hat + k4_hat
        )
        u_new_expected = torch.fft.ifft(u_new_hat_expected).real

        u_new_actual = _rk4if_step(u0, k, nu_k2, dealias_mask, dt)

        torch.testing.assert_close(u_new_actual, u_new_expected, rtol=1e-10, atol=1e-12)
