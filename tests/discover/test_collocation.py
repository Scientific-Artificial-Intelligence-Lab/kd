
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from kd.search.discover.pinn.collocation import (
    generate_collocation_points,
    generate_local_samples,
)






_CPU_DEVICE = torch.device("cpu")

BOUNDS_1D = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
BOUNDS_ASYMMETRIC = {"x": (-2.0, 3.0), "t": (0.0, 10.0)}
DEFAULT_N_POINTS = 1000
DEFAULT_CUT_RATIO = 0.05
SEED = 42







def _make_observation_coords(
    n_obs: int = 50,
    device: torch.device = _CPU_DEVICE,
    seed: int = 42,
) -> dict[str, Tensor]:
    torch.manual_seed(seed)
    return {
        "x": torch.rand(n_obs, device=device),
        "t": torch.rand(n_obs, device=device),
    }







class TestGenerateCollocationPoints:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_correct_number_of_points(self) -> None:
        pts = generate_collocation_points(
            BOUNDS_1D, n_points=DEFAULT_N_POINTS, cut_ratio=0.0, seed=SEED
        )
        assert "x" in pts
        assert "t" in pts

        assert pts["x"].shape[0] == DEFAULT_N_POINTS
        assert pts["t"].shape[0] == DEFAULT_N_POINTS

    @pytest.mark.unit
    def test_all_points_within_bounds(self) -> None:
        pts = generate_collocation_points(
            BOUNDS_ASYMMETRIC, n_points=DEFAULT_N_POINTS, cut_ratio=0.0, seed=SEED
        )
        x_lo, x_hi = BOUNDS_ASYMMETRIC["x"]
        t_lo, t_hi = BOUNDS_ASYMMETRIC["t"]
        assert pts["x"].min() >= x_lo
        assert pts["x"].max() <= x_hi
        assert pts["t"].min() >= t_lo
        assert pts["t"].max() <= t_hi

    @pytest.mark.unit
    def test_trimming_reduces_points(self) -> None:
        pts_full = generate_collocation_points(
            BOUNDS_1D, n_points=DEFAULT_N_POINTS, cut_ratio=0.0, seed=SEED
        )
        pts_trimmed = generate_collocation_points(
            BOUNDS_1D, n_points=DEFAULT_N_POINTS, cut_ratio=0.1, seed=SEED
        )
        assert pts_trimmed["x"].shape[0] < pts_full["x"].shape[0]

    @pytest.mark.unit
    def test_trimmed_points_within_inner_bounds(self) -> None:
        cut = 0.1
        pts = generate_collocation_points(
            BOUNDS_1D, n_points=5000, cut_ratio=cut, seed=SEED
        )


        x_lo, x_hi = BOUNDS_1D["x"]
        t_lo, t_hi = BOUNDS_1D["t"]
        domain_x = x_hi - x_lo
        domain_t = t_hi - t_lo
        margin_x = cut * domain_x
        margin_t = cut * domain_t

        assert pts["x"].min() >= x_lo + margin_x * 0.5
        assert pts["x"].max() <= x_hi - margin_x * 0.5
        assert pts["t"].min() >= t_lo + margin_t * 0.5
        assert pts["t"].max() <= t_hi - margin_t * 0.5

    @pytest.mark.unit
    def test_lhs_more_uniform_than_random(self) -> None:
        n = 2000
        pts = generate_collocation_points(
            BOUNDS_1D, n_points=n, cut_ratio=0.0, seed=SEED
        )


        n_bins = 10
        x_vals = pts["x"].numpy()
        hist = torch.tensor(x_vals).histc(n_bins, 0.0, 1.0)


        expected_per_bin = n / n_bins

        bin_std = hist.float().std().item()


        assert bin_std < expected_per_bin * 0.5, (
            f"LHS bin std {bin_std} too high for {n} points in {n_bins} bins"
        )

    @pytest.mark.unit
    def test_reproducible_with_same_seed(self) -> None:
        pts1 = generate_collocation_points(
            BOUNDS_1D, n_points=100, cut_ratio=0.0, seed=SEED
        )
        pts2 = generate_collocation_points(
            BOUNDS_1D, n_points=100, cut_ratio=0.0, seed=SEED
        )
        assert torch.allclose(pts1["x"], pts2["x"])
        assert torch.allclose(pts1["t"], pts2["t"])

    @pytest.mark.unit
    def test_different_seeds_differ(self) -> None:
        pts1 = generate_collocation_points(
            BOUNDS_1D, n_points=100, cut_ratio=0.0, seed=1
        )
        pts2 = generate_collocation_points(
            BOUNDS_1D, n_points=100, cut_ratio=0.0, seed=2
        )
        assert not torch.allclose(pts1["x"], pts2["x"])

    @pytest.mark.unit
    def test_empty_bounds_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_collocation_points({}, n_points=100, seed=SEED)

    @pytest.mark.unit
    def test_device_respected(self) -> None:
        device = torch.device("cpu")
        pts = generate_collocation_points(
            BOUNDS_1D, n_points=100, cut_ratio=0.0, device=device, seed=SEED
        )
        assert pts["x"].device == device
        assert pts["t"].device == device

    @pytest.mark.unit
    def test_output_dtype_float32(self) -> None:
        pts = generate_collocation_points(
            BOUNDS_1D, n_points=100, cut_ratio=0.0, seed=SEED
        )
        assert pts["x"].dtype == torch.float32
        assert pts["t"].dtype == torch.float32

    @pytest.mark.unit
    def test_higher_dimensional_bounds(self) -> None:
        bounds_3d = {"x": (0.0, 1.0), "y": (0.0, 1.0), "t": (0.0, 2.0)}
        pts = generate_collocation_points(
            bounds_3d, n_points=500, cut_ratio=0.0, seed=SEED
        )
        assert set(pts.keys()) == {"x", "y", "t"}
        assert pts["x"].shape[0] == 500
        assert pts["y"].min() >= 0.0
        assert pts["t"].max() <= 2.0

    @pytest.mark.unit
    def test_no_requires_grad(self) -> None:
        pts = generate_collocation_points(
            BOUNDS_1D, n_points=100, cut_ratio=0.0, seed=SEED
        )
        assert not pts["x"].requires_grad
        assert not pts["t"].requires_grad

    @pytest.mark.unit
    def test_single_point(self) -> None:
        pts = generate_collocation_points(
            BOUNDS_1D, n_points=1, cut_ratio=0.0, seed=SEED
        )
        assert pts["x"].shape[0] == 1
        assert pts["t"].shape[0] == 1







class TestGenerateLocalSamples:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_output_count(self) -> None:
        n_obs = 50
        multiplier = 10
        obs = _make_observation_coords(n_obs)
        pts = generate_local_samples(
            obs,
            BOUNDS_1D,
            multiplier=multiplier,
            seed=SEED,
            append_observations=False,
        )
        expected = multiplier * n_obs
        assert pts["x"].shape[0] == expected
        assert pts["t"].shape[0] == expected

    @pytest.mark.unit
    def test_points_near_observations(self) -> None:
        obs = _make_observation_coords(20)
        pts = generate_local_samples(
            obs,
            BOUNDS_1D,
            multiplier=10,
            seed=SEED,
        )

        delta_x = (BOUNDS_1D["x"][1] - BOUNDS_1D["x"][0]) / 100
        delta_t = (BOUNDS_1D["t"][1] - BOUNDS_1D["t"][0]) / 100


        for i in range(pts["x"].shape[0]):
            min_dist_x = (pts["x"][i] - obs["x"]).abs().min().item()
            min_dist_t = (pts["t"][i] - obs["t"]).abs().min().item()
            assert min_dist_x <= delta_x + 1e-6, (
                f"Point {i} x={pts['x'][i]:.4f} not within delta "
                f"of any observation"
            )
            assert min_dist_t <= delta_t + 1e-6, (
                f"Point {i} t={pts['t'][i]:.4f} not within delta "
                f"of any observation"
            )

    @pytest.mark.unit
    def test_local_samples_bounded_by_domain(self) -> None:

        obs = {
            "x": torch.tensor([0.001, 0.999]),
            "t": torch.tensor([0.001, 0.999]),
        }
        pts = generate_local_samples(
            obs,
            BOUNDS_1D,
            multiplier=20,
            seed=SEED,
        )
        x_lo, x_hi = BOUNDS_1D["x"]
        t_lo, t_hi = BOUNDS_1D["t"]
        assert pts["x"].min() >= x_lo
        assert pts["x"].max() <= x_hi
        assert pts["t"].min() >= t_lo
        assert pts["t"].max() <= t_hi

    @pytest.mark.unit
    def test_empty_observation_set(self) -> None:
        obs: dict[str, Tensor] = {
            "x": torch.empty(0),
            "t": torch.empty(0),
        }
        pts = generate_local_samples(obs, BOUNDS_1D, multiplier=10, seed=SEED)
        assert pts["x"].shape[0] == 0
        assert pts["t"].shape[0] == 0

    @pytest.mark.unit
    def test_reproducible_with_seed(self) -> None:
        obs = _make_observation_coords(30)
        pts1 = generate_local_samples(obs, BOUNDS_1D, multiplier=5, seed=SEED)
        pts2 = generate_local_samples(obs, BOUNDS_1D, multiplier=5, seed=SEED)
        assert torch.allclose(pts1["x"], pts2["x"])
        assert torch.allclose(pts1["t"], pts2["t"])

    @pytest.mark.unit
    def test_device_respected(self) -> None:
        device = torch.device("cpu")
        obs = _make_observation_coords(20, device=device)
        pts = generate_local_samples(
            obs, BOUNDS_1D, multiplier=5, device=device, seed=SEED
        )
        assert pts["x"].device == device
        assert pts["t"].device == device

    @pytest.mark.unit
    def test_output_dtype_float32(self) -> None:
        obs = _make_observation_coords(20)
        pts = generate_local_samples(obs, BOUNDS_1D, multiplier=5, seed=SEED)
        assert pts["x"].dtype == torch.float32
        assert pts["t"].dtype == torch.float32

    @pytest.mark.unit
    def test_asymmetric_bounds(self) -> None:
        obs = {
            "x": torch.tensor([0.0, 1.0, 2.0]),
            "t": torch.tensor([5.0, 7.0, 9.0]),
        }
        pts = generate_local_samples(
            obs, BOUNDS_ASYMMETRIC, multiplier=10, seed=SEED,
            append_observations=False,
        )
        assert pts["x"].shape[0] == 30
        x_lo, x_hi = BOUNDS_ASYMMETRIC["x"]
        t_lo, t_hi = BOUNDS_ASYMMETRIC["t"]
        assert pts["x"].min() >= x_lo
        assert pts["x"].max() <= x_hi
        assert pts["t"].min() >= t_lo
        assert pts["t"].max() <= t_hi







class TestTD060MultiplierDefault:

    @pytest.mark.unit
    def test_default_multiplier_is_20(self) -> None:
        n_obs = 10
        obs = _make_observation_coords(n_obs)

        pts = generate_local_samples(
            obs, BOUNDS_1D, seed=SEED, append_observations=False,
        )

        assert pts["x"].shape[0] == 20 * n_obs

    @pytest.mark.unit
    def test_full_defaults_output_count(self) -> None:
        n_obs = 10
        obs = _make_observation_coords(n_obs)
        pts = generate_local_samples(obs, BOUNDS_1D, seed=SEED)
        assert pts["x"].shape[0] == 21 * n_obs

    @pytest.mark.unit
    def test_multiplier_1_boundary(self) -> None:
        n_obs = 20
        obs = _make_observation_coords(n_obs)
        pts = generate_local_samples(
            obs, BOUNDS_1D, multiplier=1, seed=SEED,
            append_observations=False,
        )
        assert pts["x"].shape[0] == n_obs


class TestTD060AppendObservations:

    @pytest.mark.unit
    def test_append_observations_default_true(self) -> None:
        n_obs = 20
        obs = _make_observation_coords(n_obs)
        pts = generate_local_samples(
            obs, BOUNDS_1D, multiplier=10, seed=SEED,
        )

        assert pts["x"].shape[0] == 10 * n_obs + n_obs

    @pytest.mark.unit
    def test_appended_observations_match_input(self) -> None:
        n_obs = 15
        obs = _make_observation_coords(n_obs)
        pts = generate_local_samples(
            obs, BOUNDS_1D, multiplier=5, seed=SEED,
        )

        n_perturbed = 5 * n_obs
        total = n_perturbed + n_obs
        assert pts["x"].shape[0] == total
        appended_x = pts["x"][n_perturbed:]
        appended_t = pts["t"][n_perturbed:]
        assert torch.allclose(appended_x, obs["x"])
        assert torch.allclose(appended_t, obs["t"])

    @pytest.mark.unit
    def test_append_observations_false_excludes_originals(self) -> None:
        n_obs = 25
        obs = _make_observation_coords(n_obs)
        pts = generate_local_samples(
            obs, BOUNDS_1D, multiplier=10, seed=SEED,
            append_observations=False,
        )
        assert pts["x"].shape[0] == 10 * n_obs

    @pytest.mark.unit
    def test_append_observations_empty_input(self) -> None:
        obs: dict[str, Tensor] = {
            "x": torch.empty(0),
            "t": torch.empty(0),
        }
        pts = generate_local_samples(
            obs, BOUNDS_1D, multiplier=10, seed=SEED,
            append_observations=True,
        )
        assert pts["x"].shape[0] == 0

    @pytest.mark.unit
    def test_multiplier_1_with_append(self) -> None:
        n_obs = 10
        obs = _make_observation_coords(n_obs)
        pts = generate_local_samples(
            obs, BOUNDS_1D, multiplier=1, seed=SEED,
            append_observations=True,
        )
        assert pts["x"].shape[0] == 2 * n_obs

        assert torch.allclose(pts["x"][n_obs:], obs["x"])







class TestTrimSamplesDegenerate:

    @pytest.mark.unit
    def test_trim_samples_empty_result_raises(self) -> None:
        import numpy as np

        from kd.search.discover.pinn.collocation import _trim_samples

        samples = np.array(
            [[0.0, 0.0], [1.0, 1.0]], dtype=np.float64,
        )
        with pytest.raises(ValueError, match="cut_ratio"):
            _trim_samples(samples, cut_ratio=0.5)

    @pytest.mark.unit
    def test_trim_samples_cut_ratio_zero_passthrough(self) -> None:
        import numpy as np

        from kd.search.discover.pinn.collocation import _trim_samples

        samples = np.array([[0.5, 0.5]], dtype=np.float64)
        out = _trim_samples(samples, cut_ratio=0.0)
        assert out is samples
