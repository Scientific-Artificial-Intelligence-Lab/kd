
from __future__ import annotations

import pytest
import torch

from kd.data.schema import PDEDataset
from kd.data.synthetic import (
    generate_chaffee_infante_xu2020_data,
    generate_kdv_xu2020_data,
    generate_wave_xu2020_data,
)


class TestXu2020Generators:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("factory", "shape"),
        [
            (generate_kdv_xu2020_data, (32, 11)),
            (generate_wave_xu2020_data, (32, 11)),
            (generate_chaffee_infante_xu2020_data, (32, 11)),
        ],
    )
    def test_generators_return_finite_pde_dataset(
        self,
        factory: object,
        shape: tuple[int, int],
    ) -> None:
        dataset = factory(nx=shape[0], nt=shape[1], noise_level=0.0)

        assert isinstance(dataset, PDEDataset)
        assert dataset.get_shape() == shape
        assert dataset.axis_order == ["x", "t"]
        assert dataset.lhs_field == "u"
        assert dataset.lhs_axis == "t"
        assert torch.isfinite(dataset.get_field("u")).all()
        assert dataset.ground_truth

    @pytest.mark.unit
    def test_wave_matches_analytic_solution(self) -> None:
        dataset = generate_wave_xu2020_data(nx=24, nt=13, noise_level=0.0)
        x = dataset.get_coords("x")
        t = dataset.get_coords("t")
        xg, tg = torch.meshgrid(x, t, indexing="ij")
        expected = torch.sin(xg) * torch.cos(tg) + 0.5 * torch.sin(
            2.0 * xg
        ) * torch.cos(2.0 * tg)

        torch.testing.assert_close(dataset.get_field("u"), expected)

    @pytest.mark.unit
    def test_wave_is_not_proportional_to_u_tt(self) -> None:
        dataset = generate_wave_xu2020_data(nx=64, nt=33, noise_level=0.0)
        u = dataset.get_field("u")
        t = dataset.get_coords("t")
        dt = float((t[1] - t[0]).item())


        u_tt = torch.zeros_like(u)
        u_tt[:, 1:-1] = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (dt * dt)

        u_flat = u[:, 1:-1].flatten()
        u_tt_flat = u_tt[:, 1:-1].flatten()

        denom = torch.linalg.vector_norm(u_flat) * torch.linalg.vector_norm(u_tt_flat)
        assert float(denom.item()) > 0.0, "field or u_tt is identically zero"
        cos_sim = float(torch.dot(u_flat, u_tt_flat).item() / denom.item())

        assert abs(cos_sim) < 0.95, (
            f"|cosine_similarity(u, u_tt)| = {abs(cos_sim):.4f} >= 0.95: "
            "wave dataset is single-mode and DLGA can cheat by fitting "
            "u_tt = -u with one token instead of discovering u_xx."
        )

    @pytest.mark.unit
    def test_noise_level_is_relative_to_clean_signal_std(self) -> None:
        clean = generate_wave_xu2020_data(nx=40, nt=17, noise_level=0.0)
        noisy = generate_wave_xu2020_data(
            nx=40,
            nt=17,
            noise_level=0.15,
            seed=123,
        )
        diff = noisy.get_field("u") - clean.get_field("u")
        ratio = float(diff.std(correction=0) / clean.get_field("u").std(correction=0))

        assert ratio == pytest.approx(0.15, rel=0.25)
