
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from kd.core.expr.naming import parse_derivative_name
from kd.data import synthetic
from kd.data.schema import PDEDataset
from kd.data.synthetic import (
    generate_chaffee_infante_xu2020_data,
    generate_kdv_xu2020_data,
    generate_wave_xu2020_data,
)




BUILD_DATASET_GENERATORS: tuple[Callable[..., PDEDataset], ...] = (
    generate_kdv_xu2020_data,
    generate_wave_xu2020_data,
    generate_chaffee_infante_xu2020_data,
)
BUILD_DATASET_MODULES = frozenset(
    {"_kdv_xu2020.py", "_wave_xu2020.py", "_chaffee_infante.py"}
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
    @pytest.mark.parametrize(
        "factory",
        BUILD_DATASET_GENERATORS,
        ids=lambda f: f.__name__,
    )
    def test_declared_lhs_matches_ground_truth_lhs(
        self,
        factory: Callable[..., PDEDataset],
    ) -> None:
        dataset = factory(nx=32, nt=11, noise_level=0.0)
        assert dataset.ground_truth is not None

        lhs_name = dataset.ground_truth.split("=")[0].strip()
        parsed = parse_derivative_name(
            lhs_name,
            known_fields=set(dataset.fields or {}),
            known_axes=set(dataset.axes or {}),
        )
        assert parsed is not None, (
            f"{dataset.name}: ground_truth LHS {lhs_name!r} does not parse as "
            "a same-axis derivative"
        )

        assert parsed == (dataset.lhs_field, dataset.lhs_axis, dataset.lhs_order), (
            f"{dataset.name}: ground_truth '{dataset.ground_truth}' has LHS "
            f"{lhs_name!r} = {parsed}, but the dataset declares "
            f"(field={dataset.lhs_field!r}, axis={dataset.lhs_axis!r}, "
            f"order={dataset.lhs_order})"
        )

    @pytest.mark.unit
    def test_build_dataset_family_is_covered_by_lhs_contract(self) -> None:
        synthetic_dir = Path(synthetic.__file__).parent
        users = {
            path.name
            for path in synthetic_dir.glob("*.py")
            if "build_dataset(" in path.read_text(encoding="utf-8")
        } - {"_xu2020_common.py"}

        assert users == set(BUILD_DATASET_MODULES), (
            "modules using _xu2020_common.build_dataset changed: "
            f"{sorted(users)}; add the new generator to "
            "BUILD_DATASET_GENERATORS/BUILD_DATASET_MODULES so its LHS "
            "declaration is contract-tested"
        )

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
