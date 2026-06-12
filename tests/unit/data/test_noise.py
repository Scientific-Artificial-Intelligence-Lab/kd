
from __future__ import annotations

import math

import pytest
import torch

from kd.data.noise import discover_unnormalized, xu2020_relative
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType

_N = 64
_LEVEL = 0.15
_SEED = 42


def _signal(n: int = _N) -> torch.Tensor:
    base = torch.linspace(-1.3, 2.7, n, dtype=torch.float64)
    return torch.sin(base) + 0.25 * base


class TestXu2020Relative:

    @pytest.mark.unit
    def test_realized_noise_std_is_exactly_level_times_population_std(
        self,
    ) -> None:
        u = _signal()
        out = xu2020_relative(u, _LEVEL, seed=_SEED)
        noise = out - u

        realized = noise.std(correction=0)
        target = _LEVEL * u.std(correction=0)
        torch.testing.assert_close(realized, target, rtol=1e-9, atol=0.0)

    @pytest.mark.unit
    def test_target_uses_ddof0_not_ddof1(self) -> None:
        u = _signal()
        noise = xu2020_relative(u, _LEVEL, seed=_SEED) - u

        realized = noise.std(correction=0)
        wrong_target = _LEVEL * u.std(correction=1)
        assert not torch.isclose(realized, wrong_target, rtol=1e-3)

    @pytest.mark.unit
    def test_zero_level_returns_input_unchanged_no_rng(self) -> None:
        u = _signal()
        state = torch.random.get_rng_state()
        out = xu2020_relative(u, 0.0, seed=None)
        assert out is u
        assert torch.equal(torch.random.get_rng_state(), state)

    @pytest.mark.unit
    def test_same_seed_is_deterministic(self) -> None:
        u = _signal()
        a = xu2020_relative(u, _LEVEL, seed=_SEED)
        b = xu2020_relative(u, _LEVEL, seed=_SEED)
        assert torch.equal(a, b)

    @pytest.mark.unit
    def test_rejects_negative_and_nonfinite_level(self) -> None:
        u = _signal()
        with pytest.raises(ValueError, match="noise_level"):
            xu2020_relative(u, -0.1, seed=_SEED)
        with pytest.raises(ValueError, match="noise_level"):
            xu2020_relative(u, math.inf, seed=_SEED)
        with pytest.raises(ValueError, match="noise_level"):
            xu2020_relative(u, math.nan, seed=_SEED)

    @pytest.mark.unit
    def test_rejects_zero_std_signal(self) -> None:
        u = torch.ones(16, dtype=torch.float64)
        with pytest.raises(ValueError, match="zero std"):
            xu2020_relative(u, _LEVEL, seed=_SEED)


class TestDiscoverTd099:

    @pytest.mark.unit
    def test_std_mode_sigma_uses_sample_std_ddof1(self) -> None:
        u = _signal()
        out = discover_unnormalized(u, _LEVEL, seed=_SEED)

        z = torch.randn(
            u.shape,
            generator=torch.Generator().manual_seed(_SEED),
            dtype=u.dtype,
            device=u.device,
        )
        sigma = _LEVEL * torch.std(u, unbiased=True)
        assert torch.equal(out, u + sigma * z)

    @pytest.mark.unit
    def test_std_mode_does_not_use_ddof0(self) -> None:
        u = _signal()
        out = discover_unnormalized(u, _LEVEL, seed=_SEED)

        z = torch.randn(
            u.shape,
            generator=torch.Generator().manual_seed(_SEED),
            dtype=u.dtype,
            device=u.device,
        )
        wrong = u + (_LEVEL * u.std(correction=0)) * z
        assert not torch.allclose(out, wrong, rtol=1e-9, atol=0.0)

    @pytest.mark.unit
    def test_noise_is_not_renormalized_to_exact_sigma(self) -> None:
        u = _signal()
        noise = discover_unnormalized(u, _LEVEL, seed=_SEED) - u

        sigma = _LEVEL * torch.std(u, unbiased=True)
        realized = noise.std(correction=1)
        assert float((realized / sigma - 1.0).abs()) > 1e-6

    @pytest.mark.unit
    def test_max_mode_sigma_uses_abs_max(self) -> None:
        u = _signal()
        out = discover_unnormalized(u, _LEVEL, seed=_SEED, scale="max")

        z = torch.randn(
            u.shape,
            generator=torch.Generator().manual_seed(_SEED),
            dtype=u.dtype,
            device=u.device,
        )
        assert torch.equal(out, u + (_LEVEL * u.abs().max()) * z)

    @pytest.mark.unit
    def test_seed_and_generator_are_mutually_exclusive(self) -> None:
        u = _signal()
        with pytest.raises(ValueError, match="exactly one"):
            discover_unnormalized(u, _LEVEL)
        with pytest.raises(ValueError, match="exactly one"):
            discover_unnormalized(
                u,
                _LEVEL,
                seed=_SEED,
                generator=torch.Generator().manual_seed(_SEED),
            )

    @pytest.mark.unit
    def test_rejects_negative_level_and_unknown_scale(self) -> None:
        u = _signal()
        with pytest.raises(ValueError, match="non-negative"):
            discover_unnormalized(u, -0.1, seed=_SEED)
        with pytest.raises(ValueError, match="scale"):
            discover_unnormalized(u, _LEVEL, seed=_SEED, scale="bogus")


class TestRecipeDivergence:

    @pytest.mark.unit
    def test_target_stds_differ_at_same_level(self) -> None:
        u = _signal()
        target_xu = _LEVEL * u.std(correction=0)
        target_td = _LEVEL * torch.std(u, unbiased=True)
        assert not torch.isclose(target_xu, target_td, rtol=1e-3)

    @pytest.mark.unit
    def test_realized_noise_std_differs_at_same_level_and_seed(self) -> None:
        u = _signal()
        noise_xu = xu2020_relative(u, _LEVEL, seed=_SEED) - u
        noise_td = discover_unnormalized(u, _LEVEL, seed=_SEED) - u

        realized_xu = noise_xu.std(correction=0)
        realized_td = noise_td.std(correction=0)
        assert not torch.isclose(realized_xu, realized_td, rtol=1e-6)


def _two_field_dataset() -> PDEDataset:
    u = _signal(108).reshape(12, 9)
    v = (_signal(108) * 0.5 - 1.0).reshape(12, 9)
    return PDEDataset(
        name="toy",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=torch.linspace(0.0, 1.0, 12)),
            "t": AxisInfo(name="t", values=torch.linspace(0.0, 0.5, 9)),
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


class TestDelegationBitIdentical:

    @pytest.mark.unit
    def test_add_relative_noise_equals_xu2020_recipe(self) -> None:
        from kd.data.synthetic._xu2020_common import add_relative_noise

        u = _signal()
        wrapped = add_relative_noise(u, _LEVEL, seed=_SEED)
        direct = xu2020_relative(u, _LEVEL, seed=_SEED)
        assert torch.equal(wrapped, direct)

    @pytest.mark.unit
    def test_add_gaussian_noise_equals_unnormalized_recipe_per_field(self) -> None:
        from kd.search.discover.data.loader import add_gaussian_noise

        dataset = _two_field_dataset()
        assert dataset.fields is not None
        noisy = add_gaussian_noise(dataset, _LEVEL, _SEED, scale="std")
        assert noisy.fields is not None

        generator = torch.Generator().manual_seed(_SEED)
        for name, field in dataset.fields.items():
            expected = discover_unnormalized(
                field.values,
                _LEVEL,
                generator=generator,
                scale="std",
            )
            assert torch.equal(noisy.fields[name].values, expected), name
