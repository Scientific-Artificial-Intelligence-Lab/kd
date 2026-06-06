
from __future__ import annotations

import pytest
import torch

from kd.data.schema import (
    AxisInfo,
    FieldData,
    PDEDataset,
    TaskType,
)
from tests.discover._noise_helpers import add_noise_dataset, add_noise_tensor


@pytest.mark.unit
class TestAddNoiseTensor:

    def test_bit_exact_to_prior_inline_helper(self) -> None:
        u = torch.randn(64, 32)


        torch.manual_seed(42)
        expected = u + 0.5 * u.std() * torch.randn_like(u)


        torch.manual_seed(99999)
        actual = add_noise_tensor(u, 0.5, 42)

        assert torch.equal(actual, expected)

    def test_does_not_leak_into_global_rng(self) -> None:


        u = torch.randn(10)

        torch.manual_seed(12345)
        pre_call = torch.randn(5)

        torch.manual_seed(12345)
        _ = add_noise_tensor(u, 0.2, 7)
        post_call = torch.randn(5)

        assert torch.equal(pre_call, post_call), (
            "add_noise_tensor leaked into global torch RNG; "
            "downstream randomness is not independent of this call."
        )

    def test_same_seed_produces_same_output(self) -> None:
        u = torch.randn(50)
        out_a = add_noise_tensor(u, 0.1, 123)
        out_b = add_noise_tensor(u, 0.1, 123)
        assert torch.equal(out_a, out_b)

    def test_different_seeds_produce_different_output(self) -> None:
        u = torch.randn(50)
        out_a = add_noise_tensor(u, 0.1, 1)
        out_b = add_noise_tensor(u, 0.1, 2)
        assert not torch.equal(out_a, out_b)

    def test_zero_level_is_noiseless(self) -> None:
        u = torch.randn(20)
        out = add_noise_tensor(u, 0.0, 42)
        assert torch.equal(out, u)


def _tiny_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 8)
    t = torch.linspace(0.0, 0.5, 4)
    grid_x, grid_t = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(grid_x) * torch.cos(grid_t)
    v = torch.cos(grid_x) * torch.sin(grid_t)
    return PDEDataset(
        name="tiny",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={
            "u": FieldData(name="u", values=u),
            "v": FieldData(name="v", values=v),
        },
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.mark.unit
class TestAddNoiseDataset:

    def test_bit_exact_to_prior_inline_helper(self) -> None:
        dataset = _tiny_dataset()
        assert dataset.fields is not None









        torch.manual_seed(42)
        expected_fields: dict[str, torch.Tensor] = {}
        for name, fd in dataset.fields.items():
            expected_fields[name] = (
                fd.values + 0.3 * fd.values.std() * torch.randn_like(fd.values)
            )


        torch.manual_seed(7777)
        noisy = add_noise_dataset(dataset, 0.3, 42)
        assert noisy.fields is not None

        assert set(noisy.fields.keys()) == set(expected_fields.keys())
        for name, expected_tensor in expected_fields.items():
            assert torch.equal(noisy.fields[name].values, expected_tensor), (
                f"Field {name!r} diverges from pre- inline output."
            )

    def test_does_not_leak_into_global_rng(self) -> None:
        dataset = _tiny_dataset()
        torch.manual_seed(12345)
        pre_call = torch.randn(5)

        torch.manual_seed(12345)
        _ = add_noise_dataset(dataset, 0.5, 7)
        post_call = torch.randn(5)

        assert torch.equal(pre_call, post_call), (
            "add_noise_dataset leaked into global torch RNG."
        )

    def test_preserves_metadata(self) -> None:
        dataset = _tiny_dataset()
        noisy = add_noise_dataset(dataset, 0.5, 42)
        assert dataset.axes is not None
        assert noisy.axes is not None
        assert noisy.task_type == dataset.task_type
        assert list(noisy.axes.keys()) == list(dataset.axes.keys())
        assert noisy.axis_order == dataset.axis_order
        assert noisy.lhs_field == dataset.lhs_field
        assert noisy.lhs_axis == dataset.lhs_axis

        assert noisy.name == f"{dataset.name}_noisy"

    def test_noise_is_independent_across_fields(self) -> None:
        dataset = _tiny_dataset()
        noisy = add_noise_dataset(dataset, 0.5, 42)
        assert dataset.fields is not None
        assert noisy.fields is not None
        diff_u = noisy.fields["u"].values - dataset.fields["u"].values
        diff_v = noisy.fields["v"].values - dataset.fields["v"].values


        assert diff_u.shape == diff_v.shape
        assert not torch.equal(diff_u, diff_v), (
            "Both fields received the same noise tensor; helper is broken."
        )
