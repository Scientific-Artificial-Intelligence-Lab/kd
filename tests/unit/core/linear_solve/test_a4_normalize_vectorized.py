
from __future__ import annotations

import pytest
import torch

from kd.core.linear_solve.stridge import STRidgeSolver, _normalize_columns



_ORD = 2


class _ItemCallCounter:

    def __init__(self) -> None:
        self.count = 0


def _install_item_counter(
    monkeypatch: pytest.MonkeyPatch, counter: _ItemCallCounter
) -> None:
    original = torch.Tensor.item

    def _counting_item(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        counter.count += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", _counting_item)


def _seeded_system(
    n: int, d: int, dtype: torch.dtype = torch.float64, seed: int = 21
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    theta = torch.randn(n, d, generator=gen, dtype=dtype)
    coef = torch.randn(d, generator=gen, dtype=dtype)
    return theta, theta @ coef







def test_item_calls_do_not_grow_with_column_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = STRidgeSolver(tol=0.0, normalize=_ORD)

    theta_small, y_small = _seeded_system(n=200, d=5, seed=21)
    theta_large, y_large = _seeded_system(n=200, d=15, seed=21)

    counter_small = _ItemCallCounter()
    _install_item_counter(monkeypatch, counter_small)
    solver.solve(theta_small, y_small)
    count_small = counter_small.count
    monkeypatch.undo()

    counter_large = _ItemCallCounter()
    _install_item_counter(monkeypatch, counter_large)
    solver.solve(theta_large, y_large)
    count_large = counter_large.count

    assert count_large == count_small, (
        f"item() calls scaled with column count: d=5 -> {count_small}, "
        f"d=15 -> {count_large} (delta {count_large - count_small} tracks the "
        "per-column normalization loop)"
    )







def _reference_normalize(
    x0: torch.Tensor, normalize: int
) -> tuple[torch.Tensor, torch.Tensor]:
    d = x0.shape[1]
    if normalize == 0:
        mreg = torch.ones(d, 1, dtype=x0.dtype, device=x0.device)
        return x0, mreg

    mreg = torch.zeros(d, 1, dtype=x0.dtype, device=x0.device)
    x_norm = torch.zeros_like(x0)
    for i in range(d):
        col_norm = torch.linalg.norm(x0[:, i], ord=normalize).item()
        mreg[i, 0] = 1.0 / col_norm
        x_norm[:, i] = mreg[i, 0] * x0[:, i]
    return x_norm, mreg


def test_normalize_columns_matches_reference_loop_bitwise() -> None:
    shapes: list[tuple[int, int]] = [(200, 8), (5000, 20), (37, 15), (128, 3)]
    for n, d in shapes:
        gen = torch.Generator().manual_seed(100 + d)
        x0 = torch.randn(n, d, generator=gen, dtype=torch.float64)

        x_norm_prod, mreg_prod = _normalize_columns(x0, _ORD)
        x_norm_ref, mreg_ref = _reference_normalize(x0, _ORD)

        assert torch.equal(mreg_prod, mreg_ref), f"mreg mismatch at shape ({n},{d})"
        assert torch.equal(
            x_norm_prod, x_norm_ref
        ), f"x_norm mismatch at shape ({n},{d})"


    gen = torch.Generator().manual_seed(999)
    wide = torch.randn(300, 24, generator=gen, dtype=torch.float64)
    x0 = wide[:, ::2]
    x_norm_prod, mreg_prod = _normalize_columns(x0, _ORD)
    x_norm_ref, mreg_ref = _reference_normalize(x0, _ORD)
    assert torch.equal(mreg_prod, mreg_ref)
    assert torch.equal(x_norm_prod, x_norm_ref)


def test_normalize_columns_skip_branch_is_identity() -> None:
    gen = torch.Generator().manual_seed(7)
    x0 = torch.randn(50, 6, generator=gen, dtype=torch.float64)
    x_norm, mreg = _normalize_columns(x0, 0)
    assert x_norm is x0
    assert torch.equal(mreg, torch.ones(6, 1, dtype=torch.float64))
