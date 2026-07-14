
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from kd.search.sga.train import _stridge_no_debias

_ORD = 2


class _ItemCallCounter:

    def __init__(self) -> None:
        self.count = 0


def _install_item_counter(
    monkeypatch: pytest.MonkeyPatch, counter: _ItemCallCounter
) -> None:
    original = torch.Tensor.item

    def _counting_item(self: Tensor, *args: object, **kwargs: object) -> object:
        counter.count += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", _counting_item)


def _seeded_system(
    n: int, d: int, dtype: torch.dtype = torch.float32, seed: int = 33
) -> tuple[Tensor, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    theta = torch.randn(n, d, generator=gen, dtype=dtype)
    coef = torch.randn(d, generator=gen, dtype=dtype)
    return theta, theta @ coef







def test_item_calls_do_not_grow_with_column_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta_small, y_small = _seeded_system(n=200, d=5, seed=33)
    theta_large, y_large = _seeded_system(n=200, d=15, seed=33)

    counter_small = _ItemCallCounter()
    _install_item_counter(monkeypatch, counter_small)
    _stridge_no_debias(
        theta_small, y_small, lam=0.0, max_iter=10, tol=0.0, normalize=_ORD
    )
    count_small = counter_small.count
    monkeypatch.undo()

    counter_large = _ItemCallCounter()
    _install_item_counter(monkeypatch, counter_large)
    _stridge_no_debias(
        theta_large, y_large, lam=0.0, max_iter=10, tol=0.0, normalize=_ORD
    )
    count_large = counter_large.count

    assert count_large == count_small, (
        f"item() calls scaled with column count: d=5 -> {count_small}, "
        f"d=15 -> {count_large} (delta {count_large - count_small} tracks the "
        "per-column normalization loop in train.py)"
    )







def _reference_loop_normalize(
    x0: Tensor, normalize: int
) -> tuple[Tensor, Tensor]:
    d_reduced = x0.shape[1]
    if normalize == 0:
        return x0, torch.ones(d_reduced, 1, dtype=x0.dtype, device=x0.device)
    mreg = torch.zeros(d_reduced, 1, dtype=x0.dtype, device=x0.device)
    x_norm = torch.zeros_like(x0)
    for i in range(d_reduced):
        cn = torch.linalg.norm(x0[:, i], ord=normalize).item()
        mreg[i, 0] = 1.0 / cn
        x_norm[:, i] = mreg[i, 0] * x0[:, i]
    return x_norm, mreg


def _vectorized_normalize(x0: Tensor, normalize: int) -> tuple[Tensor, Tensor]:
    d_reduced = x0.shape[1]
    if normalize == 0:
        return x0, torch.ones(d_reduced, 1, dtype=x0.dtype, device=x0.device)
    norms = torch.linalg.norm(x0, ord=normalize, dim=0)
    mreg = (1.0 / norms).unsqueeze(1)
    x_norm = x0 * mreg.squeeze(-1)
    return x_norm, mreg


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
def test_vectorized_normalize_matches_loop_bitwise(dtype: torch.dtype) -> None:
    for n, d in [(200, 8), (5000, 20), (37, 15), (300, 12)]:
        gen = torch.Generator().manual_seed(400 + d)
        x0 = torch.randn(n, d, generator=gen, dtype=dtype)

        x_norm_vec, mreg_vec = _vectorized_normalize(x0, _ORD)
        x_norm_ref, mreg_ref = _reference_loop_normalize(x0, _ORD)

        assert torch.equal(
            mreg_vec, mreg_ref
        ), f"mreg mismatch: dtype={dtype}, shape=({n},{d})"
        assert torch.equal(
            x_norm_vec, x_norm_ref
        ), f"x_norm mismatch: dtype={dtype}, shape=({n},{d})"
