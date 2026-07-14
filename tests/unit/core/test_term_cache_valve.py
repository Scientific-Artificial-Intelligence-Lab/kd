
from __future__ import annotations

import torch
from torch import Tensor

from kd.core import term_cache as term_cache_module
from kd.core.term_cache import TermColumnCache


def _column(n: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.arange(n, dtype=dtype)


def test_valve_flushes_whole_cache_when_budget_would_be_exceeded() -> None:
    cache = TermColumnCache(max_bytes=16)

    cache.put("a", _column(2))
    cache.put("b", _column(2))
    cache.put("c", _column(2))

    assert "a" not in cache
    assert "b" not in cache
    assert torch.equal(cache.get("c"), _column(2))
    assert len(cache) == 1


def test_overwrite_accounting_does_not_double_count_existing_key() -> None:
    cache = TermColumnCache(max_bytes=32)

    cache.put("a", _column(3))
    cache.put("a", _column(3) + 10)
    cache.put("b", _column(5))

    assert torch.equal(cache.get("a"), _column(3) + 10)
    assert cache.get("b") is not None
    assert len(cache) == 2

    cache.put("c", _column(1))

    assert "a" not in cache
    assert "b" not in cache
    assert torch.equal(cache.get("c"), _column(1))
    assert len(cache) == 1


def test_clear_resets_byte_accounting() -> None:
    cache = TermColumnCache(max_bytes=24)

    cache.put("a", _column(4))
    cache.clear()

    cache.put("b", _column(2))
    cache.put("c", _column(2))
    cache.put("d", _column(2))

    assert cache.get("b") is not None
    assert cache.get("c") is not None
    assert cache.get("d") is not None
    assert len(cache) == 3


def test_oversized_single_column_is_stored_until_next_put_flushes_it() -> None:
    cache = TermColumnCache(max_bytes=8)

    cache.put("big", _column(3))

    assert torch.equal(cache.get("big"), _column(3))
    assert len(cache) == 1

    cache.put("small", _column(1))

    assert "big" not in cache
    assert torch.equal(cache.get("small"), _column(1))
    assert len(cache) == 1


def test_default_budget_allows_bounded_vocab_scale_without_flushing() -> None:
    assert term_cache_module.DEFAULT_MAX_BYTES == 1_073_741_824

    cache = TermColumnCache()
    for i in range(100):
        cache.put(f"term_{i}", _column(1_000))

    assert len(cache) == 100
    assert torch.equal(cache.get("term_0"), _column(1_000))
    assert torch.equal(cache.get("term_99"), _column(1_000))


def test_ownership_contract_survives_with_valve_enabled() -> None:
    cache = TermColumnCache(max_bytes=16)
    source = torch.arange(4, dtype=torch.float32, requires_grad=True)

    cache.put("u", source)
    served = cache.get("u")

    assert served is not None
    assert served is not source
    assert served.requires_grad is False
    assert torch.equal(served, torch.arange(4, dtype=torch.float32))

    with torch.no_grad():
        source.add_(100)

    assert torch.equal(served, torch.arange(4, dtype=torch.float32))
