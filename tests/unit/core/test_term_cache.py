
from __future__ import annotations

import pytest
import torch

from kd.core.term_cache import TermColumnCache


@pytest.mark.unit
def test_get_on_empty_returns_none() -> None:
    cache = TermColumnCache()
    assert cache.get("u_x") is None
    assert len(cache) == 0
    assert "u_x" not in cache


@pytest.mark.unit
def test_put_then_get_round_trips_values() -> None:
    cache = TermColumnCache()
    col = torch.arange(6, dtype=torch.float64)
    cache.put("u_x", col)

    served = cache.get("u_x")
    assert served is not None
    assert torch.equal(served, col)
    assert "u_x" in cache
    assert len(cache) == 1


@pytest.mark.unit
def test_distinct_terms_are_independent() -> None:
    cache = TermColumnCache()
    a = torch.zeros(4)
    b = torch.ones(4)
    cache.put("u", a)
    cache.put("u_x", b)

    assert len(cache) == 2
    assert torch.equal(cache.get("u"), a)
    assert torch.equal(cache.get("u_x"), b)


@pytest.mark.unit
def test_reput_same_key_does_not_grow() -> None:
    cache = TermColumnCache()
    cache.put("u_x", torch.ones(3))
    cache.put("u_x", torch.full((3,), 2.0))

    assert len(cache) == 1
    served = cache.get("u_x")
    assert served is not None
    assert torch.equal(served, torch.full((3,), 2.0))


@pytest.mark.unit
def test_clear_empties_the_cache() -> None:
    cache = TermColumnCache()
    cache.put("u", torch.ones(2))
    cache.put("u_x", torch.ones(2))
    cache.clear()

    assert len(cache) == 0
    assert cache.get("u") is None
    assert "u_x" not in cache


@pytest.mark.unit
def test_negative_missing_key_after_partial_fill() -> None:
    cache = TermColumnCache()
    cache.put("u", torch.ones(2))
    assert cache.get("never_inserted") is None
    assert "never_inserted" not in cache


@pytest.mark.unit
def test_put_stores_owned_snapshot_no_storage_aliasing() -> None:
    source = torch.ones(4)
    cache = TermColumnCache()
    cache.put("u", source)
    served = cache.get("u")
    assert served is not None
    assert served.data_ptr() != source.data_ptr(), "cached column aliases source"


    source.add_(1.0)
    assert torch.equal(served, torch.ones(4))
