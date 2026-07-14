
from __future__ import annotations

import pytest
import torch

from kd.core.term_cache import TermColumnCache


@pytest.mark.unit
def test_default_generation_is_zero_backcompat() -> None:
    cache = TermColumnCache()
    col = torch.arange(5, dtype=torch.float64)

    cache.put("u_x", col, generation=0)
    served_kw = cache.get("u_x", generation=0)
    served_default = cache.get("u_x")

    assert served_kw is not None
    assert served_default is not None
    assert torch.equal(served_kw, col)
    assert torch.equal(served_default, col)


@pytest.mark.unit
def test_column_stored_under_one_generation_misses_under_another() -> None:
    cache = TermColumnCache()
    col_gen0 = torch.ones(4, dtype=torch.float64)

    cache.put("u_x", col_gen0, generation=0)

    assert cache.get("u_x", generation=1) is None, "stale-generation hit leaked"
    served = cache.get("u_x", generation=0)
    assert served is not None
    assert torch.equal(served, col_gen0)


@pytest.mark.unit
def test_distinct_generations_of_same_term_coexist() -> None:
    cache = TermColumnCache()
    col_gen0 = torch.zeros(3, dtype=torch.float64)
    col_gen1 = torch.full((3,), 7.0, dtype=torch.float64)

    cache.put("u_x", col_gen0, generation=0)
    cache.put("u_x", col_gen1, generation=1)

    assert len(cache) == 2, "generation must partition the key space"
    served0 = cache.get("u_x", generation=0)
    served1 = cache.get("u_x", generation=1)
    assert served0 is not None and served1 is not None
    assert torch.equal(served0, col_gen0)
    assert torch.equal(served1, col_gen1)


@pytest.mark.unit
def test_get_on_empty_with_generation_returns_none() -> None:
    cache = TermColumnCache()
    assert cache.get("u_x", generation=3) is None


@pytest.mark.numerical
def test_valve_flushes_across_generations() -> None:
    cache = TermColumnCache(max_bytes=16)

    cache.put("a", torch.arange(2, dtype=torch.float32), generation=0)
    cache.put("b", torch.arange(2, dtype=torch.float32), generation=1)
    cache.put("c", torch.arange(2, dtype=torch.float32), generation=2)

    assert cache.get("a", generation=0) is None, "generation-0 entry survived flush"
    assert cache.get("b", generation=1) is None, "generation-1 entry survived flush"
    served = cache.get("c", generation=2)
    assert served is not None
    assert len(cache) == 1


@pytest.mark.numerical
def test_put_with_generation_owns_detached_snapshot() -> None:
    source = torch.ones(4, requires_grad=True)
    cache = TermColumnCache()

    cache.put("u", source, generation=2)
    served = cache.get("u", generation=2)

    assert served is not None
    assert served.requires_grad is False
    assert served.data_ptr() != source.data_ptr(), "cached column aliases source"

    with torch.no_grad():
        source.add_(100.0)
    assert torch.equal(served, torch.ones(4)), "generation snapshot was mutated"


@pytest.mark.unit
def test_reput_same_generation_and_term_overwrites() -> None:
    cache = TermColumnCache()
    cache.put("u_x", torch.ones(3), generation=1)
    cache.put("u_x", torch.full((3,), 2.0), generation=1)

    assert len(cache) == 1
    served = cache.get("u_x", generation=1)
    assert served is not None
    assert torch.equal(served, torch.full((3,), 2.0))
