
from __future__ import annotations

from typing import Any, cast

import pytest

from kd.search.llm4ed.pool import ElitePool, PoolItem


def _item(score: float, tag: str = "") -> PoolItem:
    return PoolItem(score=score, expression=f"eq_{tag or score}")


def _from_state(
    k: int, items: list[PoolItem], scores: list[float]
) -> ElitePool:


    rebuilt: ElitePool = cast(Any, ElitePool).from_state(k, items, scores)
    return rebuilt


def _evicted_pool() -> ElitePool:
    pool = ElitePool(2)
    pool.push([_item(0.6, "a"), _item(0.7, "b")])
    pool.push(_item(0.9, "c"))
    return pool


def _pooled(pool: ElitePool) -> set[tuple[float, str]]:
    return {(item.score, item.expression) for item in pool.get_top_samples()}







def test_from_state_round_trips_heap_and_ledger() -> None:
    source = _evicted_pool()
    items = source.get_top_samples()
    scores = list(source.scores)

    rebuilt = _from_state(source.k, items, scores)

    assert rebuilt.k == source.k
    assert len(rebuilt) == len(source)
    assert _pooled(rebuilt) == _pooled(source)


    assert rebuilt.scores == source.scores


def test_from_state_ledger_retains_evicted_scores_absent_from_heap() -> None:
    source = _evicted_pool()
    rebuilt = _from_state(
        source.k, source.get_top_samples(), list(source.scores)
    )
    heap_scores = {item.score for item in rebuilt.get_top_samples()}


    assert 0.6 in rebuilt.scores
    assert 0.6 not in heap_scores


def test_from_state_preserves_capacity_of_a_non_full_pool() -> None:

    pool = ElitePool(5)
    pool.push([_item(0.6, "a"), _item(0.7, "b")])
    rebuilt = _from_state(pool.k, pool.get_top_samples(), list(pool.scores))

    assert rebuilt.k == 5
    assert len(rebuilt) == 2

    rebuilt.push([_item(0.55, "c"), _item(0.8, "d"), _item(0.65, "e")])
    assert len(rebuilt) == 5


def test_from_state_rebuilds_an_empty_pool() -> None:



    rebuilt = _from_state(3, [], [])

    assert rebuilt.k == 3
    assert len(rebuilt) == 0
    assert rebuilt.scores == ()

    rebuilt.push(_item(0.7, "a"))
    assert len(rebuilt) == 1
    assert 0.7 in {item.score for item in rebuilt.get_top_samples()}







def test_push_after_from_state_obeys_edl_eviction() -> None:
    rebuilt = _from_state(
        2, _evicted_pool().get_top_samples(), list(_evicted_pool().scores)
    )



    rebuilt.push(_item(0.95, "new"))
    heap_scores = {item.score for item in rebuilt.get_top_samples()}
    assert heap_scores == {0.9, 0.95}
    assert len(rebuilt) == 2
    assert 0.95 in rebuilt.scores


    rebuilt.push(_item(0.6, "reenter"))
    assert 0.6 not in {item.score for item in rebuilt.get_top_samples()}







def test_from_state_rejects_more_items_than_capacity() -> None:


    items = [_item(0.6, "a"), _item(0.7, "b"), _item(0.8, "c")]
    with pytest.raises(ValueError):
        _from_state(2, items, [0.6, 0.7, 0.8])


def test_from_state_rejects_non_positive_capacity() -> None:

    with pytest.raises(ValueError):
        _from_state(0, [], [])
