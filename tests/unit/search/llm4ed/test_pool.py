
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kd.search.llm4ed.pool import ElitePool, PoolItem


def _item(score: float, tag: str = "") -> PoolItem:
    return PoolItem(score=score, expression=f"eq_{tag or score}")


def _pooled_scores(pool: ElitePool) -> list[float]:
    return [it.score for it in pool.get_top_samples()]


class TestPoolItemOrdering:

    def test_orders_by_score_only(self) -> None:
        assert _item(0.3, "a") < _item(0.7, "b")
        assert not (_item(0.7, "a") < _item(0.3, "b"))

    def test_equal_score_not_less_than(self) -> None:


        a, b = _item(0.5, "a"), _item(0.5, "b")
        assert not (a < b) and not (b < a)

    def test_frozen_and_carries_expression(self) -> None:
        it = PoolItem(score=0.9, expression="u + u_xx")
        assert it.score == pytest.approx(0.9)
        assert it.expression == "u + u_xx"


class TestConstruction:
    def test_empty_pool_len_zero(self) -> None:
        assert len(ElitePool(5)) == 0

    def test_empty_pool_scores_empty(self) -> None:
        assert ElitePool(5).scores == ()

    @pytest.mark.parametrize("bad_k", [0, -1, True, 2.0, "3"])
    def test_bad_k_rejected(self, bad_k: object) -> None:

        with pytest.raises((ValueError, TypeError)):
            ElitePool(bad_k)


class TestUnconditionalFill:

    def test_distinct_fill(self) -> None:
        pool = ElitePool(3)
        for s in (0.6, 0.7, 0.8):
            pool.push(_item(s))
        assert len(pool) == 3
        assert sorted(_pooled_scores(pool)) == pytest.approx([0.6, 0.7, 0.8])

    def test_duplicate_scores_admitted_during_fill(self) -> None:


        pool = ElitePool(3)
        for tag in ("a", "b", "c"):
            pool.push(_item(0.7, tag))
        assert len(pool) == 3
        assert sorted(pool.scores) == pytest.approx([0.7, 0.7, 0.7])

    def test_list_push_equivalent_to_sequential(self) -> None:



        batch = ElitePool(3)
        batch.push([_item(0.6, "a"), _item(0.7, "b"), _item(0.8, "c")])
        seq = ElitePool(3)
        for it in (_item(0.6, "a"), _item(0.7, "b"), _item(0.8, "c")):
            seq.push(it)
        assert sorted(_pooled_scores(batch)) == pytest.approx(
            sorted(_pooled_scores(seq))
        )

    def test_empty_list_push_is_noop(self) -> None:
        pool = ElitePool(3)
        pool.push([])
        assert len(pool) == 0


class TestHeapFullDedup:

    @staticmethod
    def _full_pool() -> ElitePool:
        pool = ElitePool(3)
        for s in (0.6, 0.7, 0.8):
            pool.push(_item(s))
        return pool

    def test_reject_score_below_min(self) -> None:
        pool = self._full_pool()
        pool.push(_item(0.5, "low"))
        assert len(pool) == 3
        assert 0.5 not in _pooled_scores(pool)

    def test_reject_score_equal_to_min(self) -> None:





        pool = self._full_pool()
        pool.push(_item(0.6, "tie"))
        assert len(pool) == 3
        assert sorted(_pooled_scores(pool)) == pytest.approx([0.6, 0.7, 0.8])

    def test_reject_duplicate_of_score_in_pool(self) -> None:

        pool = self._full_pool()
        pool.push(_item(0.7, "dup"))
        assert len(pool) == 3
        assert sorted(pool.scores) == pytest.approx([0.6, 0.7, 0.8])

    def test_accept_strictly_greater_evicts_min(self) -> None:
        pool = self._full_pool()
        pool.push(_item(0.9, "high"))
        assert len(pool) == 3
        assert sorted(_pooled_scores(pool)) == pytest.approx([0.7, 0.8, 0.9])

        assert 0.6 not in _pooled_scores(pool)

    def test_evicted_high_expression_carried_through(self) -> None:

        pool = self._full_pool()
        pool.push(PoolItem(score=0.95, expression="u_t = winner"))
        tops = pool.get_top_samples()
        assert tops[0].score == pytest.approx(0.95)
        assert tops[0].expression == "u_t = winner"


class TestListPushIntoFullPool:

    @staticmethod
    def _full_pool() -> ElitePool:
        pool = ElitePool(3)
        for s in (0.6, 0.7, 0.8):
            pool.push(_item(s))
        return pool

    def test_intermediate_eviction_updates_min_within_batch(self) -> None:





        pool = self._full_pool()
        pool.push([_item(0.65, "hi"), _item(0.62, "lo")])
        assert sorted(_pooled_scores(pool)) == pytest.approx([0.65, 0.7, 0.8])
        assert 0.62 not in _pooled_scores(pool)

    def test_intermediate_accepted_item_recorded_in_ledger(self) -> None:


        pool = self._full_pool()
        pool.push([_item(0.65, "hi"), _item(0.62, "lo")])
        assert 0.65 in pool.scores
        assert 0.62 not in pool.scores

    def test_batch_equals_sequential_across_full_boundary(self) -> None:



        items = [
            _item(0.6, "a"),
            _item(0.7, "b"),
            _item(0.8, "c"),
            _item(0.9, "d"),
            _item(0.75, "e"),
        ]
        batch = ElitePool(3)
        batch.push(list(items))
        seq = ElitePool(3)
        for it in items:
            seq.push(it)
        assert _pooled_scores(batch) == pytest.approx(_pooled_scores(seq))
        assert batch.scores == pytest.approx(seq.scores)


class TestAppendOnlyScoresQuirk:

    @staticmethod
    def _evicted_pool() -> ElitePool:
        pool = ElitePool(3)
        for s in (0.6, 0.7, 0.8):
            pool.push(_item(s))
        pool.push(_item(0.9, "high"))
        return pool

    def test_ledger_retains_evicted_score(self) -> None:
        pool = self._evicted_pool()

        assert 0.6 in pool.scores
        assert 0.6 not in _pooled_scores(pool)

    def test_ledger_is_append_only_superset_of_heap(self) -> None:
        pool = self._evicted_pool()

        assert sorted(pool.scores) == pytest.approx([0.6, 0.7, 0.8, 0.9])
        assert pool.scores[-1] == pytest.approx(0.9)

    def test_evicted_score_cannot_reenter(self) -> None:


        pool = self._evicted_pool()
        pool.push(_item(0.6, "reenter"))
        assert 0.6 not in _pooled_scores(pool)
        assert len(pool) == 3


class TestTopKRetrieval:
    def test_descending_order(self) -> None:
        pool = ElitePool(4)
        for s in (0.71, 0.93, 0.55, 0.82):
            pool.push(_item(s))
        scores = _pooled_scores(pool)
        assert scores == sorted(scores, reverse=True)

    def test_returns_pool_items(self) -> None:
        pool = ElitePool(2)
        pool.push(_item(0.8, "a"))
        pool.push(_item(0.9, "b"))
        tops = pool.get_top_samples()
        assert all(isinstance(it, PoolItem) for it in tops)
        assert tops[0].score == pytest.approx(0.9)


class TestPoolInvariants:

    @given(
        k=st.integers(min_value=1, max_value=6),
        scores=st.lists(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            max_size=25,
        ),
    )
    def test_len_bounded_and_descending_and_from_input(
        self, k: int, scores: list[float]
    ) -> None:
        pool = ElitePool(k)
        for i, s in enumerate(scores):
            pool.push(_item(s, str(i)))
        pooled = _pooled_scores(pool)

        assert len(pool) <= k

        assert pooled == sorted(pooled, reverse=True)

        assert set(pooled) <= set(scores)

        from collections import Counter

        heap_counts = Counter(pooled)
        ledger_counts = Counter(pool.scores)
        assert all(ledger_counts[s] >= c for s, c in heap_counts.items())
