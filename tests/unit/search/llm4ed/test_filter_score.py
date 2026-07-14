
from __future__ import annotations

import pytest

from kd.search.llm4ed.filter_score import filter_score
from kd.search.llm4ed.pool import PoolItem


def _c(score: float, tag: str = "") -> PoolItem:
    return PoolItem(score=score, expression=f"eq_{tag or score}")


class TestRewardLimit:

    def test_above_limit_kept(self) -> None:
        cache: dict[float, str] = {}
        kept = filter_score([_c(0.7), _c(0.9)], 0.5, [], cache)
        assert [c.score for c in kept] == pytest.approx([0.7, 0.9])

    def test_below_limit_dropped(self) -> None:
        cache: dict[float, str] = {}
        kept = filter_score([_c(0.3), _c(0.49)], 0.5, [], cache)
        assert kept == []

    def test_score_equal_to_limit_dropped(self) -> None:

        cache: dict[float, str] = {}
        kept = filter_score([_c(0.5, "eq")], 0.5, [], cache)
        assert kept == []

    def test_mixed_preserves_first_seen_order(self) -> None:
        cache: dict[float, str] = {}
        kept = filter_score(
            [_c(0.3, "a"), _c(0.8, "b"), _c(0.4, "c"), _c(0.6, "d")],
            0.5,
            [],
            cache,
        )
        assert [c.expression for c in kept] == ["eq_b", "eq_d"]

    def test_empty_candidates(self) -> None:
        cache: dict[float, str] = {}
        assert filter_score([], 0.5, [], cache) == []


class TestSameScoreDedup:

    def test_repeat_score_dropped(self) -> None:
        cache: dict[float, str] = {}
        kept = filter_score([_c(0.7, "a"), _c(0.7, "b")], 0.5, [], cache)
        assert [c.expression for c in kept] == ["eq_a"]

    def test_dedup_on_equal_scores(self) -> None:



        s = round(0.71237, 4)
        cache: dict[float, str] = {}
        kept = filter_score([_c(s, "a"), _c(s, "b")], 0.5, [], cache)
        assert len(kept) == 1

    def test_seen_scores_seed_blocks_admission(self) -> None:

        cache: dict[float, str] = {}
        kept = filter_score([_c(0.7, "dup"), _c(0.8, "new")], 0.5, [0.7], cache)
        assert [c.expression for c in kept] == ["eq_new"]

    def test_distinct_scores_all_kept(self) -> None:
        cache: dict[float, str] = {}
        kept = filter_score([_c(0.6), _c(0.7), _c(0.8)], 0.5, [], cache)
        assert len(kept) == 3

    def test_cache_first_wins_when_deduped(self) -> None:






        cache: dict[float, str] = {}
        filter_score([_c(0.7, "a"), _c(0.7, "b")], 0.5, [], cache)
        assert cache[0.7] == "eq_a"


class TestCacheAndSeenScores:
    def test_cache_maps_kept_score_to_expression(self) -> None:
        cache: dict[float, str] = {}
        filter_score([_c(0.7, "keep"), _c(0.3, "drop")], 0.5, [], cache)
        assert cache == {0.7: "eq_keep"}

    def test_seen_scores_not_mutated(self) -> None:


        seen = [0.9]
        cache: dict[float, str] = {}
        filter_score([_c(0.7), _c(0.8)], 0.5, seen, cache)
        assert seen == [0.9]

    def test_preexisting_cache_preserved(self) -> None:
        cache: dict[float, str] = {0.99: "old"}
        filter_score([_c(0.7, "keep")], 0.5, [], cache)
        assert cache[0.99] == "old"
        assert cache[0.7] == "eq_keep"


class TestFilterSameFalse:

    def test_repeats_kept_when_dedup_disabled(self) -> None:
        cache: dict[float, str] = {}
        kept = filter_score(
            [_c(0.7, "a"), _c(0.7, "b")], 0.5, [], cache, filter_same=False
        )
        assert len(kept) == 2

    def test_cache_last_wins_on_ties(self) -> None:


        cache: dict[float, str] = {}
        filter_score(
            [_c(0.7, "first"), _c(0.7, "last")],
            0.5,
            [],
            cache,
            filter_same=False,
        )
        assert cache[0.7] == "eq_last"

    def test_still_respects_limit(self) -> None:

        cache: dict[float, str] = {}
        kept = filter_score(
            [_c(0.3, "a"), _c(0.7, "b")], 0.5, [], cache, filter_same=False
        )
        assert [c.expression for c in kept] == ["eq_b"]
