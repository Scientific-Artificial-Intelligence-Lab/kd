
from __future__ import annotations

from kd.search.eqgpt.pool import dedup_sentence, merge_top_k









def test_dedup_drops_duplicate_term_slice() -> None:

    assert dedup_sentence([6, 2, 8, 2, 8, 1]) == [6, 2, 8, 1]


def test_dedup_is_commutative_within_a_product() -> None:

    assert dedup_sentence([7, 3, 8, 2, 8, 3, 7, 1]) == [7, 3, 8, 1]


def test_dedup_leaves_distinct_terms_untouched() -> None:
    wave = [6, 2, 20, 2, 54, 2, 8, 1]
    assert dedup_sentence(wave) == wave


def test_dedup_operator_ending_reappends_e() -> None:




    assert dedup_sentence([6, 2, 8, 2]) == [6, 2, 8, 2, 1]







def test_merge_empty_pool_sorts_descending() -> None:
    rewards, sentences = merge_top_k([], [], [0.5, 0.9, 0.7], [[1], [2], [3]], k=10)
    assert rewards == [0.9, 0.7, 0.5]
    assert sentences == [[2], [3], [1]]


def test_merge_deduplicates_by_reward_value() -> None:
    rewards, _ = merge_top_k([], [], [0.9, 0.9, 0.5], [[1], [2], [3]], k=10)
    assert rewards == [0.9, 0.5]


def test_merge_inserts_and_pops_last() -> None:
    rewards, sentences = merge_top_k([0.9, 0.5], [[1], [2]], [0.7], [[3]], k=2)
    assert rewards == [0.9, 0.7]
    assert sentences == [[1], [3]]


def test_merge_skips_reward_value_already_present() -> None:
    rewards, sentences = merge_top_k([0.9], [[1]], [0.9], [[2]], k=10)
    assert rewards == [0.9]
    assert sentences == [[1]]


def test_merge_rejects_candidate_worse_than_all_when_full() -> None:
    rewards, sentences = merge_top_k([0.9, 0.7], [[1], [2]], [0.3], [[3]], k=2)
    assert rewards == [0.9, 0.7]
    assert sentences == [[1], [2]]


def test_merge_multiple_candidates_descending() -> None:
    rewards, sentences = merge_top_k(
        [0.9, 0.5, 0.4], [[1], [2], [3]], [0.8, 0.6], [[4], [5]], k=3
    )
    assert rewards == [0.9, 0.8, 0.6]
    assert sentences == [[1], [4], [5]]












def test_merge_underfilled_pool_rejects_candidates_that_beat_nothing() -> None:
    rewards, sentences = merge_top_k([0.5], [[7]], [0.3, 0.2], [[8], [20]], k=10)
    assert rewards == [0.5]
    assert sentences == [[7]]
    assert len(rewards) == 1


def test_merge_underfilled_pool_beating_entry_stays_same_length() -> None:
    rewards, sentences = merge_top_k([0.5], [[7]], [0.7], [[8]], k=10)
    assert rewards == [0.7]
    assert sentences == [[8]]
    assert len(rewards) == 1
