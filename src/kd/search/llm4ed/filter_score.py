
from __future__ import annotations

from collections.abc import MutableMapping, Sequence

from kd.search.llm4ed.pool import PoolItem


def filter_score(
    candidates: Sequence[PoolItem],
    reward_limit: float,
    seen_scores: Sequence[float],
    cache: MutableMapping[float, str],
    *,
    filter_same: bool = True,
) -> list[PoolItem]:
    kept: list[PoolItem] = []
    running_scores = set(seen_scores)

    for candidate in candidates:
        if candidate.score > reward_limit:
            if candidate.score in running_scores and filter_same:
                continue
            running_scores.add(candidate.score)
            cache[candidate.score] = candidate.expression
            kept.append(candidate)

    return kept
