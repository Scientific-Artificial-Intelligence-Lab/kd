
from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class PoolItem:

    score: float
    expression: str = field(compare=False)


class ElitePool:

    def __init__(self, k: int) -> None:
        if type(k) is not int or k < 1:
            raise ValueError(f"k must be a positive int, got {k!r}")
        self._k = k
        self._heap: list[PoolItem] = []
        self._scores: list[float] = []

    @classmethod
    def from_state(
        cls,
        k: int,
        items: Sequence[PoolItem],
        scores: Sequence[float],
    ) -> ElitePool:
        pool = cls(k)
        members = list(items)
        if len(members) > k:
            raise ValueError(
                f"cannot restore {len(members)} items into a capacity-{k} pool"
            )
        heapq.heapify(members)
        pool._heap = members
        pool._scores = list(scores)
        return pool

    @property
    def k(self) -> int:
        return self._k

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(self._scores)

    def __len__(self) -> int:
        return len(self._heap)

    def push(self, sample: PoolItem | list[PoolItem]) -> None:
        samples = sample if isinstance(sample, list) else [sample]
        for item in samples:
            if len(self._heap) < self._k:

                heapq.heappush(self._heap, item)
                self._scores.append(item.score)
            elif item.score > self._heap[0].score and item.score not in self._scores:


                heapq.heappushpop(self._heap, item)
                self._scores.append(item.score)

    def get_top_samples(self) -> list[PoolItem]:
        return sorted(self._heap, key=lambda item: item.score, reverse=True)
