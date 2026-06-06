
from __future__ import annotations

import numpy as np

from kd.search.discover.core.tree import trim_to_natural
from kd.search.discover.ir.conversion import tokens_to_ir
from kd.search.discover.tokens.library import Library


class Deduplicator:

    def __init__(self, library: Library) -> None:
        self._library = library

    def deduplicate(
        self, actions: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        actions = np.asarray(actions, dtype=np.int32)
        if actions.ndim != 2:
            raise ValueError(
                f"actions must be 2D (B, L), got ndim={actions.ndim}"
            )

        batch_size = actions.shape[0]

        if batch_size == 0:
            return [], np.array([], dtype=np.int64)

        ir_to_idx: dict[str, int] = {}
        unique_ir_list: list[str] = []
        scatter_map = np.empty(batch_size, dtype=np.int64)

        for i in range(batch_size):
            try:
                trimmed = trim_to_natural(actions[i], self._library)
            except ValueError:
                raise ValueError(
                    f"Incomplete expression at row {i} — "
                    "upstream CandidateValidator should have filtered this."
                ) from None
            ir = tokens_to_ir(trimmed.tolist(), self._library)

            if ir not in ir_to_idx:
                ir_to_idx[ir] = len(unique_ir_list)
                unique_ir_list.append(ir)

            scatter_map[i] = ir_to_idx[ir]

        return unique_ir_list, scatter_map

    @staticmethod
    def scatter_rewards(
        unique_rewards: np.ndarray, scatter_map: np.ndarray
    ) -> np.ndarray:
        result: np.ndarray = unique_rewards[scatter_map]
        return result
