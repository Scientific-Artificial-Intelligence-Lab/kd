
from __future__ import annotations

from torch import Tensor

DEFAULT_MAX_BYTES = 1_073_741_824


class TermColumnCache:

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._columns: dict[tuple[int, str], Tensor] = {}
        self._bytes: int = 0
        self._max_bytes = max_bytes

    def get(self, term: str, generation: int = 0) -> Tensor | None:
        return self._columns.get((generation, term))

    def put(self, term: str, column: Tensor, generation: int = 0) -> None:
        key = (generation, term)
        owned = column.detach().clone()
        new_bytes = owned.numel() * owned.element_size()
        old = self._columns.get(key)
        if old is not None:
            self._bytes -= old.numel() * old.element_size()
        if self._bytes + new_bytes > self._max_bytes:
            self._columns.clear()
            self._bytes = 0
        self._columns[key] = owned
        self._bytes += new_bytes

    def clear(self) -> None:
        self._columns.clear()
        self._bytes = 0

    def __contains__(self, key: str | tuple[int, str]) -> bool:
        if isinstance(key, tuple):
            return key in self._columns
        return (0, key) in self._columns

    def __len__(self) -> int:
        return len(self._columns)
