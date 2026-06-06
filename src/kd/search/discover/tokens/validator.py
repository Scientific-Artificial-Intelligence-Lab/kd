
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kd.search.discover.core.tree import max_diff_order, natural_length
from kd.search.discover.tokens.library import Library

BoolArray = npt.NDArray[np.bool_]

_TRIVIAL_LENGTH = 1
_INCOMPLETE = -1


class CandidateValidator:

    def __init__(
        self,
        library: Library,
        max_length: int,
        max_diff_order: int | None = None,
        min_length: int | None = None,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be positive.")
        if min_length is not None and min_length < 1:
            raise ValueError("min_length must be positive when set.")
        if min_length is not None and min_length > max_length:
            raise ValueError("min_length must not exceed max_length.")
        self.library = library
        self.max_length = max_length
        self.min_length = min_length
        self._max_diff_order = max_diff_order
        self._padding_token = library.EMPTY_ACTION

    def validate(self, tokens: np.ndarray) -> BoolArray:
        token_matrix = np.asarray(tokens, dtype=np.int32)
        if token_matrix.ndim != 2:
            raise ValueError("tokens must have shape (B, L).")
        return np.asarray(
            [self.validate_single(row) for row in token_matrix],
            dtype=np.bool_,
        )

    def validate_single(self, tokens: np.ndarray) -> bool:
        token_row = self._strip_right_padding(np.asarray(tokens, dtype=np.int32))
        if token_row.ndim != 1 or token_row.size == 0:
            return False
        nat_len = natural_length(token_row, self.library)
        if nat_len == _INCOMPLETE:
            return False
        if nat_len > self.max_length:
            return False



        if self.min_length is not None and nat_len < self.min_length:
            return False
        if self._max_diff_order is not None:
            expr_tokens = token_row[:nat_len]
            if max_diff_order(expr_tokens, self.library) > self._max_diff_order:
                return False
        return nat_len > _TRIVIAL_LENGTH

    def _strip_right_padding(
        self, tokens: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        if tokens.ndim != 1:
            return tokens
        non_padding = np.flatnonzero(tokens != self._padding_token)
        if non_padding.size == 0:
            return np.empty(0, dtype=np.int32)
        return np.asarray(tokens[: non_padding[-1] + 1], dtype=np.int32)
