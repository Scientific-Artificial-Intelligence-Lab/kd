
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from kd.search.discover.tokens.library import Library

logger = logging.getLogger(__name__)







_BUFFER_CACHE_MAX_ENTRIES = 8

_OBS_DIM = 4
_PREV_ACTION_INDEX = 0
_PARENT_INDEX = 1
_SIBLING_INDEX = 2
_DANGLING_INDEX = 3


_INITIAL_DANGLING = 1




_ARITY_SLOT_CONSUMPTION = 1

Int32Array = npt.NDArray[np.int32]
Float32Array = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class _TrackerConfig:

    arities: Int32Array
    parent_adjust: Int32Array
    empty_action: int
    empty_parent: int
    empty_sibling: int


def parents_siblings(
    tokens: Int32Array,
    arities: Int32Array,
    parent_adjust: Int32Array,
    empty_parent: int,
    empty_sibling: int,
) -> tuple[Int32Array, Int32Array]:
    token_matrix = np.asarray(tokens)
    if token_matrix.ndim != 2:
        raise ValueError("tokens must have shape (B, t).")
    if token_matrix.shape[1] == 0:
        raise ValueError("tokens must contain at least one token.")
    if token_matrix.dtype != np.int32:
        raise ValueError("tokens must have dtype int32.")

    arity_vector = np.asarray(arities)
    parent_vector = np.asarray(parent_adjust)
    if arity_vector.ndim != 1 or parent_vector.ndim != 1:
        raise ValueError("arities and parent_adjust must be 1-D arrays.")
    if arity_vector.dtype != np.int32 or parent_vector.dtype != np.int32:
        raise ValueError("arities and parent_adjust must have dtype int32.")
    required_size = int(token_matrix.max()) + 1
    if arity_vector.shape[0] < required_size or parent_vector.shape[0] < required_size:
        raise ValueError(
            "arities and parent_adjust must include the EMPTY_ACTION sentinel."
        )

    batch_size, prefix_length = token_matrix.shape
    parents = np.full(batch_size, empty_parent, dtype=np.int32)
    siblings = np.full(batch_size, empty_sibling, dtype=np.int32)
    for row in range(batch_size):
        last_token = int(token_matrix[row, -1])
        if arity_vector[last_token] > 0:
            parents[row] = int(parent_vector[last_token])
            continue

        dangling = 0
        for offset in range(prefix_length):
            token_index = prefix_length - offset - 1
            token = int(token_matrix[row, token_index])
            dangling += int(arity_vector[token]) - _ARITY_SLOT_CONSUMPTION
            if dangling == 0:
                parents[row] = int(parent_vector[token])
                siblings[row] = int(token_matrix[row, token_index + 1])
                break
    return parents, siblings


class IncrementalTracker:

    def __init__(
        self,
        library: Library,
        max_length: int | None = None,
    ) -> None:
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be >= 1 when provided.")
        self._config = _build_config(library)
        self._max_length = max_length
        self._history: Int32Array | None = None
        self._buffer: Int32Array | None = None
        self._buffer_cache: OrderedDict[int, Int32Array] = OrderedDict()
        self._buffer_cache_warned = False
        self._dangling: Int32Array | None = None
        self._step_idx = 0

    def reset(self, batch_size: int) -> Float32Array:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self._step_idx = 0
        if self._max_length is None:
            self._history = np.empty((batch_size, 0), dtype=np.int32)
        else:
            self._reset_buffer(batch_size)
        self._dangling = np.full(batch_size, _INITIAL_DANGLING, dtype=np.int32)
        return _initial_obs(batch_size, self._config)

    @property
    def history(self) -> Int32Array:
        if self._max_length is None:
            if self._history is None:
                raise RuntimeError(
                    "IncrementalTracker not initialized; call reset() first"
                )
            return self._history

        buffer = _require_array(
            self._buffer,
            "IncrementalTracker not initialized; call reset() first",
        )
        return buffer[:,: self._step_idx]

    def step(self, action: Int32Array) -> Float32Array:
        dangling = _require_array(
            self._dangling, "reset() must be called before step()."
        )
        action_vector = _as_action_vector(
            action,
            expected_batch_size=self.history.shape[0],
        )
        current_history = self._append_action(action_vector)
        self._dangling = _update_dangling(dangling, action_vector, self._config.arities)
        parent, sibling = parents_siblings(
            current_history,
            self._config.arities,
            self._config.parent_adjust,
            self._config.empty_parent,
            self._config.empty_sibling,
        )
        return _stack_obs(action_vector, parent, sibling, self._dangling)

    def _reset_buffer(self, batch_size: int) -> None:
        assert self._max_length is not None
        expected_shape = (batch_size, self._max_length)
        buffer = self._buffer_cache.get(batch_size)
        if buffer is None or buffer.shape != expected_shape:
            buffer = np.empty(expected_shape, dtype=np.int32)
            self._buffer_cache[batch_size] = buffer




            while len(self._buffer_cache) > _BUFFER_CACHE_MAX_ENTRIES:
                self._buffer_cache.popitem(last=False)
                if not self._buffer_cache_warned:
                    logger.warning(
                        "IncrementalTracker._buffer_cache exceeded %d "
                        "distinct batch sizes; oldest entry evicted. "
                        "Consider building a fresh tracker per sample "
                        "instead of reusing across many batch sizes.",
                        _BUFFER_CACHE_MAX_ENTRIES,
                    )
                    self._buffer_cache_warned = True
        else:

            self._buffer_cache.move_to_end(batch_size)
        self._buffer = buffer
        self._buffer.fill(self._config.empty_action)
        self._step_idx = 0

    def _append_action(self, action_vector: Int32Array) -> Int32Array:
        if self._max_length is None:
            history = _require_array(
                self._history,
                "reset() must be called before step().",
            )
            self._history = np.concatenate((history, action_vector[:, None]), axis=1)
            return self._history

        buffer = _require_array(
            self._buffer,
            "reset() must be called before step().",
        )
        if self._step_idx >= self._max_length:
            raise RuntimeError(
                f"IncrementalTracker exceeded configured max_length={self._max_length}."
            )
        buffer[:, self._step_idx] = action_vector
        self._step_idx += 1
        return buffer[:,: self._step_idx]


class BatchTracker:

    def __init__(self, library: Library) -> None:
        self._config = _build_config(library)

    def compute_obs(self, actions: Int32Array) -> Float32Array:
        tokens = _as_action_matrix(actions)
        batch_size, sequence_length = tokens.shape
        prev_action = np.full(
            (batch_size, sequence_length),
            self._config.empty_action,
            dtype=np.int32,
        )
        if sequence_length > 1:
            prev_action[:, 1:] = tokens[:, :-1]
        parent, sibling = _compute_parent_sibling_batch(tokens, self._config)
        dangling = _compute_dangling(tokens, self._config.arities)
        return np.stack(
            [prev_action, parent, sibling, dangling],
            axis=1,
        ).astype(np.float32)


def _compute_parent_sibling_batch(
    tokens: Int32Array, config: _TrackerConfig,
) -> tuple[Int32Array, Int32Array]:
    batch_size, sequence_length = tokens.shape
    parent_matrix = np.full(
        (batch_size, sequence_length),
        config.empty_parent,
        dtype=np.int32,
    )
    sibling_matrix = np.full(
        (batch_size, sequence_length),
        config.empty_sibling,
        dtype=np.int32,
    )
    if sequence_length < 2:
        return parent_matrix, sibling_matrix

    arity_matrix = config.arities[tokens]
    cumsum_matrix = np.cumsum(arity_matrix - 1, axis=1, dtype=np.int32)



    arity_lists = arity_matrix.tolist()
    cumsum_lists = cumsum_matrix.tolist()
    token_lists = tokens.tolist()
    parent_adjust_list = config.parent_adjust.tolist()


    parent_views = [parent_matrix[row] for row in range(batch_size)]
    sibling_views = [sibling_matrix[row] for row in range(batch_size)]

    for row in range(batch_size):
        arities_row = arity_lists[row]
        cumsum_row = cumsum_lists[row]
        tokens_row = token_lists[row]
        parent_row = parent_views[row]
        sibling_row = sibling_views[row]



        last_seen: dict[int, int] = {0: 0}
        for col in range(1, sequence_length):
            prev_arity = arities_row[col - 1]
            v = cumsum_row[col - 1]
            if prev_arity > 0:

                parent_row[col] = parent_adjust_list[tokens_row[col - 1]]

            else:
                parent_pos = last_seen.get(v)
                if parent_pos is not None:
                    parent_row[col] = parent_adjust_list[tokens_row[parent_pos]]
                    sibling_row[col] = tokens_row[parent_pos + 1]

            last_seen[v] = col

    return parent_matrix, sibling_matrix


def _build_config(library: Library) -> _TrackerConfig:
    arities = np.append(library.arities, np.int32(0))
    parent_adjust = np.append(library.parent_adjust, np.int32(library.EMPTY_PARENT))
    return _TrackerConfig(
        arities=arities,
        parent_adjust=parent_adjust,
        empty_action=library.EMPTY_ACTION,
        empty_parent=library.EMPTY_PARENT,
        empty_sibling=library.EMPTY_SIBLING,
    )


def _initial_obs(batch_size: int, config: _TrackerConfig) -> Float32Array:
    obs = np.empty((batch_size, _OBS_DIM), dtype=np.float32)
    obs[:, _PREV_ACTION_INDEX] = config.empty_action
    obs[:, _PARENT_INDEX] = config.empty_parent
    obs[:, _SIBLING_INDEX] = config.empty_sibling
    obs[:, _DANGLING_INDEX] = _INITIAL_DANGLING
    return obs


def _require_array(array: Int32Array | None, message: str) -> Int32Array:
    if array is None:
        raise RuntimeError(message)
    return array


def _as_action_vector(
    action: Int32Array,
    expected_batch_size: int,
) -> Int32Array:
    action_vector = np.asarray(action, dtype=np.int32)
    if action_vector.ndim != 1:
        raise ValueError("action must have shape (B,).")
    if action_vector.shape[0] != expected_batch_size:
        raise ValueError("action batch size does not match tracker state.")
    return action_vector


def _as_action_matrix(actions: Int32Array) -> Int32Array:
    action_matrix = np.asarray(actions, dtype=np.int32)
    if action_matrix.ndim != 2:
        raise ValueError("actions must have shape (B, L).")
    if action_matrix.shape[1] == 0:
        raise ValueError("actions must contain at least one token.")
    return action_matrix


def _update_dangling(
    dangling: Int32Array,
    action: Int32Array,
    arities: Int32Array,
) -> Int32Array:
    return dangling + arities[action] - _ARITY_SLOT_CONSUMPTION


def _compute_dangling(tokens: Int32Array, arities: Int32Array) -> Int32Array:
    batch_size, sequence_length = tokens.shape
    dangling = np.full(
        (batch_size, sequence_length), _INITIAL_DANGLING, dtype=np.int32
    )
    for position in range(1, sequence_length):
        dangling[:, position] = _update_dangling(
            dangling[:, position - 1],
            tokens[:, position - 1],
            arities,
        )
    return dangling


def _stack_obs(
    prev_action: Int32Array,
    parent: Int32Array,
    sibling: Int32Array,
    dangling: Int32Array,
) -> Float32Array:
    return np.stack(
        [prev_action, parent, sibling, dangling],
        axis=1,
    ).astype(np.float32)
