
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kd.search.discover.controller.tree_state import BatchTracker
from kd.search.discover.core.batch import Batch
from kd.search.discover.tokens.library import Library
from kd.search.discover.tokens.prior import PriorSystem

Int32Array = npt.NDArray[np.int32]
BoolArray = npt.NDArray[np.bool_]

_INVALID_LENGTH_SENTINEL = np.int32(0)




_INITIAL_DANGLING = 1
_DANGLING_COMPLETE = 0


def rebuild_batch(
    actions: Int32Array,
    library: Library,
    prior_system: PriorSystem,
) -> tuple[Batch, BoolArray]:
    action_matrix = np.asarray(actions)
    if action_matrix.ndim != 2:
        raise ValueError("actions must have shape (B, L).")
    if action_matrix.shape[0] == 0:
        raise ValueError("actions must contain at least one row.")
    if action_matrix.shape[1] == 0:
        raise ValueError("actions must contain at least one token.")
    if action_matrix.dtype != np.int32:
        raise ValueError("actions must have dtype int32.")
    if prior_system.library is not library:
        raise ValueError(
            "library must be the exact prior system library instance."
        )
    _validate_token_index_range(action_matrix, library)

    obs = BatchTracker(library).compute_obs(action_matrix)
    priors = prior_system.compute_batch(action_matrix, obs)
    lengths, valid_mask = _compute_lengths_and_validity(action_matrix, library)
    batch = Batch(actions=action_matrix, obs=obs, priors=priors, lengths=lengths)
    return batch, valid_mask


def _validate_token_index_range(
    actions: Int32Array,
    library: Library,
) -> None:
    min_token = int(actions.min())
    max_token = int(actions.max())
    if min_token < 0 or max_token > library.EMPTY_ACTION:
        raise ValueError(
            "token indices must be in "
            f"[0, {library.EMPTY_ACTION}] (including EMPTY_ACTION padding), "
            f"got min={min_token}, max={max_token}"
        )


def _compute_lengths_and_validity(
    actions: Int32Array, library: Library
) -> tuple[Int32Array, BoolArray]:
    n_tokens = len(library.tokens)
    sequence_length = actions.shape[1]




    in_range = (actions >= 0) & (actions < n_tokens)




    first_oor_col = np.where(
        in_range.all(axis=1),
        sequence_length,
        np.argmin(in_range, axis=1),
    ).astype(np.int64)




    safe_actions = np.where(in_range, actions, 0)
    arity_matrix = library.arities[safe_actions].astype(np.int32)
    dangling_matrix = _INITIAL_DANGLING + np.cumsum(
        arity_matrix - 1, axis=1, dtype=np.int32
    )

    cols = np.arange(sequence_length, dtype=np.int64)[None,:]
    done_before_oor = (dangling_matrix == _DANGLING_COMPLETE) & (
        cols < first_oor_col[:, None]
    )

    has_done = done_before_oor.any(axis=1)


    first_done_col = np.argmax(done_before_oor, axis=1)

    lengths = np.where(
        has_done,
        (first_done_col + 1).astype(np.int32),
        _INVALID_LENGTH_SENTINEL,
    ).astype(np.int32)
    valid_mask = has_done.astype(np.bool_)
    return lengths, valid_mask
