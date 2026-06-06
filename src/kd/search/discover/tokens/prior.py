
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from kd.search.discover.tokens.library import Library

if TYPE_CHECKING:
    from kd.search.discover.tokens.scaffold_prior import ScaffoldPrior

_OBS_CHANNELS = 4
_PARENT_INDEX = 1
_SIBLING_INDEX = 2
_DANGLING_INDEX = 3
_INITIAL_STEP = 0
_DANGLING_ONE = 1
_ALLOW_LOGIT = np.float32(0.0)
_FORBID_LOGIT = np.float32(-np.inf)

Float32Array = npt.NDArray[np.float32]
Int32Array = npt.NDArray[np.int32]


def _zero_adjustment(batch_size: int, n_choices: int) -> Float32Array:
    return np.zeros((batch_size, n_choices), dtype=np.float32)


def _as_actions_history(actions: np.ndarray) -> Int32Array:
    action_matrix = np.asarray(actions, dtype=np.int32)
    if action_matrix.ndim != 2:
        raise ValueError("actions must have shape (B, t).")
    return action_matrix


def _as_step_obs(obs: np.ndarray) -> Float32Array:
    step_obs = np.asarray(obs, dtype=np.float32)
    if step_obs.ndim != 2 or step_obs.shape[1] != _OBS_CHANNELS:
        raise ValueError("obs must have shape (B, 4).")
    return step_obs


def _as_batch_inputs(
    actions: np.ndarray,
    obs: np.ndarray,
) -> tuple[Int32Array, Float32Array]:
    action_matrix = _as_actions_history(actions)
    batch_obs = np.asarray(obs, dtype=np.float32)
    if batch_obs.ndim != 3 or batch_obs.shape[1] != _OBS_CHANNELS:
        raise ValueError("obs must have shape (B, 4, L).")
    if batch_obs.shape[0] != action_matrix.shape[0]:
        raise ValueError("obs batch size must match actions.")
    if batch_obs.shape[2] != action_matrix.shape[1]:
        raise ValueError("obs length must match actions.")
    return action_matrix, batch_obs


def _obs_parent(obs: Float32Array) -> Int32Array:
    return np.asarray(obs[:, _PARENT_INDEX], dtype=np.int32)


def _obs_sibling(obs: Float32Array) -> Int32Array:
    return np.asarray(obs[:, _SIBLING_INDEX], dtype=np.int32)


def _obs_dangling(obs: Float32Array) -> Int32Array:
    return np.asarray(obs[:, _DANGLING_INDEX], dtype=np.int32)


def _set_forbidden(
    adjustment: Float32Array,
    row_mask: npt.NDArray[np.bool_],
    token_indices: Int32Array,
) -> None:
    if not np.any(row_mask) or token_indices.size == 0:
        return
    adjustment[np.ix_(row_mask, token_indices)] = _FORBID_LOGIT


@dataclass(frozen=True, slots=True)
class PriorContext:

    actions: Int32Array
    parent: Int32Array
    sibling: Int32Array
    dangling: Int32Array
    step_idx: int
    library: Library

    def __post_init__(self) -> None:
        if self.actions.ndim != 2:
            raise ValueError("actions must have shape (B, t).")
        if self.parent.ndim != 1 or self.sibling.ndim != 1 or self.dangling.ndim != 1:
            raise ValueError("parent, sibling, and dangling must have shape (B,).")
        if self.actions.shape[1] != self.step_idx:
            raise ValueError(
                f"actions.shape[1]={self.actions.shape[1]} must equal "
                f"step_idx={self.step_idx}"
            )
        batch_size = self.actions.shape[0]
        if self.parent.shape[0] != batch_size:
            raise ValueError(
                "batch size mismatch: "
                f"actions={batch_size}, parent={self.parent.shape[0]}, "
                f"sibling={self.sibling.shape[0]}, dangling={self.dangling.shape[0]}"
            )
        if self.sibling.shape[0] != batch_size or self.dangling.shape[0] != batch_size:
            raise ValueError(
                "batch size mismatch: "
                f"actions={batch_size}, parent={self.parent.shape[0]}, "
                f"sibling={self.sibling.shape[0]}, dangling={self.dangling.shape[0]}"
            )


class Prior(ABC):

    def __init__(self, library: Library) -> None:
        self.library = library
        self.n_choices = len(library.tokens)

    def initial_adjustment(self, batch_size: int) -> Float32Array:
        return _zero_adjustment(batch_size, self.n_choices)

    @abstractmethod
    def __call__(self, ctx: PriorContext) -> Float32Array:
        pass


class LengthConstraint(Prior):

    def __init__(
        self,
        library: Library,
        min_: int | None,
        max_: int | None,
    ) -> None:
        super().__init__(library)
        if min_ is None and max_ is None:
            raise ValueError("At least one of min_ or max_ must be set.")
        self.min_ = min_
        self.max_ = max_



        arities = library.arities
        self._tokens_by_arity: list[tuple[int, Int32Array]] = [
            (int(arity), np.flatnonzero(arities == arity).astype(np.int32))
            for arity in np.unique(arities[arities >= 1])
        ]

    def initial_adjustment(self, batch_size: int) -> Float32Array:
        adjustment = super().initial_adjustment(batch_size)
        adjustment[:, self.library.terminal_tokens] = _FORBID_LOGIT
        return adjustment

    def __call__(self, ctx: PriorContext) -> Float32Array:
        self._require_library(ctx.library)
        if ctx.step_idx == _INITIAL_STEP:
            return self.initial_adjustment(ctx.actions.shape[0])
        adjustment = _zero_adjustment(ctx.actions.shape[0], self.n_choices)
        self._apply_max(adjustment, ctx.dangling, ctx.step_idx)
        self._apply_min(adjustment, ctx.dangling, ctx.step_idx)
        return adjustment

    def _apply_max(
        self,
        adjustment: Float32Array,
        dangling: Int32Array,
        step_idx: int,
    ) -> None:
        if self.max_ is None:
            return
        current_time = step_idx - 1
        if (current_time + 2) < self.max_ // 2:
            return
        remaining = self.max_ - step_idx



        for arity_int, tokens_of_arity in self._tokens_by_arity:
            threshold = remaining - (arity_int - 1)
            forbidden = (
                dangling == threshold
                if arity_int == 1
                else dangling >= threshold
            )
            _set_forbidden(
                adjustment,
                forbidden,
                tokens_of_arity,
            )

    def _apply_min(
        self,
        adjustment: Float32Array,
        dangling: Int32Array,
        step_idx: int,
    ) -> None:
        if self.min_ is None or (step_idx + 1) >= self.min_:
            return
        _set_forbidden(
            adjustment,
            dangling == _DANGLING_ONE,
            self.library.terminal_tokens,
        )

    def _require_library(self, library: Library) -> None:
        if library is not self.library:
            raise ValueError("constraint library must match prior system library.")

class PriorSystem:

    def __init__(self, library: Library, priors: list[Prior]) -> None:
        self.library = library
        self.priors = priors
        self.n_choices = len(library.tokens)
        for prior in priors:
            if prior.library is not library:
                raise ValueError("all priors must share the same library instance.")

    def step(
        self,
        actions: np.ndarray,
        obs: np.ndarray,
        step_idx: int,
    ) -> Float32Array:
        if step_idx < 0:
            raise ValueError("step_idx must be non-negative.")
        action_matrix = _as_actions_history(actions)
        step_obs = _as_step_obs(obs)
        if action_matrix.shape[0] != step_obs.shape[0]:
            raise ValueError(
                "batch size mismatch: "
                f"actions={action_matrix.shape[0]}, obs={step_obs.shape[0]}"
            )
        if action_matrix.shape[1] != step_idx:
            raise ValueError(
                "step_idx="
                f"{step_idx} must equal actions.shape[1]={action_matrix.shape[1]}"
            )
        return self._combine(self._build_context(action_matrix, step_obs, step_idx))

    def compute_batch(self, actions: np.ndarray, obs: np.ndarray) -> Float32Array:
        action_matrix, batch_obs = _as_batch_inputs(actions, obs)
        batch_size, sequence_length = action_matrix.shape
        priors = np.empty(
            (batch_size, sequence_length, self.n_choices),
            dtype=np.float32,
        )
        for step_idx in range(sequence_length):
            ctx = self._build_context(
                action_matrix[:, :step_idx],
                batch_obs[:, :, step_idx],
                step_idx,
            )
            priors[:, step_idx,:] = self._combine(ctx)
        return priors

    def _build_context(
        self,
        actions: Int32Array,
        obs: Float32Array,
        step_idx: int,
    ) -> PriorContext:
        return PriorContext(
            actions=actions,
            parent=_obs_parent(obs),
            sibling=_obs_sibling(obs),
            dangling=_obs_dangling(obs),
            step_idx=step_idx,
            library=self.library,
        )

    def _combine(self, ctx: PriorContext) -> Float32Array:
        batch_size = ctx.actions.shape[0]
        adjustment = _zero_adjustment(batch_size, self.n_choices)
        for prior in self.priors:
            adjustment += self._prior_adjustment(prior, ctx)

        live_rows = ctx.dangling > 0
        if np.any(live_rows):
            live_all_forbidden = np.all(
                np.isneginf(adjustment[live_rows]), axis=1,
            )
            if np.any(live_all_forbidden):
                live_indices = np.where(live_rows)[0]
                dead_rows = live_indices[live_all_forbidden].tolist()
                raise ValueError(
                    "dead-end row(s) in prior system: all tokens "
                    f"forbidden at rows {dead_rows}"
                )





        if not np.all(live_rows):
            adjustment[~live_rows] = _ALLOW_LOGIT

        return adjustment.astype(np.float32, copy=False)

    def _prior_adjustment(self, prior: Prior, ctx: PriorContext) -> Float32Array:
        if ctx.step_idx == _INITIAL_STEP:
            return prior.initial_adjustment(ctx.actions.shape[0])
        return prior(ctx)




from kd.search.discover.tokens.prior_helpers import (
    DiffChildConstraint,
    DiffDescendantConstraint,
    InverseUnaryConstraint,
    RelationalConstraint,
    RepeatConstraint,
    SoftLengthPrior,
    TokenBiasPrior,
    TrigConstraint,
    ancestors,
)

__all__ = [
    "ancestors",
    "DiffChildConstraint",
    "DiffDescendantConstraint",
    "InverseUnaryConstraint",
    "LengthConstraint",
    "Prior",
    "PriorContext",
    "PriorSystem",
    "RelationalConstraint",
    "RepeatConstraint",
    "ScaffoldPrior",
    "SoftLengthPrior",
    "TokenBiasPrior",
    "TrigConstraint",
]


def __getattr__(name: str) -> Any:
    if name == "ScaffoldPrior":
        from kd.search.discover.tokens.scaffold_prior import ScaffoldPrior

        return ScaffoldPrior
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
