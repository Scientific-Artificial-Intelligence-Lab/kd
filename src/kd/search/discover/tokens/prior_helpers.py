
from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from kd.search.discover.tokens.library import Library, TokenType
from kd.search.discover.tokens.prior import (
    _INITIAL_STEP,
    Float32Array,
    Int32Array,
    Prior,
    PriorContext,
    _set_forbidden,
    _zero_adjustment,
)

BoolArray = npt.NDArray[np.bool_]

logger = logging.getLogger(__name__)


def _require_library(owner: Prior, library: Library) -> None:
    if library is not owner.library:
        raise ValueError("constraint library must match prior system library.")


def _extended_arities(library: Library) -> Int32Array:
    return np.append(library.arities, np.int32(0))


def ancestors(
    actions: np.ndarray,
    arities: np.ndarray,
    ancestor_tokens: np.ndarray,
) -> BoolArray:
    action_matrix = np.asarray(actions, dtype=np.int32)
    arity_array = np.asarray(arities, dtype=np.int32)
    watched_tokens = np.asarray(ancestor_tokens, dtype=np.int32)
    if action_matrix.ndim != 2:
        raise ValueError("actions must have shape (B, L).")
    if arity_array.ndim != 1:
        raise ValueError("arities must have shape (n_tokens + 1,).")
    if watched_tokens.ndim != 1:
        raise ValueError("ancestor_tokens must have shape (N,).")

    batch_size, sequence_length = action_matrix.shape
    if sequence_length == 0:
        return np.zeros(batch_size, dtype=np.bool_)


    arity_matrix = arity_array[action_matrix]
    in_ancestor = np.isin(action_matrix, watched_tokens)



    dangling_matrix = np.cumsum(arity_matrix - 1, axis=1, dtype=np.int32)




    threshold_matrix = dangling_matrix - arity_matrix



    suffix_min_inclusive = np.minimum.accumulate(
        dangling_matrix[:, ::-1], axis=1,
    )[:, ::-1]
    open_sentinel = np.iinfo(np.int32).max
    suffix_min_after = np.concatenate(
        [
            suffix_min_inclusive[:, 1:],
            np.full((batch_size, 1), open_sentinel, dtype=np.int32),
        ],
        axis=1,
    )

    still_open = in_ancestor & (suffix_min_after > threshold_matrix)
    return np.asarray(still_open.any(axis=1), dtype=np.bool_)


class DiffChildConstraint(Prior):

    def __init__(self, library: Library) -> None:
        super().__init__(library)
        self._diff_parents = np.asarray(
            library.parent_adjust[library.diff_tokens],
            dtype=np.int32,
        )
        self._allowed_tokens = np.asarray(
            self._build_allowed_tokens(library),
            dtype=np.int32,
        )
        allowed_set = set(int(token) for token in self._allowed_tokens.tolist())
        self._forbidden_tokens = np.asarray(
            [index for index in range(self.n_choices) if index not in allowed_set],
            dtype=np.int32,
        )

    def __call__(self, ctx: PriorContext) -> Float32Array:
        _require_library(self, ctx.library)
        adjustment = _zero_adjustment(ctx.actions.shape[0], self.n_choices)
        mask = np.isin(ctx.parent, self._diff_parents)
        _set_forbidden(adjustment, mask, self._forbidden_tokens)
        return adjustment

    @staticmethod
    def _build_allowed_tokens(library: Library) -> list[int]:
        state_tokens = [
            index
            for index, token in enumerate(library.tokens)
            if token.token_type == TokenType.TERMINAL
        ]
        allowed = set(state_tokens)
        allowed.update(int(token) for token in library.diff_tokens.tolist())
        return sorted(allowed)





_SOFT_EARLY_CUTOFF: int = 3
_SOFT_EARLY_SCALE: float = 10.0



_ADD_CANDIDATE_NAMES: frozenset[str] = frozenset(
    {"add", "add_t", "sub", "sub_t"}
)


class SoftLengthPrior(Prior):

    def __init__(self, library: Library, loc: float, scale: float) -> None:
        super().__init__(library)
        if scale <= 0.0:
            raise ValueError("scale must be positive.")
        self.loc = loc
        self.scale = scale
        self._nonterminal_tokens = np.flatnonzero(
            library.arities >= 1
        ).astype(np.int32)
        self._nonadd_tokens = np.asarray(
            [
                index
                for index, name in enumerate(library.names)
                if name not in _ADD_CANDIDATE_NAMES
            ],
            dtype=np.int32,
        )

    def initial_adjustment(self, batch_size: int) -> Float32Array:
        return self._adjustment_for_step(batch_size, float(_INITIAL_STEP))

    def __call__(self, ctx: PriorContext) -> Float32Array:
        _require_library(self, ctx.library)
        return self._adjustment_for_step(ctx.actions.shape[0], float(ctx.step_idx))

    def _adjustment_for_step(
        self,
        batch_size: int,
        step_idx: float,
    ) -> Float32Array:
        adjustment = _zero_adjustment(batch_size, self.n_choices)
        if step_idx < self.loc:
            if step_idx < _SOFT_EARLY_CUTOFF:
                penalty = np.float32(
                    -((step_idx - _SOFT_EARLY_CUTOFF) ** 2) / _SOFT_EARLY_SCALE
                )
                adjustment[:, self._nonadd_tokens] = penalty


        elif step_idx > self.loc:
            penalty = np.float32(
                -((step_idx - self.loc) ** 2) / (2.0 * self.scale)
            )
            adjustment[:, self._nonterminal_tokens] = penalty
        return adjustment


class TokenBiasPrior(Prior):

    def __init__(
        self,
        library: Library,
        token_names: list[str],
        bias: float,
    ) -> None:
        super().__init__(library)
        self.bias = float(bias)
        resolved: list[int] = []
        for name in token_names:
            try:
                resolved.append(library.name_to_index(name))
            except KeyError:
                logger.warning(
                    "TokenBiasPrior: unknown token name %r — skipping "
                    "(library.names=%r)",
                    name,
                    library.names,
                )
        self._biased_tokens: Int32Array = np.asarray(resolved, dtype=np.int32)

    def initial_adjustment(self, batch_size: int) -> Float32Array:
        return self._adjustment_for_step(batch_size, _INITIAL_STEP)

    def __call__(self, ctx: PriorContext) -> Float32Array:
        _require_library(self, ctx.library)
        return self._adjustment_for_step(ctx.actions.shape[0], ctx.step_idx)

    def _adjustment_for_step(
        self,
        batch_size: int,
        step_idx: int,
    ) -> Float32Array:
        del step_idx
        adjustment = _zero_adjustment(batch_size, self.n_choices)
        if self.bias == 0.0 or self._biased_tokens.size == 0:
            return adjustment
        adjustment[:, self._biased_tokens] = np.float32(self.bias)
        return adjustment


class RepeatConstraint(Prior):

    def __init__(
        self,
        library: Library,
        tokens: npt.NDArray[np.int32],
        max_: int,
    ) -> None:
        super().__init__(library)
        self.target_tokens = np.asarray(tokens, dtype=np.int32)
        if library.EMPTY_ACTION in self.target_tokens:
            raise ValueError(
                "EMPTY_ACTION must not be in target_tokens — it would "
                "count padding positions as real token occurrences."
            )
        self.max_ = max_

    def __call__(self, ctx: PriorContext) -> Float32Array:
        _require_library(self, ctx.library)
        adjustment = _zero_adjustment(ctx.actions.shape[0], self.n_choices)
        if ctx.actions.shape[1] == 0:
            return adjustment
        counts = np.sum(np.isin(ctx.actions, self.target_tokens), axis=1)
        _set_forbidden(adjustment, counts >= self.max_, self.target_tokens)
        return adjustment


class RelationalConstraint(Prior):

    def __init__(
        self,
        library: Library,
        targets: np.ndarray,
        effectors: np.ndarray,
        relationship: str,
    ) -> None:
        super().__init__(library)
        self.targets = np.asarray(targets, dtype=np.int32)
        self.effectors = np.asarray(effectors, dtype=np.int32)
        self.relationship = relationship
        self._extended_arities = _extended_arities(library)

    def __call__(self, ctx: PriorContext) -> Float32Array:
        _require_library(self, ctx.library)
        if self.relationship == "child":
            return self._child(ctx)
        if self.relationship == "descendant":
            return self._descendant(ctx)
        raise NotImplementedError(
            f"Relationship '{self.relationship}' not implemented. "
            "See refs/discover/dso/dso/prior.py:361 for reference."
        )

    def _child(self, ctx: PriorContext) -> Float32Array:
        adjustment = _zero_adjustment(ctx.actions.shape[0], self.n_choices)
        adj_parents = self.library.parent_adjust[self.effectors]
        _set_forbidden(adjustment, np.isin(ctx.parent, adj_parents), self.targets)
        return adjustment

    def _descendant(self, ctx: PriorContext) -> Float32Array:
        adjustment = _zero_adjustment(ctx.actions.shape[0], self.n_choices)
        mask = ancestors(ctx.actions, self._extended_arities, self.effectors)
        _set_forbidden(adjustment, mask, self.targets)
        return adjustment


class TrigConstraint(RelationalConstraint):

    def __init__(self, library: Library, block_diff: bool = True) -> None:
        tokens = (
            np.concatenate([library.trig_tokens, library.diff_tokens])
            if block_diff
            else library.trig_tokens
        )
        super().__init__(
            library,
            targets=tokens,
            effectors=tokens,
            relationship="descendant",
        )


class InverseUnaryConstraint(Prior):

    def __init__(self, library: Library) -> None:
        super().__init__(library)
        self._sub_priors: list[RelationalConstraint] = []
        for target_idx, effector_idx in library.inverse_tokens.items():
            self._sub_priors.append(
                RelationalConstraint(
                    library,
                    targets=np.array([target_idx], dtype=np.int32),
                    effectors=np.array([effector_idx], dtype=np.int32),
                    relationship="child",
                )
            )

    def __call__(self, ctx: PriorContext) -> Float32Array:
        _require_library(self, ctx.library)
        adjustment = _zero_adjustment(ctx.actions.shape[0], self.n_choices)
        for sub_prior in self._sub_priors:
            adjustment += sub_prior(ctx)
        return adjustment


class DiffDescendantConstraint(RelationalConstraint):





    _DEFAULT_FORBIDDEN_NAMES: frozenset[str] = frozenset(
        {"add", "add_t", "sub", "sub_t"}
    )

    def __init__(
        self,
        library: Library,
        extra_forbidden: list[str] | None = None,
    ) -> None:
        forbidden_names = set(self._DEFAULT_FORBIDDEN_NAMES)
        if extra_forbidden is not None:
            forbidden_names.update(extra_forbidden)
        targets = np.asarray(
            [
                index
                for index, name in enumerate(library.names)
                if name in forbidden_names
            ],
            dtype=np.int32,
        )
        super().__init__(
            library,
            targets=targets,
            effectors=library.diff_tokens,
            relationship="descendant",
        )
