from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from kd.search.discover.tokens.library import Library
from kd.search.discover.tokens.prior import (
    Float32Array,
    Int32Array,
    Prior,
    PriorContext,
)

__all__ = ["ScaffoldPrior"]

logger = logging.getLogger(__name__)



BoolArray = npt.NDArray[np.bool_]

_SCAFFOLD_CYCLE_IDX = 0
_FORBID_LOGIT = np.float32(-np.inf)
_ROOT_STEP_IDX = 0









_WARNED_UNKNOWN: set[tuple[frozenset[str], str, str]] = set()







_LEFT_SUBTREE_INITIAL_OPEN = 1







_PAD_ARITY = 0


class ScaffoldPrior(Prior):

    def __init__(
        self,
        library: Library,
        diffusion_tokens: list[str],
        reaction_tokens: list[str],
        root_tokens: list[str] | tuple[str, ...] = ("add", "sub"),
        neutral_tokens: list[str] | tuple[str, ...] = (),
    ) -> None:
        super().__init__(library)
        self._diffusion_indices: Int32Array = self._resolve_token_names(
            diffusion_tokens, field_name="diffusion_tokens",
        )
        self._reaction_indices: Int32Array = self._resolve_token_names(
            reaction_tokens, field_name="reaction_tokens",
        )
        root_names = list(root_tokens)
        self._root_indices: Int32Array = self._resolve_token_names(
            root_names, field_name="root_tokens",
        )




        if len(root_names) > 0 and self._root_indices.size == 0:
            raise ValueError(
                "ScaffoldPrior: no root tokens resolved against library. "
                "Requested root_tokens="
                f"{root_names!r}; library.names={self.library.names!r}. "
                "Fix the token names or drop the prior (default off).",
            )



        self._neutral_indices: Int32Array = self._resolve_token_names(
            list(neutral_tokens), field_name="neutral_tokens",
        )
        self._validate_neutral_disjoint(
            diffusion_tokens=diffusion_tokens,
            reaction_tokens=reaction_tokens,
            root_names=root_names,
            neutral_names=list(neutral_tokens),
        )



        if (
            self._diffusion_indices.size == 0
            and self._reaction_indices.size == 0
        ):
            logger.warning(
                "ScaffoldPrior: both diffusion_tokens and reaction_tokens "
                "resolved to empty index sets — subtree disjointness will "
                "be inert (only step-0 root constraint remains active). "
                "Supplied diffusion=%r reaction=%r against library.names=%r.",
                diffusion_tokens, reaction_tokens, self.library.names,
            )

        self._non_root_indices: Int32Array = np.setdiff1d(
            np.arange(self.n_choices, dtype=np.int32),
            self._root_indices,
            assume_unique=True,
        )




        self._reaction_forbid_on_left: Int32Array = np.setdiff1d(
            self._reaction_indices, self._neutral_indices, assume_unique=False,
        ).astype(np.int32)
        self._diffusion_forbid_on_right: Int32Array = np.setdiff1d(
            self._diffusion_indices, self._neutral_indices, assume_unique=False,
        ).astype(np.int32)




        arities_copy = np.array(library.arities, dtype=np.int32, copy=True)
        self._arities_with_pad: Int32Array = np.append(
            arities_copy, np.int32(_PAD_ARITY),
        )
        self._active: bool = False

    @classmethod
    def reset_warn_cache(cls) -> None:
        _WARNED_UNKNOWN.clear()

    def _resolve_token_names(
        self,
        names: list[str] | tuple[str, ...],
        *,
        field_name: str,
    ) -> Int32Array:
        resolved: list[int] = []
        library_vocab = frozenset(self.library.names)
        for name in names:
            try:
                resolved.append(self.library.name_to_index(name))
            except KeyError:
                key = (library_vocab, name, field_name)
                if key not in _WARNED_UNKNOWN:
                    _WARNED_UNKNOWN.add(key)
                    logger.warning(
                        "ScaffoldPrior: unknown token name %r in %s — skipping "
                        "(library.names=%r)",
                        name,
                        field_name,
                        self.library.names,
                    )
        return np.asarray(resolved, dtype=np.int32)

    def _validate_neutral_disjoint(
        self,
        *,
        diffusion_tokens: list[str] | tuple[str, ...],
        reaction_tokens: list[str] | tuple[str, ...],
        root_names: list[str] | tuple[str, ...],
        neutral_names: list[str] | tuple[str, ...],
    ) -> None:
        if self._neutral_indices.size == 0:
            return
        neutral_set = set(self._neutral_indices.tolist())
        overlap_root = neutral_set & set(self._root_indices.tolist())
        if overlap_root:
            raise ValueError(
                "ScaffoldPrior: neutral_tokens overlaps root_tokens. A "
                "neutral token must not appear in the root set — "
                "neutral overrides the left/right forbids only, not the "
                "step-0 root-only mask. "
                f"neutral_tokens={list(neutral_names)!r}, "
                f"root_tokens={list(root_names)!r}, "
                f"overlap={sorted(overlap_root)}."
            )




        _ = diffusion_tokens, reaction_tokens

    def on_cycle_start(self, cycle_idx: int) -> None:
        if isinstance(cycle_idx, bool):
            raise TypeError(
                "cycle_idx must be an int, not bool "
                "(True would coerce to 1 and silently deactivate).",
            )
        if not isinstance(cycle_idx, (int, np.integer)):
            raise TypeError(
                f"cycle_idx must be an int or numpy integer, "
                f"got {type(cycle_idx).__name__}",
            )
        if int(cycle_idx) < 0:
            raise ValueError(
                f"cycle_idx must be non-negative, got {int(cycle_idx)}",
            )
        self._active = int(cycle_idx) == _SCAFFOLD_CYCLE_IDX

    def initial_adjustment(self, batch_size: int) -> Float32Array:
        if not self._active:
            return super().initial_adjustment(batch_size)
        adjustment = np.zeros((batch_size, self.n_choices), dtype=np.float32)
        if self._non_root_indices.size > 0:
            adjustment[:, self._non_root_indices] = _FORBID_LOGIT
        return adjustment

    def __call__(self, ctx: PriorContext) -> Float32Array:
        self._require_library(ctx.library)
        batch_size = ctx.actions.shape[0]
        if not self._active:
            return np.zeros((batch_size, self.n_choices), dtype=np.float32)
        if ctx.step_idx == _ROOT_STEP_IDX:
            return self.initial_adjustment(batch_size)
        return self._subtree_adjustment(ctx)

    def _subtree_adjustment(self, ctx: PriorContext) -> Float32Array:
        batch_size = ctx.actions.shape[0]
        adjustment = np.zeros((batch_size, self.n_choices), dtype=np.float32)
        in_left = self._in_left_subtree(ctx.actions)
        in_right = ~in_left
        if np.any(in_left) and self._reaction_forbid_on_left.size > 0:
            adjustment[
                np.ix_(in_left, self._reaction_forbid_on_left)
            ] = _FORBID_LOGIT
        if np.any(in_right) and self._diffusion_forbid_on_right.size > 0:
            adjustment[
                np.ix_(in_right, self._diffusion_forbid_on_right)
            ] = _FORBID_LOGIT
        return adjustment

    def _in_left_subtree(self, actions: Int32Array) -> BoolArray:
        batch_size = actions.shape[0]
        if actions.shape[1] <= 1:


            return np.ones(batch_size, dtype=np.bool_)
        post_root = actions[:, 1:]


        safe_indices = np.where(
            post_root >= self.n_choices,
            np.int32(self.n_choices),
            post_root,
        )
        arity_matrix = self._arities_with_pad[safe_indices]

        running = _LEFT_SUBTREE_INITIAL_OPEN + np.cumsum(
            arity_matrix - 1, axis=1, dtype=np.int32,
        )


        return np.asarray(np.all(running > 0, axis=1), dtype=np.bool_)

    def _require_library(self, library: Library) -> None:
        if library is not self.library:
            raise ValueError(
                "ScaffoldPrior library must match prior-system library.",
            )
