from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from kd.search.discover.controller.lstm import LSTMController
from kd.search.discover.controller.tree_state import BatchTracker, IncrementalTracker
from kd.search.discover.core.tree import max_diff_order
from kd.search.discover.tokens.library import Library, LibraryConfig, Token, TokenType
from kd.search.discover.tokens.prior import (
    DiffChildConstraint,
    DiffDescendantConstraint,
    InverseUnaryConstraint,
    LengthConstraint,
    Prior,
    PriorContext,
    PriorSystem,
    RelationalConstraint,
    RepeatConstraint,
    SoftLengthPrior,
    TrigConstraint,
    ancestors,
)



BURGERS_CONFIG = LibraryConfig(
    coord_vars=["x", "t"],
    state_vars=["u"],
    operators=[
        "add", "mul", "sin", "cos", "diff_x", "diff_t", "n2", "neg", "div",
    ],
)


SOFT_LOC = 6.0
SOFT_SCALE = 2.0
LAP_REACHABILITY_MAX_LENGTH = 32
LAP_SAMPLING_BATCH_SIZE = 1000
LAP_SAMPLING_SEED = 8401
LAP_BINARY_CHILDREN = frozenset({"add", "sub", "mul"})


@pytest.fixture
def lib() -> Library:
    return Library.from_config(BURGERS_CONFIG)


@pytest.fixture
def ps(lib: Library) -> PriorSystem:
    return PriorSystem(
        library=lib,
        priors=[
            LengthConstraint(lib, min_=3, max_=10),
            DiffChildConstraint(lib),
        ],
    )


def _initial_obs(lib: Library, batch_size: int = 1) -> np.ndarray:
    obs = np.empty((batch_size, 4), dtype=np.float32)
    obs[:, 0] = lib.EMPTY_ACTION
    obs[:, 1] = lib.EMPTY_PARENT
    obs[:, 2] = lib.EMPTY_SIBLING
    obs[:, 3] = 1
    return obs


def _dummy_actions(batch_size: int, step_idx: int) -> np.ndarray:
    return np.zeros((batch_size, step_idx), dtype=np.int32)


def _lap_library() -> Library:
    return Library.from_config(
        LibraryConfig(
            coord_vars=["x", "y", "t"],
            state_vars=["u", "v", "gamma"],
            operators=[
                "add", "sub", "mul", "sin", "cos", "n3",
                "lap", "diff_x", "diff2_x", "diff_y", "diff2_y",
            ],
        )
    )


def _full_lap_prior_system(library: Library) -> PriorSystem:
    add_idx = library.name_to_index("add")
    return PriorSystem(
        library,
        priors=[
            LengthConstraint(
                library,
                min_=2,
                max_=LAP_REACHABILITY_MAX_LENGTH,
            ),
            DiffChildConstraint(library),
            RepeatConstraint(
                library,
                tokens=np.array([add_idx], dtype=np.int32),
                max_=5,
            ),
            TrigConstraint(library, block_diff=True),
            InverseUnaryConstraint(library),
            DiffDescendantConstraint(library),
        ],
    )


def _indices(library: Library, names: list[str]) -> list[int]:
    return [library.name_to_index(name) for name in names]


def _assert_sequence_reachable(
    library: Library,
    prior_system: PriorSystem,
    names: list[str],
) -> None:
    actions = _indices(library, names)
    tracker = IncrementalTracker(library)
    obs = tracker.reset(1)
    for step_idx, token_idx in enumerate(actions):
        adjustment = prior_system.step(tracker.history, obs, step_idx)
        assert adjustment[0, token_idx] == _allow_value(), (
            f"{names[step_idx]} forbidden at step {step_idx} while replaying "
            f"{names}"
        )
        if step_idx < len(actions) - 1:
            obs = tracker.step(np.array([token_idx], dtype=np.int32))


def _has_lap_with_binary_child(library: Library, action_row: np.ndarray) -> bool:
    meaningful = [
        int(token)
        for token in action_row.tolist()
        if int(token) != library.EMPTY_ACTION
    ]
    names = [library.names[token] for token in meaningful]
    return any(
        name == "lap" and names[index + 1] in LAP_BINARY_CHILDREN
        for index, name in enumerate(names[:-1])
    )




HARD_STEP_CASES = [
    "initial",
    "unconstrained",
    "near_max_d2",
    "near_max_d1",
    "diff_child",
    "below_min",
    "diff_below_min",
]

SOFT_STEP_CASES = [
    "soft_early",
    "soft_at_loc",
    "soft_late",
]


class TestPriorCrossValidation:

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case", HARD_STEP_CASES)
    def test_step_matches_reference(
        self, ps: PriorSystem, prior_fixture: dict[str, np.ndarray], case: str
    ) -> None:
        obs = prior_fixture[f"step_{case}_obs"]
        step_idx = int(prior_fixture[f"step_{case}_step_idx"])
        expected = prior_fixture[f"step_{case}_mask"]

        actions = _dummy_actions(obs.shape[0], step_idx)
        result = ps.step(actions, obs, step_idx)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.equivalence
    @pytest.mark.smoke
    def test_batch_matches_reference(
        self, ps: PriorSystem, prior_fixture: dict[str, np.ndarray]
    ) -> None:
        actions = prior_fixture["batch_actions"]
        obs = prior_fixture["batch_obs"]
        expected = prior_fixture["batch_masks"]

        result = ps.compute_batch(actions, obs)
        np.testing.assert_array_equal(result, expected)


class TestSoftLengthCrossValidation:

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case", SOFT_STEP_CASES)
    def test_soft_step_matches_reference(
        self, lib: Library, prior_fixture: dict[str, np.ndarray], case: str
    ) -> None:
        obs = prior_fixture[f"step_{case}_obs"]
        step_idx = int(prior_fixture[f"step_{case}_step_idx"])
        expected = prior_fixture[f"step_{case}_mask"]

        soft = SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE)
        ps = PriorSystem(lib, priors=[soft])
        actions = _dummy_actions(obs.shape[0], step_idx)
        result = ps.step(actions, obs, step_idx)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)





class TestPriorContext:

    @pytest.mark.unit
    def test_construct_and_read_fields(self, lib: Library) -> None:
        ctx = PriorContext(
            actions=np.zeros((2, 3), dtype=np.int32),
            parent=np.array([1, 2], dtype=np.int32),
            sibling=np.array([3, 4], dtype=np.int32),
            dangling=np.array([1, 2], dtype=np.int32),
            step_idx=3,
            library=lib,
        )
        assert ctx.step_idx == 3
        assert ctx.actions.shape == (2, 3)
        assert ctx.parent.tolist() == [1, 2]
        assert ctx.library is lib

    @pytest.mark.unit
    def test_frozen_raises_frozen_instance_error(self, lib: Library) -> None:
        from dataclasses import FrozenInstanceError

        ctx = PriorContext(
            actions=np.zeros((1, 0), dtype=np.int32),
            parent=np.array([lib.EMPTY_PARENT], dtype=np.int32),
            sibling=np.array([lib.EMPTY_SIBLING], dtype=np.int32),
            dangling=np.array([1], dtype=np.int32),
            step_idx=0,
            library=lib,
        )
        with pytest.raises(FrozenInstanceError):
            ctx.step_idx = 5

    @pytest.mark.unit
    def test_batch_size_consistency_enforced(self, lib: Library) -> None:
        with pytest.raises(ValueError, match="batch"):
            PriorContext(
                actions=np.zeros((2, 3), dtype=np.int32),
                parent=np.array([1, 2, 3], dtype=np.int32),
                sibling=np.array([0, 0], dtype=np.int32),
                dangling=np.array([1, 1], dtype=np.int32),
                step_idx=3,
                library=lib,
            )

    @pytest.mark.unit
    def test_step_idx_matches_actions_shape(self, lib: Library) -> None:
        with pytest.raises(ValueError, match="step_idx"):
            PriorContext(
                actions=np.zeros((1, 3), dtype=np.int32),
                parent=np.array([0], dtype=np.int32),
                sibling=np.array([0], dtype=np.int32),
                dangling=np.array([1], dtype=np.int32),
                step_idx=5,
                library=lib,
            )





def _allow_value() -> float:
    return 0.0


def _forbid_value() -> float:
    return float("-inf")


class TestLengthConstraint:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_initial_step_forbids_terminals(self, lib: Library) -> None:
        ps = PriorSystem(lib, [LengthConstraint(lib, min_=3, max_=10)])
        obs = _initial_obs(lib)
        adjustment = ps.step(_dummy_actions(1, 0), obs, step_idx=0)

        for t in lib.terminal_tokens:
            assert adjustment[0, t] == _forbid_value(), (
                f"terminal {lib.names[t]} should be forbidden"
            )
        for t in np.concatenate([lib.unary_tokens, lib.binary_tokens]):
            assert adjustment[0, t] == _allow_value()

    @pytest.mark.unit
    def test_max_forces_only_terminals(self, lib: Library) -> None:
        ps = PriorSystem(lib, [LengthConstraint(lib, min_=None, max_=10)])
        obs = np.array([[2, 0, 2, 1]], dtype=np.float32)
        adjustment = ps.step(_dummy_actions(1, 9), obs, step_idx=9)

        for t in lib.terminal_tokens:
            assert adjustment[0, t] == _allow_value()
        for t in np.concatenate([lib.unary_tokens, lib.binary_tokens]):
            assert adjustment[0, t] == _forbid_value()

    @pytest.mark.unit
    def test_min_forbids_early_termination(self, lib: Library) -> None:
        ps = PriorSystem(lib, [LengthConstraint(lib, min_=5, max_=None)])
        obs = np.array([[5, lib.EMPTY_PARENT, lib.EMPTY_SIBLING, 1]],
                       dtype=np.float32)
        adjustment = ps.step(_dummy_actions(1, 2), obs, step_idx=2)

        for t in lib.terminal_tokens:
            assert adjustment[0, t] == _forbid_value()

    @pytest.mark.unit
    def test_min_allows_terminals_once_reached(self, lib: Library) -> None:
        ps = PriorSystem(lib, [LengthConstraint(lib, min_=3, max_=None)])
        obs = np.array([[5, lib.EMPTY_PARENT, lib.EMPTY_SIBLING, 1]],
                       dtype=np.float32)
        adjustment = ps.step(_dummy_actions(1, 2), obs, step_idx=2)

        for t in lib.terminal_tokens:
            assert adjustment[0, t] == _allow_value()

    @pytest.mark.unit
    def test_unary_dangling_boundary(self, lib: Library) -> None:
        ps = PriorSystem(lib, [LengthConstraint(lib, min_=None, max_=10)])


        adj_d2 = ps.step(
            _dummy_actions(1, 8),
            np.array([[2, 0, 2, 2]], dtype=np.float32),
            step_idx=8,
        )
        for t in lib.unary_tokens:
            assert adj_d2[0, t] == _forbid_value()


        adj_d1 = ps.step(
            _dummy_actions(1, 8),
            np.array([[2, 0, 2, 1]], dtype=np.float32),
            step_idx=8,
        )
        for t in lib.unary_tokens:
            assert adj_d1[0, t] == _allow_value()

    @pytest.mark.unit
    def test_min_no_effect_when_dangling_gt_1(self, lib: Library) -> None:
        ps = PriorSystem(lib, [LengthConstraint(lib, min_=5, max_=None)])
        obs = np.array([[3, 0, lib.EMPTY_SIBLING, 3]], dtype=np.float32)
        adjustment = ps.step(_dummy_actions(1, 1), obs, step_idx=1)

        for t in lib.terminal_tokens:
            assert adjustment[0, t] == _allow_value()

    @pytest.mark.unit
    def test_max_only_initial_still_forbids_terminals(
        self, lib: Library
    ) -> None:
        ps = PriorSystem(lib, [LengthConstraint(lib, min_=None, max_=10)])
        adjustment = ps.step(_dummy_actions(1, 0), _initial_obs(lib), step_idx=0)

        for t in lib.terminal_tokens:
            assert adjustment[0, t] == _forbid_value()

    @pytest.mark.unit
    def test_max_half_activation_boundary(self, lib: Library) -> None:
        ps = PriorSystem(lib, [LengthConstraint(lib, min_=None, max_=10)])
        obs = np.array([[2, 0, 2, 5]], dtype=np.float32)


        adj_3 = ps.step(_dummy_actions(1, 3), obs, step_idx=3)
        for t in lib.binary_tokens:
            assert adj_3[0, t] == _allow_value()



        adj_4 = ps.step(_dummy_actions(1, 4), obs, step_idx=4)
        for t in lib.binary_tokens:
            assert adj_4[0, t] == _forbid_value()

    @pytest.mark.unit
    def test_ternary_operators_are_forbidden_near_max(self) -> None:
        tokens = [
            Token(name="x", arity=0, token_type=TokenType.COORDINATE),
            Token(name="u", arity=0, token_type=TokenType.TERMINAL),
            Token(name="add", arity=2, token_type=TokenType.OPERATOR),
            Token(name="diff_x", arity=1, token_type=TokenType.OPERATOR),
            Token(name="fma", arity=3, token_type=TokenType.OPERATOR),
        ]
        lib = Library(tokens)
        fma_idx = lib.name_to_index("fma")

        ps = PriorSystem(lib, [LengthConstraint(lib, min_=None, max_=10)])





        obs = np.array(
            [[0, lib.EMPTY_PARENT, lib.EMPTY_SIBLING, 1]], dtype=np.float32
        )
        adjustment = ps.step(_dummy_actions(1, 8), obs, step_idx=8)


        assert adjustment[0, fma_idx] == _forbid_value(), (
            "Arity-3 operator slipped past the length constraint — "
            "_apply_max must cover all arities, not just binary/unary."
        )













        length_only = LengthConstraint(lib, min_=None, max_=10)
        ctx_d0 = PriorContext(
            actions=np.zeros((1, 8), dtype=np.int32),
            parent=np.array([lib.EMPTY_PARENT], dtype=np.int32),
            sibling=np.array([lib.EMPTY_SIBLING], dtype=np.int32),
            dangling=np.array([0], dtype=np.int32),
            step_idx=8,
            library=lib,
        )
        adj_d0 = length_only(ctx_d0)
        add_idx = lib.name_to_index("add")


        assert adj_d0[0, fma_idx] == _forbid_value()

        assert adj_d0[0, add_idx] == _allow_value()

    @pytest.mark.unit
    def test_dangling_exceeds_remaining(self, lib: Library) -> None:
        ps = PriorSystem(lib, [LengthConstraint(lib, min_=None, max_=10)])
        obs = np.array([[2, 0, 2, 3]], dtype=np.float32)
        adjustment = ps.step(_dummy_actions(1, 9), obs, step_idx=9)


        for t in lib.binary_tokens:
            assert adjustment[0, t] == _forbid_value()

        for t in lib.unary_tokens:
            assert adjustment[0, t] == _allow_value()





class TestLengthConstraintPrecomputedArity:

    @pytest.mark.unit
    def test_tokens_by_arity_precomputed_at_init(self, lib: Library) -> None:
        constraint = LengthConstraint(lib, min_=None, max_=10)

        cached = getattr(constraint, "_tokens_by_arity", None)
        assert cached is not None, (
            "LengthConstraint must cache _tokens_by_arity at init."
        )
        cached_map = {arity: tokens for arity, tokens in cached}

        arities = lib.arities
        expected_arities = set(int(a) for a in np.unique(arities[arities >= 1]))
        assert set(cached_map.keys()) == expected_arities
        assert 0 not in cached_map, (
            "Arity-0 tokens should not appear in the max-length cache."
        )
        for arity, tokens in cached_map.items():
            expected = np.flatnonzero(arities == arity).astype(np.int32)
            np.testing.assert_array_equal(tokens, expected)





class TestDiffChildConstraint:

    @pytest.mark.unit
    def test_non_diff_parent_unconstrained(self, lib: Library) -> None:
        ps = PriorSystem(lib, [DiffChildConstraint(lib)])
        obs = np.array([[3, lib.parent_adjust[lib.name_to_index("add")],
                         lib.EMPTY_SIBLING, 2]], dtype=np.float32)
        adjustment = ps.step(_dummy_actions(1, 1), obs, step_idx=1)
        assert np.all(adjustment == _allow_value())

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_diff_parent_allows_state_and_diff_only(
        self, lib: Library
    ) -> None:
        ps = PriorSystem(lib, [DiffChildConstraint(lib)])
        pa = int(lib.parent_adjust[lib.name_to_index("diff_x")])
        obs = np.array([[7, pa, lib.EMPTY_SIBLING, 1]], dtype=np.float32)
        adjustment = ps.step(_dummy_actions(1, 1), obs, step_idx=1)

        assert adjustment[0, lib.name_to_index("u")] == _allow_value()
        assert adjustment[0, lib.name_to_index("diff_x")] == _allow_value()
        assert adjustment[0, lib.name_to_index("diff_t")] == _allow_value()
        assert adjustment[0, lib.name_to_index("x")] == _forbid_value()
        assert adjustment[0, lib.name_to_index("t")] == _forbid_value()
        assert adjustment[0, lib.name_to_index("add")] == _forbid_value()
        assert adjustment[0, lib.name_to_index("sin")] == _forbid_value()

        assert int(np.isfinite(adjustment[0]).sum()) == 3

    @pytest.mark.unit
    def test_diff_t_parent_same_restriction(self, lib: Library) -> None:
        ps = PriorSystem(lib, [DiffChildConstraint(lib)])
        pa = int(lib.parent_adjust[lib.name_to_index("diff_t")])
        obs = np.array([[8, pa, lib.EMPTY_SIBLING, 1]], dtype=np.float32)
        adjustment = ps.step(_dummy_actions(1, 1), obs, step_idx=1)
        assert int(np.isfinite(adjustment[0]).sum()) == 3

    @pytest.mark.unit
    def test_empty_parent_unconstrained(self, lib: Library) -> None:
        ps = PriorSystem(lib, [DiffChildConstraint(lib)])
        adjustment = ps.step(_dummy_actions(1, 0), _initial_obs(lib), step_idx=0)
        assert np.all(adjustment == _allow_value())

    @pytest.mark.unit
    def test_batch_mixed_parents(self, lib: Library) -> None:
        ps = PriorSystem(lib, [DiffChildConstraint(lib)])
        pa_add = int(lib.parent_adjust[lib.name_to_index("add")])
        pa_diff = int(lib.parent_adjust[lib.name_to_index("diff_x")])
        obs = np.array([
            [3, pa_add, lib.EMPTY_SIBLING, 2],
            [7, pa_diff, lib.EMPTY_SIBLING, 1],
        ], dtype=np.float32)
        adjustment = ps.step(_dummy_actions(2, 1), obs, step_idx=1)


        assert np.all(adjustment[0] == _allow_value())

        assert int(np.isfinite(adjustment[1]).sum()) == 3





class TestSoftLengthPrior:

    @pytest.mark.unit
    def test_before_loc_penalizes_nonadd_tokens(self, lib: Library) -> None:
        ps = PriorSystem(
            lib, [SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE)],
        )
        obs = _initial_obs(lib)
        adjustment = ps.step(_dummy_actions(1, 2), obs, step_idx=2)

        add_names = {"add", "add_t", "sub", "sub_t"}
        expected_penalty = np.float32(-((2 - 3) ** 2) / 10.0)
        add_seen = False
        for idx, name in enumerate(lib.names):
            if name in add_names:
                add_seen = True
                assert adjustment[0, idx] == 0.0, (
                    f"add-like {name} must be 0 at t=2"
                )
            else:
                assert adjustment[0, idx] == pytest.approx(expected_penalty), (
                    f"non-add {name}: expected {expected_penalty}, "
                    f"got {adjustment[0, idx]}"
                )
        assert add_seen, "fixture must expose an add/sub token"

    @pytest.mark.unit
    def test_after_loc_penalizes_nonterminals(self, lib: Library) -> None:
        ps = PriorSystem(
            lib, [SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE)],
        )
        obs = _initial_obs(lib)
        adjustment = ps.step(_dummy_actions(1, 9), obs, step_idx=9)

        for t in np.concatenate([lib.unary_tokens, lib.binary_tokens]):
            assert adjustment[0, t] < 0.0
            assert np.isfinite(adjustment[0, t])

        for t in lib.terminal_tokens:
            assert adjustment[0, t] == 0.0

    @pytest.mark.unit
    def test_at_loc_zero_adjustment(self, lib: Library) -> None:
        ps = PriorSystem(
            lib, [SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE)],
        )
        obs = _initial_obs(lib)
        adjustment = ps.step(_dummy_actions(1, 6), obs, step_idx=6)
        np.testing.assert_array_almost_equal(
            adjustment, np.zeros_like(adjustment), decimal=5,
        )

    @pytest.mark.unit
    def test_all_values_finite_when_alone(self, lib: Library) -> None:
        ps = PriorSystem(
            lib, [SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE)],
        )
        for step_idx in range(0, 12):
            adjustment = ps.step(
                _dummy_actions(1, step_idx), _initial_obs(lib), step_idx=step_idx,
            )
            assert np.all(np.isfinite(adjustment)), (
                f"SoftLengthPrior produced non-finite values at step {step_idx}"
            )

    @pytest.mark.unit
    def test_hard_prior_overrides_soft(self, lib: Library) -> None:
        ps = PriorSystem(
            lib,
            priors=[
                LengthConstraint(lib, min_=3, max_=10),
                SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE),
            ],
        )
        adjustment = ps.step(
            _dummy_actions(1, 0), _initial_obs(lib), step_idx=0,
        )
        for t in lib.terminal_tokens:
            assert adjustment[0, t] == _forbid_value()








SOFT_EARLY_CUTOFF = 3
SOFT_EARLY_SCALE = 10.0
_ADD_SUB_NAMES: frozenset[str] = frozenset({"add", "add_t", "sub", "sub_t"})


def _nonadd_indices(library: Library) -> list[int]:
    return [
        idx
        for idx, name in enumerate(library.names)
        if name not in _ADD_SUB_NAMES
    ]


def _add_indices(library: Library) -> list[int]:
    return [
        idx
        for idx, name in enumerate(library.names)
        if name in _ADD_SUB_NAMES
    ]


def _ref_soft_path_logprior(
    library: Library,
    token_names: list[str],
    loc: float,
    scale: float,
) -> float:
    total = 0.0
    for step_idx, name in enumerate(token_names):
        token_idx = library.name_to_index(name)
        arity = int(library.arities[token_idx])
        if step_idx < loc:
            if step_idx < SOFT_EARLY_CUTOFF:
                penalty = -((step_idx - SOFT_EARLY_CUTOFF) ** 2) / SOFT_EARLY_SCALE
                if name not in _ADD_SUB_NAMES:
                    total += penalty
        elif step_idx > loc:
            penalty = -((step_idx - loc) ** 2) / (2.0 * scale)
            if arity >= 1:
                total += penalty
    return float(total)


class TestSoftLengthPriorRefsAlignment:

    @pytest.mark.unit
    @pytest.mark.parametrize("step_idx", [0, 1, 2])
    def test_t_less_than_3_penalizes_nonadd_tokens(
        self, lib: Library, step_idx: int,
    ) -> None:
        prior = SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE)
        ps = PriorSystem(lib, priors=[prior])
        adjustment = ps.step(
            _dummy_actions(1, step_idx), _initial_obs(lib), step_idx=step_idx,
        )
        nonadd = _nonadd_indices(lib)
        add_like = _add_indices(lib)
        assert len(add_like) >= 1, "fixture must expose add or sub tokens"

        expected_penalty = np.float32(
            -((step_idx - SOFT_EARLY_CUTOFF) ** 2) / SOFT_EARLY_SCALE,
        )
        for t in nonadd:
            assert adjustment[0, t] == pytest.approx(expected_penalty), (
                f"non-add token {lib.names[t]} at step={step_idx}: "
                f"expected {expected_penalty}, got {adjustment[0, t]}"
            )
        for t in add_like:
            assert adjustment[0, t] == 0.0, (
                f"add/sub token {lib.names[t]} at step={step_idx} must be 0"
            )

    @pytest.mark.unit
    @pytest.mark.parametrize("step_idx", [3, 5])
    def test_t_3_to_loc_is_noop(self, lib: Library, step_idx: int) -> None:
        assert SOFT_EARLY_CUTOFF <= step_idx < SOFT_LOC
        ps = PriorSystem(
            lib, priors=[SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE)],
        )
        adjustment = ps.step(
            _dummy_actions(1, step_idx), _initial_obs(lib), step_idx=step_idx,
        )
        np.testing.assert_array_equal(
            adjustment, np.zeros_like(adjustment),
        )

    @pytest.mark.unit
    def test_t_3_to_loc_is_noop_wider_loc(self, lib: Library) -> None:
        wide_loc = 10.0
        ps = PriorSystem(
            lib, priors=[SoftLengthPrior(lib, loc=wide_loc, scale=SOFT_SCALE)],
        )
        for step_idx in (3, 5, 9):
            adjustment = ps.step(
                _dummy_actions(1, step_idx), _initial_obs(lib),
                step_idx=step_idx,
            )
            np.testing.assert_array_equal(
                adjustment, np.zeros_like(adjustment),
                err_msg=f"expected no-op at step={step_idx} with loc={wide_loc}",
            )

    @pytest.mark.unit
    @pytest.mark.parametrize("step_idx", [7, 9])
    def test_t_above_loc_penalizes_nonterminals(
        self, lib: Library, step_idx: int,
    ) -> None:
        ps = PriorSystem(
            lib, priors=[SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE)],
        )
        assert step_idx > SOFT_LOC
        adjustment = ps.step(
            _dummy_actions(1, step_idx), _initial_obs(lib), step_idx=step_idx,
        )
        expected_penalty = np.float32(
            -((step_idx - SOFT_LOC) ** 2) / (2.0 * SOFT_SCALE),
        )
        nonterminals = np.concatenate([lib.unary_tokens, lib.binary_tokens])
        for t in nonterminals:
            assert adjustment[0, t] == pytest.approx(expected_penalty)
        for t in lib.terminal_tokens:
            assert adjustment[0, t] == 0.0

    @pytest.mark.unit
    def test_at_loc_zero_adjustment_refs(self, lib: Library) -> None:
        ps = PriorSystem(
            lib, priors=[SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE)],
        )
        adjustment = ps.step(
            _dummy_actions(1, int(SOFT_LOC)), _initial_obs(lib),
            step_idx=int(SOFT_LOC),
        )
        np.testing.assert_array_equal(
            adjustment, np.zeros_like(adjustment),
        )

    @pytest.mark.unit
    def test_initial_adjustment_matches_step_zero(self, lib: Library) -> None:
        prior = SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE)
        ps = PriorSystem(lib, priors=[prior])
        step_zero = ps.step(_dummy_actions(1, 0), _initial_obs(lib), step_idx=0)
        direct = prior.initial_adjustment(batch_size=1)
        np.testing.assert_array_equal(step_zero, direct)

    @pytest.mark.unit
    def test_nonadd_fallback_when_library_has_no_add(self) -> None:
        bare_lib = Library.from_config(
            LibraryConfig(
                coord_vars=["x"],
                state_vars=["u"],
                operators=["mul", "sin"],
            )
        )
        prior = SoftLengthPrior(bare_lib, loc=SOFT_LOC, scale=SOFT_SCALE)
        ps = PriorSystem(bare_lib, priors=[prior])
        adjustment = ps.step(
            _dummy_actions(1, 0), _initial_obs(bare_lib), step_idx=0,
        )
        expected = np.float32(-((0 - SOFT_EARLY_CUTOFF) ** 2) / SOFT_EARLY_SCALE)

        np.testing.assert_allclose(
            adjustment, np.full_like(adjustment, expected),
        )

    @pytest.mark.unit
    def test_empirical_relative_probability_matches_refs(self) -> None:
        lib = Library.from_config(
            LibraryConfig(
                coord_vars=["x", "t"],
                state_vars=["u"],
                operators=[
                    "add", "sub", "mul", "sin", "cos",
                    "n2", "neg", "diff_x", "diff_t", "div",
                ],
            )
        )
        loc = 10.0
        scale = 5.0

        base = ["sub", "mul", "add", "sin", "cos", "n2", "neg", "diff_x"]

        def path_of_len(length: int) -> list[str]:
            return [base[i % len(base)] for i in range(length)]

        refs_lp = {
            n: _ref_soft_path_logprior(lib, path_of_len(n), loc, scale)
            for n in (3, 7, 10, 20)
        }


        prior = SoftLengthPrior(lib, loc=loc, scale=scale)
        ps = PriorSystem(lib, priors=[prior])

        def ours_lp(length: int) -> float:
            total = 0.0
            path = path_of_len(length)
            for step_idx, name in enumerate(path):
                actions = np.zeros((1, step_idx), dtype=np.int32)
                adjustment = ps.step(actions, _initial_obs(lib), step_idx)
                total += float(adjustment[0, lib.name_to_index(name)])
            return total

        ours_lp_vals = {n: ours_lp(n) for n in (3, 7, 10, 20)}

        for length, expected in refs_lp.items():
            got = ours_lp_vals[length]
            np.testing.assert_allclose(
                got, expected, rtol=0.10, atol=1e-6,
                err_msg=(
                    f"log-prior mismatch at length {length}: "
                    f"ours={got}, refs={expected}"
                ),
            )




        ratio = float(np.exp(ours_lp_vals[7] - ours_lp_vals[3]))
        assert 0.5 <= ratio <= 2.0, (
            f"P(len=7)/P(len=3) ratio {ratio:.3f} indicates refs-alignment "
            "failure (buggy impl produced ~121.5)."
        )





class TestPriorSystem:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_no_constraints_all_allowed(self, lib: Library) -> None:
        ps = PriorSystem(lib, priors=[])
        adjustment = ps.step(_dummy_actions(1, 0), _initial_obs(lib), step_idx=0)
        expected = np.zeros((1, len(lib.tokens)), dtype=np.float32)
        np.testing.assert_array_equal(adjustment, expected)

    @pytest.mark.unit
    def test_adjustment_values_are_zero_or_neg_inf(
        self, lib: Library, ps: PriorSystem,
    ) -> None:
        pa = int(lib.parent_adjust[lib.name_to_index("diff_x")])
        obs = np.array([[7, pa, lib.EMPTY_SIBLING, 1]], dtype=np.float32)
        adjustment = ps.step(_dummy_actions(1, 5), obs, step_idx=5)
        unique = set(np.unique(adjustment).tolist())
        assert unique.issubset({0.0, float("-inf")}), (
            f"hard-prior-only should yield 0/-inf, got {unique}"
        )

    @pytest.mark.unit
    def test_step_shape(self, lib: Library, ps: PriorSystem) -> None:
        B = 4
        obs = np.tile(_initial_obs(lib), (B, 1))
        adjustment = ps.step(_dummy_actions(B, 0), obs, step_idx=0)
        assert adjustment.shape == (B, len(lib.tokens))

    @pytest.mark.unit
    def test_batch_shape(self, lib: Library, ps: PriorSystem) -> None:

        actions = np.array([[5, 5, 2], [5, 5, 2]], dtype=np.int32)
        tracker = BatchTracker(lib)
        obs = tracker.compute_obs(actions)
        adjustments = ps.compute_batch(actions, obs)
        assert adjustments.shape == (2, 3, len(lib.tokens))

    @pytest.mark.unit
    def test_constraint_interaction_narrows(
        self, lib: Library, ps: PriorSystem
    ) -> None:
        pa = int(lib.parent_adjust[lib.name_to_index("diff_x")])
        obs = np.array([[7, pa, lib.EMPTY_SIBLING, 1]], dtype=np.float32)
        adjustment = ps.step(_dummy_actions(1, 1), obs, step_idx=1)

        assert adjustment[0, lib.name_to_index("diff_x")] == _allow_value()
        assert adjustment[0, lib.name_to_index("diff_t")] == _allow_value()
        assert adjustment[0, lib.name_to_index("u")] == _forbid_value()
        assert int(np.isfinite(adjustment[0]).sum()) == 2

    @pytest.mark.unit
    def test_adjustment_dtype_float32(
        self, lib: Library, ps: PriorSystem
    ) -> None:
        adjustment = ps.step(
            _dummy_actions(1, 0), _initial_obs(lib), step_idx=0,
        )
        assert adjustment.dtype == np.float32

    @pytest.mark.unit
    def test_prior_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Prior()

    @pytest.mark.unit
    def test_sum_combine_two_hard_forbidding_same_token(
        self, lib: Library,
    ) -> None:
        ps = PriorSystem(
            lib,
            priors=[
                LengthConstraint(lib, min_=3, max_=10),
                LengthConstraint(lib, min_=3, max_=10),
            ],
        )
        adjustment = ps.step(
            _dummy_actions(1, 0), _initial_obs(lib), step_idx=0,
        )

        for t in lib.terminal_tokens:
            assert adjustment[0, t] == _forbid_value()
        assert not np.any(np.isnan(adjustment))

    @pytest.mark.unit
    def test_sum_combine_hard_and_soft(self, lib: Library) -> None:
        soft_prior = SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE)
        ps = PriorSystem(
            lib,
            priors=[
                LengthConstraint(lib, min_=3, max_=10),
                soft_prior,
            ],
        )
        adjustment = ps.step(
            _dummy_actions(1, 0), _initial_obs(lib), step_idx=0,
        )

        for t in lib.terminal_tokens:
            assert adjustment[0, t] == _forbid_value(), (
                f"terminal {lib.names[t]}: expected -inf (hard dominates), "
                f"got {adjustment[0, t]}"
            )
        expected_nonadd_penalty = np.float32(-((0 - 3) ** 2) / 10.0)


        add_names = {"add", "add_t", "sub", "sub_t"}
        for t in np.concatenate([lib.unary_tokens, lib.binary_tokens]):
            name = lib.names[t]
            if name in add_names:
                assert adjustment[0, t] == 0.0, (
                    f"add-like token {name}: expected 0.0, got "
                    f"{adjustment[0, t]}"
                )
            else:
                assert adjustment[0, t] == pytest.approx(
                    expected_nonadd_penalty,
                ), (
                    f"non-add token {name}: expected {expected_nonadd_penalty}, "
                    f"got {adjustment[0, t]}"
                )

    @pytest.mark.unit
    def test_sum_combine_two_soft_priors(self, lib: Library) -> None:
        one_soft = PriorSystem(
            lib, [SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE)],
        )
        two_soft = PriorSystem(
            lib,
            priors=[
                SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE),
                SoftLengthPrior(lib, loc=SOFT_LOC, scale=SOFT_SCALE),
            ],
        )

        adj_one = one_soft.step(_dummy_actions(1, 2), _initial_obs(lib), 2)
        adj_two = two_soft.step(_dummy_actions(1, 2), _initial_obs(lib), 2)

        np.testing.assert_array_almost_equal(adj_two, 2.0 * adj_one, decimal=5)

        non_add_op_tokens = [
            idx
            for idx, name in enumerate(lib.names)
            if name not in {"add", "add_t", "sub", "sub_t"}
            and int(lib.arities[idx]) >= 1
        ]
        assert len(non_add_op_tokens) > 0, "fixture must have non-add operators"
        for t in non_add_op_tokens:
            assert adj_one[0, t] < 0.0
            assert adj_two[0, t] < adj_one[0, t]

    @pytest.mark.unit
    def test_dead_end_guard_raises(self, lib: Library) -> None:

        class _AllForbidPrior(Prior):
            def initial_adjustment(self, batch_size: int) -> np.ndarray:
                return np.full(
                    (batch_size, self.n_choices),
                    float("-inf"),
                    dtype=np.float32,
                )

            def __call__(self, ctx: PriorContext) -> np.ndarray:
                return np.full(
                    (ctx.actions.shape[0], len(ctx.library.tokens)),
                    float("-inf"),
                    dtype=np.float32,
                )

        ps = PriorSystem(lib, priors=[_AllForbidPrior(lib)])

        with pytest.raises(ValueError, match="dead"):
            ps.step(_dummy_actions(1, 0), _initial_obs(lib), step_idx=0)

        with pytest.raises(ValueError, match="dead"):
            ps.step(_dummy_actions(1, 3), _initial_obs(lib), step_idx=3)

    @pytest.mark.unit
    def test_initial_step_uses_initial_adjustment_not_call(
        self, lib: Library,
    ) -> None:
        call_count = {"initial": 0, "call": 0}

        class _TrackingPrior(Prior):
            def initial_adjustment(self, batch_size: int) -> np.ndarray:
                call_count["initial"] += 1
                return np.zeros((batch_size, self.n_choices), dtype=np.float32)

            def __call__(self, ctx: PriorContext) -> np.ndarray:
                call_count["call"] += 1


                _ = ctx.actions[:, -1]
                return np.zeros(
                    (ctx.actions.shape[0], self.n_choices),
                    dtype=np.float32,
                )

        ps = PriorSystem(lib, priors=[_TrackingPrior(lib)])

        ps.step(_dummy_actions(1, 0), _initial_obs(lib), step_idx=0)
        assert call_count["initial"] == 1
        assert call_count["call"] == 0

        ps.step(_dummy_actions(1, 1), _initial_obs(lib), step_idx=1)
        assert call_count["initial"] == 1
        assert call_count["call"] == 1

    @pytest.mark.unit
    def test_actions_step_idx_mismatch_raises(self, lib: Library) -> None:
        ps = PriorSystem(lib, priors=[LengthConstraint(lib, min_=3, max_=10)])

        bad_actions = np.zeros((1, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="step_idx"):
            ps.step(bad_actions, _initial_obs(lib), step_idx=5)

    @pytest.mark.unit
    def test_batch_size_mismatch_raises(self, lib: Library) -> None:
        ps = PriorSystem(lib, priors=[LengthConstraint(lib, min_=3, max_=10)])
        actions = np.zeros((2, 0), dtype=np.int32)
        obs = _initial_obs(lib, batch_size=3)
        with pytest.raises(ValueError, match="batch"):
            ps.step(actions, obs, step_idx=0)





class TestActionHistoryPlumbing:

    @pytest.mark.unit
    def test_prior_receives_action_history(self, lib: Library) -> None:
        captured: list[np.ndarray] = []

        class _CaptureHistoryPrior(Prior):
            def __call__(self, ctx: PriorContext) -> np.ndarray:
                captured.append(ctx.actions.copy())
                return np.zeros(
                    (ctx.actions.shape[0], len(ctx.library.tokens)),
                    dtype=np.float32,
                )

        ps = PriorSystem(lib, priors=[_CaptureHistoryPrior(lib)])
        actions = np.array([[3, 4, 2]], dtype=np.int32)
        obs = np.array([[2, 0, 2, 2]], dtype=np.float32)
        ps.step(actions, obs, step_idx=3)

        assert len(captured) == 1
        np.testing.assert_array_equal(captured[0], actions)

    @pytest.mark.unit
    def test_compute_batch_slices_history_per_step(
        self, lib: Library,
    ) -> None:
        captured_shapes: list[tuple[int, int]] = []
        initial_calls = {"count": 0}

        class _CaptureHistoryPrior(Prior):
            def initial_adjustment(self, batch_size: int) -> np.ndarray:
                initial_calls["count"] += 1
                return np.zeros(
                    (batch_size, self.n_choices), dtype=np.float32,
                )

            def __call__(self, ctx: PriorContext) -> np.ndarray:
                captured_shapes.append(tuple(ctx.actions.shape))
                return np.zeros(
                    (ctx.actions.shape[0], len(ctx.library.tokens)),
                    dtype=np.float32,
                )

        ps = PriorSystem(lib, priors=[_CaptureHistoryPrior(lib)])
        actions = np.array([[3, 4, 2, 7, 2]], dtype=np.int32)
        obs = BatchTracker(lib).compute_obs(actions)
        ps.compute_batch(actions, obs)


        assert initial_calls["count"] == 1

        assert len(captured_shapes) == 4
        for offset, shape in enumerate(captured_shapes):
            expected_step = offset + 1
            assert shape == (1, expected_step), (
                f"step {expected_step}: expected (1, {expected_step}) history, "
                f"got {shape}"
            )

    @pytest.mark.unit
    def test_tracker_history_property(self, lib: Library) -> None:
        tracker = IncrementalTracker(lib)
        tracker.reset(2)
        assert tracker.history.shape == (2, 0)

        tracker.step(np.array([3, 4], dtype=np.int32))
        assert tracker.history.shape == (2, 1)
        np.testing.assert_array_equal(tracker.history, np.array([[3], [4]]))

        tracker.step(np.array([2, 7], dtype=np.int32))
        assert tracker.history.shape == (2, 2)
        np.testing.assert_array_equal(
            tracker.history, np.array([[3, 2], [4, 7]]),
        )





class TestPriorInvariants:

    @pytest.mark.equivalence
    @pytest.mark.smoke
    def test_step_matches_batch_burgers(
        self, lib: Library, ps: PriorSystem
    ) -> None:

        actions_list = [3, 4, 2, 7, 2, 8, 2]
        B = 1

        tracker = IncrementalTracker(lib)
        obs = tracker.reset(B)
        step_adjustments = [
            ps.step(tracker.history, obs, step_idx=0),
        ]

        for t, action in enumerate(actions_list[:-1]):
            obs = tracker.step(np.array([action], dtype=np.int32))
            step_adjustments.append(
                ps.step(tracker.history, obs, step_idx=t + 1),
            )

        step_arr = np.stack(step_adjustments, axis=1)

        actions = np.array([actions_list], dtype=np.int32)
        batch_obs = BatchTracker(lib).compute_obs(actions)
        batch_arr = ps.compute_batch(actions, batch_obs)

        np.testing.assert_array_equal(step_arr, batch_arr)

    @pytest.mark.equivalence
    def test_step_matches_batch_longer(
        self, lib: Library, ps: PriorSystem
    ) -> None:

        actions_list = [3, 3, 4, 2, 2, 7, 2, 8, 2]
        B = 1

        tracker = IncrementalTracker(lib)
        obs = tracker.reset(B)
        step_adjustments = [ps.step(tracker.history, obs, step_idx=0)]
        for t, action in enumerate(actions_list[:-1]):
            obs = tracker.step(np.array([action], dtype=np.int32))
            step_adjustments.append(
                ps.step(tracker.history, obs, step_idx=t + 1),
            )
        step_arr = np.stack(step_adjustments, axis=1)

        actions = np.array([actions_list], dtype=np.int32)
        batch_obs = BatchTracker(lib).compute_obs(actions)
        batch_arr = ps.compute_batch(actions, batch_obs)

        np.testing.assert_array_equal(step_arr, batch_arr)

    @pytest.mark.equivalence
    def test_step_matches_batch_right_nested_b2(
        self, lib: Library, ps: PriorSystem
    ) -> None:
        actions = np.array([
            [3, 4, 2, 7, 2, 8, 2],
            [4, 3, 2, 2, 7, 8, 2],
        ], dtype=np.int32)
        B, T = actions.shape

        tracker = IncrementalTracker(lib)
        obs = tracker.reset(B)
        step_adjustments = [ps.step(tracker.history, obs, step_idx=0)]
        for t in range(T - 1):
            obs = tracker.step(actions[:, t])
            step_adjustments.append(
                ps.step(tracker.history, obs, step_idx=t + 1),
            )
        step_arr = np.stack(step_adjustments, axis=1)

        batch_obs = BatchTracker(lib).compute_obs(actions)
        batch_arr = ps.compute_batch(actions, batch_obs)

        np.testing.assert_array_equal(step_arr, batch_arr)

    @pytest.mark.unit
    def test_no_dead_ends(self, lib: Library, ps: PriorSystem) -> None:
        actions_list = [3, 4, 2, 7, 2, 8, 2]

        tracker = IncrementalTracker(lib)
        obs = tracker.reset(1)
        adjustment = ps.step(tracker.history, obs, step_idx=0)
        assert np.any(np.isfinite(adjustment)), "Dead end at step 0"

        for t, action in enumerate(actions_list[:-1]):
            obs = tracker.step(np.array([action], dtype=np.int32))
            adjustment = ps.step(tracker.history, obs, step_idx=t + 1)
            assert np.any(np.isfinite(adjustment)), (
                f"Dead end at step {t + 1}"
            )

    @pytest.mark.unit
    def test_batch_no_nan(self, lib: Library, ps: PriorSystem) -> None:
        actions = np.array([[3, 4, 2, 7, 2, 8, 2]], dtype=np.int32)
        obs = BatchTracker(lib).compute_obs(actions)
        adjustments = ps.compute_batch(actions, obs)
        assert not np.any(np.isnan(adjustments))

    @pytest.mark.unit
    def test_batch_values_are_zero_or_neg_inf(
        self, lib: Library, ps: PriorSystem
    ) -> None:
        actions = np.array([[3, 4, 2, 7, 2, 8, 2]], dtype=np.int32)
        obs = BatchTracker(lib).compute_obs(actions)
        adjustments = ps.compute_batch(actions, obs)
        unique = set(np.unique(adjustments).tolist())
        assert unique.issubset({0.0, float("-inf")}), (
            f"hard-prior-only batch should yield 0/-inf, got {unique}"
        )





def _obs_with_dangling(lib: Library, dangling: list[int]) -> np.ndarray:
    batch_size = len(dangling)
    obs = np.empty((batch_size, 4), dtype=np.float32)
    obs[:, 0] = lib.EMPTY_ACTION
    obs[:, 1] = lib.EMPTY_PARENT
    obs[:, 2] = lib.EMPTY_SIBLING
    obs[:, 3] = np.array(dangling, dtype=np.float32)
    return obs


def _make_all_forbidden_prior(
    library: Library,
    row_selector: Callable[[PriorContext], np.ndarray],
) -> Prior:

    class _AllForbidden(Prior):
        def __call__(self, ctx: PriorContext) -> np.ndarray:
            adjustment = np.zeros(
                (ctx.actions.shape[0], self.n_choices),
                dtype=np.float32,
            )
            mask = row_selector(ctx)
            if np.any(mask):
                adjustment[mask,:] = -np.inf
            return adjustment

    return _AllForbidden(library)


def _make_finite_penalty_prior(
    library: Library,
    row_penalties: dict[int, float],
) -> Prior:

    class _FinitePenalty(Prior):
        def __call__(self, ctx: PriorContext) -> np.ndarray:
            adjustment = np.zeros(
                (ctx.actions.shape[0], self.n_choices),
                dtype=np.float32,
            )
            for row, penalty in row_penalties.items():
                adjustment[row,:] = penalty
            return adjustment

    return _FinitePenalty(library)


class TestCombineLiveRowSemantics:

    @pytest.mark.unit
    def test_live_all_forbidden_still_raises(self, lib: Library) -> None:
        ps = PriorSystem(
            lib,
            priors=[
                _make_all_forbidden_prior(
                    lib,
                    lambda ctx: np.array([True, False], dtype=np.bool_),
                ),
            ],
        )
        actions = np.zeros((2, 3), dtype=np.int32)
        obs = _obs_with_dangling(lib, dangling=[2, 3])
        with pytest.raises(ValueError, match="dead-end"):
            ps.step(actions, obs, step_idx=3)

    @pytest.mark.unit
    def test_live_dangling_equals_one_all_forbidden_still_raises(
        self, lib: Library
    ) -> None:
        ps = PriorSystem(
            lib,
            priors=[
                _make_all_forbidden_prior(
                    lib,
                    lambda ctx: np.array([True], dtype=np.bool_),
                ),
            ],
        )
        actions = np.zeros((1, 2), dtype=np.int32)
        obs = _obs_with_dangling(lib, dangling=[1])
        with pytest.raises(ValueError, match="dead-end"):
            ps.step(actions, obs, step_idx=2)

    @pytest.mark.unit
    def test_finished_row_all_forbidden_is_silent(self, lib: Library) -> None:
        ps = PriorSystem(
            lib,
            priors=[
                _make_all_forbidden_prior(
                    lib,
                    lambda ctx: np.array([False, True], dtype=np.bool_),
                ),
            ],
        )
        actions = np.zeros((2, 3), dtype=np.int32)
        obs = _obs_with_dangling(lib, dangling=[1, 0])

        adjustment = ps.step(actions, obs, step_idx=3)

        assert adjustment.shape == (2, len(lib.tokens))

        np.testing.assert_array_equal(
            adjustment[0], np.zeros(len(lib.tokens), dtype=np.float32)
        )

        np.testing.assert_array_equal(
            adjustment[1], np.zeros(len(lib.tokens), dtype=np.float32)
        )

    @pytest.mark.unit
    def test_error_row_indices_map_to_original_batch(
        self, lib: Library
    ) -> None:
        ps = PriorSystem(
            lib,
            priors=[
                _make_all_forbidden_prior(
                    lib,
                    lambda ctx: np.array(
                        [False, False, True, True], dtype=np.bool_
                    ),
                ),
            ],
        )
        actions = np.zeros((4, 2), dtype=np.int32)


        obs = _obs_with_dangling(lib, dangling=[1, 0, 1, 0])

        with pytest.raises(ValueError, match=r"rows \[2\]"):
            ps.step(actions, obs, step_idx=2)

    @pytest.mark.unit
    def test_batch_path_negative_dangling_treated_as_finished(
        self, lib: Library
    ) -> None:
        ps = PriorSystem(
            lib,
            priors=[
                _make_all_forbidden_prior(
                    lib,
                    lambda ctx: np.array([True, True, False], dtype=np.bool_),
                ),
            ],
        )
        actions = np.zeros((3, 2), dtype=np.int32)


        obs = _obs_with_dangling(lib, dangling=[-1, -2, 2])

        adjustment = ps.step(actions, obs, step_idx=2)


        np.testing.assert_array_equal(
            adjustment[0], np.zeros(len(lib.tokens), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            adjustment[1], np.zeros(len(lib.tokens), dtype=np.float32)
        )

        np.testing.assert_array_equal(
            adjustment[2], np.zeros(len(lib.tokens), dtype=np.float32)
        )

    @pytest.mark.unit
    def test_all_rows_finished_no_crash(self, lib: Library) -> None:
        ps = PriorSystem(
            lib,
            priors=[
                _make_all_forbidden_prior(
                    lib,
                    lambda ctx: np.ones(ctx.actions.shape[0], dtype=np.bool_),
                ),
            ],
        )
        actions = np.zeros((2, 1), dtype=np.int32)
        obs = _obs_with_dangling(lib, dangling=[0, 0])

        adjustment = ps.step(actions, obs, step_idx=1)


        np.testing.assert_array_equal(
            adjustment, np.zeros((2, len(lib.tokens)), dtype=np.float32)
        )

    @pytest.mark.unit
    def test_finished_rows_zeroed_even_without_neginf(
        self, lib: Library
    ) -> None:
        ps = PriorSystem(
            lib,
            priors=[
                _make_finite_penalty_prior(
                    lib,
                    row_penalties={0: -0.5, 1: -0.7},
                ),
            ],
        )
        actions = np.zeros((2, 2), dtype=np.int32)
        obs = _obs_with_dangling(lib, dangling=[1, 0])

        adjustment = ps.step(actions, obs, step_idx=2)


        assert np.all(adjustment[0] == np.float32(-0.5))


        np.testing.assert_array_equal(
            adjustment[1], np.zeros(len(lib.tokens), dtype=np.float32)
        )

    @pytest.mark.unit
    def test_dangling_all_positive_passes_through(
        self, lib: Library
    ) -> None:
        ps = PriorSystem(
            lib,
            priors=[LengthConstraint(lib, min_=5, max_=10)],
        )
        actions = np.zeros((3, 2), dtype=np.int32)



        obs = _obs_with_dangling(lib, dangling=[1, 1, 1])

        adjustment = ps.step(actions, obs, step_idx=2)

        assert adjustment.shape == (3, len(lib.tokens))
        assert not np.any(np.isnan(adjustment))



        terminal_tokens = lib.terminal_tokens
        assert np.all(np.isneginf(adjustment[:, terminal_tokens])), (
            "live rows must retain LengthConstraint min-forbid on "
            "terminals; Option E must be a no-op here"
        )


        nonterminal_mask = np.ones(len(lib.tokens), dtype=np.bool_)
        nonterminal_mask[terminal_tokens] = False
        assert np.all(adjustment[:, nonterminal_mask] == 0.0)





class TestLibraryTrigTokens:

    @pytest.mark.unit
    def test_burgers_trig_tokens(self, lib: Library) -> None:
        expected = {lib.name_to_index("sin"), lib.name_to_index("cos")}
        assert set(lib.trig_tokens.tolist()) == expected

    @pytest.mark.unit
    def test_no_diff_in_trig_tokens(self, lib: Library) -> None:
        diff_set = set(lib.diff_tokens.tolist())
        trig_set = set(lib.trig_tokens.tolist())
        assert trig_set.isdisjoint(diff_set), (
            f"trig_tokens includes diff indices: {trig_set & diff_set}"
        )

    @pytest.mark.unit
    def test_trig_tokens_dtype_int32(self, lib: Library) -> None:
        assert lib.trig_tokens.dtype == np.int32

    @pytest.mark.unit
    def test_empty_vocab_no_trig(self) -> None:
        config = LibraryConfig(operators=["add", "diff_x"])
        lib = Library.from_config(config)
        assert lib.trig_tokens.size == 0
        assert lib.trig_tokens.dtype == np.int32

    @pytest.mark.unit
    def test_registered_trig_names_recognized(self) -> None:
        config = LibraryConfig(operators=["sin", "cos", "tan"])
        lib = Library.from_config(config)
        assert lib.trig_tokens.size == 3


class TestLibraryInverseTokens:

    @pytest.mark.unit
    def test_burgers_neg_self_inverse(self, lib: Library) -> None:
        neg_idx = lib.name_to_index("neg")
        inverse = lib.inverse_tokens
        assert isinstance(inverse, dict)
        assert inverse == {neg_idx: neg_idx}

    @pytest.mark.unit
    def test_burgers_n2_no_sqrt(self, lib: Library) -> None:
        n2_idx = lib.name_to_index("n2")
        assert n2_idx not in lib.inverse_tokens

    @pytest.mark.unit
    def test_richer_vocab_all_pairs(self) -> None:
        config = LibraryConfig(
            operators=["add", "neg", "exp", "log", "sqrt", "n2"],
        )
        lib = Library.from_config(config)
        inv = lib.inverse_tokens
        neg_idx = lib.name_to_index("neg")
        exp_idx = lib.name_to_index("exp")
        log_idx = lib.name_to_index("log")
        sqrt_idx = lib.name_to_index("sqrt")
        n2_idx = lib.name_to_index("n2")
        expected = {
            neg_idx: neg_idx,
            exp_idx: log_idx,
            log_idx: exp_idx,
            sqrt_idx: n2_idx,
            n2_idx: sqrt_idx,
        }
        assert inv == expected

    @pytest.mark.unit
    def test_inv_self_inverse(self) -> None:
        config = LibraryConfig(operators=["add", "inv"])
        lib = Library.from_config(config)
        inv_idx = lib.name_to_index("inv")
        assert lib.inverse_tokens == {inv_idx: inv_idx}

    @pytest.mark.unit
    def test_no_pairs_empty_dict(self) -> None:
        config = LibraryConfig(operators=["add", "sin", "cos"])
        lib = Library.from_config(config)
        assert lib.inverse_tokens == {}

    @pytest.mark.unit
    def test_half_pair_excluded(self) -> None:
        config = LibraryConfig(operators=["add", "exp"])
        lib = Library.from_config(config)
        assert lib.inverse_tokens == {}





def _repeat_ctx(lib: Library, actions: np.ndarray) -> PriorContext:
    batch_size = actions.shape[0]
    return PriorContext(
        actions=actions,
        parent=np.full(batch_size, lib.EMPTY_PARENT, dtype=np.int32),
        sibling=np.full(batch_size, lib.EMPTY_SIBLING, dtype=np.int32),
        dangling=np.ones(batch_size, dtype=np.int32),
        step_idx=actions.shape[1],
        library=lib,
    )

REPEAT_FIXTURE_CASES = [
    "below_max",
    "at_max",
    "above_max",
    "mixed_batch",
    "padding_safe",
    "multi_target",
    "max_zero",
]


class TestRepeatConstraint:

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case", REPEAT_FIXTURE_CASES)
    def test_cross_validation(
        self,
        lib: Library,
        repeat_fixture: dict[str, np.ndarray],
        case: str,
    ) -> None:
        actions = repeat_fixture[f"{case}_actions"]
        targets = repeat_fixture[f"{case}_targets"]
        max_ = int(repeat_fixture[f"{case}_max"])
        expected = repeat_fixture[f"{case}_expected"]

        rc = RepeatConstraint(lib, tokens=targets, max_=max_)
        result = rc(_repeat_ctx(lib, actions))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    def test_empty_history_returns_zeros(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        rc = RepeatConstraint(
            lib, tokens=np.array([add_idx], dtype=np.int32), max_=5,
        )
        ctx = PriorContext(
            actions=np.zeros((2, 0), dtype=np.int32),
            parent=np.full(2, lib.EMPTY_PARENT, dtype=np.int32),
            sibling=np.full(2, lib.EMPTY_SIBLING, dtype=np.int32),
            dangling=np.ones(2, dtype=np.int32),
            step_idx=0,
            library=lib,
        )
        result = rc(ctx)
        assert result.shape == (2, len(lib.tokens))
        assert np.all(result == 0.0)

    @pytest.mark.unit
    def test_below_threshold_no_effect(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        u_idx = lib.name_to_index("u")
        rc = RepeatConstraint(
            lib, tokens=np.array([add_idx], dtype=np.int32), max_=5,
        )
        actions = np.array([[add_idx, u_idx, add_idx, u_idx]], dtype=np.int32)
        assert np.all(rc(_repeat_ctx(lib, actions)) == 0.0)

    @pytest.mark.unit
    def test_at_threshold_forbids_targets(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        u_idx = lib.name_to_index("u")
        rc = RepeatConstraint(
            lib, tokens=np.array([add_idx], dtype=np.int32), max_=3,
        )
        actions = np.array([[add_idx, add_idx, add_idx]], dtype=np.int32)
        result = rc(_repeat_ctx(lib, actions))
        assert result[0, add_idx] == float("-inf")
        assert result[0, u_idx] == 0.0

    @pytest.mark.unit
    def test_above_threshold_still_forbidden(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        rc = RepeatConstraint(
            lib, tokens=np.array([add_idx], dtype=np.int32), max_=2,
        )
        actions = np.array(
            [[add_idx, add_idx, add_idx, add_idx, add_idx]], dtype=np.int32,
        )
        result = rc(_repeat_ctx(lib, actions))
        assert result[0, add_idx] == float("-inf")

    @pytest.mark.unit
    def test_empty_action_in_targets_raises(self, lib: Library) -> None:
        with pytest.raises(ValueError, match="EMPTY_ACTION"):
            RepeatConstraint(
                lib,
                tokens=np.array([lib.name_to_index("add"), lib.EMPTY_ACTION],
                                dtype=np.int32),
                max_=5,
            )

    @pytest.mark.unit
    def test_empty_targets_returns_zeros(self, lib: Library) -> None:
        rc = RepeatConstraint(
            lib, tokens=np.array([], dtype=np.int32), max_=1,
        )
        add_idx = lib.name_to_index("add")
        actions = np.array([[add_idx, add_idx, add_idx]], dtype=np.int32)
        assert np.all(rc(_repeat_ctx(lib, actions)) == 0.0)

    @pytest.mark.unit
    def test_max_zero_forbids_with_nonempty_history(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        u_idx = lib.name_to_index("u")
        rc = RepeatConstraint(
            lib, tokens=np.array([add_idx], dtype=np.int32), max_=0,
        )
        actions = np.array([[u_idx, u_idx]], dtype=np.int32)
        result = rc(_repeat_ctx(lib, actions))
        assert result[0, add_idx] == float("-inf")
        assert result[0, u_idx] == 0.0

    @pytest.mark.unit
    def test_max_zero_step_zero_is_exempt(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        rc = RepeatConstraint(
            lib, tokens=np.array([add_idx], dtype=np.int32), max_=0,
        )
        ctx = PriorContext(
            actions=np.zeros((1, 0), dtype=np.int32),
            parent=np.full(1, lib.EMPTY_PARENT, dtype=np.int32),
            sibling=np.full(1, lib.EMPTY_SIBLING, dtype=np.int32),
            dangling=np.ones(1, dtype=np.int32),
            step_idx=0,
            library=lib,
        )
        result = rc(ctx)
        assert result[0, add_idx] == 0.0, (
            "step 0 is exempt: empty-history early return produces zeros"
        )

    @pytest.mark.unit
    def test_output_dtype_and_shape(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        rc = RepeatConstraint(
            lib, tokens=np.array([add_idx], dtype=np.int32), max_=5,
        )
        actions = np.zeros((3, 2), dtype=np.int32)
        result = rc(_repeat_ctx(lib, actions))
        assert result.shape == (3, len(lib.tokens))
        assert result.dtype == np.float32

    @pytest.mark.unit
    def test_only_target_columns_forbidden(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        rc = RepeatConstraint(
            lib, tokens=np.array([add_idx], dtype=np.int32), max_=1,
        )
        actions = np.array([[add_idx]], dtype=np.int32)
        result = rc(_repeat_ctx(lib, actions))
        for i in range(len(lib.tokens)):
            if i == add_idx:
                assert result[0, i] == float("-inf")
            else:
                assert result[0, i] == 0.0

    @pytest.mark.equivalence
    def test_step_batch_equivalence_fires(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        u_idx = lib.name_to_index("u")
        rc = RepeatConstraint(
            lib, tokens=np.array([add_idx], dtype=np.int32), max_=2,
        )
        ps = PriorSystem(lib, priors=[rc])


        actions_list = [add_idx, add_idx, u_idx, u_idx, u_idx]

        tracker = IncrementalTracker(lib)
        obs = tracker.reset(1)
        step_adjustments = [ps.step(tracker.history, obs, step_idx=0)]
        for t, action in enumerate(actions_list[:-1]):
            obs = tracker.step(np.array([action], dtype=np.int32))
            step_adjustments.append(
                ps.step(tracker.history, obs, step_idx=t + 1),
            )
        step_arr = np.stack(step_adjustments, axis=1)

        actions = np.array([actions_list], dtype=np.int32)
        batch_obs = BatchTracker(lib).compute_obs(actions)
        batch_arr = ps.compute_batch(actions, batch_obs)

        np.testing.assert_array_equal(step_arr, batch_arr)


        assert step_arr[0, 2, add_idx] == float("-inf"), (
            "constraint should fire at step 2 (2 adds placed in history)"
        )

        assert step_arr[0, 1, add_idx] == 0.0, (
            "constraint should not fire at step 1 (only 1 add seen)"
        )





def _extended_arities(lib: Library) -> np.ndarray:
    return np.append(lib.arities, np.int32(0))


ANC_CASES = [
    "anc_1", "anc_2", "anc_3", "anc_4", "anc_5",
    "anc_6", "anc_7", "anc_8", "anc_9", "anc_10",
]


class TestAncestors:

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case", ANC_CASES)
    def test_cross_validation(
        self,
        lib: Library,
        relational_fixture: dict[str, np.ndarray],
        case: str,
    ) -> None:
        actions = relational_fixture[f"{case}_actions"]
        anc_tokens = relational_fixture[f"{case}_ancestor_tokens"]
        expected = relational_fixture[f"{case}_expected"]

        result = ancestors(actions, _extended_arities(lib), anc_tokens)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    def test_sin_open_subtree(self, lib: Library) -> None:
        sin_idx = lib.name_to_index("sin")
        actions = np.array([[sin_idx]], dtype=np.int32)
        result = ancestors(actions, _extended_arities(lib), lib.trig_tokens)
        assert result[0]

    @pytest.mark.unit
    def test_sin_complete_subtree(self, lib: Library) -> None:
        sin_idx = lib.name_to_index("sin")
        u_idx = lib.name_to_index("u")
        actions = np.array([[sin_idx, u_idx]], dtype=np.int32)
        result = ancestors(actions, _extended_arities(lib), lib.trig_tokens)
        assert not result[0]

    @pytest.mark.unit
    def test_empty_sequence(self, lib: Library) -> None:
        actions = np.zeros((1, 0), dtype=np.int32)
        result = ancestors(actions, _extended_arities(lib), lib.trig_tokens)
        assert not result[0]

    @pytest.mark.unit
    def test_nested_ancestor_outer_tracked(self, lib: Library) -> None:
        diff_idx = lib.name_to_index("diff_x")
        actions = np.array([[diff_idx, diff_idx]], dtype=np.int32)
        result = ancestors(actions, _extended_arities(lib), lib.diff_tokens)
        assert result[0]

    @pytest.mark.unit
    def test_binary_ancestor_arity_fix(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        u_idx = lib.name_to_index("u")
        actions = np.array([[add_idx, u_idx]], dtype=np.int32)
        result = ancestors(
            actions, _extended_arities(lib),
            np.array([add_idx], dtype=np.int32),
        )
        assert result[0], (
            "arity-fix: add(u,?) next is still inside add's subtree"
        )

    @pytest.mark.unit
    def test_batch_mixed_results(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        sin_idx = lib.name_to_index("sin")
        u_idx = lib.name_to_index("u")
        actions = np.array([
            [add_idx, sin_idx, u_idx],
            [add_idx, u_idx, sin_idx],
            [add_idx, u_idx, u_idx],
        ], dtype=np.int32)
        result = ancestors(actions, _extended_arities(lib), lib.trig_tokens)
        np.testing.assert_array_equal(result, [False, True, False])

    @pytest.mark.unit
    def test_output_shape_and_dtype(self, lib: Library) -> None:
        actions = np.zeros((5, 3), dtype=np.int32)
        result = ancestors(actions, _extended_arities(lib), lib.trig_tokens)
        assert result.shape == (5,)
        assert result.dtype == np.bool_

    @pytest.mark.unit
    def test_binary_ancestor_complete_subtree(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        u_idx = lib.name_to_index("u")
        actions = np.array([[add_idx, u_idx, u_idx]], dtype=np.int32)
        result = ancestors(
            actions, _extended_arities(lib),
            np.array([add_idx], dtype=np.int32),
        )
        assert not result[0], "add(u,u) subtree complete, next is outside"





def _rel_ctx(lib: Library, actions: np.ndarray, parent: np.ndarray) -> PriorContext:
    batch_size = actions.shape[0]
    return PriorContext(
        actions=actions,
        parent=parent,
        sibling=np.full(batch_size, lib.EMPTY_SIBLING, dtype=np.int32),
        dangling=np.ones(batch_size, dtype=np.int32),
        step_idx=actions.shape[1],
        library=lib,
    )


class TestRelationalConstraint:

    @pytest.mark.unit
    def test_child_relationship(self, lib: Library) -> None:
        sin_idx = lib.name_to_index("sin")
        rc = RelationalConstraint(
            lib,
            targets=lib.trig_tokens,
            effectors=lib.trig_tokens,
            relationship="child",
        )
        parent = np.array([lib.parent_adjust[sin_idx]], dtype=np.int32)
        actions = np.zeros((1, 1), dtype=np.int32)
        result = rc(_rel_ctx(lib, actions, parent))
        assert result[0, sin_idx] == float("-inf")

    @pytest.mark.unit
    def test_child_non_effector_parent(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        rc = RelationalConstraint(
            lib,
            targets=lib.trig_tokens,
            effectors=lib.trig_tokens,
            relationship="child",
        )
        parent = np.array([lib.parent_adjust[add_idx]], dtype=np.int32)
        actions = np.zeros((1, 1), dtype=np.int32)
        result = rc(_rel_ctx(lib, actions, parent))
        assert np.all(result == 0.0)

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case_num", [1, 2, 3])
    def test_child_cross_validation(
        self,
        lib: Library,
        relational_fixture: dict[str, np.ndarray],
        case_num: int,
    ) -> None:
        parent = relational_fixture[f"child_{case_num}_parent"]
        effectors = relational_fixture[f"child_{case_num}_effectors"]
        targets = relational_fixture[f"child_{case_num}_targets"]
        expected = relational_fixture[f"child_{case_num}_expected"]

        rc = RelationalConstraint(
            lib, targets=targets, effectors=effectors,
            relationship="child",
        )
        actions = np.zeros((parent.shape[0], 1), dtype=np.int32)
        result = rc(_rel_ctx(lib, actions, parent))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    def test_descendant_relationship(self, lib: Library) -> None:
        sin_idx = lib.name_to_index("sin")
        rc = RelationalConstraint(
            lib,
            targets=lib.trig_tokens,
            effectors=lib.trig_tokens,
            relationship="descendant",
        )
        actions = np.array([[sin_idx]], dtype=np.int32)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        result = rc(_rel_ctx(lib, actions, parent))
        assert result[0, sin_idx] == float("-inf")

    @pytest.mark.unit
    def test_unsupported_relationship_raises(self, lib: Library) -> None:
        rc = RelationalConstraint(
            lib,
            targets=lib.trig_tokens,
            effectors=lib.trig_tokens,
            relationship="sibling",
        )
        actions = np.zeros((1, 1), dtype=np.int32)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        with pytest.raises(NotImplementedError, match="sibling"):
            rc(_rel_ctx(lib, actions, parent))

    @pytest.mark.unit
    def test_output_dtype_shape(self, lib: Library) -> None:
        rc = RelationalConstraint(
            lib,
            targets=lib.trig_tokens,
            effectors=lib.trig_tokens,
            relationship="child",
        )
        parent = np.full(3, lib.EMPTY_PARENT, dtype=np.int32)
        actions = np.zeros((3, 2), dtype=np.int32)
        result = rc(_rel_ctx(lib, actions, parent))
        assert result.shape == (3, len(lib.tokens))
        assert result.dtype == np.float32





class TestTrigConstraint:

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case", ["trig_1", "trig_2", "trig_3", "trig_4"])
    def test_cross_validation_block_diff(
        self,
        lib: Library,
        relational_fixture: dict[str, np.ndarray],
        case: str,
    ) -> None:
        actions = relational_fixture[f"{case}_actions"]
        expected = relational_fixture[f"{case}_expected"]

        tc = TrigConstraint(lib, block_diff=True)
        parent = np.full(actions.shape[0], lib.EMPTY_PARENT, dtype=np.int32)
        result = tc(_rel_ctx(lib, actions, parent))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.equivalence
    def test_cross_validation_no_block_diff(
        self,
        lib: Library,
        relational_fixture: dict[str, np.ndarray],
    ) -> None:
        actions = relational_fixture["trig_5_actions"]
        expected = relational_fixture["trig_5_expected"]

        tc = TrigConstraint(lib, block_diff=False)
        parent = np.full(actions.shape[0], lib.EMPTY_PARENT, dtype=np.int32)
        result = tc(_rel_ctx(lib, actions, parent))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    def test_sin_inside_cos_forbids_trig_and_diff(self, lib: Library) -> None:
        sin_idx = lib.name_to_index("sin")
        cos_idx = lib.name_to_index("cos")
        diff_x_idx = lib.name_to_index("diff_x")

        tc = TrigConstraint(lib, block_diff=True)
        actions = np.array([[sin_idx, cos_idx]], dtype=np.int32)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        result = tc(_rel_ctx(lib, actions, parent))

        assert result[0, sin_idx] == float("-inf")
        assert result[0, cos_idx] == float("-inf")
        assert result[0, diff_x_idx] == float("-inf")

    @pytest.mark.unit
    def test_diff_inside_diff_forbidden(self, lib: Library) -> None:
        diff_x_idx = lib.name_to_index("diff_x")
        tc = TrigConstraint(lib, block_diff=True)
        actions = np.array([[diff_x_idx, diff_x_idx]], dtype=np.int32)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        result = tc(_rel_ctx(lib, actions, parent))
        assert result[0, diff_x_idx] == float("-inf")

    @pytest.mark.unit
    def test_trig_inside_diff_forbidden(self, lib: Library) -> None:
        diff_x_idx = lib.name_to_index("diff_x")
        sin_idx = lib.name_to_index("sin")
        cos_idx = lib.name_to_index("cos")
        tc = TrigConstraint(lib, block_diff=True)
        actions = np.array([[diff_x_idx, sin_idx]], dtype=np.int32)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        result = tc(_rel_ctx(lib, actions, parent))




        assert result[0, sin_idx] == float("-inf")
        assert result[0, cos_idx] == float("-inf")
        assert result[0, diff_x_idx] == float("-inf")

    @pytest.mark.unit
    def test_non_immediate_descendant_forbidden(self, lib: Library) -> None:
        sin_idx = lib.name_to_index("sin")
        add_idx = lib.name_to_index("add")
        tc = TrigConstraint(lib, block_diff=True)
        actions = np.array([[sin_idx, add_idx]], dtype=np.int32)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        result = tc(_rel_ctx(lib, actions, parent))
        assert result[0, sin_idx] == float("-inf"), (
            "trig still forbidden at non-immediate descendant position"
        )

    @pytest.mark.unit
    def test_no_ancestor_returns_zeros(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        u_idx = lib.name_to_index("u")
        tc = TrigConstraint(lib, block_diff=True)
        actions = np.array([[add_idx, u_idx]], dtype=np.int32)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        result = tc(_rel_ctx(lib, actions, parent))
        assert np.all(result == 0.0)

    @pytest.mark.unit
    def test_block_diff_false_allows_diff_inside_trig(
        self, lib: Library,
    ) -> None:
        sin_idx = lib.name_to_index("sin")
        tc = TrigConstraint(lib, block_diff=False)
        actions = np.array([[sin_idx]], dtype=np.int32)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        result = tc(_rel_ctx(lib, actions, parent))
        assert result[0, sin_idx] == float("-inf"), "trig still forbidden"

    @pytest.mark.unit
    def test_block_diff_false_allows_diff_nesting(self, lib: Library) -> None:
        diff_x_idx = lib.name_to_index("diff_x")
        tc = TrigConstraint(lib, block_diff=False)
        actions = np.array([[diff_x_idx, diff_x_idx]], dtype=np.int32)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        result = tc(_rel_ctx(lib, actions, parent))
        assert result[0, diff_x_idx] == 0.0, (
            "diff nesting allowed when block_diff=False"
        )
        assert result[0, diff_x_idx] == 0.0, "diff allowed when block_diff=False"

    @pytest.mark.equivalence
    def test_step_batch_equivalence(self, lib: Library) -> None:
        tc = TrigConstraint(lib, block_diff=True)
        ps = PriorSystem(lib, priors=[tc])

        sin_idx = lib.name_to_index("sin")
        add_idx = lib.name_to_index("add")
        u_idx = lib.name_to_index("u")

        actions_list = [sin_idx, add_idx, u_idx, u_idx]

        tracker = IncrementalTracker(lib)
        obs = tracker.reset(1)
        step_adjustments = [ps.step(tracker.history, obs, step_idx=0)]
        for t, action in enumerate(actions_list[:-1]):
            obs = tracker.step(np.array([action], dtype=np.int32))
            step_adjustments.append(
                ps.step(tracker.history, obs, step_idx=t + 1),
            )
        step_arr = np.stack(step_adjustments, axis=1)

        actions = np.array([actions_list], dtype=np.int32)
        batch_obs = BatchTracker(lib).compute_obs(actions)
        batch_arr = ps.compute_batch(actions, batch_obs)

        np.testing.assert_array_equal(step_arr, batch_arr)

        assert step_arr[0, 1, sin_idx] == float("-inf"), (
            "trig should be forbidden at step 1 (inside sin subtree)"
        )





class TestDiffDescendantConstraint:

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case", ["diffdes_1", "diffdes_2", "diffdes_3"])
    def test_cross_validation(
        self,
        lib: Library,
        relational_fixture: dict[str, np.ndarray],
        case: str,
    ) -> None:
        actions = relational_fixture[f"{case}_actions"]
        expected = relational_fixture[f"{case}_expected"]

        ddc = DiffDescendantConstraint(lib)
        parent = np.full(actions.shape[0], lib.EMPTY_PARENT, dtype=np.int32)
        result = ddc(_rel_ctx(lib, actions, parent))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    def test_add_forbidden_inside_diff(self, lib: Library) -> None:
        diff_x_idx = lib.name_to_index("diff_x")
        add_idx = lib.name_to_index("add")
        ddc = DiffDescendantConstraint(lib)
        actions = np.array([[diff_x_idx]], dtype=np.int32)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        result = ddc(_rel_ctx(lib, actions, parent))
        assert result[0, add_idx] == float("-inf")

    @pytest.mark.unit
    def test_mul_allowed_inside_diff(self, lib: Library) -> None:
        diff_x_idx = lib.name_to_index("diff_x")
        mul_idx = lib.name_to_index("mul")
        ddc = DiffDescendantConstraint(lib)
        actions = np.array([[diff_x_idx]], dtype=np.int32)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        result = ddc(_rel_ctx(lib, actions, parent))
        assert result[0, mul_idx] == 0.0

    @pytest.mark.unit
    def test_not_inside_diff_returns_zeros(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        u_idx = lib.name_to_index("u")
        ddc = DiffDescendantConstraint(lib)
        actions = np.array([[add_idx, u_idx]], dtype=np.int32)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        result = ddc(_rel_ctx(lib, actions, parent))
        assert np.all(result == 0.0)

    @pytest.mark.unit
    def test_extra_forbidden_extends_targets(self, lib: Library) -> None:
        diff_x_idx = lib.name_to_index("diff_x")
        mul_idx = lib.name_to_index("mul")
        ddc = DiffDescendantConstraint(lib, extra_forbidden=["mul"])
        actions = np.array([[diff_x_idx]], dtype=np.int32)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        result = ddc(_rel_ctx(lib, actions, parent))
        assert result[0, mul_idx] == float("-inf"), (
            "mul should be forbidden when passed as extra_forbidden"
        )

    @pytest.mark.equivalence
    def test_step_batch_equivalence(self, lib: Library) -> None:
        ddc = DiffDescendantConstraint(lib)
        ps = PriorSystem(lib, priors=[ddc])

        diff_x_idx = lib.name_to_index("diff_x")
        u_idx = lib.name_to_index("u")
        add_idx = lib.name_to_index("add")

        actions_list = [diff_x_idx, u_idx]

        tracker = IncrementalTracker(lib)
        obs = tracker.reset(1)
        step_adjustments = [ps.step(tracker.history, obs, step_idx=0)]
        for t, action in enumerate(actions_list[:-1]):
            obs = tracker.step(np.array([action], dtype=np.int32))
            step_adjustments.append(
                ps.step(tracker.history, obs, step_idx=t + 1),
            )
        step_arr = np.stack(step_adjustments, axis=1)

        actions = np.array([actions_list], dtype=np.int32)
        batch_obs = BatchTracker(lib).compute_obs(actions)
        batch_arr = ps.compute_batch(actions, batch_obs)

        np.testing.assert_array_equal(step_arr, batch_arr)

        assert step_arr[0, 1, add_idx] == float("-inf")





class TestInverseUnaryConstraint:

    @pytest.mark.unit
    def test_neg_as_child_of_neg_forbidden(self, lib: Library) -> None:
        neg_idx = lib.name_to_index("neg")
        iuc = InverseUnaryConstraint(lib)
        parent = np.array([lib.parent_adjust[neg_idx]], dtype=np.int32)
        actions = np.zeros((1, 1), dtype=np.int32)
        result = iuc(_rel_ctx(lib, actions, parent))
        assert result[0, neg_idx] == float("-inf")

    @pytest.mark.unit
    def test_neg_under_add_allowed(self, lib: Library) -> None:
        neg_idx = lib.name_to_index("neg")
        add_idx = lib.name_to_index("add")
        iuc = InverseUnaryConstraint(lib)
        parent = np.array([lib.parent_adjust[add_idx]], dtype=np.int32)
        actions = np.zeros((1, 1), dtype=np.int32)
        result = iuc(_rel_ctx(lib, actions, parent))
        assert result[0, neg_idx] == 0.0

    @pytest.mark.unit
    def test_non_inverse_tokens_unaffected(self, lib: Library) -> None:
        neg_idx = lib.name_to_index("neg")
        u_idx = lib.name_to_index("u")
        add_idx = lib.name_to_index("add")
        iuc = InverseUnaryConstraint(lib)
        parent = np.array([lib.parent_adjust[neg_idx]], dtype=np.int32)
        actions = np.zeros((1, 1), dtype=np.int32)
        result = iuc(_rel_ctx(lib, actions, parent))
        assert result[0, u_idx] == 0.0
        assert result[0, add_idx] == 0.0

    @pytest.mark.unit
    def test_richer_vocab_multiple_pairs(self) -> None:
        config = LibraryConfig(
            operators=["add", "neg", "exp", "log"],
        )
        lib = Library.from_config(config)
        iuc = InverseUnaryConstraint(lib)
        exp_idx = lib.name_to_index("exp")
        log_idx = lib.name_to_index("log")
        neg_idx = lib.name_to_index("neg")


        parent_exp = np.array([lib.parent_adjust[exp_idx]], dtype=np.int32)
        actions = np.zeros((1, 1), dtype=np.int32)
        result = iuc(_rel_ctx(lib, actions, parent_exp))
        assert result[0, log_idx] == float("-inf"), "log forbidden under exp"
        assert result[0, exp_idx] == 0.0, "exp allowed under exp"


        parent_log = np.array([lib.parent_adjust[log_idx]], dtype=np.int32)
        result2 = iuc(_rel_ctx(lib, actions, parent_log))
        assert result2[0, exp_idx] == float("-inf"), "exp forbidden under log"


        parent_neg = np.array([lib.parent_adjust[neg_idx]], dtype=np.int32)
        result3 = iuc(_rel_ctx(lib, actions, parent_neg))
        assert result3[0, neg_idx] == float("-inf"), "neg forbidden under neg"

    @pytest.mark.unit
    def test_empty_inverse_tokens_is_noop(self) -> None:
        config = LibraryConfig(operators=["add", "sin", "cos"])
        lib = Library.from_config(config)
        assert lib.inverse_tokens == {}
        iuc = InverseUnaryConstraint(lib)
        parent = np.full(1, lib.EMPTY_PARENT, dtype=np.int32)
        actions = np.zeros((1, 1), dtype=np.int32)
        result = iuc(_rel_ctx(lib, actions, parent))
        assert np.all(result == 0.0)

    @pytest.mark.unit
    def test_output_shape_and_dtype(self, lib: Library) -> None:
        iuc = InverseUnaryConstraint(lib)
        parent = np.full(3, lib.EMPTY_PARENT, dtype=np.int32)
        actions = np.zeros((3, 2), dtype=np.int32)
        result = iuc(_rel_ctx(lib, actions, parent))
        assert result.shape == (3, len(lib.tokens))
        assert result.dtype == np.float32

    @pytest.mark.equivalence
    def test_step_batch_equivalence(self, lib: Library) -> None:
        iuc = InverseUnaryConstraint(lib)
        ps = PriorSystem(lib, priors=[iuc])

        neg_idx = lib.name_to_index("neg")
        u_idx = lib.name_to_index("u")

        actions_list = [neg_idx, neg_idx, u_idx]

        tracker = IncrementalTracker(lib)
        obs = tracker.reset(1)
        step_adjustments = [ps.step(tracker.history, obs, step_idx=0)]
        for t, action in enumerate(actions_list[:-1]):
            obs = tracker.step(np.array([action], dtype=np.int32))
            step_adjustments.append(
                ps.step(tracker.history, obs, step_idx=t + 1),
            )
        step_arr = np.stack(step_adjustments, axis=1)

        actions = np.array([actions_list], dtype=np.int32)
        batch_obs = BatchTracker(lib).compute_obs(actions)
        batch_arr = ps.compute_batch(actions, batch_obs)

        np.testing.assert_array_equal(step_arr, batch_arr)

        assert step_arr[0, 1, neg_idx] == float("-inf")





class TestPriorSystemWithAllPriors:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_full_prior_set_no_dead_ends(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        ps = PriorSystem(
            lib,
            priors=[
                LengthConstraint(lib, min_=3, max_=10),
                DiffChildConstraint(lib),
                RepeatConstraint(
                    lib,
                    tokens=np.array([add_idx], dtype=np.int32),
                    max_=5,
                ),
                TrigConstraint(lib, block_diff=True),
                InverseUnaryConstraint(lib),
                DiffDescendantConstraint(lib),
            ],
        )

        actions_list = [add_idx, lib.name_to_index("mul"),
                        lib.name_to_index("u"), lib.name_to_index("diff_x"),
                        lib.name_to_index("u"), lib.name_to_index("diff_t"),
                        lib.name_to_index("u")]

        tracker = IncrementalTracker(lib)
        obs = tracker.reset(1)
        adjustment = ps.step(tracker.history, obs, step_idx=0)
        assert np.any(np.isfinite(adjustment)), "Dead end at step 0"

        for t, action in enumerate(actions_list[:-1]):
            obs = tracker.step(np.array([action], dtype=np.int32))
            adjustment = ps.step(tracker.history, obs, step_idx=t + 1)
            assert np.any(np.isfinite(adjustment)), f"Dead end at step {t + 1}"

    @pytest.mark.unit
    def test_full_prior_set_new_priors_fire(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        neg_idx = lib.name_to_index("neg")
        ps = PriorSystem(
            lib,
            priors=[
                LengthConstraint(lib, min_=3, max_=10),
                DiffChildConstraint(lib),
                RepeatConstraint(
                    lib,
                    tokens=np.array([add_idx], dtype=np.int32),
                    max_=5,
                ),
                TrigConstraint(lib, block_diff=True),
                InverseUnaryConstraint(lib),
                DiffDescendantConstraint(lib),
            ],
        )

        tracker = IncrementalTracker(lib)
        obs = tracker.reset(1)
        _ = ps.step(tracker.history, obs, step_idx=0)
        obs = tracker.step(np.array([neg_idx], dtype=np.int32))
        adj_step1 = ps.step(tracker.history, obs, step_idx=1)

        assert adj_step1[0, neg_idx] == float("-inf"), (
            "InverseUnary must forbid neg when parent is neg"
        )

    @pytest.mark.equivalence
    def test_full_prior_step_batch_equivalence(self, lib: Library) -> None:
        add_idx = lib.name_to_index("add")
        ps = PriorSystem(
            lib,
            priors=[
                LengthConstraint(lib, min_=3, max_=10),
                DiffChildConstraint(lib),
                RepeatConstraint(
                    lib,
                    tokens=np.array([add_idx], dtype=np.int32),
                    max_=5,
                ),
                TrigConstraint(lib, block_diff=True),
                InverseUnaryConstraint(lib),
                DiffDescendantConstraint(lib),
            ],
        )
        actions_list = [add_idx, lib.name_to_index("mul"),
                        lib.name_to_index("u"), lib.name_to_index("diff_x"),
                        lib.name_to_index("u"), lib.name_to_index("diff_t"),
                        lib.name_to_index("u")]

        tracker = IncrementalTracker(lib)
        obs = tracker.reset(1)
        step_adjustments = [ps.step(tracker.history, obs, step_idx=0)]
        for t, action in enumerate(actions_list[:-1]):
            obs = tracker.step(np.array([action], dtype=np.int32))
            step_adjustments.append(
                ps.step(tracker.history, obs, step_idx=t + 1),
            )
        step_arr = np.stack(step_adjustments, axis=1)

        actions = np.array([actions_list], dtype=np.int32)
        batch_obs = BatchTracker(lib).compute_obs(actions)
        batch_arr = ps.compute_batch(actions, batch_obs)

        np.testing.assert_array_equal(step_arr, batch_arr)





class TestLapSpecialDiffReachability:

    @pytest.mark.unit
    def test_lap_is_not_in_diff_tokens(self) -> None:
        lib = _lap_library()
        diff_names = {lib.names[index] for index in lib.diff_tokens}
        special_names = {lib.names[index] for index in lib.special_diff_tokens}

        assert "lap" not in diff_names
        assert special_names == {"lap"}

    @pytest.mark.unit
    def test_diff_child_does_not_constrain_lap_child(self) -> None:
        lib = _lap_library()
        ps = PriorSystem(lib, [DiffChildConstraint(lib)])
        _assert_sequence_reachable(lib, ps, ["lap", "add", "u", "v"])

    @pytest.mark.unit
    def test_full_prior_allows_lap_over_binary_expressions(self) -> None:
        lib = _lap_library()
        ps = _full_lap_prior_system(lib)

        _assert_sequence_reachable(lib, ps, ["lap", "add", "u", "v"])
        _assert_sequence_reachable(lib, ps, ["lap", "sub", "u", "n3", "u"])

    @pytest.mark.unit
    def test_trig_constraint_does_not_block_lap_sin_child(self) -> None:
        lib = _lap_library()
        ps = _full_lap_prior_system(lib)
        _assert_sequence_reachable(lib, ps, ["lap", "sin", "u"])

    @pytest.mark.unit
    def test_cahn_hilliard_shape_survives_full_prior_stack(self) -> None:
        lib = _lap_library()
        ps = _full_lap_prior_system(lib)

        _assert_sequence_reachable(
            lib,
            ps,
            [
                "lap",
                "sub",
                "sub",
                "n3",
                "u",
                "u",
                "mul",
                "gamma",
                "lap",
                "u",
            ],
        )

    @pytest.mark.unit
    def test_sampling_reaches_lap_binary_shape(self) -> None:
        lib = _lap_library()
        ps = _full_lap_prior_system(lib)
        controller = LSTMController(
            library=lib,
            prior_system=ps,
            num_units=8,
            num_layers=1,
            embedding_dim=4,
            observe_parent=True,
            observe_sibling=True,
            observe_action=False,
            observe_dangling=False,
            use_embedding=False,
            attention=False,
            initializer="zeros",
        )

        torch.manual_seed(LAP_SAMPLING_SEED)
        batch = controller.sample(LAP_SAMPLING_BATCH_SIZE)

        assert any(
            _has_lap_with_binary_child(lib, row)
            for row in batch.actions
        ), (
            "N=1000 uniform samples did not include lap(<binary>(...)); "
            "either lap is still masked or the deterministic seed needs review."
        )

    @pytest.mark.unit
    def test_max_diff_order_counts_lap_as_second_order(self) -> None:
        lib = _lap_library()
        tokens = np.array(_indices(lib, ["lap", "u"]), dtype=np.int32)
        assert max_diff_order(tokens, lib) == 2
