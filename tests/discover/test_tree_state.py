
import numpy as np
import pytest

from kd.search.discover.controller.tree_state import (
    BatchTracker,
    IncrementalTracker,
    parents_siblings,
)
from kd.search.discover.tokens.library import Library, LibraryConfig



X1, U1, ADD, MUL, N2, N3, SUB = range(7)





def _make_test_library() -> Library:
    return Library.from_config(
        LibraryConfig(
            coord_vars=["x1"],
            state_vars=["u1"],
            operators=["add", "mul", "n2", "n3", "sub"],
        )
    )


def _make_diff_library() -> Library:
    return Library.from_config(
        LibraryConfig(
            coord_vars=["x1"],
            state_vars=["u1"],
            operators=["add", "mul", "diff", "diff2"],
        )
    )




VALID_SEQS = [
    [X1],
    [ADD, X1, U1],
    [N2, X1],
    [ADD, MUL, U1, N2, U1, N2, U1],
    [N2, N3, N2, X1],
    [SUB, ADD, X1, U1, MUL, U1, X1],
    [MUL, ADD, X1, U1, SUB, U1, X1],
    [N2, MUL, X1, U1],
]

VALID_SEQ_IDS = [
    "terminal", "binary", "unary", "nested", "deep_unary",
    "with_sub", "alt_structure", "unary_over_binary",
]

FIXTURE_CASES = [
    "single_terminal",
    "binary_add",
    "unary_n2",
    "nested",
    "deep_unary",
    "with_sub",
]







class TestBatchTrackerCrossValidation:

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case", FIXTURE_CASES)
    def test_batch_obs_matches_reference(
        self, tree_state_fixture: dict[str, np.ndarray], case: str
    ) -> None:
        lib = _make_test_library()
        tracker = BatchTracker(lib)

        tokens = tree_state_fixture[f"{case}_tokens"]
        expected = tree_state_fixture[f"{case}_obs"]

        obs = tracker.compute_obs(tokens)
        np.testing.assert_array_equal(obs, expected)

    @pytest.mark.equivalence
    def test_batch_multi_sequence(
        self, tree_state_fixture: dict[str, np.ndarray]
    ) -> None:
        lib = _make_test_library()
        tracker = BatchTracker(lib)

        tokens = tree_state_fixture["batch_tokens"]
        expected = tree_state_fixture["batch_obs"]

        obs = tracker.compute_obs(tokens)
        np.testing.assert_array_equal(obs, expected)

    @pytest.mark.equivalence
    def test_batch_with_diff_operators(
        self, tree_state_fixture: dict[str, np.ndarray]
    ) -> None:
        lib = _make_diff_library()
        tracker = BatchTracker(lib)

        tokens = tree_state_fixture["diff_tokens"]
        expected = tree_state_fixture["diff_obs"]

        obs = tracker.compute_obs(tokens)
        np.testing.assert_array_equal(obs, expected)







class TestIncrementalTrackerCrossValidation:

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case", FIXTURE_CASES)
    def test_incremental_obs_matches_reference(
        self, tree_state_fixture: dict[str, np.ndarray], case: str
    ) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib)

        tokens = tree_state_fixture[f"{case}_tokens"]
        expected = tree_state_fixture[f"{case}_obs"]
        B, L = tokens.shape

        obs_0 = tracker.reset(B)
        np.testing.assert_array_equal(obs_0, expected[:, :, 0])

        for t in range(L - 1):
            obs_t = tracker.step(tokens[:, t])
            np.testing.assert_array_equal(
                obs_t, expected[:, :, t + 1],
                err_msg=f"Mismatch at step {t + 1} for case '{case}'",
            )

    @pytest.mark.equivalence
    def test_incremental_diff_obs_matches_reference(
        self, tree_state_fixture: dict[str, np.ndarray]
    ) -> None:
        lib = _make_diff_library()
        tracker = IncrementalTracker(lib)

        tokens = tree_state_fixture["diff_tokens"]
        expected = tree_state_fixture["diff_obs"]
        B, L = tokens.shape

        obs_0 = tracker.reset(B)
        np.testing.assert_array_equal(obs_0, expected[:, :, 0])

        for t in range(L - 1):
            obs_t = tracker.step(tokens[:, t])
            np.testing.assert_array_equal(
                obs_t, expected[:, :, t + 1],
                err_msg=f"Mismatch at step {t + 1} for diff case",
            )

    @pytest.mark.equivalence
    def test_incremental_dangling_matches_reference(
        self, tree_state_fixture: dict[str, np.ndarray]
    ) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib)

        tokens = tree_state_fixture["nested_tokens"]
        expected = tree_state_fixture["nested_obs"]
        B, L = tokens.shape

        obs = tracker.reset(B)
        assert obs[0, 3] == expected[0, 3, 0]

        for t in range(L - 1):
            obs = tracker.step(tokens[:, t])
            assert obs[0, 3] == expected[0, 3, t + 1], (
                f"Dangling mismatch at step {t + 1}: "
                f"got {obs[0, 3]}, expected {expected[0, 3, t + 1]}"
            )







class TestIncrementalBatchConsistency:

    @pytest.mark.smoke
    @pytest.mark.parametrize("seq", VALID_SEQS, ids=VALID_SEQ_IDS)
    def test_incremental_equals_batch(self, seq: list[int]) -> None:
        lib = _make_test_library()
        tokens = np.array([seq], dtype=np.int32)
        B, L = tokens.shape


        batch_obs = BatchTracker(lib).compute_obs(tokens)


        inc_tracker = IncrementalTracker(lib)
        inc_obs = np.zeros((B, 4, L), dtype=np.float32)
        inc_obs[:, :, 0] = inc_tracker.reset(B)
        for t in range(L - 1):
            inc_obs[:, :, t + 1] = inc_tracker.step(tokens[:, t])

        np.testing.assert_array_equal(inc_obs, batch_obs)

    @pytest.mark.smoke
    def test_incremental_equals_batch_multi_row(self) -> None:
        lib = _make_test_library()
        tokens = np.array(
            [
                [ADD, X1, U1],
                [N2, N2, X1],
                [MUL, X1, U1],
            ],
            dtype=np.int32,
        )
        B, L = tokens.shape

        batch_obs = BatchTracker(lib).compute_obs(tokens)

        inc_tracker = IncrementalTracker(lib)
        inc_obs = np.zeros((B, 4, L), dtype=np.float32)
        inc_obs[:, :, 0] = inc_tracker.reset(B)
        for t in range(L - 1):
            inc_obs[:, :, t + 1] = inc_tracker.step(tokens[:, t])

        np.testing.assert_array_equal(inc_obs, batch_obs)

    @pytest.mark.smoke
    def test_incremental_equals_batch_diff_library(self) -> None:
        lib = _make_diff_library()

        tokens = np.array([[2, 3, 1, 4, 1, 5, 1]], dtype=np.int32)
        B, L = tokens.shape

        batch_obs = BatchTracker(lib).compute_obs(tokens)

        inc_tracker = IncrementalTracker(lib)
        inc_obs = np.zeros((B, 4, L), dtype=np.float32)
        inc_obs[:, :, 0] = inc_tracker.reset(B)
        for t in range(L - 1):
            inc_obs[:, :, t + 1] = inc_tracker.step(tokens[:, t])

        np.testing.assert_array_equal(inc_obs, batch_obs)







class TestIncrementalTracker:

    @pytest.mark.unit
    def test_initial_obs(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib)

        obs = tracker.reset(batch_size=2)
        assert obs.shape == (2, 4)
        assert obs.dtype == np.float32

        for b in range(2):
            assert obs[b, 0] == lib.EMPTY_ACTION
            assert obs[b, 1] == lib.EMPTY_PARENT
            assert obs[b, 2] == lib.EMPTY_SIBLING
            assert obs[b, 3] == 1

    @pytest.mark.unit
    def test_single_terminal_dangling(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib)

        obs_0 = tracker.reset(1)
        assert obs_0[0, 3] == 1

        obs_1 = tracker.step(np.array([X1], dtype=np.int32))
        assert obs_1[0, 3] == 0

    @pytest.mark.unit
    def test_binary_op_dangling(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib)

        obs0 = tracker.reset(1)
        assert obs0[0, 3] == 1

        obs1 = tracker.step(np.array([ADD], dtype=np.int32))
        assert obs1[0, 3] == 2

        obs2 = tracker.step(np.array([X1], dtype=np.int32))
        assert obs2[0, 3] == 1

        obs3 = tracker.step(np.array([U1], dtype=np.int32))
        assert obs3[0, 3] == 0

    @pytest.mark.unit
    def test_unary_op_dangling(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib)

        obs0 = tracker.reset(1)
        assert obs0[0, 3] == 1

        obs1 = tracker.step(np.array([N2], dtype=np.int32))
        assert obs1[0, 3] == 1

        obs2 = tracker.step(np.array([X1], dtype=np.int32))
        assert obs2[0, 3] == 0

    @pytest.mark.unit
    def test_parent_after_operator(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib)

        tracker.reset(1)
        obs = tracker.step(np.array([ADD], dtype=np.int32))

        assert obs[0, 1] == lib.parent_adjust[ADD]
        assert obs[0, 2] == lib.EMPTY_SIBLING

    @pytest.mark.unit
    def test_sibling_after_first_child(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib)

        tracker.reset(1)
        tracker.step(np.array([ADD], dtype=np.int32))
        obs = tracker.step(np.array([X1], dtype=np.int32))


        assert obs[0, 2] == X1

        assert obs[0, 1] == lib.parent_adjust[ADD]

    @pytest.mark.unit
    def test_prev_action_in_obs(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib)

        tracker.reset(1)
        obs = tracker.step(np.array([ADD], dtype=np.int32))
        assert obs[0, 0] == ADD

        obs = tracker.step(np.array([X1], dtype=np.int32))
        assert obs[0, 0] == X1

    @pytest.mark.unit
    def test_batch_step(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib)

        tracker.reset(batch_size=3)
        actions = np.array([ADD, MUL, N2], dtype=np.int32)
        obs = tracker.step(actions)

        assert obs.shape == (3, 4)
        assert obs[0, 0] == ADD
        assert obs[1, 0] == MUL
        assert obs[2, 0] == N2

        assert obs[0, 3] == 2
        assert obs[1, 3] == 2
        assert obs[2, 3] == 1







class TestBatchTracker:

    @pytest.mark.unit
    def test_output_shape(self) -> None:
        lib = _make_test_library()
        tracker = BatchTracker(lib)

        tokens = np.array([[ADD, X1, U1]], dtype=np.int32)
        obs = tracker.compute_obs(tokens)

        assert obs.shape == (1, 4, 3)
        assert obs.dtype == np.float32

    @pytest.mark.unit
    def test_initial_position_is_empty(self) -> None:
        lib = _make_test_library()
        tracker = BatchTracker(lib)

        tokens = np.array([[ADD, X1, U1]], dtype=np.int32)
        obs = tracker.compute_obs(tokens)

        assert obs[0, 0, 0] == lib.EMPTY_ACTION
        assert obs[0, 1, 0] == lib.EMPTY_PARENT
        assert obs[0, 2, 0] == lib.EMPTY_SIBLING
        assert obs[0, 3, 0] == 1

    @pytest.mark.unit
    def test_batch_dimensions(self) -> None:
        lib = _make_test_library()
        tracker = BatchTracker(lib)

        tokens = np.array(
            [[ADD, X1, U1], [MUL, X1, U1]], dtype=np.int32
        )
        obs = tracker.compute_obs(tokens)

        assert obs.shape == (2, 4, 3)

    @pytest.mark.unit
    def test_batch_mixed_completion(self) -> None:
        lib = _make_test_library()
        tracker = BatchTracker(lib)

        tokens = np.array(
            [
                [ADD, X1, U1],
                [N2, X1, X1],
            ],
            dtype=np.int32,
        )
        obs = tracker.compute_obs(tokens)
        assert obs.shape == (2, 4, 3)


        assert obs[0, 3, 0] == 1
        assert obs[0, 3, 1] == 2
        assert obs[0, 3, 2] == 1


        assert obs[1, 3, 0] == 1
        assert obs[1, 3, 1] == 1
        assert obs[1, 3, 2] == 0



        assert obs[1, 1, 2] == lib.EMPTY_PARENT
        assert obs[1, 2, 2] == lib.EMPTY_SIBLING

    @pytest.mark.unit
    def test_handles_empty_action_padding(self) -> None:
        lib = _make_test_library()
        tracker = BatchTracker(lib)
        empty = lib.EMPTY_ACTION


        tokens = np.array([[N2, X1, empty, empty, empty]], dtype=np.int32)
        obs = tracker.compute_obs(tokens)
        assert obs.shape == (1, 4, 5)


        assert obs[0, 3, 0] == 1
        assert obs[0, 3, 1] == 1
        assert obs[0, 3, 2] == 0







class TestTreeStateInvariants:

    @pytest.mark.parametrize("seq", VALID_SEQS, ids=VALID_SEQ_IDS)
    def test_dangling_zero_iff_complete(self, seq: list[int]) -> None:
        lib = _make_test_library()
        tracker = BatchTracker(lib)

        tokens = np.array([seq], dtype=np.int32)
        obs = tracker.compute_obs(tokens)


        last_dangling = obs[0, 3, -1]
        last_arity = lib.arities[tokens[0, -1]]
        final_dangling = last_dangling + last_arity - 1
        assert final_dangling == 0

    @pytest.mark.parametrize("seq", VALID_SEQS, ids=VALID_SEQ_IDS)
    def test_parent_is_nonterminal_or_empty(self, seq: list[int]) -> None:
        lib = _make_test_library()
        tracker = BatchTracker(lib)

        tokens = np.array([seq], dtype=np.int32)
        obs = tracker.compute_obs(tokens)

        parents = obs[0, 1,:].astype(np.int32)
        max_valid_parent = lib.EMPTY_PARENT
        for p in parents:
            assert 0 <= p <= max_valid_parent, (
                f"Invalid parent: {p} (max valid = {max_valid_parent})"
            )

    @pytest.mark.parametrize("seq", VALID_SEQS, ids=VALID_SEQ_IDS)
    def test_dangling_always_positive_before_completion(
        self, seq: list[int]
    ) -> None:
        lib = _make_test_library()
        tracker = BatchTracker(lib)

        tokens = np.array([seq], dtype=np.int32)
        obs = tracker.compute_obs(tokens)

        danglings = obs[0, 3,:]
        assert np.all(danglings >= 1), (
            f"Dangling < 1 at some position: {danglings}"
        )

    @pytest.mark.parametrize("seq", VALID_SEQS, ids=VALID_SEQ_IDS)
    def test_sibling_is_valid_token_or_empty(self, seq: list[int]) -> None:
        lib = _make_test_library()
        tracker = BatchTracker(lib)

        tokens = np.array([seq], dtype=np.int32)
        obs = tracker.compute_obs(tokens)

        siblings = obs[0, 2,:].astype(np.int32)
        max_valid = lib.EMPTY_SIBLING
        for s in siblings:
            assert 0 <= s <= max_valid, (
                f"Invalid sibling: {s} (max valid = {max_valid})"
            )

    def test_sibling_across_completed_subtree(self) -> None:
        lib = _make_test_library()

        tokens = np.array([[MUL, N2, X1, U1]], dtype=np.int32)
        obs = BatchTracker(lib).compute_obs(tokens)
        assert obs[0, 2, 3] == N2

    def test_sibling_across_deep_completed_subtree(self) -> None:
        lib = _make_test_library()
        tokens = np.array([[ADD, ADD, X1, U1, U1]], dtype=np.int32)
        obs = BatchTracker(lib).compute_obs(tokens)
        assert obs[0, 2, 4] == ADD

    def test_incomplete_sequence_has_positive_final_dangling(self) -> None:
        lib = _make_test_library()
        tracker = BatchTracker(lib)


        tokens = np.array([[ADD, X1]], dtype=np.int32)
        obs = tracker.compute_obs(tokens)

        last_dangling = obs[0, 3, -1]
        last_arity = lib.arities[tokens[0, -1]]
        final_dangling = last_dangling + last_arity - 1
        assert final_dangling == 1







def _extended_arrays(lib: Library) -> tuple[np.ndarray, np.ndarray]:
    arities = np.append(lib.arities, np.int32(0))
    parent_adjust = np.append(lib.parent_adjust, np.int32(lib.EMPTY_PARENT))
    return arities, parent_adjust


class TestParentsSiblingsPureFunction:

    @pytest.mark.unit
    def test_single_row_operator_last(self) -> None:
        lib = _make_test_library()
        arities, parent_adjust = _extended_arrays(lib)

        tokens = np.array([[ADD]], dtype=np.int32)
        parents, siblings = parents_siblings(
            tokens, arities, parent_adjust, lib.EMPTY_PARENT, lib.EMPTY_SIBLING
        )

        assert parents.shape == (1,)
        assert siblings.shape == (1,)
        assert parents.dtype == np.int32
        assert siblings.dtype == np.int32
        assert parents[0] == lib.parent_adjust[ADD]
        assert siblings[0] == lib.EMPTY_SIBLING

    @pytest.mark.unit
    def test_single_row_terminal_last_backward_scan(self) -> None:
        lib = _make_test_library()
        arities, parent_adjust = _extended_arrays(lib)


        tokens = np.array([[ADD, X1]], dtype=np.int32)
        parents, siblings = parents_siblings(
            tokens, arities, parent_adjust, lib.EMPTY_PARENT, lib.EMPTY_SIBLING
        )

        assert parents[0] == lib.parent_adjust[ADD]
        assert siblings[0] == X1

    @pytest.mark.unit
    def test_deep_binary_prefix(self) -> None:
        lib = _make_test_library()
        arities, parent_adjust = _extended_arrays(lib)




        tokens = np.array([[ADD, MUL, X1, U1]], dtype=np.int32)
        parents, siblings = parents_siblings(
            tokens, arities, parent_adjust, lib.EMPTY_PARENT, lib.EMPTY_SIBLING
        )

        assert parents[0] == lib.parent_adjust[ADD]
        assert siblings[0] == MUL

    @pytest.mark.unit
    def test_unary_last_is_operator_short_circuit(self) -> None:
        lib = _make_test_library()
        arities, parent_adjust = _extended_arrays(lib)

        tokens = np.array([[N2]], dtype=np.int32)
        parents, siblings = parents_siblings(
            tokens, arities, parent_adjust, lib.EMPTY_PARENT, lib.EMPTY_SIBLING
        )

        assert parents[0] == lib.parent_adjust[N2]
        assert siblings[0] == lib.EMPTY_SIBLING

    @pytest.mark.unit
    def test_multi_row_mixed_arities(self) -> None:
        lib = _make_test_library()
        arities, parent_adjust = _extended_arrays(lib)







        tokens = np.array(
            [
                [ADD, X1],
                [MUL, U1],
                [N2, X1],
            ],
            dtype=np.int32,
        )
        parents, siblings = parents_siblings(
            tokens, arities, parent_adjust, lib.EMPTY_PARENT, lib.EMPTY_SIBLING
        )

        assert parents[0] == lib.parent_adjust[ADD]
        assert siblings[0] == X1
        assert parents[1] == lib.parent_adjust[MUL]
        assert siblings[1] == U1
        assert parents[2] == lib.EMPTY_PARENT
        assert siblings[2] == lib.EMPTY_SIBLING

    @pytest.mark.unit
    def test_diff_operator_library(self) -> None:
        lib = _make_diff_library()
        arities, parent_adjust = _extended_arrays(lib)


        d_x1, d_u1, d_add, _d_mul, d_diff, _d_diff2 = range(6)




        tokens = np.array([[d_add, d_diff, d_x1]], dtype=np.int32)
        parents, siblings = parents_siblings(
            tokens, arities, parent_adjust, lib.EMPTY_PARENT, lib.EMPTY_SIBLING
        )

        assert parents[0] == lib.parent_adjust[d_add]
        assert siblings[0] == d_diff


        assert d_u1 == 1

    @pytest.mark.unit
    def test_empty_action_padding_safe(self) -> None:
        lib = _make_test_library()
        arities, parent_adjust = _extended_arrays(lib)
        empty = lib.EMPTY_ACTION


        tokens = np.array([[N2, X1, empty, empty, empty]], dtype=np.int32)
        parents, siblings = parents_siblings(
            tokens, arities, parent_adjust, lib.EMPTY_PARENT, lib.EMPTY_SIBLING
        )

        assert parents[0] == lib.EMPTY_PARENT
        assert siblings[0] == lib.EMPTY_SIBLING

    @pytest.mark.unit
    def test_empty_tokens_raises(self) -> None:
        lib = _make_test_library()
        arities, parent_adjust = _extended_arrays(lib)

        tokens = np.zeros((3, 0), dtype=np.int32)
        with pytest.raises(ValueError):
            parents_siblings(
                tokens,
                arities,
                parent_adjust,
                lib.EMPTY_PARENT,
                lib.EMPTY_SIBLING,
            )

    @pytest.mark.unit
    def test_wrong_ndim_raises(self) -> None:
        lib = _make_test_library()
        arities, parent_adjust = _extended_arrays(lib)

        tokens = np.array([ADD, X1, U1], dtype=np.int32)
        with pytest.raises(ValueError):
            parents_siblings(
                tokens,
                arities,
                parent_adjust,
                lib.EMPTY_PARENT,
                lib.EMPTY_SIBLING,
            )

    @pytest.mark.unit
    def test_non_int32_tokens_raise_clear_error(self) -> None:
        lib = _make_test_library()
        arities, parent_adjust = _extended_arrays(lib)

        tokens = np.array([[ADD, X1, U1]], dtype=np.int64)
        with pytest.raises(ValueError, match="int32"):
            parents_siblings(
                tokens,
                arities,
                parent_adjust,
                lib.EMPTY_PARENT,
                lib.EMPTY_SIBLING,
            )

    @pytest.mark.unit
    def test_unextended_arrays_raise_clear_error(self) -> None:
        lib = _make_test_library()
        tokens = np.array([[N2, X1, lib.EMPTY_ACTION]], dtype=np.int32)

        with pytest.raises(ValueError, match="EMPTY_ACTION"):
            parents_siblings(
                tokens,
                lib.arities,
                lib.parent_adjust,
                lib.EMPTY_PARENT,
                lib.EMPTY_SIBLING,
            )







class TestIncrementalTrackerPreallocated:

    @pytest.mark.unit
    def test_constructor_with_max_length(self) -> None:
        lib = _make_test_library()
        IncrementalTracker(lib, max_length=10)

    @pytest.mark.unit
    def test_reset_preallocates_buffer(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib, max_length=10)

        tracker.reset(batch_size=3)
        hist = tracker.history
        assert hist.shape == (3, 0)
        assert hist.dtype == np.int32

    @pytest.mark.unit
    def test_step_fills_buffer_without_concat(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib, max_length=10)



        actions_seq = [
            (ADD, MUL),
            (X1, U1),
            (U1, X1),
        ]

        tracker.reset(batch_size=2)




        first_a, second_a = actions_seq[0]
        tracker.step(np.array([first_a, second_a], dtype=np.int32))
        hist = tracker.history
        assert hist.shape == (2, 1)
        assert hist.dtype == np.int32
        assert hist.base is not None, (
            "Pre-allocated path must return a view into the preallocated buffer."
        )
        initial_buffer_id = id(hist.base)
        assert int(hist[0, 0]) == first_a
        assert int(hist[1, 0]) == second_a



        for t_index, (a0, a1) in enumerate(actions_seq[1:], start=1):
            tracker.step(np.array([a0, a1], dtype=np.int32))
            hist = tracker.history
            assert hist.shape == (2, t_index + 1)
            assert hist.base is not None
            assert id(hist.base) == initial_buffer_id, (
                f"Buffer re-allocated at step {t_index}; expected in-place write."
            )
            assert int(hist[0, t_index]) == a0, (
                f"Row 0 column {t_index} not written in place."
            )
            assert int(hist[1, t_index]) == a1, (
                f"Row 1 column {t_index} not written in place."
            )

    @pytest.mark.unit
    def test_step_beyond_max_length_raises(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib, max_length=3)

        tracker.reset(batch_size=1)
        for a in [ADD, ADD, X1]:
            tracker.step(np.array([a], dtype=np.int32))

        with pytest.raises(RuntimeError, match="max_length"):
            tracker.step(np.array([X1], dtype=np.int32))

    @pytest.mark.unit
    def test_reset_reuses_buffer(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib, max_length=5)

        tracker.reset(batch_size=3)
        first_base = tracker.history.base
        assert first_base is not None
        first_buffer_id = id(first_base)


        tracker.step(np.array([ADD, MUL, N2], dtype=np.int32))
        tracker.reset(batch_size=3)

        second_base = tracker.history.base
        assert second_base is not None
        assert id(second_base) == first_buffer_id, (
            "Reset with same batch_size must reuse buffer, not allocate a new one."
        )


        assert tracker.history.shape == (3, 0)

    @pytest.mark.unit
    def test_reset_with_different_batch_size_reallocates(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib, max_length=5)

        tracker.reset(batch_size=3)
        first_base = tracker.history.base
        assert first_base is not None
        first_shape = first_base.shape

        tracker.reset(batch_size=5)
        second_base = tracker.history.base
        assert second_base is not None


        assert first_shape[0] == 3
        assert second_base.shape[0] == 5

        assert id(second_base) != id(first_base)

    @pytest.mark.unit
    def test_reset_ping_pong_reuses_seen_batch_buffer(self) -> None:
        lib = _make_test_library()
        tracker = IncrementalTracker(lib, max_length=5)

        tracker.reset(batch_size=3)
        first_base = tracker.history.base
        assert first_base is not None

        tracker.reset(batch_size=5)
        second_base = tracker.history.base
        assert second_base is not None
        assert id(second_base) != id(first_base)

        tracker.reset(batch_size=3)
        third_base = tracker.history.base
        assert third_base is not None
        assert id(third_base) == id(first_base), (
            "Tracker should cache previously seen batch-size buffers."
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "seq",
        [
            [ADD, X1, U1],
            [N2, X1],
            [ADD, MUL, U1, N2, U1, N2, U1],
            [SUB, ADD, X1, U1, MUL, U1, X1],
        ],
        ids=["binary", "unary", "nested", "with_sub"],
    )
    def test_preallocated_equals_legacy(self, seq: list[int]) -> None:
        lib = _make_test_library()
        tokens = np.array([seq], dtype=np.int32)
        B, L = tokens.shape

        legacy = IncrementalTracker(lib)
        prealloc = IncrementalTracker(lib, max_length=L)

        obs_legacy_init = legacy.reset(B)
        obs_prealloc_init = prealloc.reset(B)
        np.testing.assert_array_equal(obs_legacy_init, obs_prealloc_init)

        for t in range(L - 1):
            obs_legacy = legacy.step(tokens[:, t])
            obs_prealloc = prealloc.step(tokens[:, t])
            np.testing.assert_array_equal(
                obs_legacy,
                obs_prealloc,
                err_msg=f"Pre-alloc/legacy mismatch at step {t} for seq {seq}",
            )







class TestIncrementalTrackerBufferCacheCap:

    @pytest.mark.unit
    def test_buffer_cache_lru_hit_moves_to_end(self) -> None:
        from kd.search.discover.controller.tree_state import (
            _BUFFER_CACHE_MAX_ENTRIES,
        )

        lib = _make_test_library()
        tracker = IncrementalTracker(lib, max_length=8)


        for bs in range(1, _BUFFER_CACHE_MAX_ENTRIES + 1):
            tracker.reset(bs)
        assert list(tracker._buffer_cache.keys()) == list(
            range(1, _BUFFER_CACHE_MAX_ENTRIES + 1),
        )


        tracker.reset(1)
        expected_after_hit = list(range(2, _BUFFER_CACHE_MAX_ENTRIES + 1)) + [1]
        assert list(tracker._buffer_cache.keys()) == expected_after_hit, (
            "LRU hit must move the touched entry to the MRU end."
        )


        new_bs = _BUFFER_CACHE_MAX_ENTRIES + 2
        tracker.reset(new_bs)
        assert 2 not in tracker._buffer_cache, (
            "After the hit promoted key=1, key=2 must be the one evicted."
        )
        assert 1 in tracker._buffer_cache, (
            "key=1 was just re-reset; it must survive the next eviction."
        )
        assert new_bs in tracker._buffer_cache

    @pytest.mark.unit
    def test_buffer_cache_capped_and_warns(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        from kd.search.discover.controller.tree_state import (
            _BUFFER_CACHE_MAX_ENTRIES,
        )

        lib = _make_test_library()
        tracker = IncrementalTracker(lib, max_length=8)
        batch_sizes = list(range(1, _BUFFER_CACHE_MAX_ENTRIES + 3))

        with caplog.at_level(
            logging.WARNING, logger="kd.search.discover.controller.tree_state",
        ):
            for bs in batch_sizes:
                tracker.reset(bs)

        assert len(tracker._buffer_cache) <= _BUFFER_CACHE_MAX_ENTRIES, (
            f"cache should stay bounded at {_BUFFER_CACHE_MAX_ENTRIES}, "
            f"got {len(tracker._buffer_cache)}."
        )
        assert any(
            "buffer_cache" in rec.message.lower()
            and "batch" in rec.message.lower()
            for rec in caplog.records
        ), (
            "Expected a warning when the buffer cache is capped; "
            f"got: {[r.message for r in caplog.records]}"
        )
