
from __future__ import annotations

import numpy as np
import pytest
import torch

from kd.search.discover.controller.lstm import LSTMController
from kd.search.discover.core.batch import Batch
from kd.search.discover.search import rebuild_batch
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import (
    DiffChildConstraint,
    LengthConstraint,
    Prior,
    PriorContext,
    PriorSystem,
)

BURGERS_CONFIG = LibraryConfig(
    operators=["add", "mul", "sub", "div", "sin", "cos", "diff_x", "diff2_x"],
    state_vars=["u"],
    coord_vars=["x", "t"],
)

MAX_LENGTH = 30
MIN_LENGTH = 4
BATCH_SIZE = 16
ATOL = 1e-6
MULTI_SAMPLE_SEEDS = (7, 17, 29, 41, 53)


@pytest.fixture
def lib() -> Library:
    return Library.from_config(BURGERS_CONFIG)


@pytest.fixture
def prior_system(lib: Library) -> PriorSystem:
    return PriorSystem(
        lib,
        [
            LengthConstraint(lib, min_=MIN_LENGTH, max_=MAX_LENGTH),
            DiffChildConstraint(lib),
        ],
    )


@pytest.fixture
def controller(lib: Library, prior_system: PriorSystem) -> LSTMController:
    return LSTMController(
        library=lib,
        prior_system=prior_system,
        num_units=32,
        num_layers=1,
        embedding_dim=8,
    )


def _sample_and_rebuild(
    controller: LSTMController,
    lib: Library,
    prior_system: PriorSystem,
    batch_size: int,
    seed: int,
) -> tuple[Batch, Batch, np.ndarray]:
    torch.manual_seed(seed)
    batch = controller.sample(batch_size)
    rebuilt, valid_mask = rebuild_batch(batch.actions, lib, prior_system)
    return batch, rebuilt, valid_mask


def _same_arity_replacement(library: Library, token_index: int) -> int:
    arity = int(library.arities[token_index])
    for candidate in range(len(library.tokens)):
        if candidate == token_index:
            continue
        if int(library.arities[candidate]) == arity:
            return candidate
    raise AssertionError(f"No same-arity replacement for token {token_index}.")


class TestRebuildBatchEquivalence:

    @pytest.mark.unit
    def test_rebuild_matches_controller_sample(
        self,
        controller: LSTMController,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        batch, rebuilt, valid_mask = _sample_and_rebuild(
            controller,
            lib,
            prior_system,
            batch_size=BATCH_SIZE,
            seed=12345,
        )

        np.testing.assert_array_equal(batch.actions, rebuilt.actions)
        np.testing.assert_array_equal(batch.obs, rebuilt.obs)
        np.testing.assert_allclose(batch.priors, rebuilt.priors, atol=ATOL)
        np.testing.assert_array_equal(batch.lengths, rebuilt.lengths)
        assert valid_mask.all()

    @pytest.mark.unit
    def test_rebuild_matches_after_multiple_samples(
        self,
        controller: LSTMController,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        for seed in MULTI_SAMPLE_SEEDS:
            batch, rebuilt, valid_mask = _sample_and_rebuild(
                controller,
                lib,
                prior_system,
                batch_size=BATCH_SIZE,
                seed=seed,
            )
            np.testing.assert_array_equal(batch.actions, rebuilt.actions)
            np.testing.assert_array_equal(batch.obs, rebuilt.obs)
            np.testing.assert_allclose(batch.priors, rebuilt.priors, atol=ATOL)
            np.testing.assert_array_equal(batch.lengths, rebuilt.lengths)
            assert valid_mask.all()


class TestRebuildBatchShapeAndDtype:

    @pytest.mark.unit
    def test_returns_batch_instance(
        self,
        controller: LSTMController,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        _batch, rebuilt, _ = _sample_and_rebuild(
            controller,
            lib,
            prior_system,
            batch_size=BATCH_SIZE,
            seed=101,
        )
        assert isinstance(rebuilt, Batch)

    @pytest.mark.unit
    def test_obs_shape(
        self,
        controller: LSTMController,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        _batch, rebuilt, _ = _sample_and_rebuild(
            controller,
            lib,
            prior_system,
            batch_size=BATCH_SIZE,
            seed=102,
        )
        batch_size, sequence_length = rebuilt.actions.shape
        assert rebuilt.obs.shape == (batch_size, 4, sequence_length)
        assert rebuilt.obs.dtype == np.float32

    @pytest.mark.unit
    def test_priors_shape(
        self,
        controller: LSTMController,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        _batch, rebuilt, _ = _sample_and_rebuild(
            controller,
            lib,
            prior_system,
            batch_size=BATCH_SIZE,
            seed=103,
        )
        batch_size, sequence_length = rebuilt.actions.shape
        assert rebuilt.priors.shape == (batch_size, sequence_length, len(lib.tokens))
        assert rebuilt.priors.dtype == np.float32

    @pytest.mark.unit
    def test_lengths_shape(
        self,
        controller: LSTMController,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        _batch, rebuilt, _ = _sample_and_rebuild(
            controller,
            lib,
            prior_system,
            batch_size=BATCH_SIZE,
            seed=104,
        )
        assert rebuilt.lengths.shape == (BATCH_SIZE,)
        assert rebuilt.lengths.dtype == np.int32

    @pytest.mark.unit
    def test_lengths_are_positive(
        self,
        controller: LSTMController,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        _batch, rebuilt, _ = _sample_and_rebuild(
            controller,
            lib,
            prior_system,
            batch_size=BATCH_SIZE,
            seed=105,
        )
        assert np.all(rebuilt.lengths > 0)


class TestRebuildBatchEdgeCases:

    @pytest.mark.unit
    def test_wrong_ndim_raises(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        actions = np.array([0, 1], dtype=np.int32)
        with pytest.raises(ValueError, match="shape"):
            rebuild_batch(actions, lib, prior_system)

    @pytest.mark.unit
    def test_wrong_dtype_raises(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        actions = np.zeros((2, 3), dtype=np.int64)
        with pytest.raises(ValueError, match="dtype"):

            rebuild_batch(actions, lib, prior_system)

    @pytest.mark.unit
    def test_empty_length_raises(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        actions = np.zeros((2, 0), dtype=np.int32)
        with pytest.raises(ValueError, match="at least one token"):
            rebuild_batch(actions, lib, prior_system)

    @pytest.mark.unit
    def test_empty_batch_raises(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        actions = np.zeros((0, 2), dtype=np.int32)

        with pytest.raises(ValueError, match="at least one row"):
            rebuild_batch(actions, lib, prior_system)

    @pytest.mark.unit
    def test_negative_token_index_raises(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        actions = np.array([[-1, lib.EMPTY_ACTION]], dtype=np.int32)

        with pytest.raises(ValueError, match="token indices"):
            rebuild_batch(actions, lib, prior_system)

    @pytest.mark.unit
    def test_token_index_above_empty_action_raises(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        actions = np.array(
            [[lib.EMPTY_ACTION + 1, lib.EMPTY_ACTION]],
            dtype=np.int32,
        )

        with pytest.raises(ValueError, match="token indices"):
            rebuild_batch(actions, lib, prior_system)

    @pytest.mark.unit
    def test_mixed_library_instances_raise(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        other_lib = Library.from_config(BURGERS_CONFIG)
        actions = np.array(
            [[other_lib.name_to_index("u"), other_lib.EMPTY_ACTION]],
            dtype=np.int32,
        )

        with pytest.raises(ValueError, match="prior system library"):
            rebuild_batch(actions, other_lib, prior_system)

    @pytest.mark.unit
    def test_single_row_batch(
        self,
        controller: LSTMController,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        batch, rebuilt, valid_mask = _sample_and_rebuild(
            controller,
            lib,
            prior_system,
            batch_size=1,
            seed=201,
        )
        assert rebuilt.actions.shape[0] == 1
        np.testing.assert_array_equal(batch.obs, rebuilt.obs)
        np.testing.assert_array_equal(batch.lengths, rebuilt.lengths)
        assert valid_mask.all()

    @pytest.mark.unit
    def test_padded_rows(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        sin_idx = lib.name_to_index("sin")
        u_idx = lib.name_to_index("u")
        empty = lib.EMPTY_ACTION
        actions = np.array([[sin_idx, u_idx, empty, empty]], dtype=np.int32)

        rebuilt, valid_mask = rebuild_batch(actions, lib, prior_system)

        np.testing.assert_array_equal(rebuilt.lengths, np.array([2], dtype=np.int32))
        assert rebuilt.obs.shape == (1, 4, 4)
        assert rebuilt.priors.shape == (1, 4, len(lib.tokens))
        assert valid_mask.tolist() == [True]

    @pytest.mark.unit
    def test_rebuild_batch_with_adversarial_finished_row_prior(
        self,
        lib: Library,
    ) -> None:

        class _DeadOnEmptyActionHistoryPrior(Prior):
            def __call__(self, ctx: PriorContext) -> np.ndarray:
                adjustment = np.zeros(
                    (ctx.actions.shape[0], self.n_choices),
                    dtype=np.float32,
                )
                if ctx.actions.shape[1] == 0:
                    return adjustment
                last_token = ctx.actions[:, -1]
                mask = last_token == self.library.EMPTY_ACTION
                if np.any(mask):
                    adjustment[mask,:] = -np.inf
                return adjustment

        adv_prior_system = PriorSystem(
            lib,
            [
                LengthConstraint(lib, min_=MIN_LENGTH, max_=MAX_LENGTH),
                DiffChildConstraint(lib),
                _DeadOnEmptyActionHistoryPrior(lib),
            ],
        )

















        diff_x = lib.name_to_index("diff_x")
        u_idx = lib.name_to_index("u")
        add_idx = lib.name_to_index("add")
        empty = lib.EMPTY_ACTION
        actions = np.array(
            [
                [diff_x, u_idx, empty, empty, empty, empty],
                [add_idx, diff_x, u_idx, diff_x, diff_x, u_idx],
            ],
            dtype=np.int32,
        )

        rebuilt, valid_mask = rebuild_batch(actions, lib, adv_prior_system)

        assert isinstance(rebuilt, Batch)
        assert rebuilt.actions.shape == (2, 6)
        assert int(rebuilt.lengths[0]) == 2
        assert int(rebuilt.lengths[1]) == 6
        assert valid_mask.tolist() == [True, True]






        np.testing.assert_array_equal(
            rebuilt.priors[0, 2:,:],
            np.zeros((4, len(lib.tokens)), dtype=np.float32),
        )
        assert not np.any(np.isnan(rebuilt.priors))


class TestRebuildBatchGASimulation:

    @pytest.mark.unit
    def test_mutation_workflow(
        self,
        controller: LSTMController,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        torch.manual_seed(303)
        batch = controller.sample(8)
        mutated = batch.actions.copy()
        replacement = _same_arity_replacement(lib, int(mutated[0, 0]))
        mutated[0, 0] = replacement

        rebuilt, valid_mask = rebuild_batch(mutated, lib, prior_system)

        assert isinstance(rebuilt, Batch)
        assert rebuilt.actions.shape == batch.actions.shape
        assert rebuilt.obs.shape == batch.obs.shape
        assert rebuilt.priors.shape == batch.priors.shape
        assert rebuilt.lengths.shape == batch.lengths.shape
        assert rebuilt.actions[0, 0] == replacement
        assert int(rebuilt.lengths[0]) > 0
        assert not np.array_equal(rebuilt.obs[0], batch.obs[0])
        assert valid_mask.all()


class TestRebuildBatchMalformedRows:

    @pytest.mark.unit
    def test_malformed_row_does_not_raise(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        add_idx = lib.name_to_index("add")
        width = 6


        actions = np.full((1, width), add_idx, dtype=np.int32)

        rebuilt, valid_mask = rebuild_batch(actions, lib, prior_system)

        assert isinstance(rebuilt, Batch)
        assert valid_mask.shape == (1,)
        assert valid_mask.dtype == np.bool_
        assert not bool(valid_mask[0])

    @pytest.mark.unit
    def test_malformed_row_length_is_zero(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        add_idx = lib.name_to_index("add")
        actions = np.full((1, 5), add_idx, dtype=np.int32)

        rebuilt, _valid_mask = rebuild_batch(actions, lib, prior_system)

        assert int(rebuilt.lengths[0]) == 0

    @pytest.mark.unit
    def test_malformed_row_after_empty_padding(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        add_idx = lib.name_to_index("add")
        u_idx = lib.name_to_index("u")
        empty = lib.EMPTY_ACTION

        actions = np.array(
            [[add_idx, add_idx, u_idx, u_idx, empty, empty]], dtype=np.int32
        )

        rebuilt, valid_mask = rebuild_batch(actions, lib, prior_system)

        assert not bool(valid_mask[0])
        assert int(rebuilt.lengths[0]) == 0

    @pytest.mark.unit
    def test_mixed_batch_preserves_valid_row_outputs(
        self,
        controller: LSTMController,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        torch.manual_seed(404)
        baseline = controller.sample(BATCH_SIZE)

        add_idx = lib.name_to_index("add")
        malformed_row = np.full(
            (1, baseline.actions.shape[1]), add_idx, dtype=np.int32
        )
        mixed_actions = np.concatenate([malformed_row, baseline.actions[1:]], axis=0)

        rebuilt_mixed, valid_mask = rebuild_batch(mixed_actions, lib, prior_system)


        expected_mask = np.array([False] + [True] * (BATCH_SIZE - 1))
        np.testing.assert_array_equal(valid_mask, expected_mask)


        assert int(rebuilt_mixed.lengths[0]) == 0


        np.testing.assert_array_equal(
            rebuilt_mixed.actions[1:], baseline.actions[1:]
        )
        np.testing.assert_array_equal(
            rebuilt_mixed.lengths[1:], baseline.lengths[1:]
        )
        np.testing.assert_array_equal(rebuilt_mixed.obs[1:], baseline.obs[1:])
        np.testing.assert_allclose(
            rebuilt_mixed.priors[1:], baseline.priors[1:], atol=ATOL
        )

    @pytest.mark.unit
    def test_all_rows_malformed(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        add_idx = lib.name_to_index("add")
        actions = np.full((4, 6), add_idx, dtype=np.int32)

        rebuilt, valid_mask = rebuild_batch(actions, lib, prior_system)

        assert isinstance(rebuilt, Batch)
        assert valid_mask.shape == (4,)
        assert not valid_mask.any()
        assert np.all(rebuilt.lengths == 0)

    @pytest.mark.unit
    def test_valid_mask_shape_and_dtype(
        self,
        controller: LSTMController,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        torch.manual_seed(505)
        batch = controller.sample(8)
        _rebuilt, valid_mask = rebuild_batch(batch.actions, lib, prior_system)

        assert valid_mask.shape == (8,)
        assert valid_mask.dtype == np.bool_





class TestRebuildBatchHeterogeneousLengths:

    @pytest.mark.unit
    def test_rows_completing_at_different_positions(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        u_idx = lib.name_to_index("u")
        add_idx = lib.name_to_index("add")
        sin_idx = lib.name_to_index("sin")
        empty = lib.EMPTY_ACTION
        width = 6








        actions = np.array(
            [
                [u_idx, empty, empty, empty, empty, empty],
                [sin_idx, u_idx, empty, empty, empty, empty],
                [add_idx, u_idx, u_idx, empty, empty, empty],
                [add_idx, sin_idx, u_idx, u_idx, empty, empty],
                [add_idx, add_idx, u_idx, u_idx, sin_idx, u_idx],
                [add_idx, add_idx, u_idx, u_idx, u_idx, u_idx],
            ],
            dtype=np.int32,
        )




        expected_lengths = np.array([1, 2, 3, 4, 6, 5], dtype=np.int32)

        rebuilt, valid_mask = rebuild_batch(actions, lib, prior_system)

        np.testing.assert_array_equal(rebuilt.lengths, expected_lengths)
        np.testing.assert_array_equal(valid_mask, np.ones(width, dtype=np.bool_))

    @pytest.mark.unit
    def test_padding_before_completion_invalidates_row(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        add_idx = lib.name_to_index("add")
        u_idx = lib.name_to_index("u")
        empty = lib.EMPTY_ACTION


        actions = np.array(
            [[add_idx, empty, u_idx, u_idx]],
            dtype=np.int32,
        )

        rebuilt, valid_mask = rebuild_batch(actions, lib, prior_system)

        assert not bool(valid_mask[0])
        assert int(rebuilt.lengths[0]) == 0

    @pytest.mark.unit
    def test_valid_row_with_trailing_garbage_keeps_natural_length(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        u_idx = lib.name_to_index("u")
        add_idx = lib.name_to_index("add")



        actions = np.array([[u_idx, add_idx, u_idx, u_idx]], dtype=np.int32)

        rebuilt, valid_mask = rebuild_batch(actions, lib, prior_system)

        assert int(rebuilt.lengths[0]) == 1
        assert bool(valid_mask[0])

    @pytest.mark.unit
    def test_batch_mixing_valid_and_invalid_rows(
        self,
        lib: Library,
        prior_system: PriorSystem,
    ) -> None:
        u_idx = lib.name_to_index("u")
        add_idx = lib.name_to_index("add")
        empty = lib.EMPTY_ACTION
        actions = np.array(
            [
                [u_idx, empty, empty, empty],
                [add_idx, empty, empty, empty],
                [add_idx, u_idx, u_idx, empty],
                [add_idx, add_idx, add_idx, u_idx],
            ],
            dtype=np.int32,
        )

        rebuilt, valid_mask = rebuild_batch(actions, lib, prior_system)

        np.testing.assert_array_equal(
            rebuilt.lengths,
            np.array([1, 0, 3, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            valid_mask, np.array([True, False, True, False])
        )
