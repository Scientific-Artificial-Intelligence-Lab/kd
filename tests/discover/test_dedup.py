
from __future__ import annotations

import numpy as np
import pytest

from kd.search.discover.evaluation.dedup import Deduplicator
from kd.search.discover.tokens.library import Library, LibraryConfig



SMALL_CONFIG = LibraryConfig(
    operators=["add", "sin", "mul"],
    state_vars=["u1"],
    coord_vars=["x1"],
)



@pytest.fixture
def lib() -> Library:
    return Library.from_config(SMALL_CONFIG)


@pytest.fixture
def dedup(lib: Library) -> Deduplicator:
    return Deduplicator(lib)




SEQ_LEN = 6


def _pad_row(tokens: list[int], length: int, pad_value: int) -> list[int]:
    assert len(tokens) <= length, f"tokens length {len(tokens)} > {length}"
    return tokens + [pad_value] * (length - len(tokens))


def _make_batch(
    rows: list[list[int]], pad_value: int, length: int = SEQ_LEN
) -> np.ndarray:
    return np.array(
        [_pad_row(row, length, pad_value) for row in rows],
        dtype=np.int32,
    )





class TestDeduplicateUnit:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_all_unique_returns_identity_scatter(
        self, dedup: Deduplicator, lib: Library
    ) -> None:

        batch = _make_batch([[0], [1], [3, 0]], lib.EMPTY_ACTION)
        unique_ir, scatter = dedup.deduplicate(batch)

        assert len(unique_ir) == 3

        assert len(set(scatter.tolist())) == 3

        expected = ["x1", "u1", "sin(x1)"]
        for i, ir in enumerate(expected):
            assert unique_ir[scatter[i]] == ir

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_all_identical_returns_single_unique(
        self, dedup: Deduplicator, lib: Library
    ) -> None:
        batch = _make_batch([[3, 0], [3, 0], [3, 0]], lib.EMPTY_ACTION)
        unique_ir, scatter = dedup.deduplicate(batch)

        assert len(unique_ir) == 1
        assert unique_ir[0] == "sin(x1)"
        np.testing.assert_array_equal(scatter, [0, 0, 0])

    @pytest.mark.unit
    def test_mixed_duplicates_correct_grouping(
        self, dedup: Deduplicator, lib: Library
    ) -> None:

        batch = _make_batch(
            [[3, 0], [0], [3, 0], [1], [0]], lib.EMPTY_ACTION
        )
        unique_ir, scatter = dedup.deduplicate(batch)

        assert len(unique_ir) == 3

        assert scatter[0] == scatter[2]

        assert scatter[1] == scatter[4]

        assert len({scatter[0], scatter[1], scatter[3]}) == 3

    @pytest.mark.unit
    def test_same_prefix_different_trailing_garbage_deduplicates(
        self, dedup: Deduplicator, lib: Library
    ) -> None:
        ea = lib.EMPTY_ACTION
        batch = np.array(
            [
                [0, ea, ea, ea, ea, ea],
                [0, 0, 1, ea, ea, ea],
            ],
            dtype=np.int32,
        )
        unique_ir, scatter = dedup.deduplicate(batch)

        assert len(unique_ir) == 1
        assert unique_ir[0] == "x1"
        assert scatter[0] == scatter[1]

    @pytest.mark.unit
    def test_empty_batch(self, dedup: Deduplicator) -> None:
        batch = np.empty((0, SEQ_LEN), dtype=np.int32)
        unique_ir, scatter = dedup.deduplicate(batch)

        assert len(unique_ir) == 0
        assert scatter.shape == (0,)

    @pytest.mark.unit
    def test_incomplete_row_raises(
        self, dedup: Deduplicator, lib: Library
    ) -> None:
        ea = lib.EMPTY_ACTION

        batch = np.array([[2, 0, ea, ea, ea, ea]], dtype=np.int32)
        with pytest.raises(ValueError, match="Incomplete expression at row 0"):
            dedup.deduplicate(batch)

    @pytest.mark.unit
    def test_single_row_batch(
        self, dedup: Deduplicator, lib: Library
    ) -> None:
        batch = _make_batch([[2, 0, 1]], lib.EMPTY_ACTION)
        unique_ir, scatter = dedup.deduplicate(batch)

        assert len(unique_ir) == 1
        assert unique_ir[0] == "add(x1,u1)"
        np.testing.assert_array_equal(scatter, [0])

    @pytest.mark.unit
    def test_nested_expression(
        self, dedup: Deduplicator, lib: Library
    ) -> None:
        batch = _make_batch([[4, 3, 0, 1]], lib.EMPTY_ACTION)
        unique_ir, scatter = dedup.deduplicate(batch)

        assert len(unique_ir) == 1
        assert unique_ir[0] == "mul(sin(x1),u1)"





class TestScatterRewardsUnit:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_scatter_broadcasts_correctly(self) -> None:
        unique_rewards = np.array([0.5, 0.8, 0.1])
        scatter_map = np.array([0, 2, 0, 1, 2])
        result = Deduplicator.scatter_rewards(unique_rewards, scatter_map)
        np.testing.assert_array_equal(result, [0.5, 0.1, 0.5, 0.8, 0.1])

    @pytest.mark.unit
    def test_scatter_all_same_index(self) -> None:
        unique_rewards = np.array([0.42])
        scatter_map = np.array([0, 0, 0])
        result = Deduplicator.scatter_rewards(unique_rewards, scatter_map)
        np.testing.assert_array_equal(result, [0.42, 0.42, 0.42])

    @pytest.mark.unit
    def test_scatter_identity(self) -> None:
        unique_rewards = np.array([0.1, 0.2, 0.3])
        scatter_map = np.array([0, 1, 2])
        result = Deduplicator.scatter_rewards(unique_rewards, scatter_map)
        np.testing.assert_array_equal(result, [0.1, 0.2, 0.3])

    @pytest.mark.unit
    def test_scatter_preserves_dtype(self) -> None:
        unique_rewards = np.array([0.5, 0.8], dtype=np.float32)
        scatter_map = np.array([1, 0, 1])
        result = Deduplicator.scatter_rewards(unique_rewards, scatter_map)
        assert result.dtype == np.float32





class TestDeduplicateInvariants:

    @pytest.mark.unit
    def test_unique_count_le_batch_size(
        self, dedup: Deduplicator, lib: Library
    ) -> None:
        batch = _make_batch(
            [[0], [3, 0], [0], [1], [3, 0]], lib.EMPTY_ACTION
        )
        unique_ir, _scatter = dedup.deduplicate(batch)
        assert len(unique_ir) <= batch.shape[0]

    @pytest.mark.unit
    def test_scatter_values_in_range_and_integer(
        self, dedup: Deduplicator, lib: Library
    ) -> None:
        batch = _make_batch(
            [[3, 0], [0], [3, 0], [1]], lib.EMPTY_ACTION
        )
        unique_ir, scatter = dedup.deduplicate(batch)
        assert np.issubdtype(scatter.dtype, np.integer)
        assert scatter.min() >= 0
        assert scatter.max() < len(unique_ir)

    @pytest.mark.unit
    def test_scatter_rewards_shape(self) -> None:
        unique_rewards = np.array([0.5, 0.8])
        scatter_map = np.array([0, 1, 0, 1, 0])
        result = Deduplicator.scatter_rewards(unique_rewards, scatter_map)
        assert result.shape == (5,)

    @pytest.mark.unit
    def test_scatter_map_shape_matches_batch(
        self, dedup: Deduplicator, lib: Library
    ) -> None:
        batch = _make_batch([[0], [1], [3, 0]], lib.EMPTY_ACTION)
        _unique_ir, scatter = dedup.deduplicate(batch)
        assert scatter.shape == (batch.shape[0],)

    @pytest.mark.unit
    def test_roundtrip_dedup_scatter(
        self, dedup: Deduplicator, lib: Library
    ) -> None:

        batch = _make_batch(
            [[3, 0], [0], [3, 0]], lib.EMPTY_ACTION
        )
        unique_ir, scatter = dedup.deduplicate(batch)


        unique_rewards = np.arange(len(unique_ir), dtype=np.float64) + 1.0
        full_rewards = Deduplicator.scatter_rewards(unique_rewards, scatter)


        assert full_rewards[0] == full_rewards[2]

        assert full_rewards[0] != full_rewards[1]
        assert full_rewards.shape == (3,)
