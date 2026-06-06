from __future__ import annotations

import numpy as np
import pytest

from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.validator import CandidateValidator



BURGERS_CONFIG = LibraryConfig(
    coord_vars=["x", "t"],
    state_vars=["u"],
    operators=[
        "add", "mul", "sin", "cos", "diff_x", "diff_t", "n2", "neg", "div",
    ],
)
MAX_LENGTH = 10


@pytest.fixture
def lib() -> Library:
    return Library.from_config(BURGERS_CONFIG)


@pytest.fixture
def cv(lib: Library) -> CandidateValidator:
    return CandidateValidator(lib, max_length=MAX_LENGTH)




CASE_NAMES = [
    "valid_burgers",
    "valid_simple",
    "valid_unary",
    "valid_nested",
    "valid_at_max",
    "invalid_incomplete",
    "invalid_incomplete_unary",
    "invalid_too_long",
    "invalid_trivial",
    "invalid_trivial_coord",
]


class TestValidatorCrossValidation:

    @pytest.mark.equivalence
    @pytest.mark.parametrize("case", CASE_NAMES)
    def test_single_matches_fixture(
        self,
        cv: CandidateValidator,
        validator_fixture: dict[str, np.ndarray],
        case: str,
    ) -> None:
        tokens = validator_fixture[f"case_{case}_tokens"]
        expected = bool(validator_fixture[f"case_{case}_expected"])
        assert cv.validate_single(tokens) == expected

    @pytest.mark.equivalence
    @pytest.mark.smoke
    def test_batch_matches_fixture(
        self, cv: CandidateValidator, validator_fixture: dict[str, np.ndarray]
    ) -> None:
        tokens = validator_fixture["batch_tokens"]
        expected = validator_fixture["batch_mask"]
        result = cv.validate(tokens)
        np.testing.assert_array_equal(result, expected)




class TestValidateSingle:

    @pytest.mark.unit
    @pytest.mark.smoke
    def test_complete_expression_valid(
        self, cv: CandidateValidator, lib: Library,
    ) -> None:

        tokens = np.array([
            lib.name_to_index("mul"),
            lib.name_to_index("u"),
            lib.name_to_index("u"),
        ], dtype=np.int32)
        assert cv.validate_single(tokens) is True

    @pytest.mark.unit
    def test_incomplete_expression_invalid(
        self, cv: CandidateValidator, lib: Library
    ) -> None:

        tokens = np.array([
            lib.name_to_index("add"),
            lib.name_to_index("u"),
        ], dtype=np.int32)
        assert cv.validate_single(tokens) is False

    @pytest.mark.unit
    def test_exceeds_max_length(self, lib: Library) -> None:
        cv = CandidateValidator(lib, max_length=3)

        tokens = np.array([
            lib.name_to_index("add"),
            lib.name_to_index("u"),
            lib.name_to_index("mul"),
            lib.name_to_index("u"),
            lib.name_to_index("u"),
        ], dtype=np.int32)
        assert cv.validate_single(tokens) is False

    @pytest.mark.unit
    def test_single_terminal_trivial(
        self, cv: CandidateValidator, lib: Library
    ) -> None:
        tokens = np.array([lib.name_to_index("u")], dtype=np.int32)
        assert cv.validate_single(tokens) is False

    @pytest.mark.unit
    def test_empty_tokens_invalid(self, cv: CandidateValidator) -> None:
        tokens = np.array([], dtype=np.int32)
        assert cv.validate_single(tokens) is False

    @pytest.mark.unit
    def test_padded_expression_valid(
        self, cv: CandidateValidator, lib: Library
    ) -> None:
        ea = lib.EMPTY_ACTION
        tokens = np.array([
            lib.name_to_index("mul"),
            lib.name_to_index("u"),
            lib.name_to_index("u"),
            ea, ea, ea,
        ], dtype=np.int32)
        assert cv.validate_single(tokens) is True

    @pytest.mark.unit
    def test_exactly_at_max_length(self, lib: Library) -> None:
        cv = CandidateValidator(lib, max_length=3)
        tokens = np.array([
            lib.name_to_index("mul"),
            lib.name_to_index("u"),
            lib.name_to_index("u"),
        ], dtype=np.int32)
        assert cv.validate_single(tokens) is True

    @pytest.mark.unit
    def test_one_over_max_length(self, lib: Library) -> None:
        cv = CandidateValidator(lib, max_length=2)
        tokens = np.array([
            lib.name_to_index("mul"),
            lib.name_to_index("u"),
            lib.name_to_index("u"),
        ], dtype=np.int32)
        assert cv.validate_single(tokens) is False

    @pytest.mark.unit
    def test_unary_only_expression(
        self, cv: CandidateValidator, lib: Library
    ) -> None:

        tokens = np.array([
            lib.name_to_index("sin"),
            lib.name_to_index("cos"),
            lib.name_to_index("diff_x"),
            lib.name_to_index("u"),
        ], dtype=np.int32)
        assert cv.validate_single(tokens) is True

    @pytest.mark.unit
    def test_right_nested_valid(
        self, cv: CandidateValidator, lib: Library
    ) -> None:

        tokens = np.array([
            lib.name_to_index("add"),
            lib.name_to_index("u"),
            lib.name_to_index("add"),
            lib.name_to_index("u"),
            lib.name_to_index("diff_x"),
            lib.name_to_index("u"),
        ], dtype=np.int32)
        assert cv.validate_single(tokens) is True

    @pytest.mark.unit
    def test_all_operators_never_completes(
        self, cv: CandidateValidator, lib: Library
    ) -> None:
        tokens = np.array(
            [lib.name_to_index("add")] * 5, dtype=np.int32
        )
        assert cv.validate_single(tokens) is False

    @pytest.mark.unit
    def test_over_terminated_still_valid(
        self, cv: CandidateValidator, lib: Library
    ) -> None:
        tokens = np.array([
            lib.name_to_index("mul"),
            lib.name_to_index("u"),
            lib.name_to_index("u"),
            lib.name_to_index("u"),
        ], dtype=np.int32)
        assert cv.validate_single(tokens) is True

    @pytest.mark.unit
    def test_padding_with_operator_indices(
        self, cv: CandidateValidator, lib: Library
    ) -> None:
        tokens = np.array([
            lib.name_to_index("mul"),
            lib.name_to_index("u"),
            lib.name_to_index("u"),
            lib.name_to_index("add"),
            lib.name_to_index("add"),
        ], dtype=np.int32)
        assert cv.validate_single(tokens) is True




class TestValidateBatch:

    @pytest.mark.unit
    def test_batch_shape_and_dtype(
        self, cv: CandidateValidator, lib: Library
    ) -> None:
        tokens = np.array([[
            lib.name_to_index("mul"),
            lib.name_to_index("u"),
            lib.name_to_index("u"),
        ]], dtype=np.int32)
        result = cv.validate(tokens)
        assert result.shape == (1,)
        assert result.dtype == np.bool_

    @pytest.mark.unit
    def test_batch_mixed_valid_invalid(
        self, cv: CandidateValidator, lib: Library
    ) -> None:
        mul = lib.name_to_index("mul")
        add = lib.name_to_index("add")
        u = lib.name_to_index("u")
        tokens = np.array([
            [mul, u, u],
            [add, mul, u],
        ], dtype=np.int32)
        expected = np.array([True, False])
        result = cv.validate(tokens)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    def test_batch_all_valid(
        self, cv: CandidateValidator, lib: Library
    ) -> None:
        mul = lib.name_to_index("mul")
        u = lib.name_to_index("u")
        diff_x = lib.name_to_index("diff_x")
        tokens = np.array([
            [mul, u, u],
            [diff_x, u, 0],
        ], dtype=np.int32)
        result = cv.validate(tokens)
        assert result.all()




class TestMinLengthEnforcement:

    @pytest.mark.unit
    def test_rejects_below_min_length(self, lib: Library) -> None:
        cv = CandidateValidator(lib, max_length=10, min_length=4)
        add = lib.name_to_index("add")
        u = lib.name_to_index("u")
        tokens = np.array([add, u, u], dtype=np.int32)
        assert cv.validate_single(tokens) is False

    @pytest.mark.unit
    def test_accepts_at_min_length_boundary(self, lib: Library) -> None:
        cv = CandidateValidator(lib, max_length=10, min_length=4)
        add = lib.name_to_index("add")
        neg = lib.name_to_index("neg")
        u = lib.name_to_index("u")
        tokens = np.array([add, neg, u, u], dtype=np.int32)
        assert cv.validate_single(tokens) is True

    @pytest.mark.unit
    def test_accepts_above_min_length(self, lib: Library) -> None:
        cv = CandidateValidator(lib, max_length=10, min_length=4)
        add = lib.name_to_index("add")
        mul = lib.name_to_index("mul")
        u = lib.name_to_index("u")

        tokens = np.array([add, u, mul, u, u], dtype=np.int32)
        assert cv.validate_single(tokens) is True

    @pytest.mark.unit
    def test_default_min_length_is_none_backwards_compat(
        self, lib: Library,
    ) -> None:
        cv = CandidateValidator(lib, max_length=10)
        add = lib.name_to_index("add")
        u = lib.name_to_index("u")
        tokens = np.array([add, u, u], dtype=np.int32)
        assert cv.validate_single(tokens) is True

    @pytest.mark.unit
    def test_min_length_rejects_via_batch_api(self, lib: Library) -> None:
        cv = CandidateValidator(lib, max_length=10, min_length=4)
        add = lib.name_to_index("add")
        mul = lib.name_to_index("mul")
        u = lib.name_to_index("u")
        empty = lib.EMPTY_ACTION
        neg = lib.name_to_index("neg")
        tokens = np.array([
            [add, u, u, empty, empty],
            [add, neg, u, u, empty],
            [add, u, mul, u, u],
        ], dtype=np.int32)
        result = cv.validate(tokens)
        assert list(result) == [False, True, True]




class TestValidatorInvariant:

    @pytest.mark.equivalence
    @pytest.mark.smoke
    def test_batch_equals_single(
        self, cv: CandidateValidator, validator_fixture: dict[str, np.ndarray]
    ) -> None:
        batch = validator_fixture["batch_tokens"]
        batch_result = cv.validate(batch)
        for i in range(batch.shape[0]):
            single_result = cv.validate_single(batch[i])
            assert batch_result[i] == single_result, f"Mismatch at row {i}"

    @pytest.mark.unit
    def test_validate_returns_bool_array(
        self, cv: CandidateValidator, validator_fixture: dict[str, np.ndarray]
    ) -> None:
        batch = validator_fixture["batch_tokens"]
        result = cv.validate(batch)
        assert result.dtype == np.bool_
        assert result.shape == (batch.shape[0],)
