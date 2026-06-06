
import numpy as np
import pytest

from kd.search.discover.tokens.library import (
    Library,
    LibraryConfig,
    Token,
    TokenType,
)





def _make_burgers_nondiff_library() -> Library:
    tokens = [
        Token(name="x1", arity=0, token_type=TokenType.COORDINATE, complexity=1),
        Token(name="u1", arity=0, token_type=TokenType.TERMINAL, complexity=1),
        Token(name="add", arity=2, token_type=TokenType.OPERATOR, complexity=1),
        Token(name="mul", arity=2, token_type=TokenType.OPERATOR, complexity=1),
        Token(name="div", arity=2, token_type=TokenType.OPERATOR, complexity=2),
        Token(name="n2", arity=1, token_type=TokenType.OPERATOR, complexity=2),
        Token(name="n3", arity=1, token_type=TokenType.OPERATOR, complexity=3),
    ]
    return Library(tokens)


def _make_burgers_full_library() -> Library:
    tokens = [
        Token(name="x1", arity=0, token_type=TokenType.COORDINATE, complexity=1),
        Token(name="u1", arity=0, token_type=TokenType.TERMINAL, complexity=1),
        Token(name="add", arity=2, token_type=TokenType.OPERATOR, complexity=1),
        Token(name="mul", arity=2, token_type=TokenType.OPERATOR, complexity=1),
        Token(name="div", arity=2, token_type=TokenType.OPERATOR, complexity=2),
        Token(name="diff", arity=1, token_type=TokenType.OPERATOR, complexity=2),
        Token(name="diff2", arity=1, token_type=TokenType.OPERATOR, complexity=3),
        Token(name="diff3", arity=1, token_type=TokenType.OPERATOR, complexity=4),
        Token(name="n2", arity=1, token_type=TokenType.OPERATOR, complexity=2),
        Token(name="n3", arity=1, token_type=TokenType.OPERATOR, complexity=3),
    ]
    return Library(tokens)







class TestLibraryCrossValidation:

    @pytest.mark.equivalence
    def test_names_match_reference(
        self, burgers_fixture: dict[str, np.ndarray]
    ) -> None:
        lib = _make_burgers_nondiff_library()
        np.testing.assert_array_equal(
            np.array(lib.names), burgers_fixture["names"]
        )

    @pytest.mark.equivalence
    def test_arities_match_reference(
        self, burgers_fixture: dict[str, np.ndarray]
    ) -> None:
        lib = _make_burgers_nondiff_library()
        np.testing.assert_array_equal(lib.arities, burgers_fixture["arities"])
        assert lib.arities.dtype == np.int32

    @pytest.mark.equivalence
    def test_parent_adjust_match_reference(
        self, burgers_fixture: dict[str, np.ndarray]
    ) -> None:
        lib = _make_burgers_nondiff_library()
        np.testing.assert_array_equal(
            lib.parent_adjust, burgers_fixture["parent_adjust"]
        )
        assert lib.parent_adjust.dtype == np.int32

    @pytest.mark.equivalence
    def test_empty_values_match_reference(
        self, burgers_fixture: dict[str, np.ndarray]
    ) -> None:
        lib = _make_burgers_nondiff_library()
        assert int(burgers_fixture["EMPTY_ACTION"]) == lib.EMPTY_ACTION
        assert int(burgers_fixture["EMPTY_PARENT"]) == lib.EMPTY_PARENT
        assert int(burgers_fixture["EMPTY_SIBLING"]) == lib.EMPTY_SIBLING

    @pytest.mark.equivalence
    def test_token_group_indices_match_reference(
        self, burgers_fixture: dict[str, np.ndarray]
    ) -> None:
        lib = _make_burgers_nondiff_library()
        np.testing.assert_array_equal(
            lib.terminal_tokens, burgers_fixture["terminal_tokens"]
        )
        np.testing.assert_array_equal(
            lib.unary_tokens, burgers_fixture["unary_tokens"]
        )
        np.testing.assert_array_equal(
            lib.binary_tokens, burgers_fixture["binary_tokens"]
        )
        assert lib.terminal_tokens.dtype == np.int32
        assert lib.unary_tokens.dtype == np.int32
        assert lib.binary_tokens.dtype == np.int32

    @pytest.mark.equivalence
    def test_n_inputs_match_reference(
        self, burgers_fixture: dict[str, np.ndarray]
    ) -> None:
        lib = _make_burgers_nondiff_library()
        assert lib.n_action_inputs == int(burgers_fixture["n_action_inputs"])
        assert lib.n_parent_inputs == int(burgers_fixture["n_parent_inputs"])
        assert lib.n_sibling_inputs == int(burgers_fixture["n_sibling_inputs"])

    @pytest.mark.equivalence
    def test_diff_tokens_array_dtype(
        self, burgers_fixture: dict[str, np.ndarray]
    ) -> None:
        lib = _make_burgers_nondiff_library()
        assert lib.diff_tokens.dtype == np.int32







class TestToken:

    @pytest.mark.unit
    def test_create_operator(self) -> None:
        t = Token(name="add", arity=2, token_type=TokenType.OPERATOR, complexity=1)
        assert t.name == "add"
        assert t.arity == 2
        assert t.token_type == TokenType.OPERATOR
        assert t.complexity == 1

    @pytest.mark.unit
    def test_create_terminal(self) -> None:
        t = Token(name="u1", arity=0, token_type=TokenType.TERMINAL, complexity=1)
        assert t.arity == 0
        assert t.token_type == TokenType.TERMINAL

    @pytest.mark.unit
    def test_create_coordinate(self) -> None:
        t = Token(name="x1", arity=0, token_type=TokenType.COORDINATE, complexity=1)
        assert t.token_type == TokenType.COORDINATE

    @pytest.mark.unit
    def test_all_token_types(self) -> None:
        arities = {
            TokenType.OPERATOR: 1,
            TokenType.TERMINAL: 0,
            TokenType.COORDINATE: 0,
        }
        for tt in TokenType:
            t = Token(
                name=f"test_{tt.value}", arity=arities[tt], token_type=tt
            )
            assert t.token_type == tt

    @pytest.mark.unit
    def test_default_complexity(self) -> None:
        t = Token(name="add", arity=2, token_type=TokenType.OPERATOR)
        assert t.complexity == 1







class TestLibrary:

    @pytest.mark.unit
    def test_name_to_index_roundtrip(self) -> None:
        lib = _make_burgers_nondiff_library()
        for i, name in enumerate(lib.names):
            assert lib.name_to_index(name) == i
            assert lib.index_to_name(i) == name

    @pytest.mark.unit
    def test_getitem_by_name(self) -> None:
        lib = _make_burgers_nondiff_library()
        token = lib["add"]
        assert token.name == "add"
        assert token.arity == 2

    @pytest.mark.unit
    def test_getitem_by_index(self) -> None:
        lib = _make_burgers_nondiff_library()
        token = lib[0]
        assert token.name == "x1"

    @pytest.mark.unit
    def test_getitem_invalid_name_raises(self) -> None:
        lib = _make_burgers_nondiff_library()
        with pytest.raises(KeyError):
            lib["nonexistent"]

    @pytest.mark.unit
    def test_getitem_invalid_index_raises(self) -> None:
        lib = _make_burgers_nondiff_library()
        with pytest.raises(KeyError):
            lib[999]

    @pytest.mark.unit
    def test_getitem_negative_index_raises(self) -> None:
        lib = _make_burgers_nondiff_library()
        with pytest.raises(KeyError):
            lib[-1]

    @pytest.mark.unit
    def test_getitem_rejects_bool_index(self) -> None:
        lib = _make_burgers_nondiff_library()
        with pytest.raises(KeyError, match="bool"):
            lib[True]
        with pytest.raises(KeyError, match="bool"):
            lib[False]

    @pytest.mark.unit
    def test_empty_library(self) -> None:
        lib = Library([])
        assert len(lib.tokens) == 0
        assert len(lib.names) == 0
        assert lib.arities.shape == (0,)
        assert lib.arities.dtype == np.int32
        assert lib.parent_adjust.shape == (0,)
        assert len(lib.terminal_tokens) == 0
        assert len(lib.unary_tokens) == 0
        assert len(lib.binary_tokens) == 0
        assert len(lib.diff_tokens) == 0
        assert len(lib.special_diff_tokens) == 0
        assert lib.n_action_inputs == 1
        assert lib.n_parent_inputs == 1
        assert lib.n_sibling_inputs == 1
        assert lib.EMPTY_ACTION == 0
        assert lib.EMPTY_PARENT == 0
        assert lib.EMPTY_SIBLING == 0







class TestLibraryInvariants:

    @pytest.mark.unit
    def test_parent_adjust_neg1_iff_terminal(self) -> None:
        lib = _make_burgers_full_library()
        for i in range(len(lib.tokens)):
            if lib.arities[i] == 0:
                assert lib.parent_adjust[i] == -1, (
                    f"Token {lib.names[i]} has arity 0 but parent_adjust != -1"
                )
            else:
                assert lib.parent_adjust[i] >= 0, (
                    f"Token {lib.names[i]} has arity > 0 but parent_adjust == -1"
                )

    @pytest.mark.unit
    def test_arity_groups_partition(self) -> None:
        lib = _make_burgers_full_library()
        L = len(lib.tokens)
        total = (
            len(lib.terminal_tokens)
            + len(lib.unary_tokens)
            + len(lib.binary_tokens)
        )
        assert total == L, f"Arity groups sum to {total}, expected {L}"

    @pytest.mark.unit
    def test_empty_action_equals_L(self) -> None:
        lib = _make_burgers_full_library()
        L = len(lib.tokens)
        assert lib.EMPTY_ACTION == L







class TestDiffTokens:

    @pytest.mark.unit
    def test_diff_tokens_are_unary(self) -> None:
        lib = _make_burgers_full_library()
        for idx in lib.diff_tokens:
            assert lib.arities[idx] == 1, (
                f"Diff token {lib.names[idx]} should be unary (arity=1)"
            )

    @pytest.mark.unit
    def test_diff_token_indices(self) -> None:
        lib = _make_burgers_full_library()
        diff_names = {lib.names[i] for i in lib.diff_tokens}
        assert diff_names == {"diff", "diff2", "diff3"}

    @pytest.mark.unit
    def test_diff_tokens_in_unary_group(self) -> None:
        lib = _make_burgers_full_library()
        unary_set = set(lib.unary_tokens.tolist())
        for idx in lib.diff_tokens:
            assert idx in unary_set, (
                f"Diff token {lib.names[idx]} not in unary_tokens"
            )

    @pytest.mark.unit
    def test_lap_not_in_diff_operators(self) -> None:
        from kd.search.discover.tokens.library import _DIFF_OPERATORS

        assert "lap" not in _DIFF_OPERATORS, (
            "lap is dispatched by kd's special-operator path, not the "
            "diff_x/diff2_x path. Keeping it out of _DIFF_OPERATORS keeps "
            "DiffChildConstraint/DiffDescendantConstraint from blocking "
            "lap(add(...)) and Cahn-Hilliard-shaped expressions."
        )

    @pytest.mark.unit
    def test_lap_in_operator_arities(self) -> None:
        from kd.search.discover.tokens.library import _OPERATOR_ARITIES

        assert _OPERATOR_ARITIES["lap"] == 1

    @pytest.mark.unit
    def test_lap_in_diff_orders(self) -> None:
        from kd.search.discover.tokens.library import _DIFF_ORDERS

        assert _DIFF_ORDERS["lap"] == 2

    @pytest.mark.unit
    def test_lap_in_operator_complexities(self) -> None:
        from kd.search.discover.tokens.library import _OPERATOR_COMPLEXITIES

        assert _OPERATOR_COMPLEXITIES["lap"] == 3

    @pytest.mark.unit
    def test_lap_in_special_diff_operators(self) -> None:
        from kd.search.discover.tokens.library import _SPECIAL_DIFF_OPERATORS

        assert frozenset({"lap"}) == _SPECIAL_DIFF_OPERATORS

    @pytest.mark.unit
    def test_lap_buildable_via_config(self) -> None:
        config = LibraryConfig(
            operators=["add", "lap", "diff_x"],
            state_vars=["u"],
            coord_vars=["x"],
        )
        lib = Library.from_config(config)
        assert "lap" in lib.names
        assert lib["lap"].arity == 1
        assert lib["lap"].complexity == 3

    @pytest.mark.unit
    def test_library_special_diff_tokens_contains_lap(self) -> None:
        config = LibraryConfig(
            operators=["add", "lap", "diff_x", "diff2_x"],
            state_vars=["u"],
            coord_vars=["x", "t"],
        )
        lib = Library.from_config(config)
        special_names = {lib.names[i] for i in lib.special_diff_tokens}
        assert special_names == {"lap"}

    @pytest.mark.unit
    def test_library_diff_tokens_excludes_lap(self) -> None:
        config = LibraryConfig(
            operators=["add", "lap", "diff_x", "diff2_x"],
            state_vars=["u"],
            coord_vars=["x", "t"],
        )
        lib = Library.from_config(config)
        diff_names = {lib.names[i] for i in lib.diff_tokens}
        assert diff_names == {"diff_x", "diff2_x"}
        assert "lap" not in diff_names







class TestLibraryFromConfig:

    @pytest.mark.unit
    def test_from_config_burgers(self) -> None:
        config = LibraryConfig(
            operators=["add", "mul", "div", "n2", "n3"],
            state_vars=["u1"],
            coord_vars=["x1"],
        )
        lib = Library.from_config(config)

        assert lib.names == ["x1", "u1", "add", "mul", "div", "n2", "n3"]

    @pytest.mark.unit
    def test_from_config_with_diff(self) -> None:
        config = LibraryConfig(
            operators=["add", "mul", "diff", "diff2"],
            state_vars=["u1"],
            coord_vars=["x1"],
        )
        lib = Library.from_config(config)
        assert len(lib.diff_tokens) == 2

        for idx in lib.diff_tokens:
            assert lib.arities[idx] == 1

    @pytest.mark.unit
    def test_from_config_ordering(self) -> None:
        config = LibraryConfig(
            operators=["mul", "sin"],
            state_vars=["u1", "v1"],
            coord_vars=["x1", "y1"],
        )
        lib = Library.from_config(config)
        assert lib.names == ["x1", "y1", "u1", "v1", "mul", "sin"]
