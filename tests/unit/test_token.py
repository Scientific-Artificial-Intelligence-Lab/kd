"""Tests for Token and TokenType.

Test coverage:
- smoke: Basic instantiation
- unit: Creation, immutability, hashability, equality
"""

import pytest
import torch
from dataclasses import FrozenInstanceError

from kd2.core.ir.token import Token, TokenType


# =============================================================================
# Smoke Tests
# =============================================================================


@pytest.mark.smoke
class TestTokenSmoke:
    """Smoke tests: basic instantiation."""

    def test_token_type_exists(self) -> None:
        """TokenType enum exists and has expected members."""
        assert hasattr(TokenType, "FUNCTION")
        assert hasattr(TokenType, "VARIABLE")
        assert hasattr(TokenType, "CONSTANT")
        assert hasattr(TokenType, "DIFF")

    def test_token_class_exists(self) -> None:
        """Token class exists and is a dataclass."""
        assert hasattr(Token, "__dataclass_fields__")

    def test_token_can_be_instantiated(self) -> None:
        """Token can be instantiated with required arguments."""
        token = Token(
            name="test",
            arity=0,
            token_type=TokenType.VARIABLE,
            function=None,
        )
        assert token.name == "test"


# =============================================================================
# Unit Tests - TokenType
# =============================================================================


@pytest.mark.unit
class TestTokenType:
    """Unit tests for TokenType enum."""

    def test_function_type_value(self) -> None:
        """FUNCTION type has correct value."""
        assert TokenType.FUNCTION.value == "function"

    def test_variable_type_value(self) -> None:
        """VARIABLE type has correct value."""
        assert TokenType.VARIABLE.value == "variable"

    def test_constant_type_value(self) -> None:
        """CONSTANT type has correct value."""
        assert TokenType.CONSTANT.value == "constant"

    def test_diff_type_value(self) -> None:
        """DIFF type has correct value."""
        assert TokenType.DIFF.value == "diff"

    def test_all_types_are_distinct(self) -> None:
        """All TokenType values are distinct."""
        values = [t.value for t in TokenType]
        assert len(values) == len(set(values))


# =============================================================================
# Unit Tests - Token Creation
# =============================================================================


@pytest.mark.unit
class TestTokenCreation:
    """Unit tests for Token creation."""

    def test_create_variable_token(self) -> None:
        """Create a variable token (terminal)."""
        token = Token(
            name="x",
            arity=0,
            token_type=TokenType.VARIABLE,
            function=None,
        )
        assert token.name == "x"
        assert token.arity == 0
        assert token.token_type == TokenType.VARIABLE
        assert token.function is None
        assert token.is_commutative is False

    def test_create_constant_token(self) -> None:
        """Create a constant token (terminal)."""
        token = Token(
            name="pi",
            arity=0,
            token_type=TokenType.CONSTANT,
            function=None,
        )
        assert token.name == "pi"
        assert token.arity == 0
        assert token.token_type == TokenType.CONSTANT

    def test_create_unary_function_token(self) -> None:
        """Create a unary function token."""
        token = Token(
            name="sin",
            arity=1,
            token_type=TokenType.FUNCTION,
            function=torch.sin,
        )
        assert token.name == "sin"
        assert token.arity == 1
        assert token.token_type == TokenType.FUNCTION
        assert token.function is torch.sin

    def test_create_binary_function_token(self) -> None:
        """Create a binary function token."""
        token = Token(
            name="add",
            arity=2,
            token_type=TokenType.FUNCTION,
            function=torch.add,
            is_commutative=True,
        )
        assert token.name == "add"
        assert token.arity == 2
        assert token.is_commutative is True

    def test_create_diff_token(self) -> None:
        """Create a diff operator token."""

        def diff_x_stub(x: torch.Tensor) -> torch.Tensor:
            return x  # Stub

        token = Token(
            name="diff_x",
            arity=1,
            token_type=TokenType.DIFF,
            function=diff_x_stub,
        )
        assert token.name == "diff_x"
        assert token.arity == 1
        assert token.token_type == TokenType.DIFF


# =============================================================================
# Unit Tests - Token Immutability (frozen=True)
# =============================================================================


@pytest.mark.unit
class TestTokenImmutability:
    """Unit tests for Token immutability."""

    def test_token_name_is_frozen(self) -> None:
        """Cannot modify token name after creation."""
        token = Token(
            name="x",
            arity=0,
            token_type=TokenType.VARIABLE,
        )
        with pytest.raises(FrozenInstanceError):
            token.name = "y"  # type: ignore[misc]

    def test_token_arity_is_frozen(self) -> None:
        """Cannot modify token arity after creation."""
        token = Token(
            name="x",
            arity=0,
            token_type=TokenType.VARIABLE,
        )
        with pytest.raises(FrozenInstanceError):
            token.arity = 1  # type: ignore[misc]

    def test_token_type_is_frozen(self) -> None:
        """Cannot modify token type after creation."""
        token = Token(
            name="x",
            arity=0,
            token_type=TokenType.VARIABLE,
        )
        with pytest.raises(FrozenInstanceError):
            token.token_type = TokenType.CONSTANT  # type: ignore[misc]

    def test_token_is_commutative_is_frozen(self) -> None:
        """Cannot modify is_commutative after creation."""
        token = Token(
            name="add",
            arity=2,
            token_type=TokenType.FUNCTION,
            is_commutative=True,
        )
        with pytest.raises(FrozenInstanceError):
            token.is_commutative = False  # type: ignore[misc]


# =============================================================================
# Unit Tests - Token Hashability
# =============================================================================


@pytest.mark.unit
class TestTokenHashability:
    """Unit tests for Token hashability."""

    def test_token_is_hashable(self) -> None:
        """Token can be hashed."""
        token = Token(
            name="x",
            arity=0,
            token_type=TokenType.VARIABLE,
        )
        h = hash(token)
        assert isinstance(h, int)

    def test_token_can_be_dict_key(self) -> None:
        """Token can be used as dictionary key."""
        token = Token(
            name="x",
            arity=0,
            token_type=TokenType.VARIABLE,
        )
        d = {token: "value"}
        assert d[token] == "value"

    def test_token_can_be_set_member(self) -> None:
        """Token can be added to a set."""
        token1 = Token(name="x", arity=0, token_type=TokenType.VARIABLE)
        token2 = Token(name="y", arity=0, token_type=TokenType.VARIABLE)
        s = {token1, token2}
        assert len(s) == 2
        assert token1 in s
        assert token2 in s

    def test_same_tokens_have_same_hash(self) -> None:
        """Tokens with same properties have the same hash."""
        token1 = Token(name="x", arity=0, token_type=TokenType.VARIABLE)
        token2 = Token(name="x", arity=0, token_type=TokenType.VARIABLE)
        assert hash(token1) == hash(token2)

    def test_function_tokens_are_hashable(self) -> None:
        """Tokens with functions are still hashable."""
        token = Token(
            name="sin",
            arity=1,
            token_type=TokenType.FUNCTION,
            function=torch.sin,
        )
        h = hash(token)
        assert isinstance(h, int)


# =============================================================================
# Unit Tests - Token Equality
# =============================================================================


@pytest.mark.unit
class TestTokenEquality:
    """Unit tests for Token equality."""

    def test_same_tokens_are_equal(self) -> None:
        """Tokens with same properties are equal."""
        token1 = Token(name="x", arity=0, token_type=TokenType.VARIABLE)
        token2 = Token(name="x", arity=0, token_type=TokenType.VARIABLE)
        assert token1 == token2

    def test_different_names_not_equal(self) -> None:
        """Tokens with different names are not equal."""
        token1 = Token(name="x", arity=0, token_type=TokenType.VARIABLE)
        token2 = Token(name="y", arity=0, token_type=TokenType.VARIABLE)
        assert token1 != token2

    def test_different_arity_not_equal(self) -> None:
        """Tokens with different arity are not equal."""
        token1 = Token(name="f", arity=1, token_type=TokenType.FUNCTION)
        token2 = Token(name="f", arity=2, token_type=TokenType.FUNCTION)
        assert token1 != token2

    def test_different_type_not_equal(self) -> None:
        """Tokens with different token_type are not equal."""
        token1 = Token(name="x", arity=0, token_type=TokenType.VARIABLE)
        token2 = Token(name="x", arity=0, token_type=TokenType.CONSTANT)
        assert token1 != token2

    def test_different_commutative_not_equal(self) -> None:
        """Tokens with different is_commutative are not equal."""
        token1 = Token(
            name="add", arity=2, token_type=TokenType.FUNCTION, is_commutative=True
        )
        token2 = Token(
            name="add", arity=2, token_type=TokenType.FUNCTION, is_commutative=False
        )
        assert token1 != token2

    def test_token_not_equal_to_non_token(self) -> None:
        """Token is not equal to non-Token objects."""
        token = Token(name="x", arity=0, token_type=TokenType.VARIABLE)
        assert token != "x"
        assert token != 0
        assert token != None  # noqa: E711

    def test_function_tokens_equality(self) -> None:
        """Tokens with same function are equal."""
        token1 = Token(
            name="sin", arity=1, token_type=TokenType.FUNCTION, function=torch.sin
        )
        token2 = Token(
            name="sin", arity=1, token_type=TokenType.FUNCTION, function=torch.sin
        )
        assert token1 == token2

    def test_function_tokens_different_functions(self) -> None:
        """Tokens with different functions but same name - equality by name.

        Note: Two tokens with the same name should be considered equal
        even if they have different function references. This supports
        serialization/deserialization where function references are
        re-bound after loading.
        """
        token1 = Token(
            name="f", arity=1, token_type=TokenType.FUNCTION, function=torch.sin
        )
        token2 = Token(
            name="f", arity=1, token_type=TokenType.FUNCTION, function=torch.cos
        )
        # Equality should be based on name, arity, type, commutative - not function ref
        # This is a design decision - tokens with same identity are equal
        assert token1 == token2
