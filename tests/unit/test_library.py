"""Tests for Library.

Test coverage:
- smoke: Basic instantiation
- unit: Registration, querying, default operators
- numerical: Safe function behavior
"""

import pytest
import torch

from kd2.core.ir.token import Token, TokenType
from kd2.core.library import Library


# =============================================================================
# Smoke Tests
# =============================================================================


@pytest.mark.smoke
class TestLibrarySmoke:
    """Smoke tests: basic instantiation."""

    def test_library_can_be_instantiated(self) -> None:
        """Library can be instantiated."""
        lib = Library()
        assert lib is not None

    def test_create_default_exists(self) -> None:
        """Library.create_default class method exists."""
        assert hasattr(Library, "create_default")

    def test_create_default_returns_library(self) -> None:
        """create_default returns a Library instance."""
        lib = Library.create_default()
        assert isinstance(lib, Library)


# =============================================================================
# Unit Tests - Registration
# =============================================================================


@pytest.mark.unit
class TestLibraryRegistration:
    """Unit tests for token registration."""

    def test_register_and_get(self) -> None:
        """Register a token and retrieve it by name."""
        lib = Library()
        token = Token(name="x", arity=0, token_type=TokenType.VARIABLE)
        lib.register(token)
        retrieved = lib.get("x")
        assert retrieved == token

    def test_register_multiple_tokens(self) -> None:
        """Register multiple tokens."""
        lib = Library()
        token_x = Token(name="x", arity=0, token_type=TokenType.VARIABLE)
        token_y = Token(name="y", arity=0, token_type=TokenType.VARIABLE)
        lib.register(token_x)
        lib.register(token_y)
        assert lib.get("x") == token_x
        assert lib.get("y") == token_y

    def test_register_duplicate_raises(self) -> None:
        """Registering a token with existing name raises ValueError."""
        lib = Library()
        token = Token(name="x", arity=0, token_type=TokenType.VARIABLE)
        lib.register(token)
        with pytest.raises(ValueError, match="already exists"):
            lib.register(token)

    def test_get_not_found_raises(self) -> None:
        """Getting a non-existent token raises KeyError."""
        lib = Library()
        with pytest.raises(KeyError):
            lib.get("nonexistent")


# =============================================================================
# Unit Tests - Query Methods
# =============================================================================


@pytest.mark.unit
class TestLibraryQueries:
    """Unit tests for library query methods."""

    @pytest.fixture
    def lib_with_tokens(self) -> Library:
        """Library with various token types."""
        lib = Library()
        # Terminals (arity=0)
        lib.register(Token(name="x", arity=0, token_type=TokenType.VARIABLE))
        lib.register(Token(name="y", arity=0, token_type=TokenType.VARIABLE))
        lib.register(Token(name="pi", arity=0, token_type=TokenType.CONSTANT))
        # Unary (arity=1)
        lib.register(
            Token(name="sin", arity=1, token_type=TokenType.FUNCTION, function=torch.sin)
        )
        lib.register(
            Token(name="cos", arity=1, token_type=TokenType.FUNCTION, function=torch.cos)
        )
        # Binary (arity=2)
        lib.register(
            Token(
                name="add",
                arity=2,
                token_type=TokenType.FUNCTION,
                function=torch.add,
                is_commutative=True,
            )
        )
        return lib

    def test_get_by_arity_zero(self, lib_with_tokens: Library) -> None:
        """Get all terminals (arity=0)."""
        terminals = lib_with_tokens.get_by_arity(0)
        assert len(terminals) == 3
        names = {t.name for t in terminals}
        assert names == {"x", "y", "pi"}

    def test_get_by_arity_one(self, lib_with_tokens: Library) -> None:
        """Get all unary operators (arity=1)."""
        unary = lib_with_tokens.get_by_arity(1)
        assert len(unary) == 2
        names = {t.name for t in unary}
        assert names == {"sin", "cos"}

    def test_get_by_arity_two(self, lib_with_tokens: Library) -> None:
        """Get all binary operators (arity=2)."""
        binary = lib_with_tokens.get_by_arity(2)
        assert len(binary) == 1
        assert binary[0].name == "add"

    def test_get_by_arity_empty(self, lib_with_tokens: Library) -> None:
        """Get by arity returns empty list if no matches."""
        result = lib_with_tokens.get_by_arity(3)
        assert result == []

    def test_get_terminals(self, lib_with_tokens: Library) -> None:
        """Get all terminal tokens."""
        terminals = lib_with_tokens.get_terminals()
        assert len(terminals) == 3
        for t in terminals:
            assert t.arity == 0

    def test_get_functions(self, lib_with_tokens: Library) -> None:
        """Get all function tokens."""
        functions = lib_with_tokens.get_functions()
        assert len(functions) == 3  # sin, cos, add
        for t in functions:
            assert t.arity > 0


# =============================================================================
# Unit Tests - Default Operators
# =============================================================================


@pytest.mark.unit
class TestDefaultOperators:
    """Unit tests for default operator set."""

    @pytest.fixture
    def default_lib(self) -> Library:
        """Library with default operators."""
        return Library.create_default()

    def test_binary_operators_exist(self, default_lib: Library) -> None:
        """Default library has binary operators."""
        binary_names = {"add", "sub", "mul", "div"}
        for name in binary_names:
            token = default_lib.get(name)
            assert token.arity == 2
            assert token.token_type == TokenType.FUNCTION

    def test_unary_operators_exist(self, default_lib: Library) -> None:
        """Default library has unary operators."""
        unary_names = {"sin", "cos", "exp", "log", "n2", "n3"}
        for name in unary_names:
            token = default_lib.get(name)
            assert token.arity == 1
            assert token.token_type == TokenType.FUNCTION

    def test_commutative_flags(self, default_lib: Library) -> None:
        """add and mul are marked as commutative."""
        add = default_lib.get("add")
        mul = default_lib.get("mul")
        sub = default_lib.get("sub")
        div = default_lib.get("div")

        assert add.is_commutative is True
        assert mul.is_commutative is True
        assert sub.is_commutative is False
        assert div.is_commutative is False

    def test_operator_count(self, default_lib: Library) -> None:
        """Default library has expected number of operators."""
        # Binary: add, sub, mul, div = 4
        # Unary: sin, cos, exp, log, n2, n3 = 6
        # Total: 10
        functions = default_lib.get_functions()
        assert len(functions) == 10


# =============================================================================
# Unit Tests - Diff Token Registration
# =============================================================================


@pytest.mark.unit
class TestDiffRegistration:
    """Unit tests for differential operator registration."""

    def test_register_diff_tokens_basic(self) -> None:
        """Register diff tokens for single axis."""
        lib = Library()
        lib.register_diff_tokens(axes=["x"], fields=["u"], max_order=1)
        diff_x = lib.get("diff_x")
        assert diff_x.arity == 1
        assert diff_x.token_type == TokenType.DIFF

    def test_register_diff_tokens_multiple_axes(self) -> None:
        """Register diff tokens for multiple axes."""
        lib = Library()
        lib.register_diff_tokens(axes=["x", "t"], fields=["u"], max_order=1)
        diff_x = lib.get("diff_x")
        diff_t = lib.get("diff_t")
        assert diff_x.token_type == TokenType.DIFF
        assert diff_t.token_type == TokenType.DIFF

    def test_register_diff_tokens_higher_order(self) -> None:
        """Register higher-order diff tokens."""
        lib = Library()
        lib.register_diff_tokens(axes=["x"], fields=["u"], max_order=3)

        # Order 1: diff_x
        diff_x = lib.get("diff_x")
        assert diff_x.arity == 1

        # Order 2: diff2_x
        diff2_x = lib.get("diff2_x")
        assert diff2_x.arity == 1

        # Order 3: diff3_x
        diff3_x = lib.get("diff3_x")
        assert diff3_x.arity == 1

    def test_diff_naming_convention(self) -> None:
        """Diff tokens follow naming convention."""
        lib = Library()
        lib.register_diff_tokens(axes=["x", "y"], fields=["u"], max_order=2)

        # Order 1: diff_{axis}
        assert lib.get("diff_x").name == "diff_x"
        assert lib.get("diff_y").name == "diff_y"

        # Order 2: diff2_{axis}
        assert lib.get("diff2_x").name == "diff2_x"
        assert lib.get("diff2_y").name == "diff2_y"

    def test_diff_tokens_are_unary(self) -> None:
        """All diff tokens have arity=1."""
        lib = Library()
        lib.register_diff_tokens(axes=["x", "t"], fields=["u", "v"], max_order=2)
        diff_tokens = [t for t in lib.get_functions() if t.token_type == TokenType.DIFF]
        for token in diff_tokens:
            assert token.arity == 1, f"{token.name} should have arity=1"

    def test_register_diff_invalid_max_order(self) -> None:
        """max_order < 1 raises ValueError."""
        lib = Library()
        with pytest.raises(ValueError, match="max_order"):
            lib.register_diff_tokens(axes=["x"], fields=["u"], max_order=0)


# =============================================================================
# Numerical Tests - Safe Functions
# =============================================================================


@pytest.mark.numerical
class TestDefaultOperatorsSafety:
    """Numerical tests for safe function usage in default operators."""

    @pytest.fixture
    def default_lib(self) -> Library:
        """Library with default operators."""
        return Library.create_default()

    def test_div_by_zero_is_safe(self, default_lib: Library) -> None:
        """Division by zero does not produce NaN or Inf."""
        div = default_lib.get("div")
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([0.0, 0.0, 0.0])
        result = div.function(a, b)
        assert torch.isfinite(result).all(), "div should never produce NaN/Inf"

    def test_div_near_zero_is_safe(self, default_lib: Library) -> None:
        """Division by near-zero does not produce Inf."""
        div = default_lib.get("div")
        a = torch.tensor([1.0])
        b = torch.tensor([1e-15])
        result = div.function(a, b)
        assert torch.isfinite(result).all(), "div should handle near-zero"

    def test_exp_large_input_is_safe(self, default_lib: Library) -> None:
        """Exponential of large input does not overflow."""
        exp = default_lib.get("exp")
        x = torch.tensor([100.0, 500.0, 1000.0])
        result = exp.function(x)
        assert torch.isfinite(result).all(), "exp should clamp large inputs"

    def test_log_zero_is_safe(self, default_lib: Library) -> None:
        """Logarithm of zero does not produce -Inf."""
        log = default_lib.get("log")
        x = torch.tensor([0.0])
        result = log.function(x)
        assert torch.isfinite(result).all(), "log should handle zero"

    def test_log_negative_is_safe(self, default_lib: Library) -> None:
        """Logarithm of negative number does not produce NaN."""
        log = default_lib.get("log")
        x = torch.tensor([-1.0, -100.0])
        result = log.function(x)
        assert torch.isfinite(result).all(), "log should handle negative values"

    def test_n2_correctness(self, default_lib: Library) -> None:
        """n2 computes x^2 correctly."""
        n2 = default_lib.get("n2")
        x = torch.tensor([2.0, 3.0, -4.0])
        result = n2.function(x)
        expected = torch.tensor([4.0, 9.0, 16.0])
        torch.testing.assert_close(result, expected)

    def test_n3_correctness(self, default_lib: Library) -> None:
        """n3 computes x^3 correctly."""
        n3 = default_lib.get("n3")
        x = torch.tensor([2.0, 3.0, -2.0])
        result = n3.function(x)
        expected = torch.tensor([8.0, 27.0, -8.0])
        torch.testing.assert_close(result, expected)

    def test_trig_functions_correctness(self, default_lib: Library) -> None:
        """sin and cos produce correct results."""
        sin = default_lib.get("sin")
        cos = default_lib.get("cos")
        x = torch.tensor([0.0, torch.pi / 2, torch.pi])

        sin_result = sin.function(x)
        cos_result = cos.function(x)

        # sin(0)=0, sin(pi/2)=1, sin(pi)=0
        torch.testing.assert_close(
            sin_result, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=1e-5
        )
        # cos(0)=1, cos(pi/2)=0, cos(pi)=-1
        torch.testing.assert_close(
            cos_result, torch.tensor([1.0, 0.0, -1.0]), atol=1e-6, rtol=1e-5
        )

    def test_add_sub_mul_correctness(self, default_lib: Library) -> None:
        """Basic arithmetic operators work correctly."""
        add = default_lib.get("add")
        sub = default_lib.get("sub")
        mul = default_lib.get("mul")

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])

        torch.testing.assert_close(add.function(a, b), torch.tensor([5.0, 7.0, 9.0]))
        torch.testing.assert_close(sub.function(a, b), torch.tensor([-3.0, -3.0, -3.0]))
        torch.testing.assert_close(mul.function(a, b), torch.tensor([4.0, 10.0, 18.0]))
