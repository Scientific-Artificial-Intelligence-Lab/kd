"""Tests for GenIR (Generative Intermediate Representation).

Test coverage:
- smoke: Basic instantiation and existence
- unit: Creation, parsing, completeness, dangling, hash, equality, length
"""

import pytest
from dataclasses import FrozenInstanceError

from kd2.core.ir import GenIR, Token, TokenType
from kd2.core.library import Library


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_library() -> Library:
    """Create a Library with basic tokens for testing.

    Includes:
    - Binary operators: add, sub, mul, div (from create_default)
    - Unary operators: sin, cos, exp, log, n2, n3 (from create_default)
    - Variables: u, v, w, x, t
    """
    lib = Library.create_default()
    # Register variables
    for name in ["u", "v", "w", "x", "t"]:
        lib.register(
            Token(name=name, arity=0, token_type=TokenType.VARIABLE, function=None)
        )
    return lib


# =============================================================================
# Smoke Tests
# =============================================================================


@pytest.mark.smoke
class TestGenIRSmoke:
    """Smoke tests: basic instantiation and existence."""

    def test_gen_ir_class_exists(self) -> None:
        """GenIR class exists and is a dataclass."""
        assert hasattr(GenIR, "__dataclass_fields__")

    def test_gen_ir_creation(self) -> None:
        """GenIR can be instantiated with a tuple of tokens."""
        ir = GenIR(tokens=("add", "u", "v"))
        assert ir is not None
        assert ir.tokens == ("add", "u", "v")

    def test_gen_ir_frozen(self) -> None:
        """GenIR is immutable (frozen=True)."""
        ir = GenIR(tokens=("add", "u", "v"))
        with pytest.raises(FrozenInstanceError):
            ir.tokens = ("mul", "u", "v")  # type: ignore[misc]

    def test_gen_ir_from_string_exists(self) -> None:
        """GenIR.from_string class method exists."""
        assert hasattr(GenIR, "from_string")
        assert callable(GenIR.from_string)


# =============================================================================
# Unit Tests - from_string
# =============================================================================


@pytest.mark.unit
class TestGenIRFromString:
    """Unit tests for GenIR.from_string parsing."""

    def test_from_string_simple(self, default_library: Library) -> None:
        """Parse simple binary expression: 'add,u,v'."""
        ir = GenIR.from_string("add,u,v", default_library)
        assert ir.tokens == ("add", "u", "v")

    def test_from_string_nested(self, default_library: Library) -> None:
        """Parse nested expression: 'mul,add,u,v,w'."""
        ir = GenIR.from_string("mul,add,u,v,w", default_library)
        assert ir.tokens == ("mul", "add", "u", "v", "w")

    def test_from_string_unary(self, default_library: Library) -> None:
        """Parse unary expression: 'sin,u'."""
        ir = GenIR.from_string("sin,u", default_library)
        assert ir.tokens == ("sin", "u")

    def test_from_string_single_token(self, default_library: Library) -> None:
        """Parse single terminal: 'u'."""
        ir = GenIR.from_string("u", default_library)
        assert ir.tokens == ("u",)

    def test_from_string_complex(self, default_library: Library) -> None:
        """Parse complex expression: 'add,mul,u,v,sin,w'."""
        ir = GenIR.from_string("add,mul,u,v,sin,w", default_library)
        assert ir.tokens == ("add", "mul", "u", "v", "sin", "w")

    def test_from_string_with_spaces(self, default_library: Library) -> None:
        """Parse string with whitespace around tokens."""
        ir = GenIR.from_string("add, u, v", default_library)
        assert ir.tokens == ("add", "u", "v")

    def test_from_string_with_leading_trailing_spaces(
        self, default_library: Library
    ) -> None:
        """Parse string with leading/trailing whitespace."""
        ir = GenIR.from_string("  add,u,v  ", default_library)
        assert ir.tokens == ("add", "u", "v")


# =============================================================================
# Unit Tests - to_string
# =============================================================================


@pytest.mark.unit
class TestGenIRToString:
    """Unit tests for GenIR.to_string conversion."""

    def test_to_string_simple(self) -> None:
        """Convert simple GenIR to string."""
        ir = GenIR(tokens=("add", "u", "v"))
        assert ir.to_string() == "add,u,v"

    def test_to_string_nested(self) -> None:
        """Convert nested GenIR to string."""
        ir = GenIR(tokens=("mul", "add", "u", "v", "w"))
        assert ir.to_string() == "mul,add,u,v,w"

    def test_to_string_single(self) -> None:
        """Convert single token GenIR to string."""
        ir = GenIR(tokens=("u",))
        assert ir.to_string() == "u"

    def test_to_string_roundtrip(self, default_library: Library) -> None:
        """from_string -> to_string preserves the content."""
        original = "add,mul,u,v,sin,w"
        ir = GenIR.from_string(original, default_library)
        result = ir.to_string()
        assert result == original

    def test_to_string_roundtrip_complex(self, default_library: Library) -> None:
        """Roundtrip for complex nested expression."""
        original = "mul,add,sin,u,cos,v,exp,w"
        ir = GenIR.from_string(original, default_library)
        assert ir.to_string() == original


# =============================================================================
# Unit Tests - is_complete
# =============================================================================


@pytest.mark.unit
class TestGenIRIsComplete:
    """Unit tests for GenIR.is_complete method."""

    def test_is_complete_simple(self, default_library: Library) -> None:
        """'add,u,v' is a complete expression."""
        ir = GenIR(tokens=("add", "u", "v"))
        assert ir.is_complete(default_library) is True

    def test_is_complete_nested(self, default_library: Library) -> None:
        """'mul,add,u,v,w' is a complete expression."""
        ir = GenIR(tokens=("mul", "add", "u", "v", "w"))
        assert ir.is_complete(default_library) is True

    def test_is_complete_unary(self, default_library: Library) -> None:
        """'sin,u' is a complete expression."""
        ir = GenIR(tokens=("sin", "u"))
        assert ir.is_complete(default_library) is True

    def test_is_complete_terminal(self, default_library: Library) -> None:
        """Single terminal 'u' is a complete expression."""
        ir = GenIR(tokens=("u",))
        assert ir.is_complete(default_library) is True

    def test_is_complete_double_unary(self, default_library: Library) -> None:
        """'sin,cos,u' is a complete expression."""
        ir = GenIR(tokens=("sin", "cos", "u"))
        assert ir.is_complete(default_library) is True

    def test_incomplete_missing_operand(self, default_library: Library) -> None:
        """'add,u' is incomplete (missing one operand)."""
        ir = GenIR(tokens=("add", "u"))
        assert ir.is_complete(default_library) is False

    def test_incomplete_operator_only(self, default_library: Library) -> None:
        """'add' alone is incomplete (missing both operands)."""
        ir = GenIR(tokens=("add",))
        assert ir.is_complete(default_library) is False

    def test_incomplete_nested(self, default_library: Library) -> None:
        """'mul,add,u,v' is incomplete (missing mul's second operand)."""
        ir = GenIR(tokens=("mul", "add", "u", "v"))
        assert ir.is_complete(default_library) is False

    def test_incomplete_unary(self, default_library: Library) -> None:
        """'sin' alone is incomplete."""
        ir = GenIR(tokens=("sin",))
        assert ir.is_complete(default_library) is False

    def test_incomplete_empty(self, default_library: Library) -> None:
        """Empty GenIR is not complete."""
        ir = GenIR(tokens=())
        assert ir.is_complete(default_library) is False


# =============================================================================
# Unit Tests - dangling
# =============================================================================


@pytest.mark.unit
class TestGenIRDangling:
    """Unit tests for GenIR.dangling method.

    Algorithm: dangling = 1, for each token: dangling = dangling - 1 + arity
    """

    def test_dangling_complete_binary(self, default_library: Library) -> None:
        """'add,u,v': dangling = 1 -1+2 -1+0 -1+0 = 0."""
        ir = GenIR(tokens=("add", "u", "v"))
        assert ir.dangling(default_library) == 0

    def test_dangling_incomplete_1(self, default_library: Library) -> None:
        """'add,u': dangling = 1 -1+2 -1+0 = 1."""
        ir = GenIR(tokens=("add", "u"))
        assert ir.dangling(default_library) == 1

    def test_dangling_incomplete_2(self, default_library: Library) -> None:
        """'add': dangling = 1 -1+2 = 2."""
        ir = GenIR(tokens=("add",))
        assert ir.dangling(default_library) == 2

    def test_dangling_nested(self, default_library: Library) -> None:
        """'mul,add,u,v': dangling = 1 -1+2 -1+2 -1+0 -1+0 = 1."""
        ir = GenIR(tokens=("mul", "add", "u", "v"))
        assert ir.dangling(default_library) == 1

    def test_dangling_terminal(self, default_library: Library) -> None:
        """'u': dangling = 1 -1+0 = 0."""
        ir = GenIR(tokens=("u",))
        assert ir.dangling(default_library) == 0

    def test_dangling_complete_nested(self, default_library: Library) -> None:
        """'mul,add,u,v,w': dangling = 1 -1+2 -1+2 -1+0 -1+0 -1+0 = 0."""
        ir = GenIR(tokens=("mul", "add", "u", "v", "w"))
        assert ir.dangling(default_library) == 0

    def test_dangling_unary_complete(self, default_library: Library) -> None:
        """'sin,u': dangling = 1 -1+1 -1+0 = 0."""
        ir = GenIR(tokens=("sin", "u"))
        assert ir.dangling(default_library) == 0

    def test_dangling_unary_incomplete(self, default_library: Library) -> None:
        """'sin': dangling = 1 -1+1 = 1."""
        ir = GenIR(tokens=("sin",))
        assert ir.dangling(default_library) == 1

    def test_dangling_double_unary(self, default_library: Library) -> None:
        """'sin,cos,u': dangling = 1 -1+1 -1+1 -1+0 = 0."""
        ir = GenIR(tokens=("sin", "cos", "u"))
        assert ir.dangling(default_library) == 0

    def test_dangling_deeply_nested(self, default_library: Library) -> None:
        """'add,add,u,v,add,w,t': fully nested complete."""
        ir = GenIR(tokens=("add", "add", "u", "v", "add", "w", "t"))
        # 1 -1+2 -1+2 -1+0 -1+0 -1+2 -1+0 -1+0 = 0
        assert ir.dangling(default_library) == 0

    def test_dangling_empty(self, default_library: Library) -> None:
        """Empty GenIR: dangling = 1 (needs one expression)."""
        ir = GenIR(tokens=())
        assert ir.dangling(default_library) == 1


# =============================================================================
# Unit Tests - Hash and Equality
# =============================================================================


@pytest.mark.unit
class TestGenIRHashEquality:
    """Unit tests for GenIR hash and equality."""

    def test_hash_same_tokens(self) -> None:
        """GenIRs with same tokens have the same hash."""
        ir1 = GenIR(tokens=("add", "u", "v"))
        ir2 = GenIR(tokens=("add", "u", "v"))
        assert hash(ir1) == hash(ir2)

    def test_hash_different_tokens(self) -> None:
        """GenIRs with different tokens have different hashes."""
        ir1 = GenIR(tokens=("add", "u", "v"))
        ir2 = GenIR(tokens=("mul", "u", "v"))
        # Note: hash collisions are possible but extremely rare
        assert hash(ir1) != hash(ir2)

    def test_equality_same(self) -> None:
        """GenIRs with same tokens are equal."""
        ir1 = GenIR(tokens=("add", "u", "v"))
        ir2 = GenIR(tokens=("add", "u", "v"))
        assert ir1 == ir2

    def test_equality_different(self) -> None:
        """GenIRs with different tokens are not equal."""
        ir1 = GenIR(tokens=("add", "u", "v"))
        ir2 = GenIR(tokens=("add", "v", "u"))
        assert ir1 != ir2

    def test_equality_different_length(self) -> None:
        """GenIRs with different lengths are not equal."""
        ir1 = GenIR(tokens=("add", "u", "v"))
        ir2 = GenIR(tokens=("add", "u"))
        assert ir1 != ir2

    def test_as_dict_key(self) -> None:
        """GenIR can be used as dictionary key."""
        ir = GenIR(tokens=("add", "u", "v"))
        d = {ir: "value"}
        assert d[ir] == "value"

        # Same tokens, different instance
        ir2 = GenIR(tokens=("add", "u", "v"))
        assert d[ir2] == "value"

    def test_as_set_member(self) -> None:
        """GenIR can be added to a set."""
        ir1 = GenIR(tokens=("add", "u", "v"))
        ir2 = GenIR(tokens=("mul", "u", "v"))
        ir3 = GenIR(tokens=("add", "u", "v"))  # Same as ir1

        s = {ir1, ir2, ir3}
        assert len(s) == 2  # ir1 and ir3 are equal
        assert ir1 in s
        assert ir2 in s

    def test_equality_not_equal_to_non_genir(self) -> None:
        """GenIR is not equal to non-GenIR objects."""
        ir = GenIR(tokens=("add", "u", "v"))
        assert ir != ("add", "u", "v")  # Not equal to plain tuple
        assert ir != "add,u,v"  # Not equal to string
        assert ir != None  # noqa: E711


# =============================================================================
# Unit Tests - __len__
# =============================================================================


@pytest.mark.unit
class TestGenIRLen:
    """Unit tests for GenIR.__len__ method."""

    def test_len_simple(self) -> None:
        """Length of 'add,u,v' is 3."""
        ir = GenIR(tokens=("add", "u", "v"))
        assert len(ir) == 3

    def test_len_single(self) -> None:
        """Length of single token is 1."""
        ir = GenIR(tokens=("u",))
        assert len(ir) == 1

    def test_len_nested(self) -> None:
        """Length of nested expression."""
        ir = GenIR(tokens=("mul", "add", "u", "v", "w"))
        assert len(ir) == 5

    def test_len_complex(self) -> None:
        """Length of complex expression."""
        ir = GenIR(tokens=("add", "mul", "u", "v", "sin", "w"))
        assert len(ir) == 6

    def test_len_empty(self) -> None:
        """Length of empty GenIR is 0."""
        ir = GenIR(tokens=())
        assert len(ir) == 0


# =============================================================================
# Unit Tests - Error Handling
# =============================================================================


@pytest.mark.unit
class TestGenIRErrors:
    """Unit tests for GenIR error handling."""

    def test_from_string_unknown_token(self, default_library: Library) -> None:
        """Unknown token in string raises ValueError."""
        with pytest.raises(ValueError, match="unknown|not found|not in"):
            GenIR.from_string("add,u,unknown_token", default_library)

    def test_from_string_empty(self, default_library: Library) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            GenIR.from_string("", default_library)

    def test_from_string_whitespace_only(self, default_library: Library) -> None:
        """Whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            GenIR.from_string("   ", default_library)

    def test_from_string_multiple_unknown(self, default_library: Library) -> None:
        """Multiple unknown tokens - first one triggers error."""
        with pytest.raises(ValueError):
            GenIR.from_string("foo,bar,baz", default_library)

    def test_from_string_consecutive_commas(self, default_library: Library) -> None:
        """Consecutive commas should raise ValueError."""
        with pytest.raises(ValueError, match="empty token"):
            GenIR.from_string("add,,u,v", default_library)

    def test_from_string_trailing_comma(self, default_library: Library) -> None:
        """Trailing comma should raise ValueError."""
        with pytest.raises(ValueError, match="empty token"):
            GenIR.from_string("add,u,v,", default_library)

    def test_from_string_leading_comma(self, default_library: Library) -> None:
        """Leading comma should raise ValueError."""
        with pytest.raises(ValueError, match="empty token"):
            GenIR.from_string(",add,u,v", default_library)
