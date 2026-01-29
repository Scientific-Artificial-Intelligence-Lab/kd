"""Token and TokenType definitions for IR.

A Token represents a single symbol in the IR: function, variable,
constant, or diff operator. Tokens are immutable and hashable,
suitable for use as dictionary keys.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from torch import Tensor


class TokenType(Enum):
    """Type of token in the IR."""

    FUNCTION = "function"  # Operators like add, mul, sin
    VARIABLE = "variable"  # Input variables like u, x
    CONSTANT = "constant"  # Numeric constants
    DIFF = "diff"  # Differential operators like diff_x, diff2_x


@dataclass(frozen=True)
class Token:
    """Immutable token representing a symbol in the IR.

    Tokens are the building blocks of symbolic expressions. Each token has:
    - name: Unique identifier (e.g., "add", "x", "diff_x")
    - arity: Number of arguments (0=terminal, 1=unary, 2=binary)
    - token_type: Category of the token
    - function: Callable implementing the operation (None for terminals)
    - is_commutative: Whether argument order matters (for add, mul)

    Equality and hashing are based on name, arity, token_type, and
    is_commutative, but NOT on function reference. This supports
    serialization/deserialization where function references are
    re-bound after loading.

    Examples:
        >>> add_token = Token(
        ...     "add", 2, TokenType.FUNCTION, torch.add, is_commutative=True
        ... )
        >>> x_token = Token("x", 0, TokenType.VARIABLE, None)
        >>> diff_x = Token("diff_x", 1, TokenType.DIFF, diff_x_func)
    """

    name: str
    arity: int  # 0=terminal, 1=unary, 2=binary
    token_type: TokenType
    function: Callable[..., Tensor] | None = None  # None for terminals
    is_commutative: bool = False  # True for add, mul

    def __hash__(self) -> int:
        """Hash based on identity fields (excluding function reference).

        Function references are excluded because:
        1. Functions may not be hashable
        2. Two tokens with the same identity should hash the same
           even if they have different function references
        """
        return hash((self.name, self.arity, self.token_type, self.is_commutative))

    def __eq__(self, other: object) -> bool:
        """Equality based on identity fields (excluding function reference).

        Two tokens are equal if they have the same name, arity, token_type,
        and is_commutative. The function reference is NOT compared because:
        1. Supports serialization where functions are re-bound after loading
        2. Two tokens with the same identity should be considered equal
        """
        if not isinstance(other, Token):
            return False
        return (
            self.name == other.name
            and self.arity == other.arity
            and self.token_type == other.token_type
            and self.is_commutative == other.is_commutative
        )
