"""Token and TokenType definitions for IR.

A Token represents a single symbol in the IR: function, variable, constant, or diff operator.
Tokens are immutable and hashable, suitable for use as dictionary keys.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

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

    Examples:
        >>> add_token = Token("add", 2, TokenType.FUNCTION, torch.add, is_commutative=True)
        >>> x_token = Token("x", 0, TokenType.VARIABLE, None)
        >>> diff_x = Token("diff_x", 1, TokenType.DIFF, diff_x_func)
    """

    name: str
    arity: int  # 0=terminal, 1=unary, 2=binary
    token_type: TokenType
    function: Optional[Callable[..., Tensor]] = None  # None for terminals
    is_commutative: bool = False  # True for add, mul

    def __post_init__(self) -> None:
        """Validate token properties."""
        raise NotImplementedError("Token validation not implemented")

    def __hash__(self) -> int:
        """Hash based on name only (functions are not hashable)."""
        raise NotImplementedError("Token hash not implemented")

    def __eq__(self, other: object) -> bool:
        """Equality based on all fields except function reference."""
        raise NotImplementedError("Token equality not implemented")
