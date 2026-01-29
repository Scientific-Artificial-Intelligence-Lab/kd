"""Library of tokens for symbolic regression.

The Library manages available tokens (operators, variables, constants, diff operators)
and provides query methods for different use cases.
"""

import torch
from torch import Tensor

from kd2.core.ir.token import Token, TokenType
from kd2.core.safety import safe_div, safe_exp, safe_log


class Library:
    """Registry of tokens available for symbolic expression construction.

    The Library maintains a collection of tokens and provides methods to:
    - Register new tokens
    - Query tokens by name or properties
    - Auto-generate diff tokens for given axes/fields

    Default operators include: add, sub, mul, div, sin, cos, exp, log, n2, n3
    All numerical operations use safe_* functions to prevent NaN/Inf.

    Examples:
        >>> lib = Library.create_default()
        >>> add = lib.get("add")
        >>> unary_ops = lib.get_by_arity(1)
        >>> lib.register_diff_tokens(["x", "t"], ["u"], max_order=2)
    """

    def __init__(self) -> None:
        """Initialize an empty library."""
        self._tokens: dict[str, Token] = {}

    def register(self, token: Token) -> None:
        """Register a token in the library.

        Args:
            token: Token to register.

        Raises:
            ValueError: If a token with the same name already exists.
        """
        if token.name in self._tokens:
            raise ValueError(f"Token '{token.name}' already exists in library")
        self._tokens[token.name] = token

    def get(self, name: str) -> Token:
        """Get a token by name.

        Args:
            name: Token name to look up.

        Returns:
            The token with the given name.

        Raises:
            KeyError: If no token with the given name exists.
        """
        return self._tokens[name]

    def get_by_arity(self, arity: int) -> list[Token]:
        """Get all tokens with the specified arity.

        Args:
            arity: Number of arguments (0, 1, or 2).

        Returns:
            List of tokens with the given arity.
        """
        return [t for t in self._tokens.values() if t.arity == arity]

    def get_terminals(self) -> list[Token]:
        """Get all terminal tokens (arity=0).

        Returns:
            List of all terminals (variables, constants).
        """
        return self.get_by_arity(0)

    def get_functions(self) -> list[Token]:
        """Get all function tokens (arity > 0).

        Returns:
            List of all functions (operators, diff).
        """
        return [t for t in self._tokens.values() if t.arity > 0]

    def register_diff_tokens(
        self, axes: list[str], fields: list[str], max_order: int
    ) -> None:
        """Register differential operator tokens.

        Generates diff tokens for all axes up to the specified maximum order.
        The fields parameter is reserved for future use.

        Token naming convention:
        - Order 1: diff_{axis}  (e.g., diff_x)
        - Order 2+: diff{order}_{axis}  (e.g., diff2_x, diff3_x)

        All diff tokens have arity=1 (unary operators) and function=None.
        The actual differentiation is handled by the Executor.

        Args:
            axes: Spatial/temporal axes (e.g., ["x", "t"]).
            fields: Field names (e.g., ["u", "v"]). Reserved for future use.
            max_order: Maximum derivative order.

        Raises:
            ValueError: If max_order < 1.
        """
        if max_order < 1:
            raise ValueError(f"max_order must be >= 1, got {max_order}")

        for axis in axes:
            for order in range(1, max_order + 1):
                # Naming convention: diff_x for order 1, diff2_x for order 2+
                name = f"diff_{axis}" if order == 1 else f"diff{order}_{axis}"

                token = Token(
                    name=name,
                    arity=1,
                    token_type=TokenType.DIFF,
                    function=None,  # Handled by Executor
                    is_commutative=False,
                )
                self.register(token)

    @classmethod
    def create_default(cls) -> "Library":
        """Create a library with default operators.

        Default operators:
        - Binary (arity=2): add, sub, mul, div
        - Unary (arity=1): sin, cos, exp, log, n2, n3

        All use safe_* functions for numerical stability.

        Returns:
            Library with default operators registered.
        """
        lib = cls()

        # Binary operators
        lib.register(
            Token(
                name="add",
                arity=2,
                token_type=TokenType.FUNCTION,
                function=torch.add,
                is_commutative=True,
            )
        )
        lib.register(
            Token(
                name="sub",
                arity=2,
                token_type=TokenType.FUNCTION,
                function=torch.sub,
                is_commutative=False,
            )
        )
        lib.register(
            Token(
                name="mul",
                arity=2,
                token_type=TokenType.FUNCTION,
                function=torch.mul,
                is_commutative=True,
            )
        )
        lib.register(
            Token(
                name="div",
                arity=2,
                token_type=TokenType.FUNCTION,
                function=_safe_div_wrapper,
                is_commutative=False,
            )
        )

        # Unary operators
        lib.register(
            Token(
                name="sin",
                arity=1,
                token_type=TokenType.FUNCTION,
                function=torch.sin,
                is_commutative=False,
            )
        )
        lib.register(
            Token(
                name="cos",
                arity=1,
                token_type=TokenType.FUNCTION,
                function=torch.cos,
                is_commutative=False,
            )
        )
        lib.register(
            Token(
                name="exp",
                arity=1,
                token_type=TokenType.FUNCTION,
                function=safe_exp,
                is_commutative=False,
            )
        )
        lib.register(
            Token(
                name="log",
                arity=1,
                token_type=TokenType.FUNCTION,
                function=safe_log,
                is_commutative=False,
            )
        )
        lib.register(
            Token(
                name="n2",
                arity=1,
                token_type=TokenType.FUNCTION,
                function=_square,
                is_commutative=False,
            )
        )
        lib.register(
            Token(
                name="n3",
                arity=1,
                token_type=TokenType.FUNCTION,
                function=_cube,
                is_commutative=False,
            )
        )

        return lib


def _safe_div_wrapper(a: Tensor, b: Tensor) -> Tensor:
    """Wrapper for safe_div with default eps."""
    return safe_div(a, b)


def _square(x: Tensor) -> Tensor:
    """Compute x^2."""
    return x * x


def _cube(x: Tensor) -> Tensor:
    """Compute x^3."""
    return x * x * x
