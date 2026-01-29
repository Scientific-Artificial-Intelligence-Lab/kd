"""Library of tokens for symbolic regression.

The Library manages available tokens (operators, variables, constants, diff operators)
and provides query methods for different use cases.
"""

from typing import List

from kd2.core.ir.token import Token


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
        raise NotImplementedError("Library.__init__ not implemented")

    def register(self, token: Token) -> None:
        """Register a token in the library.

        Args:
            token: Token to register.

        Raises:
            ValueError: If a token with the same name already exists.
        """
        raise NotImplementedError("Library.register not implemented")

    def get(self, name: str) -> Token:
        """Get a token by name.

        Args:
            name: Token name to look up.

        Returns:
            The token with the given name.

        Raises:
            KeyError: If no token with the given name exists.
        """
        raise NotImplementedError("Library.get not implemented")

    def get_by_arity(self, arity: int) -> List[Token]:
        """Get all tokens with the specified arity.

        Args:
            arity: Number of arguments (0, 1, or 2).

        Returns:
            List of tokens with the given arity.
        """
        raise NotImplementedError("Library.get_by_arity not implemented")

    def get_terminals(self) -> List[Token]:
        """Get all terminal tokens (arity=0).

        Returns:
            List of all terminals (variables, constants).
        """
        raise NotImplementedError("Library.get_terminals not implemented")

    def get_functions(self) -> List[Token]:
        """Get all function tokens (arity > 0).

        Returns:
            List of all functions (operators, diff).
        """
        raise NotImplementedError("Library.get_functions not implemented")

    def register_diff_tokens(
        self, axes: List[str], fields: List[str], max_order: int
    ) -> None:
        """Register differential operator tokens.

        Generates diff tokens for all combinations of axes and fields,
        up to the specified maximum order.

        Token naming convention:
        - Order 1: diff_{axis}  (e.g., diff_x)
        - Order 2+: diff{order}_{axis}  (e.g., diff2_x, diff3_x)

        All diff tokens have arity=1 (unary operators).

        Args:
            axes: Spatial/temporal axes (e.g., ["x", "t"]).
            fields: Field names (e.g., ["u", "v"]).
            max_order: Maximum derivative order.

        Raises:
            ValueError: If max_order < 1.
        """
        raise NotImplementedError("Library.register_diff_tokens not implemented")

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
        raise NotImplementedError("Library.create_default not implemented")
