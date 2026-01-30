"""GenIR: Generative Intermediate Representation.

GenIR represents symbolic expressions as a sequence of tokens in prefix notation.
It is immutable and hashable, suitable for use in search algorithms and caching.

Examples:
    add(u, v) -> ("add", "u", "v")
    mul(add(u, v), w) -> ("mul", "add", "u", "v", "w")
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kd2.core.library import Library


@dataclass(frozen=True)
class GenIR:
    """Immutable sequence of tokens representing a symbolic expression.

    GenIR uses prefix notation where operators come before their operands.
    The sequence is stored as a tuple for immutability and hashability.

    Attributes:
        tokens: Tuple of token names in prefix order.

    Examples:
        >>> ir = GenIR(("add", "u", "v"))  # add(u, v)
        >>> ir = GenIR(("mul", "add", "u", "v", "w"))  # mul(add(u, v), w)
    """

    tokens: tuple[str, ...]

    @classmethod
    def from_string(cls, s: str, library: "Library") -> "GenIR":
        """Parse a comma-separated string into GenIR.

        Args:
            s: Comma-separated token names (e.g., "add,u,v").
            library: Library to validate token names against.

        Returns:
            GenIR instance with parsed tokens.

        Raises:
            ValueError: If any token is not found in the library.
            ValueError: If the string is empty or contains only whitespace.
        """
        # Strip leading/trailing whitespace from entire string
        s = s.strip()
        if not s:
            raise ValueError("empty token sequence")

        # Split by comma and process each token
        parts = s.split(",")
        token_names: list[str] = []

        for part in parts:
            name = part.strip()
            if not name:
                raise ValueError("empty token in sequence")
            # Validate token exists in library
            try:
                library.get(name)
            except KeyError as err:
                raise ValueError(f"Unknown token: {name}") from err
            token_names.append(name)

        return cls(tokens=tuple(token_names))

    def to_string(self) -> str:
        """Convert GenIR to comma-separated string.

        Returns:
            Comma-separated token names (e.g., "add,u,v").
        """
        return ",".join(self.tokens)

    def is_complete(self, library: "Library") -> bool:
        """Check if the expression is complete (no dangling slots).

        A complete expression has exactly as many operands as required
        by all operators in the sequence.

        Args:
            library: Library to look up token arities.

        Returns:
            True if the expression is complete, False otherwise.
        """
        return self.dangling(library) == 0

    def dangling(self, library: "Library") -> int:
        """Count the number of unfilled argument slots.

        Algorithm:
            dangling = 1
            for each token:
                dangling = dangling - 1 + arity
            return dangling

        Args:
            library: Library to look up token arities.

        Returns:
            Number of dangling slots (0 means complete).
        """
        result = 1  # Initially need one expression
        for token_name in self.tokens:
            token = library.get(token_name)
            result = result - 1 + token.arity
        return result

    def __len__(self) -> int:
        """Return the number of tokens.

        Returns:
            Number of tokens in the sequence.
        """
        return len(self.tokens)

    def __hash__(self) -> int:
        """Hash based on tokens tuple.

        Returns:
            Hash value for use in dicts and sets.
        """
        # Delegate to the tuple's hash for consistency
        return hash(self.tokens)
