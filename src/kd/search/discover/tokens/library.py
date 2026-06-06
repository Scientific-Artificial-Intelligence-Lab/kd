
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TokenType(Enum):

    OPERATOR = "operator"
    TERMINAL = "terminal"
    COORDINATE = "coordinate"





_DIFF_OPERATORS: frozenset[str] = frozenset({
    "diff", "diff2", "diff3", "diff4",
    "diff_x", "diff2_x", "diff3_x", "diff4_x",
    "diff_y", "diff2_y",
    "diff_t", "diff2_t",
})
_SPECIAL_DIFF_OPERATORS: frozenset[str] = frozenset({"lap"})

_OPERATOR_ARITIES: dict[str, int] = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "exp": 1,
    "log": 1,
    "sqrt": 1,
    "n2": 1,
    "n3": 1,
    "n4": 1,
    "n5": 1,
    "neg": 1,
    "abs": 1,
    "inv": 1,
    "tanh": 1,
    "sigmoid": 1,

    "diff": 1,
    "diff2": 1,
    "diff3": 1,
    "diff4": 1,
    "diff_x": 1,
    "diff2_x": 1,
    "diff3_x": 1,
    "diff4_x": 1,
    "diff_y": 1,
    "diff2_y": 1,
    "diff_t": 1,
    "diff2_t": 1,
    "lap": 1,
}



_DIFF_ORDERS: dict[str, int] = {
    "diff": 1, "diff2": 2, "diff3": 3, "diff4": 4,
    "diff_x": 1, "diff2_x": 2, "diff3_x": 3, "diff4_x": 4,
    "diff_y": 1, "diff2_y": 2,
    "diff_t": 1, "diff2_t": 2,
    "lap": 2,
}

_TRIG_NAMES: frozenset[str] = frozenset({
    "sin", "cos", "tan", "csc", "sec", "cot",
    "arcsin", "arccos", "arctan", "arccsc", "arcsec", "arccot",
})

_INVERSE_PAIRS: dict[str, str] = {
    "inv": "inv", "neg": "neg",
    "exp": "log", "log": "exp",
    "sqrt": "n2", "n2": "sqrt",
}


_OPERATOR_COMPLEXITIES: dict[str, int] = {
    "add": 1,
    "sub": 1,
    "mul": 1,
    "div": 2,
    "sin": 3,
    "cos": 3,
    "tan": 4,
    "exp": 4,
    "log": 4,
    "sqrt": 4,
    "n2": 2,
    "n3": 3,
    "n4": 3,
    "n5": 3,
    "neg": 1,
    "abs": 2,
    "inv": 2,
    "tanh": 4,
    "sigmoid": 4,
    "diff": 2,
    "diff2": 3,
    "diff3": 4,
    "diff4": 5,
    "diff_x": 2,
    "diff2_x": 3,
    "diff3_x": 4,
    "diff4_x": 5,
    "diff_y": 2,
    "diff2_y": 3,
    "diff_t": 2,
    "diff2_t": 3,


    "lap": 3,
}


@dataclass(frozen=True, slots=True)
class Token:

    name: str
    arity: int
    token_type: TokenType
    complexity: int = 1

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class LibraryConfig:

    operators: list[str]
    state_vars: list[str] = field(default_factory=lambda: ["u1"])
    coord_vars: list[str] = field(default_factory=lambda: ["x1"])


class Library:

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        n_tokens = len(tokens)

        self.names: list[str] = [t.name for t in tokens]
        if len(set(self.names)) != len(self.names):
            from collections import Counter
            dupes = [n for n, c in Counter(self.names).items() if c > 1]
            raise ValueError(f"Duplicate token names: {dupes}")
        self._name_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(self.names)
        }

        self.arities = np.array(
            [t.arity for t in tokens], dtype=np.int32
        )


        self.terminal_tokens = np.array(
            [i for i in range(n_tokens) if self.arities[i] == 0], dtype=np.int32
        )
        self.unary_tokens = np.array(
            [i for i in range(n_tokens) if self.arities[i] == 1], dtype=np.int32
        )
        self.binary_tokens = np.array(
            [i for i in range(n_tokens) if self.arities[i] == 2], dtype=np.int32
        )


        self.parent_adjust = np.full(n_tokens, -1, dtype=np.int32)
        count = 0
        for i in range(n_tokens):
            if self.arities[i] > 0:
                self.parent_adjust[i] = count
                count += 1


        self.diff_tokens = np.array(
            [i for i, t in enumerate(tokens) if t.name in _DIFF_OPERATORS],
            dtype=np.int32,
        )
        self.special_diff_tokens = np.array(
            [i for i, t in enumerate(tokens) if t.name in _SPECIAL_DIFF_OPERATORS],
            dtype=np.int32,
        )


        self.trig_tokens = np.array(
            [i for i, t in enumerate(tokens) if t.name in _TRIG_NAMES],
            dtype=np.int32,
        )


        self.inverse_tokens: dict[int, int] = {
            self._name_to_idx[k]: self._name_to_idx[v]
            for k, v in _INVERSE_PAIRS.items()
            if k in self._name_to_idx and v in self._name_to_idx
        }


        self.n_action_inputs: int = n_tokens + 1
        self.n_parent_inputs: int = n_tokens + 1 - len(self.terminal_tokens)
        self.n_sibling_inputs: int = n_tokens + 1
        self.EMPTY_ACTION: int = self.n_action_inputs - 1
        self.EMPTY_PARENT: int = self.n_parent_inputs - 1
        self.EMPTY_SIBLING: int = self.n_sibling_inputs - 1

    def __getitem__(self, val: str | int) -> Token:




        if isinstance(val, bool):
            raise KeyError(
                "Library cannot be indexed by bool; pass an int or str name "
                f"(got {val!r}).",
            )
        if isinstance(val, str):
            try:
                i = self._name_to_idx[val]
            except KeyError:




                raise KeyError(f"Token '{val}' does not exist")
        elif isinstance(val, (int, np.integer)):
            if val < 0 or val >= len(self.tokens):
                raise KeyError(
                    f"Token index {val} out of range [0, {len(self.tokens)})"
                )
            i = int(val)
        else:
            raise KeyError(
                f"Library must be indexed by str or int, not {type(val)}"
            )
        return self.tokens[i]

    def name_to_index(self, name: str) -> int:
        try:
            return self._name_to_idx[name]
        except KeyError:


            raise KeyError(f"Token '{name}' does not exist")

    def index_to_name(self, index: int) -> str:
        if index < 0 or index >= len(self.tokens):
            raise KeyError(f"Token index {index} out of range")
        return self.names[index]

    @classmethod
    def from_config(cls, config: LibraryConfig) -> Library:
        tokens: list[Token] = []


        for name in config.coord_vars:
            tokens.append(
                Token(name=name, arity=0, token_type=TokenType.COORDINATE)
            )


        for name in config.state_vars:
            tokens.append(
                Token(name=name, arity=0, token_type=TokenType.TERMINAL)
            )


        for name in config.operators:
            arity = _OPERATOR_ARITIES.get(name)
            if arity is None:
                raise ValueError(f"Unknown operator: '{name}'")
            complexity = _OPERATOR_COMPLEXITIES.get(name, 1)
            tokens.append(
                Token(
                    name=name,
                    arity=arity,
                    token_type=TokenType.OPERATOR,
                    complexity=complexity,
                )
            )

        return cls(tokens)
