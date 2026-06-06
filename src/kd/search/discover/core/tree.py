
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from kd.search.discover.tokens.library import _DIFF_ORDERS, Library, Token






_ADDITIVE_TOKEN_NAMES: frozenset[str] = frozenset({"add", "sub", "add_t", "sub_t"})


@dataclass(slots=True)
class TreeNode:

    token: Token
    children: list[TreeNode] = field(default_factory=list)
    parent: TreeNode | None = field(default=None, repr=False, compare=False)


@dataclass(slots=True)
class ExpressionTree:

    root: TreeNode
    _library: Library | None = field(default=None, repr=False)

    @classmethod
    def from_preorder(cls, tokens: list[int], library: Library) -> ExpressionTree:
        if not tokens:
            raise ValueError("Cannot build tree from an empty token sequence.")

        root = _build_tree_iterative(tokens, library)
        return cls(root=root, _library=library)

    def to_preorder(self) -> list[int]:
        library = _require_library(self._library)
        return [
            library.name_to_index(node.token.name)
            for node in _iter_preorder(self.root)
        ]

    def to_tokens(self) -> list[Token]:
        return [node.token for node in _iter_preorder(self.root)]

    def depth(self) -> int:
        return _depth(self.root)

    def n_nodes(self) -> int:
        return sum(1 for _ in _iter_preorder(self.root))

    def n_terms(self) -> int:
        return _count_terms(self.root)

    def is_complete(self) -> bool:
        return _is_complete(self.root)


_INCOMPLETE = -1


def natural_length(tokens: np.ndarray, library: Library) -> int:
    if tokens.size == 0:
        return _INCOMPLETE
    n_tokens = len(library.tokens)
    dangling = 1
    for position in range(tokens.shape[0]):
        token_index = int(tokens[position])
        if token_index < 0 or token_index >= n_tokens:
            return _INCOMPLETE
        dangling += int(library.arities[token_index]) - 1
        if dangling == 0:
            return position + 1
    return _INCOMPLETE


def trim_to_natural(tokens: np.ndarray, library: Library) -> np.ndarray:
    length = natural_length(tokens, library)
    if length == _INCOMPLETE:
        raise ValueError("Incomplete or empty token sequence — cannot trim.")
    return tokens[:length].copy()


def finish_tokens(tokens: list[int], library: Library) -> list[int]:
    if not tokens:
        return [_first_terminal_index(library)]

    dangling = 1
    for index, token_index in enumerate(tokens):
        dangling += library[token_index].arity - 1
        if dangling == 0:
            return tokens[: index + 1]

    terminal_index = _first_terminal_index(library)
    return tokens + [terminal_index] * dangling


def _build_tree_iterative(
    tokens: Sequence[int],
    library: Library,
) -> TreeNode:
    if not tokens:
        raise ValueError("Incomplete token sequence.")

    root = TreeNode(token=library[tokens[0]])

    stack: list[list[object]] = []
    if root.token.arity > 0:
        stack.append([root, root.token.arity])

    for token_index in tokens[1:]:
        if not stack:
            raise ValueError("Token sequence contains trailing tokens.")
        parent_frame = stack[-1]
        parent_node = parent_frame[0]
        assert isinstance(parent_node, TreeNode)
        child = TreeNode(token=library[token_index], parent=parent_node)
        parent_node.children.append(child)

        remaining = parent_frame[1]
        assert isinstance(remaining, int)
        parent_frame[1] = remaining - 1


        if child.token.arity > 0:
            stack.append([child, child.token.arity])

        while stack:
            top_remaining = stack[-1][1]
            assert isinstance(top_remaining, int)
            if top_remaining != 0:
                break
            stack.pop()

    if stack:
        raise ValueError("Incomplete token sequence.")
    return root


def _iter_preorder(root: TreeNode) -> Iterator[TreeNode]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _depth(root: TreeNode) -> int:
    best = 0
    stack: list[tuple[TreeNode, int]] = [(root, 1)]
    while stack:
        node, current_depth = stack.pop()
        if current_depth > best:
            best = current_depth
        for child in node.children:
            stack.append((child, current_depth + 1))
    return best


def _count_terms(root: TreeNode) -> int:
    count = 0
    stack: list[TreeNode] = [root]
    while stack:
        node = stack.pop()
        if node.token.name not in _ADDITIVE_TOKEN_NAMES:
            count += 1
            continue
        if len(node.children) != 2:
            raise ValueError(
                f"Malformed additive node '{node.token.name}': expected 2 "
                f"children, got {len(node.children)}."
            )
        stack.append(node.children[0])
        stack.append(node.children[1])
    return count


def _is_complete(root: TreeNode) -> bool:
    stack: list[TreeNode] = [root]
    while stack:
        node = stack.pop()
        if len(node.children) != node.token.arity:
            return False
        stack.extend(node.children)
    return True


def _first_terminal_index(library: Library) -> int:
    if len(library.terminal_tokens) == 0:
        raise ValueError("Library must contain at least one terminal token.")
    return int(library.terminal_tokens[0])


def max_diff_order(tokens: np.ndarray, library: Library) -> int:
    if tokens.size == 0:
        return 0
    n_tokens = len(library.tokens)
    best = 0

    stack: list[list[int]] = []
    for pos in range(tokens.size):
        tok_idx = int(tokens[pos])
        if tok_idx < 0 or tok_idx >= n_tokens:





            raise ValueError(
                f"max_diff_order received token index {tok_idx} at position "
                f"{pos}, out of range for library of size {n_tokens}.",
            )
        token = library.tokens[tok_idx]
        diff_ord = _DIFF_ORDERS.get(token.name, 0)
        parent_ord = stack[-1][1] if stack else 0



        current = parent_ord + diff_ord
        if current > best:
            best = current

        if stack:
            stack[-1][0] -= 1
            while stack and stack[-1][0] == 0:
                stack.pop()

        if token.arity > 0:
            stack.append([token.arity, current])
    return best


def _require_library(library: Library | None) -> Library:
    if library is None:
        raise ValueError("ExpressionTree is missing its source library.")
    return library
