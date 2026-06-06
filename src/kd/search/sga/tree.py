
from __future__ import annotations

import copy


class Node:

    __slots__ = ("name", "arity", "children")

    def __init__(
        self,
        name: str,
        arity: int,
        children: list[Node] | None = None,
    ) -> None:
        self.name = name
        self.arity = arity
        self.children: list[Node] = children if children is not None else []



    @property
    def is_leaf(self) -> bool:
        return self.arity == 0

    @property
    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(child.depth for child in self.children)

    @property
    def size(self) -> int:
        return 1 + sum(child.size for child in self.children)



    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return (
            self.name == other.name
            and self.arity == other.arity
            and self.children == other.children
        )

    def __str__(self) -> str:
        parts: list[str] = []
        self._prefix_collect(parts)
        return " ".join(parts)

    def _prefix_collect(self, parts: list[str]) -> None:
        parts.append(self.name)
        for child in self.children:
            child._prefix_collect(parts)

    def __repr__(self) -> str:
        return f"Node({self.name!r}, arity={self.arity}, children={self.children!r})"


class Tree:

    __slots__ = ("root",)

    def __init__(self, root: Node) -> None:
        self.root = root



    @property
    def depth(self) -> int:
        return self.root.depth

    @property
    def size(self) -> int:
        return self.root.size



    def copy(self) -> Tree:
        return copy.deepcopy(self)



    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tree):
            return NotImplemented
        return self.root == other.root

    def __str__(self) -> str:
        return str(self.root)

    def __repr__(self) -> str:
        return f"Tree({self.root!r})"
