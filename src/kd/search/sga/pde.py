
from __future__ import annotations

import copy

from kd.search.sga.tree import Tree


class PDE:

    __slots__ = ("terms",)

    def __init__(self, terms: list[Tree] | None = None) -> None:
        self.terms: list[Tree] = terms if terms is not None else []



    @property
    def width(self) -> int:
        return len(self.terms)



    def copy(self) -> PDE:
        return copy.deepcopy(self)



    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PDE):
            return NotImplemented
        return self.terms == other.terms

    def __str__(self) -> str:
        return " + ".join(str(t) for t in self.terms)

    def __repr__(self) -> str:
        return f"PDE(terms={self.terms!r})"
