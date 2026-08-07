
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from kd.core.equation.canonical import canonicalize_expression

COMMUTATIVE_OPS = ("add", "mul")
NON_COMMUTATIVE_OPS = ("sub", "div")
UNARY_OPS = ("neg", "n2", "n3", "diff_x", "diff2_x", "sin", "cos")
LEAVES = ("u", "a", "b", "c", "v", "w", "diffx", "u2")


Tree = str | tuple[str, tuple["Tree", ...]]


def _trees() -> st.SearchStrategy[Tree]:
    leaf = st.sampled_from(LEAVES)

    def extend(children: st.SearchStrategy[Tree]) -> st.SearchStrategy[Tree]:
        unary = st.tuples(
            st.sampled_from(UNARY_OPS),
            st.tuples(children),
        )
        binary = st.tuples(
            st.sampled_from(COMMUTATIVE_OPS + NON_COMMUTATIVE_OPS),
            st.tuples(children, children),
        )
        return st.one_of(unary, binary)

    return st.recursive(leaf, extend, max_leaves=12)


def _render(tree: Tree, *, spaced: bool) -> str:
    if isinstance(tree, str):
        return tree
    op, children = tree
    sep = ", " if spaced else ","
    return f"{op}({sep.join(_render(c, spaced=spaced) for c in children)})"


def _flip_commutative(tree: Tree) -> Tree:
    if isinstance(tree, str):
        return tree
    op, children = tree
    flipped = tuple(_flip_commutative(c) for c in children)
    if op in COMMUTATIVE_OPS:
        flipped = flipped[::-1]
    return (op, flipped)


@settings(deadline=None)
@given(tree=_trees())
def test_idempotence(tree: Tree) -> None:
    once = canonicalize_expression(_render(tree, spaced=True))
    assert canonicalize_expression(once) == once


@settings(deadline=None)
@given(tree=_trees())
def test_commutative_operand_order_invariance(tree: Tree) -> None:
    original = canonicalize_expression(_render(tree, spaced=True))
    flipped = canonicalize_expression(_render(_flip_commutative(tree), spaced=True))
    assert original == flipped


@settings(deadline=None)
@given(tree=_trees())
def test_whitespace_invariance(tree: Tree) -> None:
    spaced = canonicalize_expression(_render(tree, spaced=True))
    compact = canonicalize_expression(_render(tree, spaced=False))
    assert spaced == compact


@settings(deadline=None)
@given(tree=_trees())
def test_output_is_compact(tree: Tree) -> None:
    assert " " not in canonicalize_expression(_render(tree, spaced=True))

