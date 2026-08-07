
from __future__ import annotations

import pytest
import sympy

from kd.core.equation import canonicalize_expression
from kd.search.eqgpt.ir_map import TOKEN_IR_ATOM, sentence_to_rhs_terms
from kd.search.eqgpt.vocab import load_vocab
from kd.search.pysr.convert import pysr_sympy_to_kd_terms
from kd.search.sga.convert import pde_to_kd_expr, tree_to_kd_expr
from kd.search.sga.pde import PDE
from kd.search.sga.tree import Node, Tree






def _leaf(name: str) -> Node:
    return Node(name=name, arity=0, children=[])


def _unary(op: str, child: Node) -> Node:
    return Node(name=op, arity=1, children=[child])


def _binary(op: str, left: Node, right: Node) -> Node:
    return Node(name=op, arity=2, children=[left, right])


SGA_TREES = [
    Tree(root=_leaf("u")),
    Tree(root=_binary("*", _leaf("u"), _leaf("x"))),
    Tree(root=_binary("d", _leaf("u"), _leaf("x"))),
    Tree(root=_binary("d^2", _leaf("u"), _leaf("x"))),
    Tree(root=_binary("d", _binary("*", _leaf("u"), _leaf("u_x")), _leaf("x"))),
    Tree(root=_binary("/", _leaf("u"), _leaf("x"))),
    Tree(root=_binary("-", _leaf("u_xx"), _binary("*", _leaf("u"), _leaf("u_x")))),
    Tree(root=_unary("^2", _leaf("u"))),
]

SGA_PDES = [
    PDE(terms=[SGA_TREES[0], SGA_TREES[2]]),
    PDE(terms=[SGA_TREES[1], SGA_TREES[3], SGA_TREES[5]]),
]


@pytest.mark.parametrize("tree", SGA_TREES, ids=lambda t: tree_to_kd_expr(t))
def test_sga_tree_output_is_canonicalizable_and_stable(tree: Tree) -> None:
    emitted = tree_to_kd_expr(tree)
    canonical = canonicalize_expression(emitted)
    assert canonicalize_expression(canonical) == canonical


@pytest.mark.parametrize("pde", SGA_PDES, ids=["two_terms", "three_terms"])
def test_sga_pde_output_is_canonicalizable_and_stable(pde: PDE) -> None:
    emitted = pde_to_kd_expr(pde)
    canonical = canonicalize_expression(emitted)
    assert canonicalize_expression(canonical) == canonical


def test_constant_carrying_ir_is_outside_canonicalizable_subset() -> None:
    with pytest.raises(ValueError):
        canonicalize_expression("mul(0.5, u)")






_START_LEN = 3

EQGPT_SENTENCES = [
    ["S", "ut", "+", "ux", "+", "uxxx", "+", "(uux)xx", "E"],
    ["S", "ut", "+", "u", "*", "ux", "E"],
    ["S", "ut", "+", "uxx", "/", "x", "E"],
    ["S", "ut", "+", "uxx", "/", "x", "*", "u", "E"],
    ["S", "ut", "+", "u^2", "+", "Laplace(u)", "E"],
]


@pytest.fixture(scope="module")
def vocab():
    return load_vocab()


@pytest.mark.parametrize("words", EQGPT_SENTENCES, ids=lambda w: " ".join(w[3:-1]))
def test_eqgpt_rhs_terms_are_canonicalizable_and_stable(
    vocab,
    words: list[str],
) -> None:
    sentence = [vocab.word2id[w] for w in words]
    terms = sentence_to_rhs_terms(vocab, sentence, start_len=_START_LEN)
    assert terms
    for term in terms:
        canonical = canonicalize_expression(term)
        assert canonicalize_expression(canonical) == canonical








EQGPT_CONST_ATOM_TOKENS: frozenset[str] = frozenset()


@pytest.mark.parametrize(
    ("word", "atom"), sorted(TOKEN_IR_ATOM.items()), ids=lambda p: str(p)
)
def test_eqgpt_atom_canonicalizability_is_fully_adjudicated(
    word: str, atom: str
) -> None:
    if word in EQGPT_CONST_ATOM_TOKENS:
        with pytest.raises(ValueError):
            canonicalize_expression(atom)
    else:
        canonical = canonicalize_expression(atom)
        assert canonicalize_expression(canonical) == canonical






_PYSR_FEATURES = ["c0", "c1"]
_PYSR_TERMS = ["u", "diff_x(u)"]
_C0, _C1 = sympy.symbols("c0 c1")


@pytest.mark.parametrize(
    "expr",
    [
        _C0,
        2.5 * _C0,
        _C0 * _C1,
        sympy.sin(_C0),
        3.0 * sympy.exp(_C1) + _C0,
    ],
    ids=str,
)
def test_pysr_constant_free_terms_are_canonicalizable_and_stable(
    expr: sympy.Expr,
) -> None:
    terms = pysr_sympy_to_kd_terms(expr, _PYSR_TERMS, _PYSR_FEATURES)
    assert terms
    for term in terms:
        canonical = canonicalize_expression(term)
        assert canonicalize_expression(canonical) == canonical


def test_pysr_nested_constant_is_outside_canonicalizable_subset() -> None:
    terms = pysr_sympy_to_kd_terms(sympy.sin(2 * _C0), _PYSR_TERMS, _PYSR_FEATURES)
    assert terms == ["sin(mul(2, u))"]
    with pytest.raises(ValueError):
        canonicalize_expression(terms[0])
