
from __future__ import annotations

import pytest
from kd.search.sga.pde import PDE
from kd.search.sga.tree import Node, Tree






def _make_leaf(name: str) -> Tree:
    return Tree(root=Node(name=name, arity=0, children=[]))


def _make_mul_tree(left_name: str, right_name: str) -> Tree:
    left = Node(name=left_name, arity=0, children=[])
    right = Node(name=right_name, arity=0, children=[])
    root = Node(name="*", arity=2, children=[left, right])
    return Tree(root=root)







class TestPDEConstruction:

    @pytest.mark.smoke
    def test_pde_creation_from_list(self) -> None:
        terms = [_make_leaf("u"), _make_mul_tree("u", "x")]
        pde = PDE(terms=terms)
        assert pde.terms is terms

    @pytest.mark.smoke
    def test_pde_empty_terms(self) -> None:
        pde = PDE(terms=[])
        assert pde.terms == []







class TestPDEWidth:

    def test_width_matches_len_terms(self) -> None:
        terms = [_make_leaf("u"), _make_leaf("x"), _make_mul_tree("u", "x")]
        pde = PDE(terms=terms)
        assert pde.width == len(terms)

    def test_width_zero_for_empty(self) -> None:
        pde = PDE(terms=[])
        assert pde.width == 0

    def test_width_one(self) -> None:
        pde = PDE(terms=[_make_leaf("u")])
        assert pde.width == 1







class TestPDECopy:

    def test_copy_equals_original(self) -> None:
        terms = [_make_leaf("u"), _make_mul_tree("u", "x")]
        original = PDE(terms=terms)
        copied = original.copy()

        assert copied.width == original.width
        for orig_t, copy_t in zip(original.terms, copied.terms):
            assert orig_t == copy_t

    def test_copy_is_independent(self) -> None:
        terms = [_make_mul_tree("u", "x")]
        original = PDE(terms=terms)
        copied = original.copy()


        copied.terms[0].root.children[0] = Node(name="t", arity=0, children=[])


        assert original.terms[0].root.children[0].name == "u"

    def test_copy_terms_are_different_objects(self) -> None:
        terms = [_make_leaf("u")]
        original = PDE(terms=terms)
        copied = original.copy()
        assert copied.terms[0] is not original.terms[0]

    def test_copy_list_is_different_object(self) -> None:
        terms = [_make_leaf("u")]
        original = PDE(terms=terms)
        copied = original.copy()
        copied.terms.append(_make_leaf("x"))
        assert original.width == 1







class TestPDEStr:

    def test_str_is_nonempty(self) -> None:
        pde = PDE(terms=[_make_leaf("u"), _make_mul_tree("u", "x")])
        s = str(pde)
        assert len(s) > 0

    def test_str_contains_term_info(self) -> None:
        pde = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        s = str(pde)

        assert "u" in s
        assert "x" in s

    def test_str_empty_pde(self) -> None:
        pde = PDE(terms=[])
        s = str(pde)
        assert isinstance(s, str)







class TestPDENegative:

    def test_width_after_appending_term(self) -> None:
        pde = PDE(terms=[_make_leaf("u")])
        assert pde.width == 1
        pde.terms.append(_make_leaf("x"))
        assert pde.width == 2

    def test_copy_of_empty_pde(self) -> None:
        pde = PDE(terms=[])
        copied = pde.copy()
        assert copied.width == 0
        assert copied.terms == []
