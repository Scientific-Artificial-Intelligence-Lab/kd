
from __future__ import annotations

import pytest
import torch

from kd.search.sga.config import OP1, OP2, OPS, ROOT, SGAConfig, build_den
from kd.search.sga.genetic import (
    crossover,
    mutate,
    random_pde,
    random_tree,
    replace,
)
from kd.search.sga.pde import PDE
from kd.search.sga.tree import Node, Tree





VARS = ["u", "x", "t", "u_x", "u_t"]

DEN = build_den(axes=["x", "t"], lhs_axis="t")


def _make_rng(seed: int = 42) -> torch.Generator:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def _make_leaf(name: str) -> Tree:
    return Tree(root=Node(name=name, arity=0, children=[]))


def _make_binary_tree(op: str, left: str, right: str) -> Tree:
    left_node = Node(name=left, arity=0, children=[])
    right_node = Node(name=right, arity=0, children=[])
    root = Node(name=op, arity=2, children=[left_node, right_node])
    return Tree(root=root)


def _make_unary_tree(op: str, child: str) -> Tree:
    child_node = Node(name=child, arity=0, children=[])
    root = Node(name=op, arity=1, children=[child_node])
    return Tree(root=root)


def _collect_nodes(node: Node) -> list[Node]:
    result = [node]
    for child in node.children:
        result.extend(_collect_nodes(child))
    return result


def _collect_leaves(node: Node) -> list[Node]:
    return [n for n in _collect_nodes(node) if n.is_leaf]


def _collect_arity_map(node: Node) -> list[tuple[str, int]]:
    return [(n.name, n.arity) for n in _collect_nodes(node)]







class TestRandomTree:

    @pytest.mark.smoke
    def test_returns_tree_instance(self) -> None:
        rng = _make_rng()
        tree = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        assert isinstance(tree, Tree)

    def test_root_from_root_pool(self) -> None:
        root_names = {name for name, _ in ROOT}
        rng = _make_rng()
        for seed in range(50):
            rng.manual_seed(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=3,
                p_var=0.5,
                rng=rng,
            )
            assert tree.root.name in root_names, (
                f"Root '{tree.root.name}' not in ROOT pool {root_names}"
            )

    def test_leaves_are_variables_or_den(self) -> None:
        var_set = set(VARS)
        den_set = {name for name, _ in DEN}
        allowed = var_set | den_set
        rng = _make_rng()
        for seed in range(50):
            rng.manual_seed(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.5,
                rng=rng,
            )
            leaves = _collect_leaves(tree.root)
            assert len(leaves) > 0, "Tree must have at least one leaf"
            for leaf in leaves:
                assert leaf.name in allowed, (
                    f"Leaf '{leaf.name}' not in vars|den {allowed}"
                )

    def test_depth_respects_constraint(self) -> None:
        rng = _make_rng()
        max_depth = 3
        for seed in range(50):
            rng.manual_seed(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=max_depth,
                p_var=0.5,
                rng=rng,
            )
            assert tree.depth <= max_depth, (
                f"Tree depth {tree.depth} exceeds max {max_depth}"
            )

    def test_leaf_arity_is_zero(self) -> None:
        rng = _make_rng()
        tree = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=4,
            p_var=0.3,
            rng=rng,
        )
        for leaf in _collect_leaves(tree.root):
            assert leaf.arity == 0

    def test_operator_nodes_have_correct_children_count(self) -> None:
        rng = _make_rng()
        for seed in range(30):
            rng.manual_seed(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.5,
                rng=rng,
            )
            for node in _collect_nodes(tree.root):
                if not node.is_leaf:
                    assert len(node.children) == node.arity, (
                        f"Node '{node.name}' has arity {node.arity} "
                        f"but {len(node.children)} children"
                    )


class TestRandomTreeReproducibility:

    def test_same_seed_same_tree(self) -> None:
        seed = 123
        rng1 = _make_rng(seed)
        rng2 = _make_rng(seed)
        tree1 = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=4,
            p_var=0.5,
            rng=rng1,
        )
        tree2 = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=4,
            p_var=0.5,
            rng=rng2,
        )
        assert tree1 == tree2

    def test_different_seed_likely_different(self) -> None:
        trees = []
        for seed in range(10):
            rng = _make_rng(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.5,
                rng=rng,
            )
            trees.append(str(tree))

        assert len(set(trees)) > 1


class TestRandomTreeEdgeCases:

    def test_depth_one_is_just_root_with_leaf_children(self) -> None:
        rng = _make_rng()
        var_set = set(VARS)
        den_set = {name for name, _ in DEN}
        allowed = var_set | den_set
        tree = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=1,
            p_var=0.5,
            rng=rng,
        )

        assert tree.root.arity > 0

        for child in tree.root.children:
            assert child.is_leaf
            assert child.name in allowed

    def test_p_var_one_produces_shallow_trees(self) -> None:
        rng = _make_rng()
        tree = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=4,
            p_var=1.0,
            rng=rng,
        )


        assert tree.depth <= 1

    def test_p_var_zero_produces_deeper_trees(self) -> None:
        rng = _make_rng()
        tree = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=4,
            p_var=0.0,
            rng=rng,
        )

        assert tree.depth > 1

    def test_single_var_pool(self) -> None:
        rng = _make_rng()
        den_names = {name for name, _ in DEN}
        tree = random_tree(
            vars=["u"],
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        for leaf in _collect_leaves(tree.root):
            assert leaf.name == "u" or leaf.name in den_names







class TestRandomPDE:

    @pytest.mark.smoke
    def test_returns_pde_instance(self) -> None:
        config = SGAConfig(width=5, depth=4, p_var=0.5)
        rng = _make_rng()
        pde = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng,
        )
        assert isinstance(pde, PDE)

    def test_width_in_valid_range(self) -> None:
        config = SGAConfig(width=5, depth=4, p_var=0.5)
        for seed in range(50):
            rng = _make_rng(seed)
            pde = random_pde(
                config=config,
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                rng=rng,
            )
            assert 1 <= pde.width <= config.width, (
                f"PDE width {pde.width} not in [1, {config.width}]"
            )

    def test_all_terms_are_valid_trees(self) -> None:
        config = SGAConfig(width=5, depth=4, p_var=0.5)
        rng = _make_rng()
        pde = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng,
        )
        for term in pde.terms:
            assert isinstance(term, Tree)

    def test_terms_respect_depth_constraint(self) -> None:
        config = SGAConfig(width=5, depth=3, p_var=0.5)
        rng = _make_rng()
        pde = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng,
        )
        for term in pde.terms:
            assert term.depth <= config.depth

    def test_reproducible(self) -> None:
        config = SGAConfig(width=5, depth=4, p_var=0.5)
        rng1 = _make_rng(99)
        rng2 = _make_rng(99)
        pde1 = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng1,
        )
        pde2 = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng2,
        )
        assert pde1 == pde2







class TestMutate:

    @pytest.mark.smoke
    def test_returns_pde_instance(self) -> None:
        pde = PDE(terms=[_make_binary_tree("*", "u", "x")])
        rng = _make_rng()
        result = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=0.5,
            rng=rng,
        )
        assert isinstance(result, PDE)

    def test_does_not_modify_input(self) -> None:
        pde = PDE(terms=[_make_binary_tree("*", "u", "x")])
        original_str = str(pde)
        rng = _make_rng()
        _ = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=1.0,
            rng=rng,
        )
        assert str(pde) == original_str, "Input PDE was modified by mutate"

    def test_returns_new_object(self) -> None:
        pde = PDE(terms=[_make_binary_tree("*", "u", "x")])
        rng = _make_rng()
        result = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=0.5,
            rng=rng,
        )
        assert result is not pde

    def test_arity_preserved_after_mutation(self) -> None:
        tree = _make_binary_tree("*", "u", "x")
        pde = PDE(terms=[tree])

        original_arities = [n.arity for n in _collect_nodes(pde.terms[0].root)]

        rng = _make_rng()
        result = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=1.0,
            rng=rng,
        )

        result_arities = [n.arity for n in _collect_nodes(result.terms[0].root)]
        assert original_arities == result_arities, (
            "Mutation changed tree structure (arities differ)"
        )

    def test_high_p_mute_changes_something(self) -> None:
        tree = _make_binary_tree("*", "u", "x")
        pde = PDE(terms=[tree])

        changed = False
        for seed in range(20):
            rng = _make_rng(seed)
            result = mutate(
                pde=pde,
                vars=VARS,
                op1=OP1,
                op2=OP2,
                den=DEN,
                p_mute=1.0,
                rng=rng,
            )
            if str(result) != str(pde):
                changed = True
                break
        assert changed, "p_mute=1.0 never changed anything over 20 seeds"

    def test_zero_p_mute_preserves_tree(self) -> None:
        tree = _make_binary_tree("*", "u", "x")
        pde = PDE(terms=[tree])
        rng = _make_rng()
        result = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=0.0,
            rng=rng,
        )
        assert result == pde, "p_mute=0.0 should produce identical PDE"

    def test_leaves_remain_valid(self) -> None:
        var_set = set(VARS)
        den_set = {name for name, _ in DEN}
        allowed = var_set | den_set
        tree = _make_binary_tree("*", "u", "x")
        pde = PDE(terms=[tree])

        for seed in range(20):
            rng = _make_rng(seed)
            result = mutate(
                pde=pde,
                vars=VARS,
                op1=OP1,
                op2=OP2,
                den=DEN,
                p_mute=0.8,
                rng=rng,
            )
            for term in result.terms:
                for leaf in _collect_leaves(term.root):
                    assert leaf.name in allowed

    def test_operators_remain_from_pools(self) -> None:
        op1_names = {name for name, _ in OP1}
        op2_names = {name for name, _ in OP2}

        tree = _make_binary_tree("*", "u", "x")
        pde = PDE(terms=[tree])

        for seed in range(20):
            rng = _make_rng(seed)
            result = mutate(
                pde=pde,
                vars=VARS,
                op1=OP1,
                op2=OP2,
                den=DEN,
                p_mute=1.0,
                rng=rng,
            )
            for term in result.terms:
                for node in _collect_nodes(term.root):
                    if node.arity == 1:
                        assert node.name in op1_names
                    elif node.arity == 2:
                        assert node.name in op2_names

    def test_width_preserved(self) -> None:
        terms = [
            _make_binary_tree("*", "u", "x"),
            _make_unary_tree("^2", "u"),
        ]
        pde = PDE(terms=terms)
        rng = _make_rng()
        result = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=0.5,
            rng=rng,
        )
        assert result.width == pde.width

    def test_no_reference_sharing_with_input(self) -> None:
        tree = _make_binary_tree("*", "u", "x")
        pde = PDE(terms=[tree])
        rng = _make_rng()
        result = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=0.0,
            rng=rng,
        )

        assert result.terms[0].root is not pde.terms[0].root







class TestCrossover:

    @pytest.mark.smoke
    def test_returns_two_pdes(self) -> None:
        pde1 = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        pde2 = PDE(terms=[_make_leaf("t"), _make_leaf("u_x")])
        rng = _make_rng()
        r1, r2 = crossover(pde1, pde2, rng)
        assert isinstance(r1, PDE)
        assert isinstance(r2, PDE)

    def test_does_not_modify_inputs(self) -> None:
        pde1 = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        pde2 = PDE(terms=[_make_leaf("t"), _make_leaf("u_x")])
        str1, str2 = str(pde1), str(pde2)
        rng = _make_rng()
        _ = crossover(pde1, pde2, rng)
        assert str(pde1) == str1, "pde1 was modified by crossover"
        assert str(pde2) == str2, "pde2 was modified by crossover"

    def test_returns_new_objects(self) -> None:
        pde1 = PDE(terms=[_make_leaf("u")])
        pde2 = PDE(terms=[_make_leaf("x")])
        rng = _make_rng()
        r1, r2 = crossover(pde1, pde2, rng)
        assert r1 is not pde1
        assert r2 is not pde2

    def test_width_preserved(self) -> None:
        pde1 = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        pde2 = PDE(terms=[_make_leaf("t"), _make_leaf("u_x"), _make_leaf("u_t")])
        rng = _make_rng()
        r1, r2 = crossover(pde1, pde2, rng)
        assert r1.width == pde1.width
        assert r2.width == pde2.width

    def test_swaps_entire_terms(self) -> None:

        pde1 = PDE(terms=[_make_binary_tree("*", "u", "x"), _make_leaf("t")])
        pde2 = PDE(terms=[_make_unary_tree("^2", "u_x"), _make_leaf("u_t")])


        found_swap = False
        pde1_term_strs = {str(t) for t in pde1.terms}
        pde2_term_strs = {str(t) for t in pde2.terms}

        for seed in range(30):
            rng = _make_rng(seed)
            r1, r2 = crossover(pde1, pde2, rng)
            r1_term_strs = {str(t) for t in r1.terms}
            r2_term_strs = {str(t) for t in r2.terms}


            if (r1_term_strs & pde2_term_strs) or (r2_term_strs & pde1_term_strs):
                found_swap = True
                break

        assert found_swap, "Crossover never swapped terms over 30 seeds"

    def test_no_reference_sharing_between_outputs_and_inputs(self) -> None:
        pde1 = PDE(terms=[_make_binary_tree("*", "u", "x")])
        pde2 = PDE(terms=[_make_binary_tree("/", "t", "u_x")])
        rng = _make_rng()
        r1, r2 = crossover(pde1, pde2, rng)


        r1.terms[0].root.name = "CORRUPTED"


        assert pde1.terms[0].root.name != "CORRUPTED"
        assert pde2.terms[0].root.name != "CORRUPTED"

    def test_reproducible(self) -> None:
        pde1 = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        pde2 = PDE(terms=[_make_leaf("t"), _make_leaf("u_x")])

        rng_a = _make_rng(77)
        rng_b = _make_rng(77)
        r1a, r2a = crossover(pde1, pde2, rng_a)
        r1b, r2b = crossover(pde1, pde2, rng_b)
        assert r1a == r1b
        assert r2a == r2b







class TestReplace:

    @pytest.mark.smoke
    def test_returns_pde_instance(self) -> None:
        pde = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        rng = _make_rng()
        result = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        assert isinstance(result, PDE)

    def test_does_not_modify_input(self) -> None:
        pde = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        original_str = str(pde)
        rng = _make_rng()
        _ = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        assert str(pde) == original_str

    def test_returns_new_object(self) -> None:
        pde = PDE(terms=[_make_leaf("u")])
        rng = _make_rng()
        result = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        assert result is not pde

    def test_width_preserved(self) -> None:
        pde = PDE(terms=[_make_leaf("u"), _make_leaf("x"), _make_leaf("t")])
        rng = _make_rng()
        result = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        assert result.width == pde.width

    def test_exactly_one_term_differs(self) -> None:
        pde = PDE(
            terms=[
                _make_binary_tree("*", "u", "x"),
                _make_unary_tree("^2", "t"),
                _make_binary_tree("/", "u_x", "u_t"),
            ]
        )

        found_single_diff = False
        for seed in range(30):
            rng = _make_rng(seed)
            result = replace(
                pde=pde,
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.5,
                rng=rng,
            )
            pairs = zip(pde.terms, result.terms, strict=True)
            diffs = sum(1 for a, b in pairs if str(a) != str(b))
            if diffs == 1:
                found_single_diff = True
                break

        assert found_single_diff, (
            "replace never produced exactly 1 different term over 30 seeds"
        )

    def test_new_term_is_valid_tree(self) -> None:
        var_set = set(VARS)
        den_set = {name for name, _ in DEN}
        root_names = {name for name, _ in ROOT}
        allowed_leaves = var_set | den_set
        pde = PDE(terms=[_make_leaf("u")])
        rng = _make_rng()
        result = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )

        term = result.terms[0]
        assert isinstance(term, Tree)
        assert term.root.name in root_names | allowed_leaves
        for leaf in _collect_leaves(term.root):
            assert leaf.name in allowed_leaves

    def test_reproducible(self) -> None:
        pde = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        rng_a = _make_rng(55)
        rng_b = _make_rng(55)
        r1 = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng_a,
        )
        r2 = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng_b,
        )
        assert r1 == r2

    def test_no_reference_sharing(self) -> None:
        pde = PDE(terms=[_make_binary_tree("*", "u", "x")])
        rng = _make_rng()
        result = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        result.terms[0].root.name = "CORRUPTED"
        assert pde.terms[0].root.name == "*"







class TestGeneticOperatorInvariants:

    def test_mutate_idempotent_width(self) -> None:
        pde = PDE(terms=[_make_binary_tree("*", "u", "x"), _make_leaf("t")])
        rng = _make_rng()
        current = pde
        for _ in range(5):
            current = mutate(
                pde=current,
                vars=VARS,
                op1=OP1,
                op2=OP2,
                den=DEN,
                p_mute=0.5,
                rng=rng,
            )
            assert current.width == pde.width

    def test_all_operators_produce_finite_trees(self) -> None:
        config = SGAConfig(width=5, depth=4, p_var=0.5)
        rng = _make_rng()

        pde = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng,
        )
        for term in pde.terms:
            assert term.depth < 100, "Suspiciously deep tree (possible cycle)"
            assert term.size < 10000, "Suspiciously large tree"

    def test_chained_operations_produce_valid_pdes(self) -> None:
        config = SGAConfig(width=3, depth=3, p_var=0.5)
        rng = _make_rng(0)

        pde1 = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng,
        )
        pde2 = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng,
        )


        pde1 = mutate(
            pde=pde1,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=0.3,
            rng=rng,
        )
        pde1 = replace(
            pde=pde1,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        pde1, pde2 = crossover(pde1, pde2, rng)


        var_set = set(VARS)
        den_set = {name for name, _ in DEN}
        allowed = var_set | den_set
        for pde in (pde1, pde2):
            assert isinstance(pde, PDE)
            assert pde.width >= 1
            for term in pde.terms:
                assert isinstance(term, Tree)
                for leaf in _collect_leaves(term.root):
                    assert leaf.name in allowed







def _find_derivative_nodes(node: Node) -> list[tuple[Node, Node, Node]]:
    results: list[tuple[Node, Node, Node]] = []
    if node.name in {"d", "d^2"} and len(node.children) == 2:
        results.append((node, node.children[0], node.children[1]))
    for child in node.children:
        results.extend(_find_derivative_nodes(child))
    return results


class TestDerivativeNodeGeneration:

    def test_derivative_nodes_can_be_generated(self) -> None:
        found_deriv = False
        for seed in range(200):
            rng = _make_rng(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.3,
                rng=rng,
            )
            derivs = _find_derivative_nodes(tree.root)
            if derivs:
                found_deriv = True
                break
        assert found_deriv, "No derivative nodes generated over 200 seeds"

    def test_derivative_right_child_from_den(self) -> None:
        den_names = {name for name, _ in DEN}
        for seed in range(200):
            rng = _make_rng(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.3,
                rng=rng,
            )
            for _, _, right in _find_derivative_nodes(tree.root):
                assert right.name in den_names, (
                    f"Derivative right child '{right.name}' not in den {den_names}"
                )
                assert right.is_leaf, "Derivative right child must be a leaf"

    def test_derivative_right_child_never_lhs_axis(self) -> None:
        lhs_axis = "t"
        for seed in range(200):
            rng = _make_rng(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.3,
                rng=rng,
            )
            for _, _, right in _find_derivative_nodes(tree.root):
                assert right.name != lhs_axis, (
                    f"Derivative denominator is lhs_axis '{lhs_axis}'"
                )


class TestDerivativeNodeMutation:

    def test_mutated_derivative_right_child_stays_in_den(self) -> None:
        den_names = {name for name, _ in DEN}

        d_node = Node(
            name="d",
            arity=2,
            children=[
                Node(name="u", arity=0),
                Node(name="x", arity=0),
            ],
        )
        tree = Tree(root=d_node)
        pde = PDE(terms=[tree])

        for seed in range(50):
            rng = _make_rng(seed)
            result = mutate(
                pde=pde,
                vars=VARS,
                op1=OP1,
                op2=OP2,
                den=DEN,
                p_mute=1.0,
                rng=rng,
            )
            for term in result.terms:
                for _, _, right in _find_derivative_nodes(term.root):
                    assert right.name in den_names, (
                        f"After mutation, derivative right child "
                        f"'{right.name}' not in den"
                    )

    def test_mutation_to_d_checks_right_child_compatibility(self) -> None:
        den_names = {name for name, _ in DEN}

        tree = _make_binary_tree("*", "u", "u")
        pde = PDE(terms=[tree])

        for seed in range(50):
            rng = _make_rng(seed)
            result = mutate(
                pde=pde,
                vars=VARS,
                op1=OP1,
                op2=OP2,
                den=DEN,
                p_mute=1.0,
                rng=rng,
            )
            for term in result.terms:
                for _, _, right in _find_derivative_nodes(term.root):
                    assert right.name in den_names, (
                        f"Mutated-to-d node has invalid right child "
                        f"'{right.name}' not in den"
                    )
