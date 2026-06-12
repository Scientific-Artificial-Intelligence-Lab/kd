
from __future__ import annotations

import torch

from kd.search.sga.config import OperatorPool, SGAConfig
from kd.search.sga.pde import PDE
from kd.search.sga.tree import Node, Tree






def _rand_choice(pool_size: int, rng: torch.Generator) -> int:
    return int(torch.randint(pool_size, (1,), generator=rng).item())


def _rand_float(rng: torch.Generator) -> float:
    return float(torch.rand(1, generator=rng).item())







def _random_node(
    pool: OperatorPool,
    rng: torch.Generator,
) -> Node:
    name, arity = pool[_rand_choice(len(pool), rng)]
    return Node(name=name, arity=arity, children=[])


def _is_derivative(name: str) -> bool:
    return name in {"d", "d^2"}


def _child_pool(
    parent: Node,
    child_idx: int,
    var_pool: OperatorPool,
    den: OperatorPool,
) -> OperatorPool:
    if _is_derivative(parent.name) and child_idx == 1:
        return den
    return var_pool


def _sample_leaf(
    pool: OperatorPool,
    old_name: str,
    rng: torch.Generator,
) -> Node:
    new_node = _random_node(pool, rng)
    for _ in range(_MAX_MUTATE_RETRIES):
        if new_node.name != old_name or len(pool) <= 1:
            break
        new_node = _random_node(pool, rng)
    return new_node


def _sample_operator(
    pool: OperatorPool,
    old_name: str,
    old_arity: int,
    rng: torch.Generator,
    right_child: Node | None = None,
    den_names: set[str] | None = None,
) -> tuple[str, int]:
    new_name, new_arity = pool[_rand_choice(len(pool), rng)]
    for _ in range(_MAX_MUTATE_RETRIES):
        if new_name == old_name and len(pool) > 1:
            new_name, new_arity = pool[_rand_choice(len(pool), rng)]
            continue
        if _operator_is_valid(new_name, right_child, den_names):
            return new_name, new_arity
        new_name, new_arity = pool[_rand_choice(len(pool), rng)]
    return old_name, old_arity


def _operator_is_valid(
    name: str,
    right_child: Node | None,
    den_names: set[str] | None,
) -> bool:
    if not _is_derivative(name):
        return True
    if right_child is None or den_names is None:
        return False
    return right_child.is_leaf and right_child.name in den_names


def random_tree(
    vars: list[str],
    ops: OperatorPool,
    root: OperatorPool,
    den: OperatorPool,
    depth: int,
    p_var: float,
    rng: torch.Generator,
) -> Tree:

    var_pool: OperatorPool = tuple((v, 0) for v in vars)


    root_node = _random_node(root, rng)



    _fill_children(root_node, 1, depth, var_pool, ops, den, p_var, rng)

    return Tree(root=root_node)


def _fill_children(
    node: Node,
    current_depth: int,
    max_depth: int,
    var_pool: OperatorPool,
    ops: OperatorPool,
    den: OperatorPool,
    p_var: float,
    rng: torch.Generator,
) -> None:
    if node.arity == 0:
        return

    for j in range(node.arity):
        leaf_pool = _child_pool(node, j, var_pool, den)
        if (
            (_is_derivative(node.name) and j == 1)
            or current_depth >= max_depth - 1
            or _rand_float(rng) <= p_var
        ):
            child = _random_node(leaf_pool, rng)
        else:
            child = _random_node(ops, rng)

        node.children.append(child)
        _fill_children(
            child,
            current_depth + 1,
            max_depth,
            var_pool,
            ops,
            den,
            p_var,
            rng,
        )







def random_pde(
    config: SGAConfig,
    vars: list[str],
    ops: OperatorPool,
    root: OperatorPool,
    den: OperatorPool,
    rng: torch.Generator,
) -> PDE:

    width = int(torch.randint(1, config.width + 1, (1,), generator=rng).item())
    terms: list[Tree] = []
    for _ in range(width):
        tree = random_tree(vars, ops, root, den, config.depth, config.p_var, rng)
        terms.append(tree)
    return PDE(terms=terms)







def mutate(
    pde: PDE,
    vars: list[str],
    op1: OperatorPool,
    op2: OperatorPool,
    den: OperatorPool,
    p_mute: float,
    rng: torch.Generator,
) -> PDE:
    new_pde = pde.copy()
    var_pool: OperatorPool = tuple((v, 0) for v in vars)
    den_names = {name for name, _ in den}

    for tree in new_pde.terms:
        _mutate_subtree(tree.root, var_pool, op1, op2, den, den_names, p_mute, rng)

    return new_pde


_MAX_MUTATE_RETRIES: int = 10


def _mutate_subtree(
    node: Node,
    var_pool: OperatorPool,
    op1: OperatorPool,
    op2: OperatorPool,
    den: OperatorPool,
    den_names: set[str],
    p_mute: float,
    rng: torch.Generator,
) -> None:
    for i, child in enumerate(node.children):

        if _rand_float(rng) < p_mute:
            if child.arity == 0:
                pool = _child_pool(node, i, var_pool, den)
                node.children[i] = _sample_leaf(pool, child.name, rng)
            elif child.arity == 1:
                new_name, new_arity = _sample_operator(
                    op1,
                    child.name,
                    child.arity,
                    rng,
                )
                node.children[i] = Node(
                    name=new_name,
                    arity=new_arity,
                    children=child.children,
                )
            elif child.arity == 2:
                right_child = child.children[1] if len(child.children) == 2 else None
                new_name, new_arity = _sample_operator(
                    op2,
                    child.name,
                    child.arity,
                    rng,
                    right_child=right_child,
                    den_names=den_names,
                )
                node.children[i] = Node(
                    name=new_name,
                    arity=new_arity,
                    children=child.children,
                )
        _mutate_subtree(
            node.children[i],
            var_pool,
            op1,
            op2,
            den,
            den_names,
            p_mute,
            rng,
        )







def crossover(
    pde1: PDE,
    pde2: PDE,
    rng: torch.Generator,
) -> tuple[PDE, PDE]:
    new1 = pde1.copy()
    new2 = pde2.copy()

    if new1.width == 0 or new2.width == 0:
        return new1, new2

    idx1 = _rand_choice(new1.width, rng)
    idx2 = _rand_choice(new2.width, rng)


    new1.terms[idx1], new2.terms[idx2] = new2.terms[idx2], new1.terms[idx1]

    return new1, new2







def replace(
    pde: PDE,
    vars: list[str],
    ops: OperatorPool,
    root: OperatorPool,
    den: OperatorPool,
    depth: int,
    p_var: float,
    rng: torch.Generator,
) -> PDE:
    new_pde = pde.copy()

    if new_pde.width == 0:
        return new_pde

    idx = _rand_choice(new_pde.width, rng)
    new_tree = random_tree(vars, ops, root, den, depth, p_var, rng)
    new_pde.terms[idx] = new_tree

    return new_pde
