from __future__ import annotations

import copy
import logging
import os
from inspect import isfunction
import pdb

import numpy as np

from .tree import *
from .PDE_find import Train

import warnings
warnings.filterwarnings('ignore')


_LOGGER = logging.getLogger(__name__)
_DIVISION_STRATEGY_ENV = "KD_SGA_DIVIDE_MODE"
_DIVISION_STRATEGY_GUARD = "guard"
_DIVISION_STRATEGY_LEGACY = "legacy"
_DIVISION_STRATEGIES = {_DIVISION_STRATEGY_GUARD, _DIVISION_STRATEGY_LEGACY}
_DIVISION_ZERO_ATOL = 1e-10
_LEGACY_DIVISION_EPS = 1e-6
_DIVISION_STRATEGY_OVERRIDE: str | None = None


def _resolve_division_strategy() -> str:
    if _DIVISION_STRATEGY_OVERRIDE is not None:
        return _DIVISION_STRATEGY_OVERRIDE
    raw = os.getenv(_DIVISION_STRATEGY_ENV, _DIVISION_STRATEGY_LEGACY)
    if not raw:
        return _DIVISION_STRATEGY_LEGACY
    normalized = raw.strip().lower()
    if normalized not in _DIVISION_STRATEGIES:
        _LOGGER.debug(
            "Unknown KD_SGA division strategy '%s'; falling back to '%s'",
            normalized,
            _DIVISION_STRATEGY_LEGACY,
        )
        return _DIVISION_STRATEGY_LEGACY
    return normalized


def _set_division_strategy_for_tests(strategy: str | None) -> None:
    """Internal helper allowing tests to override division behaviour."""

    global _DIVISION_STRATEGY_OVERRIDE  # noqa: PLW0603
    if strategy is None:
        _DIVISION_STRATEGY_OVERRIDE = None
        return
    normalized = strategy.strip().lower()
    if normalized not in _DIVISION_STRATEGIES:
        raise ValueError(f"Unknown division strategy: {strategy}")
    _DIVISION_STRATEGY_OVERRIDE = normalized


class PDE:
    def __init__(self, context, depth, max_width, p_var):
        self.context = context
        self.depth = depth
        self.p_var = p_var
        self.W = np.random.randint(max_width)+1  # 1 -- width
        self.elements = []
        for i in range(0, self.W):
            # 产生W个tree，也就是W个项
            one_tree = Tree(context, depth, p_var)
            # while 'd u t' in tree.preorder:# 没用，挡不住如(sin x + u) d t；不如直接看mse，太小就扔掉
            #     tree = Tree(depth, p_var)
            self.elements.append(one_tree)

    def mutate(self, p_mute):
        for i in range(0, self.W):  # 0 -- W-1
            self.elements[i].mutate(self.context, p_mute)

    def replace(self): # 直接产生一个新的tree，替换pde中的一项
        # print('replace!')
        one_tree = Tree(self.context, self.depth, self.p_var)
        ix = np.random.randint(self.W)  # 0 -- W-1
        if len(self.elements) == 0:
            NotImplementedError('replace error')
        self.elements[ix] = one_tree

    def visualize(self): # 写出SGA产生的项的形式，包含产生的所有项，未去除系数小的项。
        name = ''
        for i in range(len(self.elements)):
            if i != 0:
                name += '+'
            name += self.elements[i].inorder
        return name

    def concise_visualize(self, context): # 写出所有项的形式，包含固定候选集和SGA，且包含系数。会区分是来自于固定候选集的还是来自于SGA生成的候选集的。如果是来自于SGA生成的候选集，需要用inorder来写出可理解的项。
        name = ''
        elements = copy.deepcopy(self.elements)
        elements, coefficients, _ = evaluate_mse(elements, context, True, return_matrix=True)
        coefficients = coefficients[:, 0]
        # print(len(elements), len(coefficients))
        for i in range(len(coefficients)):
            if np.abs(coefficients[i]) < 1e-4: # 忽略过于小的系数
                continue
            if i != 0 and name != '':
                name += ' + '
            name += str(round(np.real(coefficients[i]), 4))
            if i < context.num_default: # num_default中为一定包含的候选集
                name += context.default_names[i]
            else:
                name += elements[i-context.num_default].inorder # element是SGA生成的候选集
        return name

@profile
def evaluate_mse(a_pde, context, is_term=False, *, return_matrix=False):
    if is_term:
        terms = a_pde
    else:
        terms = a_pde.elements
    # NOTE(ND): LHS size defines the feature matrix rows for N-dim alignment.
    lhs = context.ut
    flat_size = lhs.size
    terms_values = np.zeros((flat_size, len(terms)))
    delete_ix = []
    feature_matrix = None
    division_strategy = _resolve_division_strategy()
    # NOTE(ND): Track discard reasons separately for clearer debugging.
    discarded_lhs_terms = 0
    discarded_division_terms = 0

    for ix, term in enumerate(terms):
        tree_list = term.tree
        max_depth = len(tree_list)
        term_invalid = False

        # 先搜索倒数第二层，逐层向上对数据进行运算直到顶部；排除底部空层
        for i in range(2, max_depth + 1):
            if len(tree_list[-i + 1]) == 0:
                continue

            for j in range(len(tree_list[-i])):
                if term_invalid:
                    break

                node = tree_list[-i][j]

                if node.child_num == 0:
                    continue

                if node.child_num == 1:
                    child_node = tree_list[-i + 1][node.child_st]
                    node.cache = node.cache(child_node.cache)
                    child_node.cache = child_node.var  # 重置
                    continue

                if node.child_num == 2:
                    child1 = tree_list[-i + 1][node.child_st]
                    child2 = tree_list[-i + 1][node.child_st + 1]

                    if node.name in {'d', 'd^2'}:
                        axis_name = child2.name
                        # 设计约束（Strict LHS / RHS 禁止对 lhs_axis 求导）：
                        # - 默认任务是发现 `u_t = RHS(...)`，因此 RHS 中一旦出现 `d(·, t)` 或 `d^2(·, t)`，
                        #   很容易产生退化解（例如 ut=ut）或违反论文形式。
                        # - 生成层（tree.den 过滤）会尽量避免抽到 lhs_axis，但这不是“全链路保证”：
                        #   外部构造的 tree、旧缓存、或未来 den 扩展到 N 维时都可能绕过生成层。
                        # 因此在评估层做 fail-fast：检测到 RHS 沿 lhs_axis 求导直接丢弃该 term。
                        if axis_name == getattr(context, "lhs_axis", "t"):
                            term_invalid = True
                            discarded_lhs_terms += 1
                            child1.cache = child1.var
                            child2.cache = child2.var
                            break
                        if axis_name not in context.axis_map:
                            raise NotImplementedError(
                                f"Unknown axis '{axis_name}' for derivative."
                            )
                        axis_index = context.axis_map[axis_name]
                        tmp = context.delta.get(axis_name)
                        if tmp is None:
                            raise NotImplementedError(
                                f"Missing delta for axis '{axis_name}'."
                            )

                        if not isfunction(node.cache):
                            pdb.set_trace()
                            node.cache = node.var

                        node.cache = node.cache(child1.cache, tmp, axis_index)

                    else:
                        if isfunction(child1.cache) or isfunction(child2.cache):
                            pdb.set_trace()

                        op_name = getattr(node, 'name', None)
                        if op_name == '/':
                            divisor = child2.cache
                            if np.allclose(divisor, 0.0, atol=_DIVISION_ZERO_ATOL):
                                if division_strategy == _DIVISION_STRATEGY_GUARD:
                                    term_invalid = True
                                    discarded_division_terms += 1
                                    child1.cache = child1.var
                                    child2.cache = child2.var
                                    break
                                if division_strategy == _DIVISION_STRATEGY_LEGACY:
                                    divisor = divisor + _LEGACY_DIVISION_EPS
                                    _LOGGER.debug(
                                        "Applying legacy zero-division perturbation (eps=%s)",
                                        _LEGACY_DIVISION_EPS,
                                    )
                            node.cache = node.cache(child1.cache, divisor)
                        else:
                            node.cache = node.cache(child1.cache, child2.cache)

                    child1.cache, child2.cache = child1.var, child2.var  # 重置
                    continue

                NotImplementedError()

            if term_invalid:
                break

        if term_invalid:
            delete_ix.append(ix)
            tree_list[0][0].cache = tree_list[0][0].var
            continue

        if not any(tree_list[0][0].cache.reshape(-1)):  # 如果全是0，无法收敛且无意义
            delete_ix.append(ix)
            tree_list[0][0].cache = tree_list[0][0].var  # 重置缓冲池
        else:
            terms_values[:, ix:ix + 1] = tree_list[0][0].cache.reshape(-1, 1)  # 把归并起来的该term记录下来
            tree_list[0][0].cache = tree_list[0][0].var  # 重置缓冲池

    move = 0
    for ixx in delete_ix:
        if is_term:
            terms.pop(ixx - move)
        else:
            a_pde.elements.pop(ixx - move)
            a_pde.W -= 1  # 实际宽度减一
        terms_values = np.delete(terms_values, ixx - move, axis=1)
        move += 1  # pop以后index左移

    if discarded_lhs_terms:
        _LOGGER.debug(
            "Discarded %d term(s) due to rhs derivative along lhs_axis",
            discarded_lhs_terms,
        )
    if discarded_division_terms and division_strategy == _DIVISION_STRATEGY_GUARD:
        _LOGGER.debug(
            "Discarded %d term(s) due to division guard",
            discarded_division_terms,
        )

    # 检查是否存在inf或者nan，或者terms_values是否被削没了
    if False in np.isfinite(terms_values) or terms_values.shape[1] == 0:
        error = np.inf
        aic = np.inf
        w = 0

    else:
        # 2D --> 1D
        if context.default_terms.shape[0] != flat_size:
            # NOTE(ND): Prevent silent misalignment between default terms and LHS flattening.
            raise ValueError(
                "default_terms row count does not match lhs size; "
                "check grid alignment and flatten order."
            )
        terms_values = np.hstack((context.default_terms, terms_values))
        w, loss, mse, aic = Train(
            terms_values,
            lhs.reshape(-1, 1),
            0,
            1,
            context.config.aic_ratio,
        )
        feature_matrix = terms_values

    if is_term:
        if return_matrix:
            return terms, w, feature_matrix
        return terms, w
    if return_matrix:
        return aic, w, feature_matrix
    return aic, w


if __name__ == '__main__':
    pde = PDE(depth=4, max_width=3, p_var=0.5)
    evaluate_mse(pde)
    pde.mutate(p_mute=0.1)
    pde.replace()
