from inspect import isfunction
import pdb

from .tree import *
from .PDE_find import Train

import warnings
warnings.filterwarnings('ignore')


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
        elements, coefficients = evaluate_mse(elements, context, True)
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


def evaluate_mse(a_pde, context, is_term=False):
    if is_term:
        terms = a_pde
    else:
        terms = a_pde.elements
    terms_values = np.zeros((context.u.shape[0] * context.u.shape[1], len(terms)))
    delete_ix = []
    for ix, term in enumerate(terms):
        tree_list = term.tree
        max_depth = len(tree_list)

        # 先搜索倒数第二层，逐层向上对数据进行运算直到顶部；排除底部空层
        for i in range(2, max_depth+1):
            # 如果下面一层是空的，说明这一层肯定不是非空的倒数第二层
            if len(tree_list[-i+1]) == 0:
                continue
            else: # 这一层是非空至少倒数第二层，一个一个结点看过去
                for j in range(len(tree_list[-i])):
                    # 如果这一结点没有孩子，继续看右边的结点有没有
                    if tree_list[-i][j].child_num == 0:
                        continue

                    # 这一结点有一个孩子，用自己的运算符对孩子的cache进行操作
                    elif tree_list[-i][j].child_num == 1:
                        child_node = tree_list[-i+1][tree_list[-i][j].child_st]
                        tree_list[-i][j].cache = tree_list[-i][j].cache(child_node.cache)
                        child_node.cache = child_node.var  # 重置

                    # 这一结点有一两个孩子，用自己的运算符对两孩子的cache进行操作
                    elif tree_list[-i][j].child_num == 2:
                        child1 = tree_list[-i+1][tree_list[-i][j].child_st]
                        child2 = tree_list[-i+1][tree_list[-i][j].child_st+1]

                        if tree_list[-i][j].name in {'d', 'd^2'}:
                            what_is_denominator = child2.name
                            if what_is_denominator == 't':
                                tmp = context.dt
                            elif what_is_denominator == 'x':
                                tmp = context.dx
                            else:
                                raise NotImplementedError()

                            if not isfunction(tree_list[-i][j].cache):
                                pdb.set_trace()
                                tree_list[-i][j].cache = tree_list[-i][j].var

                            tree_list[-i][j].cache = tree_list[-i][j].cache(child1.cache, tmp, what_is_denominator)

                        else:
                            if isfunction(child1.cache) or isfunction(child2.cache):
                                pdb.set_trace()
                            tree_list[-i][j].cache = tree_list[-i][j].cache(child1.cache, child2.cache)
                        child1.cache, child2.cache = child1.var, child2.var  # 重置

                    else:
                        NotImplementedError()

        if not any(tree_list[0][0].cache.reshape(-1)):  # 如果全是0，无法收敛且无意义
            delete_ix.append(ix)
            tree_list[0][0].cache = tree_list[0][0].var  # 重置缓冲池
            # print('0')
            # pdb.set_trace()
        else:
            terms_values[:, ix:ix+1] = tree_list[0][0].cache.reshape(-1, 1)  # 把归并起来的该term记录下来
            tree_list[0][0].cache = tree_list[0][0].var  # 重置缓冲池
            # print('not 0')
            # pdb.set_trace()

    move = 0
    for ixx in delete_ix:
        if is_term:
            terms.pop(ixx - move)
        else:
            a_pde.elements.pop(ixx-move)
            a_pde.W -= 1  # 实际宽度减一
        terms_values = np.delete(terms_values, ixx-move, axis=1)
        move += 1  # pop以后index左移

    # 检查是否存在inf或者nan，或者terms_values是否被削没了
    if False in np.isfinite(terms_values) or terms_values.shape[1] == 0:
        # pdb.set_trace()
        error = np.inf
        aic = np.inf
        w = 0

    else:
        # 2D --> 1D
        terms_values = np.hstack((context.default_terms, terms_values))
        w, loss, mse, aic = Train(terms_values, context.ut.reshape(context.n * context.m, 1), 0, 1, context.config.aic_ratio)

    if is_term:
        return terms, w
    else:
        return aic, w


if __name__ == '__main__':
    pde = PDE(depth=4, max_width=3, p_var=0.5)
    evaluate_mse(pde)
    pde.mutate(p_mute=0.1)
    pde.replace()

