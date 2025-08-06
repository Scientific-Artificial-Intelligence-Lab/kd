import numpy as np

# 从原始项目中导入我们需要的具体实现
# 如果未来路径有变，我们再调整这里的导入
from ..codes.pde import divide
from ..codes.PDE_find import Diff, Diff2

def cubic(inputs):
    """一个简单的三次方函数。"""
    return np.power(inputs, 3)

def get_default_library(workspace):
    """
    构建并返回SGA算法所需的默认符号库。
    这个函数取代了原始 setup.py 中定义全局符号变量的部分。

    Args:
        workspace (dict): 一个包含所有计算好数据的字典，
                          由 prepare_workspace() 函数生成。

    Returns:
        dict: 一个包含所有符号列表的字典，例如：
              {'ROOT': [...], 'OPS': [...], 'VARS': [...]}
    """
    # 从工作空间中解包我们需要的数据
    u = workspace['u']
    x = workspace['x']
    ux = workspace['ux']

    x_vec = workspace['x']
    n_x, n_t = u.shape
    # transpose((1,0)) 在旧版numpy中可能是 (m,1) -> (1,m)，再转置成 (m,1)
    # np.tile(x_vec, (n_t, 1)) -> 将 (301,) 向量复制200次，变成 (200, 301)
    # .T -> 转置成 (301, 200)
    x_grid = np.tile(x_vec, (n_t, 1)).T

    zeros = np.zeros_like(u)

    # ------------------------------------------------------------------
    # 这部分代码直接从原始 setup.py 中迁移而来，但现在它们被封装在
    # 一个函数内部，并且依赖于传入的 workspace 数据，而不是全局变量。
    # 我们也保留了之前修复过的 dtype=object。
    # ------------------------------------------------------------------

    # 保证所有变量 shape 为 (n,m)
    def ensure_2d(arr, n, m):
        arr = np.asarray(arr)
        if arr.shape == (n, m):
            return arr
        elif arr.shape == (n,):
            return arr.reshape(n, 1).repeat(m, axis=1)
        elif arr.shape == (m,):
            return arr.reshape(1, m).repeat(n, axis=0)
        elif arr.shape == (n*m,):
            return arr.reshape(n, m)
        elif arr.shape == (n*m, 1):
            return arr.reshape(n, m)
        else:
            return arr

    n, m = u.shape
    VARS = np.array([
        ['u', 0, ensure_2d(u, n, m)],
        ['x', 0, ensure_2d(x, n, m)],
        ['0', 0, ensure_2d(zeros, n, m)],
        ['ux', 0, ensure_2d(ux, n, m)]
    ], dtype=object)

    # 定义所有算子 (Operators)
    OP1 = np.array([ # 一元算子
        ['^2', 1, np.square], 
        ['^3', 1, cubic]
    ], dtype=object)
    
    OP2 = np.array([ # 二元算子
        ['+', 2, np.add], 
        ['-', 2, np.subtract], 
        ['*', 2, np.multiply], 
        ['/', 2, divide], 
        ['d', 2, Diff], 
        ['d^2', 2, Diff2]
    ], dtype=object)

    OPS = np.concatenate([OP1, OP2])

    # 定义可以作为树根节点的算子
    ROOT = np.array([
        ['*', 2, np.multiply], 
        ['d', 2, Diff], 
        ['d^2', 2, Diff2], 
        ['/', 2, divide], 
        ['^2', 1, np.square], 
        ['^3', 1, cubic]
    ], dtype=object)

    # 定义求导操作的分母项
    den = np.array([['x', 0, x]], dtype=object)
    
    # 将所有符号库打包成一个字典返回
    return {
        'VARS': VARS,
        'OPS': OPS,
        'ROOT': ROOT,
        'OP1': OP1,
        'OP2': OP2,
        'den': den
    }
