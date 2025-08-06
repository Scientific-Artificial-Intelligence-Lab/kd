import pytest
import numpy as np

from sga_refactored.main import inject_dependencies
from sga_refactored.data_utils import prepare_workspace
from sga_refactored.operators import get_default_library

# 导入原始模块
from codes import pde, tree, Data_generator

def test_inject_dependencies_basic():
    """
    验证 inject_dependencies 能正确注入 workspace 和 symbol_library 到 pde.py 和 tree.py。
    """
    # 1. 构造 workspace 和 symbol_library
    u, t, x = Data_generator.u, Data_generator.t, Data_generator.x
    workspace = prepare_workspace(u, t, x)
    symbol_library = get_default_library(workspace)

    # 2. 注入依赖
    inject_dependencies(workspace, symbol_library)

    # 3. 验证 pde.py 的 default_terms、num_default、ut 是否被正确注入
    assert hasattr(pde, "default_terms")
    assert hasattr(pde, "num_default")
    assert hasattr(pde, "ut")
    assert pde.default_terms.shape[0] == u.size
    assert pde.ut.shape == workspace['ut'].reshape(-1, 1).shape

    # 4. 验证 tree.py 的 ROOT、OPS、VARS 是否被正确注入
    assert hasattr(tree, "ROOT")
    assert hasattr(tree, "OPS")
    assert hasattr(tree, "VARS")
    assert tree.ROOT.shape[0] > 0
    assert tree.OPS.shape[0] > 0
    assert tree.VARS.shape[0] > 0

    # 5. 验证注入后可被主流程读取
    # 例如 pde.num_default 应等于 default_terms 的列数
    assert pde.num_default == pde.default_terms.shape[1]

    print("inject_dependencies 基础注入测试通过。")
