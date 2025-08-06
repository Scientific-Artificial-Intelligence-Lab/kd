import pytest
import numpy as np

# 导入我们刚刚创建的模块
from sga_refactored.operators import get_default_library

@pytest.fixture
def mock_workspace():
    """创建一个模拟的 workspace 字典，用于测试。"""
    u = np.random.rand(10, 10)
    x = np.random.rand(10, 10)
    ux = np.random.rand(10, 10)
    return {'u': u, 'x': x, 'ux': ux}


def test_get_default_library_returns_dict(mock_workspace):
    """测试函数是否返回一个字典。"""
    library = get_default_library(mock_workspace)
    assert isinstance(library, dict)

def test_library_has_all_required_keys(mock_workspace):
    """测试返回的字典是否包含了所有必需的键。"""
    library = get_default_library(mock_workspace)
    expected_keys = ['VARS', 'OPS', 'ROOT', 'OP1', 'OP2', 'den']
    for key in expected_keys:
        assert key in library

def test_vars_library_uses_workspace_data(mock_workspace):
    """测试 VARS 符号库是否正确地使用了传入的数据。"""
    library = get_default_library(mock_workspace)
    vars_lib = library['VARS']
    
    # 找到 'u' 对应的条目
    u_entry = next((item for item in vars_lib if item[0] == 'u'), None)
    assert u_entry is not None
    
    # 验证它包含的 numpy 数组是否就是我们传入的那个
    # 'is' 比较的是对象的内存地址，确保没有发生不必要的数据复制
    assert u_entry[2] is mock_workspace['u']

    # 找到 'ux' 对应的条目并验证
    ux_entry = next((item for item in vars_lib if item[0] == 'ux'), None)
    assert ux_entry is not None
    assert ux_entry[2] is mock_workspace['ux']