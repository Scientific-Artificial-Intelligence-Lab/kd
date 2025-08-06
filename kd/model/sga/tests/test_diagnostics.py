import pytest
import numpy as np
import re

from sga_refactored.diagnostics import run_diagnostics
from sga_refactored.data_utils import prepare_workspace
from codes import configure, Data_generator

@pytest.fixture(scope="module")
def golden_standard_error():
    """加载黄金标准的误差值。"""
    try:
        return np.loadtxt("tests/diagnostics_golden.txt")
    except FileNotFoundError:
        pytest.fail("diagnostics_golden.txt not found. Please run original script first.")

def test_run_diagnostics_matches_golden_standard(golden_standard_error, capsys):
    """
    测试 run_diagnostics 函数打印的误差值是否与黄金标准完全一致。
    """
    # 1. 准备输入数据
    u, t, x = Data_generator.u, Data_generator.t, Data_generator.x
    workspace = prepare_workspace(u, t, x)
    
    # 2. 调用我们的新函数
    run_diagnostics(workspace, configure)
    
    # 3. 捕获并解析输出
    captured = capsys.readouterr()
    output_text = captured.out
    
    # 使用正则表达式从打印的字符串中精确提取出数值
    match = re.search(r"Data error \(without edges\): ([-+]?\d*\.\d+|\d+)", output_text)
    
    assert match is not None, "Could not find the error value in the output string."
    
    # 将提取出的字符串转换为浮点数
    calculated_error = float(match.group(1))
    
    # 4. 与黄金标准进行比较
    print(f"\nComparing errors: Calculated={calculated_error}, GoldenStandard={golden_standard_error}")
    assert np.isclose(calculated_error, golden_standard_error, rtol=1e-7)