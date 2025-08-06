import pytest
import numpy as np

# 导入我们正在测试的目标函数
from sga_refactored.data_utils import prepare_workspace

# 导入原始的 configure 和 Data_generator 来加载与黄金标准完全一致的【输入数据】
# 这是为了确保我们的测试输入和生成黄金标准时的输入是完全一样的
from codes import configure
from codes import Data_generator

@pytest.fixture(scope="module")
def golden_standard_data():
    """
    一个 pytest fixture (测试“夹具”)。
    它负责在测试开始前加载所有黄金标准数据，并且在整个测试文件中只执行一次。
    """
    print("\nLoading golden standard data from .npy files...")
    try:
        # 从我们上一步生成的文件中加载标准答案
        ux = np.load("tests/ux_golden.npy")
        ut = np.load("tests/ut_golden.npy")
        uxx = np.load("tests/uxx_golden.npy")
        uxxx = np.load("tests/uxxx_golden.npy")
        return {"ux": ux, "ut": ut, "uxx": uxx, "uxxx": uxxx}
    except FileNotFoundError as e:
        pytest.fail(f"Golden standard .npy file not found: {e}. "
                    "Please ensure you've run the original script to generate them first.")

@pytest.fixture(scope="module")
def input_data():
    """这个 fixture 负责提供计算所需的原始输入数据 u, t, x。"""
    return Data_generator.u, Data_generator.t, Data_generator.x

def test_prepare_workspace_generates_correct_derivatives(input_data, golden_standard_data):
    """
    这是一个核心的集成测试。
    它验证我们重构的 `prepare_workspace` 函数是否产出了与黄金标准完全一致的导数。
    """
    # 1. 准备输入
    u, t, x = input_data

    # 2. 调用我们重构的函数进行计算
    print("Running the refactored 'prepare_workspace' function...")
    workspace = prepare_workspace(u, t, x)

    # 3. 进行断言比较
    print("Comparing refactored derivatives with the golden standard...")

    # 输出 ut 的最大、均方、绝对误差
    ut_ref = workspace['ut']
    ut_gold = golden_standard_data['ut']
    print("DEBUG: ut_ref.shape =", ut_ref.shape, "ut_gold.shape =", ut_gold.shape)
    print("DEBUG: ut max abs diff =", np.max(np.abs(ut_ref - ut_gold)))
    print("DEBUG: ut mean abs diff =", np.mean(np.abs(ut_ref - ut_gold)))
    print("DEBUG: ut RMSE =", np.sqrt(np.mean((ut_ref - ut_gold) ** 2)))

    # np.testing.assert_allclose 是用于比较浮点数数组的最佳工具
    # 它允许有非常小的容差（rtol），以应对不同计算方式可能带来的微小浮点误差
    np.testing.assert_allclose(workspace['ut'], golden_standard_data['ut'], rtol=1e-7,
                                err_msg="Derivative 'ut' does not match the golden standard.")
    
    np.testing.assert_allclose(workspace['ux'], golden_standard_data['ux'], rtol=1e-7,
                                err_msg="Derivative 'ux' does not match the golden standard.")
                                
    np.testing.assert_allclose(workspace['uxx'], golden_standard_data['uxx'], rtol=1e-7,
                                err_msg="Derivative 'uxx' does not match the golden standard.")
                                
    np.testing.assert_allclose(workspace['uxxx'], golden_standard_data['uxxx'], rtol=1e-7,
                                err_msg="Derivative 'uxxx' does not match the golden standard.")

    print("\n✅  Success! All derivatives calculated by the refactored module match the golden standard perfectly.")

def test_prepare_workspace_autograd_shape(input_data):
    """
    验证 prepare_workspace 在 autograd 模式下的导数 shape 是否与 finite_difference 一致。
    """
    u, t, x = input_data
    # 这里 device 和 model_path 需与 configure.py 保持一致
    from codes import configure
    device = configure.device
    model_path = configure.path

    print("Running prepare_workspace in autograd mode...")
    workspace = prepare_workspace(u, t, x, mode='autograd', model_path=model_path, device=device)
    n, m = u.shape
    assert workspace['ut'].shape == (n, m)
    assert workspace['ux'].shape == (n, m)
    assert workspace['uxx'].shape == (n, m)
    assert workspace['uxxx'].shape == (n, m)
    print("Autograd mode: all derivative shapes are correct.")
