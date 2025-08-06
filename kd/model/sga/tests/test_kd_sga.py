import pytest
import numpy as np

from sga_refactored.kd_sga import KD_SGA

def test_kd_sga_basic_fit():
    """
    验证 KD_SGA 类的初始化、参数管理和 fit 主流程可用。
    """
    # 1. 初始化
    model = KD_SGA(num=5, depth=2, width=2)
    params = model.get_params()
    assert params["num"] == 5
    assert params["depth"] == 2
    assert params["width"] == 2

    # 2. set_params
    model.set_params(num=3, width=1)
    params = model.get_params()
    assert params["num"] == 3
    assert params["width"] == 1

    # 3. fit 流程
    best_eq_obj, best_aic = model.fit(max_gen=1, verbose=False)
    assert best_eq_obj is not None
    assert isinstance(best_aic, (float, np.floating, int))

    print("KD_SGA 基础功能测试通过。")
