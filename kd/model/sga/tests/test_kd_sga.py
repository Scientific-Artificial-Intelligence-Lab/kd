import pytest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import from kd.model
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
sys.path.insert(0, kd_root)

from kd.model import KD_SGA

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
    fitted_model = model.fit(X=None, max_gen=1, verbose=False)
    assert fitted_model is model  # sklearn style: fit returns self
    assert model.best_equation_ is not None
    assert isinstance(model.best_aic_, (float, np.floating, int))

    print("KD_SGA 基础功能测试通过。")
