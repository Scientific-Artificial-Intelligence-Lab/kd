import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

import warnings

from kd.dataset import load_pde
from kd.model import KD_DSCV_SPR

warnings.filterwarnings("ignore", category=FutureWarning, module="numpy.*")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    # 1. 通过统一入口加载 Burgers 数据集
    dataset = load_pde("burgers")
    print(f"Loaded dataset 'burgers' with shape {dataset.usol.shape}")

    # 2. 初始化 KD_DSCV_SPR（Sparse + PINN 模式）
    model = KD_DSCV_SPR(
        n_iterations=1,
        n_samples_per_batch=10,
        binary_operators=["add_t", "diff_t"],
        unary_operators=["n2_t"],
        seed=0,
    )

    np.random.seed(42)

    # 3. 使用 fit_dataset 语法糖：构建稀疏数据并迭代一次 PDE+PINN 搜索
    #    为了示例快速结束，这里使用较小的 colloc_num 与 n_epochs。
    print("\nTraining KD_DSCV_SPR via fit_dataset(...) ...")
    result = model.fit_dataset(
        dataset,
        n_epochs=1,
        verbose=False,
        sample_ratio=0.1,
        colloc_num=512,
        random_state=0,
        noise_level=0.0,
    )

    expr = result.get("expression")
    reward = result.get("r")
    print(f"\nCurrent best expression is {expr} and its reward is {reward}")


if __name__ == "__main__":
    main()

