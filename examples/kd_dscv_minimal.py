import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.dataset import load_pde
from kd.model import KD_DSCV


def main():
    # 1. 通过统一入口加载 PDE 数据集
    dataset = load_pde("burgers")
    print(f"Loaded dataset 'burgers' with shape {dataset.usol.shape}")

    # 2. 初始化 KD_DSCV（Local PDE 模式）
    model = KD_DSCV(
        binary_operators=["add", "mul", "diff"],
        unary_operators=["n2"],
        n_iterations=5,
        n_samples_per_batch=500,
        seed=0,
    )

    np.random.seed(42)

    # 3. 使用 fit_dataset 语法糖执行训练
    print("\nTraining KD_DSCV via fit_dataset(...) ...")
    result = model.fit_dataset(dataset, n_epochs=11, verbose=True)

    expr = result.get("expression")
    reward = result.get("r")
    print(f"\nCurrent best expression is {expr} and its reward is {reward}")


if __name__ == "__main__":
    main()

