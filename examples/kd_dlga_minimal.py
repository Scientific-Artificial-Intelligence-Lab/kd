import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from kd.dataset import load_pde
from kd.model.kd_dlga import KD_DLGA


def main():
    # 1. 通过统一入口加载 PDE 数据集
    dataset = load_pde("kdv")
    print(f"Loaded dataset 'kdv' with shape {dataset.usol.shape}")

    # 2. 使用 fit_dataset 接口训练 DLGA 模型（推荐入口）
    model = KD_DLGA(
        operators=["u", "u_x", "u_xx", "u_xxx"],
        epi=0.1,
        input_dim=2,
        verbose=False,
        max_iter=9000,
    )

    print("\nTraining KD_DLGA via fit_dataset(...) ...")
    model.fit_dataset(dataset, sample=1000, sample_method="random")

    # 3. 打印发现的方程
    eq_latex = getattr(model, "eq_latex", "")
    if eq_latex:
        print("\nDiscovered equation (LaTeX):")
        print(eq_latex)
    else:
        best_eq = getattr(model, "best_equation_", None)
        if best_eq:
            print("\nDiscovered equation (raw string):")
            print(best_eq)
        else:
            print("\nNo equation string available; please check DLGA evolution settings.")

    # 4. 可选：在完整网格上预测，并打印简单的 MSE 作为参考
    print("\nEvaluating prediction MSE on full grid ...")
    X_full = dataset.mesh()
    u_pred = model.predict(X_full).reshape(dataset.get_size())
    mse = float(np.mean((u_pred - dataset.usol) ** 2))
    print(f"MSE on full grid: {mse:.6e}")


if __name__ == "__main__":
    main()

