"""KD_PySR 最小使用示例。

本脚本演示如何在纯 tabular 数据上使用 KD_PySR 进行符号回归。

注意：
    - 运行本示例前需要在当前环境中安装 pysr 及其依赖；
    - 这只是一个 very-minimal 的用法示例，参数配置刻意保持简单。
"""

import numpy as np

from kd.model.kd_pysr import KD_PySR


def make_synthetic_data(n_samples: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n_samples, 2))
    # 目标关系：y = x0^2 + 0.5 * x1  + 少量噪声
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] + 0.01 * rng.normal(size=n_samples)
    return X, y


def main():
    X, y = make_synthetic_data()

    model = KD_PySR(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        extra_model_kwargs={
            # 这些参数是否被支持取决于 pysr 版本，这里尽量只用常见配置。
            "progress": False,
        },
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    mse = float(np.mean((y_pred - y) ** 2))

    print("== KD_PySR minimal example ==")
    print("best_equation_:", model.best_equation_)
    print("equation_latex:", model.equation_latex())
    print("train MSE:", mse)


if __name__ == "__main__":
    main()

