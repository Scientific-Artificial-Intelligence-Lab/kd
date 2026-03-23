"""
KD Discover Regression — 替代 PySR 的符号回归

使用 KD 框架的 Discover 模型从表格数据中发现方程。
数据来自 TLC-CC 论文 (Nature Comm. 2025:16:832)。

对比 PySR 的用法：
    # PySR (原论文代码)
    model = PySRRegressor(
        binary_operators=["+", "*"],
        unary_operators=["inv(x) = 1/x"],
    )
    model.fit(Rf_t1[:, 0:2], Rf_t1[:, 2])

    # KD Discover (替代方案)
    from kd.dataset import load_regression
    from kd.model.kd_discover_regression import KD_Discover_Regression

    X, y, meta = load_regression("tlc_cc_t1")
    model = KD_Discover_Regression(
        binary_operators=["add", "sub", "mul", "div"],
        unary_operators=["inv"],
        n_iterations=50,
        n_samples_per_batch=500,
        seed=1,
        config_out={"task": {"parsimony_coeff": 0.005}},
    )
    result = model.fit(X, y, var_names=meta["var_names"])
"""

import logging
from pathlib import Path

import numpy as np

from kd.dataset import load_regression
from kd.model.kd_discover_regression import KD_Discover_Regression
from kd.model.discover.functions import PlaceholderConstant

logger = logging.getLogger(__name__)


def program_to_readable(program, var_names=None):
    """Convert a Program's traversal to a human-readable infix string."""
    tokens = list(program.traversal)

    def _fmt_const(tok):
        v = float(np.asarray(tok.value).flat[0])
        return f"{v:.4g}"

    def _build(toks):
        tok = toks.pop(0)
        if isinstance(tok, PlaceholderConstant) and tok.value is not None:
            return _fmt_const(tok)
        name = tok.name if hasattr(tok, "name") else str(tok)
        arity = tok.arity if hasattr(tok, "arity") else 0
        if arity == 0:
            if var_names and name.startswith("x"):
                idx = int(name[1:]) - 1
                if 0 <= idx < len(var_names):
                    return var_names[idx]
            return name
        if arity == 1:
            child = _build(toks)
            return f"{name}({child})"
        left = _build(toks)
        right = _build(toks)
        op_map = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        op = op_map.get(name, name)
        return f"({left} {op} {right})"

    return _build(tokens)


def discover_equation(name, seed=1):
    """Run Discover regression on a named dataset."""
    X, y, meta = load_regression(name)
    var_names = meta["var_names"]

    print(f"\n{'=' * 60}")
    print(f"数据集: {name} — {meta['description']}")
    print(f"样本数: {X.shape[0]}, 特征: {var_names}, 目标: {meta['target_name']}")
    print(f"{'=' * 60}")

    model = KD_Discover_Regression(
        binary_operators=["add", "sub", "mul", "div"],
        unary_operators=["inv"],
        n_iterations=50,
        n_samples_per_batch=500,
        seed=seed,
        config_out={"task": {"parsimony_coeff": 0.005}},
    )

    result = model.fit(X, y, var_names=var_names, verbose=True)

    readable = program_to_readable(result["program"], var_names)
    y_pred = model.predict(X)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    print(f"\n公式:   {meta['target_name']} = {readable}")
    print(f"R²:     {r2:.6f}")
    print(f"MSE:    {result['mse']:.4f}")
    print(f"Reward: {result['reward']:.6f}")

    return result, model, X, y, meta


def visualize(model, X, y, meta, output_dir="artifacts/regression_viz"):
    """Generate all visualizations for a fitted regression model."""
    from kd.viz.adapters import register_default_adapters
    from kd.viz import api as viz_api

    register_default_adapters()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = meta["target_name"]

    # 1. Equation rendering (LaTeX)
    result = viz_api.render_equation(
        model, output_dir=out, target_name=target,
    )
    logger.info("Equation: %s", result.metadata.get("latex", ""))

    # 2. Parity plot (actual vs predicted)
    result = viz_api.plot_parity(model, output_dir=out)
    summary = result.metadata.get("summary", {})
    logger.info("Parity RMSE: %.6f", summary.get("rmse", float("nan")))

    # 3. Residual analysis
    # The regression adapter auto-extracts y/y_pred from model.data_class,
    # but api.plot_residuals requires actual/predicted kwargs.
    y_pred = model.predict(X)
    viz_api.plot_residuals(model, actual=y, predicted=y_pred, output_dir=out)

    # 4. Search evolution curve
    viz_api.plot_search_evolution(model, output_dir=out)

    logger.info("Visualizations saved to %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 发现两个方程：V_S 和 V_E
    result1, model1, X1, y1, meta1 = discover_equation("tlc_cc_t1")
    result2, model2, X2, y2, meta2 = discover_equation("tlc_cc_t2")

    # 可视化
    visualize(model1, X1, y1, meta1, "artifacts/regression_viz/t1")
    visualize(model2, X2, y2, meta2, "artifacts/regression_viz/t2")
