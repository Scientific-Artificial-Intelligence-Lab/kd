"""Very small PySR + viz smoke tests.

这些测试仅在安装了 pysr 时才会运行，用于防止基础 API 或
viz adapter 被未来修改时意外破坏。
"""

import numpy as np
import pytest

pysr = pytest.importorskip("pysr", reason="KD_PySR viz tests require pysr")


def _make_synthetic():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(64, 2))
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] + 0.01 * rng.normal(size=64)
    return X, y


def test_kd_pysr_viz_equation_parity_residual(tmp_path):
    from kd.model.kd_pysr import KD_PySR
    from kd.viz import VizRequest, configure, render

    X, y = _make_synthetic()

    model = KD_PySR(
        niterations=20,
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        extra_model_kwargs={"progress": False},
    )
    model.fit(X, y)

    configure(save_dir=tmp_path)

    # equation
    eq_result = render(VizRequest(kind="equation", target=model))
    assert eq_result.has_content
    assert eq_result.paths, "equation intent should produce an image path"

    # parity
    parity_result = render(VizRequest(kind="parity", target=model))
    assert parity_result.has_content
    assert parity_result.paths, "parity intent should produce an image path"

    # residual
    residual_result = render(VizRequest(kind="residual", target=model))
    assert residual_result.has_content
    assert residual_result.paths, "residual intent should produce an image path"

