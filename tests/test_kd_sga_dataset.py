import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kd.dataset import load_pde
from kd.model.kd_sga import KD_SGA
from kd.model.sga.sgapde.config import SolverConfig
from kd.model.sga.sgapde.context import ProblemContext


class DummyContext:
    def __init__(self, config):
        self.config = config


class DummySolver:
    def __init__(self, config):
        self.config = config

    def run(self, context):
        # 确认上下文中包含配置
        assert context.config is self.config
        return "u_t = 0", 0.0


def test_fit_dataset_uses_adapter():
    dataset = load_pde('chafee-infante')

    model = KD_SGA(sga_run=1, depth=1, width=1)
    result = model.fit_dataset(
        dataset,
        context_cls=DummyContext,
        solver_cls=DummySolver,
    )

    assert result is model
    assert model.dataset_ is dataset

    np.testing.assert_allclose(model.config_.u_data, dataset.usol)
    np.testing.assert_allclose(model.config_.x_data, dataset.x)
    np.testing.assert_allclose(model.config_.t_data, dataset.t)

    assert model.best_pde_ == "u_t = 0"
    assert model.best_score_ == 0.0
    assert isinstance(model.context_, DummyContext)


def test_fit_dataset_requires_pdedataset():
    model = KD_SGA()
    with pytest.raises(TypeError):
        model.fit_dataset("not-a-dataset", context_cls=DummyContext, solver_cls=DummySolver)


def _make_inline_data(n=5, m=6):
    x = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 1.0, m)
    u = np.outer(np.sin(x), np.cos(t))
    return u, x, t


def test_solver_config_custom_inline_allows_unknown_problem():
    u, x, t = _make_inline_data()
    config = SolverConfig(problem_name="custom_sga_test", u_data=u, x_data=x, t_data=t)

    np.testing.assert_allclose(config.u, u)
    np.testing.assert_allclose(config.x, x)
    np.testing.assert_allclose(config.t, t)

    assert config.has_ground_truth is False
    assert config.right_side is None
    assert config.left_side is None
    assert config.right_side_origin is None
    assert config.left_side_origin is None


def test_problem_context_skips_ground_truth_when_absent():
    u, x, t = _make_inline_data()
    config = SolverConfig(problem_name="custom_sga_test", u_data=u, x_data=x, t_data=t)
    context = ProblemContext(config)

    # 必须构造默认特征，供 SGA 算子库使用
    assert hasattr(context, "default_terms")
    assert context.default_terms.shape[0] == u.size
    assert context.num_default == 1
    assert context.default_names == ["u"]

    # 无解析模板时不应创建 ground-truth 残差字段
    assert not hasattr(context, "right_side_full")
    assert not hasattr(context, "right_side_full_origin")
