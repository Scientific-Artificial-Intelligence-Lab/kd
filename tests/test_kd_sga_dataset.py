import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kd.dataset import load_pde
from kd.model.kd_sga import KD_SGA


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
