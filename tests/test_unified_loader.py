"""Integration-level checks that every core model consumes PDEDataset via the unified loader."""

import pytest

from kd.dataset import load_pde
from kd.model.kd_dlga import KD_DLGA
from kd.model.kd_sga import KD_SGA
from kd.model.kd_dscv import KD_DSCV, KD_DSCV_SPR
from kd.model.discover.searcher import Searcher


def test_unified_loader_dlga(monkeypatch):
    dataset = load_pde('kdv')
    X_train, y_train = dataset.sample(32)

    model = KD_DLGA(
        operators=['u', 'u_x'],
        epi=0.1,
        input_dim=2,
        verbose=False,
        max_iter=1,
    )

    captured = {}

    def fake_fit(self, X, y):
        captured['x_shape'] = X.shape
        captured['y_shape'] = y.shape
        return self

    monkeypatch.setattr(KD_DLGA, 'fit', fake_fit, raising=False)

    model.fit(X_train, y_train)

    assert captured['x_shape'] == X_train.shape
    assert captured['y_shape'] == y_train.shape


@pytest.mark.parametrize('dataset_name', ['chafee-infante', 'burgers', 'kdv'])
def test_unified_loader_sga(monkeypatch, dataset_name):
    if dataset_name != 'chafee-infante':
        pytest.importorskip('scipy')

    dataset = load_pde(dataset_name)

    class DummyContext:
        def __init__(self, config):
            self.config = config

    class DummySolver:
        def __init__(self, config):
            self.config = config

        def run(self, context):
            return "u_t = 0", 0.0

    model = KD_SGA(sga_run=1, depth=1, width=1, num=1, seed=0)
    result = model.fit_dataset(dataset, context_cls=DummyContext, solver_cls=DummySolver)

    assert result is model
    assert model.dataset_ is dataset


def test_unified_loader_dscv(monkeypatch):
    dataset = load_pde('chafee-infante')
    model = KD_DSCV(n_iterations=1, n_samples_per_batch=10)

    monkeypatch.setattr(KD_DSCV, 'make_gp_aggregator', lambda self: None, raising=False)

    def fake_search(self, n_epochs=1, verbose=True, keep_history=True):
        return {'program': None, 'expression': 'u_t = 0', 'r': 0.0}

    def fake_search_one_step(self, epoch=0, verbose=True):
        return {'program': None, 'expression': 'u_t = 0', 'r': 0.0}

    monkeypatch.setattr(Searcher, 'search', fake_search, raising=False)
    monkeypatch.setattr(Searcher, 'search_one_step', fake_search_one_step, raising=False)

    result = model.fit_from_dataset(dataset, n_epochs=1, verbose=False)

    assert result['expression'] == 'u_t = 0'
    assert model.dataset_ is dataset


def test_unified_loader_dscv_spr(monkeypatch):
    dataset = load_pde('burgers')
    model = KD_DSCV_SPR(
        n_iterations=1,
        n_samples_per_batch=10,
        binary_operators=["add_t", "diff_t"],
        unary_operators=['n2_t'],
    )

    monkeypatch.setattr(KD_DSCV_SPR, 'make_gp_aggregator', lambda self: None, raising=False)

    def fake_setup(self):
        self.config_task.setdefault('eq_num', 1)
        KD_DSCV.setup(self)

    def fake_call_iter(self, n_epochs, verbose):
        return {'program': None, 'expression': 'u_t = 0', 'r': 0.0}

    monkeypatch.setattr(KD_DSCV_SPR, 'setup', fake_setup, raising=False)
    monkeypatch.setattr(KD_DSCV_SPR, 'callIterPINN', fake_call_iter, raising=False)

    model.import_dataset(dataset, sample=50, colloc_num=64, random_state=0)
    result = model.train(n_epochs=1, verbose=False)

    assert result['expression'] == 'u_t = 0'
    assert model.dataset_ is dataset
