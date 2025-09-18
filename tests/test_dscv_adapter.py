import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kd.model.discover.task.pde.utils_fd import FiniteDiff
from kd.dataset import get_dataset_sym_true, load_pde
from kd.model.discover.adapter import DSCVRegularAdapter, DSCVSparseAdapter
from kd.model.kd_dscv import KD_DSCV, KD_DSCV_SPR
from kd.model.discover.searcher import Searcher


def test_dscv_regular_adapter_generates_expected_fields():
    dataset = load_pde('chafee-infante')

    adapter = DSCVRegularAdapter(dataset)
    data = adapter.get_data()

    assert set(data.keys()) >= {'u', 'ut', 'X', 'n_input_dim'}
    assert data['u'].shape == dataset.usol.shape
    assert data['ut'].shape == dataset.usol.shape
    assert len(data['X']) == 1
    assert data['X'][0].shape == (dataset.usol.shape[0], 1)
    assert data['n_input_dim'] == 1

    expected_sym = get_dataset_sym_true('chafee-infante')
    assert data.get('sym_true') == expected_sym

    dt = dataset.t[1] - dataset.t[0]
    expected_ut = np.zeros_like(dataset.usol)
    for idx in range(dataset.usol.shape[0]):
        expected_ut[idx, :] = FiniteDiff(dataset.usol[idx, :], dt)

    np.testing.assert_allclose(data['ut'], expected_ut)


def test_kd_dscv_import_dataset(monkeypatch):
    dataset = load_pde('chafee-infante')
    model = KD_DSCV(n_iterations=1, n_samples_per_batch=10)

    calls = {}

    def fake_setup(self):
        calls['called'] = True

    monkeypatch.setattr(KD_DSCV, 'setup', fake_setup)

    result = model.import_dataset(dataset)

    assert result is model
    assert calls.get('called') is True
    assert model.data_class.get_data()['u'].shape == dataset.usol.shape
    assert model.dataset == dataset.legacy_name
    assert model.out_path.endswith(f"discover_{dataset.legacy_name}_{model.seed}.csv")


def test_dscv_sparse_adapter_shapes():
    dataset = load_pde('burgers')

    colloc_num = 512
    adapter = DSCVSparseAdapter(
        dataset,
        sample=200,
        colloc_num=colloc_num,
        random_state=0,
    )
    data = adapter.get_data()

    assert set(data.keys()) >= {
        'X_u_train', 'u_train', 'X_f_train', 'X_u_val', 'u_val', 'lb', 'ub'
    }
    assert data['X_u_train'].shape[0] > 0
    assert data['X_u_val'].shape[0] > 0
    assert data['u_train'].shape[0] == data['X_u_train'].shape[0]
    assert data['u_val'].shape[0] == data['X_u_val'].shape[0]
    assert data['X_f_train'].shape[0] == colloc_num + data['X_u_train'].shape[0]
    assert len(data['lb']) == data['ub'].shape[0] == 2


def test_kd_dscv_spr_import_dataset(monkeypatch):
    dataset = load_pde('burgers')
    model = KD_DSCV_SPR(n_iterations=1, n_samples_per_batch=10)

    calls = {}

    def fake_setup(self):
        calls['called'] = True

    monkeypatch.setattr(KD_DSCV_SPR, 'setup', fake_setup)

    result = model.import_dataset(dataset, sample=50, colloc_num=128, random_state=1)

    assert result is model
    assert calls.get('called') is True
    sparse_data = model.data_class.get_data()
    assert sparse_data['X_f_train'].shape[0] == 128 + sparse_data['X_u_train'].shape[0]
    assert model.dataset == dataset.legacy_name
    assert model.out_path.endswith(f"discover_{dataset.legacy_name}_{model.seed}.csv")


def test_kd_dscv_train_smoke(monkeypatch):
    dataset = load_pde('chafee-infante')

    model = KD_DSCV(n_iterations=1, n_samples_per_batch=10)

    monkeypatch.setattr(KD_DSCV, 'make_gp_aggregator', lambda self: None)

    def fake_search(self, n_epochs=1, verbose=True, keep_history=True):
        return {'program': None, 'expression': 'u_t = 0', 'r': 0.0}

    def fake_search_one_step(self, epoch=0, verbose=True):
        return {'program': None, 'expression': 'u_t = 0', 'r': 0.0}

    monkeypatch.setattr(Searcher, 'search', fake_search, raising=False)
    monkeypatch.setattr(Searcher, 'search_one_step', fake_search_one_step, raising=False)

    model.import_dataset(dataset)
    result = model.train(n_epochs=1, verbose=False)

    assert result['expression'] == 'u_t = 0'


def test_kd_dscv_spr_train_smoke(monkeypatch):
    dataset = load_pde('burgers')

    model = KD_DSCV_SPR(n_iterations=1, n_samples_per_batch=10)

    monkeypatch.setattr(KD_DSCV_SPR, 'make_gp_aggregator', lambda self: None, raising=False)

    def fake_setup(self):
        # 保持 config_task 存在，避免 train 内访问失败
        self.config_task.setdefault('eq_num', 1)

    def fake_call_iter(self, n_epochs, verbose):
        return {'program': None, 'expression': 'u_t = 0', 'r': 0.0}

    monkeypatch.setattr(KD_DSCV_SPR, 'setup', fake_setup, raising=False)
    monkeypatch.setattr(KD_DSCV_SPR, 'callIterPINN', fake_call_iter, raising=False)

    model.import_dataset(dataset, sample=50, colloc_num=128, random_state=1)
    result = model.train(n_epochs=1, verbose=False)

    assert result['expression'] == 'u_t = 0'
