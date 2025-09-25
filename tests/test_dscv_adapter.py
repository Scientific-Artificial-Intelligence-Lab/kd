import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kd.model.discover.task.pde.utils_fd import FiniteDiff
from kd.dataset import get_dataset_sym_true, load_pde, PDEDataset
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
    assert model.dataset_ is dataset


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


def test_dscv_sparse_adapter_includes_full_field_metadata():
    dataset = load_pde('burgers')

    adapter = DSCVSparseAdapter(dataset, sample_ratio=0.05, colloc_num=128, random_state=0)
    data = adapter.get_data()

    assert set(data.keys()) >= {'X_star', 'u_star', 'shape'}

    expected_shape = dataset.usol.shape
    assert tuple(data['shape']) == expected_shape

    mesh_x, mesh_t = np.meshgrid(dataset.x, dataset.t, indexing='ij')
    expected_X_star = np.column_stack((mesh_x.ravel(), mesh_t.ravel()))
    expected_u_star = dataset.usol.reshape(-1, 1)

    np.testing.assert_allclose(data['X_star'], expected_X_star)
    np.testing.assert_allclose(data['u_star'], expected_u_star)


def test_dscv_sparse_adapter_respects_noise_and_sampling():
    dataset = load_pde('burgers')

    baseline_adapter = DSCVSparseAdapter(
        dataset,
        sample_ratio=0.1,
        colloc_num=64,
        random_state=0,
    )
    baseline = baseline_adapter.get_data()

    adapter = DSCVSparseAdapter(
        dataset,
        sample_ratio=0.1,
        colloc_num=64,
        random_state=0,
        noise_level=0.2,
        data_ratio=0.3,
    )
    data = adapter.get_data()

    assert data['u_train'].shape[0] == int(data['X_u_train'].shape[0])

    expected_train = max(1, int(baseline['X_u_train'].shape[0] * 0.3))
    assert data['X_u_train'].shape[0] == expected_train

    colloc = baseline['X_f_train'].shape[0] - baseline['X_u_train'].shape[0]
    assert data['X_f_train'].shape[0] == colloc + expected_train

    original = dataset.usol.reshape(-1, 1)
    noisy_flat = np.concatenate([data['u_train'], data['u_val']]).flatten()
    assert not np.allclose(noisy_flat[:original.shape[0]], original[:noisy_flat.shape[0]], atol=1e-3)


def test_dscv_sparse_adapter_handles_non_uniform_time():
    x = np.linspace(-1.0, 1.0, 6)
    t = np.array([0.0, 0.05, 0.17, 0.31, 0.5])
    X, T = np.meshgrid(x, t, indexing='ij')
    u = np.sin(X) * np.cos(T)

    dataset = PDEDataset(
        equation_name='synthetic-nu',
        pde_data=None,
        domain={'x': (float(x.min()), float(x.max())), 't': (float(t.min()), float(t.max()))},
        epi=1e-3,
        x=x,
        t=t,
        usol=u,
    )

    adapter = DSCVSparseAdapter(dataset, sample_ratio=0.2, random_state=0)
    data = adapter.get_data()

    assert data['X_u_train'].shape[1] == 2
    lb, ub = data['lb'], data['ub']
    mesh = np.column_stack((np.meshgrid(x, t, indexing='ij')[0].ravel(), np.meshgrid(x, t, indexing='ij')[1].ravel()))
    np.testing.assert_allclose(lb, mesh.min(axis=0))
    np.testing.assert_allclose(ub, mesh.max(axis=0))


def test_dscv_sparse_adapter_cut_quantile():
    dataset = load_pde('burgers')

    original_lb, original_ub = dataset.mesh_bounds()
    adapter = DSCVSparseAdapter(dataset, sample_ratio=0.1, cut_quantile=0.1, random_state=0)
    data = adapter.get_data()

    assert data['lb'][0] > original_lb[0]
    assert data['ub'][0] < original_ub[0]


def test_dscv_sparse_adapter_spline_sampling():
    dataset = load_pde('burgers')

    adapter = DSCVSparseAdapter(dataset, sample_ratio=0.2, spline_sample=True, random_state=0)
    data = adapter.get_data()

    measured = data['measured_points']
    unique_x = np.unique(measured[:, 0])
    expected = min(len(dataset.x), max(1, int(len(dataset.x) * 0.2)))
    assert unique_x.shape[0] == expected
    counts = [np.sum(measured[:, 0] == ux) for ux in unique_x]
    assert all(count == len(dataset.t) for count in counts)
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


def test_kd_dscv_spr_import_dataset_propagates_options(monkeypatch):
    dataset = load_pde('burgers')
    model = KD_DSCV_SPR(n_iterations=1, n_samples_per_batch=10)

    monkeypatch.setattr(KD_DSCV_SPR, 'setup', lambda self: None)

    baseline = DSCVSparseAdapter(
        dataset,
        sample_ratio=0.1,
        colloc_num=128,
        random_state=42,
        noise_level=0.3,
        data_ratio=0.25,
        spline_sample=True,
        cut_quantile=0.05,
    )
    baseline_data = baseline.get_data()

    model.import_dataset(
        dataset,
        sample_ratio=0.1,
        colloc_num=128,
        random_state=42,
        noise_level=0.3,
        data_ratio=0.25,
        spline_sample=True,
        cut_quantile=0.05,
    )
    sparse_data = model.data_class.get_data()

    assert sparse_data['X_u_train'].shape[0] == baseline_data['X_u_train'].shape[0]
    colloc = baseline_data['X_f_train'].shape[0] - baseline_data['X_u_train'].shape[0]
    assert sparse_data['X_f_train'].shape[0] == colloc + sparse_data['X_u_train'].shape[0]

    original_flat = dataset.usol.reshape(-1, 1)
    subset_flat = np.concatenate([sparse_data['u_train'], sparse_data['u_val']]).flatten()
    assert not np.allclose(subset_flat[:original_flat.shape[0]], original_flat[:subset_flat.shape[0]], atol=1e-3)

    assert sparse_data['sampling_strategy'] == 'spline'


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


def test_kd_dscv_fit_from_dataset(monkeypatch):
    dataset = load_pde('chafee-infante')

    model = KD_DSCV(n_iterations=1, n_samples_per_batch=10)

    monkeypatch.setattr(KD_DSCV, 'make_gp_aggregator', lambda self: None)

    def fake_search(self, n_epochs=1, verbose=True, keep_history=True):
        return {'program': None, 'expression': 'u_t = 0', 'r': 0.0}

    monkeypatch.setattr(Searcher, 'search', fake_search, raising=False)

    result = model.fit_from_dataset(dataset, n_epochs=1, verbose=False)

    assert result['expression'] == 'u_t = 0'
    assert model.dataset_ is dataset


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



def test_kd_dscv_spr_train_integration(monkeypatch):
    """End-to-end smoke test for KD_DSCV_SPR with real dataset."""
    dataset = load_pde('burgers')
    model = KD_DSCV_SPR(
        n_iterations=1,
        n_samples_per_batch=10,
        binary_operators=["add_t", "diff_t"],
        unary_operators=['n2_t'],
    )

    # Simplify heavy components while keeping PINN path intact
    monkeypatch.setattr(KD_DSCV_SPR, 'make_gp_aggregator', lambda self: None, raising=False)

    def mock_setup(self):
        # ensure metadata structures exist
        self.config_task.setdefault('eq_num', 1)
        KD_DSCV.setup(self)

    monkeypatch.setattr(KD_DSCV_SPR, 'setup', mock_setup, raising=False)

    def mock_call_iter(self, n_epochs, verbose):
        return {'program': None, 'expression': 'u_t = 0', 'r': 0.0}

    monkeypatch.setattr(KD_DSCV_SPR, 'callIterPINN', mock_call_iter, raising=False)

    model.import_dataset(dataset, sample=50, colloc_num=64, random_state=0)
    result = model.train(n_epochs=1, verbose=False)
    assert result['expression'] == 'u_t = 0'
