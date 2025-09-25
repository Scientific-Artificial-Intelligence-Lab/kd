"""Validate that core models accept every dataset registered via the unified loader."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('scipy', reason='Unified loader requires scipy.io for .mat datasets')

from kd.dataset import get_dataset_info, list_available_datasets, load_pde
from kd.model.kd_dscv import KD_DSCV, KD_DSCV_SPR
from kd.model.kd_sga import KD_SGA


ALL_DATASETS = list_available_datasets()
STATUS_ACTIVE = 'active'


@pytest.mark.parametrize("dataset_name", ALL_DATASETS)
def test_load_pde_produces_consistent_shapes(dataset_name: str):
    info = get_dataset_info(dataset_name)
    if info.get('status', STATUS_ACTIVE) != STATUS_ACTIVE:
        pytest.skip(f"dataset {dataset_name} is marked as {info.get('status')}")

    dataset = load_pde(dataset_name)
    assert dataset.usol.shape == (len(dataset.x), len(dataset.t))
    assert dataset.registry_name == dataset_name

    aliases = info.get('aliases', {}) or {}
    expected_legacy = aliases.get('legacy', dataset_name)
    assert getattr(dataset, 'legacy_name') == expected_legacy

    domain = info.get('domain')
    if domain:
        assert dataset.get_domain() == domain

    shape = info.get('shape')
    if shape:
        assert dataset.usol.shape == shape


@pytest.mark.parametrize("dataset_name", ALL_DATASETS)
def test_kd_sga_accepts_supported_datasets(dataset_name: str):
    info = get_dataset_info(dataset_name)
    if not info.get('models', {}).get('sga'):
        pytest.skip(f"dataset {dataset_name} is not supported by KD_SGA")

    dataset = load_pde(dataset_name)

    class DummyContext:
        def __init__(self, config):
            self.config = config

    class DummySolver:
        def __init__(self, config):
            self.config = config

        def run(self, context):
            return "u_t = 0", 0.0

    aliases = info.get('aliases', {}) or {}
    problem_alias = aliases.get('sga_problem') or aliases.get('legacy')
    model = KD_SGA(sga_run=1, depth=1, width=1, num=1)
    result = model.fit_dataset(
        dataset,
        problem_name=problem_alias,
        context_cls=DummyContext,
        solver_cls=DummySolver,
    )
    assert result is model
    assert model.dataset_ is dataset


@pytest.mark.parametrize("dataset_name", ALL_DATASETS)
def test_kd_dscv_regular_imports_supported(dataset_name: str, monkeypatch):
    info = get_dataset_info(dataset_name)
    if not info.get('models', {}).get('dscv'):
        pytest.skip(f"dataset {dataset_name} is not supported by KD_DSCV")

    dataset = load_pde(dataset_name)
    if dataset.t.size > 1:
        dt = np.diff(dataset.t)
        if dt.size and not np.allclose(dt, dt[0], atol=1e-8):
            pytest.skip("Regular adapter requires uniform time step")

    model = KD_DSCV(n_iterations=1, n_samples_per_batch=10)
    monkeypatch.setattr(KD_DSCV, 'make_gp_aggregator', lambda self: None, raising=False)

    called = {}

    def fake_setup(self):
        called['setup'] = True

    monkeypatch.setattr(KD_DSCV, 'setup', fake_setup, raising=False)

    result = model.import_dataset(dataset)
    assert result is model
    assert called.get('setup') is True
    data = model.data_class.get_data()
    assert data['u'].shape == dataset.usol.shape


@pytest.mark.parametrize("dataset_name", ALL_DATASETS)
def test_kd_dscv_spr_sparse_imports_supported(dataset_name: str, monkeypatch):
    info = get_dataset_info(dataset_name)
    if not info.get('models', {}).get('dscv_spr'):
        pytest.skip(f"dataset {dataset_name} is not supported by KD_DSCV_SPR")

    dataset = load_pde(dataset_name)

    model = KD_DSCV_SPR(n_iterations=1, n_samples_per_batch=10)
    monkeypatch.setattr(KD_DSCV_SPR, 'make_gp_aggregator', lambda self: None, raising=False)

    def fake_setup(self):
        self.config_task.setdefault('eq_num', 1)

    monkeypatch.setattr(KD_DSCV_SPR, 'setup', fake_setup, raising=False)

    total_points = dataset.usol.size
    sample = max(1, min(64, total_points))
    colloc = max(sample, 64)

    result = model.import_dataset(
        dataset,
        sample=sample,
        colloc_num=colloc,
        random_state=0,
    )
    assert result is model

    sparse = model.data_class.get_data()
    assert sparse['X_u_train'].shape[1] == 2
    assert sparse['u_train'].shape[0] == sparse['X_u_train'].shape[0]
