"""Smoke tests mirroring the examples/ workflows.

These tests run real (but small) training loops to ensure every supported model
can consume its registered datasets through the unified ``load_pde`` pipeline.
"""

from __future__ import annotations

import pytest

pytest.importorskip('scipy', reason='smoke tests require scipy for .mat loading')

from kd.dataset import load_pde
from kd.dataset._registry import PDE_REGISTRY
from kd.model.kd_dlga import KD_DLGA
from kd.model.kd_sga import KD_SGA
from kd.model.kd_dscv import KD_DSCV, KD_DSCV_SPR

STATUS_ACTIVE = 'active'


def collect_datasets(model_key: str) -> list[tuple[str, dict]]:
    datasets: list[tuple[str, dict]] = []
    for name, info in PDE_REGISTRY.items():
        if info.get('status', STATUS_ACTIVE) != STATUS_ACTIVE:
            continue
        if info.get('models', {}).get(model_key):
            datasets.append((name, info))
    return datasets


DLGA_DATASETS = collect_datasets('dlga')
SGA_DATASETS = collect_datasets('sga')
DSCV_DATASETS = collect_datasets('dscv')
DSCV_SPR_DATASETS = collect_datasets('dscv_spr')


@pytest.mark.parametrize('dataset_name, info', DLGA_DATASETS)
def test_kd_dlga_smoke(dataset_name: str, info: dict):
    torch = pytest.importorskip('torch', reason='KD_DLGA depends on torch')
    dataset = load_pde(dataset_name)
    nx, nt = dataset.get_size()

    model = KD_DLGA(
        operators=['u', 'u_x', 'u_xx', 'u_xxx'],
        epi=0.1,
        input_dim=2,
        verbose=False,
        max_iter=200,
    )
    sample_count = min(256, nx * nt)
    result = model.fit_dataset(dataset, sample=sample_count)

    assert result is model
    # eq_latex 对新版 KD_DLGA 是主要出口；旧字段保作后备
    assert getattr(model, 'eq_latex', None) or getattr(model, 'best_equation_', None) or getattr(model, 'equations_', None)


@pytest.mark.parametrize('dataset_name, info', SGA_DATASETS)
def test_kd_sga_dataset_smoke(dataset_name: str, info: dict):
    dataset = load_pde(dataset_name)

    aliases = info.get('aliases', {}) or {}
    problem_name = aliases.get('sga_problem') or dataset_name

    model = KD_SGA(
        sga_run=1,
        num=8,
        depth=3,
        width=3,
        max_epoch=1000,
        seed=0,
    )
    result = model.fit_dataset(dataset, problem_name=problem_name)

    assert result is model
    assert getattr(model, 'best_pde_', None) is not None
    assert getattr(model, 'dataset_', None) is dataset


@pytest.mark.parametrize('dataset_name, info', DSCV_DATASETS)
def test_kd_dscv_smoke(dataset_name: str, info: dict, monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip('torch', reason='KD_DSCV depends on torch')
    dataset = load_pde(dataset_name)

    model = KD_DSCV(
        binary_operators=['add', 'mul', 'diff'],
        unary_operators=['n2'],
        n_iterations=5,
        n_samples_per_batch=500,
        seed=0,
    )
    monkeypatch.setattr(KD_DSCV, 'make_gp_aggregator', lambda self: None, raising=False)

    model.import_dataset(dataset)
    result = model.train(n_epochs=5, verbose=False)

    assert isinstance(result, dict)
    assert 'expression' in result
    assert getattr(model, 'dataset_', None) is dataset


@pytest.mark.slow
@pytest.mark.parametrize('dataset_name, info', DSCV_SPR_DATASETS)
def test_kd_dscv_spr_smoke(dataset_name: str, info: dict, monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip('torch', reason='KD_DSCV_SPR requires torch')
    pytest.importorskip('tensorflow', reason='Mode2 pipeline depends on TensorFlow')

    dataset = load_pde(dataset_name)

    model = KD_DSCV_SPR(
        n_iterations=1,
        n_samples_per_batch=5,
        seed=0,
    )
    monkeypatch.setattr(KD_DSCV_SPR, 'make_gp_aggregator', lambda self: None, raising=False)

    model.import_dataset(
        dataset,
        sample=64,
        colloc_num=64,
        random_state=0,
        sample_ratio=0.05,
        noise_level=0.0,
    )
    result = model.train(n_epochs=1, verbose=False)

    assert isinstance(result, dict)
    assert 'expression' in result
    data = model.data_class.get_data()
    assert data['X_u_train'].shape[0] > 0
