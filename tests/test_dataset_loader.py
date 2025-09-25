import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from kd.dataset import (
    PDEDataset,
    get_dataset_info,
    get_dataset_sym_true,
    list_available_datasets,
    load_pde,
)


def test_list_available_datasets_contains_core_names():
    datasets = list_available_datasets()
    assert 'kdv' in datasets
    assert 'burgers' in datasets
    assert 'chafee-infante' in datasets


def test_get_dataset_info_unknown_raises():
    with pytest.raises(ValueError):
        get_dataset_info('non-existent-dataset')


def test_load_pde_chafee_infante_uses_npy_bundle():
    dataset = load_pde('chafee-infante')
    assert isinstance(dataset, PDEDataset)

    data = dataset.get_data()
    assert data['usol'].ndim == 2
    assert data['usol'].shape == (len(dataset.x), len(dataset.t))
    assert np.allclose(data['usol'], dataset.usol)


def test_load_pde_divide_single_npy_builds_domain():
    dataset = load_pde('PDE_divide')
    info = get_dataset_info('PDE_divide')

    assert dataset.usol.shape == info['shape']

    boundaries = dataset.get_boundaries()
    assert boundaries['x'] == pytest.approx(info['domain']['x'])
    assert boundaries['t'] == pytest.approx(info['domain']['t'])


def test_load_pde_fisher_mat_file():
    pytest.importorskip('scipy')

    dataset = load_pde('fisher')
    assert dataset.usol.shape == (len(dataset.x), len(dataset.t))
    assert get_dataset_sym_true('fisher') is not None


def test_dataset_aliases_and_names():
    dataset = load_pde('burgers')
    assert getattr(dataset, 'registry_name') == 'burgers'
    assert dataset.aliases.get('legacy') == 'Burgers2'
    assert dataset.legacy_name == 'Burgers2'
