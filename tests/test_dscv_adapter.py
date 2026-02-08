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

    result = model.fit_dataset(dataset, n_epochs=1, verbose=False)

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


def test_kd_dscv_gp_aggregator_is_disabled_in_kd(monkeypatch):
    """run_gp_agg=True 时在 KD 中应被忽略，仅发出 warning 并返回 None。"""

    model = KD_DSCV(n_iterations=1, n_samples_per_batch=10)
    model.config_gp_agg['run_gp_agg'] = True

    with pytest.warns(RuntimeWarning):
        agg = model.make_gp_aggregator()

    assert agg is None


# ==========================================================================
# N-D Adapter Tests
# ==========================================================================


def _make_nd_2d_dataset():
    """Create a 2D spatial (x, y, t) PDEDataset with known analytic solution."""
    nx, ny, nt = 10, 12, 15
    x = np.linspace(0, np.pi, nx)
    y = np.linspace(0, np.pi, ny)
    t = np.linspace(0, 1.0, nt)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.exp(-T)
    return PDEDataset(
        equation_name="synthetic-2d",
        fields_data={"u": u},
        coords_1d={"x": x, "y": y, "t": t},
        axis_order=["x", "y", "t"],
        target_field="u",
        lhs_axis="t",
    )


def _make_nd_3d_dataset():
    """Create a 3D spatial (x, y, z, t) PDEDataset."""
    nx, ny, nz, nt = 6, 7, 5, 10
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    t = np.linspace(0, 0.5, nt)
    X, Y, Z, T = np.meshgrid(x, y, z, t, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.sin(Z) * np.exp(-T)
    return PDEDataset(
        equation_name="synthetic-3d",
        fields_data={"u": u},
        coords_1d={"x": x, "y": y, "z": z, "t": t},
        axis_order=["x", "y", "z", "t"],
        target_field="u",
        lhs_axis="t",
    )


class TestDSCVAdapterND2D:
    """Tests for 2D spatial (x, y, t) adapter path."""

    def test_shapes_and_fields(self):
        dataset = _make_nd_2d_dataset()
        adapter = DSCVRegularAdapter(dataset)
        data = adapter.get_data()

        assert set(data.keys()) >= {"u", "ut", "X", "n_input_dim"}

        # dso expects u.shape = (nt, nx, ny) for 2D spatial
        nx, ny, nt = 10, 12, 15
        assert data["u"].shape == (nt, nx, ny)
        assert data["ut"].shape == (nt, nx, ny)

        # X should have 2 spatial coord columns
        assert len(data["X"]) == 2
        assert data["X"][0].shape == (nx, 1)
        assert data["X"][1].shape == (ny, 1)
        assert data["n_input_dim"] == 2

    def test_ut_numerical_accuracy(self):
        """Verify ut matches analytic derivative for u = sin(x)*cos(y)*exp(-t)."""
        dataset = _make_nd_2d_dataset()
        adapter = DSCVRegularAdapter(dataset)
        data = adapter.get_data()

        # Analytic: du/dt = -sin(x)*cos(y)*exp(-t)
        nx, ny, nt = 10, 12, 15
        x = np.linspace(0, np.pi, nx)
        y = np.linspace(0, np.pi, ny)
        t = np.linspace(0, 1.0, nt)
        X, Y, T = np.meshgrid(x, y, t, indexing="ij")
        ut_analytic_raw = -np.sin(X) * np.cos(Y) * np.exp(-T)
        # After permute: (nt, nx, ny)
        ut_analytic = np.transpose(ut_analytic_raw, (2, 0, 1))

        # FD approximation should be close to analytic (not exact due to FD error)
        np.testing.assert_allclose(data["ut"], ut_analytic, atol=0.05)


class TestDSCVAdapterND3D:
    """Tests for 3D spatial (x, y, z, t) adapter path."""

    def test_shapes_and_fields(self):
        dataset = _make_nd_3d_dataset()
        adapter = DSCVRegularAdapter(dataset)
        data = adapter.get_data()

        assert set(data.keys()) >= {"u", "ut", "X", "n_input_dim"}

        nx, ny, nz, nt = 6, 7, 5, 10
        # dso expects u.shape = (nt, nx, ny, nz) for 3D spatial
        assert data["u"].shape == (nt, nx, ny, nz)
        assert data["ut"].shape == (nt, nx, ny, nz)

        assert len(data["X"]) == 3
        assert data["X"][0].shape == (nx, 1)
        assert data["X"][1].shape == (ny, 1)
        assert data["X"][2].shape == (nz, 1)
        assert data["n_input_dim"] == 3


class TestDSCVAdapterLegacyUnchanged:
    """Confirm legacy 1D path is not affected by N-D changes."""

    def test_legacy_adapter_identical_output(self):
        dataset = load_pde("chafee-infante")
        adapter = DSCVRegularAdapter(dataset)
        data = adapter.get_data()

        # Same assertions as the original test
        assert data["u"].shape == dataset.usol.shape
        assert data["ut"].shape == dataset.usol.shape
        assert len(data["X"]) == 1
        assert data["X"][0].shape == (dataset.usol.shape[0], 1)
        assert data["n_input_dim"] == 1


class TestDSCVAdapterNDPermutation:
    """Verify correct axis permutation for non-standard axis_order."""

    def test_txy_order(self):
        """axis_order=["t","x","y"] should still produce (nt, nx, ny) output."""
        nx, ny, nt = 8, 10, 12
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        t = np.linspace(0, 0.5, nt)
        # axis_order is ["t", "x", "y"], so u.shape = (nt, nx, ny)
        T, X, Y = np.meshgrid(t, x, y, indexing="ij")
        u = np.sin(X) * np.cos(Y) * np.exp(-T)

        dataset = PDEDataset(
            equation_name="synthetic-txy",
            fields_data={"u": u},
            coords_1d={"t": t, "x": x, "y": y},
            axis_order=["t", "x", "y"],
            target_field="u",
            lhs_axis="t",
        )

        adapter = DSCVRegularAdapter(dataset)
        data = adapter.get_data()

        # Output should be (nt, nx, ny) — time first, same as input in this case
        assert data["u"].shape == (nt, nx, ny)
        assert data["n_input_dim"] == 2
        assert len(data["X"]) == 2

    def test_yxt_order(self):
        """axis_order=["y","x","t"] should permute to (nt, y, x) → (nt, nx_y, nx_x)."""
        nx, ny, nt = 8, 10, 12
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        t = np.linspace(0, 0.5, nt)
        # axis_order is ["y", "x", "t"], so u.shape = (ny, nx, nt)
        Y, X, T = np.meshgrid(y, x, t, indexing="ij")
        u = np.sin(X) * np.cos(Y) * np.exp(-T)

        dataset = PDEDataset(
            equation_name="synthetic-yxt",
            fields_data={"u": u},
            coords_1d={"y": y, "x": x, "t": t},
            axis_order=["y", "x", "t"],
            target_field="u",
            lhs_axis="t",
        )

        adapter = DSCVRegularAdapter(dataset)
        data = adapter.get_data()

        # Should be permuted to (nt, ny, nx) — time first, then spatial in order
        assert data["u"].shape == (nt, ny, nx)
        assert data["n_input_dim"] == 2
        # X list follows spatial_axes order: ["y", "x"]
        assert data["X"][0].shape == (ny, 1)
        assert data["X"][1].shape == (nx, 1)


# ==========================================================================
# function_set Auto-Selection Tests
# ==========================================================================


class TestAutoFunctionSet:
    """Tests for automatic function_set selection based on n_input_dim."""

    def test_auto_function_set_2d(self, monkeypatch):
        """2D data should auto-select Diff/Diff2/lap (not 1D diff tokens)."""
        dataset = _make_nd_2d_dataset()
        model = KD_DSCV(n_iterations=1, n_samples_per_batch=10)

        captured = {}

        original_set_task = KD_DSCV.set_task

        def capture_set_task(self):
            original_set_task(self)
            captured["function_set"] = list(self.config_task["function_set"])

        monkeypatch.setattr(KD_DSCV, "set_task", capture_set_task)
        monkeypatch.setattr(KD_DSCV, "make_gp_aggregator", lambda self: None)

        def fake_search(self, n_epochs=1, verbose=True, keep_history=True):
            return {"program": None, "expression": "u_t = 0", "r": 0.0}

        monkeypatch.setattr(Searcher, "search", fake_search, raising=False)

        model.import_dataset(dataset)

        fs = captured["function_set"]
        assert "Diff" in fs
        assert "Diff2" in fs
        assert "lap" in fs
        # 1D tokens should not be present
        assert "diff" not in fs
        assert "diff2" not in fs
        assert "diff3" not in fs
        # Non-diff tokens preserved
        assert "add" in fs
        assert "mul" in fs

    def test_auto_function_set_3d(self, monkeypatch):
        """3D data should auto-select Diff_3/Diff2_3."""
        dataset = _make_nd_3d_dataset()
        model = KD_DSCV(n_iterations=1, n_samples_per_batch=10)

        captured = {}

        original_set_task = KD_DSCV.set_task

        def capture_set_task(self):
            original_set_task(self)
            captured["function_set"] = list(self.config_task["function_set"])

        monkeypatch.setattr(KD_DSCV, "set_task", capture_set_task)
        monkeypatch.setattr(KD_DSCV, "make_gp_aggregator", lambda self: None)

        def fake_search(self, n_epochs=1, verbose=True, keep_history=True):
            return {"program": None, "expression": "u_t = 0", "r": 0.0}

        monkeypatch.setattr(Searcher, "search", fake_search, raising=False)

        model.import_dataset(dataset)

        fs = captured["function_set"]
        assert "Diff_3" in fs
        assert "Diff2_3" in fs
        assert "diff" not in fs

    def test_user_operator_overrides(self, monkeypatch):
        """User-specified operator should not be overridden by auto-selection."""
        dataset = _make_nd_2d_dataset()
        model = KD_DSCV(
            n_iterations=1,
            n_samples_per_batch=10,
            binary_operators=["add"],
            unary_operators=["Diff"],
        )

        captured = {}

        original_set_task = KD_DSCV.set_task

        def capture_set_task(self):
            original_set_task(self)
            captured["function_set"] = list(self.config_task["function_set"])

        monkeypatch.setattr(KD_DSCV, "set_task", capture_set_task)
        monkeypatch.setattr(KD_DSCV, "make_gp_aggregator", lambda self: None)

        def fake_search(self, n_epochs=1, verbose=True, keep_history=True):
            return {"program": None, "expression": "u_t = 0", "r": 0.0}

        monkeypatch.setattr(Searcher, "search", fake_search, raising=False)

        model.import_dataset(dataset)

        fs = captured["function_set"]
        assert fs == ["add", "Diff"]

    def test_1d_function_set_unchanged(self, monkeypatch):
        """1D data should keep the default function_set from config."""
        dataset = load_pde("chafee-infante")
        model = KD_DSCV(n_iterations=1, n_samples_per_batch=10)

        captured = {}

        original_set_task = KD_DSCV.set_task

        def capture_set_task(self):
            original_set_task(self)
            captured["function_set"] = list(self.config_task["function_set"])

        monkeypatch.setattr(KD_DSCV, "set_task", capture_set_task)
        monkeypatch.setattr(KD_DSCV, "make_gp_aggregator", lambda self: None)

        def fake_search(self, n_epochs=1, verbose=True, keep_history=True):
            return {"program": None, "expression": "u_t = 0", "r": 0.0}

        monkeypatch.setattr(Searcher, "search", fake_search, raising=False)

        model.import_dataset(dataset)

        fs = captured["function_set"]
        # Default config: ["add", "mul", "div", "diff","diff2", "diff3","n2","n3"]
        assert "diff" in fs
        assert "diff2" in fs
        assert "diff3" in fs

    def test_reimport_resets_function_set(self, monkeypatch):
        """Reusing a KD_DSCV instance across 2D then 1D datasets must reset tokens."""
        ds_2d = _make_nd_2d_dataset()
        ds_1d = load_pde("chafee-infante")
        model = KD_DSCV(n_iterations=1, n_samples_per_batch=10)

        captured = {}

        original_set_task = KD_DSCV.set_task

        def capture_set_task(self):
            original_set_task(self)
            captured["function_set"] = list(self.config_task["function_set"])

        monkeypatch.setattr(KD_DSCV, "set_task", capture_set_task)
        monkeypatch.setattr(KD_DSCV, "make_gp_aggregator", lambda self: None)

        def fake_search(self, n_epochs=1, verbose=True, keep_history=True):
            return {"program": None, "expression": "u_t = 0", "r": 0.0}

        monkeypatch.setattr(Searcher, "search", fake_search, raising=False)

        # First import: 2D → should get Diff/Diff2/lap
        model.import_dataset(ds_2d)
        fs_2d = captured["function_set"]
        assert "Diff" in fs_2d
        assert "diff" not in fs_2d

        # Second import: 1D → must reset back to 1D diff tokens
        model.import_dataset(ds_1d)
        fs_1d = captured["function_set"]
        assert "diff" in fs_1d
        assert "diff2" in fs_1d
        assert "Diff" not in fs_1d
        assert "Diff2" not in fs_1d


# ==========================================================================
# Edge Case Tests (from Codex review)
# ==========================================================================


class TestDSCVAdapterEdgeCases:
    """Edge cases identified by Codex reviews."""

    def test_1d_via_nd_path_rejected(self):
        """1D dataset constructed in N-D mode should be rejected by _prepare_nd()."""
        nx, nt = 10, 15
        x = np.linspace(0, 1, nx)
        t = np.linspace(0, 0.5, nt)
        X, T = np.meshgrid(x, t, indexing="ij")
        u = np.sin(X) * np.exp(-T)

        dataset = PDEDataset(
            equation_name="synthetic-1d-nd",
            fields_data={"u": u},
            coords_1d={"x": x, "t": t},
            axis_order=["x", "t"],
            target_field="u",
            lhs_axis="t",
        )

        with pytest.raises(ValueError, match="2-3 spatial dimensions"):
            DSCVRegularAdapter(dataset)

    def test_nd_minimum_time_points(self):
        """N-D adapter requires >= 3 time samples for FiniteDiff."""
        nx, ny, nt = 8, 10, 2  # only 2 time points
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        t = np.linspace(0, 0.5, nt)
        X, Y, T = np.meshgrid(x, y, t, indexing="ij")
        u = np.sin(X) * np.cos(Y) * np.exp(-T)

        dataset = PDEDataset(
            equation_name="synthetic-2d-short",
            fields_data={"u": u},
            coords_1d={"x": x, "y": y, "t": t},
            axis_order=["x", "y", "t"],
            target_field="u",
            lhs_axis="t",
        )

        with pytest.raises(ValueError, match="at least 3 samples"):
            DSCVRegularAdapter(dataset)

    def test_legacy_minimum_time_points(self):
        """Legacy adapter requires >= 3 time samples for FiniteDiff."""
        x = np.linspace(0, 1, 10)
        t = np.array([0.0, 0.5])  # only 2 time points
        X, T = np.meshgrid(x, t, indexing="ij")
        u = np.sin(X) * np.exp(-T)

        dataset = PDEDataset(
            equation_name="synthetic-1d-short",
            pde_data=None,
            domain={"x": (0.0, 1.0), "t": (0.0, 0.5)},
            epi=1e-3,
            x=x,
            t=t,
            usol=u,
        )

        with pytest.raises(ValueError, match="at least 3 temporal samples"):
            DSCVRegularAdapter(dataset)

    def test_nd_non_uniform_dt_rejected(self):
        """Non-uniform time grid should be rejected in N-D mode."""
        nx, ny = 8, 10
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        t = np.array([0.0, 0.1, 0.3, 0.7, 1.0])  # non-uniform
        X, Y, T = np.meshgrid(x, y, t, indexing="ij")
        u = np.sin(X) * np.cos(Y) * np.exp(-T)

        dataset = PDEDataset(
            equation_name="synthetic-2d-nu",
            fields_data={"u": u},
            coords_1d={"x": x, "y": y, "t": t},
            axis_order=["x", "y", "t"],
            target_field="u",
            lhs_axis="t",
        )

        with pytest.raises(ValueError, match="evenly spaced"):
            DSCVRegularAdapter(dataset)

    def test_lhs_axis_missing_rejected(self):
        """Missing lhs_axis in axis_order should raise ValueError at dataset creation."""
        nx, ny, nt = 8, 10, 12
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        t = np.linspace(0, 0.5, nt)
        X, Y, T = np.meshgrid(x, y, t, indexing="ij")
        u = np.sin(X) * np.cos(Y) * np.exp(-T)

        # PDEDataset validates lhs_axis at construction time
        with pytest.raises(ValueError, match="lhs_axis.*must be in axis_order"):
            PDEDataset(
                equation_name="synthetic-2d-bad-lhs",
                fields_data={"u": u},
                coords_1d={"x": x, "y": y, "t": t},
                axis_order=["x", "y", "t"],
                target_field="u",
                lhs_axis="z",
            )

    def test_too_many_spatial_dims_rejected(self):
        """More than 3 spatial dimensions should be rejected."""
        n = 4
        axes = ["x", "y", "z", "w", "t"]
        coords = {a: np.linspace(0, 1, n) for a in axes}
        grids = np.meshgrid(*[coords[a] for a in axes], indexing="ij")
        u = grids[0] + grids[1]  # arbitrary

        dataset = PDEDataset(
            equation_name="synthetic-4d",
            fields_data={"u": u},
            coords_1d=coords,
            axis_order=axes,
            target_field="u",
            lhs_axis="t",
        )

        with pytest.raises(ValueError, match="2-3 spatial dimensions"):
            DSCVRegularAdapter(dataset)

    def test_legacy_nan_input_rejected(self):
        """Legacy path should reject u containing NaN."""
        x = np.linspace(0, 1, 10)
        t = np.linspace(0, 0.5, 15)
        X, T = np.meshgrid(x, t, indexing="ij")
        u = np.sin(X) * np.exp(-T)
        u[3, 5] = np.nan

        dataset = PDEDataset(
            equation_name="synthetic-nan",
            pde_data=None,
            domain={"x": (0.0, 1.0), "t": (0.0, 0.5)},
            epi=1e-3,
            x=x,
            t=t,
            usol=u,
        )

        with pytest.raises(ValueError, match="NaN or Inf"):
            DSCVRegularAdapter(dataset)

    def test_nd_nan_input_rejected(self):
        """N-D path should reject u containing NaN."""
        nx, ny, nt = 8, 10, 12
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        t = np.linspace(0, 0.5, nt)
        X, Y, T = np.meshgrid(x, y, t, indexing="ij")
        u = np.sin(X) * np.cos(Y) * np.exp(-T)
        u[2, 3, 4] = np.inf

        dataset = PDEDataset(
            equation_name="synthetic-2d-inf",
            fields_data={"u": u},
            coords_1d={"x": x, "y": y, "t": t},
            axis_order=["x", "y", "t"],
            target_field="u",
            lhs_axis="t",
        )

        with pytest.raises(ValueError, match="NaN or Inf"):
            DSCVRegularAdapter(dataset)

    def test_legacy_zero_delta_t_rejected(self):
        """Legacy path should reject degenerate time grid (delta_t == 0)."""
        x = np.linspace(0, 1, 10)
        t = np.array([0.0, 0.0, 0.0])  # all identical timestamps
        X, T = np.meshgrid(x, t, indexing="ij")
        u = np.sin(X) * np.ones_like(T)

        dataset = PDEDataset(
            equation_name="synthetic-zero-dt",
            pde_data=None,
            domain={"x": (0.0, 1.0), "t": (0.0, 0.0)},
            epi=1e-3,
            x=x,
            t=t,
            usol=u,
        )

        with pytest.raises(ValueError, match="delta_t is zero"):
            DSCVRegularAdapter(dataset)

    def test_nd_zero_delta_t_rejected(self):
        """N-D path should reject degenerate time grid (delta_t == 0)."""
        nx, ny = 8, 10
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        t = np.array([1.0, 1.0, 1.0])  # all identical
        X, Y, T = np.meshgrid(x, y, t, indexing="ij")
        u = np.sin(X) * np.cos(Y) * np.ones_like(T)

        dataset = PDEDataset(
            equation_name="synthetic-2d-zero-dt",
            fields_data={"u": u},
            coords_1d={"x": x, "y": y, "t": t},
            axis_order=["x", "y", "t"],
            target_field="u",
            lhs_axis="t",
        )

        with pytest.raises(ValueError, match="zero"):
            DSCVRegularAdapter(dataset)

    def test_xty_permutation(self):
        """axis_order=["x","t","y"] — t in the middle — should still work."""
        nx, ny, nt = 8, 10, 12
        x = np.linspace(0, 1, nx)
        t = np.linspace(0, 0.5, nt)
        y = np.linspace(0, 1, ny)
        # axis_order is ["x", "t", "y"], so u.shape = (nx, nt, ny)
        X, T, Y = np.meshgrid(x, t, y, indexing="ij")
        u = np.sin(X) * np.cos(Y) * np.exp(-T)

        dataset = PDEDataset(
            equation_name="synthetic-xty",
            fields_data={"u": u},
            coords_1d={"x": x, "t": t, "y": y},
            axis_order=["x", "t", "y"],
            target_field="u",
            lhs_axis="t",
        )

        adapter = DSCVRegularAdapter(dataset)
        data = adapter.get_data()

        # Should be permuted to (nt, nx, ny)
        assert data["u"].shape == (nt, nx, ny)
        assert data["ut"].shape == (nt, nx, ny)
        assert data["n_input_dim"] == 2
        assert len(data["X"]) == 2
        assert data["X"][0].shape == (nx, 1)
        assert data["X"][1].shape == (ny, 1)

    def test_nd_spatial_axis_too_few_points_rejected(self):
        """Spatial axis with < 4 points should be rejected (Diff2 stencil needs >=4)."""
        nx, ny, nt = 3, 10, 12  # nx=3 < _MIN_SPATIAL_POINTS=4
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        t = np.linspace(0, 0.5, nt)
        X, Y, T = np.meshgrid(x, y, t, indexing="ij")
        u = np.sin(X) * np.cos(Y) * np.exp(-T)

        dataset = PDEDataset(
            equation_name="synthetic-2d-few-x",
            fields_data={"u": u},
            coords_1d={"x": x, "y": y, "t": t},
            axis_order=["x", "y", "t"],
            target_field="u",
            lhs_axis="t",
        )

        with pytest.raises(ValueError, match="at least 4"):
            DSCVRegularAdapter(dataset)

    def test_partial_nd_data_rejected(self):
        """Having fields_data but no coords_1d should raise ValueError."""
        nx, ny, nt = 8, 10, 12
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        t = np.linspace(0, 0.5, nt)
        X, Y, T = np.meshgrid(x, y, t, indexing="ij")
        u = np.sin(X) * np.cos(Y) * np.exp(-T)

        dataset = PDEDataset.__new__(PDEDataset)
        dataset.fields_data = {"u": u}
        dataset.coords_1d = None

        with pytest.raises(ValueError, match="both fields_data and coords_1d"):
            DSCVRegularAdapter(dataset)
