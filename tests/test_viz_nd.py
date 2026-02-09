"""N-D visualization tests for viz-nd Phase 1."""

from typing import Optional

import numpy as np
import pytest

import kd.viz.dscv_viz as dscv_viz_module
from kd.viz._contracts import FieldComparisonData, TimeSliceComparisonData
from kd.viz import core as viz_core
from kd.viz import registry as viz_registry
from kd.viz.adapters import DSCVVizAdapter, SGAVizAdapter
from kd.viz.discover_eq2latex import discover_program_to_latex
from kd.model.sga.sgapde.equation import SGAEquationDetails, SGAEquationTerm


# ===== Fixtures =====

@pytest.fixture(autouse=True)
def suppress_show(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: None)


def setup_function():
    viz_registry.clear_registry()


def teardown_function():
    viz_registry.clear_registry()


# ===== Contract tests: FieldComparisonData =====

class TestFieldComparisonDataContracts:
    def test_1d_backward_compat_x_coords(self):
        """Old x_coords= keyword still works."""
        x = np.linspace(0, 1, 4)
        t = np.linspace(0, 1, 3)
        field = np.arange(12).reshape(4, 3).astype(float)

        data = FieldComparisonData(
            x_coords=x, t_coords=t,
            true_field=field, predicted_field=field * 0.9,
        )
        assert data.n_spatial_dims == 1
        np.testing.assert_array_equal(data.x_coords, x)
        np.testing.assert_array_equal(data.spatial_coords[0], x)
        assert data.true_field.shape == (4, 3)

    def test_1d_positional(self):
        """Positional spatial_coords= with a single ndarray auto-wraps."""
        x = np.linspace(0, 1, 4)
        t = np.linspace(0, 1, 3)
        field = np.arange(12).reshape(4, 3).astype(float)

        data = FieldComparisonData(x, t, field, field * 0.9)
        assert data.n_spatial_dims == 1
        np.testing.assert_array_equal(data.x_coords, x)

    def test_2d_spatial(self):
        """2D spatial with spatial_coords=[x, y]."""
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 4)
        t = np.linspace(0, 1, 5)
        field = np.arange(60).reshape(3, 4, 5).astype(float)

        data = FieldComparisonData(
            spatial_coords=[x, y], t_coords=t,
            true_field=field, predicted_field=field * 0.9,
        )
        assert data.n_spatial_dims == 2
        assert data.true_field.shape == (3, 4, 5)
        np.testing.assert_array_equal(data.x_coords, x)

    def test_shape_mismatch_raises(self):
        """Wrong shape raises ValueError."""
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 4)
        t = np.linspace(0, 1, 5)
        wrong_field = np.zeros((5, 4, 3))  # wrong order

        with pytest.raises(ValueError, match="Field data must have shape"):
            FieldComparisonData(
                spatial_coords=[x, y], t_coords=t,
                true_field=wrong_field, predicted_field=wrong_field,
            )

    def test_residual_auto_computed(self):
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 4)
        t = np.linspace(0, 1, 5)
        true_f = np.ones((3, 4, 5))
        pred_f = np.ones((3, 4, 5)) * 0.5

        data = FieldComparisonData(
            spatial_coords=[x, y], t_coords=t,
            true_field=true_f, predicted_field=pred_f,
        )
        np.testing.assert_allclose(data.residual_field, 0.5)


# ===== Contract tests: TimeSliceComparisonData =====

class TestTimeSliceComparisonDataContracts:
    def test_2d_spatial(self):
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 4)
        t = np.linspace(0, 1, 5)
        field = np.arange(60).reshape(3, 4, 5).astype(float)

        data = TimeSliceComparisonData(
            spatial_coords=[x, y], t_coords=t,
            true_field=field, predicted_field=field * 0.9,
            slice_times=np.array([0.0, 0.5, 1.0]),
        )
        assert data.n_spatial_dims == 2
        np.testing.assert_array_equal(data.x_coords, x)

    def test_1d_backward_compat(self):
        x = np.linspace(0, 1, 4)
        t = np.linspace(0, 1, 3)
        field = np.arange(12).reshape(4, 3).astype(float)

        data = TimeSliceComparisonData(
            x_coords=x, t_coords=t,
            true_field=field, predicted_field=field,
            slice_times=np.array([0.0, 0.5]),
        )
        assert data.n_spatial_dims == 1


# ===== LaTeX tests =====

class TestLatexND:
    def test_to_sympy_string_Diff(self):
        """Diff(u1, x2) → Derivative(u1, x2)."""
        from kd.model.discover.stridge import Node

        class FakeToken:
            def __init__(self, name, arity=0):
                self.name = name
                self.arity = arity

        diff_node = Node(FakeToken('Diff', 2))
        u_node = Node(FakeToken('u1', 0))
        x2_node = Node(FakeToken('x2', 0))
        diff_node.children = [u_node, x2_node]

        result = diff_node.to_sympy_string()
        assert result == "Derivative(u1, x2)"

    def test_to_sympy_string_Diff2(self):
        from kd.model.discover.stridge import Node

        class FakeToken:
            def __init__(self, name, arity=0):
                self.name = name
                self.arity = arity

        diff2_node = Node(FakeToken('Diff2', 2))
        u_node = Node(FakeToken('u1', 0))
        x1_node = Node(FakeToken('x1', 0))
        diff2_node.children = [u_node, x1_node]

        result = diff2_node.to_sympy_string()
        assert result == "Derivative(u1, x1, x1)"

    def test_to_sympy_string_lap(self):
        from kd.model.discover.stridge import Node

        class FakeToken:
            def __init__(self, name, arity=0):
                self.name = name
                self.arity = arity

        lap_node = Node(FakeToken('lap', 1))
        u_node = Node(FakeToken('u1', 0))
        lap_node.children = [u_node]

        result = lap_node.to_sympy_string()
        assert result == "laplacian(u1)"

    def test_discover_program_to_latex_nd_lhs(self):
        """Dynamic LHS: lhs_axis='y' → u_{y}."""
        # Minimal mock program — invalid structure should trigger error path
        result = discover_program_to_latex(None, lhs_axis='y')
        assert "u_{y}" in result


# ===== DSCV computation tests =====

class TestDSCVComputation:
    def test_finite_difference_nd(self):
        """_finite_difference_nd computes derivative along specified axis."""
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.linspace(0, 2 * np.pi, 80)
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = np.sin(X) * np.cos(Y)  # shape (100, 80)

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # df/dx = cos(X) * cos(Y)
        df_dx = dscv_viz_module._finite_difference_nd(f, dx, order=1, axis=0)
        expected = np.cos(X) * np.cos(Y)
        np.testing.assert_allclose(df_dx[5:-5, 5:-5], expected[5:-5, 5:-5], atol=0.01)

        # df/dy = -sin(X) * sin(Y)
        df_dy = dscv_viz_module._finite_difference_nd(f, dy, order=1, axis=1)
        expected_y = -np.sin(X) * np.sin(Y)
        np.testing.assert_allclose(df_dy[5:-5, 5:-5], expected_y[5:-5, 5:-5], atol=0.01)

    def test_evaluate_term_Diff_x2(self):
        """_evaluate_term_recursively handles Diff(u1, x2) in 2D."""
        from kd.model.discover.stridge import Node

        class FakeToken:
            def __init__(self, name, arity=0):
                self.name = name
                self.arity = arity

        # Build Diff(u1, x2)
        diff_node = Node(FakeToken('Diff', 2))
        u_node = Node(FakeToken('u1', 0))
        x2_node = Node(FakeToken('x2', 0))
        diff_node.children = [u_node, x2_node]

        # 2D spatial: u = x1 * x2, so du/dx2 = x1
        nx, ny = 20, 25
        x1 = np.linspace(0, 1, nx)
        x2 = np.linspace(0, 1, ny)
        X1, X2 = np.meshgrid(x1, x2, indexing='ij')
        u_snapshot = X1 * X2  # shape (20, 25)

        result = dscv_viz_module._evaluate_term_recursively(
            diff_node, u_snapshot, x1,
            spatial_coords_list=[x1, x2],
            spatial_dxs=[x1[1] - x1[0], x2[1] - x2[0]],
        )
        # du/dx2 ≈ x1 (broadcast to (20, 25))
        expected = np.broadcast_to(x1.reshape(-1, 1), (nx, ny))
        np.testing.assert_allclose(result[2:-2, 2:-2], expected[2:-2, 2:-2], atol=0.05)

    def test_evaluate_term_lap(self):
        """_evaluate_term_recursively handles lap(u1) in 2D."""
        from kd.model.discover.stridge import Node

        class FakeToken:
            def __init__(self, name, arity=0):
                self.name = name
                self.arity = arity

        lap_node = Node(FakeToken('lap', 1))
        u_node = Node(FakeToken('u1', 0))
        lap_node.children = [u_node]

        # u = x1^2 + x2^2, laplacian = 2 + 2 = 4
        nx, ny = 30, 30
        x1 = np.linspace(0, 1, nx)
        x2 = np.linspace(0, 1, ny)
        X1, X2 = np.meshgrid(x1, x2, indexing='ij')
        u_snapshot = X1**2 + X2**2

        result = dscv_viz_module._evaluate_term_recursively(
            lap_node, u_snapshot, x1,
            spatial_coords_list=[x1, x2],
            spatial_dxs=[x1[1] - x1[0], x2[1] - x2[0]],
        )
        # Interior should be ≈ 4.0
        np.testing.assert_allclose(result[3:-3, 3:-3], 4.0, atol=0.1)

    def test_calculate_pde_fields_nd(self, monkeypatch):
        """_calculate_pde_fields works for n_input_dim=2."""
        from kd.model.discover.stridge import Node

        class FakeToken:
            def __init__(self, name, arity=0):
                self.name = name
                self.arity = arity

        # Build a simple term: u1
        u_node = Node(FakeToken('u1', 0))

        nt, nx, ny = 5, 8, 10
        x1 = np.linspace(0, 1, nx)
        x2 = np.linspace(0, 1, ny)

        u_data = np.random.randn(nt, nx, ny)
        ut_data = np.random.randn(nt, nx, ny)

        data_dict = {
            'u': u_data,
            'ut': ut_data,
            'X': [
                np.tile(x1.reshape(-1, 1), (1, ny)),  # x1 grid
                np.tile(x2.reshape(1, -1), (nx, 1)),  # x2 grid
            ],
            'n_input_dim': 2,
        }

        class FakeProgram:
            class STRidge:
                terms = [u_node]
            w = np.array([1.0])

        class FakeModel:
            class data_class:
                @staticmethod
                def get_data():
                    return data_dict

        result = dscv_viz_module._calculate_pde_fields(FakeModel(), FakeProgram())
        assert result['n_spatial_dims'] == 2
        assert result['y_hat_grid'].shape == u_data.shape
        assert len(result['spatial_coords_list']) == 2
        np.testing.assert_array_equal(result['spatial_coords_list'][0], x1)
        np.testing.assert_array_equal(result['spatial_coords_list'][1], x2)


# ===== SGA Adapter tests =====

class DummySGAContext3D:
    """3D context with axis_order=['x', 'y', 't']."""

    def __init__(self):
        nx, ny, nt = 3, 4, 5
        self.axis_order = ['x', 'y', 't']
        self.lhs_axis = 't'

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        t = np.linspace(0, 1, nt)
        self.coords_1d = {'x': x, 'y': y, 't': t}

        # Fields in axis_order layout: (x, y, t)
        self.u = np.arange(nx * ny * nt).reshape(nx, ny, nt).astype(float)
        self.u_origin = self.u + 0.1
        self.ut = self.u * 0.5
        self.ut_origin = self.u_origin * 0.5
        self.right_side_full = self.ut - 0.02
        self.right_side_full_origin = self.ut_origin - 0.01
        self.default_terms = self.u.reshape(-1, 1)
        self.default_names = ['u']
        self.num_default = 1


class StubSGA3D:
    def __init__(self, latex: str = 'u_t = u_{xx} + u_{yy}', structure: Optional[str] = None):
        self._latex = latex
        self._structure = structure or latex
        self.context_ = DummySGAContext3D()
        self.best_equation_details_ = SGAEquationDetails(
            lhs='u_t',
            terms=[SGAEquationTerm(label='u', source='default', coefficient=1.0, tree=None)],
            predicted_rhs=self.context_.ut.copy(),
        )

    def equation_latex(self, *, include_coefficients: bool = True):
        return self._latex if include_coefficients else self._structure


class TestSGAAdapterND:
    def test_field_comparison_nd(self, tmp_path):
        """2D spatial field comparison doesn't silently reject."""
        adapter = SGAVizAdapter()
        viz_registry.register_adapter(StubSGA3D, adapter)

        model = StubSGA3D()
        request = viz_core.VizRequest(
            kind='field_comparison',
            target=model,
            options={'output_dir': tmp_path},
        )
        result = viz_core.render(request)

        path = tmp_path / 'sga' / 'field_comparison.png'
        assert path.exists(), f"Expected file at {path}, warnings={result.warnings}"
        assert result.paths == [path]

        field_data = result.metadata['field_comparison_data']
        assert isinstance(field_data, FieldComparisonData)
        assert field_data.n_spatial_dims == 2
        # Shape should be (nx, ny, nt) = (3, 4, 5)
        assert field_data.true_field.shape == (3, 4, 5)

    def test_parity_nd(self, tmp_path):
        """Parity plot works for N-D context."""
        adapter = SGAVizAdapter()
        viz_registry.register_adapter(StubSGA3D, adapter)

        model = StubSGA3D()
        request = viz_core.VizRequest(
            kind='parity',
            target=model,
            options={'output_dir': tmp_path},
        )
        result = viz_core.render(request)

        path = tmp_path / 'sga' / 'parity_plot.png'
        assert path.exists()

    def test_residual_nd(self, tmp_path):
        """Residual plot works for 3D+ arrays."""
        adapter = SGAVizAdapter()
        viz_registry.register_adapter(StubSGA3D, adapter)

        model = StubSGA3D()
        request = viz_core.VizRequest(
            kind='residual',
            target=model,
            options={'output_dir': tmp_path, 'bins': 10},
        )
        result = viz_core.render(request)

        path = tmp_path / 'sga' / 'residual_analysis.png'
        assert path.exists()
        assert result.metadata['residual_data'].actual.size == model.context_.ut_origin.size


# ===== DSCV Adapter N-D tests =====

class TestDSCVAdapterND:
    def test_field_comparison_nd(self, tmp_path, monkeypatch):
        """DSCV field comparison works with 2D spatial data."""
        adapter = DSCVVizAdapter()

        class StubDSCV:
            def __init__(self):
                self.best_p = type('P', (), {'traversal': []})()
                self.searcher = type('S', (), {'r_train': [], 'best_p': self.best_p})()

        viz_registry.register_adapter(StubDSCV, adapter)
        model = StubDSCV()

        nx, ny, nt = 5, 6, 4
        x1 = np.linspace(0, 1, nx)
        x2 = np.linspace(0, 1, ny)

        def fake_fields(_model, _program):
            ut = np.arange(nx * ny * nt).reshape(nt, nx, ny).astype(float)
            return {
                'x_axis': x1,
                't_axis': np.arange(nt),
                'ut_grid': ut,
                'y_hat_grid': ut * 0.9,
                'n_spatial_dims': 2,
                'spatial_coords_list': [x1, x2],
            }

        monkeypatch.setattr(dscv_viz_module, '_calculate_pde_fields', fake_fields)

        request = viz_core.VizRequest(
            kind='field_comparison',
            target=model,
            options={'output_dir': tmp_path},
        )
        result = viz_core.render(request)
        assert result.paths
        assert (tmp_path / 'dscv' / 'field_comparison.png').exists()

    def test_residual_nd(self, tmp_path, monkeypatch):
        """DSCV residual works with 2D spatial data."""
        adapter = DSCVVizAdapter()

        class StubDSCV:
            def __init__(self):
                self.best_p = type('P', (), {'traversal': []})()
                self.searcher = type('S', (), {'r_train': [], 'best_p': self.best_p})()

        viz_registry.register_adapter(StubDSCV, adapter)
        model = StubDSCV()

        nx, ny, nt = 5, 6, 4
        x1 = np.linspace(0, 1, nx)
        x2 = np.linspace(0, 1, ny)

        def fake_fields(_model, _program):
            ut = np.arange(nx * ny * nt).reshape(nt, nx, ny).astype(float)
            yhat = ut * 0.9
            return {
                'x_axis': x1,
                't_axis': np.arange(nt),
                'ut_grid': ut,
                'y_hat_grid': yhat,
                'residual': (ut - yhat).flatten(),
                'coords': None,
                'n_spatial_dims': 2,
                'spatial_coords_list': [x1, x2],
            }

        monkeypatch.setattr(dscv_viz_module, '_calculate_pde_fields', fake_fields)

        request = viz_core.VizRequest(
            kind='residual',
            target=model,
            options={'output_dir': tmp_path},
        )
        result = viz_core.render(request)
        assert result.paths
        assert (tmp_path / 'dscv' / 'residual_analysis.png').exists()

    def test_parity_nd(self, tmp_path, monkeypatch):
        """DSCV parity works with N-D data (reshape(-1) handles it)."""
        adapter = DSCVVizAdapter()

        class StubDSCV:
            def __init__(self):
                self.best_p = type('P', (), {'traversal': []})()
                self.searcher = type('S', (), {'r_train': [], 'best_p': self.best_p})()

        viz_registry.register_adapter(StubDSCV, adapter)
        model = StubDSCV()

        def fake_fields(_model, _program):
            ut = np.arange(120).reshape(5, 6, 4).astype(float)
            return {
                'ut_grid': ut,
                'y_hat_grid': ut * 0.9,
                'n_spatial_dims': 2,
            }

        monkeypatch.setattr(dscv_viz_module, '_calculate_pde_fields', fake_fields)

        request = viz_core.VizRequest(
            kind='parity',
            target=model,
            options={'output_dir': tmp_path},
        )
        result = viz_core.render(request)
        assert result.paths
        assert (tmp_path / 'dscv' / 'parity_plot.png').exists()


# ===== Error path tests (Codex review L6) =====

class TestErrorPaths:
    def test_infer_axis_unknown_coord(self):
        """_infer_axis raises ValueError for non-x1/x2/x3 coord names."""
        from kd.model.discover.stridge import Node

        class FakeToken:
            def __init__(self, name, arity=0):
                self.name = name
                self.arity = arity

        node = Node(FakeToken('y', 0))
        with pytest.raises(ValueError, match="无法从 'y' 推断空间轴索引"):
            dscv_viz_module._infer_axis(node, {'x1': 0, 'x2': 1, 'x3': 2})

    def test_finite_difference_nd_order_5_raises(self):
        """_finite_difference_nd raises ValueError for order > 4."""
        arr = np.ones((10, 10))
        with pytest.raises(ValueError, match="只支持1-4阶导数"):
            dscv_viz_module._finite_difference_nd(arr, 0.1, order=5, axis=0)

    def test_field_comparison_data_3d_spatial(self):
        """3D spatial coords (x, y, z) contract works."""
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 4)
        z = np.linspace(0, 1, 5)
        t = np.linspace(0, 1, 2)
        field = np.arange(3 * 4 * 5 * 2).reshape(3, 4, 5, 2).astype(float)

        data = FieldComparisonData(
            spatial_coords=[x, y, z], t_coords=t,
            true_field=field, predicted_field=field * 0.9,
        )
        assert data.n_spatial_dims == 3
        assert data.true_field.shape == (3, 4, 5, 2)

    def test_field_comparison_data_list_of_lists(self):
        """list-of-lists spatial_coords is correctly interpreted as multi-axis."""
        x = [0.0, 0.5, 1.0]
        y = [0.0, 0.25, 0.5, 0.75]
        t = np.linspace(0, 1, 2)
        field = np.arange(3 * 4 * 2).reshape(3, 4, 2).astype(float)

        data = FieldComparisonData(
            spatial_coords=[x, y], t_coords=t,
            true_field=field, predicted_field=field,
        )
        assert data.n_spatial_dims == 2
        np.testing.assert_array_equal(data.spatial_coords[0], np.array(x))
        np.testing.assert_array_equal(data.spatial_coords[1], np.array(y))

    def test_time_slice_data_list_of_lists(self):
        """list-of-lists also works for TimeSliceComparisonData."""
        x = [0.0, 0.5, 1.0]
        y = [0.0, 0.25, 0.5, 0.75]
        t = np.linspace(0, 1, 2)
        field = np.arange(3 * 4 * 2).reshape(3, 4, 2).astype(float)

        data = TimeSliceComparisonData(
            spatial_coords=[x, y], t_coords=t,
            true_field=field, predicted_field=field,
            slice_times=np.array([0.5]),
        )
        assert data.n_spatial_dims == 2

    def test_sga_time_slices_nd_generates_plot(self, tmp_path):
        """time_slices works for 2D spatial (Phase 2)."""
        adapter = SGAVizAdapter()
        viz_registry.register_adapter(StubSGA3D, adapter)

        model = StubSGA3D()
        request = viz_core.VizRequest(
            kind='time_slices',
            target=model,
            options={'output_dir': tmp_path},
        )
        result = viz_core.render(request)
        assert not result.warnings, f"Unexpected warnings: {result.warnings}"
        path = tmp_path / 'sga' / 'time_slices_comparison.png'
        assert path.exists()


# =============================================================================
# Phase 2 Tests — TDD RED phase
# =============================================================================

# ===== 4D context helpers (3D spatial + time) =====

class DummySGAContext4D:
    """4D context with axis_order=['x', 'y', 'z', 't'] for 3D spatial tests."""

    def __init__(self):
        nx, ny, nz, nt = 3, 4, 5, 6
        self.axis_order = ['x', 'y', 'z', 't']
        self.lhs_axis = 't'

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        z = np.linspace(0, 1, nz)
        t = np.linspace(0, 1, nt)
        self.coords_1d = {'x': x, 'y': y, 'z': z, 't': t}

        # Fields in axis_order layout: (x, y, z, t)
        self.u = np.arange(nx * ny * nz * nt).reshape(nx, ny, nz, nt).astype(float)
        self.u_origin = self.u + 0.1
        self.ut = self.u * 0.5
        self.ut_origin = self.u_origin * 0.5
        self.right_side_full = self.ut - 0.02
        self.right_side_full_origin = self.ut_origin - 0.01
        self.default_terms = self.u.reshape(-1, 1)
        self.default_names = ['u']
        self.num_default = 1


class StubSGA4D:
    """Wraps DummySGAContext4D for 3D spatial SGA tests."""

    def __init__(self, latex: str = 'u_t = u_{xx} + u_{yy} + u_{zz}', structure: Optional[str] = None):
        self._latex = latex
        self._structure = structure or latex
        self.context_ = DummySGAContext4D()
        self.best_equation_details_ = SGAEquationDetails(
            lhs='u_t',
            terms=[SGAEquationTerm(label='u', source='default', coefficient=1.0, tree=None)],
            predicted_rhs=self.context_.ut.copy(),
        )

    def equation_latex(self, *, include_coefficients: bool = True):
        return self._latex if include_coefficients else self._structure


# ===== Phase 2 Test: SGA 3D Spatial Slice =====

class TestPhase2SGA3DSpatialSlice:
    """Phase 2: 3D spatial field_comparison and residual should produce 2D slice heatmaps."""

    def test_sga_field_comparison_3d_spatial(self, tmp_path):
        """3D spatial field comparison should generate a figure with experimental warning."""
        adapter = SGAVizAdapter()
        viz_registry.register_adapter(StubSGA4D, adapter)

        model = StubSGA4D()
        request = viz_core.VizRequest(
            kind='field_comparison',
            target=model,
            options={'output_dir': tmp_path},
        )
        result = viz_core.render(request)

        # 3D spatial should produce an experimental warning
        assert any("experimental" in w for w in result.warnings), (
            f"Expected 3D experimental warning, got: {result.warnings}"
        )

        # Should produce output file
        path = tmp_path / 'sga' / 'field_comparison.png'
        assert path.exists(), f"Expected file at {path}"
        assert result.paths == [path]

        # Metadata should contain FieldComparisonData with 3D spatial
        field_data = result.metadata['field_comparison_data']
        assert isinstance(field_data, FieldComparisonData)
        assert field_data.n_spatial_dims == 3
        # Shape: (nx, ny, nz, nt) = (3, 4, 5, 6)
        assert field_data.true_field.shape == (3, 4, 5, 6)

    def test_sga_residual_3d_spatial(self, tmp_path):
        """3D spatial residual should generate a 2D slice heatmap with experimental warning."""
        adapter = SGAVizAdapter()
        viz_registry.register_adapter(StubSGA4D, adapter)

        model = StubSGA4D()
        request = viz_core.VizRequest(
            kind='residual',
            target=model,
            options={'output_dir': tmp_path, 'bins': 10},
        )
        result = viz_core.render(request)

        # 3D spatial residual should include experimental warning
        assert any("experimental" in w for w in result.warnings), (
            f"Expected 3D experimental warning, got: {result.warnings}"
        )

        # Should produce output file
        path = tmp_path / 'sga' / 'residual_analysis.png'
        assert path.exists(), f"Expected file at {path}"

    def test_sga_field_comparison_3d_slice_axis_option(self, tmp_path):
        """3D spatial field comparison respects slice_axis and slice_index options."""
        adapter = SGAVizAdapter()
        viz_registry.register_adapter(StubSGA4D, adapter)

        model = StubSGA4D()
        request = viz_core.VizRequest(
            kind='field_comparison',
            target=model,
            options={
                'output_dir': tmp_path,
                'slice_axis': 0,       # slice along x (first spatial axis)
                'slice_index': 1,      # pick index 1 of x
            },
        )
        result = viz_core.render(request)

        # 3D experimental warning expected
        assert any("experimental" in w for w in result.warnings)
        path = tmp_path / 'sga' / 'field_comparison.png'
        assert path.exists()


# ===== Phase 2 Test: DSCV 3D Spatial Slice =====

class TestPhase2DSCV3DSpatialSlice:
    """Phase 2: DSCV field_comparison and residual should handle 3D spatial via slicing."""

    def test_dscv_field_comparison_3d_spatial(self, tmp_path, monkeypatch):
        """DSCV field comparison should work with 3D spatial data (slice to 2D)."""
        adapter = DSCVVizAdapter()

        class StubDSCV:
            def __init__(self):
                self.best_p = type('P', (), {'traversal': []})()
                self.searcher = type('S', (), {'r_train': [], 'best_p': self.best_p})()

        viz_registry.register_adapter(StubDSCV, adapter)
        model = StubDSCV()

        nx, ny, nz, nt = 3, 4, 5, 6
        x1 = np.linspace(0, 1, nx)
        x2 = np.linspace(0, 1, ny)
        x3 = np.linspace(0, 1, nz)

        def fake_fields(_model, _program):
            ut = np.arange(nt * nx * ny * nz).reshape(nt, nx, ny, nz).astype(float)
            return {
                'x_axis': x1,
                't_axis': np.arange(nt),
                'ut_grid': ut,
                'y_hat_grid': ut * 0.9,
                'residual': (ut - ut * 0.9).flatten(),
                'coords': None,
                'n_spatial_dims': 3,
                'spatial_coords_list': [x1, x2, x3],
            }

        monkeypatch.setattr(dscv_viz_module, '_calculate_pde_fields', fake_fields)

        request = viz_core.VizRequest(
            kind='field_comparison',
            target=model,
            options={'output_dir': tmp_path},
        )
        result = viz_core.render(request)

        # 3D spatial should produce an experimental warning, but still generate output
        assert any("experimental" in w for w in result.warnings), (
            f"Expected 3D experimental warning, got: {result.warnings}"
        )
        assert result.paths
        assert (tmp_path / 'dscv' / 'field_comparison.png').exists()

    def test_dscv_residual_3d_spatial_has_heatmap(self, tmp_path, monkeypatch):
        """DSCV residual for 3D spatial should produce a 2D slice heatmap, not just histogram."""
        import matplotlib.pyplot as plt
        adapter = DSCVVizAdapter()

        class StubDSCV:
            def __init__(self):
                self.best_p = type('P', (), {'traversal': []})()
                self.searcher = type('S', (), {'r_train': [], 'best_p': self.best_p})()

        viz_registry.register_adapter(StubDSCV, adapter)
        model = StubDSCV()

        nx, ny, nz, nt = 3, 4, 5, 6
        x1 = np.linspace(0, 1, nx)
        x2 = np.linspace(0, 1, ny)
        x3 = np.linspace(0, 1, nz)

        created_figures = []
        original_figure = plt.figure

        def tracking_figure(*args, **kwargs):
            fig = original_figure(*args, **kwargs)
            created_figures.append(fig)
            return fig

        monkeypatch.setattr(plt, 'figure', tracking_figure)

        def fake_fields(_model, _program):
            ut = np.arange(nt * nx * ny * nz).reshape(nt, nx, ny, nz).astype(float)
            yhat = ut * 0.9
            return {
                'x_axis': x1,
                't_axis': np.arange(nt),
                'ut_grid': ut,
                'y_hat_grid': yhat,
                'residual': (ut - yhat).flatten(),
                'coords': None,
                'n_spatial_dims': 3,
                'spatial_coords_list': [x1, x2, x3],
            }

        monkeypatch.setattr(dscv_viz_module, '_calculate_pde_fields', fake_fields)

        request = viz_core.VizRequest(
            kind='residual',
            target=model,
            options={'output_dir': tmp_path},
        )
        result = viz_core.render(request)

        assert result.paths
        path = tmp_path / 'dscv' / 'residual_analysis.png'
        assert path.exists()

        # Phase 2 requirement: the residual plot must contain a 2D heatmap subplot
        # (not just a single histogram). The current code for >2D spatial creates
        # only 1 axis (histogram). Phase 2 should create 2+ axes (heatmap + histogram).
        assert len(created_figures) > 0, "Expected at least one figure to be created"
        fig = created_figures[-1]
        n_axes = len(fig.get_axes())
        assert n_axes >= 2, (
            f"Expected at least 2 axes (heatmap + histogram) for 3D spatial residual, "
            f"got {n_axes}. Currently falls back to histogram only."
        )


# ===== Phase 2 Test: SGA Time Slices 2D =====

class TestPhase2SGATimeSlices2D:
    """Phase 2: time_slices should work for 2D spatial (heatmap) and 3D spatial (slice+heatmap)."""

    def test_sga_time_slices_2d_spatial(self, tmp_path):
        """2D spatial time_slices should produce heatmaps, not a warning."""
        adapter = SGAVizAdapter()
        viz_registry.register_adapter(StubSGA3D, adapter)

        model = StubSGA3D()  # axis_order=['x','y','t'] => 2D spatial
        request = viz_core.VizRequest(
            kind='time_slices',
            target=model,
            options={'output_dir': tmp_path, 'slice_count': 3},
        )
        result = viz_core.render(request)

        # Phase 2: should NOT return '1D spatial' warning
        assert not result.warnings, f"Expected no warnings, got: {result.warnings}"

        # Should produce output file
        path = tmp_path / 'sga' / 'time_slices_comparison.png'
        assert path.exists(), f"Expected file at {path}"
        assert result.paths == [path]

        # Metadata should contain TimeSliceComparisonData with 2D spatial
        ts_data = result.metadata.get('time_slices_data')
        assert ts_data is not None, "Expected 'time_slices_data' in metadata"
        assert isinstance(ts_data, TimeSliceComparisonData)
        assert ts_data.n_spatial_dims == 2

    def test_sga_time_slices_3d_spatial(self, tmp_path):
        """3D spatial time_slices should also work (slice spatial to 2D, then heatmap)."""
        adapter = SGAVizAdapter()
        viz_registry.register_adapter(StubSGA4D, adapter)

        model = StubSGA4D()  # axis_order=['x','y','z','t'] => 3D spatial
        request = viz_core.VizRequest(
            kind='time_slices',
            target=model,
            options={'output_dir': tmp_path, 'slice_count': 2},
        )
        result = viz_core.render(request)

        # 3D spatial should produce experimental warning
        assert any("experimental" in w for w in result.warnings), (
            f"Expected 3D experimental warning, got: {result.warnings}"
        )

        path = tmp_path / 'sga' / 'time_slices_comparison.png'
        assert path.exists(), f"Expected file at {path}"

        ts_data = result.metadata.get('time_slices_data')
        assert ts_data is not None
        assert isinstance(ts_data, TimeSliceComparisonData)
        assert ts_data.n_spatial_dims == 3


# ===== Phase 2 Test: SPR/PINN N-D =====

class TestPhase2SPRPinnND:
    """Phase 2: SPR/PINN field comparison and _calculate_pinn_fields should support N-D."""

    def test_calculate_pinn_fields_nd(self, monkeypatch):
        """_calculate_pinn_fields should return spatial_coords_list and n_spatial_dims for N-D."""
        import torch

        # Mock Program.task with 2D spatial (x1, x2) PINN data
        n_points = 50
        x1_tensor = torch.rand(n_points, 1)
        x2_tensor = torch.rand(n_points, 1)
        t_tensor = torch.rand(n_points, 1)
        ut_np = np.random.randn(n_points, 1)

        class FakeTask:
            x = [x1_tensor, x2_tensor]  # 2D spatial: list of 2 tensors
            t = t_tensor
            ut = ut_np
            u = [torch.rand(n_points, 1)]

        # Monkeypatch Program.task
        monkeypatch.setattr('kd.viz.dscv_viz.Program', type('MockProgram', (), {'task': FakeTask()}))

        class FakeBestProgram:
            y_hat_rhs = np.random.randn(n_points, 1)
            Theta = np.random.randn(n_points, 3)

            @property
            def r_ridge(self):
                return 0.5  # trigger cache fill

        result = dscv_viz_module._calculate_pinn_fields(None, FakeBestProgram())

        # Phase 2: should return N-D metadata
        assert 'n_spatial_dims' in result, "Expected 'n_spatial_dims' in result"
        assert result['n_spatial_dims'] == 2
        assert 'spatial_coords_list' in result, "Expected 'spatial_coords_list' in result"
        assert len(result['spatial_coords_list']) == 2
        # coords should be (N, n_spatial+1) = (N, 3) for 2D spatial + time
        assert result['coords'].shape == (n_points, 3)

    def test_spr_field_comparison_nd(self, tmp_path, monkeypatch):
        """SPR field comparison should use time-slice + spatial griddata for 2D spatial."""
        adapter = DSCVVizAdapter()

        class StubDSCV:
            def __init__(self):
                self.best_p = type('P', (), {'traversal': []})()
                self.searcher = type('S', (), {'r_train': [], 'best_p': self.best_p})()

        viz_registry.register_adapter(StubDSCV, adapter)
        model = StubDSCV()

        n_points = 100
        x1 = np.random.rand(n_points)
        x2 = np.random.rand(n_points)
        t = np.random.rand(n_points)
        y_true = np.sin(x1) * np.cos(x2)
        y_pred = y_true * 0.95

        def fake_pinn_fields(_model, _program):
            return {
                'residual': y_true - y_pred,
                'coords': np.column_stack([x1, x2, t]),  # (N, 3)
                'coords_x': x1,   # legacy 1D key — not enough for 2D spatial
                'coords_t': t,
                'y_true': y_true,
                'y_pred': y_pred,
                'n_spatial_dims': 2,
                'spatial_coords_list': [x1, x2],
            }

        monkeypatch.setattr(dscv_viz_module, '_calculate_pinn_fields', fake_pinn_fields)

        request = viz_core.VizRequest(
            kind='spr_field_comparison',
            target=model,
            options={'output_dir': tmp_path},
        )
        result = viz_core.render(request)

        # Phase 2: should produce a field comparison plot for 2D spatial
        assert not result.warnings, f"Expected no warnings, got: {result.warnings}"
        assert result.paths
        assert (tmp_path / 'dscv' / 'spr_field_comparison.png').exists()

        # Phase 2: metadata should contain FieldComparisonData with 2D spatial
        fc_data = result.metadata.get('field_comparison')
        assert fc_data is not None, "Expected 'field_comparison' in metadata"
        assert isinstance(fc_data, FieldComparisonData)
        assert fc_data.n_spatial_dims == 2, (
            f"Expected 2D spatial FieldComparisonData, got {fc_data.n_spatial_dims}D"
        )

    def test_spr_residual_nd_metadata(self, tmp_path, monkeypatch):
        """SPR residual should store n_spatial_dims in metadata for N-D spatial."""
        adapter = DSCVVizAdapter()

        class StubDSCV:
            def __init__(self):
                self.best_p = type('P', (), {'traversal': []})()
                self.searcher = type('S', (), {'r_train': [], 'best_p': self.best_p})()

        viz_registry.register_adapter(StubDSCV, adapter)
        model = StubDSCV()

        n_points = 80
        x1 = np.random.rand(n_points)
        x2 = np.random.rand(n_points)
        t = np.random.rand(n_points)
        y_true = np.sin(x1) * np.cos(x2)
        y_pred = y_true * 0.9
        coords_3col = np.column_stack([x1, x2, t])  # (N, 3)

        def fake_pinn_fields(_model, _program):
            return {
                'residual': y_true - y_pred,
                'coords': coords_3col,           # (N, 3) for 2D spatial + time
                'coords_x': x1,
                'coords_t': t,
                'y_true': y_true,
                'y_pred': y_pred,
                'n_spatial_dims': 2,
                'spatial_coords_list': [x1, x2],
            }

        monkeypatch.setattr(dscv_viz_module, '_calculate_pinn_fields', fake_pinn_fields)

        request = viz_core.VizRequest(
            kind='spr_residual',
            target=model,
            options={'output_dir': tmp_path},
        )
        result = viz_core.render(request)

        assert not result.warnings, f"Expected no warnings, got: {result.warnings}"
        assert result.paths

        # Phase 2: metadata should include n_spatial_dims for downstream consumers
        assert 'n_spatial_dims' in result.metadata, (
            "Expected 'n_spatial_dims' in result.metadata for SPR residual N-D"
        )
        assert result.metadata['n_spatial_dims'] == 2


# ===== Phase 2 Test: API N-D =====

class TestPhase2APINd:
    """Phase 2: API layer should accept spatial_coords parameter."""

    def test_api_field_comparison_spatial_coords(self, tmp_path, monkeypatch):
        """plot_field_comparison should accept spatial_coords as an alternative to x_coords."""
        from kd.viz import api as viz_api

        # We test that the API function accepts spatial_coords and passes it through
        # to the VizRequest options.
        captured_options = {}

        def fake_dispatch(kind, model, *, show_info, options):
            captured_options.update(options)
            return viz_core.VizResult(intent=kind)

        monkeypatch.setattr(viz_api, '_dispatch', fake_dispatch)

        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 6)
        t = np.linspace(0, 1, 4)
        field = np.zeros((5, 6, 4))

        # Phase 2: spatial_coords should be accepted as keyword argument
        viz_api.plot_field_comparison(
            None,
            spatial_coords=[x, y],
            t_coords=t,
            true_field=field,
            predicted_field=field,
            output_dir=tmp_path,
            show_info=False,
        )

        assert 'spatial_coords' in captured_options, (
            "Expected 'spatial_coords' to be passed through to options"
        )

    def test_api_time_slices_spatial_coords(self, tmp_path, monkeypatch):
        """plot_time_slices should accept spatial_coords parameter."""
        from kd.viz import api as viz_api

        captured_options = {}

        def fake_dispatch(kind, model, *, show_info, options):
            captured_options.update(options)
            return viz_core.VizResult(intent=kind)

        monkeypatch.setattr(viz_api, '_dispatch', fake_dispatch)

        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 6)
        t = np.linspace(0, 1, 4)
        field = np.zeros((5, 6, 4))

        viz_api.plot_time_slices(
            None,
            spatial_coords=[x, y],
            t_coords=t,
            true_field=field,
            predicted_field=field,
            output_dir=tmp_path,
            show_info=False,
        )

        assert 'spatial_coords' in captured_options, (
            "Expected 'spatial_coords' to be passed through to options"
        )
