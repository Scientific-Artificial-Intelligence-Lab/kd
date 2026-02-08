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
