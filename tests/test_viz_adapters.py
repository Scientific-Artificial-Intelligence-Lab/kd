import numpy as np
import pytest
import sys
import types

import kd.viz.dscv_viz as dscv_viz_module

from kd.viz import (
    FieldComparisonData,
    OptimizationHistoryData,
    ParityPlotData,
    ResidualPlotData,
    RewardEvolutionData,
    TermRelationshipData,
    TimeSliceComparisonData,
)
from kd.viz import core as viz_core
from kd.viz import registry as viz_registry
from kd.viz.adapters import DLGAVizAdapter, DSCVVizAdapter, SGAVizAdapter


class StubDLGA:
    def __init__(self):
        self.train_loss_history = [1.0, 0.8, 0.6]
        self.val_loss_history = [1.2, 1.0, 0.9]
        self.evolution_history = [
            {'fitness': 1.0, 'complexity': 5, 'population_size': 10, 'unique_modules': 4},
            {'fitness': 0.8, 'complexity': 4, 'population_size': 9, 'unique_modules': 5},
        ]
        self.Chrom = [[[0], [1]]]
        self.coef = [np.array([[1.0], [0.5]])]
        self.name = ['u_t']
        self.user_operators = ['u', 'u_x']
        base = np.linspace(0.0, 1.0, 10)
        self.metadata = {
            'u_t': base.reshape(-1),
            'u': (base + 1).reshape(-1),
            'u_x': (base - 0.5).reshape(-1),
        }


class DummyProgram:
    def __init__(self):
        self.traversal = ['u', 'add', 'u']
        self.library = type('Lib', (), {'name_arites': {'u': 0, 'add': 2}})()
        self.str_expression = 'u + u'


class StubSearch:
    def __init__(self, batches):
        self.r_train = batches
        self.r_history = [np.asarray(batch) for batch in batches]
        self.plotter = type('Plotter', (), {
            'tree_plot': lambda self_ref, program: None,
        })()
        self.best_p = DummyProgram()


class StubDSCV:
    def __init__(self):
        self.searcher = StubSearch([
            [0.5, 0.6, 0.55],
            [0.7, 0.65, 0.72],
            [0.8, 0.78, 0.81],
        ])
        self.best_p = None  # Required for equation intent (test expects LaTeX degradation)


from typing import Optional


class DummySGAContext:
    def __init__(self):
        space = np.linspace(0.0, 1.0, 3)
        time = np.linspace(0.0, 1.0, 4)
        space_grid = np.tile(space.reshape(-1, 1), (1, time.size))
        time_grid = np.tile(time.reshape(1, -1), (space.size, 1))

        self.u = np.outer(space, np.ones_like(time))
        self.u_origin = self.u + 0.1
        self.x = space_grid
        self.t = time_grid
        self.x_origin = space_grid
        self.t_origin = time_grid


class StubSGA:
    def __init__(self, latex: str = 'u_t = u_{xx}', structure: Optional[str] = None):
        self._latex = latex
        self._structure = structure or latex
        self.context_ = DummySGAContext()

    def equation_latex(self, *, include_coefficients: bool = True):
        return self._latex if include_coefficients else self._structure


def setup_function():
    viz_registry.clear_registry()


def teardown_function():
    viz_registry.clear_registry()


@pytest.fixture(autouse=True)
def suppress_show(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, 'show', lambda: None)


def register_dlga():
    adapter = DLGAVizAdapter()
    viz_registry.register_adapter(StubDLGA, adapter)
    return adapter


def test_training_curve(tmp_path):
    register_dlga()
    model = StubDLGA()

    request = viz_core.VizRequest(
        kind='training_curve',
        target=model,
        options={'output_dir': tmp_path},
    )
    result = viz_core.render(request)
    path = tmp_path / 'dlga' / 'training_loss.png'
    assert path.exists()
    assert result.paths == [path]
    assert result.metadata['points'] == len(model.train_loss_history)


def test_equation_render(tmp_path):
    register_dlga()
    model = StubDLGA()

    request = viz_core.VizRequest(
        kind='equation',
        target=model,
        options={'output_dir': tmp_path, 'font_size': 10},
    )
    result = viz_core.render(request)
    path = tmp_path / 'dlga' / 'equation.png'
    assert path.exists()
    assert 'latex' in result.metadata


def test_residual_contract(tmp_path):
    register_dlga()
    model = StubDLGA()

    actual = np.array([1.0, 0.8, 0.6, 0.5])
    predicted = np.array([0.9, 0.75, 0.55, 0.45])
    coords = np.linspace(0.0, 1.0, actual.size)

    request = viz_core.VizRequest(
        kind='residual',
        target=model,
        options={
            'output_dir': tmp_path,
            'actual': actual,
            'predicted': predicted,
            'coordinates': coords,
        },
    )
    result = viz_core.render(request)
    path = tmp_path / 'dlga' / 'residual_analysis.png'
    assert path.exists()
    residual = result.metadata['residual']
    assert isinstance(residual, ResidualPlotData)
    assert np.allclose(residual.actual - residual.predicted, residual.residuals)
    summary = result.metadata['summary']
    assert summary['count'] == actual.size


def test_optimization_contract(tmp_path):
    register_dlga()
    model = StubDLGA()

    request = viz_core.VizRequest(
        kind='optimization',
        target=model,
        options={'output_dir': tmp_path},
    )
    result = viz_core.render(request)

    path = tmp_path / 'dlga' / 'optimization_analysis.png'
    assert path.exists()
    data = result.metadata['optimization']
    assert isinstance(data, OptimizationHistoryData)
    assert data.steps.shape == data.objective.shape
    assert result.metadata['summary']['final_objective'] == pytest.approx(float(data.objective[-1]))


def test_field_comparison_contract(tmp_path):
    register_dlga()
    model = StubDLGA()

    x = np.linspace(0, 1, 4)
    t = np.linspace(0, 1, 3)
    true_field = np.arange(12).reshape(4, 3)
    pred_field = true_field * 0.9

    request = viz_core.VizRequest(
        kind='field_comparison',
        target=model,
        options={
            'output_dir': tmp_path,
            'x_coords': x,
            't_coords': t,
            'true_field': true_field,
            'predicted_field': pred_field,
        },
    )
    result = viz_core.render(request)

    path = tmp_path / 'dlga' / 'field_comparison.png'
    assert path.exists()
    data = result.metadata['field_comparison']
    assert isinstance(data, FieldComparisonData)
    assert data.true_field.shape == (x.size, t.size)
    assert np.allclose(data.residual_field, true_field - pred_field)


def test_time_slices_contract(tmp_path):
    register_dlga()
    model = StubDLGA()

    x = np.linspace(0, 1, 4)
    t = np.linspace(0, 1, 3)
    true_field = np.arange(12).reshape(4, 3)
    pred_field = true_field * 0.9

    request = viz_core.VizRequest(
        kind='time_slices',
        target=model,
        options={
            'output_dir': tmp_path,
            'x_coords': x,
            't_coords': t,
            'true_field': true_field,
            'predicted_field': pred_field,
            'slice_times': [0.0, 0.5, 1.0],
        },
    )
    result = viz_core.render(request)

    path = tmp_path / 'dlga' / 'time_slices_comparison.png'
    assert path.exists()
    data = result.metadata['time_slices']
    assert isinstance(data, TimeSliceComparisonData)
    assert np.allclose(data.true_field, true_field)


def test_derivative_relationships(tmp_path):
    register_dlga()
    model = StubDLGA()

    request = viz_core.VizRequest(
        kind='derivative_relationships',
        target=model,
        options={'output_dir': tmp_path, 'top_n_terms': 2},
    )
    result = viz_core.render(request)

    path = tmp_path / 'dlga' / 'derivative_relationships.png'
    assert path.exists()
    data = result.metadata['term_relationships']
    assert isinstance(data, TermRelationshipData)
    assert len(data.terms) == 2


def test_parity_plot(tmp_path):
    register_dlga()
    model = StubDLGA()

    request = viz_core.VizRequest(
        kind='parity',
        target=model,
        options={'output_dir': tmp_path},
    )
    result = viz_core.render(request)

    path = tmp_path / 'dlga' / 'pde_parity_plot.png'
    assert path.exists()
    data = result.metadata['parity']
    assert isinstance(data, ParityPlotData)
    assert data.actual_values.shape == data.predicted_values.shape


def test_dscv_search_evolution(tmp_path):
    adapter = DSCVVizAdapter()
    viz_registry.register_adapter(StubDSCV, adapter)

    model = StubDSCV()

    request = viz_core.VizRequest(
        kind='search_evolution',
        target=model,
        options={'output_dir': tmp_path},
    )
    result = viz_core.render(request)
    assert result.warnings or result.paths
    if result.warnings:
        assert "not implemented" in result.warnings[0]
    else:
        path = tmp_path / 'dscv' / 'reward_evolution.png'
        assert path.exists()
        data = result.metadata['reward_evolution']
        assert isinstance(data, RewardEvolutionData)
        assert data.steps.size == len(model.searcher.r_train)


def test_dscv_density(tmp_path):
    adapter = DSCVVizAdapter()
    viz_registry.register_adapter(StubDSCV, adapter)

    model = StubDSCV()

    request = viz_core.VizRequest(
        kind='density',
        target=model,
        options={'output_dir': tmp_path, 'epoches': [0, 1]},
    )
    result = viz_core.render(request)
    assert result.warnings or result.paths
    if result.warnings:
        assert 'seaborn is required' in result.warnings[0]
    else:
        path = tmp_path / 'dscv' / 'reward_density.png'
        assert path.exists()

    # fallback to tree to confirm placeholder intent still returns warning
    tree_request = viz_core.VizRequest(kind='tree', target=model, options={'output_dir': tmp_path})
    tree_result = viz_core.render(tree_request)
    assert tree_result.warnings


def test_dscv_equation_warning(tmp_path):
    adapter = DSCVVizAdapter()
    viz_registry.register_adapter(StubDSCV, adapter)

    model = StubDSCV()

    request = viz_core.VizRequest(
        kind='equation',
        target=model,
        options={'output_dir': tmp_path},
    )
    result = viz_core.render(request)
    if result.warnings:
        assert 'Failed to convert program' in result.warnings[0]
    else:
        path = tmp_path / 'dscv' / 'equation.png'
        assert path.exists()
        assert 'latex' in result.metadata


def test_dscv_residual_warning(tmp_path):
    adapter = DSCVVizAdapter()
    viz_registry.register_adapter(StubDSCV, adapter)

    model = StubDSCV()

    request = viz_core.VizRequest(
        kind='residual',
        target=model,
        options={'output_dir': tmp_path},
    )
    result = viz_core.render(request)
    assert result.warnings


def test_dscv_field_comparison(tmp_path, monkeypatch):
    adapter = DSCVVizAdapter()
    viz_registry.register_adapter(StubDSCV, adapter)

    model = StubDSCV()
    model.best_p = DummyProgram()

    def fake_fields(_model, _program):
        x = np.linspace(0, 1, 2)
        t = np.linspace(0, 1, 3)
        ut = np.arange(6).reshape(2, 3).astype(float)
        yhat = ut * 0.9
        coords = np.stack(np.meshgrid(x, t, indexing='ij'), axis=-1).reshape(-1, 2)
        return {
            'x_axis': x,
            't_axis': t,
            'ut_grid': ut,
            'y_hat_grid': yhat,
            'coords': coords,
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


def test_dscv_parity(tmp_path, monkeypatch):
    adapter = DSCVVizAdapter()
    viz_registry.register_adapter(StubDSCV, adapter)

    model = StubDSCV()
    model.best_p = DummyProgram()

    def fake_fields(_model, _program):
        x = np.linspace(0, 1, 2)
        t = np.linspace(0, 1, 3)
        ut = np.arange(6).reshape(2, 3).astype(float)
        yhat = ut * 0.9
        coords = np.stack(np.meshgrid(x, t, indexing='ij'), axis=-1).reshape(-1, 2)
        return {
            'x_axis': x,
            't_axis': t,
            'ut_grid': ut,
            'y_hat_grid': yhat,
            'coords': coords,
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


def test_dscv_spr_residual_warning(tmp_path, monkeypatch):
    adapter = DSCVVizAdapter()
    viz_registry.register_adapter(StubDSCV, adapter)

    model = StubDSCV()

    def fake_pinn_fields(_model, _program):
        return {
            'coords': np.zeros((4, 2)),
            'y_true': np.zeros(4),
            'y_pred': np.zeros(4),
        }

    monkeypatch.setattr(dscv_viz_module, '_calculate_pinn_fields', fake_pinn_fields)

    request = viz_core.VizRequest(
        kind='spr_residual',
        target=model,
        options={'output_dir': tmp_path},
    )
    result = viz_core.render(request)
    assert result.paths or result.warnings


def test_dscv_spr_field_comparison_warning(tmp_path, monkeypatch):
    adapter = DSCVVizAdapter()
    viz_registry.register_adapter(StubDSCV, adapter)

    model = StubDSCV()

    def fake_pinn_fields(_model, _program):
        return {
            'coords_x': np.array([0.0, 1.0]),
            'coords_t': np.array([0.0, 0.5, 1.0]),
            'y_true': np.arange(6).astype(float),
            'y_pred': np.arange(6).astype(float) * 0.9,
        }

    monkeypatch.setattr(dscv_viz_module, '_calculate_pinn_fields', fake_pinn_fields)

    fake_interpolate = types.SimpleNamespace(
        griddata=lambda points, values, xi, method='cubic': np.zeros_like(xi[0])
    )
    monkeypatch.setitem(sys.modules, 'scipy', types.SimpleNamespace(interpolate=fake_interpolate))
    monkeypatch.setitem(sys.modules, 'scipy.interpolate', fake_interpolate)

    request = viz_core.VizRequest(
        kind='spr_field_comparison',
        target=model,
        options={'output_dir': tmp_path},
    )
    result = viz_core.render(request)
    assert result.paths or result.warnings


def test_sga_equation(tmp_path):
    adapter = SGAVizAdapter()
    viz_registry.register_adapter(StubSGA, adapter)

    model = StubSGA('u_t = u_{xx} + u')
    request = viz_core.VizRequest(
        kind='equation',
        target=model,
        options={'output_dir': tmp_path, 'font_size': 12},
    )
    result = viz_core.render(request)

    path = tmp_path / 'sga' / 'equation.png'
    structure_path = tmp_path / 'sga' / 'equation_structure.png'
    assert path.exists()
    assert structure_path.exists()
    assert result.paths == [path, structure_path]
    assert result.metadata['latex'] == 'u_t = u_{xx} + u'
    assert result.metadata['structure_latex'] == 'u_t = u_{xx} + u'


def test_sga_field_comparison(tmp_path):
    adapter = SGAVizAdapter()
    viz_registry.register_adapter(StubSGA, adapter)

    model = StubSGA()
    request = viz_core.VizRequest(
        kind='field_comparison',
        target=model,
        options={'output_dir': tmp_path},
    )
    result = viz_core.render(request)

    path = tmp_path / 'sga' / 'field_comparison.png'
    assert path.exists()
    assert result.paths == [path]
    field_data = result.metadata['field_comparison_data']
    assert isinstance(field_data, FieldComparisonData)
    assert field_data.predicted_field.shape == model.context_.u.shape
    assert field_data.true_field.shape == model.context_.u_origin.shape


def test_sga_time_slices(tmp_path):
    adapter = SGAVizAdapter()
    viz_registry.register_adapter(StubSGA, adapter)

    model = StubSGA()
    request = viz_core.VizRequest(
        kind='time_slices',
        target=model,
        options={'output_dir': tmp_path, 'slice_times': [0.0, 0.5, 1.0]},
    )
    result = viz_core.render(request)

    path = tmp_path / 'sga' / 'time_slices_comparison.png'
    assert path.exists()
    time_data = result.metadata['time_slices_data']
    assert isinstance(time_data, TimeSliceComparisonData)
    summary = result.metadata['time_slices_summary']
    assert np.allclose(summary['requested_slice_times'], [0.0, 0.5, 1.0])
    assert np.allclose(time_data.slice_times, summary['actual_slice_times'])
