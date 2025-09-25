import numpy as np
import pytest

from kd.viz import FieldComparisonData, OptimizationHistoryData, ResidualPlotData
from kd.viz import core as viz_core
from kd.viz import registry as viz_registry
from kd.viz.adapters import DLGAVizAdapter, DSCVVizAdapter


class StubDLGA:
    def __init__(self):
        self.train_loss_history = [1.0, 0.8, 0.6]
        self.val_loss_history = [1.2, 1.0, 0.9]
        self.evolution_history = [
            {'fitness': 1.0, 'complexity': 5, 'population_size': 10, 'unique_modules': 4},
            {'fitness': 0.8, 'complexity': 4, 'population_size': 9, 'unique_modules': 5},
        ]
        self.Chrom = [[[0]]]
        self.coef = [np.array([[1.0]])]
        self.name = ['u_t']
        self.user_operators = ['u']


class StubDSCV:
    pass


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


def test_dscv_placeholder_warning():
    adapter = DSCVVizAdapter()
    viz_registry.register_adapter(StubDSCV, adapter)

    result = viz_core.render(viz_core.VizRequest(kind='tree', target=StubDSCV()))
    assert not result.has_content
    assert result.warnings
    assert 'does not support intent' in result.warnings[0]
