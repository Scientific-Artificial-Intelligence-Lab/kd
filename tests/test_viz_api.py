from pathlib import Path

import numpy as np
import pytest

from kd.viz import api as viz_api
from kd.viz import core as viz_core


@pytest.fixture(autouse=True)
def suppress_emit(monkeypatch):
    monkeypatch.setattr(viz_api, '_emit_info', lambda *args, **kwargs: None)


def test_plot_training_curve_dispatch(monkeypatch):
    captured = {}

    def fake_render(request):
        captured['kind'] = request.kind
        captured['options'] = request.options
        return viz_core.VizResult(intent=request.kind, paths=[Path('foo.png')])

    monkeypatch.setattr(viz_api, 'render', fake_render)

    model = object()
    result = viz_api.plot_training_curve(model, output_dir='out')

    assert result.paths
    assert captured['kind'] == 'training_curve'
    assert captured['options'] == {'output_dir': 'out'}


def test_render_equation_warning_passthrough(monkeypatch):
    warning = 'unsupported'

    def fake_render(request):
        return viz_core.VizResult(intent=request.kind, warnings=[warning])

    monkeypatch.setattr(viz_api, 'render', fake_render)

    result = viz_api.render_equation(object(), show_info=False)
    assert result.warnings == [warning]


def test_plot_residuals_injects_arrays(monkeypatch):
    captured = {}

    def fake_render(request):
        captured['kind'] = request.kind
        captured['options'] = request.options
        return viz_core.VizResult(intent=request.kind, paths=[Path('bar.png')])

    monkeypatch.setattr(viz_api, 'render', fake_render)

    actual = np.array([1.0, 0.8])
    predicted = np.array([0.9, 0.75])
    coords = np.array([0.1, 0.2])

    viz_api.plot_residuals(
        object(),
        actual=actual,
        predicted=predicted,
        coordinates=coords,
        bins=32,
    )

    assert captured['kind'] == 'residual'
    assert captured['options']['actual'] is actual
    assert captured['options']['predicted'] is predicted
    assert captured['options']['coordinates'] is coords
    assert captured['options']['bins'] == 32
