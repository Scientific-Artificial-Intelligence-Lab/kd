"""High-level helper functions for KD visualization intents."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .core import VizRequest, VizResult, render


def plot_training_curve(model: Any, *, show_info: bool = True, **options: Any) -> VizResult:
    return _dispatch('training_curve', model, show_info=show_info, options=options)


def plot_validation_curve(model: Any, *, show_info: bool = True, **options: Any) -> VizResult:
    return _dispatch('validation_curve', model, show_info=show_info, options=options)


def plot_search_evolution(model: Any, *, show_info: bool = True, **options: Any) -> VizResult:
    return _dispatch('search_evolution', model, show_info=show_info, options=options)


def plot_optimization(model: Any, *, show_info: bool = True, **options: Any) -> VizResult:
    return _dispatch('optimization', model, show_info=show_info, options=options)


def plot_time_slices(
    model: Any,
    *,
    x_coords: Any,
    t_coords: Any,
    true_field: Any,
    predicted_field: Any,
    slice_times: Optional[Any] = None,
    show_info: bool = True,
    **options: Any,
) -> VizResult:
    payload = dict(options)
    payload['x_coords'] = x_coords
    payload['t_coords'] = t_coords
    payload['true_field'] = true_field
    payload['predicted_field'] = predicted_field
    if slice_times is not None:
        payload['slice_times'] = slice_times
    return _dispatch('time_slices', model, show_info=show_info, options=payload)


def render_equation(model: Any, *, show_info: bool = True, **options: Any) -> VizResult:
    return _dispatch('equation', model, show_info=show_info, options=options)


def plot_residuals(
    model: Any,
    *,
    actual: Any,
    predicted: Any,
    coordinates: Optional[Any] = None,
    show_info: bool = True,
    **options: Any,
) -> VizResult:
    payload = dict(options)
    payload['actual'] = actual
    payload['predicted'] = predicted
    if coordinates is not None:
        payload['coordinates'] = coordinates
    return _dispatch('residual', model, show_info=show_info, options=payload)


def plot_field_comparison(
    model: Any,
    *,
    x_coords: Any,
    t_coords: Any,
    true_field: Any,
    predicted_field: Any,
    residual_field: Optional[Any] = None,
    show_info: bool = True,
    **options: Any,
) -> VizResult:
    payload = dict(options)
    payload['x_coords'] = x_coords
    payload['t_coords'] = t_coords
    payload['true_field'] = true_field
    payload['predicted_field'] = predicted_field
    if residual_field is not None:
        payload['residual_field'] = residual_field
    return _dispatch('field_comparison', model, show_info=show_info, options=payload)


def plot_derivative_relationships(
    model: Any,
    *,
    top_n_terms: int = 4,
    show_info: bool = True,
    **options: Any,
) -> VizResult:
    payload = dict(options)
    payload['top_n_terms'] = top_n_terms
    return _dispatch('derivative_relationships', model, show_info=show_info, options=payload)


def plot_parity(
    model: Any,
    *,
    title: Optional[str] = None,
    show_info: bool = True,
    **options: Any,
) -> VizResult:
    payload = dict(options)
    if title is not None:
        payload['title'] = title
    return _dispatch('parity', model, show_info=show_info, options=payload)


def _dispatch(kind: str, model: Any, *, show_info: bool, options: Dict[str, Any]) -> VizResult:
    request = VizRequest(kind=kind, target=model, options=options)
    result = render(request)
    if show_info:
        _emit_info(kind, result)
    return result


def _emit_info(kind: str, result: VizResult) -> None:
    if result.warnings:
        print(f"[kd.viz.{kind}] warning: {'; '.join(result.warnings)}")
        return
    if result.paths:
        joined = ', '.join(str(path) for path in result.paths)
        print(f"[kd.viz.{kind}] saved: {joined}")
    elif result.figure is not None:
        print(f"[kd.viz.{kind}] figure ready")
    else:
        print(f"[kd.viz.{kind}] produced no output")


__all__ = [
    'plot_training_curve',
    'plot_validation_curve',
    'plot_search_evolution',
    'plot_optimization',
    'plot_residuals',
    'plot_field_comparison',
    'plot_time_slices',
    'plot_derivative_relationships',
    'plot_parity',
    'render_equation',
]
