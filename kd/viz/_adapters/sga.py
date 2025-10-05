"""SGA model visualization adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..equation_renderer import render_latex_to_image
from ..core import FieldComparisonData, TimeSliceComparisonData, VizResult


class SGAVizAdapter:
    capabilities: Iterable[str] = {
        'equation',
        'field_comparison',
        'time_slices',
    }

    def __init__(self, *, subdir: str = 'sga') -> None:
        self._subdir = subdir

    def render(self, request, ctx):  # type: ignore[override]
        handler = {
            'equation': self._equation,
            'field_comparison': self._field_comparison,
            'time_slices': self._time_slices,
        }.get(request.kind)

        if handler is None:
            return VizResult(
                intent=request.kind,
                warnings=[f"SGA adapter does not support intent '{request.kind}'."],
            )
        return handler(request.target, ctx)

    def _resolve_output(self, ctx, filename: str) -> Path:
        base = ctx.options.get('output_dir')
        if base is None:
            return ctx.save_path(f'{self._subdir}/{filename}')
        base_path = Path(base) / self._subdir
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path / filename

    def _equation(self, model, ctx) -> VizResult:
        if not hasattr(model, 'equation_latex'):
            return VizResult(intent='equation', warnings=['Model does not expose equation LaTeX helper.'])

        try:
            latex_plain = model.equation_latex()
            structure_plain = model.equation_latex(include_coefficients=False)
        except Exception as exc:  # pragma: no cover - defensive fallback
            return VizResult(intent='equation', warnings=[f'Failed to obtain SGA equation: {exc}'])

        output_path = self._resolve_output(ctx, 'equation.png')
        structure_path = self._resolve_output(ctx, 'equation_structure.png')
        try:
            render_latex_to_image(
                latex_plain,
                output_path=str(output_path),
                font_size=ctx.options.get('font_size', 16),
                show=False,
            )
            render_latex_to_image(
                structure_plain,
                output_path=str(structure_path),
                font_size=ctx.options.get('font_size', 16),
                show=False,
            )
        except Exception as exc:  # pragma: no cover - runtime dependency issues
            return VizResult(intent='equation', warnings=[f'Failed to render LaTeX equation: {exc}'])

        metadata = {
            'latex': latex_plain,
            'structure_latex': structure_plain,
        }
        return VizResult(intent='equation', paths=[output_path, structure_path], metadata=metadata)

    def _field_comparison(self, model, ctx) -> VizResult:
        context = getattr(model, 'context_', None)
        if context is None:
            return VizResult(intent='field_comparison', warnings=['Model context is unavailable; call fit() first.'])

        data = self._build_field_comparison_data(context)
        if data is None:
            return VizResult(intent='field_comparison', warnings=['Context does not expose metadata/original fields.'])

        fig, axes = plt.subplots(1, 2, figsize=ctx.options.get('figsize', (14, 5)), sharey=True)
        fig.suptitle(ctx.options.get('title', 'Metadata vs Original Field'), fontsize=16)

        vmin = float(np.nanmin([data.true_field.min(), data.predicted_field.min()]))
        vmax = float(np.nanmax([data.true_field.max(), data.predicted_field.max()]))

        im0 = axes[0].pcolormesh(
            data.t_coords,
            data.x_coords,
            data.predicted_field,
            cmap=ctx.options.get('cmap', 'viridis'),
            vmin=vmin,
            vmax=vmax,
            shading='auto',
        )
        axes[0].set_title('Metadata Field')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Space')
        fig.colorbar(im0, ax=axes[0], label='Value')

        im1 = axes[1].pcolormesh(
            data.t_coords,
            data.x_coords,
            data.true_field,
            cmap=ctx.options.get('cmap', 'viridis'),
            vmin=vmin,
            vmax=vmax,
            shading='auto',
        )
        axes[1].set_title('Original Field')
        axes[1].set_xlabel('Time')
        fig.colorbar(im1, ax=axes[1], label='Value')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_path = self._resolve_output(ctx, 'field_comparison.png')
        fig.savefig(str(output_path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        plt.close(fig)

        metadata = {
            'summary': {
                'predicted_shape': data.predicted_field.shape,
                'original_shape': data.true_field.shape,
                'vmin': vmin,
                'vmax': vmax,
            },
            'field_comparison_data': data,
        }
        return VizResult(intent='field_comparison', paths=[output_path], metadata=metadata)

    def _time_slices(self, model, ctx) -> VizResult:
        context = getattr(model, 'context_', None)
        if context is None:
            return VizResult(intent='time_slices', warnings=['Model context is unavailable; call fit() first.'])

        baseline = self._build_field_comparison_data(context)
        if baseline is None:
            return VizResult(intent='time_slices', warnings=['Context does not expose metadata/original fields.'])

        slice_option = ctx.options.get('slice_times')
        if slice_option is None:
            slice_count = int(ctx.options.get('slice_count', 3))
            slice_count = max(1, slice_count)
            t_min = float(baseline.t_coords.min())
            t_max = float(baseline.t_coords.max())
            raw_requested = np.linspace(t_min, t_max, slice_count)
        else:
            raw_requested = np.asarray(slice_option, dtype=float)
            if raw_requested.size == 0:
                raw_requested = baseline.t_coords

        requested_times = np.asarray(raw_requested, dtype=float)

        # Map requested times to nearest available grid points
        t_coords = baseline.t_coords.reshape(-1)
        t_min, t_max = float(t_coords.min()), float(t_coords.max())
        requested_times = np.clip(requested_times, t_min, t_max)
        requested_times = np.unique(np.ravel(requested_times))
        if requested_times.size == 0:
            requested_times = t_coords

        indices = [int(np.argmin(np.abs(t_coords - t_val))) for t_val in requested_times]
        actual_times = np.asarray([t_coords[idx] for idx in indices], dtype=float)

        t_coords = baseline.t_coords.reshape(-1)
        x_coords = baseline.x_coords.reshape(-1)

        fig, axes = plt.subplots(
            1,
            len(indices),
            figsize=ctx.options.get('figsize', (max(4 * len(indices), 6), 4)),
            sharey=True,
        )
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        for ax, idx, t_val in zip(axes, indices, actual_times):
            ax.plot(x_coords, baseline.true_field[:, idx], 'b-', linewidth=2, label='Original')
            ax.plot(x_coords, baseline.predicted_field[:, idx], 'r--', linewidth=2, label='Metadata')
            ax.set_xlabel(ctx.options.get('space_label', 'Space'))
            ax.set_title(f't = {t_val:.3f}')
            ax.grid(True, linestyle='--', alpha=0.5)

        axes[0].set_ylabel(ctx.options.get('field_label', 'u(x, t)'))
        legend_target = axes[len(indices) // 2] if len(indices) > 1 else axes[0]
        legend_target.legend()

        fig.tight_layout()

        output_path = self._resolve_output(ctx, 'time_slices_comparison.png')
        fig.savefig(str(output_path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        plt.close(fig)

        metadata = {
            'time_slices_summary': {
                'requested_slice_times': np.asarray(raw_requested, dtype=float).tolist(),
                'actual_slice_times': actual_times.tolist(),
                'space_points': baseline.x_coords.shape[0],
                'time_points': baseline.t_coords.shape[0],
            },
            'time_slices_data': TimeSliceComparisonData(
                x_coords=baseline.x_coords,
                t_coords=baseline.t_coords,
                true_field=baseline.true_field,
                predicted_field=baseline.predicted_field,
                slice_times=actual_times,
            ),
        }
        return VizResult(intent='time_slices', paths=[output_path], metadata=metadata)

    def _build_field_comparison_data(self, context) -> FieldComparisonData | None:
        metadata_field = getattr(context, 'u', None)
        original_field = getattr(context, 'u_origin', None)
        if metadata_field is None or original_field is None:
            return None

        metadata_field = np.asarray(metadata_field)
        original_field = np.asarray(original_field)
        if metadata_field.ndim != 2 or original_field.ndim != 2:
            return None

        x_coords = self._extract_axis(getattr(context, 'x', None), metadata_field.shape[0], axis='space')
        t_coords = self._extract_axis(getattr(context, 't', None), metadata_field.shape[1], axis='time')

        return FieldComparisonData(
            x_coords=x_coords,
            t_coords=t_coords,
            true_field=original_field,
            predicted_field=metadata_field,
        )

    def _extract_axis(self, grid, expected_size: int, *, axis: str) -> np.ndarray:
        if grid is None:
            return np.linspace(0.0, 1.0, expected_size)
        arr = np.asarray(grid)
        if arr.ndim == 2:
            if axis == 'space':
                candidate = arr[:, 0] if arr.shape[1] >= 1 else None
            else:
                candidate = arr[0, :] if arr.shape[0] >= 1 else None
            if candidate is not None and candidate.size == expected_size:
                return candidate
        if arr.ndim == 1:
            if arr.size == expected_size:
                return arr
        return np.linspace(0.0, 1.0, expected_size)


__all__ = ['SGAVizAdapter']
