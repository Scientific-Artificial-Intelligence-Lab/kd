"""Model adapters bridging KD visualization façade to existing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from . import dlga_viz
from ._contracts import FieldComparisonData, OptimizationHistoryData, ResidualPlotData
from .core import VizResult
from .dlga_eq2latex import dlga_eq2latex
from .equation_renderer import render_latex_to_image
from .registry import register_adapter

__all__ = [
    'DLGAVizAdapter',
    'DSCVVizAdapter',
    'register_default_adapters',
]


class DLGAVizAdapter:
    capabilities = {
        'training_curve',
        'validation_curve',
        'search_evolution',
        'equation',
        'residual',
        'optimization',
        'field_comparison',
    }

    def __init__(self, *, subdir: str = 'dlga') -> None:
        self._subdir = subdir

    def render(self, request, ctx):  # type: ignore[override]
        model = request.target
        kind = request.kind
        handler = {
            'training_curve': self._training_curve,
            'validation_curve': self._validation_curve,
            'search_evolution': self._search_evolution,
            'equation': self._equation,
            'residual': self._residual,
            'optimization': self._optimization,
            'field_comparison': self._field_comparison,
        }.get(kind)
        if handler is None:
            return VizResult(
                intent=kind,
                warnings=[f"DLGA adapter does not support intent '{kind}'."],
                metadata={'capabilities': sorted(self.capabilities)},
            )
        return handler(model, ctx)

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------
    def _training_curve(self, model, ctx) -> VizResult:
        history = getattr(model, 'train_loss_history', None)
        if not history:
            return VizResult(intent='training_curve', warnings=['No training history found.'])
        out_dir, path = self._resolve_output(ctx, 'training_loss.png')
        dlga_viz.plot_training_loss(model, output_dir=str(out_dir))
        return VizResult(intent='training_curve', paths=[path], metadata={'points': len(history)})

    def _validation_curve(self, model, ctx) -> VizResult:
        history = getattr(model, 'val_loss_history', None)
        if not history:
            return VizResult(intent='validation_curve', warnings=['No validation history found.'])
        out_dir, path = self._resolve_output(ctx, 'validation_loss.png')
        dlga_viz.plot_validation_loss(model, output_dir=str(out_dir))
        return VizResult(intent='validation_curve', paths=[path], metadata={'points': len(history)})

    def _search_evolution(self, model, ctx) -> VizResult:
        history = getattr(model, 'evolution_history', None)
        if not history:
            return VizResult(intent='search_evolution', warnings=['No evolution history found.'])
        out_dir, path = self._resolve_output(ctx, 'evolution_analysis.png')
        dlga_viz.plot_evolution(model, output_dir=str(out_dir))
        return VizResult(
            intent='search_evolution',
            paths=[path],
            metadata={'generations': len(history)},
        )

    def _optimization(self, model, ctx) -> VizResult:
        data = self._build_optimization_data(model)
        if data is None:
            return VizResult(intent='optimization', warnings=['No optimization history available.'])

        fig = self._plot_optimization(data, ctx)
        _, path = self._resolve_output(ctx, 'optimization_analysis.png')
        fig.savefig(str(path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        try:  # pragma: no cover - defensive cleanup
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

        metadata = {
            'optimization': data,
            'summary': {
                'initial_objective': float(data.objective[0]),
                'final_objective': float(data.objective[-1]),
                'best_objective': float(np.min(data.objective)),
            },
        }
        return VizResult(intent='optimization', paths=[path], metadata=metadata)

    def _field_comparison(self, model, ctx) -> VizResult:
        data = self._build_field_comparison_data(ctx)
        if data is None:
            return VizResult(intent='field_comparison', warnings=['Field comparison data is required.'])

        _, path = self._resolve_output(ctx, 'field_comparison.png')
        dlga_viz.plot_pde_comparison(
            data.x_coords,
            data.t_coords,
            data.true_field,
            data.predicted_field,
            output_dir=str(path.parent),
        )
        produced = path.parent / 'pde_comparison.png'
        if produced.exists() and not path.exists():
            produced.rename(path)
        metadata = {
            'field_comparison': data,
        }
        return VizResult(intent='field_comparison', paths=[path], metadata=metadata)

    def _equation(self, model, ctx) -> VizResult:
        chrom = getattr(model, 'Chrom', None)
        coef = getattr(model, 'coef', None)
        names = getattr(model, 'name', None)
        if not chrom or not coef or not names:
            return VizResult(intent='equation', warnings=['No discovered equation available.'])

        index = int(ctx.options.get('solution_index', 0) or 0)
        try:
            chromosome = chrom[index]
            coefficients = coef[index]
            lhs = names[index]
        except (IndexError, TypeError):
            return VizResult(intent='equation', warnings=[f'Solution index {index} is invalid.'])

        operators = getattr(model, 'user_operators', None)
        latex = dlga_eq2latex(
            chromosome=chromosome,
            coefficients=coefficients,
            lhs_name_str=lhs,
            operator_names=operators,
        )
        _, path = self._resolve_output(ctx, 'equation.png')
        render_latex_to_image(latex, output_path=str(path), font_size=ctx.options.get('font_size', 16))
        return VizResult(intent='equation', paths=[path], metadata={'latex': latex, 'index': index})

    def _residual(self, model, ctx) -> VizResult:
        data = self._build_residual_data(ctx)
        if data is None:
            return VizResult(intent='residual', warnings=['Residual data is required.'])

        fig = self._plot_residual(data, ctx)
        _, path = self._resolve_output(ctx, 'residual_analysis.png')
        fig.savefig(str(path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        try:  # close figure to avoid GUI leakage
            import matplotlib.pyplot as plt  # noqa: WPS433
            plt.close(fig)
        except Exception:  # pragma: no cover
            pass

        residual_flat = data.residuals.reshape(-1)
        summary = {
            'mean': float(np.mean(residual_flat)),
            'std': float(np.std(residual_flat)),
            'max_abs': float(np.max(np.abs(residual_flat))),
            'count': int(residual_flat.size),
        }
        return VizResult(
            intent='residual',
            paths=[path],
            metadata={'residual': data, 'summary': summary},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_output(self, ctx, filename: str) -> Tuple[Path, Path]:
        base = ctx.options.get('output_dir')
        if base is not None:
            base_path = Path(base)
            if self._subdir:
                base_path = base_path / self._subdir
            base_path.mkdir(parents=True, exist_ok=True)
            return base_path, base_path / filename
        relative = f"{self._subdir}/{filename}" if self._subdir else filename
        path = ctx.save_path(relative)
        return path.parent, path

    def _build_residual_data(self, ctx) -> Optional[ResidualPlotData]:
        actual = ctx.options.get('actual')
        if actual is None:
            actual = ctx.options.get('actual_values')
        predicted = ctx.options.get('predicted')
        if predicted is None:
            predicted = ctx.options.get('predicted_values')
        if actual is None or predicted is None:
            return None
        coords = ctx.options.get('coordinates')
        residuals = ctx.options.get('residuals')
        metadata: Dict[str, str] = {
            'title': ctx.options.get('title', 'Residual and Parity Analysis'),
            'parity_title': ctx.options.get('parity_title', 'Parity Plot'),
            'residual_title': ctx.options.get('residual_title', 'Residual Distribution'),
        }
        if residuals is None:
            return ResidualPlotData.from_actual_predicted(
                actual,
                predicted,
                input_coordinates=coords,
                metadata=metadata,
            )
        return ResidualPlotData(
            actual=np.asarray(actual),
            predicted=np.asarray(predicted),
            residuals=np.asarray(residuals),
            input_coordinates=coords,
            metadata=metadata,
        )

    def _plot_residual(self, data: ResidualPlotData, ctx):
        import matplotlib.pyplot as plt

        actual = data.actual.reshape(-1)
        predicted = data.predicted.reshape(-1)
        residuals = data.residuals.reshape(-1)

        fig = plt.figure(figsize=ctx.options.get('figsize', (12, 5)))
        scatter_kwargs = {
            'alpha': 0.65,
            's': 20,
            'edgecolors': 'none',
        }

        ax1 = fig.add_subplot(1, 2, 1)
        coords = data.input_coordinates
        if coords is not None:
            coord_arr = np.asarray(coords)
            color = coord_arr[:, 0] if coord_arr.ndim > 1 else coord_arr
            sc = ax1.scatter(
                actual,
                predicted,
                c=color,
                cmap=ctx.options.get('cmap', 'viridis'),
                **scatter_kwargs,
            )
            fig.colorbar(sc, ax=ax1, label=ctx.options.get('colorbar_label', 'coordinate'))
        else:
            ax1.scatter(actual, predicted, color='#1f77b4', **scatter_kwargs)

        min_val = float(np.min(np.concatenate([actual, predicted])))
        max_val = float(np.max(np.concatenate([actual, predicted])))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        ax1.set(
            xlabel=ctx.options.get('actual_label', 'Actual'),
            ylabel=ctx.options.get('predicted_label', 'Predicted'),
            title=data.metadata.get('parity_title', 'Parity Plot'),
        )

        ax2 = fig.add_subplot(1, 2, 2)
        bins = int(ctx.options.get('bins', 40))
        ax2.hist(
            residuals,
            bins=bins,
            color='#1f77b4',
            alpha=0.7,
            edgecolor='black',
        )
        ax2.set(
            xlabel=ctx.options.get('residual_label', 'Residual'),
            ylabel='Frequency',
            title=data.metadata.get('residual_title', 'Residual Distribution'),
        )
        ax2.axvline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.6)

        return fig

    def _build_optimization_data(self, model) -> Optional[OptimizationHistoryData]:
        history = getattr(model, 'evolution_history', None)
        if not history:
            return None

        steps = np.arange(len(history))
        objective = []
        complexity = []
        population = []
        unique = []

        for entry in history:
            value = entry.get('fitness')
            if value is None:
                value = entry.get('objective')
            if value is None:
                return None
            objective.append(value)
            complexity.append(entry.get('complexity'))
            population.append(entry.get('population_size'))
            unique.append(entry.get('unique_modules'))

        def sanitize(values):
            if any(v is None for v in values):
                return None
            return np.asarray(values)

        return OptimizationHistoryData(
            steps=steps,
            objective=np.asarray(objective),
            complexity=sanitize(complexity),
            population_size=sanitize(population),
            unique_modules=sanitize(unique),
        )

    def _plot_optimization(self, data: OptimizationHistoryData, ctx):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=ctx.options.get('figsize', (12, 5)))
        ax_left, ax_right = axes

        style_kwargs = {
            'marker': 'o',
            'markersize': 3,
            'alpha': 0.7,
        }

        pop = data.population_size
        unique = data.unique_modules
        if pop is not None or unique is not None:
            twin = ax_left.twinx()
            lines = []
            labels = []
            if pop is not None:
                l1 = ax_left.plot(
                    data.steps,
                    pop,
                    color='#1f77b4',
                    label='Population Size',
                    **style_kwargs,
                )
                lines.extend(l1)
                labels.append('Population Size')
                ax_left.set_ylabel('Population Size', color='#1f77b4')
            if unique is not None:
                l2 = twin.plot(
                    data.steps,
                    unique,
                    color='#ff7f0e',
                    label='Unique Modules',
                    **style_kwargs,
                )
                lines.extend(l2)
                labels.append('Unique Modules')
                twin.set_ylabel('Unique Modules', color='#ff7f0e')
            ax_left.legend(lines, labels, loc='upper right')
        else:
            ax_left.plot(data.steps, data.objective, color='#1f77b4', **style_kwargs)

        ax_left.set_xlabel('Step')
        ax_left.set_title('Optimization Diversity')
        ax_left.grid(True, linestyle='--', alpha=0.5)

        ax_right.plot(
            data.steps,
            data.objective,
            color='#2ca02c',
            linewidth=2,
            marker='o',
            markersize=3,
            alpha=0.8,
        )
        ax_right.set_xlabel('Step')
        ax_right.set_ylabel('Objective')
        ax_right.set_title('Objective Evolution')
        ax_right.grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()
        return fig

    def _build_field_comparison_data(self, ctx) -> Optional[FieldComparisonData]:
        x_coords = ctx.options.get('x_coords')
        t_coords = ctx.options.get('t_coords')
        true_field = ctx.options.get('true_field')
        if true_field is None:
            true_field = ctx.options.get('u_true')
        predicted_field = ctx.options.get('predicted_field')
        if predicted_field is None:
            predicted_field = ctx.options.get('u_pred')

        if x_coords is None or t_coords is None or true_field is None or predicted_field is None:
            return None

        residual_field = ctx.options.get('residual_field')
        metadata: Dict[str, Any] = {
            'title': ctx.options.get('title', 'Field Comparison'),
            'true_label': ctx.options.get('true_label', 'True Field'),
            'pred_label': ctx.options.get('pred_label', 'Predicted Field'),
        }
        return FieldComparisonData(
            x_coords=x_coords,
            t_coords=t_coords,
            true_field=true_field,
            predicted_field=predicted_field,
            residual_field=residual_field,
            metadata=metadata,
        )


class DSCVVizAdapter:
    capabilities: Iterable[str] = ()

    def render(self, request, ctx):  # type: ignore[override]
        return VizResult(
            intent=request.kind,
            warnings=['KD_DSCV visualization not integrated yet.'],
        )


def register_default_adapters() -> None:
    try:
        from kd.model.kd_dlga import KD_DLGA  # type: ignore
    except Exception:  # pragma: no cover
        KD_DLGA = None  # type: ignore
    else:
        register_adapter(KD_DLGA, DLGAVizAdapter())

    try:
        from kd.model.kd_dscv import KD_DSCV  # type: ignore
    except Exception:  # pragma: no cover
        KD_DSCV = None  # type: ignore
    else:
        register_adapter(KD_DSCV, DSCVVizAdapter())
