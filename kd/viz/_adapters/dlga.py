"""DLGA model adapter bridging legacy helpers with the unified façade."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .. import dlga_viz
from .._contracts import (
    FieldComparisonData,
    OptimizationHistoryData,
    ParityPlotData,
    ResidualPlotData,
    RewardEvolutionData,
    TermContribution,
    TermRelationshipData,
    TimeSliceComparisonData,
)
from ..core import VizResult
from ..dlga_eq2latex import dlga_eq2latex
from ..equation_renderer import render_latex_to_image


class DLGAVizAdapter:
    capabilities: Iterable[str] = {
        'training_curve',
        'validation_curve',
        'search_evolution',
        'equation',
        'residual',
        'optimization',
        'field_comparison',
        'time_slices',
        'derivative_relationships',
        'parity',
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
            'time_slices': self._time_slices,
            'derivative_relationships': self._derivative_relationships,
            'parity': self._parity,
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
        metadata = {
            'generations': len(history),
        }
        return VizResult(intent='search_evolution', paths=[path], metadata=metadata)

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
        return VizResult(intent='field_comparison', paths=[path], metadata={'field_comparison': data})

    def _time_slices(self, model, ctx) -> VizResult:
        data = self._build_time_slice_data(ctx)
        if data is None:
            return VizResult(intent='time_slices', warnings=['Time slice data is required.'])

        import matplotlib.pyplot as plt

        slice_times = data.slice_times
        t_coords = data.t_coords.reshape(-1)
        indices = [int(np.argmin(np.abs(t_coords - t_val))) for t_val in slice_times]

        fig, axes = plt.subplots(
            1,
            len(indices),
            figsize=ctx.options.get('figsize', (max(4 * len(indices), 6), 4)),
            sharey=True,
        )
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        for ax, idx, t_val in zip(axes, indices, slice_times):
            ax.plot(data.x_coords, data.true_field[:, idx], 'b-', linewidth=2, label='Exact')
            ax.plot(data.x_coords, data.predicted_field[:, idx], 'r--', linewidth=2, label='Prediction')
            ax.set_xlabel(ctx.options.get('space_label', 'Space'))
            ax.set_title(f't = {t_val:.3f}')
            ax.grid(True, linestyle='--', alpha=0.5)

        axes[0].set_ylabel(ctx.options.get('field_label', 'u(x, t)'))
        legend_target = axes[len(indices) // 2] if len(indices) > 1 else axes[0]
        legend_target.legend()

        fig.tight_layout()

        _, path = self._resolve_output(ctx, 'time_slices_comparison.png')
        fig.savefig(str(path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        try:
            plt.close(fig)
        except Exception:
            pass

        enriched = TimeSliceComparisonData(
            x_coords=data.x_coords,
            t_coords=data.t_coords,
            true_field=data.true_field,
            predicted_field=data.predicted_field,
            slice_times=slice_times,
            metadata={**data.metadata, 'slice_indices': indices},
        )
        return VizResult(intent='time_slices', paths=[path], metadata={'time_slices': enriched})

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
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

        residual_flat = data.residuals.reshape(-1)
        summary = {
            'mean': float(np.mean(residual_flat)),
            'std': float(np.std(residual_flat)),
            'max_abs': float(np.max(np.abs(residual_flat))),
            'count': int(residual_flat.size),
        }
        return VizResult(intent='residual', paths=[path], metadata={'residual': data, 'summary': summary})

    def _derivative_relationships(self, model, ctx) -> VizResult:
        data = self._build_term_relationship_data(model, ctx)
        if data is None:
            return VizResult(
                intent='derivative_relationships',
                warnings=['Model metadata is incomplete; cannot derive term relationships.'],
            )

        import matplotlib.pyplot as plt

        terms = data.terms
        ncols = min(3, len(terms))
        nrows = int(np.ceil(len(terms) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=ctx.options.get('figsize', (5 * ncols, 4.5 * nrows)),
            squeeze=False,
        )
        axes_flat = axes.flatten()
        fig.suptitle(ctx.options.get('title', f'Relationship of Terms with {data.lhs_label}'), fontsize=16)

        for ax, term in zip(axes_flat, terms):
            sc = ax.scatter(term.values, data.lhs_values, alpha=0.35, s=20, c=term.values)
            ax.set_xlabel(f"Term: {term.label}")
            ax.set_ylabel(data.lhs_label)
            ax.set_title(f"Coefficient: {term.coefficient:.4f}")
            ax.grid(True, linestyle='--', alpha=0.5)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes_flat[len(terms):]:
            ax.set_visible(False)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        _, path = self._resolve_output(ctx, 'derivative_relationships.png')
        fig.savefig(str(path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

        return VizResult(intent='derivative_relationships', paths=[path], metadata={'term_relationships': data})

    def _parity(self, model, ctx) -> VizResult:
        data = self._build_parity_data(model)
        if data is None:
            return VizResult(intent='parity', warnings=['Model metadata is incomplete; cannot build parity plot.'])

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=ctx.options.get('figsize', (7, 7)))
        ax.scatter(data.predicted_values, data.actual_values, alpha=0.35, s=20, label='Prediction vs Actual')

        min_val = float(np.min([data.actual_values.min(), data.predicted_values.min()]))
        max_val = float(np.max([data.actual_values.max(), data.predicted_values.max()]))
        ref = np.linspace(min_val, max_val, 100)
        ax.plot(ref, ref, 'r--', linewidth=1.5, label='y = x')

        lhs_label = data.metadata.get('lhs_label', 'LHS')
        ax.set_xlabel(ctx.options.get('predicted_label', 'Predicted RHS Values'))
        ax.set_ylabel(ctx.options.get('actual_label', f'True LHS Values ({lhs_label})'))
        ax.set_title(ctx.options.get('title', 'Parity Plot'))
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', 'box')

        _, path = self._resolve_output(ctx, 'pde_parity_plot.png')
        fig.savefig(str(path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

        residual_stats = {
            'mean_residual': float(np.mean(data.residuals)),
            'max_abs_residual': float(np.max(np.abs(data.residuals))),
            'rmse': float(np.sqrt(np.mean(np.square(data.residuals)))),
        }
        return VizResult(intent='parity', paths=[path], metadata={'parity': data, 'summary': residual_stats})

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
                lines.extend(
                    ax_left.plot(
                        data.steps,
                        pop,
                        color='#1f77b4',
                        label='Population Size',
                        **style_kwargs,
                    )
                )
                labels.append('Population Size')
                ax_left.set_ylabel('Population Size', color='#1f77b4')
            if unique is not None:
                twin.plot(
                    data.steps,
                    unique,
                    color='#ff7f0e',
                    label='Unique Modules',
                    **style_kwargs,
                )
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

        if any(val is None for val in (x_coords, t_coords, true_field, predicted_field)):
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

    def _build_time_slice_data(self, ctx) -> Optional[TimeSliceComparisonData]:
        x_coords = ctx.options.get('x_coords')
        t_coords = ctx.options.get('t_coords')
        true_field = ctx.options.get('true_field')
        predicted_field = ctx.options.get('predicted_field')
        if any(val is None for val in (x_coords, t_coords, true_field, predicted_field)):
            return None

        slice_times = ctx.options.get('slice_times')
        t_arr = np.asarray(t_coords)
        if slice_times is None:
            requested = int(ctx.options.get('num_slices', 3) or 3)
            count = max(1, min(requested, t_arr.size))
            slice_times = np.linspace(t_arr.min(), t_arr.max(), num=count)

        metadata: Dict[str, Any] = {
            'title': ctx.options.get('title', 'Time Slice Comparison'),
        }
        return TimeSliceComparisonData(
            x_coords=x_coords,
            t_coords=t_arr,
            true_field=true_field,
            predicted_field=predicted_field,
            slice_times=slice_times,
            metadata=metadata,
        )

    def _build_term_relationship_data(self, model, ctx) -> Optional[TermRelationshipData]:
        chrom = getattr(model, 'Chrom', None)
        coef = getattr(model, 'coef', None)
        names = getattr(model, 'name', None)
        operators = getattr(model, 'user_operators', None)
        metadata = getattr(model, 'metadata', None)

        if not chrom or not coef or not names or metadata is None or not operators:
            return None

        best_chrom = chrom[0]
        lhs_name = names[0]
        if lhs_name not in metadata:
            return None

        lhs_values = np.asarray(metadata[lhs_name]).reshape(-1)
        raw_coefficients = coef[0] if isinstance(coef, (list, tuple)) else coef
        coefficients = np.asarray(raw_coefficients).reshape(-1)

        contributions: List[TermContribution] = []
        for idx, module in enumerate(best_chrom):
            if idx >= coefficients.size:
                break
            if any(gene >= len(operators) for gene in module):
                continue
            try:
                term_arrays = [np.asarray(metadata[operators[gene]]) for gene in module]
            except KeyError:
                continue
            term_values = np.prod(term_arrays, axis=0).reshape(-1)
            label = ' * '.join(operators[gene] for gene in module) or '1'
            contributions.append(
                TermContribution(
                    label=label,
                    values=term_values,
                    coefficient=float(coefficients[idx]),
                )
            )

        if not contributions:
            return None

        contributions.sort(key=lambda term: abs(term.coefficient), reverse=True)
        top_n = int(ctx.options.get('top_n_terms', 4) or 4)
        chosen = contributions[: max(1, top_n)]

        meta = {
            'lhs_label': lhs_name,
            'total_terms': len(contributions),
        }
        return TermRelationshipData(
            lhs_values=lhs_values,
            lhs_label=lhs_name,
            terms=chosen,
            metadata=meta,
        )

    def _build_parity_data(self, model) -> Optional[ParityPlotData]:
        chrom = getattr(model, 'Chrom', None)
        coef = getattr(model, 'coef', None)
        names = getattr(model, 'name', None)
        operators = getattr(model, 'user_operators', None)
        metadata = getattr(model, 'metadata', None)

        if not chrom or not coef or not names or metadata is None or not operators:
            return None

        best_chrom = chrom[0]
        lhs_name = names[0]
        if lhs_name not in metadata:
            return None

        lhs_values = np.asarray(metadata[lhs_name]).reshape(-1)
        raw_coefficients = coef[0] if isinstance(coef, (list, tuple)) else coef
        coefficients = np.asarray(raw_coefficients).reshape(-1)
        rhs_values = np.zeros_like(lhs_values)
        for idx, module in enumerate(best_chrom):
            if idx >= coefficients.size:
                break
            if any(gene >= len(operators) for gene in module):
                continue
            try:
                term_arrays = [np.asarray(metadata[operators[gene]]) for gene in module]
            except KeyError:
                continue
            term_values = np.prod(term_arrays, axis=0).reshape(-1)
            rhs_values = rhs_values + float(coefficients[idx]) * term_values

        return ParityPlotData.from_actual_predicted(
            lhs_values,
            rhs_values,
            metadata={'lhs_label': lhs_name},
        )

    def _build_reward_evolution(self, model) -> Optional[RewardEvolutionData]:
        history = getattr(model, 'evolution_history', None)
        if not history:
            return None

        steps = np.arange(len(history))
        best = []
        max_values = []
        mean_values = []
        cumulative_best = -np.inf
        for entry in history:
            rewards = entry.get('fitness_history') or entry.get('rewards')
            if rewards is None:
                return None
            rewards_arr = np.asarray(rewards)
            max_val = float(np.max(rewards_arr))
            mean_val = float(np.mean(rewards_arr))
            cumulative_best = max(cumulative_best, max_val)
            best.append(cumulative_best)
            max_values.append(max_val)
            mean_values.append(mean_val)

        return RewardEvolutionData(
            steps=steps,
            best_reward=np.asarray(best),
            max_reward=np.asarray(max_values),
            mean_reward=np.asarray(mean_values),
        )
