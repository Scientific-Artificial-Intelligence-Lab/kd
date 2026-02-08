"""DSCV model adapter skeleton for the unified visualization façade."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .. import dscv_viz
from .._contracts import (
    FieldComparisonData,
    ParityPlotData,
    RewardEvolutionData,
    ResidualPlotData,
)
from ..core import VizResult
from ..discover_eq2latex import discover_program_to_latex
from ..equation_renderer import render_latex_to_image


class DSCVVizAdapter:
    """Placeholder DSCV adapter exposing planned capabilities.

    Each intent currently returns a structured warning so that downstream
    callers can detect support without triggering legacy visualizers. Actual
    rendering logic will be implemented incrementally.
    """

    capabilities: Iterable[str] = {
        'search_evolution',
        'density',
        'tree',
        'equation',
        'residual',
        'field_comparison',
        'parity',
        'spr_residual',
        'spr_field_comparison',
    }

    def render(self, request, ctx):  # type: ignore[override]
        handler = {
            'search_evolution': self._search_evolution,
            'density': self._density,
            'tree': self._tree,
            'equation': self._equation,
            'residual': self._residual,
            'field_comparison': self._field_comparison,
            'parity': self._parity,
            'spr_residual': self._spr_residual,
            'spr_field_comparison': self._spr_field_comparison,
        }.get(request.kind)

        if handler is None:
            return VizResult(
                intent=request.kind,
                warnings=[
                    "DSCV adapter stub: intent '%s' not implemented yet." % request.kind
                ],
                metadata={'capabilities': sorted(self.capabilities)},
            )

        return handler(request.target, ctx)

    # ------------------------------------------------------------------
    # Implemented intents
    # ------------------------------------------------------------------
    def _search_evolution(self, model, ctx) -> VizResult:
        data = self._build_reward_evolution(model)
        if data is None:
            return VizResult(
                intent='search_evolution',
                warnings=['No reward history available for search evolution plot.'],
            )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=ctx.options.get('figsize', (10, 6)))
        ax.plot(data.steps, data.best_reward, color='black', linewidth=2, label='Best Reward')
        if data.max_reward is not None:
            ax.plot(
                data.steps,
                data.max_reward,
                color='#F47E62',
                linestyle='-.',
                linewidth=2,
                label='Maximum Reward',
            )
        if data.mean_reward is not None:
            ax.plot(
                data.steps,
                data.mean_reward,
                color='#4F8FBA',
                linestyle='--',
                linewidth=2,
                label='Average Reward',
            )

        ax.set_xlabel('Iteration Count')
        ax.set_ylabel('Reward Value')
        ax.set_title(ctx.options.get('title', 'Reward Evolution'))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', frameon=False)

        _, path = self._resolve_output(ctx, 'reward_evolution.png')
        fig.savefig(str(path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        try:
            plt.close(fig)
        except Exception:
            pass

        return VizResult(
            intent='search_evolution',
            paths=[path],
            metadata={'reward_evolution': data},
        )

    def _density(self, model, ctx) -> VizResult:
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
        except ImportError:  # noqa: WPS433
            return VizResult(
                intent='density',
                warnings=['seaborn is required for reward density plots.'],
            )

        history = getattr(getattr(model, 'searcher', None), 'r_history', None)
        if not history:
            return VizResult(
                intent='density',
                warnings=['No reward history available for density plot.'],
            )

        epoches = ctx.options.get('epoches')
        data = history if epoches is None else [history[i] for i in epoches if i < len(history)]
        if not data:
            return VizResult(intent='density', warnings=['No data available for requested epochs.'])

        fig, ax = plt.subplots(figsize=ctx.options.get('figsize', (8, 5)))
        for idx, rewards in enumerate(data):
            sns.kdeplot(
                np.asarray(rewards),
                ax=ax,
                label=f'Epoch {epoches[idx] if epoches else idx}',
                fill=False,
            )

        ax.set_xlabel('Reward')
        ax.set_ylabel('Density')
        ax.set_title(ctx.options.get('title', 'Reward Density Evolution'))
        ax.grid(True, linestyle='--', alpha=0.4)
        if len(data) > 1:
            ax.legend(loc='best', frameon=False)

        _, path = self._resolve_output(ctx, 'reward_density.png')
        fig.savefig(str(path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        try:
            plt.close(fig)
        except Exception:
            pass

        metadata = {
            'epoches': epoches if epoches is not None else list(range(len(data))),
        }
        return VizResult(intent='density', paths=[path], metadata=metadata)

    def _tree(self, model, ctx) -> VizResult:
        try:
            import graphviz  # noqa: F401
        except ImportError:
            return VizResult(
                intent='tree',
                warnings=['graphviz is required for expression tree rendering.'],
            )

        program = self._get_best_program(model)
        if program is None:
            return VizResult(intent='tree', warnings=['No best program available for tree rendering.'])

        searcher = getattr(model, 'searcher', None)
        plotter = getattr(searcher, 'plotter', None)
        if plotter is None or not hasattr(plotter, 'tree_plot'):
            return VizResult(intent='tree', warnings=['Tree plotter is unavailable on the model searcher.'])

        graph = plotter.tree_plot(program)
        if graph is None:
            return VizResult(intent='tree', warnings=['Tree plotter did not return a graph object.'])

        try:
            png_bytes = graph.pipe(format='png')
        except Exception as exc:  # pragma: no cover - runtime dependency issues
            return VizResult(
                intent='tree',
                warnings=[f'Failed to render expression tree: {exc}'],
            )

        _, path = self._resolve_output(ctx, 'expression_tree.png')
        path.write_bytes(png_bytes)

        metadata = {
            'expression': getattr(program, 'str_expression', None),
        }
        return VizResult(intent='tree', paths=[path], metadata=metadata)

    def _equation(self, model, ctx) -> VizResult:
        program = self._get_best_program(model)
        if program is None:
            return VizResult(intent='equation', warnings=['No best program available for equation rendering.'])

        try:
            latex = discover_program_to_latex(program)
        except Exception as exc:  # pragma: no cover - dependent on program internals
            return VizResult(intent='equation', warnings=[f'Failed to convert program to LaTeX: {exc}'])

        _, path = self._resolve_output(ctx, 'equation.png')
        try:
            render_latex_to_image(
                latex,
                output_path=str(path),
                font_size=ctx.options.get('font_size', 16),
                show=False,
            )
        except Exception as exc:  # pragma: no cover - runtime dependency issues
            return VizResult(intent='equation', warnings=[f'Failed to render LaTeX equation: {exc}'])

        return VizResult(intent='equation', paths=[path], metadata={'latex': latex})

    def _residual(self, model, ctx) -> VizResult:
        program = self._get_best_program(model)
        if program is None:
            return VizResult(intent='residual', warnings=['No best program available for residual analysis.'])

        try:
            fields = dscv_viz._calculate_pde_fields(model, program)
        except Exception as exc:  # pragma: no cover - heavy upstream dependency
            return VizResult(intent='residual', warnings=[f'Unable to compute PDE fields: {exc}'])

        n_spatial_dims = fields.get('n_spatial_dims', 1)
        residuals = np.asarray(fields['residual']).reshape(-1)
        actual = np.asarray(fields['ut_grid']).reshape(-1)
        predicted = np.asarray(fields['y_hat_grid']).reshape(-1)

        try:
            coords = fields.get('coords')
            if coords is not None:
                coords = np.asarray(coords)
            residual_data = ResidualPlotData.from_actual_predicted(
                actual, predicted,
                input_coordinates=coords,
            )
        except Exception as exc:
            return VizResult(intent='residual', warnings=[f'Failed to build residual data: {exc}'])

        import matplotlib.pyplot as plt

        if n_spatial_dims == 1:
            fig = self._residual_plot_1d(residuals, coords, ctx)
        else:
            fig = self._residual_plot_nd(fields, residuals, ctx)

        _, path = self._resolve_output(ctx, 'residual_analysis.png')
        fig.savefig(str(path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        try:
            plt.close(fig)
        except Exception:
            pass

        summary = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'max_abs': float(np.max(np.abs(residuals))),
            'count': int(residuals.size),
        }
        return VizResult(intent='residual', paths=[path], metadata={'residual': residual_data, 'summary': summary})

    def _residual_plot_1d(self, residuals, coords, ctx):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=ctx.options.get('figsize', (12, 5)))
        ax1 = fig.add_subplot(1, 2, 1)
        sc = ax1.scatter(
            coords[:, 1], coords[:, 0], c=residuals,
            cmap=ctx.options.get('cmap', 'coolwarm'), s=15, alpha=0.8,
        )
        fig.colorbar(sc, ax=ax1, label='Residual ($u_t$ - RHS)')
        ax1.set_xlabel('Time (index)')
        ax1.set_ylabel('Space')
        ax1.set_title('Spatiotemporal Residual Distribution')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist(residuals, bins=int(ctx.options.get('bins', 40)), color='#1f77b4', alpha=0.7, edgecolor='black', density=True)
        ax2.set_xlabel('Residual Value')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Residual Distribution')
        ax2.axvline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
        fig.tight_layout()
        return fig

    def _residual_plot_nd(self, fields, residuals, ctx):
        """2D spatial: heatmap of residuals at a selected time step + histogram."""
        import matplotlib.pyplot as plt

        n_spatial_dims = fields.get('n_spatial_dims', 2)
        if n_spatial_dims > 2:
            # Fallback: histogram only for >2D spatial
            fig, ax = plt.subplots(figsize=ctx.options.get('figsize', (7, 5)))
            ax.hist(residuals, bins=int(ctx.options.get('bins', 40)), color='#1f77b4', alpha=0.7, edgecolor='black', density=True)
            ax.set_xlabel('Residual Value')
            ax.set_ylabel('Probability Density')
            ax.set_title('Residual Distribution (>2D spatial — heatmap not available)')
            ax.axvline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
            return fig

        ut_grid = fields['ut_grid']
        y_hat_grid = fields['y_hat_grid']
        spatial_coords = fields['spatial_coords_list']
        t_axis = fields['t_axis']

        residual_grid = ut_grid - y_hat_grid
        # Pick middle time step
        t_idx = residual_grid.shape[0] // 2

        fig = plt.figure(figsize=ctx.options.get('figsize', (12, 5)))
        ax1 = fig.add_subplot(1, 2, 1)

        y_coords = spatial_coords[0]
        x_coords = spatial_coords[1]
        residual_slice = residual_grid[t_idx]
        vabs = float(np.max(np.abs(residual_slice)))
        im = ax1.pcolormesh(
            x_coords, y_coords, residual_slice,
            cmap=ctx.options.get('cmap', 'coolwarm'),
            vmin=-vabs, vmax=vabs, shading='auto',
        )
        fig.colorbar(im, ax=ax1, label='Residual')
        ax1.set_title(f'Residual (t={t_axis[t_idx]:.2f})')
        ax1.set_xlabel('$x_2$')
        ax1.set_ylabel('$x_1$')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist(residuals, bins=int(ctx.options.get('bins', 40)), color='#1f77b4', alpha=0.7, edgecolor='black', density=True)
        ax2.set_xlabel('Residual Value')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Residual Distribution')
        ax2.axvline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
        fig.tight_layout()
        return fig

    def _field_comparison(self, model, ctx) -> VizResult:
        program = self._get_best_program(model)
        if program is None:
            return VizResult(intent='field_comparison', warnings=['No best program available for field comparison.'])

        try:
            fields = dscv_viz._calculate_pde_fields(model, program)
        except Exception as exc:
            return VizResult(intent='field_comparison', warnings=[f'Unable to compute PDE fields: {exc}'])

        n_spatial_dims = fields.get('n_spatial_dims', 1)

        try:
            t_axis = np.asarray(fields['t_axis'])
            true_field = np.asarray(fields['ut_grid'])
            pred_field = np.asarray(fields['y_hat_grid'])

            if n_spatial_dims == 1:
                x_axis = np.asarray(fields['x_axis'])
                data = FieldComparisonData(
                    x_coords=x_axis, t_coords=t_axis,
                    true_field=true_field, predicted_field=pred_field,
                )
            else:
                spatial_coords = fields['spatial_coords_list']
                # DSCV N-D returns (nt, *spatial) — transpose to (*spatial, t) for contract
                n_dims = true_field.ndim
                perm = list(range(1, n_dims)) + [0]
                data = FieldComparisonData(
                    spatial_coords=spatial_coords, t_coords=t_axis,
                    true_field=np.transpose(true_field, perm),
                    predicted_field=np.transpose(pred_field, perm),
                )
        except Exception as exc:
            return VizResult(intent='field_comparison', warnings=[f'Failed to build field comparison data: {exc}'])

        import matplotlib.pyplot as plt

        if n_spatial_dims == 1:
            return self._field_comparison_1d(data, ctx)

        # --- 2D spatial: show selected time steps as 2D heatmaps ---
        return self._field_comparison_nd(data, ctx)

    def _field_comparison_1d(self, data: FieldComparisonData, ctx) -> VizResult:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=ctx.options.get('figsize', (14, 6)), sharey=True)
        fig.suptitle(ctx.options.get('title', 'Predicted vs True Field'), fontsize=16)

        vmin = float(np.min([data.true_field.min(), data.predicted_field.min()]))
        vmax = float(np.max([data.true_field.max(), data.predicted_field.max()]))

        im0 = axes[0].pcolormesh(data.t_coords, data.x_coords, data.true_field, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        fig.colorbar(im0, ax=axes[0], label='Value')
        axes[0].set_title('True Field ($u_t$)')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Space')

        im1 = axes[1].pcolormesh(data.t_coords, data.x_coords, data.predicted_field, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        fig.colorbar(im1, ax=axes[1], label='Value')
        axes[1].set_title('Predicted Field (RHS)')
        axes[1].set_xlabel('Time')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        _, path = self._resolve_output(ctx, 'field_comparison.png')
        fig.savefig(str(path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        try:
            plt.close(fig)
        except Exception:
            pass

        return VizResult(intent='field_comparison', paths=[path], metadata={'field_comparison': data})

    def _field_comparison_nd(self, data: FieldComparisonData, ctx) -> VizResult:
        """2D spatial field comparison: show 2D heatmaps at selected time steps."""
        import matplotlib.pyplot as plt

        if data.n_spatial_dims > 2:
            return VizResult(
                intent='field_comparison',
                warnings=[f'field_comparison currently supports up to 2D spatial (got {data.n_spatial_dims}D).'],
            )

        nt = data.t_coords.size
        # Pick up to 3 time steps: first, middle, last
        if nt <= 3:
            t_indices = list(range(nt))
        else:
            t_indices = [0, nt // 2, nt - 1]

        n_cols = len(t_indices)
        fig, axes = plt.subplots(2, n_cols, figsize=ctx.options.get('figsize', (5 * n_cols, 8)))
        fig.suptitle(ctx.options.get('title', 'Predicted vs True Field (2D spatial)'), fontsize=16)
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        vmin = float(np.min([data.true_field.min(), data.predicted_field.min()]))
        vmax = float(np.max([data.true_field.max(), data.predicted_field.max()]))
        cmap = ctx.options.get('cmap', 'viridis')

        y_coords = data.spatial_coords[0]
        x_coords = data.spatial_coords[1]

        for col_idx, t_idx in enumerate(t_indices):
            true_slice = data.true_field[..., t_idx]
            pred_slice = data.predicted_field[..., t_idx]

            im0 = axes[0, col_idx].pcolormesh(
                x_coords, y_coords, true_slice,
                cmap=cmap, vmin=vmin, vmax=vmax, shading='auto',
            )
            axes[0, col_idx].set_title(f'True (t={data.t_coords[t_idx]:.2f})')
            axes[0, col_idx].set_xlabel('$x_2$')
            if col_idx == 0:
                axes[0, col_idx].set_ylabel('$x_1$')

            im1 = axes[1, col_idx].pcolormesh(
                x_coords, y_coords, pred_slice,
                cmap=cmap, vmin=vmin, vmax=vmax, shading='auto',
            )
            axes[1, col_idx].set_title(f'Predicted (t={data.t_coords[t_idx]:.2f})')
            axes[1, col_idx].set_xlabel('$x_2$')
            if col_idx == 0:
                axes[1, col_idx].set_ylabel('$x_1$')

        fig.colorbar(im0, ax=axes.ravel().tolist(), label='Value', shrink=0.8)
        fig.tight_layout(rect=[0, 0.03, 0.92, 0.95])
        _, path = self._resolve_output(ctx, 'field_comparison.png')
        fig.savefig(str(path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        try:
            plt.close(fig)
        except Exception:
            pass

        return VizResult(intent='field_comparison', paths=[path], metadata={'field_comparison': data})

    def _parity(self, model, ctx) -> VizResult:
        program = self._get_best_program(model)
        if program is None:
            return VizResult(intent='parity', warnings=['No best program available for parity plot.'])

        try:
            fields = dscv_viz._calculate_pde_fields(model, program)
        except Exception as exc:
            return VizResult(intent='parity', warnings=[f'Unable to compute PDE fields: {exc}'])

        try:
            actual = np.asarray(fields['ut_grid']).reshape(-1)
            predicted = np.asarray(fields['y_hat_grid']).reshape(-1)
            data = ParityPlotData.from_actual_predicted(actual, predicted, metadata={'lhs_label': 'u_t'})
        except Exception as exc:
            return VizResult(intent='parity', warnings=[f'Failed to build parity data: {exc}'])

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=ctx.options.get('figsize', (7, 7)))
        ax.scatter(data.predicted_values, data.actual_values, alpha=0.35, s=15, label='Prediction vs Actual')
        min_val = float(np.min([data.actual_values.min(), data.predicted_values.min()]))
        max_val = float(np.max([data.actual_values.max(), data.predicted_values.max()]))
        ref = np.linspace(min_val, max_val, 100)
        ax.plot(ref, ref, 'r--', linewidth=1.5, label='y = x')
        ax.set_xlabel('Predicted RHS Values')
        ax.set_ylabel('True LHS Values (u_t)')
        ax.set_title(ctx.options.get('title', 'Parity Plot'))
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', 'box')

        _, path = self._resolve_output(ctx, 'parity_plot.png')
        fig.savefig(str(path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        try:
            plt.close(fig)
        except Exception:
            pass

        return VizResult(intent='parity', paths=[path], metadata={'parity': data})

    def _spr_residual(self, model, ctx) -> VizResult:
        try:
            pinn_fields = dscv_viz._calculate_pinn_fields(model, self._get_best_program(model))
        except Exception as exc:
            return VizResult(intent='spr_residual', warnings=[f'Unable to compute PINN fields: {exc}'])

        try:
            actual = np.asarray(pinn_fields['y_true']).reshape(-1)
            predicted = np.asarray(pinn_fields['y_pred']).reshape(-1)
            coords = np.asarray(pinn_fields['coords'])
            data = ResidualPlotData.from_actual_predicted(actual, predicted, input_coordinates=coords)
        except Exception as exc:
            return VizResult(intent='spr_residual', warnings=[f'Failed to build SPR residual data: {exc}'])

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=ctx.options.get('figsize', (12, 5)))
        ax1 = fig.add_subplot(1, 2, 1)
        sc = ax1.scatter(data.actual, data.predicted, alpha=0.4, s=15)
        min_val = float(np.min(np.concatenate([data.actual, data.predicted])))
        max_val = float(np.max(np.concatenate([data.actual, data.predicted])))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('SPR Parity Plot')
        fig.colorbar(sc, ax=ax1, label='sample')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist(data.residuals, bins=int(ctx.options.get('bins', 40)), color='#1f77b4', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Frequency')
        ax2.set_title('SPR Residual Distribution')
        ax2.axvline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.6)

        fig.tight_layout()
        _, path = self._resolve_output(ctx, 'spr_residual_analysis.png')
        fig.savefig(str(path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        try:
            plt.close(fig)
        except Exception:
            pass

        return VizResult(intent='spr_residual', paths=[path], metadata={'residual': data})

    def _spr_field_comparison(self, model, ctx) -> VizResult:
        try:
            pinn_fields = dscv_viz._calculate_pinn_fields(model, self._get_best_program(model))
        except Exception as exc:
            return VizResult(intent='spr_field_comparison', warnings=[f'Unable to compute PINN fields: {exc}'])

        try:
            x = np.asarray(pinn_fields['coords_x']).reshape(-1)
            t = np.asarray(pinn_fields['coords_t']).reshape(-1)
            true_vals = np.asarray(pinn_fields['y_true']).reshape(-1)
            pred_vals = np.asarray(pinn_fields['y_pred']).reshape(-1)
        except Exception as exc:
            return VizResult(intent='spr_field_comparison', warnings=[f'Failed to read PINN field data: {exc}'])

        try:
            from scipy.interpolate import griddata
        except Exception as exc:  # pragma: no cover - optional dependency
            return VizResult(intent='spr_field_comparison', warnings=[f'scipy is required for SPR field comparison: {exc}'])

        grid_x, grid_t = np.mgrid[x.min():x.max():100j, t.min():t.max():100j]
        grid_true = griddata((x, t), true_vals, (grid_x, grid_t), method='cubic')
        grid_pred = griddata((x, t), pred_vals, (grid_x, grid_t), method='cubic')

        if grid_true is None or grid_pred is None:
            return VizResult(intent='spr_field_comparison', warnings=['Failed to interpolate SPR field data.'])

        data = FieldComparisonData(
            x_coords=np.linspace(x.min(), x.max(), grid_true.shape[0]),
            t_coords=np.linspace(t.min(), t.max(), grid_true.shape[1]),
            true_field=np.nan_to_num(grid_true),
            predicted_field=np.nan_to_num(grid_pred),
        )

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=ctx.options.get('figsize', (14, 6)), sharey=True)
        fig.suptitle(ctx.options.get('title', 'SPR Predicted vs True Field'), fontsize=16)

        vmin = float(np.nanmin([data.true_field, data.predicted_field]))
        vmax = float(np.nanmax([data.true_field, data.predicted_field]))

        im0 = axes[0].pcolormesh(data.t_coords, data.x_coords, data.true_field, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        fig.colorbar(im0, ax=axes[0], label='Value')
        axes[0].set_title('True Field (SPR)')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Space')

        im1 = axes[1].pcolormesh(data.t_coords, data.x_coords, data.predicted_field, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        fig.colorbar(im1, ax=axes[1], label='Value')
        axes[1].set_title('Predicted Field (SPR)')
        axes[1].set_xlabel('Time')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        _, path = self._resolve_output(ctx, 'spr_field_comparison.png')
        fig.savefig(str(path), dpi=ctx.options.get('dpi', 300), bbox_inches='tight')
        try:
            plt.close(fig)
        except Exception:
            pass

        return VizResult(intent='spr_field_comparison', paths=[path], metadata={'field_comparison': data})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_output(self, ctx, filename: str):
        from pathlib import Path

        base = ctx.options.get('output_dir')
        if base is not None:
            base_path = Path(base) / 'dscv'
            base_path.mkdir(parents=True, exist_ok=True)
            return base_path, base_path / filename
        path = ctx.save_path(f'dscv/{filename}')
        return path.parent, path

    def _build_reward_evolution(self, model) -> Optional[RewardEvolutionData]:
        searcher = getattr(model, 'searcher', None)
        if searcher is None or not getattr(searcher, 'r_train', None):
            return None

        rewards = searcher.r_train
        if not rewards:
            return None

        steps = np.arange(1, len(rewards) + 1)
        max_vals = []
        mean_vals = []
        best_vals = []
        cumulative_best = -np.inf
        for batch in rewards:
            arr = np.asarray(batch)
            max_val = float(np.max(arr))
            mean_val = float(np.mean(arr))
            cumulative_best = max(cumulative_best, max_val)
            max_vals.append(max_val)
            mean_vals.append(mean_val)
            best_vals.append(cumulative_best)

        return RewardEvolutionData(
            steps=steps,
            best_reward=np.asarray(best_vals),
            max_reward=np.asarray(max_vals),
            mean_reward=np.asarray(mean_vals),
        )

    def _get_best_program(self, model):
        best = getattr(model, 'best_p', None)
        if isinstance(best, list):
            best = best[0] if best else None
        if best is None:
            searcher = getattr(model, 'searcher', None)
            if searcher is not None:
                best = getattr(searcher, 'best_p', None)
                if isinstance(best, list):
                    best = best[0] if best else None
        return best
