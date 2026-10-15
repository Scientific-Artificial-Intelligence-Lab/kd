
from __future__ import annotations

from kd.viz.plugin_panels import PanelSet, SeriesPanel

_X_LABEL = "Iteration"





_PANELS = PanelSet(
    (
        SeriesPanel(
            name="reward_convergence",
            title="Reward (Mean of Top ε)",
            description=(
                "Per-iteration mean reward of the top-ε risk-seeking subset the "
                "policy gradient trains on — kd's analogue of what the reference "
                "implementation calls 'Mean of Top ε' (r_avg_sub; kd selects the "
                "subset after invalid-row filtering, delta, so the subsets "
                "can differ for larger batches), NOT the batch reward. The subset "
                "is every valid row whose reward is at or above the (1-ε) quantile "
                "of the valid rewards (numpy 'higher' method): 1 + floor(ε·(n-1)) "
                "rows for n distinct valid rewards, more when rewards tie at the "
                "quantile. When that is a "
                "single row this curve equals the per-iteration maximum and, under "
                "the 'R_e' baseline, coincides with the Reward Baseline plot; in "
                "general it lies between the (1-ε) quantile and that maximum. The "
                "whole-batch signal is the 'Reward (Full-Batch Mean)' plot."
            ),
            metric="reward",
            x_label=_X_LABEL,
        ),
        SeriesPanel(
            name="reward_full_mean",
            title="Reward (Full-Batch Mean)",
            description=(
                "Per-iteration mean reward over the FULL sampled batch (invalid "
                "expressions carry the invalid-reward fill) — the reference "
                "implementation's r_avg_full. Unlike the top-ε curve this one "
                "moves with the whole policy, so the two panels together show "
                "elite vs population progress."
            ),
            metric="reward_full",
            x_label=_X_LABEL,
        ),
        SeriesPanel(
            name="entropy_loss_decay",
            title="Entropy Loss",
            description=(
                "Entropy regularizer loss component (= -weight x mean policy "
                "entropy, typically negative). Direction-neutral title on "
                "purpose: the curve RISES toward zero as the policy's entropy "
                "collapses — it does not 'decay' for a healthy run."
            ),
            metric="entropy_loss",
            x_label=_X_LABEL,
        ),
        SeriesPanel(
            name="baseline_ewma",
            title="Reward Baseline",
            description=(
                "Scalar reward baseline used by the RSPG policy-gradient "
                "estimator. NOTE: the trailing ``_ewma`` in the plot name is a "
                "legacy artifact retained for backward compatibility with an "
                "earlier plot-naming scheme — the actual series semantics depend on "
                "``RSPGStrategy.baseline``: ``'R_e'`` (default) is a "
                "risk-seeking quantile, ``'ewma_R'`` is an exponential moving "
                "average, ``'combined'`` sums both. Read the configured "
                "baseline mode before interpreting the curve; the human-readable "
                "plot title (``Reward Baseline``) is mode-neutral on purpose."
            ),
            metric="baseline",
            x_label=_X_LABEL,
        ),
    )
)

list_plot_infos = _PANELS.list_plot_infos
render = _PANELS.render
get_data = _PANELS.get_data

__all__ = [
    "get_data",
    "list_plot_infos",
    "render",
]
