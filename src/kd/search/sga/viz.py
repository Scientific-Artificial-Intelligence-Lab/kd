
from __future__ import annotations

from kd.search.surrogate_log import (
    SURROGATE_BEST_EPOCH_KEY,
    SURROGATE_EPOCH_KEY,
    SURROGATE_TRAIN_LOSS_KEY,
    SURROGATE_VAL_LOSS_KEY,
)
from kd.viz.plugin_panels import PanelSet, SeriesPanel, SurrogateCurvePanel
















GEN_BEST_AIC_KEY = "gen_best_aic"
GEN_MEAN_AIC_KEY = "gen_mean_aic"
GEN_BEST_NMSE_KEY = "gen_best_nmse"
N_VALID_KEY = "n_valid"
N_UNIQUE_KEY = "n_unique"
GEN_MEAN_COMPLEXITY_KEY = "gen_mean_complexity"




POP_MEAN_AIC_KEY = "pop_mean_aic"






REPEAT_CROSS_KEY = "repeat_cross"
REPEAT_CHANGE_KEY = "repeat_change"
LOGGED_METRICS: tuple[str, ...] = (
    GEN_BEST_AIC_KEY,
    GEN_MEAN_AIC_KEY,
    GEN_BEST_NMSE_KEY,
    N_VALID_KEY,
    N_UNIQUE_KEY,
    GEN_MEAN_COMPLEXITY_KEY,
    POP_MEAN_AIC_KEY,
    REPEAT_CROSS_KEY,
    REPEAT_CHANGE_KEY,
)





BEST_AIC_KEY = "best_aic"


_X_LABEL = "Generation"








_PANELS = PanelSet(
    (
        SeriesPanel(
            name="population_diversity",
            title="Offspring Diversity",
            description=(
                "Distinct candidate-expression count among this generation's "
                "evaluated offspring (the staged batch, post-dedup; failed "
                "evaluations count too). A collapsing count signals premature "
                "convergence / loss of search diversity. NOT a population "
                "statistic — the count can exceed the population size."
            ),
            metric=N_UNIQUE_KEY,
            x_label=_X_LABEL,
            count_axis=True,
        ),
        SeriesPanel(
            name="complexity_evolution",
            title="Complexity Evolution",
            description=(
                "Mean number of STRidge-selected (non-zero) terms across this "
                "generation's valid offspring — a sparsity / bloat monitor for "
                "the evolving PDE candidates. All-invalid generations appear as "
                "gaps (nothing was measured)."
            ),
            metric=GEN_MEAN_COMPLEXITY_KEY,
            x_label=_X_LABEL,
        ),
        SeriesPanel(
            name="fitness_spread",
            title="Fitness Spread",
            description=(
                "Per-generation MEAN AIC over the surviving population's finite "
                "scores (lower is better). The platform 'convergence' plot draws "
                "the single best AIC; this curve shows the level of the whole "
                "surviving population — under elitist truncation it declines "
                "alongside the best curve, and the distance between the two is "
                "how far the population trails its optimum. Generations with no "
                "finite-scored survivors (e.g. an all-invalid start) appear as "
                "gaps (their +inf sentinel is masked)."
            ),
            metric=POP_MEAN_AIC_KEY,
            x_label=_X_LABEL,
        ),
        SurrogateCurvePanel(
            name="surrogate_training",
            title="Surrogate Training Curve",
            description=(
                "FieldModel surrogate training loss "
                "vs epoch (log-y): train and — when a validation split exists — "
                "validation MSE, with the best-validation epoch marked. The autograd "
                "derivatives (u_x, u_t) the GA reads from come from this network. "
                "Absent (No data panel) for finite-diff mode or when the surrogate "
                "was pre-trained (user-supplied field_model), not trained by the "
                "platform builder."
            ),
            epoch_key=SURROGATE_EPOCH_KEY,
            train_key=SURROGATE_TRAIN_LOSS_KEY,
            val_key=SURROGATE_VAL_LOSS_KEY,
            best_epoch_key=SURROGATE_BEST_EPOCH_KEY,
        ),
    )
)

list_plot_infos = _PANELS.list_plot_infos
render = _PANELS.render
get_data = _PANELS.get_data

__all__ = [
    "BEST_AIC_KEY",
    "GEN_BEST_AIC_KEY",
    "GEN_BEST_NMSE_KEY",
    "GEN_MEAN_AIC_KEY",
    "GEN_MEAN_COMPLEXITY_KEY",
    "LOGGED_METRICS",
    "N_UNIQUE_KEY",
    "N_VALID_KEY",
    "POP_MEAN_AIC_KEY",
    "REPEAT_CHANGE_KEY",
    "REPEAT_CROSS_KEY",
    "get_data",
    "list_plot_infos",
    "render",
]
