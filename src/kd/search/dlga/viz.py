
from __future__ import annotations

from kd.search.surrogate_log import (
    SURROGATE_BEST_EPOCH_KEY,
    SURROGATE_EPOCH_KEY,
    SURROGATE_TRAIN_LOSS_KEY,
    SURROGATE_VAL_LOSS_KEY,
)
from kd.viz.plugin_panels import PanelSet, SeriesPanel, SurrogateCurvePanel










GEN_BEST_FITNESS_KEY = "gen_best_fitness"
GEN_MEAN_FITNESS_KEY = "gen_mean_fitness"
GEN_BEST_NMSE_KEY = "gen_best_nmse"
N_VALID_KEY = "n_valid"
N_UNIQUE_KEY = "n_unique"
GEN_MEAN_COMPLEXITY_KEY = "gen_mean_complexity"
LHS_UT_KEY = "lhs_ut"
LHS_UTT_KEY = "lhs_utt"
LOGGED_METRICS: tuple[str, ...] = (
    GEN_BEST_FITNESS_KEY,
    GEN_MEAN_FITNESS_KEY,
    GEN_BEST_NMSE_KEY,
    N_VALID_KEY,
    N_UNIQUE_KEY,
    GEN_MEAN_COMPLEXITY_KEY,
    LHS_UT_KEY,
    LHS_UTT_KEY,
)


_X_LABEL = "Generation"












_PANELS = PanelSet(
    (
        SeriesPanel(
            name="fitness_spread",
            title="Fitness Spread",
            description=(
                "Per-generation MEAN GA fitness (NMSE + epsilon*length; lower is "
                "better) over valid individuals, on a LOG y-axis: single "
                "blown-up individuals (svd_null_space normalization) push the "
                "mean into the thousands and a linear axis flattens the range "
                "the search actually works in. Complements the platform "
                "'convergence' plot, which draws the single global best (the GA "
                "keeps its best genome, so the per-generation best is that same "
                "curve). A generation appears as a gap whenever its mean is "
                "non-finite — "
                "one inf-fitness individual suffices, not only all-invalid "
                "generations."
            ),
            metric=GEN_MEAN_FITNESS_KEY,
            x_label=_X_LABEL,
            log_y=True,
        ),
        SeriesPanel(
            name="population_diversity",
            title="Population Diversity",
            description=(
                "Distinct candidate-expression count per generation over the whole "
                "population. A collapsing count signals premature convergence."
            ),
            metric=N_UNIQUE_KEY,
            x_label=_X_LABEL,
            count_axis=True,
        ),
        SeriesPanel(
            name="complexity_evolution",
            title="Complexity Evolution",
            description=(
                "Mean retained-term count (EvaluationResult.complexity — terms "
                "kept in the final fitted equation) of valid individuals per "
                "generation. NOT the genome gene count that the epsilon fitness "
                "penalty acts on: module-internal bloat (more genes per term) "
                "moves the penalty but not this curve. All-invalid generations "
                "appear as gaps (nothing was measured)."
            ),
            metric=GEN_MEAN_COMPLEXITY_KEY,
            x_label=_X_LABEL,
        ),
        SurrogateCurvePanel(
            name="surrogate_training",
            title="Surrogate Training Curve",
            description=(
                "NN_1 surrogate (Xu 2020 §2.2.1) training loss vs epoch (log-y): "
                "train and — when a validation split exists — validation MSE, with "
                "the best-validation epoch marked. This is the 'deep learning' "
                "evidence of the DLGA pipeline — the autograd-derivative source the "
                "GA reads from. Absent (No data panel) when the surrogate was "
                "pre-trained or not trained by the platform builder."
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
    "GEN_BEST_FITNESS_KEY",
    "GEN_BEST_NMSE_KEY",
    "GEN_MEAN_COMPLEXITY_KEY",
    "GEN_MEAN_FITNESS_KEY",
    "LHS_UTT_KEY",
    "LHS_UT_KEY",
    "LOGGED_METRICS",
    "N_UNIQUE_KEY",
    "N_VALID_KEY",
    "get_data",
    "list_plot_infos",
    "render",
]
