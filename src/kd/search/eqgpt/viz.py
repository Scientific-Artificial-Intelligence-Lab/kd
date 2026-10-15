
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kd.viz.extension import PlotInfo
from kd.viz.gap_notes import (
    NO_MEASUREMENT,
    GapVocabulary,
    all_gap_note,
    append_subtitle,
    gap_phrase,
    is_measured,
    partial_gap_note,
)
from kd.viz.plugin_panels import (
    BandPanel,
    PanelSet,
    SeriesPanel,
    draw_no_data,
    sanitize_y,
    series_gap_annotator,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes







POOL_BEST_KEY = "pool_best"
POOL_MEDIAN_KEY = "pool_median"
POOL_WORST_KEY = "pool_worst"
FINETUNE_LOSS_KEY = "finetune_loss"










N_DROPPED_KEY = "n_dropped"
LOGGED_METRICS: tuple[str, ...] = (
    POOL_BEST_KEY,
    POOL_MEDIAN_KEY,
    POOL_WORST_KEY,
    FINETUNE_LOSS_KEY,
    N_DROPPED_KEY,
)






_POOL_GAPS = GapVocabulary(
    unit_plural="epochs", missing="had no pool", nothing="no pool"
)
_FINETUNE_GAPS = GapVocabulary(
    unit_plural="epochs",
    missing="skipped the fine-tune",
    nothing="no fine-tune",
)






_PER_CASE_PLOT_NAME = "per_case_reward"
_PER_CASE_TITLE = "Per-case Reward"
PER_CASE_PLOT_INFO = PlotInfo(
    name=_PER_CASE_PLOT_NAME,
    title=_PER_CASE_TITLE,
    description=(
        "Reward of the discovered structure fitted to each wave-breaking case "
        "on its own -- one bar per experiment. A uniformly high band shows a "
        "single discovered law explaining every case."
    ),
)
_PER_CASE_XLABEL = "case"
_PER_CASE_YLABEL = "reward"
_PER_CASE_LABEL_FONTSIZE = 6





_PER_CASE_SCOPE_CAPTION = (
    "score = mean reward over the scored cases; R^2/MSE = primary case only"
)
_PER_CASE_SCOPE_WARNING = (
    f"plugin plot '{_PER_CASE_PLOT_NAME}': headline score is the mean reward "
    "over the scored cases (n/a cases excluded); R^2/MSE metrics are "
    "primary-case only"
)


def _annotate_reward_convergence(ax: Axes, series: list[Any]) -> None:
    from kd.viz.plots.convergence import flat_value

    measured = [index for index, value in enumerate(series) if is_measured(value)]
    gaps = len(series) - len(measured)
    if not measured:
        note = all_gap_note(len(series), _POOL_GAPS)
    else:
        constant = flat_value(series)
        if constant is None:
            if gaps == 0:
                return
            note = partial_gap_note(gaps, len(series), _POOL_GAPS)
        elif gaps:
            note = (
                f"best reward constant at {constant:.4g} across the "
                f"{len(measured)} measured epochs "
                f"({gap_phrase(gaps, len(series), _POOL_GAPS)})"
            )
        else:
            note = (
                f"best reward constant at {constant:.4g} (reached at the first epoch)"
            )
    append_subtitle(ax, note)


_X_LABEL = "Epoch"


_PANELS = PanelSet(
    (
        SeriesPanel(
            name="reward_convergence",
            title="Reward Convergence",
            description=(
                "Best top-k pool reward per epoch (the search's primary signal)."
            ),
            metric=POOL_BEST_KEY,
            x_label=_X_LABEL,
            annotate=_annotate_reward_convergence,
        ),
        BandPanel(
            name="pool_reward_spread",
            title="Pool Reward Spread",
            description=(
                "Top-k pool reward band (best / median / worst) per epoch -- pool "
                "quality and diversity as the GPT proposer evolves."
            ),
            metrics=(POOL_BEST_KEY, POOL_MEDIAN_KEY, POOL_WORST_KEY),
            x_label=_X_LABEL,
            ylabel="reward",
            gaps=_POOL_GAPS,
        ),
        SeriesPanel(
            name="finetune_loss",
            title="Fine-tune Loss",
            description=(
                "Mean cross-entropy loss of the per-epoch GPT fine-tune on the "
                "top-k pool."
            ),
            metric=FINETUNE_LOSS_KEY,
            x_label=_X_LABEL,
            annotate=series_gap_annotator(_FINETUNE_GAPS),
        ),
    )
)

list_plot_infos = _PANELS.list_plot_infos
render = _PANELS.render
get_data = _PANELS.get_data


def render_per_case_reward(ax: Axes, per_case: dict[str, float]) -> list[str]:
    ax.set_xlabel(_PER_CASE_XLABEL)
    ax.set_ylabel(_PER_CASE_YLABEL)
    ax.set_title(_PER_CASE_TITLE)
    names = list(per_case)
    if not names or not any(is_measured(per_case[name]) for name in names):
        reason = "no candidate" if not names else "no survivor cases"
        return draw_no_data(ax, PER_CASE_PLOT_INFO.name, reason)
    positions = list(range(len(names)))




    heights = [
        per_case[name] if is_measured(per_case[name]) else NO_MEASUREMENT
        for name in names
    ]
    ax.bar(positions, heights)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_short_case(name) for name in names],
        rotation=90,
        fontsize=_PER_CASE_LABEL_FONTSIZE,
    )
    for position, name in enumerate(names):
        if not is_measured(per_case[name]):








            ax.text(
                position,
                0.02,
                "n/a",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=_PER_CASE_LABEL_FONTSIZE,
            )
    append_subtitle(ax, _PER_CASE_SCOPE_CAPTION)
    return [_PER_CASE_SCOPE_WARNING]


def per_case_data(per_case: dict[str, float]) -> dict[str, Any]:
    names = list(per_case)
    return {
        "x": names,
        "y": [sanitize_y(per_case[name]) for name in names],
        "xlabel": _PER_CASE_XLABEL,
        "ylabel": _PER_CASE_YLABEL,
        "title": _PER_CASE_TITLE,
    }


def _short_case(name: str) -> str:
    return name[2:] if name.startswith("N_") else name


__all__ = [
    "FINETUNE_LOSS_KEY",
    "LOGGED_METRICS",
    "N_DROPPED_KEY",
    "PER_CASE_PLOT_INFO",
    "POOL_BEST_KEY",
    "POOL_MEDIAN_KEY",
    "POOL_WORST_KEY",
    "get_data",
    "list_plot_infos",
    "per_case_data",
    "render",
    "render_per_case_reward",
]
