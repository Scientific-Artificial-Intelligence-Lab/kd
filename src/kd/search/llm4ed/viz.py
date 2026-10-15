
from __future__ import annotations

from typing import Any

from kd.search.llm4ed.score import (
    ERROR_ABNORMAL_COEF,
    ERROR_INEXPRESSIBLE,
    ERROR_LSTSQ,
    ERROR_NON_FINITE,
    ERROR_PARSE,
    ERROR_UNDEFINED_OPERANDS,
    ERROR_UNDEFINED_OPERATORS,
)
from kd.viz.gap_notes import GapVocabulary
from kd.viz.plugin_panels import (
    BandPanel,
    PanelSet,
    SeriesPanel,
    series_gap_annotator,
)









POOL_BEST_KEY = "pool_best"
POOL_MEDIAN_KEY = "pool_median"
POOL_WORST_KEY = "pool_worst"
N_INVALID_KEY = "n_invalid"
N_LLM_CALLS_KEY = "n_llm_calls"
N_VALID_KEY = "n_valid"









N_INVALID_UNDEFINED_OPERANDS_KEY = "n_invalid_undefined_operands"
N_INVALID_UNDEFINED_OPERATORS_KEY = "n_invalid_undefined_operators"
N_INVALID_INEXPRESSIBLE_KEY = "n_invalid_inexpressible"
N_INVALID_PARSE_KEY = "n_invalid_parse"
N_INVALID_NON_FINITE_KEY = "n_invalid_non_finite"
N_INVALID_LSTSQ_KEY = "n_invalid_lstsq"
N_INVALID_ABNORMAL_COEF_KEY = "n_invalid_abnormal_coef"
N_INVALID_OTHER_KEY = "n_invalid_other"




INVALID_KEY_BY_LABEL: dict[str, str] = {
    ERROR_UNDEFINED_OPERANDS: N_INVALID_UNDEFINED_OPERANDS_KEY,
    ERROR_UNDEFINED_OPERATORS: N_INVALID_UNDEFINED_OPERATORS_KEY,
    ERROR_INEXPRESSIBLE: N_INVALID_INEXPRESSIBLE_KEY,
    ERROR_PARSE: N_INVALID_PARSE_KEY,
    ERROR_NON_FINITE: N_INVALID_NON_FINITE_KEY,
    ERROR_LSTSQ: N_INVALID_LSTSQ_KEY,
    ERROR_ABNORMAL_COEF: N_INVALID_ABNORMAL_COEF_KEY,
}


INVALID_SPREAD_METRICS: tuple[str, ...] = (
    *INVALID_KEY_BY_LABEL.values(),
    N_INVALID_OTHER_KEY,
)

LOGGED_METRICS: tuple[str, ...] = (
    POOL_BEST_KEY,
    POOL_MEDIAN_KEY,
    POOL_WORST_KEY,
    N_INVALID_KEY,
    N_LLM_CALLS_KEY,
    N_VALID_KEY,
    *INVALID_SPREAD_METRICS,
)

_X_LABEL = "Round"






_POOL_GAPS = GapVocabulary(
    unit_plural="rounds", missing="had no pool", nothing="no pool"
)


def _counts(n_invalid: list[Any]) -> list[Any]:
    return [float(value) for value in n_invalid]


def _cumulative(n_llm_calls: list[Any]) -> list[Any]:
    cumulative: list[Any] = []
    running = 0.0
    for value in n_llm_calls:
        running += float(value)
        cumulative.append(running)
    return cumulative







_PANELS = PanelSet(
    (
        BandPanel(
            name="pool_reward_spread",
            title="Pool Reward Spread",
            description=(
                "Elite-pool reward band (best / median / worst) per round -- pool "
                "quality and diversity as the LLM proposer evolves."
            ),
            metrics=(POOL_BEST_KEY, POOL_MEDIAN_KEY, POOL_WORST_KEY),
            x_label=_X_LABEL,
            ylabel="reward",
            gaps=_POOL_GAPS,
        ),
        SeriesPanel(
            name="invalid_count",
            title="Invalid Count",
            description=(
                "Number of invalid scoring events per round (n_invalid): parse / "
                "solve failures across all resamples. Rising counts flag the LLM "
                "proposing more unusable equations."
            ),
            metric=N_INVALID_KEY,
            x_label=_X_LABEL,
            ylabel="invalid events",
            count_axis=True,
            derive=_counts,
            annotate=series_gap_annotator(
                GapVocabulary(
                    unit_plural="rounds",
                    missing="recorded no invalid count",
                    nothing="no invalid count",
                )
            ),
        ),
        SeriesPanel(
            name="llm_calls",
            title="Cumulative LLM Calls",
            description=(
                "Total provider completions issued through each round (running sum "
                "of per-round n_llm_calls) -- the search's whole-run budget spend."
            ),
            metric=N_LLM_CALLS_KEY,
            x_label=_X_LABEL,
            ylabel="cumulative calls",
            count_axis=True,
            derive=_cumulative,
            annotate=series_gap_annotator(
                GapVocabulary(
                    unit_plural="rounds",
                    missing="recorded no call count",
                    nothing="no call count",
                )
            ),
        ),
    )
)

list_plot_infos = _PANELS.list_plot_infos
render = _PANELS.render
get_data = _PANELS.get_data


__all__ = [
    "INVALID_KEY_BY_LABEL",
    "INVALID_SPREAD_METRICS",
    "LOGGED_METRICS",
    "N_INVALID_ABNORMAL_COEF_KEY",
    "N_INVALID_INEXPRESSIBLE_KEY",
    "N_INVALID_KEY",
    "N_INVALID_LSTSQ_KEY",
    "N_INVALID_NON_FINITE_KEY",
    "N_INVALID_OTHER_KEY",
    "N_INVALID_PARSE_KEY",
    "N_INVALID_UNDEFINED_OPERANDS_KEY",
    "N_INVALID_UNDEFINED_OPERATORS_KEY",
    "N_LLM_CALLS_KEY",
    "N_VALID_KEY",
    "POOL_BEST_KEY",
    "POOL_MEDIAN_KEY",
    "POOL_WORST_KEY",
    "get_data",
    "list_plot_infos",
    "render",
]
