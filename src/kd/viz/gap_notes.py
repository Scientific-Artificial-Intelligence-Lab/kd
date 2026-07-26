
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes





NO_MEASUREMENT: Final[float] = float("nan")


@dataclass(frozen=True)
class GapVocabulary:

    unit_plural: str
    missing: str
    nothing: str


def is_measured(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def measured_flags(series: Sequence[Any]) -> list[bool]:
    return [is_measured(value) for value in series]


def band_measured_flags(
    series_group: Iterable[Sequence[Any]], length: int
) -> list[bool]:
    flags = [False] * length
    for series in series_group:
        for index, value in enumerate(series[:length]):
            if is_measured(value):
                flags[index] = True
    return flags


def gap_phrase(n_gaps: int, n_samples: int, vocab: GapVocabulary) -> str:
    return f"{n_gaps} of {n_samples} {vocab.unit_plural} {vocab.missing}"


def all_gap_note(n_samples: int, vocab: GapVocabulary) -> str:
    return (
        f"{vocab.nothing} in any of the {n_samples} {vocab.unit_plural} "
        f"(nothing measured)"
    )


def partial_gap_note(n_gaps: int, n_samples: int, vocab: GapVocabulary) -> str:
    return f"{gap_phrase(n_gaps, n_samples, vocab)} (no measurement to plot there)"


def append_subtitle(ax: Axes, note: str) -> None:
    ax.set_title(f"{ax.get_title()}\n{note}", fontsize="medium")


def annotate_gaps(ax: Axes, measured: Sequence[bool], vocab: GapVocabulary) -> None:
    n_samples = len(measured)
    n_gaps = n_samples - sum(measured)
    if n_gaps == 0:
        return
    note = (
        all_gap_note(n_samples, vocab)
        if n_gaps == n_samples
        else partial_gap_note(n_gaps, n_samples, vocab)
    )
    append_subtitle(ax, note)


__all__ = [
    "GapVocabulary",
    "all_gap_note",
    "annotate_gaps",
    "append_subtitle",
    "band_measured_flags",
    "gap_phrase",
    "is_measured",
    "measured_flags",
    "partial_gap_note",
]
