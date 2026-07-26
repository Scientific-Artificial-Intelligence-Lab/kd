
from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from kd.api import Model
from kd.harness.plan import PlanEntry

if TYPE_CHECKING:
    from kd.data.schema import PDEDataset
    from kd.search.records import RunRecord

logger = logging.getLogger(__name__)



STATUS_COMPLETED: Final[str] = "completed"
STATUS_RAISED: Final[str] = "raised"
STATUS_NO_RECORD: Final[str] = "no_record"

_ERROR_MESSAGE_MAX_CHARS: Final[int] = 2000


@dataclass(frozen=True, kw_only=True)
class EpisodeOutcome:

    entry_index: int
    entry: PlanEntry
    status: str
    record: RunRecord | None
    error_type: str | None
    error_message: str | None
    wallclock_seconds: float


def run_episode(
    *,
    entry: PlanEntry,
    entry_index: int,
    dataset: PDEDataset,
    model_factory: Callable[..., Any] = Model,
    device: str | None = None,
) -> EpisodeOutcome:
    start = time.perf_counter()
    try:


        model_kwargs = copy.deepcopy(entry.model_kwargs)



        sibling_kwargs: dict[str, Any] = {}
        if device is not None:
            sibling_kwargs["device"] = device
        model = model_factory(
            algorithm=entry.instrument,
            seed=entry.seed,
            verbose=False,
            **model_kwargs,
            **sibling_kwargs,
        )
        model.fit(dataset)
        record: RunRecord | None = model.result_.run_record
    except Exception as exc:
        elapsed = time.perf_counter() - start
        error_type = type(exc).__qualname__
        error_message = str(exc)[:_ERROR_MESSAGE_MAX_CHARS]
        logger.info(
            "episode %d (%s/%s seed=%d) raised %s after %.3fs",
            entry_index,
            entry.instrument,
            entry.dataset_ref,
            entry.seed,
            error_type,
            elapsed,
        )
        return EpisodeOutcome(
            entry_index=entry_index,
            entry=entry,
            status=STATUS_RAISED,
            record=None,
            error_type=error_type,
            error_message=error_message,
            wallclock_seconds=elapsed,
        )

    elapsed = time.perf_counter() - start
    if record is None:
        logger.info(
            "episode %d (%s/%s seed=%d) produced no run_record after %.3fs",
            entry_index,
            entry.instrument,
            entry.dataset_ref,
            entry.seed,
            elapsed,
        )
        return EpisodeOutcome(
            entry_index=entry_index,
            entry=entry,
            status=STATUS_NO_RECORD,
            record=None,
            error_type=None,
            error_message=None,
            wallclock_seconds=elapsed,
        )

    logger.info(
        "episode %d (%s/%s seed=%d) completed in %.3fs",
        entry_index,
        entry.instrument,
        entry.dataset_ref,
        entry.seed,
        elapsed,
    )
    return EpisodeOutcome(
        entry_index=entry_index,
        entry=entry,
        status=STATUS_COMPLETED,
        record=record,
        error_type=None,
        error_message=None,
        wallclock_seconds=elapsed,
    )


__all__ = [
    "STATUS_COMPLETED",
    "STATUS_NO_RECORD",
    "STATUS_RAISED",
    "EpisodeOutcome",
    "run_episode",
]
