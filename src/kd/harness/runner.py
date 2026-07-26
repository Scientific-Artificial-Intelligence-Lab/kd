
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import kd
from kd.api import Model
from kd.harness.episode import EpisodeOutcome, run_episode
from kd.harness.plan import ExperimentPlan
from kd.harness.store import EvidenceStore, environment_fingerprint

if TYPE_CHECKING:
    from kd.data.schema import PDEDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class PlanRunResult:

    store_root: Path
    outcomes: tuple[EpisodeOutcome, ...]


def _supported_instruments() -> set[str]:
    return {schema["algorithm"] for schema in kd.instrument_schemas()}


def _preflight(plan: ExperimentPlan, datasets: Mapping[str, PDEDataset]) -> None:
    supported = _supported_instruments()
    violations: list[str] = []
    for index, entry in enumerate(plan.entries):
        if entry.instrument not in supported:
            violations.append(
                f"entry[{index}]: unknown instrument {entry.instrument!r} "
                f"(supported: {sorted(supported)!r})"
            )
        if entry.dataset_ref not in datasets:
            violations.append(
                f"entry[{index}]: dataset_ref {entry.dataset_ref!r} not in "
                f"provided datasets (available: {sorted(datasets)!r})"
            )
    if violations:
        joined = "\n ".join(violations)
        raise ValueError(
            f"plan pre-flight failed ({len(violations)} violation(s)):\n {joined}"
        )


def run_plan(
    plan: ExperimentPlan,
    *,
    datasets: Mapping[str, PDEDataset],
    store_root: Path,
    model_factory: Callable[..., Any] = Model,
    device: str | None = None,
) -> PlanRunResult:
    _preflight(plan, datasets)

    store = EvidenceStore.create(
        Path(store_root),
        plan=plan,
        env=environment_fingerprint(),
    )
    logger.info(
        "run_plan: executing %d entries of plan %r into %s",
        len(plan.entries),
        plan.name,
        store.root,
    )

    outcomes: list[EpisodeOutcome] = []
    for entry_index, entry in enumerate(plan.entries):



        outcome = run_episode(
            entry=entry,
            entry_index=entry_index,
            dataset=datasets[entry.dataset_ref],
            model_factory=model_factory,
            device=device,
        )
        store.add_outcome(outcome)
        outcomes.append(outcome)

    return PlanRunResult(store_root=store.root, outcomes=tuple(outcomes))


__all__ = [
    "PlanRunResult",
    "run_plan",
]
