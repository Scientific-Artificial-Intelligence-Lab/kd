
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from kdagent.data_source import is_file_ref, resolve_input
from kdagent.lineage import SegmentLedger
from kdagent.surrogates import SurrogateRegistry


class Session:

    def __init__(
        self,
        workspace: Path,
        dataset_id: str | None = None,
        time_budget_seconds: float | None = None,
        recursion_limit: int | None = None,
    ) -> None:
        self.workspace = workspace
        self.pinned_dataset_id = dataset_id
        self.input_source = (
            resolve_input(workspace, dataset_id)
            if dataset_id is not None and is_file_ref(dataset_id)
            else None
        )
        self.time_budget_seconds = time_budget_seconds
        self.recursion_limit = recursion_limit
        self._started = time.monotonic()
        self.prior_annotation: dict[str, dict[str, Any]] | None = None
        self.successful_runs: dict[str, dict[str, Any]] = {}
        self.ledger = SegmentLedger(workspace)
        selected = self.ledger.selected
        self.answer_dataset_id: str | None = (
            None if selected is None else self.ledger.entry(selected).dataset
        )
        self.surrogates = SurrogateRegistry(workspace)

    def budget(self) -> dict[str, Any]:
        elapsed = round(time.monotonic() - self._started, 1)
        return {
            "elapsed_seconds": elapsed,
            "time_budget_seconds": self.time_budget_seconds,
            "remaining_seconds": (
                None
                if self.time_budget_seconds is None
                else round(self.time_budget_seconds - elapsed, 1)
            ),
            "graph_step_limit": self.recursion_limit,
        }
