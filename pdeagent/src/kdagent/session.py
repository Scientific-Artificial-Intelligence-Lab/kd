
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from kd.models import DLGA_NN_1

from kdagent.config import Llm4edConfig
from kdagent.data_source import is_file_ref, resolve_input
from kdagent.lineage import SegmentLedger
from kdagent.surrogates import SurrogateRegistry

DEVICES = ("cpu", "cuda")


def check_training_dtype(value: object) -> None:
    replace(DLGA_NN_1, training_dtype=cast(Any, value))


def effective_training_dtype(run_choice: str | None, value: object) -> str | None:
    check_training_dtype(value)
    given = cast("str | None", value)
    if run_choice is None:
        return given
    if given is not None and given != run_choice:
        raise ValueError(
            f"training_dtype {given!r} differs from this run's training "
            f"precision {run_choice!r} (kd-agent run --training-dtype); every "
            "network this run trains uses the run's precision, so omit "
            "training_dtype to inherit it"
        )
    return run_choice


class Session:

    def __init__(
        self,
        workspace: Path,
        dataset_id: str | None = None,
        time_budget_seconds: float | None = None,
        recursion_limit: int | None = None,
        *,
        device: str = "cpu",
        training_dtype: str | None = None,
        llm4ed: Llm4edConfig | None = None,
    ) -> None:
        if device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
        check_training_dtype(training_dtype)
        self.workspace = workspace
        self.device = device
        self.training_dtype = training_dtype
        self.llm4ed = llm4ed
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
        self.action_experience: dict[str, Any] | None = None
        self.action_memory_sha256: str | None = None
        self.action_features: dict[str, Any] = {}
        self.action_presented: set[str] = set()
        self.action_pipelines: dict[str, dict[str, Any]] = {}
        self.successful_runs: dict[str, dict[str, Any]] = {}
        self.sealed_lhs: dict[str, dict[str, Any] | None] = {}
        self.sealed_laws: dict[str, dict[str, Any]] = {}
        self.sealed_fits: dict[str, dict[str, Any]] = {}
        self.derived_inputs: dict[str, str] = {}
        self.search_inputs: dict[str, dict[str, Any]] = {}
        self.search_verify: dict[str, dict[str, float] | None] = {}
        self.receipts: dict[str, dict[str, Any]] = {}
        self.evidence: dict[str, dict[str, Any]] = {}
        self.sketches: dict[str, dict[str, Any]] = {}
        self.user_sketch_id: str | None = None
        self.active_sketch_id: str | None = None
        self.workers_started = 0
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
