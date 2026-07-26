
from __future__ import annotations

from pathlib import Path
from typing import Any

from kd.harness.dispatch import (
    DispatchDatasetSpec,
    build_dispatch_manifest,
)
from kd.harness.dispatch_log import DispatchLog, WorkerLogRow
from kd.harness.dispatch_report import build_batch_report, render_dispatch_markdown

from kd.harness.plan import ExperimentPlan, PlanEntry
from kd.harness.report import build_store_report

from ._helpers import build_sealed_store, make_record

_ENV6 = {
    "CUDA_VISIBLE_DEVICES": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
}


def _plan() -> ExperimentPlan:
    return ExperimentPlan(
        name="reportplan",
        entries=(
            PlanEntry(instrument="sga", dataset_ref="a", seed=0, model_kwargs={}),
            PlanEntry(instrument="sga", dataset_ref="b", seed=1, model_kwargs={}),
        ),
    )


def _manifest(plan: ExperimentPlan) -> Any:
    specs = {
        "a": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={}),
        "b": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={}),
    }
    return build_dispatch_manifest(plan, dataset_specs=specs, n_workers=2)


def _row(shard_id: str, **overrides: Any) -> WorkerLogRow:
    kwargs: dict[str, Any] = {
        "shard_id": shard_id,
        "pid": 100,
        "exit_code": 0,
        "kill_reason": None,
        "sigkill_used": False,
        "started_at": "2026-07-22T00:00:00+00:00",
        "finished_at": "2026-07-22T00:00:01+00:00",
        "wallclock_seconds": 1.0,
        "timeout_seconds": None,
        "env": dict(_ENV6),
        "heavy": False,
    }
    kwargs.update(overrides)
    return WorkerLogRow(**kwargs)


def _log(plan: ExperimentPlan, **overrides: Any) -> DispatchLog:
    kwargs: dict[str, Any] = {
        "plan_hash": plan.plan_hash(),
        "workers": (
            _row("shard-00"),
            _row(
                "shard-01",
                exit_code=-9,
                kill_reason="timeout",
                sigkill_used=True,
                timeout_seconds=1.0,
                heavy=True,
            ),
        ),
        "lost_entries": (41,),
        "heavy_peak": 1,
        "started_at": "2026-07-22T00:00:00+00:00",
        "finished_at": "2026-07-22T00:00:03+00:00",
    }
    kwargs.update(overrides)
    return DispatchLog(**kwargs)


def _store(plan: ExperimentPlan, tmp_path: Path) -> Any:
    return build_sealed_store(
        tmp_path / "store",
        entries=list(plan.entries),
        records={0: make_record("sga", seed=0), 1: make_record("sga", seed=1)},
        name=plan.name,
    )





def test_render_surfaces_worker_rows_kill_and_lost(tmp_path: Path) -> None:
    plan = _plan()
    report = render_dispatch_markdown(_manifest(plan), _log(plan))
    assert "shard-00" in report
    assert "shard-01" in report
    assert "timeout" in report
    assert "41" in report





def test_build_batch_report_is_store_report_plus_dispatch_section(
    tmp_path: Path,
) -> None:
    plan = _plan()
    store = _store(plan, tmp_path)
    manifest = _manifest(plan)
    log = _log(plan)

    composed = build_batch_report(store, manifest, log)
    expected = (
        build_store_report(store) + "\n" + render_dispatch_markdown(manifest, log)
    )
    assert composed == expected

    assert composed.startswith(build_store_report(store))





def test_render_is_byte_deterministic(tmp_path: Path) -> None:
    plan = _plan()
    manifest = _manifest(plan)
    log = _log(plan)
    assert render_dispatch_markdown(manifest, log) == render_dispatch_markdown(
        manifest, log
    )





def test_render_degenerate_all_clean_no_lost(tmp_path: Path) -> None:
    plan = _plan()
    manifest = _manifest(plan)
    log = _log(plan, workers=(_row("shard-00"), _row("shard-01")), lost_entries=())
    report = render_dispatch_markdown(manifest, log)
    assert "shard-00" in report
    assert "shard-01" in report
    assert report.strip() != ""
