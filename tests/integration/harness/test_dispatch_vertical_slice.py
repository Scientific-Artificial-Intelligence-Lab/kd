
from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import pytest
import torch
from kd.harness.dispatch import (
    DispatchDatasetSpec,
    build_dispatch_manifest,
    resolve_dataset,
    write_dispatch_manifest,
)
from kd.harness.dispatch_report import build_batch_report
from kd.harness.dispatcher import run_dispatch
from kd.harness.merge import merge_shards

from kd.harness.plan import ExperimentPlan, PlanEntry
from kd.harness.report import build_store_report
from kd.harness.runner import run_plan
from kd.harness.store import EvidenceStore

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_SGA = {
    "generations": 2,
    "population": 8,
    "depth": 3,
    "width": 4,
    "maxit": 3,
    "str_iters": 3,
    "d_tol": 0.5,
}
_INSTRUMENT_KWARGS: dict[str, dict[str, object]] = {
    "sga": _SGA,
    "pysindy": {},
    "discover": {"generations": 2},
}

_PYSINDY_AVAILABLE = importlib.util.find_spec("pysindy") is not None
_INSTRUMENTS = (
    ("sga", "pysindy", "discover") if _PYSINDY_AVAILABLE else ("sga", "discover")
)












_CUDA = torch.cuda.is_available()
_PIN_TABLE = ["0", "0"] if _CUDA else ["", ""]
_SLICE_DEVICE = "cuda" if _CUDA else None
_SEEDS = (0, 1)
_SPECS = {
    "grid_a": DispatchDatasetSpec(
        loader="kd.generate_burgers_data",
        kwargs={"nx": 32, "nt": 16, "nu": 0.1, "seed": 0},
    ),
    "grid_b": DispatchDatasetSpec(
        loader="kd.generate_burgers_data",
        kwargs={"nx": 24, "nt": 16, "nu": 0.1, "seed": 0},
    ),
}
_REFS = ("grid_a", "grid_b")


def _build_plan() -> ExperimentPlan:
    entries = tuple(
        PlanEntry(
            instrument=instrument,
            dataset_ref=ref,
            seed=seed,
            model_kwargs=dict(_INSTRUMENT_KWARGS[instrument]),
        )
        for instrument in _INSTRUMENTS
        for ref in _REFS
        for seed in _SEEDS
    )
    return ExperimentPlan(name="dispatch-vertical-slice", entries=entries)


@dataclass(frozen=True)
class _Slice:
    plan: ExperimentPlan
    serial: EvidenceStore
    merged: EvidenceStore
    manifest: object
    log: object


@pytest.fixture(scope="module")
def slice_(tmp_path_factory: pytest.TempPathFactory) -> _Slice:
    plan = _build_plan()
    base = tmp_path_factory.mktemp("dispatch_slice")



    datasets = {ref: resolve_dataset(_SPECS[ref]) for ref in _REFS}
    serial_root = base / "serial"
    run_plan(plan, datasets=datasets, store_root=serial_root, device=_SLICE_DEVICE)
    serial = EvidenceStore.load(serial_root)


    batch_root = base / "batch"
    manifest = build_dispatch_manifest(
        plan, dataset_specs=_SPECS, n_workers=2, pin_table=_PIN_TABLE
    )
    write_dispatch_manifest(manifest, batch_root)
    result = run_dispatch(batch_root)
    merged = merge_shards(
        [batch_root / "dispatch.json"], merged_root=base / "merged"
    )
    return _Slice(
        plan=plan, serial=serial, merged=merged, manifest=manifest, log=result.log
    )


def test_plan_hash_matches(slice_: _Slice) -> None:
    assert slice_.merged.plan_hash == slice_.serial.plan_hash == slice_.plan.plan_hash()


def test_per_entry_attempt_status_matches(slice_: _Slice) -> None:
    serial_status = {a["entry_index"]: a["status"] for a in slice_.serial.attempts}
    merged_status = {a["entry_index"]: a["status"] for a in slice_.merged.attempts}
    assert merged_status == serial_status


def test_evidence_and_run_spec_hash_sets_and_alignment(slice_: _Slice) -> None:
    assert set(slice_.merged.records) == set(slice_.serial.records)
    for i in slice_.serial.records:
        assert (
            slice_.merged.records[i].evidence_hash
            == slice_.serial.records[i].evidence_hash
        )
        assert (
            slice_.merged.records[i].run_spec_hash
            == slice_.serial.records[i].run_spec_hash
        )
    serial_ev = {r.evidence_hash for r in slice_.serial.records.values()}
    merged_ev = {r.evidence_hash for r in slice_.merged.records.values()}
    assert merged_ev == serial_ev


def test_ledger_two_workers_all_clean(slice_: _Slice) -> None:
    assert len(slice_.log.workers) == 2
    assert all(w.exit_code == 0 for w in slice_.log.workers)
    assert slice_.log.lost_entries == ()


def test_batch_report_has_store_and_dispatch_sections(slice_: _Slice) -> None:
    report = build_batch_report(slice_.merged, slice_.manifest, slice_.log)
    assert report.startswith(build_store_report(slice_.merged))
    assert "shard-00" in report
    assert "shard-01" in report
