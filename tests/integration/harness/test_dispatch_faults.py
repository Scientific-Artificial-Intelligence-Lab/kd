
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest
from kd.harness.dispatch import (
    DispatchDatasetSpec,
    build_dispatch_manifest,
    write_dispatch_manifest,
)
from kd.harness.dispatcher import (
    assemble_worker_env,
    dispatch_plan,
    run_dispatch,
    worker_command,
)
from kd.harness.merge import ShardMissingError, merge_shards

from kd.harness.plan import ExperimentPlan, PlanEntry
from kd.harness.store import EvidenceStore

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SLEEPY = "tests.integration.harness._slow_loader.load_sleepy"
_SGA = {
    "generations": 2,
    "population": 8,
    "depth": 3,
    "width": 4,
    "maxit": 3,
    "str_iters": 3,
    "d_tol": 0.5,
}


def _poll_attempts(index_path: Path, *, minimum: int, deadline_s: float) -> int:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            attempts = json.loads(index_path.read_text(encoding="utf-8"))["attempts"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            attempts = []
        if len(attempts) >= minimum:
            return len(attempts)
        time.sleep(0.5)
    return -1





def test_kill_worker_then_retry_converges(tmp_path: Path) -> None:



    plan = ExperimentPlan(
        name="faults-kill",
        entries=(
            PlanEntry(instrument="sga", dataset_ref="warm", seed=0, model_kwargs=_SGA),
            PlanEntry(instrument="sga", dataset_ref="gate", seed=1, model_kwargs=_SGA),
            PlanEntry(instrument="sga", dataset_ref="gate", seed=2, model_kwargs=_SGA),
        ),
    )
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    _burgers = {"nx": 32, "nt": 16, "nu": 0.1, "seed": 0}
    gated_specs = {
        "warm": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs=_burgers),
        "gate": DispatchDatasetSpec(
            loader="tests.integration.harness._gate_loader.load_gate",
            kwargs={"gate_dir": str(gate_dir), **_burgers},
        ),
    }

    normal_specs = {
        "warm": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs=_burgers),
        "gate": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs=_burgers),
    }
    batch1 = tmp_path / "batch1"
    manifest = build_dispatch_manifest(plan, dataset_specs=gated_specs, n_workers=1)
    write_dispatch_manifest(manifest, batch1)
    shard = manifest.shards[0]

    with (batch1 / "logs" / "shard-00.out").open("w") as out, (
        batch1 / "logs" / "shard-00.err"
    ).open("w") as err:
        proc = subprocess.Popen(
            worker_command(
                shard, batch1 / "dispatch.json", python_executable=sys.executable
            ),
            env=assemble_worker_env(shard),
            cwd=_REPO_ROOT,
            stdout=out,
            stderr=err,
        )
        try:
            index_path = batch1 / "shards" / "shard-00" / "index.json"


            assert _poll_attempts(index_path, minimum=1, deadline_s=60) >= 1
            proc.kill()
        finally:
            proc.wait(timeout=30)


    shard_store = EvidenceStore.load(batch1 / "shards" / "shard-00")
    assert 0 < len(shard_store.attempts) < 3

    merged1 = merge_shards([batch1 / "dispatch.json"], merged_root=tmp_path / "merged1")
    done = {a["entry_index"] for a in merged1.attempts}
    lost = [i for i in range(3) if i not in done]
    assert lost, "the kill must have dropped at least one entry"


    batch2 = tmp_path / "batch2"
    dispatch_plan(
        plan,
        batch_root=batch2,
        dataset_specs=normal_specs,
        n_workers=1,
        entry_indices=lost,
    )

    merged2 = merge_shards(
        [batch1 / "dispatch.json", batch2 / "dispatch.json"],
        merged_root=tmp_path / "merged2",
    )
    assert set(merged2.records) == {0, 1, 2}

    for i in done & set(merged1.records):
        assert merged2.records[i].record_hash == merged1.records[i].record_hash





def test_timeout_records_kill_reason_and_missing_shard(tmp_path: Path) -> None:
    plan = ExperimentPlan(
        name="faults-timeout",
        entries=tuple(
            PlanEntry(instrument="sga", dataset_ref="slow", seed=s, model_kwargs=_SGA)
            for s in range(2)
        ),
    )
    specs = {"slow": DispatchDatasetSpec(loader=_SLEEPY, kwargs={"seconds": 30})}
    batch = tmp_path / "batch"
    manifest = build_dispatch_manifest(
        plan,
        dataset_specs=specs,
        n_workers=1,
        default_timeout_seconds=1.0,
        grace_seconds=1.0,
    )
    write_dispatch_manifest(manifest, batch)

    result = run_dispatch(batch)

    worker = result.log.workers[0]
    assert worker.kill_reason == "timeout"

    assert result.log.lost_entries == (0, 1)
    assert not (batch / "shards" / "shard-00").exists()

    with pytest.raises(ShardMissingError):
        merge_shards([batch / "dispatch.json"], merged_root=tmp_path / "m_strict")
    merged = merge_shards(
        [batch / "dispatch.json"],
        merged_root=tmp_path / "m_lenient",
        require_all_shards=False,
    )
    assert merged.records == {}





def test_heavy_gate_serializes_two_heavy_workers(tmp_path: Path) -> None:
    plan = ExperimentPlan(
        name="faults-heavy",
        entries=(
            PlanEntry(instrument="sga", dataset_ref="a", seed=0, model_kwargs=_SGA),
            PlanEntry(instrument="sga", dataset_ref="b", seed=1, model_kwargs=_SGA),
        ),
    )
    specs = {
        "a": DispatchDatasetSpec(loader=_SLEEPY, kwargs={"seconds": 2}),
        "b": DispatchDatasetSpec(loader=_SLEEPY, kwargs={"seconds": 2}),
    }
    batch = tmp_path / "batch"
    manifest = build_dispatch_manifest(
        plan,
        dataset_specs=specs,
        n_workers=2,
        heavy_keys={("sga", "a"), ("sga", "b")},
        max_concurrent_heavy=1,
    )
    write_dispatch_manifest(manifest, batch)

    result = run_dispatch(batch)

    assert result.log.heavy_peak == 1
    rows = sorted(
        result.log.workers, key=lambda w: datetime.fromisoformat(w.started_at)
    )
    first_end = datetime.fromisoformat(rows[0].finished_at)
    second_start = datetime.fromisoformat(rows[1].started_at)

    assert first_end <= second_start
