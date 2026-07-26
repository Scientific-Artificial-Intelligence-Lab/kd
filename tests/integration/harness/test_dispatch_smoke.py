
from __future__ import annotations

import json
from pathlib import Path

import pytest
from kd.harness.dispatch import DispatchDatasetSpec
from kd.harness.dispatch_log import read_dispatch_log
from kd.harness.dispatcher import dispatch_plan
from kd.harness.merge import merge_shards

from kd.harness.plan import ExperimentPlan, PlanEntry

pytestmark = pytest.mark.integration

_SGA = {
    "generations": 2,
    "population": 8,
    "depth": 3,
    "width": 4,
    "maxit": 3,
    "str_iters": 3,
    "d_tol": 0.5,
}


def test_dispatch_smoke_single_cpu_worker(tmp_path: Path) -> None:
    plan = ExperimentPlan(
        name="dispatch-smoke",
        entries=tuple(
            PlanEntry(
                instrument="sga", dataset_ref="burgers", seed=s, model_kwargs=_SGA
            )
            for s in (0, 1)
        ),
    )
    snapshot_path = tmp_path / "child_env.json"
    specs = {
        "burgers": DispatchDatasetSpec(
            loader="tests.integration.harness._env_probe_loader."
            "load_with_env_snapshot",
            kwargs={
                "snapshot_path": str(snapshot_path),
                "nx": 32,
                "nt": 16,
                "nu": 0.1,
                "seed": 0,
            },
        )
    }
    batch_root = tmp_path / "batch"

    result = dispatch_plan(
        plan, batch_root=batch_root, dataset_specs=specs, n_workers=1
    )


    worker = result.log.workers[0]
    assert worker.exit_code == 0
    assert worker.env["OMP_NUM_THREADS"] == "1"
    assert worker.env["MKL_NUM_THREADS"] == "1"
    assert worker.env["OPENBLAS_NUM_THREADS"] == "1"
    assert worker.env["NUMEXPR_NUM_THREADS"] == "1"
    assert worker.env["CUDA_VISIBLE_DEVICES"] == ""
    assert worker.env["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"





    child_env = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert child_env == worker.env


    log = read_dispatch_log(batch_root / "dispatch-log.json")
    assert log.plan_hash == plan.plan_hash()


    merged = merge_shards(
        [batch_root / "dispatch.json"], merged_root=tmp_path / "merged"
    )
    assert merged.plan_hash == plan.plan_hash()
    assert set(merged.records) == {0, 1}
