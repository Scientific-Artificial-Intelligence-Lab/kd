
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from kd.harness.dispatch import (
    DispatchDatasetSpec,
    ShardSpec,
    build_dispatch_manifest,
    write_dispatch_manifest,
)
from kd.harness.dispatch_log import read_dispatch_log
from kd.harness.dispatcher import (
    DispatchRunError,
    _Dispatch,
    assemble_worker_env,
    dispatch_plan,
    run_dispatch,
    select_startable,
    worker_command,
)

from kd.harness import dispatcher as dispatcher_module
from kd.harness.plan import ExperimentPlan, PlanEntry


def _shard(
    shard_id: str = "shard-00",
    *,
    cuda: str = "",
    heavy: bool = False,
    memory_max_gb: float | None = None,
) -> ShardSpec:
    return ShardSpec(
        shard_id=shard_id,
        entry_indices=(0,),
        cuda_visible_devices=cuda,
        device="cuda" if cuda else None,
        heavy=heavy,
        timeout_seconds=None,
        memory_max_gb=memory_max_gb,
    )





def test_four_blas_threads_are_pinned_to_one(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    env = assemble_worker_env(_shard(cuda="0"))
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"


def test_cuda_visible_devices_is_shard_pin() -> None:
    assert assemble_worker_env(_shard(cuda="0"))["CUDA_VISIBLE_DEVICES"] == "0"

    assert assemble_worker_env(_shard(cuda=""))["CUDA_VISIBLE_DEVICES"] == ""


def test_nvml_cuda_check_is_injected_for_every_shard() -> None:


    assert (
        assemble_worker_env(_shard(cuda=""))["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"
    )
    assert (
        assemble_worker_env(_shard(cuda="0"))["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"
    )


def test_other_env_keys_are_inherited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KD_TEST_INHERIT_MARKER", "keepme")
    env = assemble_worker_env(_shard())
    assert env["KD_TEST_INHERIT_MARKER"] == "keepme"


def test_parent_os_environ_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    snapshot = dict(os.environ)
    assemble_worker_env(_shard(cuda="0"))
    assert dict(os.environ) == snapshot

    assert os.environ["OMP_NUM_THREADS"] == "8"





def test_worker_command_base_argv_is_exact() -> None:
    manifest_path = Path("/batch/dispatch.json")
    cmd = worker_command(
        _shard("shard-03"), manifest_path, python_executable="/usr/bin/python3"
    )
    assert cmd == [
        "/usr/bin/python3",
        "-m",
        "kd.harness.worker",
        "--dispatch",
        str(manifest_path),
        "--shard",
        "shard-03",
    ]


def test_worker_command_wraps_with_systemd_run_when_memory_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dispatcher_module.shutil, "which", lambda _name: "/usr/bin/systemd-run"
    )
    cmd = worker_command(
        _shard("shard-00", memory_max_gb=4.0),
        Path("/batch/dispatch.json"),
        python_executable="/usr/bin/python3",
    )
    assert cmd[:7] == [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "-p",
        "MemoryMax=4.0G",
    ]
    assert cmd[7:10] == ["/usr/bin/python3", "-m", "kd.harness.worker"]


def test_worker_command_fails_loud_when_systemd_run_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:


    monkeypatch.setattr(dispatcher_module.shutil, "which", lambda _name: None)
    with pytest.raises(DispatchRunError, match="systemd-run"):
        worker_command(
            _shard("shard-00", memory_max_gb=4.0),
            Path("/batch/dispatch.json"),
            python_executable="/usr/bin/python3",
        )





def test_non_heavy_always_startable_regardless_of_running() -> None:
    pending = [_shard("shard-00"), _shard("shard-01")]
    startable = select_startable(pending, heavy_running=5, max_concurrent_heavy=1)
    assert set(startable) == set(pending)


def test_no_heavy_startable_when_cap_full() -> None:
    heavy = [_shard("shard-00", heavy=True), _shard("shard-01", heavy=True)]
    startable = select_startable(heavy, heavy_running=1, max_concurrent_heavy=1)
    assert all(not s.heavy for s in startable)


def test_a_heavy_is_startable_when_cap_has_room() -> None:
    pending = [_shard("shard-00", heavy=True), _shard("shard-01")]
    startable = select_startable(pending, heavy_running=0, max_concurrent_heavy=1)
    assert any(s.heavy for s in startable)

    assert any(not s.heavy for s in startable)





def test_dispatch_plan_builds_writes_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = ExperimentPlan(
        name="wire",
        entries=(
            PlanEntry(
                instrument="sga", dataset_ref="a", seed=0, model_kwargs={}
            ),
        ),
    )
    specs = {"a": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={})}
    batch_root = tmp_path / "batch"

    calls: list[dict[str, Any]] = []
    sentinel = object()

    def _recorder(root: Path, *, python_executable: str | None = None) -> Any:
        calls.append({"batch_root": root, "python_executable": python_executable})
        return sentinel

    monkeypatch.setattr(dispatcher_module, "run_dispatch", _recorder)

    result = dispatch_plan(
        plan,
        batch_root=batch_root,
        dataset_specs=specs,
        python_executable="/opt/py",
        n_workers=1,
    )


    assert (batch_root / "dispatch.json").is_file()

    assert calls == [{"batch_root": batch_root, "python_executable": "/opt/py"}]
    assert result is sentinel


def test_dispatch_plan_routes_build_kwargs_only_to_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:


    plan = ExperimentPlan(
        name="wire2",
        entries=(
            PlanEntry(instrument="sga", dataset_ref="a", seed=0, model_kwargs={}),
            PlanEntry(instrument="sga", dataset_ref="b", seed=1, model_kwargs={}),
        ),
    )
    specs = {
        "a": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={}),
        "b": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={}),
    }
    monkeypatch.setattr(dispatcher_module, "run_dispatch", lambda *a, **k: object())

    dispatch_plan(
        plan,
        batch_root=tmp_path / "batch",
        dataset_specs=specs,
        n_workers=2,
        pin_table=["", "0"],
    )

    from kd.harness.dispatch import read_dispatch_manifest

    manifest = read_dispatch_manifest(tmp_path / "batch" / "dispatch.json")
    assert len(manifest.shards) == 2
    devices = {s.shard_id: s.cuda_visible_devices for s in manifest.shards}
    assert devices == {"shard-00": "", "shard-01": "0"}


def test_build_dispatch_manifest_is_importable_for_wiring() -> None:

    assert callable(build_dispatch_manifest)





class _FakeProc:

    def __init__(self, *, dies_on_terminate: bool) -> None:
        self.pid = 4321
        self._rc: int | None = None
        self._dies_on_terminate = dies_on_terminate
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self._rc

    def terminate(self) -> None:
        self.terminated = True
        if self._dies_on_terminate:
            self._rc = -15

    def kill(self) -> None:
        self.killed = True
        self._rc = -9

    def wait(self, timeout: float | None = None) -> int:
        if self._rc is None:
            self._rc = -9
        return self._rc

    @property
    def returncode(self) -> int | None:
        return self._rc


def _running_info(
    shard: ShardSpec, proc: _FakeProc, *, kill_reason: str | None = None
) -> dict[str, Any]:
    return {
        "proc": proc,
        "shard": shard,
        "out_f": io.StringIO(),
        "err_f": io.StringIO(),
        "start_monotonic": time.monotonic(),
        "started_at": "2026-07-22T00:00:00+00:00",
        "kill_reason": kill_reason,
        "sigkill_used": False,
        "terminate_deadline": None,
    }


def test_interrupt_all_escalates_reaps_and_preserves_timeout(tmp_path: Path) -> None:



    plan = ExperimentPlan(
        name="interrupt",
        entries=tuple(
            PlanEntry(instrument="sga", dataset_ref="a", seed=s, model_kwargs={})
            for s in range(3)
        ),
    )
    specs = {"a": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={})}
    batch = tmp_path / "batch"
    manifest = build_dispatch_manifest(
        plan, dataset_specs=specs, allocation=[[0], [1], [2]], grace_seconds=0.2
    )
    write_dispatch_manifest(manifest, batch)
    dispatch = _Dispatch(manifest, batch, sys.executable)

    graceful = _FakeProc(dies_on_terminate=True)
    stubborn = _FakeProc(dies_on_terminate=False)
    timed_out = _FakeProc(dies_on_terminate=True)
    shards = {s.shard_id: s for s in manifest.shards}
    dispatch._running = {
        "shard-00": _running_info(shards["shard-00"], graceful),
        "shard-01": _running_info(shards["shard-01"], stubborn),
        "shard-02": _running_info(
            shards["shard-02"], timed_out, kill_reason="timeout"
        ),
    }

    dispatch._interrupt_all()

    assert graceful.terminated and not graceful.killed
    assert stubborn.terminated and stubborn.killed
    assert timed_out.terminated
    rows = dispatch._rows
    assert rows["shard-00"].kill_reason == "dispatcher_interrupted"
    assert rows["shard-00"].sigkill_used is False
    assert rows["shard-01"].kill_reason == "dispatcher_interrupted"
    assert rows["shard-01"].sigkill_used is True

    assert rows["shard-02"].kill_reason == "timeout"
    assert dispatch._running == {}


def test_run_dispatch_interrupt_writes_final_ledger_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:


    plan = ExperimentPlan(
        name="interrupt-run",
        entries=(
            PlanEntry(instrument="sga", dataset_ref="a", seed=0, model_kwargs={}),
        ),
    )
    specs = {"a": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={})}
    batch = tmp_path / "batch"
    manifest = build_dispatch_manifest(
        plan, dataset_specs=specs, n_workers=1, grace_seconds=0.2
    )
    write_dispatch_manifest(manifest, batch)

    class _KIProc(_FakeProc):
        def __init__(self) -> None:
            super().__init__(dies_on_terminate=False)
            self._polls = 0

        def poll(self) -> int | None:
            self._polls += 1
            if self._polls == 1:
                raise KeyboardInterrupt
            return self._rc

    proc = _KIProc()

    def _fake_popen(
        cmd: list[str], env: Any = None, stdout: Any = None, stderr: Any = None
    ) -> _KIProc:
        return proc

    monkeypatch.setattr(dispatcher_module.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(dispatcher_module, "_POLL_INTERVAL_SECONDS", 0.0)

    with pytest.raises(KeyboardInterrupt):
        run_dispatch(batch)

    assert proc.terminated and proc.killed
    log = read_dispatch_log(batch / "dispatch-log.json")
    assert log.finished_at is not None
    worker = log.workers[0]
    assert worker.kill_reason == "dispatcher_interrupted"
    assert worker.sigkill_used is True





def test_run_dispatch_refuses_when_ledger_already_exists(tmp_path: Path) -> None:


    plan = ExperimentPlan(
        name="once",
        entries=(
            PlanEntry(instrument="sga", dataset_ref="a", seed=0, model_kwargs={}),
        ),
    )
    specs = {"a": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={})}
    batch = tmp_path / "batch"
    manifest = build_dispatch_manifest(plan, dataset_specs=specs, n_workers=1)
    write_dispatch_manifest(manifest, batch)
    (batch / "dispatch-log.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DispatchRunError, match="already exists"):
        run_dispatch(batch)


def test_run_dispatch_sweeps_systemd_run_precondition_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:


    plan = ExperimentPlan(
        name="memcap",
        entries=(
            PlanEntry(instrument="sga", dataset_ref="a", seed=0, model_kwargs={}),
        ),
    )
    specs = {"a": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={})}
    batch = tmp_path / "batch"
    manifest = build_dispatch_manifest(
        plan, dataset_specs=specs, n_workers=1, memory_max_gb=8.0
    )
    write_dispatch_manifest(manifest, batch)
    monkeypatch.setattr(dispatcher_module.shutil, "which", lambda _name: None)
    with pytest.raises(DispatchRunError, match="systemd-run"):
        run_dispatch(batch)

    assert not (batch / "shards" / "shard-00").exists()
    assert not (batch / "dispatch-log.json").exists()





def _one_shard_batch(tmp_path: Path) -> tuple[Any, Path]:
    plan = ExperimentPlan(
        name="claim",
        entries=(
            PlanEntry(instrument="sga", dataset_ref="a", seed=0, model_kwargs={}),
        ),
    )
    specs = {"a": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={})}
    batch = tmp_path / "batch"
    manifest = build_dispatch_manifest(plan, dataset_specs=specs, n_workers=1)
    write_dispatch_manifest(manifest, batch)
    return manifest, batch


def test_claim_ledger_is_exclusive_and_writes_running_ledger(tmp_path: Path) -> None:


    manifest, batch = _one_shard_batch(tmp_path)
    _Dispatch(manifest, batch, sys.executable).claim_ledger()

    claimed = read_dispatch_log(batch / "dispatch-log.json")
    assert claimed.finished_at is None
    assert claimed.workers == ()
    assert claimed.plan_hash == manifest.plan_hash

    with pytest.raises(DispatchRunError, match="already exists"):
        _Dispatch(manifest, batch, sys.executable).claim_ledger()


def test_run_dispatch_claims_ledger_before_first_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:


    _manifest, batch = _one_shard_batch(tmp_path)
    ledger_seen_at_spawn: list[bool] = []

    class _InstantProc(_FakeProc):
        def poll(self) -> int | None:
            return 0

    def _fake_popen(
        cmd: list[str], env: Any = None, stdout: Any = None, stderr: Any = None
    ) -> _InstantProc:
        ledger_seen_at_spawn.append((batch / "dispatch-log.json").exists())
        return _InstantProc(dies_on_terminate=True)

    monkeypatch.setattr(dispatcher_module.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(dispatcher_module, "_POLL_INTERVAL_SECONDS", 0.0)

    run_dispatch(batch)
    assert ledger_seen_at_spawn == [True]


def test_run_dispatch_still_refuses_pre_existing_ledger(tmp_path: Path) -> None:

    _manifest, batch = _one_shard_batch(tmp_path)
    (batch / "dispatch-log.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DispatchRunError, match="already exists"):
        run_dispatch(batch)





def test_interrupt_all_isolates_a_raising_child_and_writes_ledger(
    tmp_path: Path,
) -> None:



    plan = ExperimentPlan(
        name="isolate",
        entries=(
            PlanEntry(instrument="sga", dataset_ref="a", seed=0, model_kwargs={}),
            PlanEntry(instrument="sga", dataset_ref="b", seed=1, model_kwargs={}),
        ),
    )
    specs = {
        "a": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={}),
        "b": DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={}),
    }
    batch = tmp_path / "batch"
    manifest = build_dispatch_manifest(
        plan, dataset_specs=specs, allocation=[[0], [1]], grace_seconds=0.2
    )
    write_dispatch_manifest(manifest, batch)
    dispatch = _Dispatch(manifest, batch, sys.executable)

    class _RaisingTerminate(_FakeProc):
        def terminate(self) -> None:
            raise OSError("terminate boom")

    raising = _RaisingTerminate(dies_on_terminate=False)
    normal = _FakeProc(dies_on_terminate=True)
    shards = {s.shard_id: s for s in manifest.shards}
    dispatch._running = {
        "shard-00": _running_info(shards["shard-00"], raising),
        "shard-01": _running_info(shards["shard-01"], normal),
    }

    dispatch._interrupt_all()



    assert normal.terminated
    assert raising.killed
    assert dispatch._running == {}
    assert set(dispatch._rows) == {"shard-00", "shard-01"}


    dispatch._write_log(finished_at="2026-07-22T00:00:05+00:00", lost=())
    reloaded = read_dispatch_log(batch / "dispatch-log.json")
    assert {w.shard_id for w in reloaded.workers} == {"shard-00", "shard-01"}
