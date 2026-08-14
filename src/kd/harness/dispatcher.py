
from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Final

from kd.harness._dispatch_schema import DispatchDatasetSpec, ShardSpec
from kd.harness.dispatch import (
    DispatchManifest,
    DispatchManifestError,
    build_dispatch_manifest,
    read_dispatch_manifest,
    write_dispatch_manifest,
)
from kd.harness.dispatch_log import (
    _DISPATCH_LOG_V1_WORKER_ENV_KEYS,
    DispatchLog,
    WorkerLogRow,
    write_dispatch_log,
)
from kd.harness.plan import ExperimentPlan

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS: Final[float] = 1.0
_MANIFEST_FILENAME: Final[str] = "dispatch.json"
_LOG_FILENAME: Final[str] = "dispatch-log.json"


class DispatchRunError(RuntimeError):
    pass


@dataclass(frozen=True, kw_only=True)
class DispatchResult:

    batch_root: Path
    manifest: DispatchManifest
    log: DispatchLog





def assemble_worker_env(shard: ShardSpec) -> dict[str, str]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = shard.cuda_visible_devices




    env["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    return env


def worker_command(
    shard: ShardSpec,
    manifest_path: Path,
    *,
    python_executable: str,
) -> list[str]:
    base = [
        python_executable,
        "-m",
        "kd.harness.worker",
        "--dispatch",
        str(manifest_path),
        "--shard",
        shard.shard_id,
    ]
    if shard.memory_max_gb is None:
        return base
    prefix = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "-p",
        f"MemoryMax={shard.memory_max_gb}G",
    ]
    return prefix + base


def select_startable(
    pending: Sequence[ShardSpec],
    heavy_running: int,
    max_concurrent_heavy: int,
) -> list[ShardSpec]:
    room = max_concurrent_heavy - heavy_running
    startable: list[ShardSpec] = []
    for shard in pending:
        if not shard.heavy:
            startable.append(shard)
        elif room > 0:
            startable.append(shard)
            room -= 1
    return startable


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _row_env(shard: ShardSpec) -> dict[str, str]:
    assembled = assemble_worker_env(shard)
    return {key: assembled[key] for key in _DISPATCH_LOG_V1_WORKER_ENV_KEYS}


def _compute_lost_entries(manifest: DispatchManifest, batch_root: Path) -> list[int]:
    lost: list[int] = []
    for shard in manifest.shards:
        index_path = batch_root / "shards" / shard.shard_id / "index.json"
        delivered: set[int] = set()
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            for attempt in data["attempts"]:
                delivered.add(attempt["entry_index"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError):
            delivered = set()
        for local, full in enumerate(shard.entry_indices):
            if local not in delivered:
                lost.append(full)
    return sorted(lost)





class _Dispatch:

    def __init__(
        self,
        manifest: DispatchManifest,
        batch_root: Path,
        python_executable: str,
    ) -> None:
        self._manifest = manifest
        self._batch_root = batch_root
        self._manifest_path = batch_root / _MANIFEST_FILENAME
        self._log_path = batch_root / _LOG_FILENAME
        self._logs_dir = batch_root / "logs"
        self._python = python_executable
        self._grace = manifest.resources.grace_seconds
        self._max_heavy = manifest.resources.max_concurrent_heavy
        self._rows: dict[str, WorkerLogRow] = {}
        self._running: dict[str, dict[str, Any]] = {}
        self._heavy_running = 0
        self._heavy_peak = 0
        self._started_at = _now_iso()

    def _write_log(
        self, *, finished_at: str | None = None, lost: Sequence[int] = ()
    ) -> DispatchLog:
        log = DispatchLog(
            plan_hash=self._manifest.plan_hash,
            workers=tuple(self._rows.values()),
            lost_entries=tuple(lost),
            heavy_peak=self._heavy_peak,
            started_at=self._started_at,
            finished_at=finished_at,
        )
        write_dispatch_log(log, self._log_path)
        return log

    def claim_ledger(self) -> None:
        initial = DispatchLog(
            plan_hash=self._manifest.plan_hash,
            workers=(),
            lost_entries=(),
            heavy_peak=0,
            started_at=self._started_at,
            finished_at=None,
        )
        payload = json.dumps(
            initial.to_dict(), indent=2, allow_nan=False, sort_keys=True
        )
        try:
            fd = os.open(
                self._log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
            )
        except FileExistsError as exc:
            raise DispatchRunError(
                f"{_LOG_FILENAME} already exists at {self._log_path}; refusing to "
                f"re-run a batch on the same root (this would overwrite the run "
                f"ledger and can double-spawn the same shard roots). The retry "
                f"flow is a NEW sibling batch_root built with "
                f"entry_indices=<lost entries>, then merge_shards([first, retry], "
                f"...)"
            ) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)

    def _spawn(self, shard: ShardSpec) -> bool:
        command = worker_command(
            shard, self._manifest_path, python_executable=self._python
        )
        started_at = _now_iso()




        out_f: IO[Any] | None = None
        err_f: IO[Any] | None = None
        try:
            out_f = (self._logs_dir / f"{shard.shard_id}.out").open(
                "w", encoding="utf-8"
            )
            err_f = (self._logs_dir / f"{shard.shard_id}.err").open(
                "w", encoding="utf-8"
            )
            proc = subprocess.Popen(
                command,
                env=assemble_worker_env(shard),
                stdout=out_f,
                stderr=err_f,
            )
        except OSError as exc:
            if out_f is not None:
                _close(out_f)
            if err_f is not None:
                _close(err_f)
            logger.warning("spawn failed for shard %s: %s", shard.shard_id, exc)
            self._rows[shard.shard_id] = WorkerLogRow(
                shard_id=shard.shard_id,
                pid=None,
                exit_code=None,
                kill_reason="spawn_failed",
                sigkill_used=False,
                started_at=started_at,
                finished_at=_now_iso(),
                wallclock_seconds=0.0,
                timeout_seconds=shard.timeout_seconds,
                env=_row_env(shard),
                heavy=shard.heavy,
            )
            return False
        self._running[shard.shard_id] = {
            "proc": proc,
            "shard": shard,
            "out_f": out_f,
            "err_f": err_f,
            "start_monotonic": time.monotonic(),
            "started_at": started_at,
            "kill_reason": None,
            "sigkill_used": False,
            "terminate_deadline": None,
        }
        self._rows[shard.shard_id] = WorkerLogRow(
            shard_id=shard.shard_id,
            pid=proc.pid,
            exit_code=None,
            kill_reason=None,
            sigkill_used=False,
            started_at=started_at,
            finished_at=None,
            wallclock_seconds=None,
            timeout_seconds=shard.timeout_seconds,
            env=_row_env(shard),
            heavy=shard.heavy,
        )
        return True

    def _finalize(self, shard_id: str, exit_code: int | None) -> None:
        info = self._running.pop(shard_id)
        _close(info["out_f"])
        _close(info["err_f"])
        shard: ShardSpec = info["shard"]
        self._rows[shard_id] = WorkerLogRow(
            shard_id=shard_id,
            pid=info["proc"].pid,
            exit_code=exit_code,
            kill_reason=info["kill_reason"],
            sigkill_used=info["sigkill_used"],
            started_at=info["started_at"],
            finished_at=_now_iso(),
            wallclock_seconds=time.monotonic() - info["start_monotonic"],
            timeout_seconds=shard.timeout_seconds,
            env=_row_env(shard),
            heavy=shard.heavy,
        )
        if shard.heavy:
            self._heavy_running -= 1

    def _poll_running(self) -> bool:
        finished = False
        for shard_id in list(self._running):
            info = self._running[shard_id]
            proc = info["proc"]
            shard: ShardSpec = info["shard"]
            exit_code = proc.poll()
            if exit_code is None:
                self._enforce_timeout(info, proc, shard)
                continue
            self._finalize(shard_id, exit_code)
            finished = True
        return finished

    def _enforce_timeout(
        self, info: dict[str, Any], proc: subprocess.Popen[bytes], shard: ShardSpec
    ) -> None:
        now = time.monotonic()
        if (
            info["kill_reason"] is None
            and shard.timeout_seconds is not None
            and now - info["start_monotonic"] > shard.timeout_seconds
        ):


            if proc.poll() is None:
                proc.terminate()
                info["kill_reason"] = "timeout"
                info["terminate_deadline"] = now + self._grace
        elif (
            info["kill_reason"] == "timeout"
            and info["terminate_deadline"] is not None
            and now > info["terminate_deadline"]
            and proc.poll() is None
        ):
            proc.kill()
            info["sigkill_used"] = True

    def run(self) -> DispatchLog:
        pending = list(self._manifest.shards)
        try:
            while pending or self._running:
                startable = select_startable(
                    pending, self._heavy_running, self._max_heavy
                )
                if startable:
                    started_ids: set[str] = set()
                    for shard in startable:
                        running = self._spawn(shard)
                        started_ids.add(shard.shard_id)
                        if running and shard.heavy:
                            self._heavy_running += 1
                            self._heavy_peak = max(
                                self._heavy_peak, self._heavy_running
                            )
                    pending = [s for s in pending if s.shard_id not in started_ids]
                    self._write_log()
                if not self._running:
                    continue
                time.sleep(_POLL_INTERVAL_SECONDS)
                if self._poll_running():
                    self._write_log()
        except BaseException:





            self._interrupt_all()
            lost = _compute_lost_entries(self._manifest, self._batch_root)
            self._write_log(finished_at=_now_iso(), lost=lost)
            raise
        lost = _compute_lost_entries(self._manifest, self._batch_root)
        return self._write_log(finished_at=_now_iso(), lost=lost)

    def _interrupt_all(self) -> None:
        deadline = time.monotonic() + self._grace
        for info in self._running.values():
            if _proc_poll(info["proc"]) is None:
                _proc_terminate(info["proc"])



                if info["kill_reason"] is None:
                    info["kill_reason"] = "dispatcher_interrupted"
        while time.monotonic() < deadline and any(
            _proc_poll(info["proc"]) is None for info in self._running.values()
        ):
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                break
        for info in self._running.values():
            if _proc_poll(info["proc"]) is None:
                _proc_kill(info["proc"])
                info["sigkill_used"] = True
            _proc_wait(info["proc"])
        for shard_id in list(self._running):
            self._finalize(shard_id, self._running[shard_id]["proc"].returncode)


def _close(handle: IO[Any]) -> None:
    with contextlib.suppress(OSError):
        handle.close()


def _proc_poll(proc: subprocess.Popen[bytes]) -> int | None:
    try:
        return proc.poll()
    except OSError:
        return None


def _proc_terminate(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(OSError):
        proc.terminate()


def _proc_kill(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(OSError):
        proc.kill()


def _proc_wait(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(OSError):
        proc.wait()


def run_dispatch(
    batch_root: Path,
    *,
    python_executable: str | None = None,
) -> DispatchResult:
    batch_root = Path(batch_root)
    manifest_path = batch_root / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise DispatchRunError(f"no {_MANIFEST_FILENAME} at {manifest_path}")
    try:
        manifest = read_dispatch_manifest(manifest_path)
    except DispatchManifestError as exc:
        raise DispatchRunError(
            f"unreadable manifest at {manifest_path}: {exc}"
        ) from exc
    if not (batch_root / "shards").is_dir() or not (batch_root / "logs").is_dir():
        raise DispatchRunError(f"batch_root is missing shards/ or logs/: {batch_root}")
    if (
        any(shard.memory_max_gb is not None for shard in manifest.shards)
        and shutil.which("systemd-run") is None
    ):




        raise DispatchRunError(
            "one or more shards request memory_max_gb but systemd-run is not "
            "available; refusing to run without the requested cgroup memory cap"
        )
    python_executable = python_executable or sys.executable
    dispatch = _Dispatch(manifest, batch_root, python_executable)


    dispatch.claim_ledger()
    log = dispatch.run()



    from kd.harness.dispatch_report import write_dispatch_report

    write_dispatch_report(manifest, log, batch_root)
    return DispatchResult(batch_root=batch_root, manifest=manifest, log=log)


def dispatch_plan(
    plan: ExperimentPlan,
    *,
    batch_root: Path,
    dataset_specs: Mapping[str, DispatchDatasetSpec],
    python_executable: str | None = None,
    **build_kwargs: Any,
) -> DispatchResult:
    manifest = build_dispatch_manifest(
        plan, dataset_specs=dataset_specs, **build_kwargs
    )
    write_dispatch_manifest(manifest, Path(batch_root))
    return run_dispatch(Path(batch_root), python_executable=python_executable)


__all__ = [
    "DispatchResult",
    "DispatchRunError",
    "assemble_worker_env",
    "dispatch_plan",
    "run_dispatch",
    "select_startable",
    "worker_command",
]
