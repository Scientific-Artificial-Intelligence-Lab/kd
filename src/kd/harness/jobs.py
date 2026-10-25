
from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Event

from kd.harness.schedule import (
    FifoBackfillQueue,
    Job,
    ResourceLimits,
    ResourceQueues,
    Running,
    SchedulerState,
    choose_gpu,
    resolve_gpu_pool,
)
from kd.harness.supervise import ProcessResult, ProcessSupervisor


@dataclass(frozen=True, kw_only=True)
class JobResult:

    job: Job
    gpu: int | None
    process: ProcessResult


def _limits_for(jobs: Sequence[Job], limits: ResourceLimits) -> ResourceLimits:
    if not any(job.device == "cuda" for job in jobs):
        return limits
    if limits.gpu_slots == 0:

        raise ValueError("a cuda job cannot run with gpu_slots=0")
    return resolve_gpu_pool(limits, os.environ)


def run_jobs(
    jobs: Sequence[Job],
    *,
    limits: ResourceLimits,
    launch: Callable[[Job, int | None], ProcessSupervisor],
    finished: Callable[[JobResult], None],
    observe: Callable[[SchedulerState], None] | None = None,
    poll_seconds: float = 0.5,
) -> tuple[str, ...]:
    limits = _limits_for(jobs, limits)
    pending = deque(jobs)
    queue = (
        ResourceQueues() if limits.policy == "resource_queues" else FifoBackfillQueue()
    )
    active: dict[Future[ProcessResult], Running] = {}
    ran: list[str] = []
    cancelled = Event()
    state = "aborted"

    def publish(state: str) -> None:
        if observe is not None:
            observe(queue.state(pending, list(active.values()), limits, state))

    try:
        with ThreadPoolExecutor(max_workers=limits.slots) as pool:
            try:
                while pending or active:
                    publish("running")
                    while pending:
                        running = list(active.values())
                        job = queue.next_job(pending, running, limits)
                        if job is None:
                            break
                        gpu = choose_gpu(job, running, limits)
                        supervisor = launch(job, gpu)
                        try:
                            future = pool.submit(
                                supervisor.wait,
                                poll_seconds=poll_seconds,
                                cancelled=cancelled,
                            )
                        except BaseException:


                            supervisor.abort()
                            raise
                        active[future] = (job, gpu)
                        ran.append(job.key)
                        publish("running")
                    done, _ = wait(
                        active,
                        return_when=FIRST_COMPLETED,
                        timeout=1.0 if limits.policy == "resource_queues" else None,
                    )
                    for future in [f for f in active if f in done]:
                        job, gpu = active.pop(future)
                        finished(JobResult(job=job, gpu=gpu, process=future.result()))
            except BaseException:
                cancelled.set()
                raise
        state = "finished"
    finally:


        active.clear()
        publish(state)
    return tuple(ran)


__all__ = ["JobResult", "run_jobs"]
