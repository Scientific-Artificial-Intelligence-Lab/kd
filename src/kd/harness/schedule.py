
from __future__ import annotations

from collections import Counter, deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal

Device = Literal["cpu", "cuda"]
Policy = Literal["fifo", "backfill", "resource_queues"]

_POLICIES: Final = ("fifo", "backfill", "resource_queues")
_DEVICES: Final[tuple[Device, ...]] = ("cpu", "cuda")




_THREAD_PINS: Final = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHON_JULIACALL_THREADS",
    "JULIA_NUM_THREADS",
)


PINNED_ENVIRONMENT: Final = (*_THREAD_PINS, "PYTHONHASHSEED")


@dataclass(frozen=True, kw_only=True)
class ResourceLimits:

    slots: int = 1
    memory_budget_gb: float | None = None
    gpu_slots: int = 0
    gpus: tuple[int, ...] | None = None
    jobs_per_gpu: int = 1
    policy: Policy = "fifo"

    def __post_init__(self) -> None:


        if self.policy not in _POLICIES:
            raise ValueError(f"policy must be one of {_POLICIES}; got {self.policy!r}")
        if self.slots < 1 or self.gpu_slots < 0 or self.jobs_per_gpu < 1:
            raise ValueError("slots and jobs_per_gpu must be >= 1, gpu_slots >= 0")
        if self.memory_budget_gb is not None and self.memory_budget_gb <= 0:
            raise ValueError("memory_budget_gb must be positive or None")
        if self.gpus is not None:
            if not self.gpus or len(set(self.gpus)) != len(self.gpus):
                raise ValueError("gpus must be non-empty distinct card indices")
            if self.gpu_slots > len(self.gpus) * self.jobs_per_gpu:
                raise ValueError(
                    f"gpu_slots={self.gpu_slots} exceeds the {len(self.gpus)} "
                    f"listed cards times jobs_per_gpu={self.jobs_per_gpu}"
                )


@dataclass(frozen=True, kw_only=True)
class Job:

    key: str
    device: Device
    memory_estimate_gb: float = 0.0
    payload: Any = None



Running = tuple[Job, int | None]


@dataclass(frozen=True, kw_only=True)
class QueueHead:

    key: str
    reasons: list[str]


@dataclass(frozen=True, kw_only=True)
class SchedulerState:

    state: str
    active: dict[str, int]
    pending: dict[str, int]
    waiting: dict[str, QueueHead | None]
    reserved_for: str | None
    estimated_memory_gb: float
    limits: ResourceLimits


def admission_reasons(
    job: Job, active: Collection[Running], limits: ResourceLimits
) -> list[str]:
    reasons = []
    if len(active) >= limits.slots:
        reasons.append("slots")
    if (
        job.device == "cuda"
        and sum(running.device == "cuda" for running, _ in active) >= limits.gpu_slots
    ):
        reasons.append("gpu_capacity")
    if (
        limits.memory_budget_gb is not None
        and active
        and job.memory_estimate_gb
        + sum(running.memory_estimate_gb for running, _ in active)
        > limits.memory_budget_gb
    ):
        reasons.append("memory")
    return reasons


def fits(job: Job, active: Collection[Running], limits: ResourceLimits) -> bool:
    return not admission_reasons(job, active, limits)


@dataclass
class FifoBackfillQueue:

    overtaken: dict[str, int] = field(default_factory=dict)

    def next_job(
        self, pending: deque[Job], active: Collection[Running], limits: ResourceLimits
    ) -> Job | None:
        if len(active) >= limits.slots:
            return None
        skipped: list[str] = []
        for index, job in enumerate(pending):
            if fits(job, active, limits):
                del pending[index]
                self.overtaken.pop(job.key, None)
                for earlier in skipped:
                    self.overtaken[earlier] = self.overtaken.get(earlier, 0) + 1
                return job
            if (
                limits.policy == "fifo"
                or self.overtaken.get(job.key, 0) >= limits.slots
            ):
                break
            skipped.append(job.key)
        return None

    def state(
        self,
        pending: deque[Job],
        active: Collection[Running],
        limits: ResourceLimits,
        state: str,
    ) -> SchedulerState:
        return _scheduler_state(pending, active, limits, state, reserved_for=None)


def _queue_heads(pending: deque[Job]) -> dict[str, Job]:
    heads: dict[str, Job] = {}
    for job in pending:
        heads.setdefault(job.device, job)
    return heads


@dataclass
class ResourceQueues:

    preferred: str | None = None
    memory_passes: dict[str, int] = field(default_factory=dict)
    reserved_for: str | None = None

    def next_job(
        self, pending: deque[Job], active: Collection[Running], limits: ResourceLimits
    ) -> Job | None:
        heads = _queue_heads(pending)
        reasons = {
            device: admission_reasons(job, active, limits)
            for device, job in heads.items()
        }
        for device in sorted(heads, key=lambda device: device != self.preferred):
            job = heads[device]
            if reasons[device] or (
                self.reserved_for is not None and self.reserved_for != job.key
            ):
                continue
            for other, head in heads.items():
                if (
                    other != device
                    and "memory" in reasons[other]
                    and "gpu_capacity" not in reasons[other]
                ):
                    passes = self.memory_passes.get(head.key, 0) + 1
                    self.memory_passes[head.key] = passes
                    if passes >= limits.slots:
                        self.reserved_for = head.key
            pending.remove(job)
            self.memory_passes.pop(job.key, None)
            if self.reserved_for == job.key:
                self.reserved_for = None
            self.preferred = "cuda" if device == "cpu" else "cpu"
            return job
        return None

    def state(
        self,
        pending: deque[Job],
        active: Collection[Running],
        limits: ResourceLimits,
        state: str,
    ) -> SchedulerState:
        return _scheduler_state(
            pending, active, limits, state, reserved_for=self.reserved_for
        )


def _scheduler_state(
    pending: deque[Job],
    active: Collection[Running],
    limits: ResourceLimits,
    state: str,
    *,
    reserved_for: str | None,
) -> SchedulerState:
    waiting: dict[str, QueueHead | None] = dict.fromkeys(_DEVICES)
    for device, job in _queue_heads(pending).items():
        reasons = admission_reasons(job, active, limits)
        if reserved_for is not None and reserved_for != job.key:
            reasons.append("memory_reservation")
        waiting[device] = QueueHead(key=job.key, reasons=reasons)
    return SchedulerState(
        state=state,
        active={d: sum(job.device == d for job, _ in active) for d in _DEVICES},
        pending={d: sum(job.device == d for job in pending) for d in _DEVICES},
        waiting=waiting,
        reserved_for=reserved_for,
        estimated_memory_gb=sum((job.memory_estimate_gb for job, _ in active), 0.0),
        limits=limits,
    )


def _visible_gpus(environ: Mapping[str, str]) -> tuple[int, ...] | None:
    value = environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens or not all(token.isdigit() for token in tokens):
        return None
    return tuple(int(token) for token in tokens)


def resolve_gpu_pool(
    limits: ResourceLimits, environ: Mapping[str, str]
) -> ResourceLimits:
    visible = _visible_gpus(environ)
    if limits.gpus is not None:
        if visible is not None and not set(limits.gpus) <= set(visible):
            raise ValueError(
                f"gpus {list(limits.gpus)} names cards outside "
                f"CUDA_VISIBLE_DEVICES={environ['CUDA_VISIBLE_DEVICES']!r}"
            )
        return limits
    if visible is None:
        if limits.gpu_slots > 1:
            raise ValueError(
                "gpu_slots above 1 needs the cards: list them in gpus or run "
                "under CUDA_VISIBLE_DEVICES naming them by index"
            )
        return limits
    if limits.gpu_slots > len(visible) * limits.jobs_per_gpu:
        raise ValueError(
            f"gpu_slots={limits.gpu_slots} exceeds the {len(visible)} cards of "
            f"CUDA_VISIBLE_DEVICES={environ['CUDA_VISIBLE_DEVICES']!r} times "
            f"jobs_per_gpu={limits.jobs_per_gpu}"
        )
    return replace(limits, gpus=visible)


def choose_gpu(
    job: Job, active: Collection[Running], limits: ResourceLimits
) -> int | None:
    gpus = limits.gpus
    if job.device != "cuda" or gpus is None:
        return None
    held = Counter(gpu for _, gpu in active if gpu is not None)
    return min(
        (gpu for gpu in gpus if held[gpu] < limits.jobs_per_gpu),
        key=lambda gpu: (held[gpu], gpus.index(gpu)),
    )


def child_environment(
    base: Mapping[str, str],
    *,
    blas_threads: int,
    python_hash_seed: int | None,
    gpu: int | None,
) -> dict[str, str]:
    env = dict(base)
    for name in _THREAD_PINS:
        env[name] = str(blas_threads)
    if python_hash_seed is not None:
        env["PYTHONHASHSEED"] = str(python_hash_seed)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


__all__ = [
    "PINNED_ENVIRONMENT",
    "FifoBackfillQueue",
    "Job",
    "QueueHead",
    "ResourceLimits",
    "ResourceQueues",
    "Running",
    "SchedulerState",
    "admission_reasons",
    "child_environment",
    "choose_gpu",
    "fits",
    "resolve_gpu_pool",
]
