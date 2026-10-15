
from __future__ import annotations

_CPU_ALLOC_FRAGMENT = "DefaultCPUAllocator"


def is_cpu_alloc_failure(exc: Exception) -> bool:
    return _CPU_ALLOC_FRAGMENT in str(exc)
