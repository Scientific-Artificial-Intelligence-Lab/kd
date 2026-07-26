
from __future__ import annotations

import time
from typing import Any

import kd


def load_sleepy(
    *,
    seconds: float = 0.0,
    nx: int = 32,
    nt: int = 8,
    nu: float = 0.1,
    seed: int = 0,
) -> Any:
    time.sleep(seconds)
    return kd.generate_burgers_data(nx=nx, nt=nt, nu=nu, seed=seed)
