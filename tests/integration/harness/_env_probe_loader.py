
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import kd


_LEDGER_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTORCH_NVML_BASED_CUDA_CHECK",
)


def load_with_env_snapshot(
    *,
    snapshot_path: str,
    nx: int = 32,
    nt: int = 16,
    nu: float = 0.1,
    seed: int = 0,
) -> Any:
    snapshot = {key: os.environ.get(key) for key in _LEDGER_ENV_KEYS}
    Path(snapshot_path).write_text(json.dumps(snapshot), encoding="utf-8")
    return kd.generate_burgers_data(nx=nx, nt=nt, nu=nu, seed=seed)
