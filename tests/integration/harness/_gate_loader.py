
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import kd
from kd.data.schema import PDEDataset



_BLOCK_TIMEOUT_S: float = 120.0


class _GateDataset(PDEDataset):

    def __getattribute__(self, name: str) -> Any:
        instance_dict = object.__getattribute__(self, "__dict__")
        if name == "fields" and instance_dict.get("_gate_armed", False):
            instance_dict["_gate_armed"] = False
            started = instance_dict.get("_gate_started")
            if started is not None:
                Path(started).touch()
            release = instance_dict.get("_gate_release")
            deadline = time.monotonic() + _BLOCK_TIMEOUT_S
            while time.monotonic() < deadline:
                if release is not None and Path(release).exists():
                    break
                time.sleep(0.05)
        return object.__getattribute__(self, name)


def load_gate(
    *,
    gate_dir: str,
    nx: int = 32,
    nt: int = 16,
    nu: float = 0.1,
    seed: int = 0,
) -> PDEDataset:
    dataset = kd.generate_burgers_data(nx=nx, nt=nt, nu=nu, seed=seed)
    gate = Path(gate_dir)
    dataset.__class__ = _GateDataset
    dataset._gate_started = str(gate / "gate_started")
    dataset._gate_release = str(gate / "gate_release")
    dataset._gate_armed = True
    return dataset
