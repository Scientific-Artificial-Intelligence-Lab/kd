
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from kd.data._field_names import is_reserved_field_name, reserved_field_name_message
from kd.data.source import DatasetSource

TRAJECTORY_FINGERPRINT_DOMAIN: Final = "kd-trajectories-v1"
_FINGERPRINT_PREFIX: Final = TRAJECTORY_FINGERPRINT_DOMAIN + ":"


@dataclass(frozen=True, kw_only=True)
class Trajectory:

    id: str
    t: np.ndarray
    state: np.ndarray
    params: Mapping[str, float]

    def __post_init__(self) -> None:
        where = f"trajectory {self.id!r}"
        if self.t.ndim != 1:
            raise ValueError(f"{where}: t must be 1-D; got shape {self.t.shape}")
        if self.t.shape[0] < 3:
            raise ValueError(f"{where}: needs at least 3 points; has {self.t.shape[0]}")
        if not np.all(np.isfinite(self.t)):
            raise ValueError(f"{where}: t is not finite everywhere")
        if not np.all(np.diff(self.t) > 0):
            raise ValueError(f"{where}: t is not strictly increasing")
        if self.state.ndim != 2 or self.state.shape[1] != self.t.shape[0]:
            raise ValueError(
                f"{where}: state must be 2-D with shape (n_state, {self.t.shape[0]}); "
                f"got shape {self.state.shape}"
            )
        if not np.all(np.isfinite(self.state)):
            raise ValueError(f"{where}: state is not finite everywhere")
        for name, value in self.params.items():
            if not math.isfinite(value):
                raise ValueError(f"{where}: params[{name!r}] = {value!r} is not finite")


@dataclass(frozen=True, kw_only=True)
class TrajectorySet:

    name: str
    source: str
    time_var: str
    state_vars: tuple[str, ...]
    param_names: tuple[str, ...]
    trajectories: tuple[Trajectory, ...]
    file_source: DatasetSource | None = None

    def __post_init__(self) -> None:
        self._check_names()
        for trajectory in self.trajectories:
            where = f"trajectory {trajectory.id!r}"
            if trajectory.state.shape[0] != len(self.state_vars):
                raise ValueError(
                    f"{where}: state has shape {trajectory.state.shape}, "
                    f"but there are {len(self.state_vars)} state variables"
                )
            if set(trajectory.params) != set(self.param_names):
                raise ValueError(
                    f"{where}: params {sorted(trajectory.params)} are not "
                    f"the set's param_names {sorted(self.param_names)}"
                )

    def _check_names(self) -> None:
        if not self.trajectories:
            raise ValueError(f"set {self.name!r} needs at least one trajectory")
        ids = [trajectory.id for trajectory in self.trajectories]
        if len(set(ids)) != len(ids):
            raise ValueError(f"set {self.name!r} has duplicate trajectory ids: {ids}")
        if not self.state_vars:
            raise ValueError(f"set {self.name!r} needs at least one state variable")
        for field in ("state_vars", "param_names"):
            names = getattr(self, field)
            if len(set(names)) != len(names):
                raise ValueError(f"set {self.name!r} has duplicate {field}: {names}")
        names = (*self.state_vars, *self.param_names, self.time_var)
        if len(set(names)) != len(names):
            raise ValueError(
                f"set {self.name!r}: state, parameter and time names must be "
                f"distinct; got {names}"
            )
        for name in names:
            if is_reserved_field_name(name):
                label = "state, parameter or time name"
                raise ValueError(
                    f"set {self.name!r}: "
                    + reserved_field_name_message(name, label=label)
                )


def derivative_target(state: str) -> str:
    return f"d{state}_dt"


def _metadata(trajectories: TrajectorySet) -> dict[str, Any]:
    return {
        "name": trajectories.name,
        "source": trajectories.source,
        "time_var": trajectories.time_var,
        "state_vars": list(trajectories.state_vars),
        "param_names": list(trajectories.param_names),
        "trajectories": [
            {
                "id": trajectory.id,
                "params": {
                    name: trajectory.params[name] for name in trajectories.param_names
                },
            }
            for trajectory in trajectories.trajectories
        ],
    }


def trajectory_set_fingerprint(trajectories: TrajectorySet) -> str:
    digest = hashlib.sha256(
        json.dumps(_metadata(trajectories), sort_keys=True).encode()
    )
    for trajectory in trajectories.trajectories:
        for array in (trajectory.t, trajectory.state):
            digest.update(json.dumps([array.dtype.str, list(array.shape)]).encode())
            digest.update(array.tobytes(order="C"))
    return _FINGERPRINT_PREFIX + digest.hexdigest()


__all__ = [
    "Trajectory",
    "TrajectorySet",
    "derivative_target",
    "trajectory_set_fingerprint",
]
