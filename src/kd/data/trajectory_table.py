
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from kd.data.derivatives.finite_diff import coordinate_diff
from kd.data.regression import TabularDataset
from kd.data.trajectory import Trajectory, TrajectorySet, derivative_target


def _trim_slice(n_points: int, trim: int, trajectory_id: str) -> slice:
    if n_points <= 2 * trim:
        raise ValueError(
            f"trajectory {trajectory_id!r} has {n_points} points; trimming "
            f"{trim} per end needs more than {2 * trim}"
        )
    return slice(trim, n_points - trim)


def _differentiate_state(t: np.ndarray, state: np.ndarray) -> np.ndarray:


    return coordinate_diff(
        torch.as_tensor(state, dtype=torch.float64),
        torch.as_tensor(t, dtype=torch.float64),
        axis=0,
        order=1,
    ).numpy()


def _regression_rows(
    state: np.ndarray,
    derivative: np.ndarray,
    parameters: Sequence[float] | np.ndarray,
    keep: slice,
) -> tuple[np.ndarray, np.ndarray]:
    constants = np.tile(np.asarray(parameters, dtype=np.float64), (state.shape[1], 1))
    return np.hstack([state.T, constants])[keep], derivative[keep]


def _rows(
    trajectory: Trajectory, index: int, param_names: tuple[str, ...], trim: int
) -> tuple[np.ndarray, np.ndarray]:
    keep = _trim_slice(len(trajectory.t), trim, trajectory.id)
    derivative = _differentiate_state(trajectory.t, trajectory.state[index])
    parameters = [trajectory.params[name] for name in param_names]
    return _regression_rows(trajectory.state, derivative, parameters, keep)


def trajectories_to_tabular(
    trajectories: TrajectorySet, target_state: str, *, trim: int = 0
) -> TabularDataset:
    if target_state not in trajectories.state_vars:
        raise ValueError(
            f"{target_state!r} is not a state of {trajectories.state_vars}"
        )
    if trim < 0:
        raise ValueError(f"trim must be >= 0; got {trim}")
    index = trajectories.state_vars.index(target_state)
    rows = [
        _rows(trajectory, index, trajectories.param_names, trim)
        for trajectory in trajectories.trajectories
    ]
    target = derivative_target(target_state)
    return TabularDataset(
        X=np.concatenate([features for features, _ in rows], axis=0),
        y=np.concatenate([derivative for _, derivative in rows]),
        var_names=(*trajectories.state_vars, *trajectories.param_names),
        target_name=target,
        name=trajectories.name,
        source=trajectories.source,
        description=(
            f"One row per time point of each of {len(trajectories.trajectories)} "
            f"trajectories, {trim} point(s) dropped at each end; the columns are "
            f"the states and the per-trajectory parameter values. The target "
            f"{target} is the time derivative of {target_state}, estimated with "
            "kd's finite differences over each trajectory's time array."
        ),
        file_source=trajectories.file_source,
    )


__all__ = ["trajectories_to_tabular"]
