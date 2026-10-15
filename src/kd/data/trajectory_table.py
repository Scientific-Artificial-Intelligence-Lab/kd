
from __future__ import annotations

import numpy as np
import torch

from kd.data.derivatives.finite_diff import coordinate_diff
from kd.data.regression import TabularDataset
from kd.data.trajectory import Trajectory, TrajectorySet, derivative_target


def _rows(
    trajectory: Trajectory, index: int, param_names: tuple[str, ...], trim: int
) -> tuple[np.ndarray, np.ndarray]:
    n_points = trajectory.t.shape[0]
    if n_points <= 2 * trim:
        raise ValueError(
            f"trajectory {trajectory.id!r} has {n_points} points; trimming "
            f"{trim} per end needs more than {2 * trim}"
        )
    keep = slice(trim, n_points - trim)


    derivative = coordinate_diff(
        torch.as_tensor(trajectory.state[index], dtype=torch.float64),
        torch.as_tensor(trajectory.t, dtype=torch.float64),
        axis=0,
        order=1,
    ).numpy()
    values = [trajectory.params[name] for name in param_names]
    constants = np.tile(np.asarray(values, dtype=np.float64), (n_points, 1))
    return np.hstack([trajectory.state.T, constants])[keep], derivative[keep]


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
    )


__all__ = ["trajectories_to_tabular"]
