
from __future__ import annotations

import numpy as np

from kd.data.containers import Inventory, read_real
from kd.data.tabular_bridge import _validate_names
from kd.data.trajectory import Trajectory, TrajectorySet, derivative_target


def load_trajectory_set(
    inventory: Inventory,
    *,
    coords: dict[str, str],
    fields: dict[str, str],
    trajectory_id: str | None,
    params: dict[str, str] | None,
    name: str,
) -> TrajectorySet:
    if len(coords) != 1 or not fields:
        raise ValueError(
            "trajectory requires one time coordinate and non-empty fields; "
            f"got coords={coords}, fields={fields}"
        )
    parameters = {} if params is None else params
    for state in fields:
        _validate_names([*fields, *parameters], derivative_target(state))
    ((time_var, time_key),) = coords.items()
    container = inventory.container
    t = read_real(inventory.entry(time_key), container)
    states = {
        key: read_real(inventory.entry(key), container) for key in fields.values()
    }
    values = {
        label: read_real(inventory.entry(key), container)
        for label, key in parameters.items()
    }
    ids = None if trajectory_id is None else _ids(inventory, trajectory_id)
    if t.ndim == 1 and all(state.ndim == 1 for state in states.values()):
        trajectories = _long_table(time_key, t, states, ids, values)
    else:
        trajectories = _matrices(time_key, t, states, ids, values)
    return TrajectorySet(
        name=name,
        source="",
        time_var=time_var,
        state_vars=tuple(fields),
        param_names=tuple(parameters),
        trajectories=tuple(trajectories),
    )


def _ids(inventory: Inventory, key: str) -> list[str]:
    entry = inventory.entry(key)
    if np.dtype(entry.dtype).kind in "US":
        text = np.asarray(entry.read())
        if text.dtype.kind == "S":
            text = np.char.decode(text, "utf-8")
        if text.ndim != 1:
            raise ValueError(
                f"trajectory id array {key!r} must be a vector; got shape {text.shape}"
            )
        ids = [str(value) for value in text]
        for row, value in enumerate(ids):
            if not value:
                raise ValueError(f"trajectory id {key!r} is missing at row {row}")
        return ids
    numbers = read_real(entry, inventory.container)
    if numbers.ndim != 1:
        raise ValueError(
            f"trajectory id array {key!r} must be a vector; got shape {numbers.shape}"
        )
    for row, value in enumerate(numbers):
        if np.isnan(value):
            raise ValueError(f"trajectory id {key!r} is missing at row {row}")
        if not np.isfinite(value) or value != np.round(value):
            raise ValueError(
                f"trajectory id {value!r} in {key!r} (row {row}) is not an "
                "integer; numeric ids must be integers"
            )
    return [str(int(value)) for value in numbers]


def _shape_error(time_key: str, t: np.ndarray, states: dict[str, np.ndarray]) -> str:
    shapes = ", ".join(f"{key!r} {state.shape}" for key, state in states.items())
    return (
        f"trajectory arrays disagree in shape: time {time_key!r} has shape "
        f"{t.shape}; states {shapes}. Expected time (n,) with states (n,) or "
        "(k, n), or time and states (k, n)"
    )


def _per_trajectory(
    label: str, values: np.ndarray, count: int, what: str
) -> np.ndarray | float:
    if values.size == 1:
        return float(values.reshape(-1)[0])
    if values.shape != (count,):
        raise ValueError(
            f"parameter {label!r} has shape {values.shape}; expected a scalar "
            f"or one value per {what} ({count})"
        )
    return values


def _long_table(
    time_key: str,
    t: np.ndarray,
    states: dict[str, np.ndarray],
    ids: list[str] | None,
    params: dict[str, np.ndarray],
) -> list[Trajectory]:
    n_rows = t.shape[0]
    if any(state.shape != (n_rows,) for state in states.values()):
        raise ValueError(_shape_error(time_key, t, states))
    if ids is not None and len(ids) != n_rows:
        raise ValueError(
            f"trajectory id column has shape ({len(ids)},); the time column has "
            f"shape {t.shape}"
        )
    columns = {
        label: _per_trajectory(label, values, n_rows, "row")
        for label, values in params.items()
    }
    rows_by_id: dict[str, list[int]] = {}
    for row, trajectory in enumerate(["00"] * n_rows if ids is None else ids):
        rows_by_id.setdefault(trajectory, []).append(row)
    trajectories = []
    for trajectory, rows in rows_by_id.items():
        index = np.asarray(rows)
        trajectories.append(
            Trajectory(
                id=trajectory,
                t=t[index],
                state=np.vstack([state[index] for state in states.values()]),
                params={
                    label: _constant(label, values, index, trajectory)
                    for label, values in columns.items()
                },
            )
        )
    return trajectories


def _constant(
    label: str, values: np.ndarray | float, index: np.ndarray, trajectory: str
) -> float:
    if isinstance(values, float):
        return values
    selected = values[index]
    if not np.array_equal(
        selected, np.full_like(selected, selected[0]), equal_nan=True
    ):
        raise ValueError(f"parameter {label!r} varies within trajectory {trajectory!r}")
    return float(selected[0])


def _matrices(
    time_key: str,
    t: np.ndarray,
    states: dict[str, np.ndarray],
    ids: list[str] | None,
    params: dict[str, np.ndarray],
) -> list[Trajectory]:
    shapes = {state.shape for state in states.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError(_shape_error(time_key, t, states))
    ((count, n_points),) = shapes
    if t.shape not in {(n_points,), (count, n_points)}:
        raise ValueError(_shape_error(time_key, t, states))
    if ids is not None and len(ids) != count:
        raise ValueError(
            f"trajectory id array has shape ({len(ids)},); the states have "
            f"{count} trajectories (shape {(count, n_points)})"
        )
    names = [f"{row:02d}" for row in range(count)] if ids is None else ids
    columns = {
        label: _per_trajectory(label, values, count, "trajectory")
        for label, values in params.items()
    }
    return [
        Trajectory(
            id=names[row],
            t=t if t.ndim == 1 else t[row],
            state=np.vstack([state[row] for state in states.values()]),
            params={
                label: values if isinstance(values, float) else float(values[row])
                for label, values in columns.items()
            },
        )
        for row in range(count)
    ]


__all__ = ["load_trajectory_set"]
