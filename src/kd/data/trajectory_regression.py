
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import accumulate
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from kd.data.derivatives.autograd import AutogradProvider
from kd.data.regression import TabularDataset
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.data.tabular_bridge import _validate_names
from kd.data.trajectory import (
    Trajectory,
    TrajectorySet,
    derivative_target,
    trajectory_set_fingerprint,
)
from kd.data.trajectory_table import (
    _differentiate_state,
    _regression_rows,
    _trim_slice,
)
from kd.models import FieldModel, FieldModelTrainer, TrainingResult

FORMAT = "kd-trajectory-regression-v1"
_PROJECTION_PREFIX = "ode-regression-v1:"
NumericArrays = dict[str, NDArray[np.float64]]


def _positive_integer(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


@dataclass(frozen=True, kw_only=True)
class TrajectoryNNConfig:

    seed: int = 0
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    lr: float = 1e-3
    max_epochs: int = 1000
    patience: int | None = None
    val_ratio: float = 0.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        _positive_integer(self.max_epochs, "max_epochs")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        for size in self.hidden_sizes:
            _positive_integer(size, "hidden_sizes")
        object.__setattr__(self, "lr", float(self.lr))
        object.__setattr__(self, "val_ratio", float(self.val_ratio))
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.val_ratio) or not 0 <= self.val_ratio < 1:
            raise ValueError("val_ratio must be finite and in [0, 1)")
        if self.patience is not None:
            _positive_integer(self.patience, "patience")
            if self.val_ratio == 0:
                raise ValueError("patience requires a positive val_ratio")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _numeric_digest(metadata: dict[str, Any], arrays: NumericArrays) -> str:
    digest = hashlib.sha256()
    for part in [FORMAT.encode(), _json_bytes(metadata)]:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    for key, array in sorted(arrays.items()):
        description = _json_bytes([key, array.dtype.str, list(array.shape)])
        digest.update(len(description).to_bytes(8, "big"))
        digest.update(description)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _derived_seed(base: int, trajectory_id: str) -> int:
    payload = json.dumps(
        [base, trajectory_id], separators=(",", ":"), ensure_ascii=False
    )
    return (
        int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
        % 2147483647
    )


class TrajectoryRegression:

    def __init__(
        self,
        *,
        metadata: dict[str, Any],
        arrays: NumericArrays,
        training_reports: dict[str, TrainingResult],
    ) -> None:
        self._metadata = json.loads(_json_bytes(metadata))
        self._arrays = {
            name: np.array(value, dtype=np.float64, order="C", copy=True)
            for name, value in arrays.items()
        }
        for value in self._arrays.values():
            if not np.isfinite(value).all():
                raise ValueError("trajectory preprocessing produced nonfinite values")
            value.setflags(write=False)
        self._training_reports = copy.deepcopy(training_reports)
        self._content_id = FORMAT + ":" + _numeric_digest(self._metadata, self._arrays)
        self._projections: dict[str, dict[str, Any]] = {}

    @property
    def content_id(self) -> str:
        return self._content_id

    @property
    def metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._metadata)

    @property
    def training_reports(self) -> dict[str, TrainingResult]:
        return copy.deepcopy(self._training_reports)

    def table(
        self,
        target_state: str,
        *,
        state_source: Literal["observed", "surrogate"] = "observed",
        trim: int = 2,
    ) -> TabularDataset:
        self._validate_projection(target_state, state_source, trim)
        meta = self._metadata
        index = meta["state_vars"].index(target_state)
        rows = []
        bounds = meta["row_boundaries"]
        for i, trajectory_id in enumerate(meta["trajectory_ids"]):
            start, end = bounds[i: i + 2]
            keep = _trim_slice(end - start, trim, trajectory_id)
            rows.append(
                _regression_rows(
                    self._arrays[state_source][:, start:end],
                    self._arrays["derivatives"][index, start:end],
                    self._arrays["parameters"][i],
                    keep,
                )
            )
        x = np.concatenate([row[0] for row in rows])
        y = np.concatenate([row[1] for row in rows])
        name = self._projection_name(target_state, state_source, trim, x, y)
        return TabularDataset(
            X=x,
            y=y,
            var_names=tuple(meta["state_vars"] + meta["param_names"]),
            target_name=derivative_target(target_state),
            name=name,
            source=meta["source"],
            description=(
                f"{meta['name']}: {meta['derivatives']} derivatives; "
                f"{state_source} RHS states; trim={trim} per trajectory end."
            ),
        )

    def _projection_name(
        self,
        target_state: str,
        state_source: str,
        trim: int,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> str:
        projection = {
            "content_id": self.content_id,
            "target_state": target_state,
            "state_source": state_source,
            "trim": trim,
        }
        name = _PROJECTION_PREFIX + _numeric_digest(projection, {"X": x, "y": y})
        self._projections[name] = {
            "id": name,
            "target_state": target_state,
            "state_source": state_source,
            "trim": trim,
        }
        return name

    def _validate_projection(self, target: str, source: str, trim: int) -> None:
        if target not in self._metadata["state_vars"]:
            raise ValueError(
                f"{target!r} is not a state of {self._metadata['state_vars']}"
            )
        if source not in ("observed", "surrogate"):
            raise ValueError(f"unknown state_source {source!r}")
        if source == "surrogate" and self._metadata["derivatives"] != "autograd":
            raise ValueError("surrogate states require autograd preprocessing")
        if type(trim) is not int or trim < 0:
            raise ValueError("trim must be a nonnegative integer")


def _train_trajectory(
    trajectory: Trajectory,
    time_var: str,
    states: tuple[str, ...],
    config: TrajectoryNNConfig,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], TrainingResult]:
    device = torch.device(config.device)
    time = torch.tensor(trajectory.t.copy(), dtype=torch.float64, device=device)
    targets = {
        name: torch.tensor(
            trajectory.state[i].copy(), dtype=torch.float64, device=device
        )
        for i, name in enumerate(states)
    }

    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = FieldModel(
            coord_names=[time_var],
            field_names=list(states),
            hidden_sizes=list(config.hidden_sizes),
            activation=config.activation,
        ).to(device=device, dtype=torch.float64)
    trainer = FieldModelTrainer(model, lr=config.lr, weight_decay=0.0)
    report = trainer.fit(
        coords={time_var: time},
        targets=targets,
        max_epochs=config.max_epochs,
        patience=config.patience,
        val_ratio=config.val_ratio,
        seed=seed,
        restore_best=False,
        device=device,
    )
    model.eval()
    surrogate, derivatives = _materialize_model(
        model, trajectory.id, time_var, time, targets
    )
    return surrogate, derivatives, report


def _materialize_model(
    model: FieldModel,
    trajectory_id: str,
    time_var: str,
    time: torch.Tensor,
    targets: dict[str, torch.Tensor],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    states = tuple(targets)
    dataset = PDEDataset(
        name=trajectory_id,
        task_type=TaskType.ODE,
        topology=DataTopology.SCATTERED,
        axes={time_var: AxisInfo(time_var, time)},
        axis_order=[time_var],
        fields={name: FieldData(name, value) for name, value in targets.items()},
        lhs_field=states[0],
        lhs_axis=time_var,
        lhs_order=1,
    )
    coords = {time_var: time.detach().clone().requires_grad_(True)}
    provider = AutogradProvider(model, coords, dataset, max_order=1)
    surrogate = np.stack(
        [provider.get_field(name).detach().cpu().numpy().copy() for name in states]
    )
    derivatives = np.stack(
        [
            provider.get_derivative(name, time_var, 1).detach().cpu().numpy().copy()
            for name in states
        ]
    )
    return surrogate, derivatives


def _prepare_arrays(
    data: TrajectorySet,
    config: TrajectoryNNConfig | None,
) -> tuple[NumericArrays, dict[str, TrainingResult], dict[str, int]]:
    derivatives, surrogates = [], []
    reports, seeds = {}, {}
    for trajectory in data.trajectories:
        if config is None:
            derivatives.append(
                np.stack(
                    [
                        _differentiate_state(trajectory.t, state)
                        for state in trajectory.state
                    ]
                )
            )
        else:
            seed = _derived_seed(config.seed, trajectory.id)
            surrogate, derivative, report = _train_trajectory(
                trajectory, data.time_var, data.state_vars, config, seed
            )
            derivatives.append(derivative)
            surrogates.append(surrogate)
            reports[trajectory.id], seeds[trajectory.id] = report, seed
    arrays = {
        "time": np.concatenate([t.t for t in data.trajectories]),
        "observed": np.concatenate([t.state for t in data.trajectories], axis=1),
        "derivatives": np.concatenate(derivatives, axis=1),
        "parameters": np.array(
            [[t.params[name] for name in data.param_names] for t in data.trajectories],
            dtype=np.float64,
        ),
    }
    if config is not None:
        arrays["surrogate"] = np.concatenate(surrogates, axis=1)
    return arrays, reports, seeds


def prepare_trajectory_regression(
    trajectories: TrajectorySet,
    *,
    derivatives: Literal["finite_diff", "autograd"] = "finite_diff",
    nn_config: TrajectoryNNConfig | None = None,
) -> TrajectoryRegression:
    if derivatives not in ("finite_diff", "autograd"):
        raise ValueError(f"unknown derivatives {derivatives!r}")
    if derivatives == "finite_diff" and nn_config is not None:
        raise ValueError("nn_config is only valid with derivatives='autograd'")
    for state in trajectories.state_vars:
        _validate_names(
            [*trajectories.state_vars, *trajectories.param_names],
            derivative_target(state),
        )
    config = (
        (nn_config if nn_config is not None else TrajectoryNNConfig())
        if derivatives == "autograd"
        else None
    )
    fingerprint = trajectory_set_fingerprint(trajectories)
    arrays, reports, seeds = _prepare_arrays(trajectories, config)
    metadata = {
        "source_fingerprint": fingerprint,
        "name": trajectories.name,
        "source": trajectories.source,
        "derivatives": derivatives,
        "nn_config": None if config is None else asdict(config),
        "time_var": trajectories.time_var,
        "state_vars": list(trajectories.state_vars),
        "param_names": list(trajectories.param_names),
        "trajectory_ids": [t.id for t in trajectories.trajectories],
        "row_boundaries": [0, *accumulate(len(t.t) for t in trajectories.trajectories)],
        "derived_seeds": seeds,
    }
    return TrajectoryRegression(
        metadata=metadata, arrays=arrays, training_reports=reports
    )
