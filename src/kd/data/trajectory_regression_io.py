
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from kd.data.tabular_bridge import _validate_names
from kd.data.trajectory import (
    TRAJECTORY_FINGERPRINT_DOMAIN,
    Trajectory,
    TrajectorySet,
    derivative_target,
)
from kd.data.trajectory_regression import (
    FORMAT,
    NumericArrays,
    TrajectoryNNConfig,
    TrajectoryRegression,
    _derived_seed,
    _json_bytes,
    _numeric_digest,
)
from kd.models import TrainingResult

_METADATA_KEYS = frozenset(
    {
        "source_fingerprint",
        "name",
        "source",
        "derivatives",
        "nn_config",
        "time_var",
        "state_vars",
        "param_names",
        "trajectory_ids",
        "row_boundaries",
        "derived_seeds",
    }
)
_ENVELOPE_KEYS = frozenset(
    {
        "format",
        "content_id",
        "metadata",
        "training_reports",
        "arrays_sha256",
        "projections",
    }
)


def save_trajectory_regression(
    result: TrajectoryRegression,
    directory: str | Path,
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    numeric_path = target / "arrays.npz"

    np.savez(numeric_path, **cast(dict[str, Any], result._arrays))
    envelope = {
        "format": FORMAT,
        "content_id": result.content_id,
        "metadata": result.metadata,
        "training_reports": {
            key: asdict(value) for key, value in result.training_reports.items()
        },
        "arrays_sha256": hashlib.sha256(numeric_path.read_bytes()).hexdigest(),
        "projections": list(result._projections.values()),
    }
    (target / "metadata.json").write_bytes(_json_bytes(envelope) + b"\n")
    return target


def _validate_recipe(metadata: dict[str, Any]) -> TrajectoryNNConfig | None:
    if metadata.keys() != _METADATA_KEYS:
        raise ValueError("trajectory metadata has missing or unknown fields")
    kind = metadata["derivatives"]
    if kind not in ("finite_diff", "autograd"):
        raise ValueError("metadata derivatives must be finite_diff or autograd")
    fingerprint = metadata["source_fingerprint"]
    prefix, separator, digest = fingerprint.partition(":")
    if prefix != TRAJECTORY_FINGERPRINT_DOMAIN or separator != ":" or len(digest) != 64:
        raise ValueError("invalid source_fingerprint")
    int(digest, 16)
    for state in metadata["state_vars"]:
        _validate_names(
            metadata["state_vars"] + metadata["param_names"], derivative_target(state)
        )
    if kind == "finite_diff":
        if metadata["nn_config"] is not None or metadata["derived_seeds"]:
            raise ValueError("FD metadata cannot carry an NN recipe")
        return None
    config_values = dict(metadata["nn_config"])
    config_values["hidden_sizes"] = tuple(config_values["hidden_sizes"])
    config = TrajectoryNNConfig(**config_values)
    seeds = {
        name: _derived_seed(config.seed, name) for name in metadata["trajectory_ids"]
    }
    if metadata["derived_seeds"] != seeds:
        raise ValueError("NN derived seeds do not match recipe and trajectory IDs")
    return config


def _validate_shapes(metadata: dict[str, Any], arrays: NumericArrays) -> None:
    ids, states = metadata["trajectory_ids"], metadata["state_vars"]
    bounds = metadata["row_boundaries"]
    if not ids or len(bounds) != len(ids) + 1 or bounds[0] != 0:
        raise ValueError("invalid trajectory row boundaries")
    if any(type(value) is not int for value in bounds):
        raise ValueError("trajectory row boundaries must be integers")
    if any(end - start < 3 for start, end in zip(bounds[:-1], bounds[1:], strict=True)):
        raise ValueError("each trajectory needs at least 3 points")
    shapes = {
        "time": (bounds[-1],),
        "observed": (len(states), bounds[-1]),
        "derivatives": (len(states), bounds[-1]),
        "parameters": (len(ids), len(metadata["param_names"])),
    }
    if metadata["derivatives"] == "autograd":
        shapes["surrogate"] = shapes["observed"]
    elif any(
        end - start < 5 for start, end in zip(bounds[:-1], bounds[1:], strict=True)
    ):
        raise ValueError("finite_diff trajectories need at least 5 points")
    if arrays.keys() != shapes.keys():
        raise ValueError("numeric arrays do not match the derivative recipe")
    for name, shape in shapes.items():
        if arrays[name].shape != shape or arrays[name].dtype != np.float64:
            raise ValueError(f"numeric array {name!r} has incompatible shape or dtype")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"numeric array {name!r} has nonfinite values")


def _validate_trajectory_data(metadata: dict[str, Any], arrays: NumericArrays) -> None:
    bounds = metadata["row_boundaries"]
    trajectories = tuple(
        Trajectory(
            id=trajectory_id,
            t=arrays["time"][bounds[i]: bounds[i + 1]],
            state=arrays["observed"][:, bounds[i]: bounds[i + 1]],
            params=dict(
                zip(metadata["param_names"], arrays["parameters"][i], strict=True)
            ),
        )
        for i, trajectory_id in enumerate(metadata["trajectory_ids"])
    )
    TrajectorySet(
        name=metadata["name"],
        source=metadata["source"],
        time_var=metadata["time_var"],
        state_vars=tuple(metadata["state_vars"]),
        param_names=tuple(metadata["param_names"]),
        trajectories=trajectories,
    )


def _restore_result(
    envelope: dict[str, Any], arrays: NumericArrays
) -> TrajectoryRegression:
    metadata = envelope["metadata"]
    actual_id = FORMAT + ":" + _numeric_digest(metadata, arrays)
    if actual_id != envelope["content_id"]:
        raise ValueError("trajectory content digest does not match metadata and arrays")
    config = _validate_recipe(metadata)
    _validate_shapes(metadata, arrays)
    _validate_trajectory_data(metadata, arrays)
    reports = {
        name: TrainingResult(**values)
        for name, values in envelope["training_reports"].items()
    }
    expected_reports = set() if config is None else set(metadata["trajectory_ids"])
    if reports.keys() != expected_reports:
        raise ValueError("training reports do not match trajectory recipe")
    result = TrajectoryRegression(
        metadata=metadata, arrays=arrays, training_reports=reports
    )
    for projection in envelope["projections"]:
        if projection.keys() != {"id", "target_state", "state_source", "trim"}:
            raise ValueError("invalid saved projection metadata")
        table = result.table(
            projection["target_state"],
            state_source=projection["state_source"],
            trim=projection["trim"],
        )
        if table.name != projection["id"]:
            raise ValueError("projection content digest mismatch")
    return result


def load_trajectory_regression(directory: str | Path) -> TrajectoryRegression:
    target = Path(directory)
    envelope = json.loads((target / "metadata.json").read_text())
    if not isinstance(envelope, dict) or envelope.keys() != _ENVELOPE_KEYS:
        raise ValueError("invalid trajectory-regression metadata envelope")
    if envelope["format"] != FORMAT:
        raise ValueError(
            f"unsupported trajectory-regression format {envelope['format']!r}"
        )
    numeric_path = target / "arrays.npz"
    if (
        hashlib.sha256(numeric_path.read_bytes()).hexdigest()
        != envelope["arrays_sha256"]
    ):
        raise ValueError("arrays.npz SHA256 checksum mismatch")
    with np.load(numeric_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    try:
        return _restore_result(envelope, arrays)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"invalid trajectory-regression metadata: {exc}") from exc
