"""MetaNN training helpers for N-dim structured grids."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import Net


def _as_1d_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError("Expected 1D coordinate arrays.")
    return arr


def _build_coord_grids(
    coords_1d: Dict[str, np.ndarray],
    axis_order: Sequence[str],
    grid_shape: Tuple[int, ...],
) -> Dict[str, np.ndarray]:
    coord_grids: Dict[str, np.ndarray] = {}
    axis_map = {axis: idx for idx, axis in enumerate(axis_order)}
    for axis, index in axis_map.items():
        coord = coords_1d[axis]
        reshape = [1] * len(grid_shape)
        reshape[index] = coord.shape[0]
        coord_grids[axis] = np.broadcast_to(coord.reshape(reshape), grid_shape)
    return coord_grids


def prepare_training_data(
    coords_1d: Dict[str, Iterable[float]],
    field: np.ndarray,
    *,
    axis_order: Optional[Sequence[str]] = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Flatten N-dim structured data into (X, y) for MetaNN."""
    coords = {axis: _as_1d_array(values) for axis, values in coords_1d.items()}
    axis_order_list = list(axis_order) if axis_order else list(coords.keys())
    if len(axis_order_list) != len(set(axis_order_list)):
        raise ValueError("axis_order entries must be unique.")
    if set(axis_order_list) != set(coords.keys()):
        raise ValueError("axis_order must match coords_1d keys.")

    grid_shape = tuple(coords[axis].shape[0] for axis in axis_order_list)
    field_arr = np.asarray(field, dtype=float)
    if field_arr.shape != grid_shape:
        raise ValueError(
            f"Field shape {field_arr.shape} does not match grid shape {grid_shape}."
        )

    coord_grids = _build_coord_grids(coords, axis_order_list, grid_shape)

    # NOTE(ND): Use C-order flattening for consistent (X, y) alignment.
    X = np.stack(
        [coord_grids[axis].reshape(-1) for axis in axis_order_list],
        axis=1,
    )
    y = field_arr.reshape(-1, 1)

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std_safe = np.where(x_std == 0.0, 1.0, x_std)

    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std_safe = np.where(y_std == 0.0, 1.0, y_std)

    if normalize:
        X = (X - x_mean) / x_std_safe
        y = (y - y_mean) / y_std_safe

    stats = {
        "axis_order": np.array(axis_order_list),
        "x_mean": x_mean,
        "x_std": x_std_safe,
        "y_mean": y_mean,
        "y_std": y_std_safe,
        "normalized": np.array([normalize]),
    }
    return X, y, stats


def train_metanet(
    coords_1d: Dict[str, Iterable[float]],
    field: np.ndarray,
    *,
    axis_order: Optional[Sequence[str]] = None,
    hidden_dim: int = 50,
    max_epoch: int = 1000,
    train_ratio: float = 1.0,
    normalize: bool = True,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[Net, Dict[str, np.ndarray]]:
    """Train a MetaNN model on N-dim structured data."""
    if device is None:
        device = torch.device("cpu")

    # NOTE: Training consumes RNG for weight init / data split. Do not leak RNG state
    # to the caller (important for GA reproducibility).
    np_rng = np.random.RandomState(seed)

    X, y, stats = prepare_training_data(
        coords_1d,
        field,
        axis_order=axis_order,
        normalize=normalize,
    )

    cuda_devices: List[int] = []
    if torch.cuda.is_available():
        cuda_devices = list(range(torch.cuda.device_count()))

    fork_rng = getattr(torch.random, "fork_rng", None)
    if fork_rng is None:
        cpu_state = torch.get_rng_state()
        cuda_state = None
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
        try:
            torch.manual_seed(seed)
            num_feature = X.shape[1]
            model = Net(num_feature, hidden_dim, 1).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters())

            total = X.shape[0]
            if train_ratio <= 0 or train_ratio > 1:
                raise ValueError("train_ratio must be within (0, 1].")
            split = int(total * train_ratio)
            indices = np.arange(total)
            np_rng.shuffle(indices)
            train_idx = indices[:split]

            x_train = torch.from_numpy(X[train_idx]).float().to(device)
            y_train = torch.from_numpy(y[train_idx]).float().to(device)

            for _ in range(max_epoch):
                optimizer.zero_grad()
                pred = model(x_train)
                loss = criterion(pred, y_train)
                loss.backward()
                optimizer.step()
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        return model, stats

    with fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)

        num_feature = X.shape[1]
        model = Net(num_feature, hidden_dim, 1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        total = X.shape[0]
        if train_ratio <= 0 or train_ratio > 1:
            raise ValueError("train_ratio must be within (0, 1].")
        split = int(total * train_ratio)
        indices = np.arange(total)
        np_rng.shuffle(indices)
        train_idx = indices[:split]

        x_train = torch.from_numpy(X[train_idx]).float().to(device)
        y_train = torch.from_numpy(y[train_idx]).float().to(device)

        for _ in range(max_epoch):
            optimizer.zero_grad()
            pred = model(x_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()

        return model, stats


__all__ = [
    "prepare_training_data",
    "train_metanet",
]
