
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from kd.models.field_model import FieldModel






V1_HIDDEN_LAYERS: int = 6
V1_NEURONS: int = 60
V1_INPUT_DIM: int = 2
V1_OUTPUT_DIM: int = 1
V1_ACTIVATION: str = "sin"




V1_COORD_NAMES: tuple[str, ...] = ("t", "x")
V1_FIELD_NAMES: tuple[str, ...] = ("u",)


def load_v1_field_model(checkpoint_path: str | Path) -> FieldModel:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"v1 wave-breaking checkpoint not found: {path}")

    loaded: object = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise ValueError(
            f"checkpoint must contain a tensor state-dict, got "
            f"{type(loaded).__name__}."
        )
    state = _validate_v1_state_dict(loaded)

    model = FieldModel(
        coord_names=list(V1_COORD_NAMES),
        field_names=list(V1_FIELD_NAMES),
        hidden_sizes=[V1_NEURONS] * V1_HIDDEN_LAYERS,
        activation=V1_ACTIVATION,
    )
    model.load_state_dict(_map_v1_state_to_field_model(state, model), strict=True)
    model.eval()
    return model


def _expected_v1_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "Layers.0.weight": (V1_NEURONS, V1_INPUT_DIM),
        "Layers.0.bias": (V1_NEURONS,),
    }
    for layer_idx in range(1, V1_HIDDEN_LAYERS):
        shapes[f"Layers.{layer_idx}.weight"] = (V1_NEURONS, V1_NEURONS)
        shapes[f"Layers.{layer_idx}.bias"] = (V1_NEURONS,)
    head_idx = V1_HIDDEN_LAYERS
    shapes[f"Layers.{head_idx}.weight"] = (V1_OUTPUT_DIM, V1_NEURONS)
    shapes[f"Layers.{head_idx}.bias"] = (V1_OUTPUT_DIM,)
    return shapes


def _validate_v1_state_dict(raw: Mapping[object, object]) -> dict[str, Tensor]:
    tensor_state: dict[str, Tensor] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(
                f"checkpoint state-dict keys must be strings, got "
                f"{type(key).__name__}."
            )
        if not isinstance(value, Tensor):
            raise ValueError(
                f"checkpoint key '{key}' must contain a tensor, got "
                f"{type(value).__name__}."
            )
        tensor_state[key] = value

    expected = _expected_v1_shapes()
    missing = sorted(set(expected) - set(tensor_state))
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {missing}.")
    extra = sorted(set(tensor_state) - set(expected))
    if extra:
        raise ValueError(f"checkpoint has unexpected keys: {extra}.")

    for key, expected_shape in expected.items():
        actual_shape = tuple(tensor_state[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"checkpoint key '{key}' has shape {actual_shape}, "
                f"expected {expected_shape}."
            )
    return tensor_state


def _map_v1_state_to_field_model(
    state: Mapping[str, Tensor],
    model: FieldModel,
) -> dict[str, Tensor]:
    mapped = dict(model.state_dict())
    for layer_idx in range(V1_HIDDEN_LAYERS):
        trunk_idx = layer_idx * 2
        mapped[f"trunk.{trunk_idx}.weight"] = state[f"Layers.{layer_idx}.weight"]
        mapped[f"trunk.{trunk_idx}.bias"] = state[f"Layers.{layer_idx}.bias"]
    head_idx = V1_HIDDEN_LAYERS
    mapped["head.weight"] = state[f"Layers.{head_idx}.weight"]
    mapped["head.bias"] = state[f"Layers.{head_idx}.bias"]
    return mapped


__all__ = ["load_v1_field_model"]
