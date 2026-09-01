
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import torch
from torch import nn

from kd.models.field_model import FieldModel, Rational, _FuncModule

SURROGATE_SCHEME: Final[str] = "kd-surrogate-v1"

_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "scheme",
        "coord_names",
        "field_names",
        "hidden_sizes",
        "activation",
        "dtype",
        "state_dict",
    }
)


_DTYPES: Final[dict[str, torch.dtype]] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

_ACTIVATION_NAMES: Final[dict[type[nn.Module], str]] = {
    nn.Tanh: "tanh",
    nn.ReLU: "relu",
    Rational: "rational",
}


def _hidden_sizes(model: FieldModel) -> list[int]:
    return [
        layer.out_features for layer in model.trunk if isinstance(layer, nn.Linear)
    ]


def _activation_name(model: FieldModel) -> str:
    for layer in model.trunk:
        if isinstance(layer, nn.Linear):
            continue
        if isinstance(layer, _FuncModule):
            if layer._func is torch.sin:
                return "sin"
            raise ValueError(
                f"cannot serialize FieldModel activation {layer._func!r}"
            )
        name = _ACTIVATION_NAMES.get(type(layer))
        if name is not None:
            return name
        raise ValueError(
            f"cannot serialize FieldModel activation {type(layer).__name__}"
        )
    raise ValueError("FieldModel trunk carries no activation module")


def save_field_model(model: FieldModel, path: str | Path) -> Path:
    dtype = next(model.parameters()).dtype
    payload = {
        "scheme": SURROGATE_SCHEME,
        "coord_names": list(model.coord_names),
        "field_names": list(model.field_names),
        "hidden_sizes": _hidden_sizes(model),
        "activation": _activation_name(model),
        "dtype": str(dtype).removeprefix("torch."),
        "state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
    }
    target = Path(path)
    tmp_path = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target)
    return target


def load_field_model(path: str | Path) -> FieldModel:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(
            f"{SURROGATE_SCHEME} file must hold a mapping, got "
            f"{type(payload).__name__}"
        )
    missing = sorted(_PAYLOAD_KEYS - payload.keys())
    if missing:
        raise ValueError(f"{SURROGATE_SCHEME} payload is missing keys: {missing}")
    extra = sorted(payload.keys() - _PAYLOAD_KEYS)
    if extra:
        raise ValueError(f"{SURROGATE_SCHEME} payload has unexpected keys: {extra}")
    scheme = payload["scheme"]
    if scheme != SURROGATE_SCHEME:
        raise ValueError(
            f"unsupported surrogate scheme {scheme!r}; expected {SURROGATE_SCHEME!r}"
        )
    dtype = _DTYPES.get(payload["dtype"])
    if dtype is None:
        raise ValueError(
            f"{SURROGATE_SCHEME} payload names unsupported dtype "
            f"{payload['dtype']!r}; expected one of {sorted(_DTYPES)}"
        )
    model = FieldModel(
        coord_names=list(payload["coord_names"]),
        field_names=list(payload["field_names"]),
        hidden_sizes=list(payload["hidden_sizes"]),
        activation=payload["activation"],
    ).to(dtype=dtype)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"{SURROGATE_SCHEME} state_dict does not fit the declared "
            f"architecture: {exc}"
        ) from exc
    model.eval()
    return model


__all__ = ["SURROGATE_SCHEME", "load_field_model", "save_field_model"]
