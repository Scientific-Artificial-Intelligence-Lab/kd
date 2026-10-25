
from __future__ import annotations

import hashlib
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kd.data.schema import PDEDataset, compute_dataset_fingerprint
from kd.models._training_precision import omit_default_training_dtype
from kd.models.field_model import FieldModel
from kd.models.serialize import (
    StoredSurrogate,
    load_stored_surrogate,
    save_field_model,
)
from kd.models.trainer import TrainingResult
from kd.search.dlga.config import DLGAConfig


@dataclass(frozen=True)
class TrainedSurrogate:

    module: FieldModel
    config: DLGAConfig
    stopped_at_deadline: bool
    training: TrainingResult | None


def dlga_surrogate_identity(dataset: PDEDataset, config: DLGAConfig) -> dict[str, Any]:



    from kd.core.platform.builder import resolve_lhs_defaults
    from kd.search.dlga.plugin import DLGAPlugin

    resolved = resolve_lhs_defaults(dataset)
    surrogate = omit_default_training_dtype(
        {
            name: _plain(getattr(config, name))
            for name in sorted(DLGAPlugin.descriptor.surrogate_fields)
        }
    )
    return {
        "algorithm": "dlga",
        "dataset": compute_dataset_fingerprint(resolved),
        "target_sha256": _target_sha256(resolved),
        "surrogate": surrogate,
        "seed": int(config.seed),
    }


def _target_sha256(resolved: PDEDataset) -> str:
    target = resolved.get_field(resolved.lhs_field).detach().cpu().contiguous()
    return hashlib.sha256(target.numpy().tobytes()).hexdigest()


def _plain(value: object) -> object:
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value


def save_dlga_surrogate(
    module: FieldModel,
    path: str | Path,
    *,
    dataset: PDEDataset,
    config: DLGAConfig,
    stopped_at_deadline: bool = False,
) -> Path:
    identity = dlga_surrogate_identity(dataset, config)
    if stopped_at_deadline:
        identity["stopped_at_deadline"] = True
    return save_field_model(module, path, identity=identity)


def load_dlga_surrogate(
    path: str | Path,
    *,
    dataset: PDEDataset,
    config: DLGAConfig,
    device: str | torch.device | None = None,
) -> FieldModel:
    stored = load_stored_surrogate(path, device=device)
    require_surrogate_identity(stored, dataset, config)
    return stored.module


def require_surrogate_identity(
    stored: StoredSurrogate,
    dataset: PDEDataset,
    config: DLGAConfig,
    *,
    ignore: Collection[str] = (),
) -> None:
    if stored.identity is None:
        raise ValueError(
            f"stored surrogate {stored.path} carries no identity, so it cannot be "
            "verified against this dataset and configuration; write it with "
            "save_dlga_surrogate (or Model.train_surrogate(save_to=)), or inject "
            "its module directly to use it unverified"
        )
    expected = dlga_surrogate_identity(dataset, config)
    differing = [
        key for key in _differing_keys(stored.identity, expected) if key not in ignore
    ]
    if not differing:
        return
    raise ValueError(
        f"stored surrogate {stored.path} was trained on another dataset or "
        f"surrogate configuration; differing identity keys: {differing}"
    )


def _differing_keys(stored: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in sorted(stored.keys() | expected.keys()):
        if key == "surrogate":
            inner_stored = stored.get(key) or {}
            inner_expected = expected.get(key) or {}
            keys.extend(
                f"surrogate.{name}"
                for name in sorted(inner_stored.keys() | inner_expected.keys())
                if inner_stored.get(name) != inner_expected.get(name)
            )
        elif stored.get(key) != expected.get(key):
            keys.append(key)
    return keys


__all__ = [
    "TrainedSurrogate",
    "dlga_surrogate_identity",
    "load_dlga_surrogate",
    "require_surrogate_identity",
    "save_dlga_surrogate",
]
