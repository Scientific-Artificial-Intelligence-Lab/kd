
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from kd.search.resume_policy import CONFIG_ARTIFACT_KEYS, config_artifact_overlay
from kd.search.run_spec import (
    CONFIG_CANON_SCHEME,
    ConfigCanonicalizationError,
    canonicalize_config,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CHECKPOINT_VERSION",
    "atomic_torch_save",
    "build_checkpoint_payload",
]

CHECKPOINT_VERSION = 1
_CHECKPOINT_PATTERN = "checkpoint_{iteration:06d}.pt"
_CHECKPOINT_FINAL = "checkpoint_final.pt"



_CHECKPOINT_TMP_SUFFIX = ".tmp"


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(path.name + _CHECKPOINT_TMP_SUFFIX)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to clean up staging file %s", tmp_path)
        raise


def build_checkpoint_payload(iteration: int, algorithm: Any) -> dict[str, Any]:
    config_snapshot = _config_snapshot(algorithm)
    return {
        "version": CHECKPOINT_VERSION,
        "iteration": iteration,
        "algorithm_state": algorithm.state,
        "best_score": algorithm.best_score,
        "best_expression": algorithm.best_expression,
        "algorithm": _algorithm_name(algorithm),
        "config": config_snapshot,
        "config_canon_scheme": (
            CONFIG_CANON_SCHEME if config_snapshot is not None else None
        ),
    }


def _config_snapshot(algorithm: Any) -> dict[str, Any] | None:
    config = getattr(algorithm, "config", None)
    if not isinstance(config, Mapping):
        return None
    try:
        snapshot = canonicalize_config(dict(config))
    except ConfigCanonicalizationError:
        return None
    algorithm_name = _algorithm_name(algorithm)
    if algorithm_name is not None and CONFIG_ARTIFACT_KEYS.get(algorithm_name):
        artifacts = getattr(algorithm, "artifacts", None)
        config_artifact_overlay(snapshot, algorithm_name, artifacts)
    return snapshot


def _algorithm_name(algorithm: Any) -> str | None:
    config = getattr(algorithm, "config", None)
    if not isinstance(config, Mapping):
        return None
    name = config.get("algorithm")
    return name if isinstance(name, str) else None
