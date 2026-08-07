
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from kd.search.callbacks import CHECKPOINT_VERSION, build_checkpoint_payload
from kd.search.dlga import DLGAPlugin
from kd.search.run_spec import CONFIG_CANON_SCHEME, canonicalize_config
from kd.search.sga import SGAConfig, SGAPlugin


class _StubAlgorithm:

    config: dict[str, Any] = {"algorithm": "stub"}
    state: dict[str, Any] = {"weights": [1, 2, 3]}
    best_score = 0.5
    best_expression = "u_t = -u"


def test_payload_wires_all_eight_keys() -> None:


    payload = build_checkpoint_payload(7, _StubAlgorithm())
    assert payload == {
        "version": CHECKPOINT_VERSION,
        "iteration": 7,
        "algorithm_state": {"weights": [1, 2, 3]},
        "best_score": 0.5,
        "best_expression": "u_t = -u",
        "algorithm": "stub",
        "config": {"algorithm": "stub"},
        "config_canon_scheme": CONFIG_CANON_SCHEME,
    }


def test_payload_config_snapshot_matches_canonicalized_plugin_config() -> None:
    plugin = SGAPlugin(SGAConfig())
    payload = build_checkpoint_payload(3, plugin)
    assert payload["config"] == canonicalize_config(dict(plugin.config))
    assert payload["config_canon_scheme"] == CONFIG_CANON_SCHEME


def test_payload_config_snapshot_overlays_injected_model_identity() -> None:
    plugin = DLGAPlugin(surrogate_model=nn.Sequential(nn.Linear(2, 1)))
    payload = build_checkpoint_payload(1, plugin)
    surrogate = payload["config"]["surrogate_model"]


    assert "sha256" in surrogate
    assert "artifact" not in surrogate
    assert payload["config_canon_scheme"] == CONFIG_CANON_SCHEME


class _RaisingArtifactsWriter:

    state: dict[str, Any] = {"weights": [1]}
    best_score = 0.0
    best_expression = ""

    def __init__(self, algorithm: str, extra: dict[str, Any]) -> None:
        self._config = {"algorithm": algorithm, **extra}

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)

    @property
    def artifacts(self) -> dict[str, Any]:
        raise FileNotFoundError("provenance asset read must not run per write")


def test_config_snapshot_skips_artifacts_for_algorithm_without_config_keys() -> None:
    writer = _RaisingArtifactsWriter("eqgpt", {"sparsity_alpha": 0.02})
    payload = build_checkpoint_payload(2, writer)
    assert payload["config"] == canonicalize_config(dict(writer.config))
    assert payload["config_canon_scheme"] == CONFIG_CANON_SCHEME


def test_config_snapshot_does_read_artifacts_for_config_artifact_algorithm() -> None:
    writer = _RaisingArtifactsWriter("sga", {"num": 4})
    with pytest.raises(FileNotFoundError):
        build_checkpoint_payload(2, writer)


def test_payload_rejects_non_mapping_config_loudly() -> None:
    writer = MagicMock()
    writer.state = {"weights": [1]}
    with pytest.raises(TypeError, match="Mapping"):
        build_checkpoint_payload(0, writer)


def test_runner_and_callback_emit_via_single_builder() -> None:
    import inspect

    from kd.search import callbacks, runner

    runner_src = inspect.getsource(runner.ExperimentRunner.save_checkpoint)
    cb_src = inspect.getsource(callbacks.CheckpointCallback)
    assert "build_checkpoint_payload(" in runner_src


    assert cb_src.count("build_checkpoint_payload(") >= 2
    assert "_CHECKPOINT_VERSION" not in runner_src
