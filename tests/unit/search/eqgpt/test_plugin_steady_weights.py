
from __future__ import annotations

from pathlib import Path

import pytest

from kd.search.eqgpt.backend import (
    DEFAULT_WEIGHTS_FILENAME,
    FakeGPTBackend,
    RealGPTBackend,
    resolve_asset_path,
)
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.eqgpt.vocab import load_vocab

pytestmark = pytest.mark.unit


def _capture_resolved_path(monkeypatch) -> list[Path]:
    resolved_paths: list[Path] = []

    def fake_from_assets(
        cls, config, *, weights_path=None, asset_dir=None, device=None
    ) -> FakeGPTBackend:
        del cls, config, device
        resolved_paths.append(
            resolve_asset_path(weights_path=weights_path, asset_dir=asset_dir)
        )
        return FakeGPTBackend(load_vocab().size, seed=0)

    monkeypatch.setattr(RealGPTBackend, "from_assets", classmethod(fake_from_assets))
    return resolved_paths


def test_steady_explicit_weights_path_is_an_allowed_reproduction_opt_in(
    tmp_path: Path, monkeypatch
) -> None:
    weights = tmp_path / DEFAULT_WEIGHTS_FILENAME
    weights.touch()
    resolved = _capture_resolved_path(monkeypatch)
    config = EqGPTConfig.steady_preset("smile", weights_path=weights)

    backend = EqGPTPlugin(config)._build_default_backend(load_vocab())

    assert isinstance(backend, FakeGPTBackend)
    assert resolved == [weights]


def test_steady_env_resolved_full_corpus_checkpoint_is_allowed(
    tmp_path: Path, monkeypatch
) -> None:
    weights = tmp_path / "gpt_model" / DEFAULT_WEIGHTS_FILENAME
    weights.parent.mkdir()
    weights.touch()
    monkeypatch.setenv("KD_EQGPT_ASSET_DIR", str(tmp_path))
    resolved = _capture_resolved_path(monkeypatch)
    config = EqGPTConfig.steady_preset("smile")

    backend = EqGPTPlugin(config)._build_default_backend(load_vocab())

    assert isinstance(backend, FakeGPTBackend)
    assert resolved == [weights]


def test_steady_silent_default_without_any_asset_still_fails_loudly(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("KD_EQGPT_ASSET_DIR", raising=False)
    config = EqGPTConfig.steady_preset("smile", asset_dir=tmp_path / "empty")

    with pytest.raises((ValueError, FileNotFoundError)) as exc_info:
        EqGPTPlugin(config)._build_default_backend(load_vocab())

    message = str(exc_info.value).lower()
    assert "weight" in message or "asset" in message
    assert "protocol mismatch" not in message
