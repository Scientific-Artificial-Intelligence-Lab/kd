
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kd.search.eqgpt.backend import (
    ASSET_ENV_VAR,
    DEFAULT_WEIGHTS_FILENAME,
    FakeGPTBackend,
    GPTBackend,
    RealGPTBackend,
    resolve_asset_path,
)
from kd.search.eqgpt.gpt import GPTConfig

_VOCAB = 8







@pytest.mark.smoke
def test_backends_conform_to_protocol() -> None:
    assert issubclass(FakeGPTBackend, GPTBackend)
    assert issubclass(RealGPTBackend, GPTBackend)


def test_asset_env_var_name_is_locked() -> None:
    assert ASSET_ENV_VAR == "KD_EQGPT_ASSET_DIR"


def test_import_kd_without_assets_does_not_crash(tmp_path: Path) -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "KD_EQGPT_ASSET_DIR": str(tmp_path / "does_not_exist"),
    }

    proc = subprocess.run(
        [sys.executable, "-c", "import kd; import kd.search.eqgpt; print('ok')"],
        capture_output=True,
        text=True,
        env={**env, "PYTHONPATH": ":".join(sys.path)},
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout







def test_fake_backend_logits_depend_on_prefix() -> None:
    fake = FakeGPTBackend(_VOCAB, seed=0)
    a = fake.next_token_logits([5, 6, 2, 7])
    b = fake.next_token_logits([5, 3, 2, 7])
    assert a.shape == (_VOCAB,)
    assert not torch.allclose(a, b)


def test_fake_backend_next_token_matches_forward_last_row() -> None:
    fake = FakeGPTBackend(_VOCAB, seed=0)
    prefix = [5, 6, 2, 7]
    tokens = torch.tensor([prefix], dtype=torch.long)
    torch.testing.assert_close(
        fake.next_token_logits(prefix),
        fake.forward_logits(tokens)[0, -1],
        rtol=0.0,
        atol=0.0,
    )


def test_fake_backend_exposes_trainable_parameters() -> None:
    fake = FakeGPTBackend(_VOCAB, seed=0)
    params = list(fake.parameters())
    assert params
    assert all(isinstance(p, torch.Tensor) for p in params)
    assert all(p.requires_grad for p in params)


def test_fake_backend_same_prefix_is_deterministic() -> None:
    fake = FakeGPTBackend(_VOCAB, seed=0)
    prefix = [5, 6, 2, 7]
    torch.testing.assert_close(
        fake.next_token_logits(prefix),
        fake.next_token_logits(prefix),
        rtol=0.0,
        atol=0.0,
    )


def test_fake_backend_forward_logits_shape() -> None:
    fake = FakeGPTBackend(_VOCAB, seed=0)
    tokens = torch.tensor([[5, 6, 2, 7, 1]], dtype=torch.long)
    logits = fake.forward_logits(tokens)
    assert logits.shape == (1, 5, _VOCAB)
    assert torch.isfinite(logits).all()


def test_fake_backend_state_dict_is_detached_snapshot() -> None:
    fake = FakeGPTBackend(_VOCAB, seed=0)
    prefix = [5, 6, 2, 7]
    original = fake.next_token_logits(prefix).clone()
    saved = fake.state_dict()
    pre_mutation = {k: v.clone() for k, v in saved.items()}

    with torch.no_grad():
        next(iter(fake.parameters())).add_(1.0)

    for key, value in saved.items():
        torch.testing.assert_close(value, pre_mutation[key], rtol=0.0, atol=0.0)

    fake.load_state_dict(saved)
    torch.testing.assert_close(
        fake.next_token_logits(prefix), original, rtol=0.0, atol=0.0
    )


def test_fake_backend_forward_grad_enabled_next_token_detached() -> None:
    fake = FakeGPTBackend(_VOCAB, seed=0)
    tokens = torch.tensor([[5, 6, 2, 7, 1]], dtype=torch.long)
    loss = fake.forward_logits(tokens).sum()
    loss.backward()
    params = list(fake.parameters())
    assert params
    assert all(p.grad is not None for p in params)

    out = fake.next_token_logits([5, 6, 2, 7])
    assert out.grad_fn is None
    assert out.requires_grad is False


def test_fake_backend_next_token_empty_prefix_raises() -> None:
    fake = FakeGPTBackend(_VOCAB, seed=0)
    with pytest.raises(ValueError, match="empty prefix"):
        fake.next_token_logits([])


def test_fake_backend_state_dict_roundtrip_changes_then_restores_behaviour() -> None:
    fake = FakeGPTBackend(_VOCAB, seed=0)
    prefix = [5, 6, 2, 7]
    original = fake.next_token_logits(prefix).clone()
    saved = {k: v.clone() for k, v in fake.state_dict().items()}

    other = FakeGPTBackend(_VOCAB, seed=999)
    fake.load_state_dict(other.state_dict())
    changed = fake.next_token_logits(prefix)
    assert not torch.allclose(original, changed)

    fake.load_state_dict(saved)
    restored = fake.next_token_logits(prefix)
    torch.testing.assert_close(restored, original, rtol=0.0, atol=0.0)








def _make_weights(root: Path) -> Path:
    gpt_dir = root / "gpt_model"
    gpt_dir.mkdir(parents=True, exist_ok=True)
    weights = gpt_dir / DEFAULT_WEIGHTS_FILENAME
    weights.write_bytes(b"stub")
    return weights


def test_resolve_prefers_explicit_weights_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ASSET_ENV_VAR, raising=False)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"stub")
    assert resolve_asset_path(weights_path=weights) == weights


def test_resolve_uses_asset_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ASSET_ENV_VAR, raising=False)
    weights = _make_weights(tmp_path)
    assert resolve_asset_path(asset_dir=tmp_path) == weights


def test_resolve_uses_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = _make_weights(tmp_path)
    monkeypatch.setenv(ASSET_ENV_VAR, str(tmp_path))
    assert resolve_asset_path() == weights


def test_resolve_explicit_weights_path_wins_over_env_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit.pt"
    explicit.write_bytes(b"stub")
    _make_weights(tmp_path / "asset")
    _make_weights(tmp_path / "env")
    _make_weights(tmp_path / "fb")
    monkeypatch.setenv(ASSET_ENV_VAR, str(tmp_path / "env"))
    resolved = resolve_asset_path(
        weights_path=explicit,
        asset_dir=tmp_path / "asset",
        fallback_dir=tmp_path / "fb",
    )
    assert resolved == explicit


def test_resolve_asset_dir_wins_over_env_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _make_weights(tmp_path / "asset")
    _make_weights(tmp_path / "env")
    _make_weights(tmp_path / "fb")
    monkeypatch.setenv(ASSET_ENV_VAR, str(tmp_path / "env"))
    resolved = resolve_asset_path(
        asset_dir=tmp_path / "asset", fallback_dir=tmp_path / "fb"
    )
    assert resolved == asset


def test_resolve_env_wins_over_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_weights = _make_weights(tmp_path / "env")
    _make_weights(tmp_path / "fb")
    monkeypatch.setenv(ASSET_ENV_VAR, str(tmp_path / "env"))
    assert resolve_asset_path(fallback_dir=tmp_path / "fb") == env_weights


def test_resolve_explicit_nonexistent_weights_path_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ASSET_ENV_VAR, raising=False)
    with pytest.raises(FileNotFoundError):
        resolve_asset_path(weights_path=tmp_path / "nope.pt")


def test_resolve_missing_asset_raises_with_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ASSET_ENV_VAR, raising=False)
    with pytest.raises(FileNotFoundError) as exc:
        resolve_asset_path(
            asset_dir=tmp_path / "empty_asset", fallback_dir=tmp_path / "empty_fb"
        )
    assert ASSET_ENV_VAR in str(exc.value)


def test_real_backend_from_assets_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ASSET_ENV_VAR, raising=False)
    with pytest.raises((FileNotFoundError, OSError)):
        RealGPTBackend.from_assets(
            GPTConfig(vocab_size=57),
            asset_dir=tmp_path / "empty",
        )
