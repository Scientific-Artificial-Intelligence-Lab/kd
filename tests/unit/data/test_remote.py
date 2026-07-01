from __future__ import annotations

import hashlib
import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

import kd
import kd.data as data
from kd.data.remote import _hf_client, fetch_hub_file
from kd.data.remote import _loaders as remote_loaders

pytestmark = pytest.mark.unit

_DownloadFunc = Callable[..., str]


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    download: _DownloadFunc,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    fake_hub = ModuleType("huggingface_hub")

    def hf_hub_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return download(**kwargs)

    fake_hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    return calls


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_heat_mat(path: Path) -> None:
    import scipy.io as sio

    x = np.linspace(0.0, 0.15, 4)[None,:]
    t = np.linspace(0.0, 0.2, 5)[None,:]
    u = np.arange(20, dtype=np.float64).reshape(4, 5)
    sio.savemat(path, {"x": x, "t": t, "usol": u})


def _write_fisher_mat(path: Path) -> None:
    import scipy.io as sio

    x = np.linspace(-1.0, 1.0, 4)[None,:]
    t = np.linspace(0.0, 0.2, 3)[:, None]
    u = np.arange(12, dtype=np.float64).reshape(3, 4)
    sio.savemat(path, {"x": x, "t": t, "U": u, "D": [[0.02]], "r": [[10.0]]})


def test_fetch_hub_file_success_returns_downloaded_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded_file = tmp_path / "downloaded.mat"
    downloaded_file.write_bytes(b"remote data")
    cache_dir = tmp_path / "cache"

    def download(**kwargs: object) -> str:
        return str(downloaded_file)

    calls = _install_fake_hub(monkeypatch, download)

    result = fetch_hub_file(
        "org/dataset",
        "data/file.mat",
        revision="abc123",
        cache_dir=cache_dir,
    )

    assert result == downloaded_file
    assert calls == [
        {
            "repo_id": "org/dataset",
            "filename": "data/file.mat",
            "revision": "abc123",
            "repo_type": "dataset",
            "cache_dir": str(cache_dir),
            "local_files_only": False,
        }
    ]


def test_fetch_hub_file_accepts_matching_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"checksum verified"
    downloaded_file = tmp_path / "checked.mat"
    downloaded_file.write_bytes(content)

    def download(**kwargs: object) -> str:
        return str(downloaded_file)

    _install_fake_hub(monkeypatch, download)

    result = fetch_hub_file(
        "org/dataset",
        "checked.mat",
        revision="abc123",
        expected_sha256=_digest(content),
    )

    assert result == downloaded_file
    assert downloaded_file.exists()


def test_fetch_hub_file_deletes_file_on_checksum_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"corrupt payload"
    downloaded_file = tmp_path / "corrupt.mat"
    downloaded_file.write_bytes(content)
    expected = "0" * 64
    actual = _digest(content)

    def download(**kwargs: object) -> str:
        return str(downloaded_file)

    _install_fake_hub(monkeypatch, download)

    with pytest.raises(
        ValueError,
        match=f"expected {expected}, actual {actual}",
    ):
        fetch_hub_file(
            "org/dataset",
            "corrupt.mat",
            revision="abc123",
            expected_sha256=expected,
        )

    assert not downloaded_file.exists()


def test_fetch_hub_file_offline_uses_cache_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached_file = tmp_path / "cached.mat"
    cached_file.write_bytes(b"cached data")

    def download(**kwargs: object) -> str:
        return str(cached_file)

    calls = _install_fake_hub(monkeypatch, download)

    result = fetch_hub_file(
        "org/dataset",
        "cached.mat",
        revision="abc123",
        offline=True,
    )

    assert result == cached_file
    assert calls[0]["local_files_only"] is True


def test_fetch_hub_file_offline_miss_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def download(**kwargs: object) -> str:
        raise RuntimeError("not in cache")

    _install_fake_hub(monkeypatch, download)

    with pytest.raises(FileNotFoundError, match="offline=True forbids network"):
        fetch_hub_file(
            "org/dataset",
            "missing.mat",
            revision="abc123",
            offline=True,
        )


def test_fetch_hub_file_rejects_missing_revision() -> None:
    with pytest.raises(ValueError, match="revision is required"):
        fetch_hub_file(
            "org/dataset",
            "file.mat",
            revision=None,
        )


def test_fetch_hub_file_missing_huggingface_hub_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = importlib.import_module

    def missing_hub_import(name: str, package: str | None = None) -> ModuleType:
        if name == "huggingface_hub":
            raise ModuleNotFoundError(
                "No module named 'huggingface_hub'",
                name=name,
            )
        return original_import(name, package)

    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    monkeypatch.setattr(_hf_client.importlib, "import_module", missing_hub_import)

    with pytest.raises(ImportError, match="uv sync --extra hub"):
        fetch_hub_file("org/dataset", "file.mat", revision="abc123")


@pytest.mark.parametrize(
    ("dataset_id", "writer", "expected_shape", "expected_file", "expected_sha"),
    [
        (
            "llm4ed-fisher",
            _write_fisher_mat,
            (4, 3),
            "llm4ed/fisher_groundtruth.mat",
            "884e0b06d5bfcc586db8f38a0a5fb8de417e7be25154342a534c0296353ca257",
        ),
        (
            "llm4ed-fisher-nonlinear",
            _write_fisher_mat,
            (4, 3),
            "llm4ed/fisher_nonlin_groundtruth.mat",
            "c8a8eb8024e5a18b8a08a495a5485454fcd5e1dbda58a014524f3c7b111927db",
        ),
        (
            "llm4ed-heat",
            _write_heat_mat,
            (4, 5),
            "llm4ed/Heat_equation.mat",
            "ae4f570537a2bbf0c32de96e26f7c22b22bf16ef283e63cfa8d07b21c405dff8",
        ),
    ],
)
def test_load_from_hub_fetches_pinned_spec_and_builds_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: str,
    writer: Callable[[Path], None],
    expected_shape: tuple[int, int],
    expected_file: str,
    expected_sha: str,
) -> None:
    mat_path = tmp_path / "remote.mat"
    writer(mat_path)
    calls: list[dict[str, object]] = []

    def fake_fetch_hub_file(
        repo_id: str,
        filename: str,
        *,
        revision: str,
        cache_dir: Path | None = None,
        expected_sha256: str | None = None,
        offline: bool = False,
    ) -> Path:
        calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
                "cache_dir": cache_dir,
                "expected_sha256": expected_sha256,
                "offline": offline,
            }
        )
        return mat_path

    monkeypatch.setattr(remote_loaders, "fetch_hub_file", fake_fetch_hub_file)

    dataset = data.load_from_hub(dataset_id)

    assert calls == [
        {
            "repo_id": "timeoutHao/KD-data",
            "filename": expected_file,
            "revision": "6c7dbe4f032e14fea2194644b09e74a4f254dd42",
            "cache_dir": None,
            "expected_sha256": expected_sha,
            "offline": False,
        }
    ]
    assert dataset.name == dataset_id
    assert dataset.axis_order == ["x", "t"]
    assert dataset.lhs_field == "u"
    assert dataset.lhs_axis == "t"
    assert tuple(dataset.get_field("u").shape) == expected_shape
    assert dataset.ground_truth == data.get_dataset(dataset_id).equation


def test_load_from_hub_preserves_fisher_raw_tx_orientation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mat_path = tmp_path / "fisher.mat"
    _write_fisher_mat(mat_path)
    monkeypatch.setattr(
        remote_loaders,
        "fetch_hub_file",
        lambda *args, **kwargs: mat_path,
    )

    dataset = data.load_from_hub("llm4ed-fisher")

    expected = torch.arange(12, dtype=torch.float64).reshape(3, 4).T
    torch.testing.assert_close(dataset.get_field("u"), expected)


def test_load_from_hub_rejects_builtin_dataset() -> None:
    with pytest.raises(ValueError, match="only loads remote datasets"):
        data.load_from_hub("burgers")


def test_top_level_remote_exports_match_data_module() -> None:
    assert kd.load_from_hub is data.load_from_hub
    assert kd.list_remote_datasets is data.list_remote_datasets
