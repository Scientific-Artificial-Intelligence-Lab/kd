
from __future__ import annotations

import hashlib
import importlib
import logging
from pathlib import Path
from typing import Protocol, cast

logger = logging.getLogger(__name__)

_HASH_CHUNK_SIZE = 1024 * 1024


class _HuggingFaceHubModule(Protocol):

    def hf_hub_download(
        self,
        *,
        repo_id: str,
        filename: str,
        revision: str,
        repo_type: str,
        cache_dir: str | None,
        local_files_only: bool,
    ) -> str: ...


def _import_huggingface_hub() -> _HuggingFaceHubModule:
    try:
        module = importlib.import_module("huggingface_hub")
    except ModuleNotFoundError as exc:
        if exc.name == "huggingface_hub":
            raise ImportError(
                "Remote dataset fetching requires the optional dependency "
                "`huggingface_hub`; install it with `uv sync --extra hub`."
            ) from exc
        raise
    return cast(_HuggingFaceHubModule, module)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _delete_corrupt_cache_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        logger.debug("Corrupt cache file already removed: %s", path)
    except OSError as exc:
        logger.warning("Failed to delete corrupt cache file %s: %s", path, exc)


def fetch_hub_file(
    repo_id: str,
    filename: str,
    *,
    revision: str,
    cache_dir: Path | None = None,
    expected_sha256: str | None = None,
    offline: bool = False,
) -> Path:
    if revision is None or not revision.strip():
        raise ValueError("revision is required for reproducible hub downloads")

    hub = _import_huggingface_hub()
    cache_dir_arg = str(cache_dir) if cache_dir is not None else None

    logger.debug(
        "Fetching Hugging Face dataset file repo_id=%s filename=%s revision=%s "
        "offline=%s",
        repo_id,
        filename,
        revision,
        offline,
    )
    try:
        downloaded = hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="dataset",
            cache_dir=cache_dir_arg,
            local_files_only=offline,
        )
    except Exception as exc:
        if offline:
            raise FileNotFoundError(
                f"Hub file {repo_id}/{filename}@{revision} is not cached; "
                "offline=True forbids network access."
            ) from exc
        raise

    path = Path(downloaded)
    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()
        actual = _sha256(path)
        if actual != expected:
            _delete_corrupt_cache_file(path)
            raise ValueError(
                f"Checksum mismatch for {path}: expected {expected}, actual {actual}"
            )

    return path
