
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)



KD_HUB_REPO_ID: Final[str] = "timeoutHao/KD-data"

_HASH_CHUNK_SIZE = 1024 * 1024






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


def _require_revision(revision: str) -> None:
    if not revision.strip():
        raise ValueError("revision is required for reproducible hub downloads")


def fetch_hub_file(
    repo_id: str,
    filename: str,
    *,
    revision: str,
    cache_dir: Path | None = None,
    expected_sha256: str | None = None,
    offline: bool = False,
) -> Path:
    _require_revision(revision)
    from huggingface_hub import hf_hub_download

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
        downloaded = hf_hub_download(
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


def fetch_hub_tree(repo_id: str, subdir: str, *, revision: str) -> Path:
    _require_revision(revision)
    from huggingface_hub import snapshot_download

    logger.debug(
        "Fetching Hugging Face dataset tree repo_id=%s subdir=%s revision=%s",
        repo_id,
        subdir,
        revision,
    )
    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"{subdir}/**"],
    )
    tree = Path(snapshot) / subdir
    if not tree.is_dir():
        raise FileNotFoundError(
            f"Hub tree {repo_id}/{subdir}@{revision} is not present at {tree}: "
            "the download did not happen (Hub unreachable?) or the directory "
            "does not exist in the repository."
        )
    return tree
