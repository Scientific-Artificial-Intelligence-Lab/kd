
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kd.search._checkpoint_manifest_verify import (
    CKPTMAN_CONFIG_HASH_SCHEME as CKPTMAN_CONFIG_HASH_SCHEME,
)
from kd.search._checkpoint_manifest_verify import (
    CKPTMAN_SCHEMA_VERSION as CKPTMAN_SCHEMA_VERSION,
)
from kd.search._checkpoint_manifest_verify import (
    CKPTMAN_SCHEME as CKPTMAN_SCHEME,
)
from kd.search._checkpoint_manifest_verify import (
    FINAL_STATUS_COMPLETED as FINAL_STATUS_COMPLETED,
)
from kd.search._checkpoint_manifest_verify import (
    FINAL_STATUS_CRASHED as FINAL_STATUS_CRASHED,
)
from kd.search._checkpoint_manifest_verify import (
    FINAL_STATUS_VOCABULARY as FINAL_STATUS_VOCABULARY,
)
from kd.search._checkpoint_manifest_verify import (
    KIND_FINAL as KIND_FINAL,
)
from kd.search._checkpoint_manifest_verify import (
    KIND_PERIODIC as KIND_PERIODIC,
)
from kd.search._checkpoint_manifest_verify import (
    KIND_VOCABULARY as KIND_VOCABULARY,
)
from kd.search._checkpoint_manifest_verify import (
    MANIFEST_FILENAME as MANIFEST_FILENAME,
)
from kd.search._checkpoint_manifest_verify import (
    CheckpointManifestEntry as CheckpointManifestEntry,
)
from kd.search._checkpoint_manifest_verify import (
    CheckpointManifestError as CheckpointManifestError,
)
from kd.search._checkpoint_manifest_verify import (
    load_checkpoint_manifest as load_checkpoint_manifest,
)
from kd.search.checkpoint_payload import _CHECKPOINT_TMP_SUFFIX as _TMP_SUFFIX

logger = logging.getLogger(__name__)

__all__ = [
    "CKPTMAN_CONFIG_HASH_SCHEME",
    "CKPTMAN_SCHEMA_VERSION",
    "CKPTMAN_SCHEME",
    "FINAL_STATUS_COMPLETED",
    "FINAL_STATUS_CRASHED",
    "FINAL_STATUS_VOCABULARY",
    "KIND_FINAL",
    "KIND_PERIODIC",
    "KIND_VOCABULARY",
    "MANIFEST_FILENAME",
    "CheckpointManifestEntry",
    "CheckpointManifestError",
    "CheckpointManifestWriter",
    "build_manifest_entry",
    "config_hash_of_snapshot",
    "load_checkpoint_manifest",
]







def config_hash_of_snapshot(snapshot: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(snapshot), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(
        f"{CKPTMAN_CONFIG_HASH_SCHEME}:{canonical}".encode()
    ).hexdigest()







def build_manifest_entry(
    payload: Mapping[str, Any],
    *,
    filename: str,
    kind: str,
    final_status: str | None,
) -> CheckpointManifestEntry:
    from kd import __version__ as kd_version

    config = payload.get("config")

    raw_score = payload.get("best_score")
    if (
        type(raw_score) is not bool
        and isinstance(raw_score, (int, float))
        and math.isfinite(raw_score)
    ):
        best_score: float | None = float(raw_score)
    else:
        best_score = None

    raw_expression = payload.get("best_expression")
    best_expression = (
        raw_expression
        if isinstance(raw_expression, str) and raw_expression
        else None
    )

    raw_algorithm = payload.get("algorithm")
    algorithm = raw_algorithm if isinstance(raw_algorithm, str) else None

    seed: int | None = None
    library_fingerprint: str | None = None
    config_hash: str | None = None
    if isinstance(config, dict):
        raw_seed = config.get("seed")
        if type(raw_seed) is int:
            seed = raw_seed
        raw_fingerprint = config.get("library_fingerprint")
        if isinstance(raw_fingerprint, str):
            library_fingerprint = raw_fingerprint
        try:
            config_hash = config_hash_of_snapshot(config)
        except (TypeError, ValueError):
            config_hash = None

    created_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )

    return CheckpointManifestEntry(
        filename=filename,
        kind=kind,
        final_status=final_status,
        iteration=payload["iteration"],
        best_score=best_score,
        best_expression=best_expression,
        created_at=created_at,
        algorithm=algorithm,
        seed=seed,
        config_hash=config_hash,
        library_fingerprint=library_fingerprint,
        kd_version=kd_version,
    )







class CheckpointManifestWriter:

    def __init__(
        self, directory: Path, entries: list[CheckpointManifestEntry]
    ) -> None:
        self._directory = directory
        self._entries = entries

    @classmethod
    def create(cls, directory: str | Path) -> CheckpointManifestWriter:
        directory = Path(directory)
        if directory.exists():
            if not directory.is_dir():
                raise CheckpointManifestError(
                    f"checkpoint directory path is not a directory: {directory}"
                )
            if any(directory.iterdir()):
                raise CheckpointManifestError(
                    "checkpoint directory is not empty (reuse is not supported; "
                    f"use a fresh directory per run): {directory}"
                )
        directory.mkdir(parents=True, exist_ok=True)
        writer = cls(directory, [])
        writer._persist()
        return writer

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def entries(self) -> tuple[CheckpointManifestEntry, ...]:
        return tuple(self._entries)

    def append(self, entry: CheckpointManifestEntry) -> None:
        if any(existing.filename == entry.filename for existing in self._entries):
            raise CheckpointManifestError(
                f"checkpoint {entry.filename!r} already listed in the manifest"
            )
        if entry.kind == KIND_FINAL and any(
            existing.kind == KIND_FINAL for existing in self._entries
        ):
            raise CheckpointManifestError(
                "a final entry already exists in the manifest (one run per "
                "directory)"
            )
        self._entries.append(entry)
        self._persist()

    def prune_periodic(self, keep_last_n: int) -> None:
        periodic = [e for e in self._entries if e.kind == KIND_PERIODIC]
        excess = len(periodic) - keep_last_n
        if excess <= 0:
            return
        remove_names = {e.filename for e in periodic[:excess]}
        self._entries = [
            e for e in self._entries if e.filename not in remove_names
        ]
        self._persist()
        for name in remove_names:
            try:
                (self._directory / name).unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Failed to unlink pruned checkpoint %s",
                    self._directory / name,
                )

    def _persist(self) -> None:
        payload = {
            "scheme": CKPTMAN_SCHEME,
            "schema_version": CKPTMAN_SCHEMA_VERSION,
            "entries": [entry.to_dict() for entry in self._entries],
        }
        tmp_path = self._directory / f"{MANIFEST_FILENAME}{_TMP_SUFFIX}"
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False, sort_keys=True)
        os.replace(tmp_path, self._directory / MANIFEST_FILENAME)
