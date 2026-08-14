
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from kd.search._record_schema import (
    RecordSchemaError,
    StrictDecodeError,
    strict_keys,
    validate_lineage,
)




from kd.search.checkpoint_payload import _CHECKPOINT_TMP_SUFFIX as _TMP_SUFFIX





CKPTMAN_SCHEME: Final[str] = "kd-ckptman-v1"






CKPTMAN_SCHEMA_VERSION: Final[int] = 2
MANIFEST_FILENAME: Final[str] = "manifest.json"

KIND_PERIODIC: Final[str] = "periodic"
KIND_FINAL: Final[str] = "final"
KIND_VOCABULARY: Final[frozenset[str]] = frozenset({KIND_PERIODIC, KIND_FINAL})

FINAL_STATUS_COMPLETED: Final[str] = "completed"
FINAL_STATUS_CRASHED: Final[str] = "crashed"
FINAL_STATUS_VOCABULARY: Final[frozenset[str]] = frozenset(
    {FINAL_STATUS_COMPLETED, FINAL_STATUS_CRASHED}
)

CKPTMAN_CONFIG_HASH_SCHEME: Final[str] = "kd-confighash-v1"

_SHA256_HEX_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")







_CKPTMAN_V1_FIELDS: Final[tuple[str, ...]] = (
    "filename",
    "kind",
    "final_status",
    "iteration",
    "best_score",
    "best_expression",
    "created_at",
    "algorithm",
    "seed",
    "config_hash",
    "library_fingerprint",
    "kd_version",
)
_CKPTMAN_V1_FIELD_SET: Final[frozenset[str]] = frozenset(_CKPTMAN_V1_FIELDS)




_CKPTMAN_HEADER_KEYS_V1: Final[frozenset[str]] = frozenset(
    {"scheme", "schema_version", "entries"}
)
_CKPTMAN_HEADER_KEYS_V2: Final[frozenset[str]] = _CKPTMAN_HEADER_KEYS_V1 | {
    "lineage"
}
_CKPTMAN_HEADER_KEYS_BY_VERSION: Final[dict[int, frozenset[str]]] = {
    1: _CKPTMAN_HEADER_KEYS_V1,
    2: _CKPTMAN_HEADER_KEYS_V2,
}


class CheckpointManifestError(RecordSchemaError):
    pass


@dataclass(frozen=True)
class CheckpointManifestEntry:

    filename: str
    kind: str
    final_status: str | None
    iteration: int
    best_score: float | None
    best_expression: str | None
    created_at: str
    algorithm: str | None
    seed: int | None
    config_hash: str | None
    library_fingerprint: str | None
    kd_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str) or not self.filename:
            raise CheckpointManifestError(
                f"filename must be a non-empty str; got {self.filename!r}"
            )
        path = Path(self.filename)
        if path.name != self.filename or ".." in path.parts:
            raise CheckpointManifestError(
                "filename must be a bare name (no path separators or '..'); "
                f"got {self.filename!r}"
            )
        if self.filename == MANIFEST_FILENAME:
            raise CheckpointManifestError(
                f"filename must not be the manifest file itself: {self.filename!r}"
            )
        if self.filename.endswith(_TMP_SUFFIX):
            raise CheckpointManifestError(
                f"filename must not end with the staging suffix {_TMP_SUFFIX!r}: "
                f"{self.filename!r}"
            )




        if not isinstance(self.kind, str) or self.kind not in KIND_VOCABULARY:
            raise CheckpointManifestError(
                f"kind must be one of {sorted(KIND_VOCABULARY)!r}; "
                f"got {self.kind!r}"
            )

        if self.kind == KIND_PERIODIC:
            if self.final_status is not None:
                raise CheckpointManifestError(
                    "final_status must be None for a periodic entry; "
                    f"got {self.final_status!r}"
                )
        elif (
            not isinstance(self.final_status, str)
            or self.final_status not in FINAL_STATUS_VOCABULARY
        ):
            raise CheckpointManifestError(
                "final_status must be one of "
                f"{sorted(FINAL_STATUS_VOCABULARY)!r} for a final entry; "
                f"got {self.final_status!r}"
            )

        if type(self.iteration) is not int or self.iteration < 0:
            raise CheckpointManifestError(
                "iteration must be a non-negative int (bool rejected); "
                f"got {self.iteration!r}"
            )

        if self.best_score is not None and (
            type(self.best_score) is bool
            or not isinstance(self.best_score, (int, float))
            or not math.isfinite(self.best_score)
        ):
            raise CheckpointManifestError(
                f"best_score must be a finite number or None; got {self.best_score!r}"
            )

        if self.best_expression is not None and (
            not isinstance(self.best_expression, str) or not self.best_expression
        ):
            raise CheckpointManifestError(
                "best_expression must be a non-empty str or None; "
                f"got {self.best_expression!r}"
            )

        if not isinstance(self.created_at, str) or not self.created_at:
            raise CheckpointManifestError(
                f"created_at must be a non-empty str; got {self.created_at!r}"
            )

        if self.algorithm is not None and (
            not isinstance(self.algorithm, str) or not self.algorithm
        ):
            raise CheckpointManifestError(
                f"algorithm must be a non-empty str or None; got {self.algorithm!r}"
            )

        if self.seed is not None and type(self.seed) is not int:
            raise CheckpointManifestError(
                f"seed must be an int or None (bool rejected); got {self.seed!r}"
            )

        if self.config_hash is not None and (
            not isinstance(self.config_hash, str)
            or _SHA256_HEX_RE.fullmatch(self.config_hash) is None
        ):
            raise CheckpointManifestError(
                "config_hash must be a 64-char lowercase sha256 hex or None; "
                f"got {self.config_hash!r}"
            )

        if self.library_fingerprint is not None and (
            not isinstance(self.library_fingerprint, str)
            or not self.library_fingerprint
        ):
            raise CheckpointManifestError(
                "library_fingerprint must be a non-empty str or None; "
                f"got {self.library_fingerprint!r}"
            )

        if not isinstance(self.kd_version, str) or not self.kd_version:
            raise CheckpointManifestError(
                f"kd_version must be a non-empty str; got {self.kd_version!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in _CKPTMAN_V1_FIELDS}

    @classmethod
    def from_dict(cls, data: object) -> CheckpointManifestEntry:
        if not isinstance(data, dict):
            raise CheckpointManifestError(
                f"checkpoint manifest entry must be a dict; got {type(data).__name__}"
            )
        try:
            strict_keys(
                data,
                object_name="checkpoint manifest entry",
                required=_CKPTMAN_V1_FIELD_SET,
            )
        except StrictDecodeError as exc:
            raise CheckpointManifestError(str(exc)) from exc
        return cls(**{field: data[field] for field in _CKPTMAN_V1_FIELDS})


def load_checkpoint_manifest(
    directory: str | Path,
) -> tuple[CheckpointManifestEntry, ...]:
    directory = Path(directory)
    if not directory.is_dir():
        raise CheckpointManifestError(
            f"checkpoint directory is not a directory: {directory}"
        )
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise CheckpointManifestError(f"no checkpoint manifest at {manifest_path}")

    with manifest_path.open(encoding="utf-8") as handle:
        try:
            data = json.load(handle)



        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CheckpointManifestError(
                f"checkpoint manifest is invalid JSON ({manifest_path}): {exc}"
            ) from exc

    if not isinstance(data, dict):
        raise CheckpointManifestError(
            "checkpoint manifest payload must be a JSON object"
        )


    version = data.get("schema_version")
    if type(version) is not int or version not in _CKPTMAN_HEADER_KEYS_BY_VERSION:
        raise CheckpointManifestError(
            f"unsupported checkpoint manifest schema_version: got {version!r}; "
            f"supported: {sorted(_CKPTMAN_HEADER_KEYS_BY_VERSION)!r}"
        )
    try:
        strict_keys(
            data,
            object_name="checkpoint manifest header",
            required=_CKPTMAN_HEADER_KEYS_BY_VERSION[version],
        )
    except StrictDecodeError as exc:
        raise CheckpointManifestError(str(exc)) from exc

    scheme = data["scheme"]
    if scheme != CKPTMAN_SCHEME:
        raise CheckpointManifestError(
            f"unsupported checkpoint manifest scheme: got {scheme!r}; "
            f"expected {CKPTMAN_SCHEME!r}"
        )
    if version >= 2:




        validate_lineage(data["lineage"], error_cls=CheckpointManifestError)

    raw_entries = data["entries"]
    if not isinstance(raw_entries, list):
        raise CheckpointManifestError(
            "checkpoint manifest entries must be a JSON array"
        )

    entries: list[CheckpointManifestEntry] = []
    seen: set[str] = set()
    final_count = 0
    for index, raw in enumerate(raw_entries):
        try:
            entry = CheckpointManifestEntry.from_dict(raw)
        except CheckpointManifestError as exc:
            raise CheckpointManifestError(
                f"checkpoint manifest entry [{index}]: {exc}"
            ) from exc
        if entry.filename in seen:
            raise CheckpointManifestError(
                f"duplicate checkpoint filename in manifest: {entry.filename!r}"
            )
        seen.add(entry.filename)
        if entry.kind == KIND_FINAL:
            final_count += 1
            if final_count > 1:
                raise CheckpointManifestError(
                    "more than one final entry in the checkpoint manifest"
                )
        entries.append(entry)

    for entry in entries:
        if not (directory / entry.filename).is_file():
            raise CheckpointManifestError(
                "missing checkpoint file named by the manifest: "
                f"{directory / entry.filename}"
            )

    referenced = {entry.filename for entry in entries}
    for child in sorted(directory.iterdir()):
        if not child.is_file() or child.name == MANIFEST_FILENAME:
            continue
        if child.name not in referenced:
            raise CheckpointManifestError(
                f"unreferenced checkpoint file (orphan) not named by the "
                f"manifest: {child}"
            )

    return tuple(entries)
