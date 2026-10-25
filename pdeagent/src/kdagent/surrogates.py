
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REGISTRY_FORMAT = "kdagent-surrogates-v2"

REGISTRY_DIRNAME = "surrogates"

REGISTRY_FILENAME = "registry.jsonl"

_RECIPE_KEYS: frozenset[str] = frozenset({"dataset_id", "algorithm", "seed", "params"})

NETWORK_ALGORITHM = "network"


def recipe_key(recipe: dict[str, Any]) -> str:
    if set(recipe) != _RECIPE_KEYS:
        raise ValueError(
            f"a surrogate recipe has exactly the keys {sorted(_RECIPE_KEYS)}, "
            f"got {sorted(recipe)}"
        )
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def surrogate_id_of(recipe: dict[str, Any]) -> str:
    return f"{recipe['algorithm']}-{recipe_key(recipe)}"


def trained_surrogate_id(recipe: dict[str, Any], run_dir_name: str) -> str:
    canonical = json.dumps(
        {"recipe": recipe, "run_dir": run_dir_name},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{recipe['algorithm']}-{digest}"


def stopped_at_deadline(entry: SurrogateEntry) -> bool:
    return entry.identity is not None and bool(
        entry.identity.get("stopped_at_deadline", False)
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class SurrogateEntry:

    surrogate_id: str
    recipe: dict[str, Any]
    recipe_key: str
    key: str | None
    file: str
    file_sha256: str
    train_seconds: float
    epochs: int
    final_loss: float
    created_at: str
    identity: dict[str, Any] | None
    source_run: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entry_from_dict(payload: dict[str, Any]) -> SurrogateEntry:
    return SurrogateEntry(
        surrogate_id=payload["surrogate_id"],
        recipe=payload["recipe"],
        recipe_key=payload["recipe_key"],
        key=payload["key"],
        file=payload["file"],
        file_sha256=payload["file_sha256"],
        train_seconds=payload["train_seconds"],
        epochs=payload["epochs"],
        final_loss=payload["final_loss"],
        created_at=payload["created_at"],
        identity=payload["identity"],
        source_run=payload.get("source_run"),
    )


class SurrogateRegistry:

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._directory = workspace / REGISTRY_DIRNAME
        self._path = self._directory / REGISTRY_FILENAME
        self._entries: dict[str, SurrogateEntry] = {}
        if self._path.is_file():
            self._replay()

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def directory(self) -> Path:
        return self._directory

    def file_for(self, surrogate_id: str) -> Path:
        return self._directory / f"{surrogate_id}.pt"

    def register(
        self,
        *,
        recipe: dict[str, Any],
        key: str | None,
        file_sha256: str,
        train_seconds: float,
        epochs: int,
        final_loss: float,
        created_at: str,
        identity: dict[str, Any] | None,
        surrogate_id: str | None = None,
        source_run: str | None = None,
    ) -> SurrogateEntry:
        if surrogate_id is None:
            surrogate_id = surrogate_id_of(recipe)
        entry = SurrogateEntry(
            surrogate_id=surrogate_id,
            recipe=deepcopy(recipe),
            recipe_key=recipe_key(recipe),
            key=key,
            file=self.file_for(surrogate_id).relative_to(self._workspace).as_posix(),
            file_sha256=file_sha256,
            train_seconds=train_seconds,
            epochs=epochs,
            final_loss=final_loss,
            created_at=created_at,
            identity=deepcopy(identity),
            source_run=source_run,
        )
        self._append(entry.to_dict())
        self._entries[surrogate_id] = entry
        return entry

    def trainings_of(self, recipe: dict[str, Any]) -> tuple[SurrogateEntry, ...]:
        own = surrogate_id_of(recipe)
        key = recipe_key(recipe)
        rows = sorted(
            (
                entry
                for entry in self._entries.values()
                if entry.recipe_key == key and not stopped_at_deadline(entry)
            ),
            key=lambda entry: entry.surrogate_id != own,
        )
        return tuple(rows)

    def find_by_recipe(self, recipe: dict[str, Any]) -> SurrogateEntry | None:
        return next(iter(self.trainings_of(recipe)), None)

    def entry(self, surrogate_id: str) -> SurrogateEntry:
        self._require_registered(surrogate_id)
        return self._entries[surrogate_id]

    def path(self, surrogate_id: str) -> Path:
        return self._workspace / self.entry(surrogate_id).file

    def ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def _require_registered(self, surrogate_id: str) -> None:
        if surrogate_id not in self._entries:
            raise KeyError(
                f"surrogate {surrogate_id!r} is not in this workspace's registry; "
                f"the registered ones are {list(self._entries)}"
            )

    def _append(self, row: dict[str, Any]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"format": REGISTRY_FORMAT, **row}, ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _replay(self) -> None:
        text = self._path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{self._path} line {number}: "
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}not a JSON object ({exc})") from exc
            tag = payload.get("format")
            if tag != REGISTRY_FORMAT:
                raise ValueError(
                    f"{where}format {tag!r} is not the {REGISTRY_FORMAT!r} this "
                    "registry writes; a registry from an older controller does "
                    "not replay, so retrain its surrogates in a fresh workspace"
                )
            entry = _entry_from_dict(payload)
            self._entries[entry.surrogate_id] = entry


__all__ = [
    "NETWORK_ALGORITHM",
    "REGISTRY_DIRNAME",
    "REGISTRY_FILENAME",
    "REGISTRY_FORMAT",
    "SurrogateEntry",
    "SurrogateRegistry",
    "file_sha256",
    "recipe_key",
    "stopped_at_deadline",
    "surrogate_id_of",
    "trained_surrogate_id",
]
