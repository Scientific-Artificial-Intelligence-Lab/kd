
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["DatasetSource", "file_sha256"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, kw_only=True)
class DatasetSource:

    path: str
    sha256: str
    container: str
    layout: str
    hints: dict[str, Any]
    select: dict[str, int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "container": self.container,
            "layout": self.layout,
            "hints": dict(self.hints),
            "select": None if self.select is None else dict(self.select),
        }
