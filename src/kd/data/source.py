
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["DatasetSource"]


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
