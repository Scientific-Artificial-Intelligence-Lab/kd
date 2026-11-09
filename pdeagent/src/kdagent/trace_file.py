
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def append_row(path: Path, row: dict[str, Any]) -> None:
    line = json.dumps(row, default=str, ensure_ascii=False) + "\n"
    with path.open("ab", buffering=0) as sink:
        rest = memoryview(line.encode("utf-8"))
        while rest:
            rest = rest[sink.write(rest):]


def read_rows(path: Path) -> list[dict[str, Any]]:
    *lines, cut = path.read_bytes().split(b"\n")
    if cut:
        logger.warning(
            "%s: skipped a last line of %d bytes without a newline, a row the "
            "tracer was writing when the process stopped",
            path,
            len(cut),
        )
    return [json.loads(line) for line in lines]
