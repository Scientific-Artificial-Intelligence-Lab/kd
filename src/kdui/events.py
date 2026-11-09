
from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pydantic import JsonValue

_SCAN_BLOCK = 64 * 1024


class CursorError(ValueError):
    pass


@dataclass(frozen=True)
class RecordPage:

    records: list[dict[str, JsonValue]]
    next_cursor: str
    has_more: bool
    reset: bool
    skipped: int


@dataclass(frozen=True)
class TextPage:

    text: str
    next_cursor: str
    has_more: bool
    reset: bool


@dataclass(frozen=True)
class _Chunk:
    data: bytes
    end: int
    has_more: bool
    reset: bool


def _stream(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def _encode(stream: str, offset: int) -> str:
    raw = json.dumps([stream, offset]).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode(cursor: str) -> tuple[str, int]:
    try:
        decoded = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    except (ValueError, binascii.Error, UnicodeEncodeError) as exc:
        raise CursorError(f"not a cursor: {cursor!r}") from exc
    match decoded:
        case [str(stream), int(offset)] if offset >= 0:
            return stream, offset
    raise CursorError(f"not a cursor: {cursor!r}")


def _newline_after(handle: BinaryIO, start: int) -> int | None:
    handle.seek(start)
    position = start
    while block := handle.read(_SCAN_BLOCK):
        index = block.find(b"\n")
        if index >= 0:
            return position + index + 1
        position += len(block)
    return None


def _read_chunk(path: Path, cursor: str | None, max_bytes: int) -> _Chunk:
    stream = _stream(path)
    offset, reset = 0, False
    if cursor is not None:
        cursor_stream, offset = _decode(cursor)
        if cursor_stream != stream:
            offset, reset = 0, True
    if not path.exists():
        return _Chunk(b"", 0, False, reset or offset > 0)
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        if offset > size:
            offset, reset = 0, True
        handle.seek(offset)
        window = handle.read(max_bytes)
        cut = window.rfind(b"\n")


        end = offset + cut + 1 if cut >= 0 else _newline_after(handle, offset) or offset
        handle.seek(offset)
        data = handle.read(end - offset)
        has_more = _newline_after(handle, end) is not None
    return _Chunk(data, end, has_more, reset)


def read_records(path: Path, cursor: str | None, *, max_bytes: int) -> RecordPage:
    chunk = _read_chunk(path, cursor, max_bytes)
    records: list[dict[str, JsonValue]] = []
    skipped = 0
    for line in chunk.data.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        if isinstance(record, dict):
            records.append(record)
        else:
            skipped += 1
    return RecordPage(
        records=records,
        next_cursor=_encode(_stream(path), chunk.end),
        has_more=chunk.has_more,
        reset=chunk.reset,
        skipped=skipped,
    )


def read_text(path: Path, cursor: str | None, *, max_bytes: int) -> TextPage:
    chunk = _read_chunk(path, cursor, max_bytes)
    return TextPage(
        text=chunk.data.decode("utf-8", errors="replace"),
        next_cursor=_encode(_stream(path), chunk.end),
        has_more=chunk.has_more,
        reset=chunk.reset,
    )
