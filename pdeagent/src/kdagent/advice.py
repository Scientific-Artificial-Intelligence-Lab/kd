
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

Advice = dict[str, Any]
_TASK_KINDS = frozenset({"pde-evolution", "pde-steady", "tabular-regression", "ode"})
_MODES = frozenset(
    {
        "default",
        "autograd",
        "mode1",
        "tabular",
        "single_wave",
        "wave_multicase",
        "steady",
    }
)
_DOCUMENT_KEYS = frozenset(
    {
        "format",
        "version",
        "advice_id",
        "built_at",
        "builder",
        "basis",
        "usage",
        "cost_anchors",
    }
)
_ITEM_KEYS = frozenset({"algorithm", "mode", "task_kind", "text"})
_SHAPE_KEYS = frozenset({"points", "n_axes", "lhs_order", "n_fields"})


def _keys(
    value: dict[str, Any], allowed: frozenset[str], where: str, *, required: bool = True
) -> None:
    unknown = value.keys() - allowed
    missing = allowed - value.keys() if required else set()
    if unknown or missing:
        raise ValueError(
            f"{where}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}; rebuild the advice document"
        )


def _decode(payload: bytes) -> Advice:
    document: Advice = json.loads(payload)
    _keys(document, _DOCUMENT_KEYS, "advice")
    if document["format"] != "kd-experience-advice-v1":
        raise ValueError(
            "advice format must be kd-experience-advice-v1; rebuild the document"
        )
    if type(document["version"]) is not int or document["version"] != 1:
        raise ValueError("advice version must be 1; rebuild the document")
    _keys(
        document["basis"],
        frozenset({"matrices", "rules_version", "excluded_groups"}),
        "basis",
    )
    for group in ("usage", "cost_anchors"):
        for item in document[group]:
            _keys(
                item,
                _ITEM_KEYS | ({"measured"} if group == "cost_anchors" else set()),
                group,
            )
            if item["mode"] is not None and item["mode"] not in _MODES:
                raise ValueError(
                    f"{group}: unknown mode {item['mode']!r}; use a declared mode"
                )
            if item["task_kind"] is not None and item["task_kind"] not in _TASK_KINDS:
                raise ValueError(
                    f"{group}: unknown task_kind {item['task_kind']!r}; use a supported task kind"
                )
            if group == "cost_anchors":
                measured = item["measured"]
                _keys(measured, frozenset({"shape", "device", "matrix"}), "measured")

                _keys(measured["shape"], _SHAPE_KEYS, "shape", required=False)
    return document


def load_advice(path: str | Path) -> Advice:
    return _decode(Path(path).read_bytes())


def prime(path: str | Path, workspace: str | Path) -> tuple[dict[str, str], Advice]:
    path, workspace = Path(path), Path(workspace)
    if path.resolve().is_relative_to(workspace.resolve()):
        raise ValueError(
            f"keep advice outside the workspace {workspace}: move {path} before starting the run"
        )
    payload = path.read_bytes()
    return {
        "advice_path": str(path),
        "advice_sha256": hashlib.sha256(payload).hexdigest(),
    }, _decode(payload)
