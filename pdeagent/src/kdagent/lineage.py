
from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kd.core.equation import Form, LawSignature, LhsSpec

TREE_FILENAME = "tree.jsonl"

TREE_FORMAT = "kd-segtree-v1"

PLATFORM_FRAME = "kd-evaluate-terms-full-grid-v1"

_EVENTS = ("segment", "prune", "select")


def signature_from_dict(payload: dict[str, Any]) -> LawSignature:
    native_lhs = payload["native_lhs"]
    return LawSignature(
        version=payload["version"],
        structure_key=payload["structure_key"],
        terms=tuple(payload["terms"]),
        coefficients=tuple(payload["coefficients"]),
        native_form=Form(payload["native_form"]),
        native_lhs=(
            None
            if native_lhs is None
            else LhsSpec(
                field=native_lhs["field"],
                axis=native_lhs["axis"],
                order=native_lhs["order"],
            )
        ),
    )


@dataclass(frozen=True)
class SegmentSummary:

    best_score: float | None
    segment_best: float | None
    score_direction: str
    iterations_without_improvement: int
    segments_without_improvement: int
    signature: LawSignature | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_score": self.best_score,
            "segment_best": self.segment_best,
            "score_direction": self.score_direction,
            "iterations_without_improvement": self.iterations_without_improvement,
            "segments_without_improvement": self.segments_without_improvement,
            "signature": (None if self.signature is None else self.signature.to_dict()),
        }


def _summary_from_dict(payload: dict[str, Any]) -> SegmentSummary:
    signature = payload["signature"]
    return SegmentSummary(
        best_score=payload["best_score"],
        segment_best=payload["segment_best"],
        score_direction=payload["score_direction"],
        iterations_without_improvement=payload["iterations_without_improvement"],
        segments_without_improvement=payload["segments_without_improvement"],
        signature=None if signature is None else signature_from_dict(signature),
    )


@dataclass(frozen=True)
class SegmentEntry:

    run_id: str
    run_dir: str
    parent_run_id: str | None
    resume_from: dict[str, str | int] | None
    instrument: str
    dataset: str
    seed: int
    status: str
    nmse: float | None
    created_at: str
    summary: SegmentSummary | None
    reseed: bool = False
    decision_parent: str | None = None
    action: str = "fresh"
    params: dict[str, Any] | None = None
    law: dict[str, Any] | None = None
    report_projection: dict[str, Any] | None = None
    sketch: dict[str, Any] | None = None
    surrogate: str | None = None
    platform_nmse: float | None = None
    platform_frame: str | None = None
    platform_coefficients: list[float] | None = None
    cost: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["summary"] = None if self.summary is None else self.summary.to_dict()
        return payload


def _entry_from_dict(payload: dict[str, Any]) -> SegmentEntry:
    summary = payload["summary"]
    return SegmentEntry(
        run_id=payload["run_id"],
        run_dir=payload["run_dir"],
        parent_run_id=payload["parent_run_id"],
        resume_from=payload["resume_from"],
        instrument=payload["instrument"],
        dataset=payload["dataset"],
        seed=payload["seed"],
        status=payload["status"],
        nmse=payload["nmse"],
        created_at=payload["created_at"],
        summary=None if summary is None else _summary_from_dict(summary),
        reseed=payload["reseed"],
        decision_parent=payload["decision_parent"],
        action=payload["action"],
        params=payload["params"],
        law=payload["law"],
        report_projection=payload["report_projection"],


        sketch=payload.get("sketch"),
        surrogate=payload.get("surrogate"),
        platform_nmse=payload.get("platform_nmse"),
        platform_frame=payload.get("platform_frame"),
        platform_coefficients=payload.get("platform_coefficients"),
        cost=payload.get("cost"),
    )


class SegmentLedger:

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._path = workspace / TREE_FILENAME
        self._entries: dict[str, SegmentEntry] = {}
        self._pruned: dict[str, str] = {}
        self._selected: tuple[str, str] | None = None
        if self._path.is_file():
            self._replay()

    def record(
        self,
        *,
        run_id: str,
        run_dir: str,
        parent_run_id: str | None,
        resume_from: dict[str, str | int] | None,
        instrument: str,
        dataset: str,
        seed: int,
        status: str,
        nmse: float | None,
        created_at: str,
        summary: SegmentSummary | None,
        reseed: bool = False,
        decision_parent: str | None = None,
        action: str = "fresh",
        params: dict[str, Any] | None = None,
        law: dict[str, Any] | None = None,
        report_projection: dict[str, Any] | None = None,
        sketch: dict[str, Any] | None = None,
        surrogate: str | None = None,
        platform_nmse: float | None = None,
        platform_frame: str | None = None,
        platform_coefficients: list[float] | None = None,
        cost: dict[str, Any] | None = None,
    ) -> SegmentEntry:
        entry = SegmentEntry(
            run_id=run_id,
            run_dir=run_dir,
            parent_run_id=parent_run_id,
            resume_from=deepcopy(resume_from),
            instrument=instrument,
            dataset=dataset,
            seed=seed,
            status=status,
            nmse=(nmse if nmse is None or math.isfinite(nmse) else None),
            created_at=created_at,
            summary=summary,
            reseed=reseed,
            decision_parent=decision_parent,
            action=action,




            params=deepcopy(params) or None,
            law=deepcopy(law),
            report_projection=deepcopy(report_projection),
            sketch=deepcopy(sketch),
            surrogate=surrogate,
            platform_nmse=(
                platform_nmse
                if platform_nmse is None or math.isfinite(platform_nmse)
                else None
            ),
            platform_frame=platform_frame,
            platform_coefficients=(
                None if platform_coefficients is None else list(platform_coefficients)
            ),
            cost=deepcopy(cost),
        )
        self._require_not_recorded(run_id)
        self._require_parent(entry)
        self._require_decision_parent(entry)
        self._append({"event": "segment", **entry.to_dict()})
        self._entries[run_id] = entry
        return entry

    def prune(self, run_id: str, *, reason: str) -> None:
        self._require_recorded(run_id, "prune")
        self._append({"event": "prune", "run_id": run_id, "reason": reason})
        self._pruned[run_id] = reason

    def select(self, run_id: str, *, reason: str) -> None:
        self._require_recorded(run_id, "select")
        if self.is_pruned(run_id):
            raise ValueError(
                f"segment {run_id!r} is pruned and cannot be selected; prune "
                "removed it from the running, so selecting it would publish an "
                "abandoned result"
            )
        self._append({"event": "select", "run_id": run_id, "reason": reason})
        self._selected = (run_id, reason)

    def is_pruned(self, run_id: str) -> bool:
        if run_id not in self._entries:
            return False
        current: str | None = run_id
        while current is not None:
            if current in self._pruned:
                return True
            current = self._entries[current].parent_run_id
        return False

    @property
    def selected(self) -> str | None:
        if self._selected is None:
            return None
        run_id = self._selected[0]
        return None if self.is_pruned(run_id) else run_id

    def entry(self, run_id: str) -> SegmentEntry:
        return self._entries[run_id]

    def segments(self) -> tuple[SegmentEntry, ...]:
        return tuple(self._entries.values())

    def children(self, run_id: str) -> tuple[SegmentEntry, ...]:
        return tuple(
            entry for entry in self._entries.values() if entry.parent_run_id == run_id
        )

    def roots(self) -> tuple[SegmentEntry, ...]:
        return tuple(
            entry for entry in self._entries.values() if entry.parent_run_id is None
        )

    def chain(self, run_id: str) -> tuple[SegmentEntry, ...]:
        current = self._entries[run_id]
        reversed_chain = [current]
        while current.parent_run_id is not None:
            current = self._entries[current.parent_run_id]
            reversed_chain.append(current)
        return tuple(reversed(reversed_chain))

    def lineage_cost(self, run_id: str) -> dict[str, Any]:
        evaluations = 0
        search_seconds = 0.0
        unknown = 0
        chain = self.chain(run_id)
        for entry in chain:
            cost = entry.cost
            if cost is None or cost.get("evaluations") is None:
                unknown += 1
                continue
            evaluations += int(cost["evaluations"])
            search_seconds += float(cost.get("search_seconds") or 0.0)
        return {
            "evaluations": evaluations,
            "search_seconds": search_seconds,
            "segments": len(chain),
            "unknown": unknown,
        }

    def resumable_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.run_id
            for entry in self._entries.values()
            if (self._workspace / entry.run_dir).is_dir()
        )

    def to_dict(self) -> dict[str, Any]:
        selected: dict[str, Any] | None = None
        if self._selected is not None and not self.is_pruned(self._selected[0]):
            selected = {"run_id": self._selected[0], "reason": self._selected[1]}
        return {
            "segments": [entry.to_dict() for entry in self._entries.values()],
            "pruned": [
                {"run_id": run_id, "reason": reason}
                for run_id, reason in self._pruned.items()
            ],
            "selected": selected,
        }

    def _append(self, event: dict[str, Any]) -> None:
        line = json.dumps({"format": TREE_FORMAT, **event}, ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _require_recorded(self, run_id: str, event: str) -> None:
        if run_id not in self._entries:
            raise KeyError(
                f"cannot {event} run {run_id!r}: this ledger has no such "
                f"segment; the recorded ones are {list(self._entries)}"
            )

    def _require_not_recorded(self, run_id: str, where: str = "") -> None:
        if run_id in self._entries:
            raise ValueError(
                f"{where}segment {run_id!r} is already on this ledger; a run "
                "directory is filed once, and a second segment event for it "
                "would overwrite the row (and un-prune a pruned lineage)"
            )

    def _require_parent(self, entry: SegmentEntry, where: str = "") -> None:
        parent = entry.parent_run_id
        if parent is not None and parent not in self._entries:
            raise ValueError(
                f"{where}segment {entry.run_id!r} resumes from {parent!r}, "
                "which no segment event recorded"
            )

    def _require_decision_parent(self, entry: SegmentEntry, where: str = "") -> None:
        decision_parent = entry.decision_parent
        if decision_parent is not None and decision_parent not in self._entries:
            raise ValueError(
                f"{where}segment {entry.run_id!r} answers to "
                f"{decision_parent!r}, which no segment event recorded"
            )

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
            self._apply(payload, where)

    def _apply(self, payload: dict[str, Any], where: str) -> None:
        tag = payload.get("format")
        if tag != TREE_FORMAT:
            raise ValueError(
                f"{where}format {tag!r} is not the {TREE_FORMAT!r} this ledger writes"
            )
        event = payload.get("event")
        if event == "segment":
            entry = _entry_from_dict(payload)
            self._require_not_recorded(entry.run_id, where)
            self._require_parent(entry, where)
            self._require_decision_parent(entry, where)
            self._entries[entry.run_id] = entry
            return
        if event not in _EVENTS:
            raise ValueError(f"{where}event {event!r} is none of {list(_EVENTS)}")
        run_id = payload["run_id"]
        if run_id not in self._entries:
            raise ValueError(
                f"{where}{event} names run {run_id!r}, which no segment event recorded"
            )
        if event == "prune":
            self._pruned[run_id] = payload["reason"]
        else:
            self._selected = (run_id, payload["reason"])
