
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BUDGET_FORMAT = "kd-budget-v1"
BUDGET_FILENAME = "budget.jsonl"

_EVENTS = frozenset({"open", "reserve", "settle", "charge"})


class BudgetExhausted(RuntimeError):
    pass


def price_seconds(seconds: float, *, seconds_per_evaluation: float) -> int:
    if seconds < 0:
        raise ValueError(f"seconds must be >= 0, got {seconds!r}")
    if seconds_per_evaluation <= 0:
        raise ValueError(
            "seconds_per_evaluation must be positive, got "
            f"{seconds_per_evaluation!r}"
        )
    return math.ceil(seconds / seconds_per_evaluation)


@dataclass(frozen=True)
class Reservation:

    label: str
    kind: str
    cap: int


class EvaluationPocket:

    def __init__(self, workspace: Path, *, total: int | None = None) -> None:
        self._workspace = Path(workspace)
        self._path = self._workspace / BUDGET_FILENAME
        self._total: int | None = None
        self._spent = 0
        self._outstanding: dict[str, Reservation] = {}
        self._settled: dict[str, dict[str, Any]] = {}
        self._charges: dict[str, dict[str, Any]] = {}
        if self._path.is_file():
            self._replay()
        if total is not None:
            if self._total is None:
                self._require_positive(total, "total")
                self._append({"event": "open", "total": int(total)})
                self._total = int(total)
            elif int(total) != self._total:
                raise ValueError(
                    f"{self._path} was opened with total={self._total}; a pocket "
                    f"cannot be reopened with total={total}. Use a new workspace "
                    "for a new budget"
                )
        if self._total is None:
            raise ValueError(f"{self._path} has no budget yet: pass total= to open one")



    @property
    def total(self) -> int:
        assert self._total is not None
        return self._total

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def reserved(self) -> int:
        return sum(r.cap for r in self._outstanding.values())

    @property
    def remaining(self) -> int:
        return self.total - self._spent - self.reserved

    @property
    def outstanding(self) -> tuple[Reservation, ...]:
        return tuple(self._outstanding.values())

    def settled(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._settled.values())

    def charges(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._charges.values())



    def reserve(
        self,
        label: str,
        *,
        kind: str = "segment",
        request: int | None = None,
        minimum: int = 1,
    ) -> Reservation:
        if not label:
            raise ValueError("a reservation needs a non-empty label")
        if self._label_taken(label):
            raise ValueError(
                f"label {label!r} is already used by this pocket; "
                "labels are unique so a settle cannot be misfiled"
            )
        self._require_positive(minimum, "minimum")
        if request is not None:
            self._require_positive(request, "request")
        remaining = self.remaining
        if remaining < minimum:
            raise BudgetExhausted(
                f"cannot reserve {label!r}: {remaining} of {self.total} "
                f"evaluations remain (spent {self._spent}, outstanding "
                f"{self.reserved}), below the minimum of {minimum}"
            )
        cap = remaining if request is None else min(int(request), remaining)
        reservation = Reservation(label=label, kind=kind, cap=cap)
        self._append(
            {
                "event": "reserve",
                "label": label,
                "kind": kind,
                "cap": cap,
                "remaining_after": remaining - cap,
            }
        )
        self._outstanding[label] = reservation
        return reservation

    def settle(
        self,
        reservation: Reservation | str,
        *,
        evaluations: int | None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        label = reservation if isinstance(reservation, str) else reservation.label
        if label not in self._outstanding:
            raise KeyError(
                f"cannot settle {label!r}: not an outstanding reservation "
                f"(outstanding: {list(self._outstanding)})"
            )
        held = self._outstanding[label]
        if evaluations is not None and int(evaluations) < 0:
            raise ValueError(f"evaluations must be >= 0, got {evaluations}")
        charged = held.cap if evaluations is None else int(evaluations)
        row = {
            "event": "settle",
            "label": label,
            "kind": held.kind,
            "cap": held.cap,
            "run_id": run_id,
            "evaluations": None if evaluations is None else int(evaluations),
            "charged": charged,
            "remaining_after": self.remaining + held.cap - charged,
        }
        self._append(row)
        del self._outstanding[label]
        self._spent += charged
        self._settled[label] = row
        return row

    def charge(
        self,
        label: str,
        *,
        evaluations: int,
        kind: str = "surrogate_training",
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not label:
            raise ValueError("a charge needs a non-empty label")
        if self._label_taken(label):
            raise ValueError(
                f"label {label!r} is already used by this pocket; "
                "labels are unique so a settle cannot be misfiled"
            )
        self._require_positive(evaluations, "evaluations")
        row: dict[str, Any] = {
            "event": "charge",
            "label": label,
            "kind": kind,
            "evaluations": int(evaluations),
            "charged": int(evaluations),
            "detail": detail,
            "remaining_after": self.remaining - int(evaluations),
        }
        self._append(row)
        self._spent += int(evaluations)
        self._charges[label] = row
        return row

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "spent": self._spent,
            "reserved": self.reserved,
            "remaining": self.remaining,
            "outstanding": [r.__dict__ for r in self._outstanding.values()],
            "settled": list(self._settled.values()),
            "charges": list(self._charges.values()),
        }



    def _label_taken(self, label: str) -> bool:
        return (
            label in self._outstanding
            or label in self._settled
            or label in self._charges
        )

    @staticmethod
    def _require_positive(value: Any, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    def _append(self, event: dict[str, Any]) -> None:
        line = json.dumps(
            {
                "format": BUDGET_FORMAT,
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                **event,
            },
            ensure_ascii=False,
        )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _replay(self) -> None:
        with self._path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
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
        if tag != BUDGET_FORMAT:
            raise ValueError(
                f"{where}format {tag!r} is not the {BUDGET_FORMAT!r} this pocket writes"
            )
        event = payload.get("event")
        if event not in _EVENTS:
            raise ValueError(f"{where}event {event!r} is none of {sorted(_EVENTS)}")
        if event == "open":
            if self._total is not None:
                raise ValueError(f"{where}a second open event")
            self._total = int(payload["total"])
            return
        if self._total is None:
            raise ValueError(f"{where}{event} before the open event")
        label = payload["label"]
        if event == "charge":
            if label in self._charges:
                raise ValueError(f"{where}label {label!r} charged twice")
            row = {
                key: value
                for key, value in payload.items()
                if key not in {"format", "at"}
            }
            self._spent += int(payload["charged"])
            self._charges[label] = row
            return
        if event == "reserve":
            if label in self._outstanding or label in self._settled:
                raise ValueError(f"{where}label {label!r} reserved twice")
            self._outstanding[label] = Reservation(
                label=label, kind=payload["kind"], cap=int(payload["cap"])
            )
            return
        if label not in self._outstanding:
            raise ValueError(f"{where}settle for {label!r} without a reservation")
        row = {
            key: value for key, value in payload.items() if key not in {"format", "at"}
        }
        del self._outstanding[label]
        self._spent += int(payload["charged"])
        self._settled[label] = row
