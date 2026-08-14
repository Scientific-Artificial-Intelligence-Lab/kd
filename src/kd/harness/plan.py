
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Final

from kd.core.strict_keys import strict_keys as _strict_keys_core
from kd.search.records import StrictDecodeError
from kd.search.run_spec import canonicalize_config

PLAN_SCHEMA_VERSION: Final[int] = 1
PLAN_HASH_SCHEME: Final[str] = "kd-plan-v1"




_RESERVED_MODEL_KWARGS: Final[frozenset[str]] = frozenset(
    {
        "algorithm", "seed", "config", "callbacks", "provider",
        "surrogate_model", "checkpoint_dir", "checkpoint_every",
        "checkpoint_keep_last", "phases_path", "verbose", "device",
    }
)












_PLAN_V1_FIELDS: Final[frozenset[str]] = frozenset(
    {"plan_schema_version", "name", "entries"}
)
_PLAN_V1_ENTRY_FIELDS: Final[frozenset[str]] = frozenset(
    {"instrument", "dataset_ref", "seed", "model_kwargs"}
)


def _strict_keys(
    data: dict[str, Any], *, object_name: str, required: frozenset[str]
) -> None:
    _strict_keys_core(
        data,
        object_name=object_name,
        required=required,
        error_cls=StrictDecodeError,
    )


def _as_dict(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StrictDecodeError(f"{field} must be an object")
    return value


@dataclass(frozen=True, kw_only=True)
class PlanEntry:

    instrument: str
    dataset_ref: str
    seed: int
    model_kwargs: dict[str, Any]

    def __post_init__(self) -> None:
        for field in ("instrument", "dataset_ref"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"PlanEntry.{field} must be a non-empty str; got {value!r}"
                )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(
                f"PlanEntry.seed must be an int (bool rejected); got {self.seed!r}"
            )
        if not isinstance(self.model_kwargs, dict):
            raise ValueError(
                "PlanEntry.model_kwargs must be a dict; "
                f"got {type(self.model_kwargs).__name__}"
            )
        reserved = _RESERVED_MODEL_KWARGS & self.model_kwargs.keys()
        if reserved:
            raise ValueError(
                "PlanEntry.model_kwargs must not contain reserved key(s): "
                f"{sorted(reserved)!r}"
            )




        object.__setattr__(
            self, "model_kwargs", canonicalize_config(self.model_kwargs)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "dataset_ref": self.dataset_ref,
            "seed": self.seed,
            "model_kwargs": deepcopy(self.model_kwargs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanEntry:
        _strict_keys(data, object_name="plan entry", required=_PLAN_V1_ENTRY_FIELDS)
        return cls(
            instrument=data["instrument"],
            dataset_ref=data["dataset_ref"],
            seed=data["seed"],
            model_kwargs=_as_dict(
                data["model_kwargs"], field="plan entry model_kwargs"
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ExperimentPlan:

    name: str
    entries: tuple[PlanEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"ExperimentPlan.name must be a non-empty str; got {self.name!r}"
            )
        if type(self.entries) is not tuple:
            raise ValueError(
                "ExperimentPlan.entries must be a tuple; "
                f"got {type(self.entries).__name__}"
            )
        if not self.entries:
            raise ValueError("ExperimentPlan.entries must be non-empty")
        for index, entry in enumerate(self.entries):
            if not isinstance(entry, PlanEntry):
                raise ValueError(
                    f"ExperimentPlan.entries[{index}] must be a PlanEntry; "
                    f"got {type(entry).__name__}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_schema_version": PLAN_SCHEMA_VERSION,
            "name": self.name,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentPlan:
        if not isinstance(data, dict):
            raise StrictDecodeError("plan must be an object")
        version = data.get("plan_schema_version")
        if type(version) is not int or version != PLAN_SCHEMA_VERSION:


            raise StrictDecodeError(
                f"Unsupported plan_schema_version: got {version!r}; "
                f"supported versions: {[PLAN_SCHEMA_VERSION]!r}"
            )
        _strict_keys(data, object_name="plan", required=_PLAN_V1_FIELDS)
        entries_data = data["entries"]
        if not isinstance(entries_data, list):
            raise StrictDecodeError("plan.entries must be an array")
        entries = tuple(
            PlanEntry.from_dict(_as_dict(item, field=f"plan.entries[{index}]"))
            for index, item in enumerate(entries_data)
        )
        return cls(name=data["name"], entries=entries)

    def plan_hash(self) -> str:
        return _plan_v1_hash(self)


def _plan_v1_payload(plan: ExperimentPlan) -> dict[str, Any]:






    full_payload = plan.to_dict()
    payload = {field: full_payload[field] for field in _PLAN_V1_FIELDS}
    payload["entries"] = [
        {field: entry_dict[field] for field in _PLAN_V1_ENTRY_FIELDS}
        for entry_dict in full_payload["entries"]
    ]
    return payload


def _plan_v1_hash(plan: ExperimentPlan) -> str:
    canonical = json.dumps(
        _plan_v1_payload(plan),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    hash_input = f"{PLAN_HASH_SCHEME}:{canonical}"
    hashed_bytes = hash_input.encode("utf-8")
    return hashlib.sha256(hashed_bytes).hexdigest()


__all__ = [
    "PLAN_HASH_SCHEME",
    "PLAN_SCHEMA_VERSION",
    "ExperimentPlan",
    "PlanEntry",
]
