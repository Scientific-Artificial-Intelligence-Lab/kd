
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Final

from kd.harness.plan import ExperimentPlan
from kd.search.run_spec import canonicalize_config

DISPATCH_ARTIFACT_TAG: Final[str] = "kd-dispatch-v1"
DISPATCH_SCHEMA_VERSION: Final[int] = 1

_DISPATCH_V1_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "dispatch_schema_version",
        "plan",
        "plan_hash",
        "shards",
        "datasets",
        "resources",
    }
)
_DISPATCH_V1_SHARD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "shard_id",
        "entry_indices",
        "cuda_visible_devices",
        "device",
        "heavy",
        "timeout_seconds",
        "memory_max_gb",
    }
)
_DISPATCH_V1_DATASET_KEYS: Final[frozenset[str]] = frozenset({"loader", "kwargs"})
_DISPATCH_V1_RESOURCES_KEYS: Final[frozenset[str]] = frozenset(
    {"max_concurrent_heavy", "grace_seconds"}
)


SHARD_ID_RE: Final[re.Pattern[str]] = re.compile(r"^shard-\d{2,}$")


class DispatchManifestError(ValueError):
    pass


def _strict_keys(
    data: Any, *, object_name: str, required: frozenset[str]
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise DispatchManifestError(f"{object_name} must be a JSON object")
    actual = frozenset(data)
    unknown = actual - required
    if unknown:
        keys = ", ".join(repr(key) for key in sorted(unknown))
        raise DispatchManifestError(f"Unknown {object_name} field(s): {keys}")
    missing = required - actual
    if missing:
        keys = ", ".join(repr(key) for key in sorted(missing))
        raise DispatchManifestError(f"Missing required {object_name} field(s): {keys}")
    return data


def _require_finite_positive(value: Any, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DispatchManifestError(f"{field} must be a real number; got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise DispatchManifestError(
            f"{field} must be finite and positive; got {value!r}"
        )


@dataclass(frozen=True, kw_only=True)
class DispatchDatasetSpec:

    loader: str
    kwargs: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.loader, str) or not self.loader:
            raise DispatchManifestError(
                "DispatchDatasetSpec.loader must be a non-empty str; "
                f"got {self.loader!r}"
            )
        if not isinstance(self.kwargs, dict):
            raise DispatchManifestError(
                "DispatchDatasetSpec.kwargs must be a dict; "
                f"got {type(self.kwargs).__name__}"
            )
        object.__setattr__(self, "kwargs", canonicalize_config(self.kwargs))


@dataclass(frozen=True, kw_only=True)
class ShardSpec:

    shard_id: str
    entry_indices: tuple[int, ...]
    cuda_visible_devices: str
    device: str | None
    heavy: bool
    timeout_seconds: float | None
    memory_max_gb: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.shard_id, str) or not SHARD_ID_RE.match(self.shard_id):
            raise DispatchManifestError(
                f"shard_id must match '^shard-\\d{{2,}}$'; got {self.shard_id!r}"
            )
        if type(self.entry_indices) is not tuple or not self.entry_indices:
            raise DispatchManifestError(
                f"shard {self.shard_id!r} entry_indices must be a non-empty tuple"
            )
        previous: int | None = None
        for index in self.entry_indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise DispatchManifestError(
                    f"shard {self.shard_id!r} entry_indices must be ints "
                    f"(bool rejected); got {index!r}"
                )
            if index < 0:
                raise DispatchManifestError(
                    f"shard {self.shard_id!r} entry_indices must be non-negative; "
                    f"got {index!r}"
                )
            if previous is not None and index <= previous:
                raise DispatchManifestError(
                    f"shard {self.shard_id!r} entry_indices must be strictly "
                    f"increasing; got {self.entry_indices!r}"
                )
            previous = index
        if not isinstance(self.cuda_visible_devices, str):
            raise DispatchManifestError(
                f"shard {self.shard_id!r} cuda_visible_devices must be a str"
            )
        if self.device is not None and (
            not isinstance(self.device, str) or not self.device
        ):
            raise DispatchManifestError(
                f"shard {self.shard_id!r} device must be null or a non-empty str"
            )
        if not isinstance(self.heavy, bool):
            raise DispatchManifestError(f"shard {self.shard_id!r} heavy must be a bool")
        if self.timeout_seconds is not None:
            _require_finite_positive(
                self.timeout_seconds,
                field=f"shard {self.shard_id!r} timeout_seconds",
            )
        if self.memory_max_gb is not None:
            _require_finite_positive(
                self.memory_max_gb, field=f"shard {self.shard_id!r} memory_max_gb"
            )


@dataclass(frozen=True, kw_only=True)
class DispatchResources:

    max_concurrent_heavy: int
    grace_seconds: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_concurrent_heavy, bool)
            or type(self.max_concurrent_heavy) is not int
            or self.max_concurrent_heavy < 1
        ):
            raise DispatchManifestError(
                "resources max_concurrent_heavy must be an int >= 1; "
                f"got {self.max_concurrent_heavy!r}"
            )
        _require_finite_positive(self.grace_seconds, field="resources grace_seconds")


@dataclass(frozen=True, kw_only=True)
class DispatchManifest:

    plan: ExperimentPlan
    plan_hash: str
    shards: tuple[ShardSpec, ...]
    datasets: dict[str, DispatchDatasetSpec]
    resources: DispatchResources

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ExperimentPlan):
            raise DispatchManifestError("manifest plan must be an ExperimentPlan")
        if not isinstance(self.plan_hash, str) or not self.plan_hash:
            raise DispatchManifestError("manifest plan_hash must be a non-empty str")
        recomputed = self.plan.plan_hash()
        if self.plan_hash != recomputed:
            raise DispatchManifestError(
                f"plan_hash mismatch: manifest {self.plan_hash!r} != recomputed "
                f"{recomputed!r}"
            )
        if type(self.shards) is not tuple or not self.shards:
            raise DispatchManifestError("manifest shards must be a non-empty tuple")
        seen_ids: set[str] = set()
        n_entries = len(self.plan.entries)
        seen_indices: set[int] = set()
        for shard in self.shards:
            if not isinstance(shard, ShardSpec):
                raise DispatchManifestError("each shard must be a ShardSpec")
            if shard.shard_id in seen_ids:
                raise DispatchManifestError(f"duplicate shard_id {shard.shard_id!r}")
            seen_ids.add(shard.shard_id)
            for index in shard.entry_indices:
                if not 0 <= index < n_entries:
                    raise DispatchManifestError(
                        f"shard {shard.shard_id!r} entry_indices value {index} out "
                        f"of range for plan with {n_entries} entries"
                    )
                if index in seen_indices:
                    raise DispatchManifestError(
                        f"overlapping entry_indices across shards: {index} "
                        f"appears more than once"
                    )
                seen_indices.add(index)
        if not isinstance(self.datasets, dict):
            raise DispatchManifestError("manifest datasets must be a mapping")
        expected_refs = {entry.dataset_ref for entry in self.plan.entries}
        actual_refs = set(self.datasets)
        if actual_refs != expected_refs:
            unknown = sorted(actual_refs - expected_refs)
            missing = sorted(expected_refs - actual_refs)
            raise DispatchManifestError(
                f"datasets key set must equal plan dataset refs "
                f"(unknown={unknown!r}, missing={missing!r})"
            )
        for ref, spec in self.datasets.items():
            if not isinstance(spec, DispatchDatasetSpec):
                raise DispatchManifestError(
                    f"dataset {ref!r} must be a DispatchDatasetSpec"
                )
        if not isinstance(self.resources, DispatchResources):
            raise DispatchManifestError(
                "manifest resources must be a DispatchResources"
            )


def _shard_payload(shard: ShardSpec) -> dict[str, Any]:
    return {
        "shard_id": shard.shard_id,
        "entry_indices": list(shard.entry_indices),
        "cuda_visible_devices": shard.cuda_visible_devices,
        "device": shard.device,
        "heavy": shard.heavy,
        "timeout_seconds": shard.timeout_seconds,
        "memory_max_gb": shard.memory_max_gb,
    }


def manifest_to_payload(manifest: DispatchManifest) -> dict[str, Any]:
    return {
        "artifact": DISPATCH_ARTIFACT_TAG,
        "dispatch_schema_version": DISPATCH_SCHEMA_VERSION,
        "plan": manifest.plan.to_dict(),
        "plan_hash": manifest.plan_hash,
        "shards": [_shard_payload(shard) for shard in manifest.shards],
        "datasets": {
            ref: {"loader": spec.loader, "kwargs": dict(spec.kwargs)}
            for ref, spec in manifest.datasets.items()
        },
        "resources": {
            "max_concurrent_heavy": manifest.resources.max_concurrent_heavy,
            "grace_seconds": manifest.resources.grace_seconds,
        },
    }


def _decode_shard(payload: Any) -> ShardSpec:
    data = _strict_keys(payload, object_name="shard", required=_DISPATCH_V1_SHARD_KEYS)
    indices = data["entry_indices"]
    if not isinstance(indices, list):
        raise DispatchManifestError("shard entry_indices must be a JSON array")
    return ShardSpec(
        shard_id=data["shard_id"],
        entry_indices=tuple(indices),
        cuda_visible_devices=data["cuda_visible_devices"],
        device=data["device"],
        heavy=data["heavy"],
        timeout_seconds=data["timeout_seconds"],
        memory_max_gb=data["memory_max_gb"],
    )


def _decode_dataset(payload: Any) -> DispatchDatasetSpec:
    data = _strict_keys(
        payload, object_name="dataset", required=_DISPATCH_V1_DATASET_KEYS
    )
    return DispatchDatasetSpec(loader=data["loader"], kwargs=data["kwargs"])


def _decode_resources(payload: Any) -> DispatchResources:
    data = _strict_keys(
        payload, object_name="resources", required=_DISPATCH_V1_RESOURCES_KEYS
    )
    return DispatchResources(
        max_concurrent_heavy=data["max_concurrent_heavy"],
        grace_seconds=data["grace_seconds"],
    )


def decode_manifest_payload(payload: Any) -> DispatchManifest:
    _strict_keys(payload, object_name="dispatch manifest", required=_DISPATCH_V1_KEYS)
    if payload["artifact"] != DISPATCH_ARTIFACT_TAG:
        raise DispatchManifestError(
            f"artifact tag mismatch: got {payload['artifact']!r}, "
            f"expected {DISPATCH_ARTIFACT_TAG!r}"
        )
    version = payload["dispatch_schema_version"]
    if type(version) is not int or version != DISPATCH_SCHEMA_VERSION:
        raise DispatchManifestError(
            f"unsupported dispatch_schema_version: got {version!r}; "
            f"supported: {[DISPATCH_SCHEMA_VERSION]!r}"
        )
    plan_payload = payload["plan"]
    if not isinstance(plan_payload, dict):
        raise DispatchManifestError("manifest plan must be a JSON object")
    try:
        plan = ExperimentPlan.from_dict(plan_payload)
    except Exception as exc:
        raise DispatchManifestError(f"undecodable embedded plan: {exc}") from exc

    shards_payload = payload["shards"]
    if not isinstance(shards_payload, list) or not shards_payload:
        raise DispatchManifestError("manifest shards must be a non-empty JSON array")
    shards = tuple(_decode_shard(item) for item in shards_payload)

    datasets_payload = payload["datasets"]
    if not isinstance(datasets_payload, dict):
        raise DispatchManifestError("manifest datasets must be a JSON object")
    datasets = {ref: _decode_dataset(spec) for ref, spec in datasets_payload.items()}

    resources = _decode_resources(payload["resources"])
    return DispatchManifest(
        plan=plan,
        plan_hash=payload["plan_hash"],
        shards=shards,
        datasets=datasets,
        resources=resources,
    )
