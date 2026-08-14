
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from kd.core.strict_keys import strict_keys as _strict_keys_core
from kd.harness.plan import ExperimentPlan
from kd.harness.recording import RecordingOptions
from kd.search.run_spec import canonicalize_config

DISPATCH_ARTIFACT_TAG: Final[str] = "kd-dispatch-v1"





DISPATCH_SCHEMA_VERSION: Final[int] = 2

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
_DISPATCH_V2_KEYS: Final[frozenset[str]] = _DISPATCH_V1_KEYS | {"recording"}
_DISPATCH_KEYS_BY_VERSION: Final[dict[int, frozenset[str]]] = {
    1: _DISPATCH_V1_KEYS,
    2: _DISPATCH_V2_KEYS,
}
_DISPATCH_V2_RECORDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "events_every_n",
        "checkpoint_every",
        "checkpoint_keep_last",
        "phases",
        "catalog",
        "resume_from",
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
    _strict_keys_core(
        data,
        object_name=object_name,
        required=required,
        error_cls=DispatchManifestError,
    )
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
class DispatchRecording:

    options: RecordingOptions
    catalog: str | None = None
    resume_from: dict[int, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.options, RecordingOptions):
            raise DispatchManifestError(
                "recording options must be a RecordingOptions; "
                f"got {type(self.options).__name__}"
            )
        if self.catalog is not None:
            if not isinstance(self.catalog, str) or not self.catalog:
                raise DispatchManifestError(
                    f"recording catalog must be null or a non-empty str; "
                    f"got {self.catalog!r}"
                )
            if Path(self.catalog).is_absolute():
                raise DispatchManifestError(
                    "recording catalog must be relative to the batch root "
                    f"(manifest relocatability); got {self.catalog!r}"
                )
        if self.resume_from is not None:
            if not isinstance(self.resume_from, dict):
                raise DispatchManifestError(
                    "recording resume_from must be null or an object"
                )
            for index, path in self.resume_from.items():
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise DispatchManifestError(
                        "recording resume_from keys must be non-negative ints "
                        f"(bool rejected); got {index!r}"
                    )
                if not isinstance(path, str) or not path:
                    raise DispatchManifestError(
                        f"recording resume_from[{index}] must be a non-empty "
                        f"str; got {path!r}"
                    )


@dataclass(frozen=True, kw_only=True)
class DispatchManifest:

    plan: ExperimentPlan
    plan_hash: str
    shards: tuple[ShardSpec, ...]
    datasets: dict[str, DispatchDatasetSpec]
    resources: DispatchResources
    recording: DispatchRecording | None = None

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
        if self.recording is not None:
            if not isinstance(self.recording, DispatchRecording):
                raise DispatchManifestError(
                    "manifest recording must be a DispatchRecording or None"
                )
            if self.recording.resume_from is not None:
                for index in self.recording.resume_from:
                    if index not in seen_indices:
                        raise DispatchManifestError(
                            f"recording resume_from index {index} is not covered "
                            "by any shard entry_indices"
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


def _recording_payload(recording: DispatchRecording | None) -> dict[str, Any] | None:
    if recording is None:
        return None
    return {
        "events_every_n": recording.options.events_every_n,
        "checkpoint_every": recording.options.checkpoint_every,
        "checkpoint_keep_last": recording.options.checkpoint_keep_last,
        "phases": recording.options.phases,
        "catalog": recording.catalog,

        "resume_from": (
            None
            if recording.resume_from is None
            else {str(k): v for k, v in recording.resume_from.items()}
        ),
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
        "recording": _recording_payload(manifest.recording),
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


def _decode_recording(payload: Any) -> DispatchRecording | None:
    if payload is None:
        return None
    data = _strict_keys(
        payload, object_name="recording", required=_DISPATCH_V2_RECORDING_KEYS
    )
    try:
        options = RecordingOptions(
            events_every_n=data["events_every_n"],
            checkpoint_every=data["checkpoint_every"],
            checkpoint_keep_last=data["checkpoint_keep_last"],
            phases=data["phases"],
        )
    except ValueError as exc:
        raise DispatchManifestError(f"recording options: {exc}") from exc
    resume_raw = data["resume_from"]
    resume_from: dict[int, str] | None = None
    if resume_raw is not None:
        if not isinstance(resume_raw, dict):
            raise DispatchManifestError("recording resume_from must be an object")
        resume_from = {}
        malformed_key: str | None = None
        for key, value in resume_raw.items():
            if not isinstance(key, str):
                raise DispatchManifestError(
                    f"recording resume_from key {key!r} is not a str"
                )
            if re.fullmatch(r"0|[1-9][0-9]*", key) is None:
                malformed_key = key
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise DispatchManifestError(
                    f"recording resume_from key {key!r} is not an int"
                ) from exc
            resume_from[index] = value
        if len(resume_from) != len(resume_raw):
            raise DispatchManifestError(
                "recording resume_from contains a duplicate index"
            )
        if malformed_key is not None:
            raise DispatchManifestError(
                f"recording resume_from key {malformed_key!r} is not a "
                "canonical non-negative integer"
            )
    return DispatchRecording(
        options=options, catalog=data["catalog"], resume_from=resume_from
    )


def decode_manifest_payload(payload: Any) -> DispatchManifest:
    if not isinstance(payload, dict):
        raise DispatchManifestError("dispatch manifest must be a JSON object")

    version = payload.get("dispatch_schema_version")
    if type(version) is not int or version not in _DISPATCH_KEYS_BY_VERSION:
        raise DispatchManifestError(
            f"unsupported dispatch_schema_version: got {version!r}; "
            f"supported: {sorted(_DISPATCH_KEYS_BY_VERSION)!r}"
        )
    _strict_keys(
        payload,
        object_name="dispatch manifest",
        required=_DISPATCH_KEYS_BY_VERSION[version],
    )
    if payload["artifact"] != DISPATCH_ARTIFACT_TAG:
        raise DispatchManifestError(
            f"artifact tag mismatch: got {payload['artifact']!r}, "
            f"expected {DISPATCH_ARTIFACT_TAG!r}"
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
    recording = (
        _decode_recording(payload["recording"]) if version >= 2 else None
    )
    return DispatchManifest(
        plan=plan,
        plan_hash=payload["plan_hash"],
        shards=shards,
        datasets=datasets,
        resources=resources,
        recording=recording,
    )
