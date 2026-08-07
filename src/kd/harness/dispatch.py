
from __future__ import annotations

import importlib
import json
import os
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import kd
from kd.core.jsonsafe import JSON_INDENT_SPACES
from kd.harness._dispatch_schema import (
    _DISPATCH_V1_DATASET_KEYS,
    _DISPATCH_V1_KEYS,
    _DISPATCH_V1_RESOURCES_KEYS,
    _DISPATCH_V1_SHARD_KEYS,
    DISPATCH_ARTIFACT_TAG,
    DISPATCH_SCHEMA_VERSION,
    DispatchDatasetSpec,
    DispatchManifest,
    DispatchManifestError,
    DispatchResources,
    ShardSpec,
    decode_manifest_payload,
    manifest_to_payload,
)
from kd.harness.plan import ExperimentPlan

__all__ = [
    "DISPATCH_ARTIFACT_TAG",
    "DISPATCH_SCHEMA_VERSION",
    "_DISPATCH_V1_DATASET_KEYS",
    "_DISPATCH_V1_KEYS",
    "_DISPATCH_V1_RESOURCES_KEYS",
    "_DISPATCH_V1_SHARD_KEYS",
    "DatasetResolverError",
    "DispatchDatasetSpec",
    "DispatchAllocationError",
    "DispatchManifest",
    "DispatchManifestError",
    "DispatchResources",
    "ShardSpec",
    "build_dispatch_manifest",
    "read_dispatch_manifest",
    "resolve_dataset",
    "write_dispatch_manifest",
]

_MANIFEST_FILENAME: Final[str] = "dispatch.json"


class DispatchAllocationError(ValueError):
    pass


class DatasetResolverError(ValueError):
    pass





def _validate_entry_indices(
    plan: ExperimentPlan, entry_indices: Sequence[int] | None
) -> list[int]:
    n_entries = len(plan.entries)
    if entry_indices is None:
        return list(range(n_entries))
    selected = list(entry_indices)
    if not selected:
        raise DispatchAllocationError("entry_indices must be non-empty when provided")
    previous: int | None = None
    for index in selected:
        if isinstance(index, bool) or not isinstance(index, int):
            raise DispatchAllocationError(
                f"entry_indices must be ints (bool rejected); got {index!r}"
            )
        if not 0 <= index < n_entries:
            raise DispatchAllocationError(
                f"entry_indices value {index} out of range for plan with "
                f"{n_entries} entries"
            )
        if previous is not None and index <= previous:
            raise DispatchAllocationError(
                f"entry_indices must be strictly increasing; got {selected!r}"
            )
        previous = index
    return selected


def _slots_from_n_workers(
    plan: ExperimentPlan, selected: list[int], n_workers: int
) -> list[list[int]]:
    if isinstance(n_workers, bool) or not isinstance(n_workers, int) or n_workers < 1:
        raise DispatchAllocationError(
            f"n_workers must be an int >= 1; got {n_workers!r}"
        )
    selected_set = set(selected)
    group_order: list[str] = []
    groups: dict[str, list[int]] = {}
    for index in range(len(plan.entries)):
        if index not in selected_set:
            continue
        ref = plan.entries[index].dataset_ref
        if ref not in groups:
            groups[ref] = []
            group_order.append(ref)
        groups[ref].append(index)

    slots: list[list[int]] = [[] for _ in range(n_workers)]
    for group_position, ref in enumerate(group_order):
        slots[group_position % n_workers].extend(groups[ref])
    for slot_position, slot in enumerate(slots):
        if not slot:
            raise DispatchAllocationError(
                f"slot {slot_position} is empty (n_workers={n_workers} exceeds the "
                f"number of dataset groups); lower n_workers or use manual allocation"
            )
    return [sorted(slot) for slot in slots]


def _slots_from_allocation(
    selected: list[int], allocation: Sequence[Sequence[int]], *, n_entries: int
) -> list[list[int]]:
    slots: list[list[int]] = []
    seen: set[int] = set()
    selected_set = set(selected)
    for slot_position, raw_slot in enumerate(allocation):
        slot = list(raw_slot)
        if not slot:
            raise DispatchAllocationError(f"allocation slot {slot_position} is empty")
        for index in slot:
            if isinstance(index, bool) or not isinstance(index, int):
                raise DispatchAllocationError(
                    f"allocation slot {slot_position} entry {index!r} is not an int"
                )
            if index not in selected_set:



                if 0 <= index < n_entries:
                    raise DispatchAllocationError(
                        f"allocation slot {slot_position} entry {index} is a valid "
                        f"plan index but not in this batch's selected subset "
                        f"{sorted(selected_set)!r}"
                    )
                raise DispatchAllocationError(
                    f"allocation slot {slot_position} entry {index} is out of the "
                    f"plan range [0, {n_entries})"
                )
            if index in seen:
                raise DispatchAllocationError(
                    f"allocation slots overlap at entry {index}"
                )
            seen.add(index)
        slots.append(sorted(slot))
    if seen != selected_set:
        raise DispatchAllocationError(
            "allocation union must cover exactly the selected entries "
            f"(missing={sorted(selected_set - seen)!r}, "
            f"extra={sorted(seen - selected_set)!r})"
        )
    return slots


def _heavy_instruments() -> set[str]:
    return {
        schema["algorithm"]
        for schema in kd.instrument_schemas()
        if schema.get("cost_class") == "heavy"
    }


def build_dispatch_manifest(
    plan: ExperimentPlan,
    *,
    dataset_specs: Mapping[str, DispatchDatasetSpec],
    n_workers: int | None = None,
    allocation: Sequence[Sequence[int]] | None = None,
    pin_table: Sequence[str] | None = None,
    heavy_keys: Collection[tuple[str, str]] | None = None,
    max_concurrent_heavy: int = 1,
    default_timeout_seconds: float | None = None,
    timeout_overrides: Mapping[str, float] | None = None,
    memory_max_gb: float | None = None,
    grace_seconds: float = 30.0,
    entry_indices: Sequence[int] | None = None,
) -> DispatchManifest:
    if (n_workers is None) == (allocation is None):
        raise DispatchAllocationError(
            "exactly one of n_workers or allocation must be provided"
        )
    selected = _validate_entry_indices(plan, entry_indices)
    if allocation is not None:
        slots = _slots_from_allocation(
            selected, allocation, n_entries=len(plan.entries)
        )
    else:
        assert n_workers is not None
        slots = _slots_from_n_workers(plan, selected, n_workers)

    n_shards = len(slots)
    if timeout_overrides is not None:
        valid_shard_ids = {f"shard-{i:02d}" for i in range(n_shards)}
        unknown_ids = set(timeout_overrides) - valid_shard_ids
        if unknown_ids:


            raise DispatchAllocationError(
                f"timeout_overrides names unknown shard id(s) "
                f"{sorted(unknown_ids)!r}; valid ids are "
                f"{sorted(valid_shard_ids)!r}"
            )
    if pin_table is not None:
        pins = list(pin_table)
        if len(pins) != n_shards:
            raise DispatchAllocationError(
                f"pin_table length {len(pins)} must equal the shard count {n_shards}"
            )
    else:
        pins = ["" for _ in range(n_shards)]

    heavy_instruments = _heavy_instruments() if heavy_keys is None else set()
    heavy_key_set = set(heavy_keys) if heavy_keys is not None else set()

    def _entry_is_heavy(index: int) -> bool:
        entry = plan.entries[index]
        if heavy_keys is None:
            return entry.instrument in heavy_instruments
        return (entry.instrument, entry.dataset_ref) in heavy_key_set

    overrides = dict(timeout_overrides) if timeout_overrides is not None else {}
    shards: list[ShardSpec] = []
    for shard_position, slot in enumerate(slots):
        shard_id = f"shard-{shard_position:02d}"
        cuda = pins[shard_position]
        shards.append(
            ShardSpec(
                shard_id=shard_id,
                entry_indices=tuple(slot),
                cuda_visible_devices=cuda,
                device="cuda" if cuda else None,
                heavy=any(_entry_is_heavy(index) for index in slot),
                timeout_seconds=overrides.get(shard_id, default_timeout_seconds),
                memory_max_gb=memory_max_gb,
            )
        )

    return DispatchManifest(
        plan=plan,
        plan_hash=plan.plan_hash(),
        shards=tuple(shards),
        datasets=dict(dataset_specs),
        resources=DispatchResources(
            max_concurrent_heavy=max_concurrent_heavy, grace_seconds=grace_seconds
        ),
    )





def write_dispatch_manifest(manifest: DispatchManifest, batch_root: Path) -> Path:
    root = Path(batch_root)
    if root.exists():
        if not root.is_dir():
            raise DispatchManifestError(f"batch_root is not a directory: {root}")
        if any(root.iterdir()):
            raise DispatchManifestError(
                f"batch_root is not empty (manifests are immutable): {root}"
            )
    root.mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)

    manifest_path = root / _MANIFEST_FILENAME
    tmp_path = root / f"{_MANIFEST_FILENAME}.tmp"
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest_to_payload(manifest),
            handle,
            indent=JSON_INDENT_SPACES,
            allow_nan=False,
            sort_keys=True,
        )
    os.replace(tmp_path, manifest_path)
    return manifest_path


def _reject_constant(token: str) -> Any:
    raise DispatchManifestError(
        f"manifest contains a non-standard JSON numeric token {token!r} "
        "(NaN / Infinity / -Infinity are not permitted)"
    )


def read_dispatch_manifest(path: Path) -> DispatchManifest:
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise DispatchManifestError(f"manifest is not valid JSON: {exc}") from exc
    return decode_manifest_payload(payload)





def resolve_dataset(spec: DispatchDatasetSpec) -> Any:
    loader = spec.loader
    if "." not in loader:
        raise DatasetResolverError(
            f"loader must be a dotted path (module.attr); got {loader!r}"
        )
    module_path, _, attr = loader.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise DatasetResolverError(
            f"cannot import loader module {module_path!r}: {exc}"
        ) from exc
    try:
        target = getattr(module, attr)
    except AttributeError as exc:
        raise DatasetResolverError(
            f"loader module {module_path!r} has no attribute {attr!r}"
        ) from exc
    if not callable(target):
        raise DatasetResolverError(f"loader target {loader!r} is not callable")
    try:
        return target(**spec.kwargs)
    except Exception as exc:
        raise DatasetResolverError(
            f"loader {loader!r} raised {type(exc).__qualname__}: {exc}"
        ) from exc
