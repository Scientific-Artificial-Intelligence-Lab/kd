
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from kd.harness.dispatch import (
    DispatchAllocationError,
    DispatchManifest,
    read_dispatch_manifest,
)
from kd.harness.plan import ExperimentPlan, PlanEntry
from kd.harness.store import (
    EvidenceStore,
    EvidenceStoreError,
    environment_fingerprint,
)
from kd.search.records import RunRecord

logger = logging.getLogger(__name__)


class DispatchMergeError(RuntimeError):
    pass


class ShardMissingError(DispatchMergeError):
    pass


class ShardMappingError(DispatchMergeError):
    pass


class MergeReplayError(DispatchMergeError):
    pass


@dataclass(frozen=True, kw_only=True)
class _ReplayOutcome:

    entry_index: int
    entry: PlanEntry
    status: str
    record: RunRecord | None
    error_type: str | None
    error_message: str | None
    wallclock_seconds: float


def _decode_manifests(
    manifest_paths: list[Path],
) -> tuple[list[tuple[DispatchManifest, Path]], ExperimentPlan]:
    if not manifest_paths:
        raise DispatchMergeError("merge requires at least one manifest path")
    pairs: list[tuple[DispatchManifest, Path]] = []
    for path in manifest_paths:
        path = Path(path)
        manifest = read_dispatch_manifest(path)
        pairs.append((manifest, path.parent))
    full_plan = pairs[0][0].plan
    full_hash = pairs[0][0].plan_hash
    for manifest, _ in pairs[1:]:
        if manifest.plan_hash != full_hash or manifest.plan != full_plan:
            raise DispatchMergeError(
                "manifests disagree on the plan (plan_hash / structure mismatch); "
                "merge requires a single shared plan"
            )
    return pairs, full_plan


def _verify_shard_mapping(
    shard_store: EvidenceStore,
    entry_indices: tuple[int, ...],
    full_plan: ExperimentPlan,
    *,
    manifest_dir: Path,
    shard_id: str,
) -> None:
    if len(shard_store.plan.entries) != len(entry_indices):
        raise ShardMappingError(
            f"shard {shard_id!r} in {manifest_dir}: sub-plan has "
            f"{len(shard_store.plan.entries)} entries, expected "
            f"{len(entry_indices)}"
        )
    for local, full in enumerate(entry_indices):
        if shard_store.plan.entries[local] != full_plan.entries[full]:
            raise ShardMappingError(
                f"shard {shard_id!r} in {manifest_dir}: local entry {local} does "
                f"not match full entry {full}"
            )


def merge_shards(
    manifest_paths: Sequence[Path],
    *,
    merged_root: Path,
    require_all_shards: bool = True,
) -> EvidenceStore:
    pairs, full_plan = _decode_manifests(list(manifest_paths))

    outcomes: dict[int, _ReplayOutcome] = {}
    expected_hash: dict[int, str] = {}
    for manifest, manifest_dir in pairs:
        for shard in manifest.shards:
            shard_root = manifest_dir / "shards" / shard.shard_id
            try:
                shard_store = EvidenceStore.load(shard_root)
            except EvidenceStoreError as exc:
                if require_all_shards:
                    raise ShardMissingError(
                        f"shard {shard.shard_id!r} in {manifest_dir} is missing or "
                        f"failed to load: {exc}"
                    ) from exc
                logger.info(
                    "merge: skipping unloadable shard %s (%s)", shard.shard_id, exc
                )
                continue

            _verify_shard_mapping(
                shard_store,
                shard.entry_indices,
                full_plan,
                manifest_dir=manifest_dir,
                shard_id=shard.shard_id,
            )

            for attempt in shard_store.attempts:
                local = attempt["entry_index"]
                full = shard.entry_indices[local]
                if full in outcomes:
                    raise DispatchAllocationError(
                        f"two shards delivered an outcome for full entry {full}: "
                        f"shard {shard.shard_id!r} in {manifest_dir} conflicts with "
                        f"an earlier shard"
                    )
                record = shard_store.records.get(local)
                outcomes[full] = _ReplayOutcome(
                    entry_index=full,
                    entry=full_plan.entries[full],
                    status=attempt["status"],
                    record=record,
                    error_type=attempt["error_type"],
                    error_message=attempt["error_message"],
                    wallclock_seconds=attempt["wallclock_seconds"],
                )
                if record is not None:
                    expected_hash[full] = record.record_hash

    merged = EvidenceStore.create(
        Path(merged_root), plan=full_plan, env=environment_fingerprint()
    )
    for full in sorted(outcomes):
        merged.add_outcome(outcomes[full])

    reloaded = EvidenceStore.load(Path(merged_root))
    for full, sealed_hash in expected_hash.items():
        if reloaded.records[full].record_hash != sealed_hash:
            raise MergeReplayError(
                f"record_hash changed for full entry {full} across merge replay: "
                f"{reloaded.records[full].record_hash!r} != {sealed_hash!r}"
            )
    logger.info(
        "merge: replayed %d outcomes (%d records) into %s",
        len(outcomes),
        len(expected_hash),
        merged_root,
    )
    return reloaded


__all__ = [
    "DispatchMergeError",
    "MergeReplayError",
    "ShardMappingError",
    "ShardMissingError",
    "merge_shards",
]
