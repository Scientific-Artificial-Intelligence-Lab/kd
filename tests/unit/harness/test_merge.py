
from __future__ import annotations

from pathlib import Path

import pytest
from kd.harness.dispatch import (
    DispatchDatasetSpec,
    DispatchAllocationError,
    build_dispatch_manifest,
    write_dispatch_manifest,
)
from kd.harness.merge import (
    DispatchMergeError,
    MergeReplayError,
    ShardMappingError,
    ShardMissingError,
    merge_shards,
)

from kd.harness.plan import ExperimentPlan, PlanEntry
from kd.harness.store import (
    EvidenceStore,
    EvidenceStoreError,
    environment_fingerprint,
)

from ._helpers import build_sealed_store, make_record


def _entry(seed: int, ref: str = "a", instrument: str = "sga") -> PlanEntry:
    return PlanEntry(
        instrument=instrument, dataset_ref=ref, seed=seed, model_kwargs={}
    )


def _four_entry_plan() -> ExperimentPlan:
    return ExperimentPlan(
        name="merge-full",
        entries=(
            _entry(0, "a"),
            _entry(1, "b"),
            _entry(2, "a"),
            _entry(3, "b"),
        ),
    )


def _specs(plan: ExperimentPlan) -> dict[str, DispatchDatasetSpec]:
    refs = {e.dataset_ref for e in plan.entries}
    return {r: DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={}) for r in refs}


def _rec_for(entry: PlanEntry, **kw: object) -> object:
    return make_record(instrument=entry.instrument, seed=entry.seed, **kw)


def _populate(
    batch_root: Path,
    manifest: object,
    plan: ExperimentPlan,
    records_by_full: dict[int, object],
) -> None:
    for shard in manifest.shards:
        shard_root = batch_root / "shards" / shard.shard_id
        entries = [plan.entries[i] for i in shard.entry_indices]
        records = {
            local: records_by_full[full]
            for local, full in enumerate(shard.entry_indices)
            if records_by_full.get(full) is not None
        }
        build_sealed_store(
            shard_root,
            entries=entries,
            records=records,
            name=f"{plan.name}::{shard.shard_id}",
        )


def _build_batch(
    tmp_path: Path,
    name: str,
    plan: ExperimentPlan,
    *,
    allocation: list[list[int]] | None = None,
    entry_indices: list[int] | None = None,
    records_by_full: dict[int, object],
) -> Path:
    kwargs: dict[str, object] = {"dataset_specs": _specs(plan)}
    if allocation is not None:
        kwargs["allocation"] = allocation
    else:
        kwargs["n_workers"] = 1
        if entry_indices is not None:
            kwargs["entry_indices"] = entry_indices
    manifest = build_dispatch_manifest(plan, **kwargs)
    batch_root = tmp_path / name
    written = write_dispatch_manifest(manifest, batch_root)
    _populate(batch_root, manifest, plan, records_by_full)
    return Path(written)





def test_full_replay_preserves_records_and_hashes(tmp_path: Path) -> None:
    plan = _four_entry_plan()
    recs = {i: _rec_for(plan.entries[i]) for i in range(4)}
    manifest_path = _build_batch(
        tmp_path, "batch0", plan, allocation=[[0, 2], [1, 3]], records_by_full=recs
    )

    merged = merge_shards([manifest_path], merged_root=tmp_path / "merged")

    assert isinstance(merged, EvidenceStore)
    assert merged.plan_hash == plan.plan_hash()
    assert set(merged.records) == {0, 1, 2, 3}
    for i in range(4):
        assert merged.records[i].record_hash == recs[i].record_hash
    assert [a["status"] for a in merged.attempts] == ["completed"] * 4
    assert [a["entry_index"] for a in merged.attempts] == [0, 1, 2, 3]





def test_partial_shard_leaves_missing_entry_unattempted(tmp_path: Path) -> None:
    plan = _four_entry_plan()

    recs = {0: _rec_for(plan.entries[0]), 1: _rec_for(plan.entries[1]),
            3: _rec_for(plan.entries[3])}
    manifest_path = _build_batch(
        tmp_path, "batch0", plan, allocation=[[0, 2], [1, 3]], records_by_full=recs
    )

    merged = merge_shards([manifest_path], merged_root=tmp_path / "merged")

    assert set(merged.records) == {0, 1, 3}

    assert all(a["entry_index"] != 2 for a in merged.attempts)





def _three_entry_plan() -> ExperimentPlan:
    return ExperimentPlan(
        name="merge-three",
        entries=(_entry(0, "a"), _entry(1, "a"), _entry(2, "a")),
    )


def test_cross_manifest_outcome_overlap_fails(tmp_path: Path) -> None:
    plan = _three_entry_plan()
    a = _build_batch(
        tmp_path, "batchA", plan, entry_indices=[0, 1],
        records_by_full={0: _rec_for(plan.entries[0]), 1: _rec_for(plan.entries[1])},
    )
    b = _build_batch(
        tmp_path, "batchB", plan, entry_indices=[1, 2],
        records_by_full={1: _rec_for(plan.entries[1]), 2: _rec_for(plan.entries[2])},
    )

    with pytest.raises(DispatchAllocationError):
        merge_shards([a, b], merged_root=tmp_path / "merged")





def test_retry_form_overlapping_alloc_disjoint_outcome_merges_clean(
    tmp_path: Path,
) -> None:
    plan = _three_entry_plan()

    a = _build_batch(
        tmp_path, "batchA", plan, entry_indices=[0, 1, 2],
        records_by_full={0: _rec_for(plan.entries[0])},
    )


    b = _build_batch(
        tmp_path, "batchB", plan, entry_indices=[1, 2],
        records_by_full={1: _rec_for(plan.entries[1]), 2: _rec_for(plan.entries[2])},
    )
    merged = merge_shards([a, b], merged_root=tmp_path / "merged")
    assert set(merged.records) == {0, 1, 2}





def test_mapping_mismatch_fails(tmp_path: Path) -> None:
    plan = _four_entry_plan()
    manifest = build_dispatch_manifest(
        plan, dataset_specs=_specs(plan), allocation=[[0, 2], [1, 3]]
    )
    batch_root = tmp_path / "batch0"
    written = write_dispatch_manifest(manifest, batch_root)


    bad_entry = _entry(99, "a")
    build_sealed_store(
        batch_root / "shards" / "shard-00",
        entries=[bad_entry, plan.entries[2]],
        records={0: make_record(instrument="sga", seed=99)},
        name=f"{plan.name}::shard-00",
    )
    build_sealed_store(
        batch_root / "shards" / "shard-01",
        entries=[plan.entries[1], plan.entries[3]],
        records={0: _rec_for(plan.entries[1]), 1: _rec_for(plan.entries[3])},
        name=f"{plan.name}::shard-01",
    )
    with pytest.raises(ShardMappingError):
        merge_shards([Path(written)], merged_root=tmp_path / "merged")





def test_missing_shard_fails_by_default(tmp_path: Path) -> None:
    plan = _four_entry_plan()
    manifest = build_dispatch_manifest(
        plan, dataset_specs=_specs(plan), allocation=[[0, 2], [1, 3]]
    )
    batch_root = tmp_path / "batch0"
    written = write_dispatch_manifest(manifest, batch_root)

    build_sealed_store(
        batch_root / "shards" / "shard-00",
        entries=[plan.entries[0], plan.entries[2]],
        records={0: _rec_for(plan.entries[0]), 1: _rec_for(plan.entries[2])},
        name=f"{plan.name}::shard-00",
    )
    with pytest.raises(ShardMissingError):
        merge_shards([Path(written)], merged_root=tmp_path / "merged")


def test_require_all_shards_false_skips_missing_shard(tmp_path: Path) -> None:
    plan = _four_entry_plan()
    manifest = build_dispatch_manifest(
        plan, dataset_specs=_specs(plan), allocation=[[0, 2], [1, 3]]
    )
    batch_root = tmp_path / "batch0"
    written = write_dispatch_manifest(manifest, batch_root)
    build_sealed_store(
        batch_root / "shards" / "shard-00",
        entries=[plan.entries[0], plan.entries[2]],
        records={0: _rec_for(plan.entries[0]), 1: _rec_for(plan.entries[2])},
        name=f"{plan.name}::shard-00",
    )
    merged = merge_shards(
        [Path(written)], merged_root=tmp_path / "merged", require_all_shards=False
    )

    assert set(merged.records) == {0, 2}





def test_same_ref_different_fingerprint_fails_at_merged_load(tmp_path: Path) -> None:


    plan = ExperimentPlan(name="ident", entries=(_entry(0, "a"), _entry(1, "a")))
    manifest = build_dispatch_manifest(
        plan, dataset_specs=_specs(plan), allocation=[[0], [1]]
    )
    batch_root = tmp_path / "batch0"
    written = write_dispatch_manifest(manifest, batch_root)
    build_sealed_store(
        batch_root / "shards" / "shard-00",
        entries=[plan.entries[0]],
        records={0: make_record(
            instrument="sga", seed=0, dataset_cache_fingerprint="sha256:aaa"
        )},
        name=f"{plan.name}::shard-00",
    )
    build_sealed_store(
        batch_root / "shards" / "shard-01",
        entries=[plan.entries[1]],
        records={0: make_record(
            instrument="sga", seed=1, dataset_cache_fingerprint="sha256:bbb"
        )},
        name=f"{plan.name}::shard-01",
    )
    with pytest.raises(EvidenceStoreError, match="dataset"):
        merge_shards([Path(written)], merged_root=tmp_path / "merged")





def test_plan_hash_disagreement_across_manifests_fails(tmp_path: Path) -> None:
    plan_a = _three_entry_plan()
    plan_b = ExperimentPlan(
        name="merge-three-DIFFERENT",
        entries=(_entry(0, "a"), _entry(1, "a"), _entry(2, "a")),
    )
    assert plan_a.plan_hash() != plan_b.plan_hash()
    a = _build_batch(
        tmp_path, "batchA", plan_a, entry_indices=[0],
        records_by_full={0: _rec_for(plan_a.entries[0])},
    )
    b = _build_batch(
        tmp_path, "batchB", plan_b, entry_indices=[0],
        records_by_full={0: _rec_for(plan_b.entries[0])},
    )
    with pytest.raises(DispatchMergeError):
        merge_shards([a, b], merged_root=tmp_path / "merged")





def test_merged_root_must_be_fresh(tmp_path: Path) -> None:
    plan = _four_entry_plan()
    recs = {i: _rec_for(plan.entries[i]) for i in range(4)}
    manifest_path = _build_batch(
        tmp_path, "batch0", plan, allocation=[[0, 2], [1, 3]], records_by_full=recs
    )
    merged_root = tmp_path / "merged"
    merged_root.mkdir()
    (merged_root / "stray.txt").write_text("occupied")
    with pytest.raises(EvidenceStoreError):
        merge_shards([manifest_path], merged_root=merged_root)


def test_empty_manifest_list_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DispatchMergeError):
        merge_shards([], merged_root=tmp_path / "merged")





def test_two_merges_produce_identical_index_bytes(tmp_path: Path) -> None:
    plan = _four_entry_plan()
    recs = {i: _rec_for(plan.entries[i]) for i in range(4)}
    manifest_path = _build_batch(
        tmp_path, "batch0", plan, allocation=[[0, 2], [1, 3]], records_by_full=recs
    )
    merge_shards([manifest_path], merged_root=tmp_path / "mergedA")
    merge_shards([manifest_path], merged_root=tmp_path / "mergedB")
    a = (tmp_path / "mergedA" / "index.json").read_bytes()
    b = (tmp_path / "mergedB" / "index.json").read_bytes()
    assert a == b





def test_merged_env_is_merge_host_fingerprint(tmp_path: Path) -> None:
    plan = _four_entry_plan()
    recs = {i: _rec_for(plan.entries[i]) for i in range(4)}
    manifest_path = _build_batch(
        tmp_path, "batch0", plan, allocation=[[0, 2], [1, 3]], records_by_full=recs
    )
    merged = merge_shards([manifest_path], merged_root=tmp_path / "merged")
    assert merged.env == environment_fingerprint()





def test_merge_replay_error_is_a_merge_error_subclass() -> None:
    assert issubclass(MergeReplayError, DispatchMergeError)
    assert issubclass(ShardMissingError, DispatchMergeError)
    assert issubclass(ShardMappingError, DispatchMergeError)


def test_merge_replay_error_fires_on_reload_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:




    import dataclasses

    plan = _four_entry_plan()
    recs = {i: _rec_for(plan.entries[i]) for i in range(4)}
    manifest_path = _build_batch(
        tmp_path, "batch0", plan, allocation=[[0, 2], [1, 3]], records_by_full=recs
    )
    merged_root = tmp_path / "merged"

    original_load = EvidenceStore.load

    def _tampering_load(root: Path) -> EvidenceStore:
        store = original_load(root)
        if Path(root) == merged_root:



            key = next(iter(store._records))
            store._records[key] = dataclasses.replace(
                store._records[key], record_hash="0" * 64
            )
        return store

    monkeypatch.setattr(EvidenceStore, "load", staticmethod(_tampering_load))
    with pytest.raises(MergeReplayError, match="record_hash changed"):
        merge_shards([manifest_path], merged_root=merged_root)
