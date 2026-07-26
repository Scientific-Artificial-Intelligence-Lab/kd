
from __future__ import annotations

from pathlib import Path

import pytest
from kd.harness.dispatch import (
    DatasetResolverError,
    DispatchDatasetSpec,
    DispatchAllocationError,
    DispatchManifest,
    DispatchManifestError,
    build_dispatch_manifest,
    resolve_dataset,
    write_dispatch_manifest,
)

from kd.data.schema import PDEDataset
from kd.harness.plan import ExperimentPlan, PlanEntry


def _entry(instrument: str, ref: str, seed: int) -> PlanEntry:
    return PlanEntry(
        instrument=instrument, dataset_ref=ref, seed=seed, model_kwargs={}
    )


def _spec(refs: set[str]) -> dict[str, DispatchDatasetSpec]:
    return {
        ref: DispatchDatasetSpec(loader="kd.generate_burgers_data", kwargs={}) for ref in refs
    }





def test_build_is_deterministic_object_and_bytes(tmp_path: Path) -> None:
    plan = ExperimentPlan(
        name="det",
        entries=(_entry("sga", "a", 0), _entry("sga", "b", 1)),
    )
    specs = _spec({"a", "b"})
    first = build_dispatch_manifest(plan, dataset_specs=specs, n_workers=2)
    second = build_dispatch_manifest(plan, dataset_specs=specs, n_workers=2)
    assert first == second

    b1 = Path(write_dispatch_manifest(first, tmp_path / "b1")).read_bytes()
    b2 = Path(write_dispatch_manifest(second, tmp_path / "b2")).read_bytes()
    assert b1 == b2





def test_group_by_ref_first_appearance_rotation() -> None:


    plan = ExperimentPlan(
        name="rot",
        entries=(
            _entry("sga", "a", 0),
            _entry("sga", "b", 1),
            _entry("sga", "a", 2),
            _entry("sga", "c", 3),
            _entry("sga", "b", 4),
        ),
    )
    manifest = build_dispatch_manifest(
        plan, dataset_specs=_spec({"a", "b", "c"}), n_workers=2
    )
    shards = {s.shard_id: s.entry_indices for s in manifest.shards}
    assert shards == {"shard-00": (0, 2, 3), "shard-01": (1, 4)}


def test_shard_ids_are_zero_padded_slot_order() -> None:
    plan = ExperimentPlan(
        name="pad", entries=(_entry("sga", "a", 0), _entry("sga", "b", 1))
    )
    manifest = build_dispatch_manifest(
        plan, dataset_specs=_spec({"a", "b"}), n_workers=2
    )
    assert [s.shard_id for s in manifest.shards] == ["shard-00", "shard-01"]





def test_empty_slot_fails_loud() -> None:

    plan = ExperimentPlan(
        name="one_group",
        entries=(_entry("sga", "a", 0), _entry("sga", "a", 1)),
    )
    with pytest.raises(DispatchAllocationError):
        build_dispatch_manifest(plan, dataset_specs=_spec({"a"}), n_workers=2)





def _four_entry_plan() -> ExperimentPlan:
    return ExperimentPlan(
        name="four",
        entries=(
            _entry("sga", "a", 0),
            _entry("sga", "b", 1),
            _entry("sga", "a", 2),
            _entry("sga", "b", 3),
        ),
    )


def test_manual_allocation_is_honored() -> None:
    plan = _four_entry_plan()
    manifest = build_dispatch_manifest(
        plan, dataset_specs=_spec({"a", "b"}), allocation=[[0, 2], [1, 3]]
    )
    shards = {s.shard_id: s.entry_indices for s in manifest.shards}
    assert shards == {"shard-00": (0, 2), "shard-01": (1, 3)}


def test_allocation_and_n_workers_are_mutually_exclusive() -> None:
    plan = _four_entry_plan()
    with pytest.raises(DispatchAllocationError):
        build_dispatch_manifest(
            plan,
            dataset_specs=_spec({"a", "b"}),
            n_workers=2,
            allocation=[[0, 1], [2, 3]],
        )


def test_neither_n_workers_nor_allocation_fails() -> None:
    plan = _four_entry_plan()
    with pytest.raises(DispatchAllocationError):
        build_dispatch_manifest(plan, dataset_specs=_spec({"a", "b"}))


def test_allocation_overlap_fails() -> None:
    plan = _four_entry_plan()
    with pytest.raises(DispatchAllocationError):
        build_dispatch_manifest(
            plan, dataset_specs=_spec({"a", "b"}), allocation=[[0, 1], [1, 2]]
        )


def test_allocation_out_of_range_fails() -> None:
    plan = _four_entry_plan()
    with pytest.raises(DispatchAllocationError):
        build_dispatch_manifest(
            plan, dataset_specs=_spec({"a", "b"}), allocation=[[0, 99], [1, 2]]
        )


def test_allocation_union_must_cover_all_entries() -> None:
    plan = _four_entry_plan()
    with pytest.raises(DispatchAllocationError):
        build_dispatch_manifest(
            plan, dataset_specs=_spec({"a", "b"}), allocation=[[0], [1]]
        )


def test_allocation_rejects_empty_slot() -> None:
    plan = _four_entry_plan()
    with pytest.raises(DispatchAllocationError):
        build_dispatch_manifest(
            plan, dataset_specs=_spec({"a", "b"}), allocation=[[0, 1, 2, 3], []]
        )





def test_pin_table_derives_device() -> None:
    plan = ExperimentPlan(
        name="pin", entries=(_entry("sga", "a", 0), _entry("sga", "b", 1))
    )
    manifest = build_dispatch_manifest(
        plan, dataset_specs=_spec({"a", "b"}), n_workers=2, pin_table=["", "0"]
    )
    by_id = {s.shard_id: s for s in manifest.shards}
    assert by_id["shard-00"].cuda_visible_devices == ""
    assert by_id["shard-00"].device is None
    assert by_id["shard-01"].cuda_visible_devices == "0"
    assert by_id["shard-01"].device == "cuda"


def test_pin_table_default_is_all_cpu() -> None:
    plan = ExperimentPlan(
        name="cpu", entries=(_entry("sga", "a", 0), _entry("sga", "b", 1))
    )
    manifest = build_dispatch_manifest(
        plan, dataset_specs=_spec({"a", "b"}), n_workers=2
    )
    for shard in manifest.shards:
        assert shard.cuda_visible_devices == ""
        assert shard.device is None


def test_pin_table_length_must_match_shard_count() -> None:
    plan = ExperimentPlan(
        name="pinbad", entries=(_entry("sga", "a", 0), _entry("sga", "b", 1))
    )
    with pytest.raises(DispatchAllocationError):
        build_dispatch_manifest(
            plan, dataset_specs=_spec({"a", "b"}), n_workers=2, pin_table=["0"]
        )





def test_default_heavy_from_cost_class_and_any_entry_rule() -> None:


    plan = ExperimentPlan(
        name="heavy",
        entries=(
            _entry("sga", "a", 0),
            _entry("discover", "a", 1),
            _entry("pysindy", "b", 2),
        ),
    )
    manifest = build_dispatch_manifest(
        plan, dataset_specs=_spec({"a", "b"}), n_workers=2
    )
    by_id = {s.shard_id: s for s in manifest.shards}
    assert by_id["shard-00"].heavy is True
    assert by_id["shard-01"].heavy is False


def test_explicit_heavy_keys_override_cost_class() -> None:


    plan = ExperimentPlan(
        name="heavy2",
        entries=(
            _entry("sga", "a", 0),
            _entry("discover", "a", 1),
            _entry("pysindy", "b", 2),
        ),
    )
    manifest = build_dispatch_manifest(
        plan,
        dataset_specs=_spec({"a", "b"}),
        n_workers=2,
        heavy_keys={("pysindy", "b")},
    )
    by_id = {s.shard_id: s for s in manifest.shards}
    assert by_id["shard-00"].heavy is False
    assert by_id["shard-01"].heavy is True





def test_timeout_override_beats_default() -> None:
    plan = ExperimentPlan(
        name="to", entries=(_entry("sga", "a", 0), _entry("sga", "b", 1))
    )
    manifest = build_dispatch_manifest(
        plan,
        dataset_specs=_spec({"a", "b"}),
        n_workers=2,
        default_timeout_seconds=100.0,
        timeout_overrides={"shard-01": 5.0},
    )
    by_id = {s.shard_id: s for s in manifest.shards}
    assert by_id["shard-00"].timeout_seconds == 100.0
    assert by_id["shard-01"].timeout_seconds == 5.0


def test_default_timeout_is_none() -> None:
    plan = ExperimentPlan(name="noto", entries=(_entry("sga", "a", 0),))
    manifest = build_dispatch_manifest(
        plan, dataset_specs=_spec({"a"}), n_workers=1
    )
    assert manifest.shards[0].timeout_seconds is None


def test_timeout_overrides_unknown_shard_id_fails_loud() -> None:


    plan = ExperimentPlan(
        name="to-typo", entries=(_entry("sga", "a", 0), _entry("sga", "b", 1))
    )
    with pytest.raises(DispatchAllocationError, match="shard-1"):
        build_dispatch_manifest(
            plan,
            dataset_specs=_spec({"a", "b"}),
            n_workers=2,
            timeout_overrides={"shard-1": 5.0},
        )





def test_entry_indices_selects_a_subset() -> None:
    plan = _four_entry_plan()
    manifest = build_dispatch_manifest(
        plan, dataset_specs=_spec({"a", "b"}), n_workers=1, entry_indices=[1, 3]
    )
    covered = sorted(i for s in manifest.shards for i in s.entry_indices)
    assert covered == [1, 3]





def test_write_refuses_non_empty_root(tmp_path: Path) -> None:
    plan = ExperimentPlan(name="w", entries=(_entry("sga", "a", 0),))
    manifest = build_dispatch_manifest(plan, dataset_specs=_spec({"a"}), n_workers=1)
    root = tmp_path / "batch"
    root.mkdir()
    (root / "stray.txt").write_text("occupied")
    with pytest.raises(DispatchManifestError):
        write_dispatch_manifest(manifest, root)





def test_resolve_dataset_happy_path() -> None:


    spec = DispatchDatasetSpec(
        loader="kd.generate_burgers_data",
        kwargs={"nx": 32, "nt": 8, "nu": 0.1, "seed": 0},
    )
    dataset = resolve_dataset(spec)
    assert isinstance(dataset, PDEDataset)


def test_resolve_dataset_rejects_loader_without_dot() -> None:


    with pytest.raises(DatasetResolverError):
        resolve_dataset(DispatchDatasetSpec(loader="nodot", kwargs={}))


def test_resolve_dataset_rejects_import_failure() -> None:
    with pytest.raises(DatasetResolverError):
        resolve_dataset(
            DispatchDatasetSpec(loader="kd_no_such_module_zzz.load", kwargs={})
        )


def test_resolve_dataset_rejects_missing_attribute() -> None:
    with pytest.raises(DatasetResolverError):
        resolve_dataset(
            DispatchDatasetSpec(loader="kd.this_attr_does_not_exist_zzz", kwargs={})
        )


def test_resolve_dataset_rejects_non_callable_target() -> None:

    with pytest.raises(DatasetResolverError):
        resolve_dataset(DispatchDatasetSpec(loader="kd.__version__", kwargs={}))


def test_resolve_dataset_wraps_loader_exception() -> None:

    with pytest.raises(DatasetResolverError):
        resolve_dataset(
            DispatchDatasetSpec(
                loader="kd.generate_burgers_data", kwargs={"nx": -5, "nt": -5}
            )
        )





def test_build_returns_dispatch_manifest() -> None:
    plan = ExperimentPlan(name="ty", entries=(_entry("sga", "a", 0),))
    manifest = build_dispatch_manifest(plan, dataset_specs=_spec({"a"}), n_workers=1)
    assert isinstance(manifest, DispatchManifest)
    assert manifest.plan_hash == plan.plan_hash()
