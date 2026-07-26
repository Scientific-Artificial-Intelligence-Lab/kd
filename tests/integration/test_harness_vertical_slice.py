
from __future__ import annotations

import importlib.util
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

import kd
from kd.harness import (
    EvidenceStore,
    EvidenceStoreError,
    ExperimentPlan,
    PlanEntry,
    PlanRunResult,
    build_plan_report,
    run_plan,
)

logger = logging.getLogger(__name__)

_DATASET_REF = "burgers_tiny"
_SEEDS = (0, 1, 2)












_INSTRUMENT_KWARGS: dict[str, dict[str, object]] = {
    "sga": {
        "generations": 2,
        "population": 8,
        "depth": 3,
        "width": 4,
        "maxit": 3,
        "str_iters": 3,
        "d_tol": 0.5,
    },
    "pysindy": {},
    "discover": {"generations": 2},
}







_PYSINDY_AVAILABLE = importlib.util.find_spec("pysindy") is not None
_INSTRUMENT_ORDER = (
    ("sga", "pysindy", "discover") if _PYSINDY_AVAILABLE else ("sga", "discover")
)


_EXPECTED_EPISODES = len(_INSTRUMENT_ORDER) * len(_SEEDS)
_EXPECTED_GROUPS = len(_INSTRUMENT_ORDER)


@dataclass(frozen=True)
class _SliceArtifacts:

    plan: ExperimentPlan
    run_result: PlanRunResult
    store_root: Path
    wallclock_seconds: float


def _build_plan() -> ExperimentPlan:
    entries = tuple(
        PlanEntry(
            instrument=instrument,
            dataset_ref=_DATASET_REF,
            seed=seed,
            model_kwargs=dict(_INSTRUMENT_KWARGS[instrument]),
        )
        for instrument in _INSTRUMENT_ORDER
        for seed in _SEEDS
    )
    return ExperimentPlan(name="harness-vertical-slice", entries=entries)


@pytest.fixture(scope="module")
def slice_artifacts(tmp_path_factory: pytest.TempPathFactory) -> _SliceArtifacts:
    dataset = kd.generate_burgers_data(nx=32, nt=16, nu=0.1, seed=0)
    plan = _build_plan()
    store_root = tmp_path_factory.mktemp("harness_slice") / "store"

    start = time.perf_counter()
    run_result = run_plan(
        plan,
        datasets={_DATASET_REF: dataset},
        store_root=store_root,
    )
    wallclock = time.perf_counter() - start



    print(f"SLICE_WALLCLOCK_SECONDS={wallclock:.3f}")
    logger.info("harness vertical slice wallclock: %.3fs", wallclock)

    return _SliceArtifacts(
        plan=plan,
        run_result=run_result,
        store_root=store_root,
        wallclock_seconds=wallclock,
    )


@pytest.mark.integration
def test_all_episodes_completed(slice_artifacts: _SliceArtifacts) -> None:
    outcomes = slice_artifacts.run_result.outcomes
    assert len(outcomes) == _EXPECTED_EPISODES

    non_completed = [
        (o.entry.instrument, o.entry.seed, o.status, o.error_type, o.error_message)
        for o in outcomes
        if o.status != "completed"
    ]
    assert not non_completed, f"non-completed episodes: {non_completed}"


@pytest.mark.integration
def test_store_load_verifies_and_yields_all_records(
    slice_artifacts: _SliceArtifacts,
) -> None:
    store = EvidenceStore.load(slice_artifacts.store_root)

    assert len(store.attempts) == _EXPECTED_EPISODES
    assert len(store.records) == _EXPECTED_EPISODES

    entries = slice_artifacts.plan.entries
    for index, entry in enumerate(entries):
        record = store.records[index]
        assert record.evidence.instrument == entry.instrument
        assert record.evidence.seed == entry.seed


@pytest.mark.integration
def test_plan_hash_roundtrips_through_the_store(
    slice_artifacts: _SliceArtifacts,
) -> None:
    store = EvidenceStore.load(slice_artifacts.store_root)
    original = slice_artifacts.plan.plan_hash()



    assert store.plan_hash == original
    assert store.plan.plan_hash() == original


@pytest.mark.integration
def test_report_has_all_groups_and_rows(
    slice_artifacts: _SliceArtifacts,
) -> None:
    store = EvidenceStore.load(slice_artifacts.store_root)
    report = build_plan_report(
        plan=store.plan,
        plan_hash=store.plan_hash,
        env=store.env,
        attempts=store.attempts,
        records=store.records,
    )

    group_headers = [line for line in report.splitlines() if line.startswith("### ")]
    assert len(group_headers) == _EXPECTED_GROUPS
    for instrument in _INSTRUMENT_ORDER:
        assert f"### {instrument}" in report



    data_rows = [
        line
        for line in report.splitlines()
        if line.startswith("|")
        and "---" not in line
        and not line.startswith("| instrument ")
    ]
    assert len(data_rows) == _EXPECTED_EPISODES, (
        f"expected {_EXPECTED_EPISODES} data rows, got {len(data_rows)}:\n"
        + "\n".join(data_rows)
    )


    assert "### failures" not in report


@pytest.mark.integration
def test_tamper_flip_one_byte_fails_load(
    slice_artifacts: _SliceArtifacts, tmp_path: Path
) -> None:
    tampered_root = tmp_path / "tampered_store"
    shutil.copytree(slice_artifacts.store_root, tampered_root)


    EvidenceStore.load(tampered_root)

    record_files = sorted((tampered_root / "records").glob("entry-*.json"))
    assert record_files, "expected at least one sealed record file"
    victim = record_files[0]

    raw = bytearray(victim.read_bytes())


    flip_at = len(raw) // 2
    raw[flip_at] ^= 0x01
    victim.write_bytes(raw)

    with pytest.raises(EvidenceStoreError):
        EvidenceStore.load(tampered_root)
