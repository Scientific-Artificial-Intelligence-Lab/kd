
from __future__ import annotations

import json
from pathlib import Path

import pytest

import kd
from kd.data.schema import PDEDataset
from kd.data.synthetic import generate_burgers_data
from kd.search import (
    IterationEventEmitter,
    append_catalog_row,
    catalog_row_from_result,
    create_run_dir,
    finalize_run_dir,
    load_checkpoint_manifest,
    run_id_of_run_dir,
)
from kd.search.checkpoint_manifest import KIND_FINAL, MANIFEST_FILENAME

pytestmark = pytest.mark.integration

_POPULATION = 6


@pytest.fixture(scope="module")
def dataset() -> PDEDataset:
    return generate_burgers_data(nx=24, nt=24, nu=0.1, seed=0)


def _recorded_fit(
    runs_root: Path,
    run_id: str,
    dataset: PDEDataset,
    *,
    resume_from: Path | None = None,
    parent_run_id: str | None = None,
) -> dict:
    paths = create_run_dir(runs_root / run_id)
    model = kd.Model(
        algorithm="sga",
        generations=1,
        population=_POPULATION,
        seed=0,
        verbose=False,
        checkpoint_dir=paths.checkpoints,
        checkpoint_every=1,
        phases_path=paths.phases,
        callbacks=[IterationEventEmitter(jsonl_path=paths.events)],
    )
    model.fit(dataset, resume_from=resume_from)
    result = model.result_
    manifest = finalize_run_dir(
        paths,
        result,
        run_id=run_id,
        instrument="sga",
        status="completed",
        lineage=result.manifest.resume_source if result.manifest else None,
    )
    append_catalog_row(
        runs_root / "catalog.jsonl",
        catalog_row_from_result(
            result,
            run_id=run_id,
            created_at=manifest["created_at"],
            instrument="sga",
            status="completed",
            run_dir=run_id,
            record_path=f"{run_id}/record.json",
            parent_run_id=parent_run_id,
            resume_from=(
                str(resume_from) if resume_from is not None else None
            ),
        ),
    )
    return manifest


def test_recorded_fit_then_resume_carries_lineage(
    tmp_path: Path, dataset: PDEDataset
) -> None:
    runs_root = tmp_path / "runs"


    first = _recorded_fit(runs_root, "sga-run1", dataset)
    root1 = runs_root / "sga-run1"
    for name in ("manifest.json", "record.json", "events.jsonl",
                 "phases.jsonl", "recorder.json"):
        assert (root1 / name).is_file(), f"missing {name}"
    assert first["lineage"] is None
    assert first["artifacts"]["events.jsonl"] is not None

    phases = [
        json.loads(line)["phase"]
        for line in (root1 / "phases.jsonl").read_text().splitlines()
    ]
    assert phases == ["fit_started", "search_started", "search_ended"]



    entries = load_checkpoint_manifest(root1 / "checkpoints")
    listed = {entry.filename for entry in entries} | {MANIFEST_FILENAME}
    on_disk = {p.name for p in (root1 / "checkpoints").iterdir()}
    assert on_disk == listed
    final = next(e for e in entries if e.kind == KIND_FINAL)


    source = root1 / "checkpoints" / final.filename
    second = _recorded_fit(
        runs_root,
        "sga-run2",
        dataset,
        resume_from=source,
        parent_run_id=run_id_of_run_dir(root1),
    )

    lineage = second["lineage"]
    assert lineage is not None
    assert lineage["resume_from"] == str(source)
    assert lineage["source_run_id"] == "sga-run1"
    assert lineage["source_final_status"] == "completed"
    assert lineage["source_config_hash"] == final.config_hash
    assert lineage["source_iteration"] == final.iteration


    ledger = json.loads(
        (runs_root / "sga-run2" / "checkpoints" / MANIFEST_FILENAME).read_text()
    )
    assert ledger["schema_version"] == 2
    assert ledger["lineage"] == lineage


    rows = [
        json.loads(line)
        for line in (runs_root / "catalog.jsonl").read_text().splitlines()
    ]
    assert [row["run_id"] for row in rows] == ["sga-run1", "sga-run2"]
    assert rows[1]["parent_run_id"] == "sga-run1"
    assert rows[0]["record_hash"] is not None

    assert rows[0]["nmse"] is not None


def test_shallow_relative_resume_does_not_probe_cwd_run_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = create_run_dir(tmp_path / "unrelated-run")
    finalize_run_dir(
        paths,
        None,
        run_id="unrelated-run",
        instrument="sga",
        status="raised",
    )
    monkeypatch.chdir(paths.root)

    lineage = kd.Model(algorithm="sga")._build_resume_lineage(
        Path("../legacy.pt")
    )

    assert lineage["source_run_id"] is None
