
from __future__ import annotations

import re
from pathlib import Path

import pytest

import kd
from kd.data.schema import PDEDataset
from kd.data.synthetic import generate_burgers_data
from kd.search.checkpoint_manifest import (
    FINAL_STATUS_COMPLETED,
    KIND_FINAL,
    load_checkpoint_manifest,
)

pytestmark = pytest.mark.integration

_POPULATION = 6
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def dataset() -> PDEDataset:
    return generate_burgers_data(nx=24, nt=24, nu=0.1, seed=0)


def test_controller_selects_via_manifest_and_resumes(
    tmp_path: Path, dataset: PDEDataset
) -> None:
    write_dir = tmp_path / "run1"
    kd.Model(
        algorithm="sga",
        generations=1,
        population=_POPULATION,
        checkpoint_dir=write_dir,
        checkpoint_every=1,
        seed=0,
        verbose=False,
    ).fit(dataset)

    entries = load_checkpoint_manifest(write_dir)
    finals = [e for e in entries if e.kind == KIND_FINAL]
    assert len(finals) == 1
    final = finals[0]


    assert final.final_status == FINAL_STATUS_COMPLETED
    assert final.algorithm == "sga"
    assert type(final.seed) is int
    assert final.config_hash is not None
    assert _SHA256_RE.match(final.config_hash)
    assert final.kd_version



    resume_target = write_dir / final.filename
    assert resume_target.is_file()

    resume_dir = tmp_path / "run2"
    resumed = kd.Model(
        algorithm="sga",
        generations=1,
        population=_POPULATION,
        checkpoint_dir=resume_dir,
        checkpoint_every=1,
        seed=0,
        verbose=False,
    ).fit(dataset, resume_from=resume_target)
    assert resumed.best_expr_


    resumed_entries = load_checkpoint_manifest(resume_dir)
    assert any(e.kind == KIND_FINAL for e in resumed_entries)
