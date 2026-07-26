
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import kd
from kd.data.schema import PDEDataset
from kd.data.synthetic import generate_burgers_data
from kd.search.callbacks import build_checkpoint_payload
from kd.search.checkpoint_manifest import (
    MANIFEST_FILENAME,
    CheckpointManifestError,
)
from kd.search.dlga import DLGAPlugin

pytestmark = pytest.mark.integration

_POPULATION = 6


@pytest.fixture(scope="module")
def dataset() -> PDEDataset:
    return generate_burgers_data(nx=24, nt=24, nu=0.1, seed=0)


@pytest.fixture(scope="module")
def base_checkpoint(
    dataset: PDEDataset, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    write_dir = tmp_path_factory.mktemp("g2b_write")
    kd.Model(
        algorithm="sga",
        generations=1,
        population=_POPULATION,
        checkpoint_dir=write_dir,
        seed=0,
        verbose=False,
    ).fit(dataset)
    final = write_dir / "checkpoint_final.pt"
    assert final.exists()
    return final


def _resume(
    tmp_path: Path,
    dataset: PDEDataset,
    checkpoint: Path,
    **overrides: object,
) -> Path:
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    kwargs: dict[str, object] = {
        "algorithm": "sga",
        "generations": 1,
        "population": _POPULATION,
        "checkpoint_dir": resume_dir,
        "seed": 0,
        "verbose": False,
    }
    kwargs.update(overrides)
    kd.Model(**kwargs).fit(dataset, resume_from=checkpoint)
    return resume_dir







def test_w1_same_config_resume_runs(
    tmp_path: Path, dataset: PDEDataset, base_checkpoint: Path
) -> None:
    resume_dir = _resume(tmp_path, dataset, base_checkpoint)
    assert (resume_dir / "checkpoint_final.pt").exists()







def test_w2_resume_safe_population_change_is_allowed(
    tmp_path: Path, dataset: PDEDataset, base_checkpoint: Path
) -> None:
    resume_dir = _resume(tmp_path, dataset, base_checkpoint, population=_POPULATION + 4)
    assert (resume_dir / "checkpoint_final.pt").exists()







def test_w3_init_only_change_is_rejected_before_build(
    tmp_path: Path, dataset: PDEDataset, base_checkpoint: Path
) -> None:
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    with pytest.raises(ValueError, match="init_only") as exc:
        kd.Model(
            algorithm="sga",
            generations=1,
            population=_POPULATION,
            aic_ratio=999.0,
            checkpoint_dir=resume_dir,
            seed=0,
            verbose=False,
        ).fit(dataset, resume_from=base_checkpoint)
    assert "aic_ratio" in str(exc.value)


    assert not any(resume_dir.iterdir())







def test_w4_identity_autograd_rejected_before_surrogate_training(
    tmp_path: Path, dataset: PDEDataset, base_checkpoint: Path
) -> None:
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    with pytest.raises(ValueError, match="new lineage") as exc:
        kd.Model(
            algorithm="sga",
            generations=1,
            population=_POPULATION,
            derivatives="autograd",
            checkpoint_dir=resume_dir,
            seed=0,
            verbose=False,
        ).fit(dataset, resume_from=base_checkpoint)
    assert "use_autograd" in str(exc.value)


    assert not any(resume_dir.iterdir())







def _strip_config(checkpoint: Path, dest: Path) -> Path:
    payload = torch.load(checkpoint, weights_only=False)
    payload.pop("config", None)
    payload.pop("config_canon_scheme", None)
    torch.save(payload, dest)
    return dest


def test_w5_legacy_checkpoint_without_config_loads_unchecked(
    tmp_path: Path, dataset: PDEDataset, base_checkpoint: Path
) -> None:
    legacy = _strip_config(base_checkpoint, tmp_path / "legacy.pt")
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()




    kd.Model(
        algorithm="sga",
        generations=1,
        population=_POPULATION,
        aic_ratio=999.0,
        checkpoint_dir=resume_dir,
        seed=0,
        verbose=False,
    ).fit(dataset, resume_from=legacy)
    assert (resume_dir / "checkpoint_final.pt").exists()







def test_w6_foreign_config_scheme_is_rejected(
    tmp_path: Path, dataset: PDEDataset, base_checkpoint: Path
) -> None:
    payload = torch.load(base_checkpoint, weights_only=False)
    assert payload["config"] is not None
    payload["config_canon_scheme"] = "kd-config-v2"
    foreign = tmp_path / "foreign.pt"
    torch.save(payload, foreign)

    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    with pytest.raises(ValueError, match="config_canon_scheme"):
        kd.Model(
            algorithm="sga",
            generations=1,
            population=_POPULATION,
            checkpoint_dir=resume_dir,
            seed=0,
            verbose=False,
        ).fit(dataset, resume_from=foreign)
    assert not any(resume_dir.iterdir())














def _clone_manifest_dir(base_checkpoint: Path, dest: Path) -> Path:
    shutil.copytree(base_checkpoint.parent, dest)
    assert (dest / MANIFEST_FILENAME).is_file()
    return dest


def test_fence_resume_listed_file_from_manifest_dir_succeeds(
    tmp_path: Path, dataset: PDEDataset, base_checkpoint: Path
) -> None:


    resumed = kd.Model(
        algorithm="sga",
        generations=1,
        population=_POPULATION,
        seed=0,
        verbose=False,
    ).fit(dataset, resume_from=base_checkpoint)
    assert resumed.best_expr_


def test_fence_resume_unlisted_name_from_manifest_dir_is_rejected(
    tmp_path: Path, dataset: PDEDataset, base_checkpoint: Path
) -> None:





    clone = _clone_manifest_dir(base_checkpoint, tmp_path / "clone_unlisted")
    bogus = clone / "checkpoint_999999.pt"
    with pytest.raises(CheckpointManifestError, match="not listed"):
        kd.Model(
            algorithm="sga",
            generations=1,
            population=_POPULATION,
            seed=0,
            verbose=False,
        ).fit(dataset, resume_from=bogus)


def test_fence_resume_from_legacy_no_manifest_dir_is_byte_identical(
    tmp_path: Path, dataset: PDEDataset, base_checkpoint: Path
) -> None:


    legacy = tmp_path / "legacy_dir"
    legacy.mkdir()
    shutil.copy(base_checkpoint, legacy / "checkpoint_final.pt")
    assert not (legacy / MANIFEST_FILENAME).exists()
    resumed = kd.Model(
        algorithm="sga",
        generations=1,
        population=_POPULATION,
        seed=0,
        verbose=False,
    ).fit(dataset, resume_from=legacy / "checkpoint_final.pt")
    assert resumed.best_expr_


def test_fence_resume_from_torn_manifest_dir_fails_loud(
    tmp_path: Path, dataset: PDEDataset, base_checkpoint: Path
) -> None:


    clone = _clone_manifest_dir(base_checkpoint, tmp_path / "clone_torn")
    (clone / "orphan.pt").write_bytes(b"stray")
    with pytest.raises(CheckpointManifestError, match="orphan"):
        kd.Model(
            algorithm="sga",
            generations=1,
            population=_POPULATION,
            seed=0,
            verbose=False,
        ).fit(dataset, resume_from=clone / "checkpoint_final.pt")







def test_w7_injected_surrogate_identity_change_rejected_before_build(
    tmp_path: Path, dataset: PDEDataset
) -> None:
    model_a = nn.Sequential(nn.Linear(2, 1))
    model_b = nn.Sequential(nn.Linear(2, 1))
    with torch.no_grad():
        for param in model_a.parameters():
            param.fill_(1.0)
        for param in model_b.parameters():
            param.fill_(2.0)

    checkpoint = tmp_path / "dlga_model_a.pt"
    payload = build_checkpoint_payload(1, DLGAPlugin(surrogate_model=model_a))
    torch.save(payload, checkpoint)

    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    with pytest.raises(ValueError, match="identity_breaking") as exc:
        kd.Model(
            algorithm="dlga",
            generations=1,
            surrogate_model=model_b,
            checkpoint_dir=resume_dir,
            seed=0,
            verbose=False,
        ).fit(dataset, resume_from=checkpoint)
    assert "surrogate_model" in str(exc.value)

    assert not any(resume_dir.iterdir())
