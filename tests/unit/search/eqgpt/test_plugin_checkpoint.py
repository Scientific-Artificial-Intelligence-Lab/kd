
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_HELPER = Path(__file__).resolve().parent / "_h5_helper.py"


def _run(mode: str, path: Path, home: Path) -> tuple[str, str]:
    proc = subprocess.run(
        [sys.executable, str(_HELPER), mode, str(path)],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": ":".join(sys.path),
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),


            "PYTHONHASHSEED": "0",
        },
    )
    assert proc.returncode == 0, f"{mode} failed:\n{proc.stderr}"
    full, weights = proc.stdout.strip().split()
    return full, weights


@pytest.mark.slow
def test_checkpoint_resume_equivalence(tmp_path: Path) -> None:
    straight5_full, straight5_weights = _run("straight5", tmp_path / "x.pt", tmp_path)
    straight2_full, straight2_weights = _run("straight2", tmp_path / "x.pt", tmp_path)

    assert straight5_full != straight2_full





    assert straight2_weights != straight5_weights

    phase1_full, phase1_weights = _run("phase1", tmp_path / "ckpt.pt", tmp_path)




    assert phase1_full == straight2_full
    assert phase1_weights == straight2_weights

    resumed_full, _ = _run("phase2", tmp_path / "ckpt.pt", tmp_path)
    assert resumed_full == straight5_full
